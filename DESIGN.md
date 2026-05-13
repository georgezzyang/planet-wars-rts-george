# Decoupled UCT (+ PUCT) for Planet Wars RTS

Design notes and empirical results for the MCTS-family entry to the AAMAS 2026
Planet Wars RTS competition. Single-author research record; intent is that this
document is enough to (a) reproduce every number below from the code in this
repo, (b) understand every design choice and the alternative that was rejected,
(c) pick up the next iteration without re-deriving anything.

This document starts at UCT. The prior NaiveMCTS-family iterations (flat
NaiveSampling, tree NaiveMCTS) are documented in commit history and the
comparison report against `santiontanon/microrts`, but are not the design
baseline going forward.

---

## 1. Domain

Planet Wars RTS is a real-time strategy game with the following structure
relevant to algorithm choice:

- **Action interface.** Each game tick, each player issues exactly ONE
  `Action(playerId, sourcePlanetId, destinationPlanetId, numShips)` — i.e. one
  transporter launch from one planet to one target. Or `DoNothing`.
- **Time budget.** 50 ms per tick. 20 Hz. Game runs up to 2000 ticks ≈ 100 s.
- **Action effect is delayed.** Transporters take `distance / transporterSpeed`
  ticks to arrive. With typical params, ~10–30 tick delay between issuing a
  launch and seeing its result on the battlefield.
- **Two-player simultaneous moves.** Both players' actions for tick t are
  collected, then `ForwardModel.step(actions: Map<Player, Action>)` advances
  the state. Not turn-based.
- **Action validity.** The forward model silently drops actions for which
  `source.transporter != null`, `source.owner != player`, or
  `source.nShips < action.numShips`. Important: a wrongly-attributed action
  vanishes without error.
- **Game state.** `GameState` is a list of `Planet`s plus a tick counter.
  Each `Planet` has owner, ship count (continuous Double), position (Vec2d),
  growth rate, radius, and a possibly-null `Transporter`. State is fully
  observable in the AAMAS 2026 track.
- **Randomized params per match.** Per the AAMAS 2026 spec: planets 10–30,
  initial neutral ratio 0.25–0.35, growth 0.05–0.2, transporter speed 2.0–5.0.
  Agents must generalise across this range.

How this compares to MicroRTS (where the NaiveMCTS literature originates):
single-action-per-tick instead of multi-unit joint actions, much smaller joint
action space (~100–300 pairs), delayed action effects, true simultaneous
moves, tighter time budget. These mostly invert the assumptions NaiveMCTS was
designed for; see the comparison report in commit history.

## 2. Why UCT, not NaiveMCTS

Empirical and structural reasons in summary:

- NaiveSampling factorises the per-player joint action into independent
  per-factor MABs. This is the right move when the joint space has 10^7 entries
  (MicroRTS). Here the per-player joint space is ~25 `(src, tgt)` pairs — small
  enough to enumerate. Factorisation pays bias without buying tractability.
- The flat NaiveSampling variant (no tree, root-only NaiveSampling) reached
  70% win rate in a 6-agent league; the faithful tree NaiveMCTS reached 41.7%.
  Tree-NaiveMCTS performed WORSE than flat — visit dilution across joint
  children, plus the bias from factorising independent factors that aren't.
- Time discount on the evaluation function was missing in the NaiveMCTS impl;
  added back here. See `timeDiscountGamma` / `timeDiscountScale` below.

Decoupled UCT (Shafiei et al. 2009) is the clean simultaneous-move MCTS
baseline: at every node, each player runs an independent UCB1 over its own
flat enumerated action set. No factorisation. PUCT (Silver et al. 2017) sits
naturally on top as an optional prior-augmented variant.

## 3. Algorithm: Decoupled UCT (vanilla)

File: `app/src/main/java/competition_entry/UCTAgent.java`, with
`useHeuristicPrior = false` (default).

### 3.1 Tree node

Each tree node represents a state reachable by some path of joint actions from
the root. State is NOT stored on the node — it is reconstructed each iteration
by `rootState.deepCopy()` plus replaying the path's joint actions through
`ForwardModel`. Trades a small CPU cost (~µs per step) for not paying memory
to keep states on millions of potential children. The `ForwardModel` is fast
enough that this is the right trade for Planet Wars-scale state.

### 3.2 Per-player action enumeration

When a node is first visited, both players' legal actions are enumerated:

- `mySources` / `oppSources`: planet IDs owned by this player with no
  outbound transporter and ≥ 1 ship.
- `myTargets` / `oppTargets`: opponent-owned planets ONLY by default, with
  neutrals as a fallback if the opponent has no planets. This was a 20-point
  finding from an earlier ablation: CarefulRandomAgent (62% win rate) targets
  only opponents; BetterRandomAgent (6%) targets opponents and neutrals. Attacking
  neutrals is a strategic trap at these game params — pay garrison cost, then
  the captured planet sits at zero ships while the opponent keeps growing.

The arm list per player is the Cartesian product `mySources × myTargets`, plus
one explicit `DoNothing` arm appended at the end. Encoded as
`(srcIdx << 8) | tgtIdx`, with a dedicated `DO_NOTHING_PAIR = 0xFFFF` sentinel.

### 3.3 Decoupled selection

At each node, ME and OPP each run their own independent UCB1 over their flat
arm list:

```
U(a) = Q(a) + ucbC * sqrt( ln(N) / n_a )
```

with first-play-urgency = forced exploration of any unvisited arm.

The joint child key is `(myArm << 16) | (oppArm & 0xFFFF)`.

### 3.4 Iteration loop

Standard MCTS:

1. **SELECT**: Walk down the tree using Decoupled UCB1, applying joint actions
   to a working state copy. Stop at first joint child that does not yet exist.
2. **EXPAND**: Create the missing child node, populate its legal actions.
   Single-node expansion per iteration.
3. **ROLLOUT**: From the leaf, run heuristic policies (`GreedyHeuristicAgent`
   for both ME and OPP) for `rolloutSteps` ticks or until terminal.
4. **EVALUATE**: `reward = (myShips - oppShips) * γ^(elapsed_ticks / scale)`.
   Time discount makes 50-tick decisive victories rank above 500-tick stalls
   with the same terminal ship diff.
5. **BACKPROP**: Walk back up the trajectory. Each node along the path gets
   `visits++`, `totalReward += reward`. The arm taken at that node gets its
   per-player stats updated. Opponent's MAB receives NEGATED reward (zero-sum
   minimax: opponent maximizing negated-reward is equivalent to minimizing
   my reward).

### 3.5 Final action selection at root

Robust child policy: return the action whose root-arm has the highest visit
count. More stable than max-mean-reward when arm visit counts are uneven.

## 4. PUCT extension

File: same `UCTAgent.java`, with `useHeuristicPrior = true`. Switches the per-arm
selection formula from UCB1 to AlphaGo-Zero-style PUCT:

```
U(a) = Q(a) + priorWeight * P(a) * sqrt(N) / (1 + n_a)
```

- `Q(a)` = mean reward; first-play-urgency = 0 (no forced exploration —
  the prior carries it).
- `N` = total visits at the node (per player).
- `P(a)` = prior probability over arms, computed once when the node is
  populated. Softmax of per-arm domain scores.
- `priorWeight` = the "c_puct" constant. Currently 2.0.

### 4.1 Per-arm score

For each `(src, tgt)` arm:

```
attack_ships     = src.nShips * shipsFraction
traversal_ticks  = distance(src, tgt) / transporter_speed
defense_estimate = tgt.nShips + tgt.growthRate * traversal_ticks
net_capture      = attack_ships - defense_estimate
distance_penalty = distance / 800.0
growth_bonus     = tgt.growthRate * 50.0

score(src, tgt)  = net_capture + growth_bonus - 2.0 * distance_penalty
```

DoNothing gets a flat `score = -2.0`. Scores are softmax'd with
`priorTemperature = 2.0` to produce P(a). Sum-normalised. Degenerate fallback
to uniform if the softmax explodes.

The factors are exactly the same domain knowledge GreedyHeuristicAgent uses;
the prior is essentially "what GreedyHeuristicAgent thinks about each move",
turned into a probability distribution.

## 5. Implementation notes

### 5.1 Single class, gated variants

Both vanilla UCT and PUCT live in `UCTAgent.java`, gated by `useHeuristicPrior`.
`getAgentType()` returns different strings (`UCT-decoupled-v1` vs `UCT-puct-v2`)
so the same class can appear twice in the round-robin league. Reduces code
duplication; both share rollout, eval, backprop, tree-walk.

### 5.2 Java/Kotlin interop

The repo is Kotlin (the framework), but our agent is in Java
(`app/src/main/java/`). The Kotlin Gradle plugin auto-detects the Java source
set. Notes:

- `Action` (Kotlin data class) is constructed directly with `new Action(player,
  srcId, tgtId, ships)` rather than via the companion object, to dodge the
  `Action.Companion.doNothing()` Java-from-Kotlin awkwardness.
- `Player`, `Planet` are Kotlin enums/data classes — readable from Java via
  the auto-generated getters (`planet.getNShips()`, `planet.getOwner()`).
- `PlanetWarsPlayer` (Kotlin abstract class) holds `protected var player` and
  `protected var params`; Java subclass uses `getPlayer()` / `getParams()`.

### 5.3 Player-switch caching bug — documented for posterity

`RoundRobinLeague` reuses the SAME agent instance as both Player 1 and
Player 2 across pair iterations. The first naive implementation cached
`opponentRolloutPolicy.prepareToPlayAs(opp, ...)` behind a `modelsReady`
boolean, so when the agent flipped to playing the other color, the rollout
policies' internal `player` field stayed stale → GreedyHeuristic generated
actions from the WRONG player's planets → ForwardModel silently dropped every
action → rollouts degenerated to 30 ticks of DoNothing.

Symptom: pair timings asymmetric by 4–5x. UCT(P1) vs Greedy(P2) ran 95s for
6 games; Greedy(P1) vs UCT(P2) ran 258s. Win rate signal corrupted
proportionally.

Fix: replaced the boolean cache with a `rolloutPoliciesPreparedFor` Player
field that detects player change and re-prepares the rollout policies. Same
bug almost certainly exists in any other agent that caches per-player setup
across calls (e.g. `SimpleEvoAgent`'s `bestSolution` shift buffer).

Lesson: never assume an agent instance has a stable player identity across
`getAction` calls in a multi-pair tournament. Either re-prepare every call or
detect changes explicitly.

### 5.4 Hyperparameters

| Name                  | Value          | Effect                                  |
|-----------------------|----------------|-----------------------------------------|
| `timeBudgetMillis`    | 45             | 5 ms safety margin under server's 50 ms |
| `rolloutSteps`        | 30             | Rollout horizon                         |
| `maxTreeDepth`        | 8              | Tree descent cap                        |
| `ucbC`                | √2             | UCB1 exploration constant (vanilla)     |
| `shipsFraction`       | 0.5            | Fraction of source ships sent           |
| `timeDiscountGamma`   | 0.99           | Eval = ship_diff × γ^(elapsed / scale)  |
| `timeDiscountScale`   | 10.0           | Discount horizon                        |
| `useHeuristicPrior`   | false / true   | Switch between vanilla / PUCT           |
| `priorWeight`         | 2.0            | PUCT exploration term scaling           |
| `priorTemperature`    | 2.0            | Softmax temperature for per-arm scores  |

None of these have been swept. They are first-cut guesses, except
`shipsFraction` which matches every other agent in the framework.

### 5.5 Build / run

- `./gradlew :app:classes` — incremental compile.
- `./gradlew :app:benchmarkNaiveMCTS` — runs the local league
  defined in `app/src/main/kotlin/competition_entry/LocalBenchmark.kt`.
  Despite the task name (legacy), it benchmarks whatever agents the file
  lists.
- `./gradlew :app:shadowJar` then `java -jar app/build/libs/client-server.jar`
  — launches the WebSocket server on :8080 for competition deployment.
  Entry point: `competition_entry.RunEntryAsServerKt`, which constructs
  the agent configured in that file. Currently `UCTAgent()` (vanilla).

JDK 21 toolchain (Foojay auto-provisioned by Gradle). Launcher JVM must be
≤ JDK 21 because Gradle 8.12 does not support newer JDKs as a launcher;
local development uses `C:\Users\…\.jdks\jdk-21.0.11+10`.

## 6. Empirical results

All benchmarks: 12 planets, 1200 max ticks, GreedyHeuristic + EvoAgent +
CarefulRandom as references. League pots scaled to `points / nGames * 100`.

### 6.1 Vanilla UCT (N = 30 per pair, 4 agents, 180 games / agent)

| Rank | Agent                | Win % | Points |
|------|----------------------|-------|--------|
| 1    | UCT-decoupled-v1     | 62.2  | 112    |
| 2    | EvoAgent (RHEA)      | 58.9  | 106    |
| 3    | GreedyHeuristic      | 54.4  | 98     |
| 4    | CarefulRandom        | 24.4  | 44     |

UCT leads RHEA by 3.3 percentage points. With N=180 per agent, the standard
error of a 60% baseline is ~3.7%; the gap is NOT statistically significant
(p ≈ 0.26 single-sided, two-proportion test). Read this as "UCT and RHEA
plausibly equal at this setting"; the 5.6-point lead seen at N=6 was sampling
noise.

### 6.2 PUCT vs vanilla UCT (N = 10 per pair, 5 agents, 80 games / agent)

| Rank | Agent                | Win % | Points |
|------|----------------------|-------|--------|
| 1    | UCT-puct-v2          | 80.0  | 64     |
| 2    | EvoAgent (RHEA)      | 50.0  | 40     |
| 3    | GreedyHeuristic      | 48.8  | 39     |
| 4    | UCT-decoupled-v1     | 47.5  | 38     |
| 5    | CarefulRandom        | 23.8  | 19     |

PUCT leads RHEA by 30 points, vanilla UCT by 32.5. With N=80 per agent,
standard error of 80% is ~4.5%; the gap is comfortably significant.

Three observations from this league:

1. The +32-point PUCT-over-vanilla gap is the headline. The prior carries
   most of the lift, but a vanilla MCTS that already uses GreedyHeuristic
   IN ROLLOUTS still trails by this much. Conclusion: **heuristic-in-
   rollouts is not enough; heuristic must also bias tree expansion**.
   At a 50 ms / ~1000-iteration budget, UCB1 from-scratch cannot find the
   action structure that the heuristic encodes for free.

2. Vanilla UCT, RHEA, and GreedyHeuristic cluster at 47–50%, basically
   indistinguishable when PUCT dominates the league. Without PUCT in the
   mix, vanilla UCT was at 62%; with PUCT in the mix, it drops to 47.5%.
   Compositional effect — a league reranks every agent's win rate when a
   strong newcomer enters.

3. PUCT vs Greedy = +31 points isolates the search contribution on top of
   the prior. The prior alone (Greedy) is at 48.8%; the prior plus 1000
   iterations of PUCT-guided search is at 80%. Search ITSELF adds ~31
   points of value when given a non-uniform starting distribution.

### 6.3 Historical numbers from earlier benchmarks (different leagues, NOT directly comparable)

For context only:

- FlatNaiveSampling-v3: 70.0% in a 6-agent league with weaker randoms.
- NaiveMCTS-tree-v1 (faithful Ontañón impl): 41.7% in same 6-agent league.
- These leagues had the player-switch caching bug present in all agents
  except EvoAgent — actual relative strengths may differ from reported.

## 7. Known issues and caveats

1. **Sample size.** PUCT was confirmed at N=10/pair which gives wide CIs.
   PUCT's lead is large enough to survive easily, but if we want a paper
   number, redo at N=30+.
2. **Hyperparameters not swept.** All values in §5.4 are first guesses.
   `priorWeight=2.0`, `priorTemperature=2.0`, `ucbC=√2`, `rolloutSteps=30`,
   `timeBudgetMillis=45` — every one of these has a plausibly-better value.
3. **Tree depth.** Anecdotally rarely exceeds 2 in 50 ms / ~1000 iter.
   `maxTreeDepth = 8` is a safety cap that is never hit. Worth instrumenting.
4. **Opponent modelling.** Each tree node maintains opp's MAB and updates
   with negated reward — assumes opp is also doing UCT-like search. Actual
   opponents in this competition are heuristics (RHEA, Greedy, scripts).
   The opp-MAB at the root is therefore learning a fiction. May or may not
   matter — needs ablation.
5. **Asymmetric pair timings, still slightly.** UCT-vs-Evo went 513s P1 /
   367s P2 at N=30 — 40% asymmetry that hasn't been root-caused. Could be
   EvoAgent's shift-buffer carrying stale state across pairs (same bug
   family as §5.3 but in `SimpleEvoAgent`). Win-rate signal probably OK,
   timing is the diagnostic.
6. **No instrumentation.** Currently no internal stats logged: iter count
   per tick, average tree depth, prior entropy, action-revisit frequency.
   Worth adding for any next paper-grade experiment.

## 8. Potential next steps

### Cheap (≤ 1 hour each)

- **Hyperparameter grid:** `priorWeight ∈ {1, 2, 4}` × `priorTemperature ∈
  {1, 2, 4}` × `ucbC ∈ {0.7, 1.4, 2.8}`. Run mini-leagues; pick best.
- **Time-budget sweep:** PUCT at 10, 25, 45, 100, 250 ms per tick. Sketches
  the search-budget vs strength curve. Useful for the paper.
- **Larger-N PUCT confirmation:** redo §6.2 at N=30 to get tight CIs.
- **Instrumentation:** log per-tick iter count, max tree depth reached,
  prior entropy of top arm. Surface in benchmark output.

### Medium (1–2 days each)

- **Prior ablation.** Replace the per-arm score formula component-by-
  component: drop `net_capture`; drop `distance_penalty`; drop
  `growth_bonus`; replace prior with uniform; replace prior with random.
  Each ablation isolates one design choice's contribution. Tabular result.
- **Asymmetric tree (no opp MAB).** Replace the opp-side NaiveSampling at
  each tree node with a fixed call to GreedyHeuristicAgent. Tests whether
  the self-play opp model at the tree is helping or hurting given real
  opponents are heuristics.
- **Reimplement RHEA from scratch.** Read SimpleEvoAgent source; build our
  own RHEA with a fair-sized shift buffer, possibly islands, possibly
  script-seeded initial population. Compare to PUCT and to SimpleEvoAgent.
  Tests whether the existing EvoAgent baseline is optimally tuned.
- **Better rollout policy.** Currently both sides play Greedy in rollouts;
  this may make rollout outcomes too correlated. Try Random, RandomBiased,
  asymmetric (me=Greedy, opp=Random), and shorter rollouts with a value
  function.

### Substantial (3+ days)

- **Learned prior.** Replace the hand-coded per-arm score with a small
  policy network trained from PUCT self-play. The full AlphaGo-Zero loop
  at smaller scale. Needs: self-play infrastructure, training pipeline,
  inference budget within the 50 ms / tick constraint. Could plausibly
  push PUCT past 90%.
- **Learned value head.** Train a value network on (state → win prob) from
  self-play, replace rollout with `V(s)` evaluation. Decouples search budget
  from rollout horizon. Same infrastructure cost as the learned prior.

### Open research questions worth a section in a paper

- Why does short-budget MCTS so dramatically depend on a prior? Formal
  characterisation of the regime where heuristic-in-rollouts is insufficient
  vs heuristic-in-selection is required.
- How does PUCT advantage scale with time budget? Does it plateau or keep
  growing? At what point does vanilla UCT catch up?
- Is the "MCTS-needs-prior" finding a Planet-Wars-specific quirk or a
  general property of small-action-space simultaneous-move games at tight
  budgets?

## 9. File layout

```
app/src/main/java/competition_entry/
  UCTAgent.java                 # Decoupled UCT + PUCT (single class, gated)
  FlatNaiveSamplingAgent.java   # earlier baseline; kept for historical ablation
  NaiveMCTSAgent.java           # faithful tree-based Ontañón 2013 impl; abandoned
                                # (perform 41.7% in earlier league)

app/src/main/kotlin/competition_entry/
  RunEntryAsServer.kt           # WebSocket entry — currently launches UCTAgent
  GreedyHeuristicAgent.kt       # reference heuristic (also used as rollout policy)
  LocalBenchmark.kt             # round-robin league runner
  CarefulPartialAgent.kt        # for partial-obs track (unused this round)
  RunPartialAsServer.kt         # ditto

app/build.gradle.kts            # shadowJar Main-Class points at RunEntryAsServerKt
Dockerfile                       # self-contained multi-stage build for submission
```

## 10. Reproducing the numbers in this doc

```bash
# Set JAVA_HOME to a JDK 21 install
./gradlew :app:classes              # incremental compile
./gradlew :app:benchmarkNaiveMCTS   # runs LocalBenchmark.kt
```

The benchmark task name is legacy and DOES NOT mean it only runs NaiveMCTS;
it runs whatever agent list is configured in `LocalBenchmark.kt`. The current
configuration runs `[UCTAgent, UCTAgent(useHeuristicPrior=true), GreedyHeuristic,
SimpleEvoAgent, CarefulRandomAgent]` at 10 games / pair.

To reproduce §6.1: set `GAMES_PER_PAIR = 30` and remove the PUCT variant.
To reproduce §6.2: leave as-is.

Commit hash at the time of this writing: `057badf` on the `george` remote
(`georgezzyang/planet-wars-rts-george`).
