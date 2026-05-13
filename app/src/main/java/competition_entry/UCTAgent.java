package competition_entry;

import games.planetwars.agents.Action;
import games.planetwars.agents.PlanetWarsAgent;
import games.planetwars.agents.PlanetWarsPlayer;
import games.planetwars.core.ForwardModel;
import games.planetwars.core.GameState;
import games.planetwars.core.Planet;
import games.planetwars.core.Player;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Decoupled UCT for Planet Wars (Shafiei et al. 2009 style).
 *
 * The "plain MCTS baseline": straight UCT, no NaiveSampling factorization, no
 * heuristic priors, no PUCT bias. The cleanest reference point against which to
 * compare more elaborate variants.
 *
 * Tree structure
 * --------------
 * Each node = a unique state reached by a particular sequence of joint actions
 * from the root. State is reconstructed per iteration (deepCopy root + replay
 * along the path) - same trade-off as NaiveMCTSAgent.
 *
 * Action representation per node, per player
 * ------------------------------------------
 * The full per-player action set is ENUMERATED as a flat list of
 * (sourceIdx, targetIdx) pairs, plus one explicit DoNothing arm at the end.
 * No factorization into separate source / target MABs. Each arm has its own
 * UCB1 stats (visits, totalReward).
 *
 * Decoupled selection (the "Decoupled" in Decoupled UCT)
 * ------------------------------------------------------
 * At each node, ME and OPP each run their own independent UCB1 selection over
 * their own action set. The joint action is (myArm, oppArm). This is the
 * standard adaptation of UCT to simultaneous-move 2-player games where solving
 * the true matrix-game equilibrium per node would be too expensive.
 *
 * Iteration
 * ---------
 *   1. SELECT: descend the tree by Decoupled UCB1 at every node.
 *   2. EXPAND: when the joint child does not exist, create one (single-node
 *              expansion per iteration, standard MCTS).
 *   3. ROLLOUT: from the leaf, both players play GreedyHeuristic until
 *               rolloutSteps or terminal.
 *   4. BACKPROP: update visits + totalReward at every node, plus the per-player
 *                arm stats for the action taken at that node. Opponent's stats
 *                use NEGATED reward (zero-sum minimax). Final reward is
 *                TIME-DISCOUNTED so faster wins beat slow ones with the same
 *                terminal ship diff.
 *
 * Differences from {@link NaiveMCTSAgent}
 * ---------------------------------------
 * - No three-MAB NaiveSampling at each node; one flat UCB1 over (src,tgt) pairs.
 * - Adds time-discounted evaluation (gamma^t) - Diff #4 from the MicroRTS comparison.
 * - Removes the epsilon-greedy mix between local and global MABs.
 *
 * Final action selection at root: ROBUST CHILD = highest-visit arm.
 */
public class UCTAgent extends PlanetWarsPlayer {

    // ---- Hyperparameters ----
    public long timeBudgetMillis = 45;
    public int rolloutSteps = 30;
    public int maxTreeDepth = 8;
    public double ucbC = Math.sqrt(2.0);
    public double shipsFraction = 0.5;
    public double timeDiscountGamma = 0.99;
    public double timeDiscountScale = 10.0;     // discount = gamma ^ (elapsed / scale)

    // ---- PUCT enhancement (off by default = vanilla UCT) ----
    /** When true, switches arm selection from UCB1 to PUCT (AlphaGo Zero style):
     *  U(a) = Q(a) + priorWeight * P(a) * sqrt(N) / (1 + n_a)
     *  where P(a) is a heuristic prior derived from the same domain logic as
     *  GreedyHeuristicAgent (distance + defense estimate + growth bonus). */
    public boolean useHeuristicPrior = false;
    /** Softmax temperature on the raw heuristic scores. Lower = sharper prior. */
    public double priorTemperature = 2.0;
    /** PUCT exploration constant — the c in c * P(a) * sqrt(N) / (1 + n_a). */
    public double priorWeight = 2.0;
    /** When true (with useHeuristicPrior also true) use the v14 strong prior:
     *  v14 weightGrowth=100, preserveProductionSources penalty, counterattackRisk penalty,
     *  attackMomentum bonus, gameStage modulation, minTargetGrowth filter at 0.05.
     *  This is the HeuristicAgent v14 scoring formula imported as a PUCT prior. */
    public boolean priorV14 = false;
    /** When true, persist the tree across getAction calls. After each tick, infer the joint
     *  action that was actually played (my action + opp's action from state diff), find the
     *  corresponding child of the persistent root, and promote it as the new root. The
     *  promoted root keeps all its accumulated visits → MCTS effectively gets cumulative
     *  search budget across the entire game instead of 50ms/tick fresh starts. RHEA-style
     *  shift-buffer adaptation for MCTS. */
    public boolean useTreeReuse = false;

    // ---- Models ----
    private final PlanetWarsAgent opponentRolloutPolicy = new GreedyHeuristicAgent();
    private final PlanetWarsAgent myRolloutPolicy = new GreedyHeuristicAgent();

    // ---- Per-call ----
    private Player me;
    private Player opp;
    /** Last me-Player we prepared the rollout policies as. null = never prepared. */
    private Player rolloutPoliciesPreparedFor = null;

    // ---- Tree-reuse state (persists across getAction calls) ----
    private Node persistentRoot = null;
    private GameState lastState = null;
    private Action lastMyAction = null;

    // Sentinel arm value meaning DoNothing. Stored at the LAST index of each player's arm list.
    private static final int DO_NOTHING_PAIR = 0xFFFF;

    @Override
    public Action getAction(GameState rootState) {
        long deadline = System.currentTimeMillis() + timeBudgetMillis;
        me = getPlayer();
        opp = (me == Player.Player1) ? Player.Player2 : Player.Player1;
        // Re-prepare rollout policies whenever the player assignment changes (RoundRobinLeague
        // reuses the same agent instance for both colors). Without this, the rollout policies'
        // internal `player` field stays stale → GreedyHeuristic samples sources from the wrong
        // side → ForwardModel silently drops every action → rollouts degenerate to DoNothing.
        if (rolloutPoliciesPreparedFor != me) {
            opponentRolloutPolicy.prepareToPlayAs(opp, getParams(), "internal-rollout-opp");
            myRolloutPolicy.prepareToPlayAs(me, getParams(), "internal-rollout-me");
            rolloutPoliciesPreparedFor = me;
            // Same bug class — tree built for prior player assignment is invalid after switch.
            persistentRoot = null;
            lastState = null;
            lastMyAction = null;
        }

        Node root = null;
        if (useTreeReuse && persistentRoot != null && lastState != null && lastMyAction != null) {
            root = tryReuseRoot(rootState);
        }
        if (root == null) {
            root = new Node();
            root.populateLegal(rootState);
        }
        if (root.mySources.length == 0 || root.myTargets.length == 0) {
            if (useTreeReuse) {
                persistentRoot = root;
                lastState = rootState.deepCopy();
                lastMyAction = doNothing();
            }
            return doNothing();
        }

        Random rng = ThreadLocalRandom.current();
        while (System.currentTimeMillis() < deadline) {
            iterate(root, rootState, rng);
        }
        Action best = root.bestMyActionAtRoot(rootState);
        if (useTreeReuse) {
            persistentRoot = root;
            lastState = rootState.deepCopy();
            lastMyAction = best;
        }
        return best;
    }

    /**
     * Find the child of persistentRoot corresponding to (lastMyAction, inferredOppAction)
     * and return it as the new root. Returns null if no matching child exists (then caller
     * builds fresh).
     */
    private Node tryReuseRoot(GameState currentState) {
        Action inferredOppAction = inferOppAction(lastState, currentState);
        int myArm = findArmIdx(persistentRoot.mySources, persistentRoot.myTargets, lastMyAction);
        int oppArm = findArmIdx(persistentRoot.oppSources, persistentRoot.oppTargets, inferredOppAction);
        if (myArm < 0 || oppArm < 0) return null;
        int childKey = (myArm << 16) | (oppArm & 0xFFFF);
        Node next = persistentRoot.children.get(childKey);
        if (next == null || next.mySources == null) return null;
        return next;
    }

    /**
     * Look for a newly-launched opp transporter by diffing planets between before/after.
     * A planet whose transporter was null in `before` and non-null in `after`, with
     * transporter.owner == opp, indicates opp launched there this tick.
     * 0 or 1 such planets expected (each player launches at most one transporter per tick).
     */
    private Action inferOppAction(GameState before, GameState after) {
        List<Planet> beforeP = before.getPlanets();
        List<Planet> afterP = after.getPlanets();
        int n = Math.min(beforeP.size(), afterP.size());
        for (int i = 0; i < n; i++) {
            Planet pOld = beforeP.get(i);
            Planet pNew = afterP.get(i);
            if (pOld.getTransporter() == null && pNew.getTransporter() != null) {
                if (pNew.getTransporter().getOwner() == opp) {
                    return new Action(opp,
                            pNew.getId(),
                            pNew.getTransporter().getDestinationIndex(),
                            pNew.getTransporter().getNShips());
                }
            }
        }
        return new Action(Player.Neutral, -1, -1, 0.0);   // opp did nothing
    }

    /**
     * Given an action and the old root's source/target arrays, return the arm index in the
     * old root's encoding scheme (srcIdx * targetCount + tgtIdx), or sources.length *
     * targets.length for DoNothing. Returns -1 if the action's planets aren't in the old
     * root's legal sets (means tree-reuse should rebuild).
     */
    private static int findArmIdx(int[] sources, int[] targets, Action action) {
        if (action.getSourcePlanetId() == -1) {
            // DoNothing — last index in the arm array (or index 0 when there were no real arms)
            int realCount = sources.length * targets.length;
            return realCount;       // == 0 if sources or targets empty
        }
        int srcIdx = -1;
        for (int i = 0; i < sources.length; i++) {
            if (sources[i] == action.getSourcePlanetId()) { srcIdx = i; break; }
        }
        int tgtIdx = -1;
        for (int i = 0; i < targets.length; i++) {
            if (targets[i] == action.getDestinationPlanetId()) { tgtIdx = i; break; }
        }
        if (srcIdx < 0 || tgtIdx < 0) return -1;
        return srcIdx * targets.length + tgtIdx;
    }

    /** Single MCTS iteration: select -> expand -> rollout -> backprop. */
    private void iterate(Node root, GameState rootState, Random rng) {
        GameState sim = rootState.deepCopy();
        ForwardModel model = new ForwardModel(sim, getParams());
        int rolloutStartTick = sim.getGameTick();

        ArrayList<Node> path = new ArrayList<>(maxTreeDepth + 1);
        ArrayList<int[]> arms = new ArrayList<>(maxTreeDepth);  // each entry: [myArmIdx, oppArmIdx]
        path.add(root);

        Node node = root;
        int depth = 0;
        while (depth < maxTreeDepth && !model.isTerminal()) {
            int myArmIdx = node.selectMyUcb();
            int oppArmIdx = node.selectOppUcb();
            arms.add(new int[]{myArmIdx, oppArmIdx});

            applyJoint(model, sim, node, myArmIdx, oppArmIdx);

            int childKey = (myArmIdx << 16) | (oppArmIdx & 0xFFFF);
            Node child = node.children.get(childKey);
            if (child == null) {
                child = new Node();
                child.populateLegal(sim);
                node.children.put(childKey, child);
                path.add(child);
                break;
            }
            path.add(child);
            node = child;
            depth++;
        }

        // ROLLOUT with heuristic policies on both sides
        for (int t = 0; t < rolloutSteps && !model.isTerminal(); t++) {
            HashMap<Player, Action> joint = new HashMap<>(2);
            joint.put(me, myRolloutPolicy.getAction(sim));
            joint.put(opp, opponentRolloutPolicy.getAction(sim));
            model.step(joint);
        }

        // Time-discounted evaluation: prefer faster victories
        double rawReward = model.getShips(me) - model.getShips(opp);
        int elapsed = sim.getGameTick() - rolloutStartTick;
        double reward = rawReward * Math.pow(timeDiscountGamma, elapsed / timeDiscountScale);

        // BACKPROP
        for (int i = 0; i < path.size(); i++) {
            Node n = path.get(i);
            n.visits++;
            n.totalReward += reward;
            if (i < arms.size()) {
                int[] a = arms.get(i);
                n.updateMy(a[0], reward);
                n.updateOpp(a[1], -reward);
            }
        }
    }

    private void applyJoint(ForwardModel model, GameState sim, Node node, int myArmIdx, int oppArmIdx) {
        Action myAction = node.armToAction(myArmIdx, sim, /*forMe*/ true);
        Action oppAction = node.armToAction(oppArmIdx, sim, /*forMe*/ false);
        HashMap<Player, Action> joint = new HashMap<>(2);
        joint.put(me, myAction);
        joint.put(opp, oppAction);
        model.step(joint);
    }

    private static Action doNothing() {
        return new Action(Player.Neutral, -1, -1, 0.0);
    }

    @Override
    public String getAgentType() {
        StringBuilder sb = new StringBuilder();
        if (!useHeuristicPrior) sb.append("UCT-decoupled-v1");
        else if (priorV14) sb.append("UCT-puct-v14");
        else sb.append("UCT-puct-v2");
        if (useTreeReuse) sb.append("+reuse");
        return sb.toString();
    }

    // =====================================================================
    // Inner classes
    // =====================================================================

    /**
     * One node in the joint-action tree. Each player has a flat UCB1 MAB over their
     * action set: all (sourceIdx, targetIdx) pairs plus one DoNothing arm at the end.
     */
    private final class Node {
        int[] mySources, myTargets;
        int[] oppSources, oppTargets;

        // Flat enumerated arms per player: each entry encodes (srcIdx << 8) | tgtIdx,
        // or DO_NOTHING_PAIR for the dedicated DoNothing arm at the last index.
        int[] myArmEncoding;
        int[] oppArmEncoding;

        // UCB1 stats for ME
        double[] myTotal;
        int[] myVisits;
        long myTotalVisits;

        // UCB1 stats for OPP
        double[] oppTotal;
        int[] oppVisits;
        long oppTotalVisits;

        // Heuristic prior P(a) per arm — null when useHeuristicPrior is false (pure UCB1).
        double[] myPrior;
        double[] oppPrior;

        final HashMap<Integer, Node> children = new HashMap<>();
        int visits;
        double totalReward;

        void populateLegal(GameState s) {
            mySources = legalSources(s, me);
            myTargets = legalOppTargetsFor(s, me);
            oppSources = legalSources(s, opp);
            oppTargets = legalOppTargetsFor(s, opp);

            myArmEncoding = enumerateArms(mySources.length, myTargets.length);
            oppArmEncoding = enumerateArms(oppSources.length, oppTargets.length);

            myTotal = new double[myArmEncoding.length];
            myVisits = new int[myArmEncoding.length];
            oppTotal = new double[oppArmEncoding.length];
            oppVisits = new int[oppArmEncoding.length];

            if (useHeuristicPrior) {
                if (priorV14) {
                    myPrior = computePriorV14(s, mySources, myTargets, myArmEncoding, me, opp);
                    oppPrior = computePriorV14(s, oppSources, oppTargets, oppArmEncoding, opp, me);
                } else {
                    myPrior = computePrior(s, mySources, myTargets, myArmEncoding);
                    oppPrior = computePrior(s, oppSources, oppTargets, oppArmEncoding);
                }
            }
        }

        int selectMyUcb() {
            return select(myArmEncoding.length, myVisits, myTotal, myTotalVisits, myPrior);
        }

        int selectOppUcb() {
            return select(oppArmEncoding.length, oppVisits, oppTotal, oppTotalVisits, oppPrior);
        }

        void updateMy(int armIdx, double reward) {
            myTotal[armIdx] += reward;
            myVisits[armIdx]++;
            myTotalVisits++;
        }

        void updateOpp(int armIdx, double reward) {
            oppTotal[armIdx] += reward;
            oppVisits[armIdx]++;
            oppTotalVisits++;
        }

        Action armToAction(int armIdx, GameState s, boolean forMe) {
            int[] enc = forMe ? myArmEncoding : oppArmEncoding;
            int[] sources = forMe ? mySources : oppSources;
            int[] targets = forMe ? myTargets : oppTargets;
            Player who = forMe ? me : opp;

            int pair = enc[armIdx];
            if (pair == DO_NOTHING_PAIR) return doNothing();
            int srcIdx = (pair >>> 8) & 0xFF;
            int tgtIdx = pair & 0xFF;
            int srcPid = sources[srcIdx];
            int tgtPid = targets[tgtIdx];
            double ships = s.getPlanets().get(srcPid).getNShips() * shipsFraction;
            return new Action(who, srcPid, tgtPid, ships);
        }

        /** Robust child: pick the most-visited arm at root. */
        Action bestMyActionAtRoot(GameState s) {
            int bestIdx = -1;
            int bestVisits = 0;
            for (int i = 0; i < myArmEncoding.length; i++) {
                if (myVisits[i] > bestVisits) {
                    bestVisits = myVisits[i];
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) return doNothing();
            return armToAction(bestIdx, s, /*forMe*/ true);
        }
    }

    /**
     * Arm selection. Two modes depending on whether a prior is supplied:
     *   prior == null  -> vanilla UCB1 (Q(a) + c * sqrt(ln N / n_a)), force-explores unvisited.
     *   prior != null  -> PUCT (Q(a) + priorWeight * P(a) * sqrt(N) / (1 + n_a)). Unvisited arms
     *                     use first-play-urgency = 0; their bonus = priorWeight * P(a) * sqrt(N),
     *                     so high-prior arms are tried earlier without the UCB1 "must visit each
     *                     arm at least once" rule.
     */
    private int select(int n, int[] visits, double[] total, long totalVisits, double[] prior) {
        if (prior == null) {
            for (int i = 0; i < n; i++) if (visits[i] == 0) return i;
            int best = 0;
            double bestVal = Double.NEGATIVE_INFINITY;
            double logT = Math.log(totalVisits);
            for (int i = 0; i < n; i++) {
                double mean = total[i] / visits[i];
                double ucb = mean + ucbC * Math.sqrt(logT / visits[i]);
                if (ucb > bestVal) { bestVal = ucb; best = i; }
            }
            return best;
        }
        // PUCT
        int best = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        double sqrtN = Math.sqrt(Math.max(1L, totalVisits));
        for (int i = 0; i < n; i++) {
            double q = (visits[i] > 0) ? (total[i] / visits[i]) : 0.0;
            double bonus = priorWeight * prior[i] * sqrtN / (1.0 + visits[i]);
            double u = q + bonus;
            if (u > bestVal) { bestVal = u; best = i; }
        }
        return best;
    }

    /**
     * Heuristic prior over a player's arms, computed once when a node is populated.
     * The raw score for each (src, tgt) arm uses the same domain factors as
     * GreedyHeuristicAgent — net capture margin, distance, growth rate — but
     * combined into a per-pair score. DoNothing gets a flat low score. Softmax
     * with priorTemperature converts scores into a probability distribution.
     */
    private double[] computePrior(GameState s, int[] sources, int[] targets, int[] armEncoding) {
        double[] scores = new double[armEncoding.length];
        double speed = getParams().getTransporterSpeed();

        for (int i = 0; i < armEncoding.length; i++) {
            int pair = armEncoding[i];
            if (pair == DO_NOTHING_PAIR) {
                scores[i] = -2.0;        // mild discouragement, not exclusion
                continue;
            }
            int srcIdx = (pair >>> 8) & 0xFF;
            int tgtIdx = pair & 0xFF;
            Planet src = s.getPlanets().get(sources[srcIdx]);
            Planet tgt = s.getPlanets().get(targets[tgtIdx]);
            double distance = src.getPosition().distance(tgt.getPosition());
            double traversalTicks = distance / speed;
            double estimatedDefense = tgt.getNShips() + tgt.getGrowthRate() * traversalTicks;
            double myAttackShips = src.getNShips() * shipsFraction;
            double netCapture = myAttackShips - estimatedDefense;   // positive => can take it
            double normalizedDistance = distance / 800.0;           // typical max ~640px
            double growthBonus = tgt.getGrowthRate() * 50.0;         // up to ~10 at max growth
            scores[i] = netCapture + growthBonus - 2.0 * normalizedDistance;
        }

        // Softmax with temperature
        double maxScore = Double.NEGATIVE_INFINITY;
        for (double sc : scores) if (sc > maxScore) maxScore = sc;
        double sum = 0.0;
        double[] probs = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            probs[i] = Math.exp((scores[i] - maxScore) / priorTemperature);
            sum += probs[i];
        }
        if (sum <= 0.0) {
            // degenerate fallback: uniform
            for (int i = 0; i < probs.length; i++) probs[i] = 1.0 / probs.length;
        } else {
            for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        }
        return probs;
    }

    /**
     * v14 prior: imports the strong-heuristic scoring (weightGrowth=100,
     * preserveProductionSources, counterattackRisk, attackMomentum, gameStage
     * modulation, minTargetGrowth=0.05 filter) into PUCT.
     *
     * Per-arm score features:
     *   + net_capture                          (attack_ships - estimated_defense)
     *   + growth_bonus                         (target.growth × effWeightGrowth)
     *   - distance penalty                     (effWeightDistance × distance / 800)
     *   - preserveProductionSources penalty    (src.growth × 30)
     *   - counterattackRisk penalty            (3 × #opp planets within 300 of target)
     *   + attackMomentum bonus                 (2 × #my planets within 300 of target)
     * Targets with growth < 0.05 score very low (filter).
     * gameStage: tick<200 → weightGrowth×1.5; tick>800 → weightDistance×1.5.
     */
    private double[] computePriorV14(GameState s, int[] sources, int[] targets, int[] armEncoding,
                                      Player forPlayer, Player otherPlayer) {
        double[] scores = new double[armEncoding.length];
        double speed = getParams().getTransporterSpeed();

        double effWeightGrowth = 100.0;
        double effWeightDistance = 2.0;
        int tick = s.getGameTick();
        if (tick < 200) effWeightGrowth *= 1.5;
        else if (tick > 800) effWeightDistance *= 1.5;

        for (int i = 0; i < armEncoding.length; i++) {
            int pair = armEncoding[i];
            if (pair == DO_NOTHING_PAIR) {
                scores[i] = -2.0;
                continue;
            }
            int srcIdx = (pair >>> 8) & 0xFF;
            int tgtIdx = pair & 0xFF;
            Planet src = s.getPlanets().get(sources[srcIdx]);
            Planet tgt = s.getPlanets().get(targets[tgtIdx]);

            // minTargetGrowth=0.05 filter — drop very low-growth targets to near-zero probability
            if (tgt.getGrowthRate() < 0.05) {
                scores[i] = -20.0;
                continue;
            }

            double distance = src.getPosition().distance(tgt.getPosition());
            double traversalTicks = distance / speed;
            double estimatedDefense = tgt.getNShips() + tgt.getGrowthRate() * traversalTicks;
            double myAttackShips = src.getNShips() * shipsFraction;
            double netCapture = myAttackShips - estimatedDefense;
            double normalizedDistance = distance / 800.0;
            double growthBonus = tgt.getGrowthRate() * effWeightGrowth;

            double score = netCapture + growthBonus - effWeightDistance * normalizedDistance;

            // preserveProductionSources
            score -= src.getGrowthRate() * 30.0;

            // counterattackRisk: opp neighbors of target
            int oppNearby = 0;
            for (Planet p : s.getPlanets()) {
                if (p.getOwner() != otherPlayer) continue;
                if (p == tgt) continue;
                if (p.getPosition().distance(tgt.getPosition()) < 300.0) oppNearby++;
            }
            score -= oppNearby * 3.0;

            // attackMomentum: forPlayer neighbors of target
            int myNearby = 0;
            for (Planet p : s.getPlanets()) {
                if (p.getOwner() != forPlayer) continue;
                if (p == src) continue;
                if (p.getPosition().distance(tgt.getPosition()) < 300.0) myNearby++;
            }
            score += myNearby * 2.0;

            scores[i] = score;
        }

        double maxScore = Double.NEGATIVE_INFINITY;
        for (double sc : scores) if (sc > maxScore) maxScore = sc;
        double sum = 0.0;
        double[] probs = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            probs[i] = Math.exp((scores[i] - maxScore) / priorTemperature);
            sum += probs[i];
        }
        if (sum <= 0.0) {
            for (int i = 0; i < probs.length; i++) probs[i] = 1.0 / probs.length;
        } else {
            for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        }
        return probs;
    }

    /** Build the flat arm-encoding list: one entry per (src,tgt) pair, plus DoNothing at the end. */
    private static int[] enumerateArms(int sourceCount, int targetCount) {
        if (sourceCount == 0 || targetCount == 0) {
            return new int[]{DO_NOTHING_PAIR};
        }
        int[] arms = new int[sourceCount * targetCount + 1];
        int k = 0;
        for (int s = 0; s < sourceCount; s++) {
            for (int t = 0; t < targetCount; t++) {
                arms[k++] = ((s & 0xFF) << 8) | (t & 0xFF);
            }
        }
        arms[k] = DO_NOTHING_PAIR;
        return arms;
    }

    // -------- Legal action enumeration --------

    private static int[] legalSources(GameState s, Player who) {
        int n = 0;
        for (Planet p : s.getPlanets()) {
            if (p.getOwner() == who && p.getTransporter() == null && p.getNShips() >= 1.0) n++;
        }
        int[] out = new int[n];
        int i = 0;
        for (Planet p : s.getPlanets()) {
            if (p.getOwner() == who && p.getTransporter() == null && p.getNShips() >= 1.0) {
                out[i++] = p.getId();
            }
        }
        return out;
    }

    /** Targets for player who: opponent planets first, neutrals only as fallback. */
    private static int[] legalOppTargetsFor(GameState s, Player who) {
        Player otherSide = (who == Player.Player1) ? Player.Player2 : Player.Player1;
        ArrayList<Integer> oppPlanets = new ArrayList<>();
        ArrayList<Integer> neutrals = new ArrayList<>();
        for (Planet p : s.getPlanets()) {
            if (p.getOwner() == otherSide) oppPlanets.add(p.getId());
            else if (p.getOwner() == Player.Neutral) neutrals.add(p.getId());
        }
        ArrayList<Integer> chosen = !oppPlanets.isEmpty() ? oppPlanets : neutrals;
        int[] out = new int[chosen.size()];
        for (int i = 0; i < chosen.size(); i++) out[i] = chosen.get(i);
        return out;
    }
}
