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

/**
 * Flat Monte Carlo with heuristic-driven candidate pruning.
 *
 * Why this exists, in one sentence: combine our strongest standalone heuristic (v14, the
 * 83.3% one) with shallow simulation to refine decisions near the heuristic's blind spots,
 * without paying the cost of a full UCT tree.
 *
 * Algorithm
 * ---------
 *   1. Enumerate (src, tgt) candidate actions, same set as HeuristicAgent.
 *   2. Score every candidate with the heuristic, keep the top-K (K small).
 *   3. Distribute the remaining time budget uniformly over the top-K. For each candidate:
 *        - simulate(state, candidate_action) for `rolloutSteps` ticks using HeuristicAgent
 *          as the rollout policy for BOTH players.
 *        - record final ship-diff × time-discount.
 *   4. Pick the candidate with the highest mean reward.
 *
 * Relationship to other agents
 * ----------------------------
 *   - HeuristicAgent: step 1+2 alone, no simulation. Our 83.3%/82.8% baseline.
 *   - FlatNaiveSamplingAgent: like this, but with NaiveSampling bandit + UCB1 over factored
 *     (src, tgt) MABs. Has factorisation bias.
 *   - UCTAgent (puct mode): tree + PUCT prior. Best ceiling but most complex.
 *   - FlatMCAgent (this): heuristic for pruning, flat rollouts for refinement. Bridge.
 */
public class FlatMCAgent extends PlanetWarsPlayer {

    public long timeBudgetMillis = 45;
    public int rolloutSteps = 30;
    public int topK = 8;              // number of heuristic-best candidates to simulate
    public double shipsFraction = 0.5;
    public double timeDiscountGamma = 0.99;
    public double timeDiscountScale = 10.0;

    /** Heuristic used for (a) ranking candidates, (b) BOTH-sides rollout policy. */
    private final HeuristicAgent heuristicMe = makeStrongHeuristic();
    private final HeuristicAgent heuristicOpp = makeStrongHeuristic();
    private Player rolloutPoliciesPreparedFor = null;

    private static HeuristicAgent makeStrongHeuristic() {
        // v14 best: minTargetGrowth=0.05 with v11 features
        HeuristicAgent h = new HeuristicAgent();
        h.minTargetGrowth = 0.05;
        h.weightGrowth = 100.0;
        h.preserveProductionSources = true;
        h.counterattackRisk = true;
        h.attackMomentum = true;
        h.gameStageAwareness = true;
        h.threatResponse = true;
        return h;
    }

    @Override
    public Action getAction(GameState state) {
        long deadline = System.currentTimeMillis() + timeBudgetMillis;
        Player me = getPlayer();
        Player opp = (me == Player.Player1) ? Player.Player2 : Player.Player1;

        if (rolloutPoliciesPreparedFor != me) {
            heuristicMe.prepareToPlayAs(me, getParams(), "flatmc-me");
            heuristicOpp.prepareToPlayAs(opp, getParams(), "flatmc-opp");
            rolloutPoliciesPreparedFor = me;
        }

        // Step 1: enumerate candidate actions
        List<Action> candidates = enumerateCandidates(state, me, opp);
        if (candidates.isEmpty()) return doNothing();

        // Step 2: score with heuristic, keep top-K
        // Use the heuristic's getAction on a series of mutated states isn't trivial; instead
        // we score each candidate using the same per-arm scoring logic as HeuristicAgent
        // exposes via its scoreCandidate helper, then keep the K highest.
        List<ScoredAction> scored = new ArrayList<>();
        for (Action a : candidates) {
            double s = scoreCandidate(state, a, me);
            scored.add(new ScoredAction(a, s));
        }
        scored.sort((x, y) -> Double.compare(y.score, x.score));
        int k = Math.min(topK, scored.size());
        if (k == 1) return scored.get(0).action;     // only one candidate, no simulation needed

        // Step 3: distribute rollouts across top-K
        double[] sumReward = new double[k];
        int[] simCount = new int[k];
        int round = 0;
        while (System.currentTimeMillis() < deadline) {
            // round-robin over the K candidates so each gets balanced sims
            int idx = round % k;
            double r = simulate(state, scored.get(idx).action, me, opp);
            sumReward[idx] += r;
            simCount[idx]++;
            round++;
        }

        // Step 4: pick candidate with highest mean reward (fall back to heuristic order on ties)
        int best = 0;
        double bestMean = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            if (simCount[i] == 0) continue;
            double mean = sumReward[i] / simCount[i];
            if (mean > bestMean) { bestMean = mean; best = i; }
        }
        return scored.get(best).action;
    }

    /** Enumerate (src, tgt) candidate actions following the same target restriction as HeuristicAgent. */
    private List<Action> enumerateCandidates(GameState state, Player me, Player opp) {
        List<Planet> mySources = new ArrayList<>();
        List<Planet> oppTargets = new ArrayList<>();
        List<Planet> neutralTargets = new ArrayList<>();
        for (Planet p : state.getPlanets()) {
            if (p.getOwner() == me && p.getTransporter() == null && p.getNShips() >= 1.0) {
                mySources.add(p);
            } else if (p.getOwner() == opp) {
                oppTargets.add(p);
            } else if (p.getOwner() == Player.Neutral) {
                neutralTargets.add(p);
            }
        }
        List<Planet> targets = !oppTargets.isEmpty() ? oppTargets : neutralTargets;
        if (mySources.isEmpty() || targets.isEmpty()) return new ArrayList<>();

        List<Action> out = new ArrayList<>();
        for (Planet src : mySources) {
            for (Planet tgt : targets) {
                double ships = src.getNShips() * shipsFraction;
                out.add(new Action(me, src.getId(), tgt.getId(), ships));
            }
        }
        return out;
    }

    /** Quick heuristic score: net_capture + growth_bonus - distance_penalty (v11/v14 defaults). */
    private double scoreCandidate(GameState state, Action a, Player me) {
        Planet src = state.getPlanets().get(a.getSourcePlanetId());
        Planet tgt = state.getPlanets().get(a.getDestinationPlanetId());
        double distance = src.getPosition().distance(tgt.getPosition());
        double speed = getParams().getTransporterSpeed();
        double traversalTicks = distance / speed;
        double defense = tgt.getNShips() + tgt.getGrowthRate() * traversalTicks;
        double attackShips = src.getNShips() * shipsFraction;
        double netCapture = attackShips - defense;
        double normDist = distance / 800.0;
        double growthBonus = tgt.getGrowthRate() * 100.0;       // matches v14's weightGrowth=100
        return netCapture + growthBonus - 2.0 * normDist;
    }

    /** Single simulation: apply candidate action + opponent heuristic action, roll out with
     *  HeuristicAgent for both sides, return time-discounted ship diff. */
    private double simulate(GameState rootState, Action firstAction, Player me, Player opp) {
        GameState sim = rootState.deepCopy();
        ForwardModel model = new ForwardModel(sim, getParams());
        int startTick = sim.getGameTick();

        Action firstOppAction = heuristicOpp.getAction(sim);
        Map<Player, Action> joint = new HashMap<>(2);
        joint.put(me, firstAction);
        joint.put(opp, firstOppAction);
        model.step(joint);

        for (int t = 1; t < rolloutSteps && !model.isTerminal(); t++) {
            joint.clear();
            joint.put(me, heuristicMe.getAction(sim));
            joint.put(opp, heuristicOpp.getAction(sim));
            model.step(joint);
        }

        double rawReward = model.getShips(me) - model.getShips(opp);
        int elapsed = sim.getGameTick() - startTick;
        return rawReward * Math.pow(timeDiscountGamma, elapsed / timeDiscountScale);
    }

    private static Action doNothing() {
        return new Action(Player.Neutral, -1, -1, 0.0);
    }

    @Override
    public String getAgentType() {
        return "FlatMC-topK" + topK + "-v1";
    }

    private static final class ScoredAction {
        final Action action;
        final double score;
        ScoredAction(Action a, double s) { action = a; score = s; }
    }
}
