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
 * Flat NaiveMCTS for Planet Wars.
 *
 * The action space is decomposed into two factors:
 *   - source planet (mine, no active transporter)
 *   - target planet (anyone but mine)
 * Each tick we pick exactly ONE (source, target) pair for the player. Because
 * the per-tick joint action is just "one transporter launch", a tree-MCTS gives
 * very few visits per tree branch within 50 ms. Flat MCTS spends the entire
 * budget on the root decision, which is the right trade-off here.
 *
 * Three MABs (Naive Sampling, Ontanon 2013):
 *   - source MAB        — UCB1 over my legal source planets
 *   - target MAB        — UCB1 over candidate target planets
 *   - global pair MAB   — UCB1 over (src, tgt) pairs that have actually been sampled
 *
 * ε-greedy at the top: with prob ε explore via the local MABs (lets new pairs
 * appear), else exploit the best global arm so far. This is the canonical
 * Naive Sampling structure adapted for our single-action setting.
 */
public class NaiveMCTSAgent extends PlanetWarsPlayer {

    public long timeBudgetMillis = 45;
    public int rolloutSteps = 30;             // shorter horizon now that rollouts use a meaningful policy
    public double epsilonGlobal = 0.3;
    public double ucbC = Math.sqrt(2.0);
    public double shipsFraction = 0.5;

    /** Opponent model used inside rollouts. GreedyHeuristic is ~70% vs random — a far more realistic stand-in than DoNothing. */
    private final PlanetWarsAgent opponentModel = new GreedyHeuristicAgent();
    /** Policy used for ME during rollouts after the first chosen action. Same script — both sides play greedy. */
    private final PlanetWarsAgent myRolloutPolicy = new GreedyHeuristicAgent();
    private boolean modelsReady = false;

    @Override
    public Action getAction(GameState state) {
        long deadline = System.currentTimeMillis() + timeBudgetMillis;

        Player me = getPlayer();
        Player opp = (me == Player.Player1) ? Player.Player2 : Player.Player1;
        if (!modelsReady) {
            opponentModel.prepareToPlayAs(opp, getParams(), "internal-rollout-opp");
            myRolloutPolicy.prepareToPlayAs(me, getParams(), "internal-rollout-me");
            modelsReady = true;
        }

        // Targets RESTRICTED to opponent-owned planets. Local benchmark showed CarefulRandomAgent
        // (opponent-only targeting) at 62% vs BetterRandomAgent (opponent+neutral) at 6% — a
        // 56-point gap purely from the target filter. Attacking neutrals is a strategic trap with
        // these game params: you pay the garrison cost, then the captured planet sits at 0 ships
        // while the opponent keeps growing.
        // Fallback: if opponent has zero planets we've won; if we still have free transporters with
        // no opponent target, allow attacking neutrals as a last resort to keep the game moving.
        List<Integer> sourceIds = new ArrayList<>();
        List<Integer> targetIds = new ArrayList<>();
        List<Integer> neutralIds = new ArrayList<>();
        for (Planet p : state.getPlanets()) {
            if (p.getOwner() == me && p.getTransporter() == null && p.getNShips() >= 1.0) {
                sourceIds.add(p.getId());
            } else if (p.getOwner() == opp) {
                targetIds.add(p.getId());
            } else if (p.getOwner() == Player.Neutral) {
                neutralIds.add(p.getId());
            }
        }
        if (sourceIds.isEmpty()) return doNothing();
        if (targetIds.isEmpty()) {
            if (neutralIds.isEmpty()) return doNothing();
            targetIds = neutralIds; // last-resort fallback
        }

        Mab sourceMab = new Mab(sourceIds.size());
        Mab targetMab = new Mab(targetIds.size());
        Map<Long, ArmStats> globalArms = new HashMap<>();
        long globalVisits = 0;

        Random rng = ThreadLocalRandom.current();
        Map<Player, Action> jointBuf = new HashMap<>(2);

        while (System.currentTimeMillis() < deadline) {
            int srcIdx;
            int tgtIdx;
            if (globalArms.isEmpty() || rng.nextDouble() < epsilonGlobal) {
                srcIdx = sourceMab.selectUcb(ucbC);
                tgtIdx = targetMab.selectUcb(ucbC);
            } else {
                long bestKey = -1L;
                double bestVal = Double.NEGATIVE_INFINITY;
                double logV = Math.log(Math.max(1L, globalVisits));
                for (Map.Entry<Long, ArmStats> e : globalArms.entrySet()) {
                    ArmStats s = e.getValue();
                    double ucb = (s.totalReward / s.visits) + ucbC * Math.sqrt(logV / s.visits);
                    if (ucb > bestVal) { bestVal = ucb; bestKey = e.getKey(); }
                }
                srcIdx = (int) (bestKey >>> 32);
                tgtIdx = (int) (bestKey & 0xFFFFFFFFL);
            }

            int srcPid = sourceIds.get(srcIdx);
            int tgtPid = targetIds.get(tgtIdx);

            double reward = simulate(state, me, opp, srcPid, tgtPid, rng, jointBuf);

            sourceMab.update(srcIdx, reward);
            targetMab.update(tgtIdx, reward);
            long key = (((long) srcIdx) << 32) | ((long) tgtIdx & 0xFFFFFFFFL);
            globalArms.computeIfAbsent(key, k -> new ArmStats()).update(reward);
            globalVisits++;
        }

        long bestKey = -1L;
        double bestMean = Double.NEGATIVE_INFINITY;
        int bestVisits = 0;
        for (Map.Entry<Long, ArmStats> e : globalArms.entrySet()) {
            ArmStats s = e.getValue();
            if (s.visits < 2) continue;
            double mean = s.totalReward / s.visits;
            if (mean > bestMean) { bestMean = mean; bestKey = e.getKey(); bestVisits = s.visits; }
        }
        if (bestKey < 0L) {
            for (Map.Entry<Long, ArmStats> e : globalArms.entrySet()) {
                if (e.getValue().visits > bestVisits) {
                    bestVisits = e.getValue().visits;
                    bestKey = e.getKey();
                }
            }
        }
        if (bestKey < 0L) return doNothing();

        int srcIdx = (int) (bestKey >>> 32);
        int tgtIdx = (int) (bestKey & 0xFFFFFFFFL);
        Planet srcPlanet = state.getPlanets().get(sourceIds.get(srcIdx));
        double ships = srcPlanet.getNShips() * shipsFraction;
        return new Action(me, sourceIds.get(srcIdx), targetIds.get(tgtIdx), ships);
    }

    private double simulate(GameState rootState, Player me, Player opp,
                            int srcId, int tgtId,
                            @SuppressWarnings("unused") Random rng,
                            Map<Player, Action> jointBuf) {
        GameState sim = rootState.deepCopy();
        ForwardModel model = new ForwardModel(sim, getParams());

        Planet src = sim.getPlanets().get(srcId);
        double ships = src.getNShips() * shipsFraction;
        Action myAction = new Action(me, srcId, tgtId, ships);
        Action oppAction = opponentModel.getAction(sim);

        jointBuf.clear();
        jointBuf.put(me, myAction);
        jointBuf.put(opp, oppAction);
        model.step(jointBuf);

        for (int t = 1; t < rolloutSteps && !model.isTerminal(); t++) {
            jointBuf.clear();
            jointBuf.put(me, myRolloutPolicy.getAction(sim));
            jointBuf.put(opp, opponentModel.getAction(sim));
            model.step(jointBuf);
        }
        return model.getShips(me) - model.getShips(opp);
    }

    private static Action doNothing() {
        return new Action(Player.Neutral, -1, -1, 0.0);
    }

    @Override
    public String getAgentType() {
        return "NaiveMCTS-flat-v3";
    }

    private static final class ArmStats {
        double totalReward = 0;
        int visits = 0;
        void update(double r) { totalReward += r; visits += 1; }
    }

    private static final class Mab {
        final int n;
        final double[] total;
        final int[] visits;
        long totalVisits = 0;
        Mab(int n) { this.n = n; total = new double[n]; visits = new int[n]; }
        int selectUcb(double C) {
            for (int i = 0; i < n; i++) if (visits[i] == 0) return i;
            int best = 0;
            double bestVal = Double.NEGATIVE_INFINITY;
            double logT = Math.log(totalVisits);
            for (int i = 0; i < n; i++) {
                double ucb = (total[i] / visits[i]) + C * Math.sqrt(logT / visits[i]);
                if (ucb > bestVal) { bestVal = ucb; best = i; }
            }
            return best;
        }
        void update(int i, double r) { total[i] += r; visits[i] += 1; totalVisits++; }
    }
}
