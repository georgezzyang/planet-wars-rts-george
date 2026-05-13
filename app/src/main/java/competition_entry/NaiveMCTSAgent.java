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
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * NaiveMCTS for Planet Wars (Ontanon 2013, Algorithm 3) - faithful tree-based version.
 *
 * Tree structure
 * --------------
 * Each node = a unique state reached by a particular sequence of joint actions from
 * the root. State is NOT stored at the node; it is reconstructed each iteration by
 * deepCopying the root state and re-applying the path's joint actions through the
 * forward model. This trades CPU (cheap forward model) for memory.
 *
 * Each node holds an INDEPENDENT NaiveSampling state for BOTH players (decoupled UCT
 * for simultaneous-move games).
 *
 * NaiveSampling (per player, per node)
 * ------------------------------------
 *   sourceMab : UCB1 over my legal source planets, plus one extra "DoNothing" arm at
 *               the end (index = sourceCount).
 *   targetMab : UCB1 over candidate target planets.
 *   globalArms: map from encoded (srcIdx, tgtIdx) pair -> ArmStats. Records the
 *               (src,tgt) pairs actually sampled so far at this node.
 *   epsilon-greedy mix: with prob epsilon EXPLORE via the local MABs (lets new pairs
 *               appear), else EXPLOIT by UCB1 over the global pair MAB.
 *
 * Action encoding within a node
 * -----------------------------
 *   Per-player arm: (srcIdx in low 8 bits) | (tgtIdx in high 8 bits) ... actually
 *                   stored as ((srcIdx & 0xFF) << 8) | (tgtIdx & 0xFF).
 *   srcIdx == sourceCount  ->  DoNothing (tgtIdx ignored, normalised to 0 for keying).
 *   Joint child key:  (myArm << 16) | oppArm.
 *
 * Iteration
 * ---------
 *   1. SELECT: descend the tree by NaiveSampling each player at every node.
 *   2. EXPAND: when the joint child does not exist, create one (single-node expansion
 *              per iteration, standard MCTS).
 *   3. ROLLOUT: from the leaf, both players play the heuristic policy
 *               (GreedyHeuristicAgent) until rolloutSteps or terminal.
 *   4. BACKPROP: for every node along the path, update visits + totalReward, and
 *                update both per-player MABs with the joint arm taken from that node.
 *                Opponent's MABs are updated with NEGATED reward (zero-sum minimax:
 *                opponent wants to minimize my reward).
 *
 * Final action selection at root: ROBUST CHILD = highest visit count over root's
 * myGlobal arms. Standard MCTS choice; more stable than max mean reward when arms
 * have wildly different visit counts.
 */
public class NaiveMCTSAgent extends PlanetWarsPlayer {

    // ---- Hyperparameters ----
    public long timeBudgetMillis = 45;
    public int rolloutSteps = 30;
    public int maxTreeDepth = 8;             // safety cap on tree descent
    public double epsilonGlobal = 0.3;       // probability of exploring via local MABs
    public double ucbC = Math.sqrt(2.0);     // UCB1 exploration constant
    public double shipsFraction = 0.5;       // fraction of source ships to send

    // ---- Models reused across calls (set up once) ----
    private final PlanetWarsAgent opponentRolloutPolicy = new GreedyHeuristicAgent();
    private final PlanetWarsAgent myRolloutPolicy = new GreedyHeuristicAgent();
    private boolean modelsReady = false;

    // ---- Per-call (kept as fields so inner Node can read me/opp) ----
    private Player me;
    private Player opp;

    @Override
    public Action getAction(GameState rootState) {
        long deadline = System.currentTimeMillis() + timeBudgetMillis;
        me = getPlayer();
        opp = (me == Player.Player1) ? Player.Player2 : Player.Player1;
        if (!modelsReady) {
            opponentRolloutPolicy.prepareToPlayAs(opp, getParams(), "internal-rollout-opp");
            myRolloutPolicy.prepareToPlayAs(me, getParams(), "internal-rollout-me");
            modelsReady = true;
        }

        Node root = new Node();
        root.populateLegal(rootState);
        if (root.mySources.length == 0) return doNothing();

        Random rng = ThreadLocalRandom.current();
        while (System.currentTimeMillis() < deadline) {
            iterate(root, rootState, rng);
        }
        return root.bestMyActionAtRoot(rootState);
    }

    /** Single MCTS iteration: select -> expand -> rollout -> backprop. */
    private void iterate(Node root, GameState rootState, Random rng) {
        GameState sim = rootState.deepCopy();
        ForwardModel model = new ForwardModel(sim, getParams());

        ArrayList<Node> path = new ArrayList<>(maxTreeDepth + 1);
        ArrayList<int[]> arms = new ArrayList<>(maxTreeDepth);  // each entry: [myArm, oppArm]
        path.add(root);

        Node node = root;
        int depth = 0;
        while (depth < maxTreeDepth && !model.isTerminal()) {
            int myArm = node.naiveSampleMy(rng);
            int oppArm = node.naiveSampleOpp(rng);
            arms.add(new int[]{myArm, oppArm});

            applyJoint(model, sim, node, myArm, oppArm);

            int childKey = (myArm << 16) | (oppArm & 0xFFFF);
            Node child = node.children.get(childKey);
            if (child == null) {
                // EXPANSION: one new node per iteration.
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

        // ROLLOUT from current sim state with heuristic policy on both sides.
        for (int t = 0; t < rolloutSteps && !model.isTerminal(); t++) {
            HashMap<Player, Action> joint = new HashMap<>(2);
            joint.put(me, myRolloutPolicy.getAction(sim));
            joint.put(opp, opponentRolloutPolicy.getAction(sim));
            model.step(joint);
        }
        double reward = model.getShips(me) - model.getShips(opp);

        // BACKPROP: update visits + totalReward at every node, MABs at every node where an
        // action was taken (i.e. all nodes except possibly the leaf-after-expansion).
        for (int i = 0; i < path.size(); i++) {
            Node n = path.get(i);
            n.visits++;
            n.totalReward += reward;
            if (i < arms.size()) {
                int[] a = arms.get(i);
                n.updateMy(a[0], reward);
                n.updateOpp(a[1], -reward);     // opponent minimizes my reward (zero-sum)
            }
        }
    }

    private void applyJoint(ForwardModel model, GameState sim, Node node, int myArm, int oppArm) {
        Action myAction = node.armToAction(myArm, sim, /*forMe*/ true);
        Action oppAction = node.armToAction(oppArm, sim, /*forMe*/ false);
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
        return "NaiveMCTS-tree-v1";
    }

    // =====================================================================
    // Inner classes
    // =====================================================================

    /** One node in the joint-action tree. State is not stored - reconstructed per iteration. */
    private final class Node {
        // Cached on first visit
        int[] mySources, myTargets;
        int[] oppSources, oppTargets;

        // NaiveSampling state for ME
        Mab mySrcMab, myTgtMab;
        HashMap<Integer, ArmStats> myGlobal;
        long myGlobalVisits;

        // NaiveSampling state for OPP
        Mab oppSrcMab, oppTgtMab;
        HashMap<Integer, ArmStats> oppGlobal;
        long oppGlobalVisits;

        // Tree
        final HashMap<Integer, Node> children = new HashMap<>();
        int visits;
        double totalReward;

        void populateLegal(GameState s) {
            mySources = legalSources(s, me);
            myTargets = legalOppTargetsFor(s, me);
            oppSources = legalSources(s, opp);
            oppTargets = legalOppTargetsFor(s, opp);

            // +1 in source MAB for the explicit DoNothing arm at index = sourceCount.
            mySrcMab = new Mab(mySources.length + 1);
            myTgtMab = new Mab(Math.max(1, myTargets.length));
            myGlobal = new HashMap<>();
            myGlobalVisits = 0;

            oppSrcMab = new Mab(oppSources.length + 1);
            oppTgtMab = new Mab(Math.max(1, oppTargets.length));
            oppGlobal = new HashMap<>();
            oppGlobalVisits = 0;
        }

        int naiveSampleMy(Random rng) {
            return naiveSample(mySrcMab, myTgtMab, myGlobal, myGlobalVisits, mySources.length, rng);
        }

        int naiveSampleOpp(Random rng) {
            return naiveSample(oppSrcMab, oppTgtMab, oppGlobal, oppGlobalVisits, oppSources.length, rng);
        }

        void updateMy(int arm, double reward) {
            int srcIdx = (arm >>> 8) & 0xFF;
            int tgtIdx = arm & 0xFF;
            mySrcMab.update(srcIdx, reward);
            if (srcIdx < mySources.length) {
                myTgtMab.update(tgtIdx, reward);
            }
            int normKey = normalizeArmKey(srcIdx, tgtIdx, mySources.length);
            myGlobal.computeIfAbsent(normKey, k -> new ArmStats()).update(reward);
            myGlobalVisits++;
        }

        void updateOpp(int arm, double reward) {
            int srcIdx = (arm >>> 8) & 0xFF;
            int tgtIdx = arm & 0xFF;
            oppSrcMab.update(srcIdx, reward);
            if (srcIdx < oppSources.length) {
                oppTgtMab.update(tgtIdx, reward);
            }
            int normKey = normalizeArmKey(srcIdx, tgtIdx, oppSources.length);
            oppGlobal.computeIfAbsent(normKey, k -> new ArmStats()).update(reward);
            oppGlobalVisits++;
        }

        Action armToAction(int arm, GameState s, boolean forMe) {
            int srcIdx = (arm >>> 8) & 0xFF;
            int tgtIdx = arm & 0xFF;
            int[] sources = forMe ? mySources : oppSources;
            int[] targets = forMe ? myTargets : oppTargets;
            Player who = forMe ? me : opp;
            if (srcIdx >= sources.length) return doNothing();
            if (targets.length == 0) return doNothing();
            int srcPid = sources[srcIdx];
            int tgtPid = targets[Math.min(tgtIdx, targets.length - 1)];
            double ships = s.getPlanets().get(srcPid).getNShips() * shipsFraction;
            return new Action(who, srcPid, tgtPid, ships);
        }

        Action bestMyActionAtRoot(GameState s) {
            // Robust child: highest visit count over global arms at root.
            int bestArm = -1;
            int bestVisits = 0;
            for (Map.Entry<Integer, ArmStats> e : myGlobal.entrySet()) {
                if (e.getValue().visits > bestVisits) {
                    bestVisits = e.getValue().visits;
                    bestArm = e.getKey();
                }
            }
            if (bestArm < 0) return doNothing();
            return armToAction(bestArm, s, /*forMe*/ true);
        }
    }

    /** NaiveSampling at one node for one player. Returns encoded (srcIdx<<8 | tgtIdx). */
    private int naiveSample(Mab srcMab, Mab tgtMab,
                            HashMap<Integer, ArmStats> global, long globalVisits,
                            int sourceCount, Random rng) {
        int srcArm, tgtArm;
        if (global.isEmpty() || rng.nextDouble() < epsilonGlobal) {
            // EXPLORE: independent UCB1 on each factor MAB
            srcArm = srcMab.selectUcb(ucbC);
            tgtArm = tgtMab.selectUcb(ucbC);
        } else {
            // EXPLOIT: UCB1 over (src,tgt) pairs already sampled at this node
            int bestKey = -1;
            double bestVal = Double.NEGATIVE_INFINITY;
            double logV = Math.log(Math.max(1L, globalVisits));
            for (Map.Entry<Integer, ArmStats> e : global.entrySet()) {
                ArmStats st = e.getValue();
                double ucb = (st.totalReward / st.visits) + ucbC * Math.sqrt(logV / st.visits);
                if (ucb > bestVal) { bestVal = ucb; bestKey = e.getKey(); }
            }
            srcArm = (bestKey >>> 8) & 0xFF;
            tgtArm = bestKey & 0xFF;
        }
        return ((srcArm & 0xFF) << 8) | (tgtArm & 0xFF);
    }

    /** Normalize the arm key so all DoNothing samples (any tgtIdx) collapse to one global entry. */
    private static int normalizeArmKey(int srcIdx, int tgtIdx, int sourceCount) {
        if (srcIdx >= sourceCount) {
            return (sourceCount & 0xFF) << 8;     // tgtIdx forced to 0
        }
        return ((srcIdx & 0xFF) << 8) | (tgtIdx & 0xFF);
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

    /** Targets for player who: opponent's planets first, neutrals only as fallback. */
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

    // -------- Inner data --------

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
