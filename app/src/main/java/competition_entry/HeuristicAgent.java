package competition_entry;

import games.planetwars.agents.Action;
import games.planetwars.agents.PlanetWarsPlayer;
import games.planetwars.core.GameState;
import games.planetwars.core.Planet;
import games.planetwars.core.Player;

import java.util.ArrayList;
import java.util.List;

/**
 * Standalone heuristic agent. NO MCTS, NO rollout, NO search.
 *
 * Why this exists
 * ---------------
 * The PUCT prior in {@link UCTAgent} computes a per-arm score that biases tree
 * expansion. To iterate on that scoring formula we don't actually need the
 * search loop in the way — testing the scoring as a standalone deterministic
 * agent is ~25x faster per benchmark (seconds-to-minutes vs ~25 min with MCTS).
 *
 * Once the standalone heuristic is dialed in, drop the same scoring formula
 * back into {@link UCTAgent#computePrior} for compound gains.
 *
 * Algorithm
 * ---------
 *   1. Enumerate (src, tgt) pairs: my planets with free transporters, opponent
 *      planets as targets (neutral fallback).
 *   2. Score every pair with {@link #scoreAction}.
 *   3. Pick the highest-scoring pair. If best score is hopelessly negative,
 *      return DoNothing.
 *   4. Determine ship count (half by default; "just enough" if smartShipCount).
 *
 * Compared to the existing GreedyHeuristicAgent
 * ---------------------------------------------
 * - Considers (src, tgt) jointly instead of "pick max-ships source, then pick
 *   best target from there". A long-range strong source might be worse than
 *   a close-range medium source against a particular target.
 * - Target candidates restricted to opponents (with neutral fallback) instead
 *   of always including neutrals. Earlier ablation showed this is a +56-point
 *   difference (CarefulRandom vs BetterRandom).
 * - Additive higher-is-better score with explicit per-feature weights, so we
 *   can ablate any single feature by setting its weight to 0.
 *
 * Public fields are intentionally tunable so the benchmark can spin up
 * multiple variants in one league.
 */
public class HeuristicAgent extends PlanetWarsPlayer {

    // ---- v1 Feature flags ----
    public boolean smartShipCount = false;            // "just enough" — found to HURT in v1 league
    public boolean awareOfIncoming = false;           // -- HURT in v1 league
    public boolean sourceVulnerabilityPenalty = false;

    // ---- v2 Feature flags ----
    /** "Better" smart ship count: send max(half, defense_estimate + postCaptureGarrison).
     *  Always at least half, but MORE if defense requires.  Solves the v1 smart-mode failure
     *  where light captures left zero-garrison planets vulnerable to immediate counter-attack. */
    public boolean strongerSmartShipCount = false;
    public double postCaptureGarrison = 5.0;

    /** Require source to have at least this many ships to be considered as a launch source.
     *  GreedyHeuristic uses 10; we default to 1 (any free planet).  Set to 10 to ablate. */
    public double minSourceShips = 1.0;

    /** Dynamic ship fraction based on lead/deficit.  When leading by 2x ships, send less
     *  (conserve, consolidate).  When behind by 2x, send more (catch up via aggression). */
    public boolean strategicContext = false;

    // ---- v3 Feature flags ----
    /** Require target to have ≥ this growth rate. Filters out low-value captures. 0 = no filter. */
    public double minTargetGrowth = 0.0;

    /** Stricter abort: also DoNothing if best (src,tgt) score is positive but BELOW this.
     *  abortThreshold is already in place; this is a tightening that demands a "good enough"
     *  action, not just "least bad". Default 0 = act on any score above existing abortThreshold. */
    public double minActionScore = Double.NEGATIVE_INFINITY;

    /** Skip target if any of OUR ships are already en-route to it (don't double-attack a target
     *  that we're already capturing). */
    public boolean skipAlreadyAttacked = false;

    /** Skip source if it's the closest planet to any opponent planet (don't drain frontline). */
    public boolean preserveFrontline = false;

    // ---- v7 feature flags (new mechanisms) ----
    /** Bonus to targets that multiple of my sources can converge on (within 1.2x of best dist).
     *  Rewards coordinated multi-source captures of high-defense targets. */
    public boolean multiSourcePower = false;
    public double multiSourceWeight = 5.0;

    /** Penalize sending from high-growth sources (production hubs). */
    public boolean preserveProductionSources = false;
    public double productionPenaltyWeight = 30.0;

    /** Replace the fixed 50-tick growth horizon with a distance-discounted estimate of
     *  game time remaining after the transporter arrives. More principled bonus. */
    public boolean distanceDiscountedGrowth = false;
    public double estimatedGameTime = 600.0;

    // ---- Tunable weights ----
    public double weightNetCapture = 1.0;
    public double weightGrowth = 50.0;
    public double weightDistance = 2.0;

    // ---- Behaviour constants ----
    public double shipsFraction = 0.5;       // baseline ship fraction
    public double safetyMargin = 1.0;        // extra ships beyond defense_estimate (smart mode)
    public double abortThreshold = -10.0;    // if best score < this, DoNothing

    @Override
    public Action getAction(GameState state) {
        Player me = getPlayer();
        Player opp = (me == Player.Player1) ? Player.Player2 : Player.Player1;

        List<Planet> mySources = new ArrayList<>();
        List<Planet> oppTargets = new ArrayList<>();
        List<Planet> neutralTargets = new ArrayList<>();
        for (Planet p : state.getPlanets()) {
            if (p.getOwner() == me && p.getTransporter() == null && p.getNShips() >= minSourceShips) {
                mySources.add(p);
            } else if (p.getOwner() == opp) {
                oppTargets.add(p);
            } else if (p.getOwner() == Player.Neutral) {
                neutralTargets.add(p);
            }
        }
        if (mySources.isEmpty()) return doNothing();

        // Strategic context: compute dynamic ship fraction based on lead/deficit
        double effectiveShipsFraction = shipsFraction;
        if (strategicContext) {
            double myTotal = 0, oppTotal = 0;
            for (Planet p : state.getPlanets()) {
                if (p.getOwner() == me) myTotal += p.getNShips();
                else if (p.getOwner() == opp) oppTotal += p.getNShips();
            }
            if (oppTotal > 0) {
                double ratio = myTotal / oppTotal;
                if (ratio > 2.0) effectiveShipsFraction = 0.3;       // way ahead: conserve
                else if (ratio < 0.5) effectiveShipsFraction = 0.7;  // way behind: all-in
            }
        }

        List<Planet> targets = !oppTargets.isEmpty() ? oppTargets : neutralTargets;
        if (targets.isEmpty()) return doNothing();

        // v3: filter targets by minimum growth rate
        if (minTargetGrowth > 0.0) {
            List<Planet> filtered = new ArrayList<>();
            for (Planet t : targets) if (t.getGrowthRate() >= minTargetGrowth) filtered.add(t);
            if (!filtered.isEmpty()) targets = filtered;
        }

        // v3: skip targets that already have a friendly transporter en route
        if (skipAlreadyAttacked) {
            java.util.Set<Integer> alreadyAttacked = new java.util.HashSet<>();
            for (Planet p : state.getPlanets()) {
                if (p.getTransporter() != null && p.getTransporter().getOwner() == me) {
                    alreadyAttacked.add(p.getTransporter().getDestinationIndex());
                }
            }
            List<Planet> filtered = new ArrayList<>();
            for (Planet t : targets) if (!alreadyAttacked.contains(t.getId())) filtered.add(t);
            if (!filtered.isEmpty()) targets = filtered;
        }

        // v3: preserve frontline — exclude sources that are closest to ANY opponent planet
        if (preserveFrontline && !oppTargets.isEmpty() && mySources.size() > 1) {
            List<Planet> nonFrontline = new ArrayList<>();
            for (Planet src : mySources) {
                boolean isFrontline = false;
                for (Planet oppP : oppTargets) {
                    double minDistToOpp = Double.POSITIVE_INFINITY;
                    Planet closestToOpp = null;
                    for (Planet myP : mySources) {
                        double d = myP.getPosition().distance(oppP.getPosition());
                        if (d < minDistToOpp) { minDistToOpp = d; closestToOpp = myP; }
                    }
                    if (closestToOpp == src) { isFrontline = true; break; }
                }
                if (!isFrontline) nonFrontline.add(src);
            }
            if (!nonFrontline.isEmpty()) mySources = nonFrontline;
        }

        double bestScore = Double.NEGATIVE_INFINITY;
        Planet bestSrc = null;
        Planet bestTgt = null;
        for (Planet src : mySources) {
            for (Planet tgt : targets) {
                double s = scoreAction(src, tgt, state, me);
                if (s > bestScore) {
                    bestScore = s;
                    bestSrc = src;
                    bestTgt = tgt;
                }
            }
        }
        if (bestSrc == null) return doNothing();
        if (bestScore < abortThreshold) return doNothing();
        if (bestScore < minActionScore) return doNothing();

        double ships = computeShipsToSend(bestSrc, bestTgt, effectiveShipsFraction);
        if (ships < 1.0) return doNothing();
        return new Action(me, bestSrc.getId(), bestTgt.getId(), ships);
    }

    /** Score a single (src, tgt) pair. Higher = better. */
    private double scoreAction(Planet src, Planet tgt, GameState state, Player me) {
        double distance = src.getPosition().distance(tgt.getPosition());
        double speed = getParams().getTransporterSpeed();
        double traversalTicks = distance / speed;
        double estimatedDefense = tgt.getNShips() + tgt.getGrowthRate() * traversalTicks;

        if (awareOfIncoming) {
            // Account for friendly ships already in flight to this target
            double inflightToTgt = 0.0;
            for (Planet p : state.getPlanets()) {
                if (p.getTransporter() != null
                        && p.getTransporter().getOwner() == me
                        && p.getTransporter().getDestinationIndex() == tgt.getId()) {
                    inflightToTgt += p.getTransporter().getNShips();
                }
            }
            estimatedDefense = Math.max(0.0, estimatedDefense - inflightToTgt);
        }

        double attackShips = src.getNShips() * shipsFraction;
        double netCapture = attackShips - estimatedDefense;
        double normalizedDistance = distance / 800.0;

        double growthBonus;
        if (distanceDiscountedGrowth) {
            // Growth bonus weighted by remaining game time after arrival
            double remainingAfterArrival = Math.max(50.0, estimatedGameTime - state.getGameTick() - traversalTicks);
            growthBonus = tgt.getGrowthRate() * remainingAfterArrival * (weightGrowth / 50.0);
        } else {
            growthBonus = tgt.getGrowthRate() * weightGrowth;
        }

        double score = weightNetCapture * netCapture
                     + growthBonus
                     - weightDistance * normalizedDistance;

        if (sourceVulnerabilityPenalty) {
            double afterAttack = src.getNShips() - attackShips;
            if (afterAttack < 5.0) {
                score -= (5.0 - afterAttack) * 0.5;
            }
        }

        if (preserveProductionSources) {
            score -= src.getGrowthRate() * productionPenaltyWeight;
        }

        if (multiSourcePower) {
            // Count my free sources within 1.2x of my distance to this target
            int nearbyCount = 0;
            for (Planet otherSrc : state.getPlanets()) {
                if (otherSrc == src) continue;
                if (otherSrc.getOwner() != me) continue;
                if (otherSrc.getTransporter() != null) continue;
                if (otherSrc.getNShips() < minSourceShips) continue;
                double otherDist = otherSrc.getPosition().distance(tgt.getPosition());
                if (otherDist <= distance * 1.2) nearbyCount++;
            }
            score += nearbyCount * multiSourceWeight;
        }

        return score;
    }

    /**
     * Ships to send. Three modes by priority:
     *   strongerSmartShipCount: max(half, defense_estimate + postCaptureGarrison) — never UNDER
     *                          half, but MORE if defense is heavy.  Guarantees a meaningful
     *                          post-capture garrison.
     *   smartShipCount:        min(half, defense_estimate + safetyMargin) — v1 attempt, hurts.
     *   default:               half of source.
     */
    private double computeShipsToSend(Planet src, Planet tgt, double fraction) {
        double half = src.getNShips() * fraction;
        if (!strongerSmartShipCount && !smartShipCount) return half;

        double distance = src.getPosition().distance(tgt.getPosition());
        double speed = getParams().getTransporterSpeed();
        double traversalTicks = distance / speed;
        double estimatedDefense = tgt.getNShips() + tgt.getGrowthRate() * traversalTicks;

        if (strongerSmartShipCount) {
            double needed = estimatedDefense + postCaptureGarrison;
            return Math.max(1.0, Math.max(half, Math.min(src.getNShips(), needed)));
        }
        // legacy smartShipCount: "just enough" — known to lose
        double justEnough = estimatedDefense + safetyMargin;
        return Math.max(1.0, Math.min(half, justEnough));
    }

    private static Action doNothing() {
        return new Action(Player.Neutral, -1, -1, 0.0);
    }

    @Override
    public String getAgentType() {
        StringBuilder sb = new StringBuilder("Heuristic");
        if (smartShipCount) sb.append("+smart");
        if (strongerSmartShipCount) sb.append("+smart2");
        if (awareOfIncoming) sb.append("+inflight");
        if (sourceVulnerabilityPenalty) sb.append("+vulnPen");
        if (minSourceShips > 1.0) sb.append("+minSrc").append((int) minSourceShips);
        if (strategicContext) sb.append("+ctx");
        if (minTargetGrowth > 0.0) sb.append("+minG").append(minTargetGrowth);
        if (minActionScore > Double.NEGATIVE_INFINITY) sb.append("+minAct").append((int) minActionScore);
        if (skipAlreadyAttacked) sb.append("+skipDup");
        if (preserveFrontline) sb.append("+frontline");
        if (multiSourcePower) sb.append("+multiSrc");
        if (preserveProductionSources) sb.append("+preserveProd");
        if (distanceDiscountedGrowth) sb.append("+distGrowth");
        if (weightGrowth != 50.0) sb.append("+g").append((int) weightGrowth);
        if (weightDistance != 2.0) sb.append("+d").append(weightDistance);
        return sb.toString();
    }
}
