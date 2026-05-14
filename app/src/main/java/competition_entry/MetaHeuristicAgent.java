package competition_entry;

import games.planetwars.agents.Action;
import games.planetwars.agents.PlanetWarsPlayer;
import games.planetwars.core.GameParams;
import games.planetwars.core.GameState;
import games.planetwars.core.Planet;
import games.planetwars.core.Player;

import java.util.List;

/**
 * Meta-heuristic: pick a HeuristicAgent config based on map fingerprint at game start.
 *
 * Why this beats per-tick weight tuning (v12 fingerprint approach):
 *   v12-fingerprint adjusted ONE config's weights per map. Small effect because the
 *   adjustment magnitudes were small relative to the heuristic's existing strength.
 *   MetaHeuristic switches between ENTIRE configs — different feature sets, different
 *   defaults — so the adjustment is structural rather than incremental.
 *
 * Algorithm:
 *   - At first getAction call (re-set on each prepareToPlayAs), compute the map
 *     fingerprint (mean inter-planet distance, growth std, mean growth, planet count).
 *   - Pick a config from the portfolio based on rules derived from per-map analysis.
 *   - Configure an internal HeuristicAgent with that config and use it for the rest
 *     of the game.
 *
 * Routing rules (derived from v18 per-map data, 30 seeds vs Greedy):
 *   1. DEFAULT: minG=0.05 + v11 features. Wins 25/30 of evaluated maps. Best baseline.
 *   2. Sparse + high-growth map (dist > 330 AND meanGrowth > 0.06):
 *      switch to minG=0.07. Recovers seed 11 (only-minG.07-wins map). Doesn't hurt
 *      other dist>330 maps (seeds 1, 24) since minG.07 also wins on those.
 *
 * Expected uplift: ~+3.3 points over minG=0.05 alone (recovers 1 of 5 lost maps).
 * More rules can be added as more per-map data becomes available.
 */
public class MetaHeuristicAgent extends PlanetWarsPlayer {

    private HeuristicAgent inner;
    private boolean configured = false;
    private String chosenConfigName = "?";

    @Override
    public Action getAction(GameState state) {
        if (!configured) {
            applyConfigForMap(state);
            configured = true;
        }
        return inner.getAction(state);
    }

    @Override
    public String prepareToPlayAs(Player p, GameParams params, String opponent) {
        super.prepareToPlayAs(p, params, opponent);
        configured = false;       // re-route for the new game
        return getAgentType();
    }

    private void applyConfigForMap(GameState state) {
        Fp fp = computeFingerprint(state);

        // Apply routing rules in order (first-match)
        if (fp.meanInterPlanetDistance > 330.0 && fp.meanGrowth > 0.06) {
            chosenConfigName = "sparse-high-growth: minG.07";
            inner = makeStrictFilterConfig();
        } else {
            chosenConfigName = "default: minG.05+v11";
            inner = makeDefaultConfig();
        }
        inner.prepareToPlayAs(getPlayer(), getParams(), "meta-internal");
    }

    /** Default config — minG=0.05 + v11 features. Wins ~83% of maps vs Greedy. */
    private static HeuristicAgent makeDefaultConfig() {
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

    /** Strict-filter config — minG=0.07 + v11. For sparse maps with dense high-growth targets. */
    private static HeuristicAgent makeStrictFilterConfig() {
        HeuristicAgent h = new HeuristicAgent();
        h.minTargetGrowth = 0.07;
        h.weightGrowth = 100.0;
        h.preserveProductionSources = true;
        h.counterattackRisk = true;
        h.attackMomentum = true;
        h.gameStageAwareness = true;
        h.threatResponse = true;
        return h;
    }

    @Override
    public String getAgentType() {
        return "MetaHeuristic-v1";
    }

    // ---- Fingerprint helpers ----
    private static Fp computeFingerprint(GameState state) {
        List<Planet> planets = state.getPlanets();
        int n = planets.size();
        double sumG = 0;
        for (Planet p : planets) sumG += p.getGrowthRate();
        double meanG = sumG / n;
        double sumSq = 0;
        for (Planet p : planets) {
            double d = p.getGrowthRate() - meanG;
            sumSq += d * d;
        }
        double stdG = Math.sqrt(sumSq / n);
        double sumDist = 0;
        int pairs = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                sumDist += planets.get(i).getPosition().distance(planets.get(j).getPosition());
                pairs++;
            }
        }
        double meanDist = pairs > 0 ? sumDist / pairs : 0.0;
        return new Fp(meanG, stdG, meanDist, n);
    }

    private static final class Fp {
        final double meanGrowth, growthStd, meanInterPlanetDistance;
        final int numPlanets;
        Fp(double mg, double gs, double mid, int n) {
            this.meanGrowth = mg; this.growthStd = gs; this.meanInterPlanetDistance = mid; this.numPlanets = n;
        }
    }
}
