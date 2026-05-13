package competition_entry

import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.GameParams
import games.planetwars.core.GameState
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Planet
import games.planetwars.core.Player
import games.planetwars.runners.GameRunner
import kotlin.math.sqrt

/**
 * Per-map analysis: run each candidate on K different fixed-seed maps and report per-map
 * win rate, plus aggregate (mean + std across maps).
 *
 * Why this exists:
 *   The round-robin LocalBenchmark generates a new random map per game. The win rate it
 *   reports is averaged across many unrelated map geometries. Features that help on
 *   sparse maps may hurt on dense maps; the aggregate hides this.
 *
 *   MapBenchmark fixes a map seed (via GameStateFactory(seed=K)) and runs games on THAT
 *   geometry. Per-seed differences expose feature-map interactions. Cross-seed mean is
 *   our most map-robust estimate.
 *
 * Notes:
 *   - HeuristicAgent variants are DETERMINISTIC. On a fixed map vs a deterministic
 *     opponent (Greedy, another Heuristic), all N games per seed are identical. So we
 *     get only K independent samples (one per map), not K*N.
 *   - To get variance on the same map, pair against a STOCHASTIC opponent like
 *     CarefulRandomAgent. Each game then differs because of the opponent's RNG.
 *   - Each candidate plays both sides (Player1 / Player2) per seed to wash out
 *     player-position effects.
 *
 * Run: ./gradlew :app:benchmarkMaps
 */
fun main() {
    // Use 10 map seeds when search agents are in the pool (UCT/FlatMC are ~30s/game);
    // use 30 for fast-only benchmarks.
    val mapSeeds = (1L..10L).toList()
    val gameParams = GameParams(
        numPlanets = 12,
        maxTicks = 1200,
        newMapEachRun = false,                  // re-use same gameState across N games per pair
    )

    // v13 candidates: keep top configs from v11/v12 + weight variants for tuning
    val v11Base = { HeuristicAgent().apply {
        minTargetGrowth = 0.07; weightGrowth = 100.0
        preserveProductionSources = true
        counterattackRisk = true
        attackMomentum = true
        gameStageAwareness = true
        threatResponse = true
    }}
    // v16: small comparison set — heuristic ceiling vs PUCT variants (UCT slow)
    val candidates: List<Pair<String, () -> PlanetWarsAgent>> = listOf(
        "Heuristic-best" to { v11Base().apply { minTargetGrowth = 0.05 } },
        "UCT-vanilla" to { UCTAgent() },
        "PUCT-v1-prior" to { UCTAgent().apply { useHeuristicPrior = true } },
        "PUCT-v14-prior" to { UCTAgent().apply { useHeuristicPrior = true; priorV14 = true } },
    )

    // Fixed opponent. Use a stochastic one (CarefulRandom) so different N games on the
    // same seed produce different outcomes — gives variance estimate. Plus Greedy
    // (deterministic) reference for sanity.
    data class Opponent(val name: String, val factory: () -> PlanetWarsAgent, val stochastic: Boolean)
    val opponents = listOf(
        Opponent("Greedy", { GreedyHeuristicAgent() }, false),
        Opponent("CarefulRandom", { CarefulRandomAgent() }, true),
    )

    for (opp in opponents) {
        val gamesPerSide = if (opp.stochastic) 3 else 1   // stochastic needs samples; deterministic doesn't
        println("\n========================================================================")
        println("Opponent: ${opp.name}   gamesPerSide=$gamesPerSide   mapSeeds=${mapSeeds.size}")
        println("========================================================================")

        // Header with fingerprint columns + candidate columns
        print("seed\tdist\tgStd\tgMean\tn\t")
        for ((name, _) in candidates) print("$name\t")
        println()

        // Per-candidate aggregate: mean and stddev across seeds
        val perCandidate = mutableMapOf<String, MutableList<Double>>()
        for ((name, _) in candidates) perCandidate[name] = mutableListOf()

        for (seed in mapSeeds) {
            // Compute & print fingerprint of this seed's map
            val fpState = GameStateFactory(gameParams, seed = seed).createGame()
            val fp = computeFingerprintExternal(fpState)
            print("$seed\t${"%.0f".format(fp.meanDist)}\t${"%.3f".format(fp.growthStd)}\t${"%.3f".format(fp.meanGrowth)}\t${fp.n}\t")
            for ((name, makeCandidate) in candidates) {
                // Play candidate as P1
                val candidateA = makeCandidate()
                val oppA = opp.factory()
                val runnerA = GameRunner(candidateA, oppA, gameParams)
                runnerA.gameState = GameStateFactory(gameParams, seed = seed).createGame()
                val scoresA = runnerA.runGames(gamesPerSide)
                val winsA = scoresA[Player.Player1] ?: 0

                // Play candidate as P2 (symmetric)
                val candidateB = makeCandidate()
                val oppB = opp.factory()
                val runnerB = GameRunner(oppB, candidateB, gameParams)
                runnerB.gameState = GameStateFactory(gameParams, seed = seed).createGame()
                val scoresB = runnerB.runGames(gamesPerSide)
                val winsB = scoresB[Player.Player2] ?: 0

                val totalGames = 2 * gamesPerSide
                val winRate = (winsA + winsB).toDouble() / totalGames * 100.0
                perCandidate[name]!!.add(winRate)
                print("${"%.0f".format(winRate)}\t")
            }
            println()
        }

        println("\nAggregate vs ${opp.name}:")
        for ((name, rates) in perCandidate) {
            val mean = rates.average()
            val variance = rates.map { (it - mean) * (it - mean) }.average()
            val std = sqrt(variance)
            println("${name.padEnd(20)}  mean=${"%5.1f".format(mean)}%  std=${"%5.1f".format(std)}")
        }
    }
}

private data class FpExt(val meanGrowth: Double, val growthStd: Double, val meanDist: Double, val n: Int)

private fun computeFingerprintExternal(state: GameState): FpExt {
    val planets = state.planets
    val n = planets.size
    val meanG = planets.map { it.growthRate }.average()
    val gStd = sqrt(planets.map { (it.growthRate - meanG).let { d -> d * d } }.average())
    var sumDist = 0.0
    var pairs = 0
    for (i in 0 until n) for (j in i + 1 until n) {
        sumDist += planets[i].position.distance(planets[j].position); pairs++
    }
    val meanD = if (pairs > 0) sumDist / pairs else 0.0
    return FpExt(meanG, gStd, meanD, n)
}
