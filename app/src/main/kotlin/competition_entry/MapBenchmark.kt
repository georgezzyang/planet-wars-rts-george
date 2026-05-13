package competition_entry

import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.GameParams
import games.planetwars.core.GameStateFactory
import games.planetwars.core.Player
import games.planetwars.runners.GameRunner

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
    val mapSeeds = (1L..30L).toList()           // 30 distinct map geometries
    val gameParams = GameParams(
        numPlanets = 12,
        maxTicks = 1200,
        newMapEachRun = false,                  // re-use same gameState across N games per pair
    )

    // Candidates: vanilla + key configs we want to compare
    val candidates: List<Pair<String, () -> PlanetWarsAgent>> = listOf(
        "vanilla" to { HeuristicAgent() },
        "g100" to { HeuristicAgent().apply { weightGrowth = 100.0 } },
        "preserveProd" to { HeuristicAgent().apply { preserveProductionSources = true } },
        "v6-best" to { HeuristicAgent().apply {
            minSourceShips = 10.0; minTargetGrowth = 0.07; weightGrowth = 100.0
        }},
        "v10-all5" to { HeuristicAgent().apply {
            minSourceShips = 10.0; minTargetGrowth = 0.07; weightGrowth = 100.0
            preserveProductionSources = true
            counterattackRisk = true
            attackMomentum = true
            gameStageAwareness = true
            threatResponse = true
        }},
        "v11-minus-minSrc" to { HeuristicAgent().apply {
            minTargetGrowth = 0.07; weightGrowth = 100.0
            preserveProductionSources = true
            counterattackRisk = true
            attackMomentum = true
            gameStageAwareness = true
            threatResponse = true
        }},
        "v12-fingerprint" to { HeuristicAgent().apply {
            minTargetGrowth = 0.07; weightGrowth = 100.0
            preserveProductionSources = true
            counterattackRisk = true
            attackMomentum = true
            gameStageAwareness = true
            threatResponse = true
            useMapFingerprint = true
        }},
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

        // Header
        print("seed\t")
        for ((name, _) in candidates) print("$name\t")
        println()

        // Per-candidate aggregate: mean and stddev across seeds
        val perCandidate = mutableMapOf<String, MutableList<Double>>()
        for ((name, _) in candidates) perCandidate[name] = mutableListOf()

        for (seed in mapSeeds) {
            print("$seed\t")
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
            val std = Math.sqrt(variance)
            println("${name.padEnd(20)}  mean=${"%5.1f".format(mean)}%  std=${"%5.1f".format(std)}")
        }
    }
}
