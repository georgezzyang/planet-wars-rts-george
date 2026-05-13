package competition_entry

import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.evo.SimpleEvoAgent
import games.planetwars.agents.random.BetterRandomAgent
import games.planetwars.agents.random.CarefulRandomAgent
import games.planetwars.core.GameParams
import games.planetwars.runners.LeagueResult
import games.planetwars.runners.LeagueWriter
import games.planetwars.runners.RoundRobinLeague

/**
 * Local self-evaluation: pit our NaiveMCTSAgent against the standard baselines
 * so we can read its strength before paying the round-trip to the AAMAS server.
 *
 * Tweak GAMES_PER_PAIR / planet count below to trade time for statistical power.
 *
 * Run: ./gradlew :app:benchmarkNaiveMCTS
 */
fun main() {
    val gameParams = GameParams(numPlanets = 12, maxTicks = 1200)
    // FAST HEURISTIC ITERATION MODE
    // All agents below are deterministic / non-search, so a game takes ~1s instead of ~30s.
    // This lets us iterate scoring tweaks at 25-30x the speed of a PUCT benchmark.
    // Once the heuristic is dialed in, the SAME scoring formula goes back into
    // UCTAgent.computePrior() for compound gains.
    val GAMES_PER_PAIR = 30

    // v7: new mechanisms + EvoAgent (RHEA) joins for top-end check.
    // "best" = minSrc10 + minG0.07 + g100 (winner from v6).
    val best = { HeuristicAgent().apply {
        minSourceShips = 10.0
        minTargetGrowth = 0.07
        weightGrowth = 100.0
    }}
    val agents: MutableList<PlanetWarsAgent> = mutableListOf(
        best(),                                                                              // current best (control)
        best().apply { multiSourcePower = true },                                            // +multi-source coord
        best().apply { preserveProductionSources = true },                                   // +production penalty
        best().apply { distanceDiscountedGrowth = true },                                    // +game-time-aware growth
        best().apply {                                                                       // all 3 new combined
            multiSourcePower = true
            preserveProductionSources = true
            distanceDiscountedGrowth = true
        },
        HeuristicAgent(),                                                                    // vanilla
        GreedyHeuristicAgent(),
        SimpleEvoAgent(                                                                      // RHEA — top baseline
            useShiftBuffer = true,
            nEvals = 50,
            sequenceLength = 400,
            probMutation = 0.8,
        ),
    )

    println("=== Benchmark: NaiveMCTS vs baselines ===")
    println("planets=${gameParams.numPlanets}, maxTicks=${gameParams.maxTicks}, gamesPerPair=$GAMES_PER_PAIR")
    println("agents: ${agents.map { it.getAgentType() }}")

    val league = RoundRobinLeague(
        agents,
        gameParams = gameParams,
        gamesPerPair = GAMES_PER_PAIR,
        runRemoteAgents = false,
    )
    val results = league.runRoundRobin()

    val writer = LeagueWriter()
    val leagueResult = LeagueResult(results.values.toList())
    val md = writer.generateMarkdownTable(leagueResult)
    println()
    println(md)

    println("\n--- sorted by points ---")
    val sorted = results.toList().sortedByDescending { it.second.points }
    for ((_, e) in sorted) {
        println("${e.agentName} : points=${e.points} games=${e.nGames}")
    }
}
