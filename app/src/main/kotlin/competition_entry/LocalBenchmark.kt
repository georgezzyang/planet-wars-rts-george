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
    val GAMES_PER_PAIR = 100

    // v10: clean final comparison. Top candidates vs stable references, N=100 per pair for
    // tight CIs (std err ~1.6%). 8 agents, no weight collisions in getAgentType.
    val base = { HeuristicAgent().apply {
        minSourceShips = 10.0
        minTargetGrowth = 0.07
        weightGrowth = 100.0
    }}
    val agents: MutableList<PlanetWarsAgent> = mutableListOf(
        // candidates (top of each evolution)
        base(),                                                                              // v6 winner
        base().apply { preserveProductionSources = true },                                   // v7 winner
        base().apply { preserveProductionSources = true; counterattackRisk = true },        // v8/v9 candidate: preserveProd + counter
        base().apply {                                                                       // v8 all-4 winner
            preserveProductionSources = true
            counterattackRisk = true
            attackMomentum = true
            gameStageAwareness = true
            threatResponse = true
        },
        // stable references
        HeuristicAgent(),                                                                    // vanilla
        GreedyHeuristicAgent(),
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
