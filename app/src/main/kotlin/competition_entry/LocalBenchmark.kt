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
    val GAMES_PER_PAIR = 10

    val agents: MutableList<PlanetWarsAgent> = mutableListOf(
        UCTAgent(),                                    // vanilla Decoupled UCT (UCB1)
        UCTAgent().apply { useHeuristicPrior = true }, // PUCT variant — same code, prior on
        GreedyHeuristicAgent(),
        SimpleEvoAgent(
            useShiftBuffer = true,
            nEvals = 50,
            sequenceLength = 400,
            probMutation = 0.8,
        ),
        CarefulRandomAgent(),
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
