package competition_entry

import json_rmi.GameAgentServer

fun main() {
    val server = GameAgentServer(port = 8080, agentClass = NaiveMCTSAgent::class)
    server.start(wait = true)
}
