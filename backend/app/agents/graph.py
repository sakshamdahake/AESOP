from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.scout.agent import scout_node
from app.agents.critic.node import critic_node


def synthesizer_node(state: AgentState) -> AgentState:
    # Phase 2
    return state


def routing_logic(state: AgentState) -> str:
    if state.critic_decision == "retrieve_more" and state.iteration_count < state.max_iterations:
        return "scout"
    return "synthesize"


graph = StateGraph(AgentState)

graph.add_node("scout", scout_node)
graph.add_node("critic", critic_node)
graph.add_node("synthesize", synthesizer_node)

graph.set_entry_point("scout")

graph.add_edge("scout", "critic")

graph.add_conditional_edges(
    "critic",
    routing_logic,
    {
        "scout": "scout",
        "synthesize": "synthesize",
    },
)

graph.add_edge("synthesize", END)

aesop_graph = graph.compile()
