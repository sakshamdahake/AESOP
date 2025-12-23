from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.scout.agent import scout_node
from app.agents.critic.node import critic_node
from app.agents.synthesizer.agent import synthesizer_node


# ============================
# CRAG Routing Logic
# ============================

def routing_logic(state: AgentState) -> str:
    """
    Decide next step based on Critic's global decision.

    Rules:
    - If Critic says retrieve_more AND we have iterations left → scout
    - Otherwise → synthesize
    """
    if (
        state.critic_decision == "retrieve_more"
        and state.iteration_count < state.max_iterations
    ):
        return "scout"

    return "synthesize"


# ============================
# LangGraph Construction
# ============================

graph = StateGraph(AgentState)

graph.add_node("scout", scout_node)
graph.add_node("critic", critic_node)
graph.add_node("synthesize", synthesizer_node)

graph.set_entry_point("scout")

# Scout → Critic
graph.add_edge("scout", "critic")

# Critic → (Scout | Synthesizer)
graph.add_conditional_edges(
    "critic",
    routing_logic,
    {
        "scout": "scout",
        "synthesize": "synthesize",
    },
)

# Synthesizer → END
graph.add_edge("synthesize", END)

aesop_graph = graph.compile()
