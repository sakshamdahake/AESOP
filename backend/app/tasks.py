from app.agents.graph import aesop_graph
from app.agents.state import AgentState


def run_review(query: str) -> AgentState:
    initial_state = AgentState(query=query)
    final_state = aesop_graph.invoke(initial_state)
    return final_state
