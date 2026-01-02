"""
Task runners for AESOP system.
"""

import uuid

from app.agents.graph import aesop_graph
from app.agents.orchestrator_graph import orchestrator_graph
from app.agents.state import AgentState, OrchestratorState


def run_review(query: str) -> AgentState:
    """
    Original single-turn review (backward compatible).
    """
    initial_state = AgentState(query=query)
    final_state = aesop_graph.invoke(initial_state)
    return final_state


def run_orchestrated_review(
    query: str,
    session_id: str = None,
) -> dict:
    """
    Session-aware multi-turn review with intent classification.
    
    Handles:
    - Chat (greetings, thanks, system questions)
    - Utility (reformat existing output)
    - Research (full literature search)
    - Follow-up research (context-aware queries)
    
    Args:
        query: User message
        session_id: Optional session ID for follow-ups
        
    Returns:
        dict with response, session_id, route_taken, intent, etc.
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    initial_state = OrchestratorState(
        query=query,
        session_id=session_id,
    )
    
    # LangGraph returns a dict
    final_state = orchestrator_graph.invoke(initial_state)
    
    # Determine the response based on route taken
    route_taken = final_state.get("route_taken") or "unknown"
    
    if route_taken == "chat":
        response = final_state.get("chat_response") or ""
    elif route_taken == "utility":
        response = final_state.get("utility_response") or ""
    else:
        response = final_state.get("synthesis_output") or ""
    
    # Determine paper count
    papers_count = 0
    if final_state.get("merged_papers"):
        papers_count = len(final_state["merged_papers"])
    elif final_state.get("retrieved_papers"):
        papers_count = len(final_state["retrieved_papers"])
    elif final_state.get("session_context"):
        session_ctx = final_state["session_context"]
        if hasattr(session_ctx, "retrieved_papers"):
            papers_count = len(session_ctx.retrieved_papers)
        elif isinstance(session_ctx, dict):
            papers_count = len(session_ctx.get("retrieved_papers", []))
    
    return {
        "response": response,
        "session_id": session_id,
        "route_taken": route_taken,
        "intent": final_state.get("intent"),
        "intent_confidence": final_state.get("intent_confidence"),
        "papers_count": papers_count,
        "critic_decision": final_state.get("critic_decision"),
        "avg_quality": final_state.get("avg_quality"),
    }