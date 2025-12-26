"""
Router LangGraph node.
"""

from langchain_aws import ChatBedrock

from app.agents.state import OrchestratorState
from app.agents.router.agent import RouterAgent
from app.services.session import get_session_service
from app.logging import logger


# LLM: Claude Haiku (same as Scout)
llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
)

router_agent = RouterAgent(llm)
session_service = get_session_service()


def router_node(state: OrchestratorState) -> OrchestratorState:
    """
    Entry point node - routes query to appropriate execution path.
    
    SYNC, immutable state pattern.
    """
    logger.info(
        "ROUTER_NODE_START",
        extra={
            "session_id": state.session_id,
            "query": state.query[:50],
        },
    )
    
    # Fetch session context from Redis
    session_context = session_service.get_session(state.session_id)
    
    # Run router classification
    decision = router_agent.route(
        current_query=state.query,
        session_context=session_context,
    )
    
    logger.info(
        "ROUTER_NODE_END",
        extra={
            "route": decision.route,
            "similarity": decision.similarity_score,
            "is_new_session": decision.is_new_session,
        },
    )
    
    # Create new state (immutable pattern)
    new_state = state.model_copy(deep=True)
    new_state.router_decision = decision
    new_state.session_context = session_context
    new_state.route_taken = decision.route
    
    return new_state