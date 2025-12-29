"""
Utility LangGraph node.
"""

from langchain_aws import ChatBedrock

from app.agents.state import OrchestratorState
from app.agents.utility.agent import UtilityAgent
from app.logging import logger


# LLM: Claude Haiku
llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
)

utility_agent = UtilityAgent(llm)


def utility_node(state: OrchestratorState) -> OrchestratorState:
    """
    Utility node - Transforms existing output.
    
    SYNC, immutable state pattern.
    """
    logger.info(
        "UTILITY_NODE_START",
        extra={
            "session_id": state.session_id,
            "query": state.query[:50] if state.query else "",
        },
    )
    
    # Transform content
    result = utility_agent.transform(
        request=state.query,
        session_context=state.session_context,
    )
    
    logger.info(
        "UTILITY_NODE_END",
        extra={
            "result_length": len(result),
        },
    )
    
    # Create new state (immutable pattern)
    new_state = state.model_copy(deep=True)
    new_state.utility_response = result
    new_state.route_taken = "utility"
    
    return new_state