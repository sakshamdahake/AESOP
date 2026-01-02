"""
Chat LangGraph node.
"""

from langchain_aws import ChatBedrock

from app.agents.state import OrchestratorState
from app.agents.chat.agent import ChatAgent
from app.logging import logger


# LLM: Claude Haiku (cost-efficient for chat)
llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
)

chat_agent = ChatAgent(llm)


def chat_node(state: OrchestratorState) -> OrchestratorState:
    """
    Chat node - Handles general conversation.
    
    SYNC, immutable state pattern.
    """
    logger.info(
        "CHAT_NODE_START",
        extra={
            "session_id": state.session_id,
            "query": state.query[:50] if state.query else "",
        },
    )
    
    # Generate response
    response = chat_agent.respond(
        message=state.query,
        session_context=state.session_context,
    )
    
    logger.info(
        "CHAT_NODE_END",
        extra={
            "response_length": len(response),
        },
    )
    
    # Create new state (immutable pattern)
    new_state = state.model_copy(deep=True)
    new_state.chat_response = response
    new_state.route_taken = "chat"
    
    return new_state