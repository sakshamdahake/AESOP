"""
Intent classification LangGraph node.
"""

from langchain_aws import ChatBedrock

from app.agents.state import OrchestratorState
from app.agents.intent.agent import IntentClassifier
from app.services.session import get_session_service
from app.logging import logger


# LLM: Claude Haiku (fast, cost-effective for classification)
llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
)

intent_classifier = IntentClassifier(llm)
session_service = get_session_service()


def intent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Entry point node - classifies user intent before routing.
    
    Uses hybrid approach:
    1. Fast-path patterns for trivial cases
    2. Keyword analysis for clear-cut cases
    3. LLM for nuanced classification
    
    SYNC, immutable state pattern.
    """
    logger.info(
        "INTENT_NODE_START",
        extra={
            "session_id": state.session_id or "new",
            "user_input": state.query[:50] if state.query else "",
        },
    )
    
    # Fetch session context from Redis (if session_id provided)
    session_context = None
    if state.session_id:
        session_context = session_service.get_session(state.session_id)
        if session_context:
            logger.debug(
                "INTENT_SESSION_FOUND",
                extra={
                    "original_query": session_context.original_query[:50],
                    "turn_count": session_context.turn_count,
                },
            )
    
    # Classify intent
    intent, confidence, reasoning = intent_classifier.classify(
        message=state.query,
        session_context=session_context,
    )
    
    logger.info(
        "INTENT_NODE_END",
        extra={
            "intent": intent,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
        },
    )
    
    # Create new state (immutable pattern)
    new_state = state.model_copy(deep=True)
    new_state.intent = intent
    new_state.intent_confidence = confidence
    new_state.intent_reasoning = reasoning
    new_state.session_context = session_context
    
    return new_state