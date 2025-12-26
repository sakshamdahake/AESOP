"""
Context Q&A LangGraph node.
"""

from langchain_aws import ChatBedrock

from app.agents.state import OrchestratorState
from app.agents.context_qa.agent import ContextQAAgent
from app.logging import logger


# LLM: Claude Haiku (cost-efficient for simple Q&A)
llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
)

context_qa_agent = ContextQAAgent(llm)


def context_qa_node(state: OrchestratorState) -> OrchestratorState:
    """
    Route C node - Answer from cached context only.
    
    SYNC, immutable state pattern.
    """
    logger.info(
        "CONTEXT_QA_NODE_START",
        extra={
            "session_id": state.session_id,
            "query": state.query[:50],
            "papers_in_context": len(state.session_context.retrieved_papers) if state.session_context else 0,
        },
    )
    
    answer = context_qa_agent.answer(
        current_query=state.query,
        session_context=state.session_context,
    )
    
    logger.info(
        "CONTEXT_QA_NODE_END",
        extra={
            "answer_length": len(answer),
        },
    )
    
    # Create new state (immutable pattern)
    new_state = state.model_copy(deep=True)
    new_state.synthesis_output = answer
    
    return new_state