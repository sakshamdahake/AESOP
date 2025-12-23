from langchain_aws import ChatBedrock
from langchain_core.tracers.context import tracing_v2_enabled

from app.agents.state import AgentState
from app.agents.critic.agent import CriticAgent
from app.logging import logger


# ============================
# LLM configuration (Bedrock)
# ============================

llm = ChatBedrock(
    model="amazon.nova-pro-v1:0",
    region_name="us-east-1",
)

critic = CriticAgent(llm)


# ============================
# LangGraph Critic Node (SYNC)
# ============================

def critic_node(state: AgentState) -> AgentState:
    """
    Synchronous LangGraph Critic node.

    CRITICAL LANGGRAPH RULE:
    - NEVER mutate state in-place
    - ALWAYS return a NEW state object
    """

    with tracing_v2_enabled(project_name="aesop-dev"):
        logger.info(
            "CRITIC_NODE_START",
            extra={
                "iteration": state.iteration_count,
                "num_papers": len(state.retrieved_papers),
            },
        )

        try:
            # --------------------------------------------------
            # Case 1: No papers retrieved → force retrieval
            # --------------------------------------------------
            if not state.retrieved_papers:
                new_state = state.model_copy(deep=True)

                new_state.critic_decision = "retrieve_more"
                new_state.critic_explanation = (
                    "No papers retrieved; additional search required."
                )

                logger.info(
                    "CRITIC_NODE_NO_PAPERS",
                    extra={"iteration": state.iteration_count},
                )

                return new_state

            # --------------------------------------------------
            # CRAG grading (PASS PAPERS, NOT ABSTRACTS)
            # --------------------------------------------------
            result = critic.grade_batch(
                research_question=state.query,
                papers=state.retrieved_papers,
                iteration=state.iteration_count,
            )

            grades = result["grades"]
            decision = result["decision"]

            # --------------------------------------------------
            # Explainability metrics
            # --------------------------------------------------
            if grades:
                avg_quality = sum(
                    (g.relevance_score + g.methodology_score) / 2
                    for g in grades
                ) / len(grades)

                discard_ratio = (
                    sum(
                        1
                        for g in grades
                        if g.recommendation.name == "DISCARD"
                    )
                    / len(grades)
                )
            else:
                avg_quality = None
                discard_ratio = None

            # --------------------------------------------------
            # IMPORTANT: create NEW state object
            # --------------------------------------------------
            new_state = state.model_copy(deep=True)

            new_state.grades = grades
            new_state.critic_decision = decision
            new_state.avg_quality = (
                round(avg_quality, 3) if avg_quality is not None else None
            )
            new_state.discard_ratio = (
                round(discard_ratio, 3)
                if discard_ratio is not None
                else None
            )

            new_state.critic_explanation = (
                f"CRAG decision='{decision}' | "
                f"avg_quality={new_state.avg_quality}, "
                f"discard_ratio={new_state.discard_ratio}, "
                f"iteration={state.iteration_count}"
            )

            logger.info(
                "CRITIC_NODE_END",
                extra={
                    "decision": decision,
                    "avg_quality": new_state.avg_quality,
                    "discard_ratio": new_state.discard_ratio,
                    "iteration": state.iteration_count,
                },
            )

            return new_state

        except Exception as e:
            logger.exception(
                "CRITIC_NODE_ERROR",
                extra={
                    "iteration": state.iteration_count,
                    "error": str(e),
                },
            )
            raise
