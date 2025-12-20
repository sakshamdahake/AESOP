from langchain_openai import ChatOpenAI

from app.agents.state import AgentState
from app.agents.critic.agent import CriticAgent

# ============================
# Mega-LLM configuration
# ============================

BASE_URL = "https://ai.megallm.io/v1"
AGENT_MODEL = "openai-gpt-oss-120b"

llm = ChatOpenAI(
    model=AGENT_MODEL,
    base_url=BASE_URL,
    temperature=0.2,
)

critic = CriticAgent(llm)


# ============================
# LangGraph Critic Node (SYNC)
# ============================

def critic_node(state: AgentState) -> AgentState:
    """
    Synchronous LangGraph Critic node.

    Responsibilities:
    - Extract abstracts
    - Run CRAG grading
    - Update state for routing + explainability
    """

    # Safety: no papers retrieved
    if not state.retrieved_papers:
        state.critic_decision = "retrieve_more"
        state.critic_explanation = (
            "No papers retrieved; additional search required."
        )
        return state

    abstracts = [paper.abstract for paper in state.retrieved_papers]

    # CRAG grading (SYNC)
    result = critic.grade_batch(
        research_question=state.query,
        abstracts=abstracts,
        iteration=state.iteration_count,
    )

    # Core outputs
    state.grades = result["grades"]
    state.critic_decision = result["decision"]

    # Explainability (optional but valuable)
    grades = result["grades"]

    if grades:
        avg_quality = sum(
            (g.relevance_score + g.methodology_score) / 2
            for g in grades
        ) / len(grades)

        discard_ratio = (
            sum(1 for g in grades if g.recommendation.name == "DISCARD")
            / len(grades)
        )

        state.avg_quality = round(avg_quality, 3)
        state.discard_ratio = round(discard_ratio, 3)

        state.critic_explanation = (
            f"CRAG decision='{state.critic_decision}' | "
            f"avg_quality={state.avg_quality}, "
            f"discard_ratio={state.discard_ratio}, "
            f"iteration={state.iteration_count}"
        )

    else:
        state.avg_quality = None
        state.discard_ratio = None
        state.critic_explanation = (
            "No valid grades produced; triggering retrieval."
        )

    return state
