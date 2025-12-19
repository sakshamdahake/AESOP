from langchain_openai import ChatOpenAI

from app.agents.state import AgentState
from app.agents.critic.agent import CriticAgent

# Mega-LLM config (MATCH Scout)
BASE_URL = "https://ai.megallm.io/v1"
AGENT_MODEL = "openai-gpt-oss-120b"

llm = ChatOpenAI(
    model=AGENT_MODEL,
    base_url=BASE_URL,
    temperature=0.2,
)

critic = CriticAgent(llm)


async def critic_node(state: AgentState) -> AgentState:
    if not state.retrieved_papers:
        state.critic_decision = "retrieve_more"
        return state

    abstracts = [paper.abstract for paper in state.retrieved_papers]

    result = await critic.grade_batch(
        research_question=state.query,
        abstracts=abstracts,
    )

    state.grades = result["grades"]
    state.critic_decision = result["decision"]

    return state
