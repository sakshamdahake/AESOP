from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from app.agents.state import AgentState
from app.agents.synthesizer.prompts import SYNTHESIS_PROMPT
from app.agents.synthesizer.utils import (
    build_graded_papers,
    format_papers_for_prompt,
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)


def synthesizer_node(state: AgentState) -> AgentState:
    graded_papers = build_graded_papers(
        state.retrieved_papers,
        state.grades,
    )

    if not graded_papers:
        state.synthesis_output = (
            "Insufficient high-quality evidence available "
            "to generate a structured review."
        )
        return state

    papers_block = format_papers_for_prompt(graded_papers)

    prompt = PromptTemplate.from_template(SYNTHESIS_PROMPT)

    response = llm.invoke(
        prompt.format(
            query=state.query,
            papers=papers_block,
        )
    )

    state.synthesis_output = response.content
    return state
