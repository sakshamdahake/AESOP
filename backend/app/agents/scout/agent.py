from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.agents.state import AgentState
from app.agents.scout.prompts import QUERY_EXPANSION_PROMPT
from app.agents.scout.tools import pubmed_search, pubmed_fetch

#Mega-LLM configs
BASE_URL = "https://ai.megallm.io/v1"
AGENT_MODEL = "openai-gpt-oss-120b"

llm = ChatOpenAI(
    model=AGENT_MODEL,
    base_url= BASE_URL,
    temperature=0.2,
)

def scout_node(state: AgentState) -> AgentState:
    prompt = PromptTemplate.from_template(QUERY_EXPANSION_PROMPT)

    response = llm.invoke(
        prompt.format(query=state.query)
    )

    expanded_queries = [
        line.strip("- ").strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    retrieved_papers = []
    for query in expanded_queries:
        pmids = pubmed_search(query)
        retrieved_papers.extend(pubmed_fetch(pmids))

    state.expanded_queries = expanded_queries
    state.retrieved_papers = retrieved_papers
    state.iteration_count += 1

    return state
