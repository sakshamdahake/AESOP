import json

from langchain_core.prompts import PromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_aws import ChatBedrock

from app.agents.state import AgentState
from app.agents.scout.prompts import QUERY_EXPANSION_PROMPT
from app.agents.scout.tools import pubmed_search, pubmed_fetch
from app.logging import logger


# ============================
# LLM configuration (Bedrock)
# ============================

llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
)


# ============================
# LangGraph Scout Node (SYNC)
# ============================

def scout_node(state: AgentState) -> AgentState:
    """
    Synchronous LangGraph Scout node.

    HARD CONTRACT:
    - LLM MUST return a JSON array of query strings
    - Any violation is a hard failure
    - State is NEVER mutated in-place
    """

    with tracing_v2_enabled(project_name="aesop-dev"):
        logger.info(
            "SCOUT_NODE_START",
            extra={
                "iteration": state.iteration_count,
                "query": state.query,
            },
        )

        try:
            # --------------------------------------------------
            # Query expansion
            # --------------------------------------------------
            prompt = PromptTemplate.from_template(
                QUERY_EXPANSION_PROMPT
            )

            logger.info(
                "SCOUT_QUERY_EXPANSION_START",
                extra={"iteration": state.iteration_count},
            )

            response = llm.invoke(
                prompt.format(query=state.query)
            )

            raw_output = response.content.strip()

            # --------------------------------------------------
            # HARD JSON CONTRACT ENFORCEMENT
            # --------------------------------------------------
            if not (raw_output.startswith("[") and raw_output.endswith("]")):
                raise RuntimeError(
                    "Scout LLM violated JSON-only output contract.\n"
                    f"Raw output:\n{raw_output}"
                )

            expanded_queries = json.loads(raw_output)

            if not isinstance(expanded_queries, list) or not expanded_queries:
                raise RuntimeError(
                    "Scout output is not a non-empty JSON array."
                )

            if not all(isinstance(q, str) for q in expanded_queries):
                raise RuntimeError(
                    "Scout output array contains non-string elements."
                )

            logger.info(
                "SCOUT_QUERY_EXPANSION_END",
                extra={
                    "iteration": state.iteration_count,
                    "num_queries": len(expanded_queries),
                },
            )

            # --------------------------------------------------
            # Retrieval from PubMed
            # --------------------------------------------------
            retrieved_papers = []

            for q in expanded_queries:
                logger.info(
                    "SCOUT_PUBMED_SEARCH",
                    extra={
                        "iteration": state.iteration_count,
                        "expanded_query": q,
                    },
                )

                pmids = pubmed_search(q)

                logger.info(
                    "SCOUT_PUBMED_FETCH",
                    extra={
                        "iteration": state.iteration_count,
                        "pmid_count": len(pmids),
                    },
                )

                retrieved_papers.extend(pubmed_fetch(pmids))

            # --------------------------------------------------
            # IMPORTANT: create NEW state object
            # --------------------------------------------------
            new_state = state.model_copy(deep=True)

            new_state.expanded_queries = expanded_queries
            new_state.retrieved_papers = retrieved_papers
            new_state.iteration_count = state.iteration_count + 1

            logger.info(
                "SCOUT_NODE_END",
                extra={
                    "iteration": new_state.iteration_count,
                    "num_queries": len(expanded_queries),
                    "num_papers": len(retrieved_papers),
                },
            )

            return new_state

        except Exception as e:
            logger.exception(
                "SCOUT_NODE_ERROR",
                extra={
                    "iteration": state.iteration_count,
                    "error": str(e),
                },
            )
            raise
