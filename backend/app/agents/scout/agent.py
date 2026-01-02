import json
import re

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
# JSON Parsing Utilities
# ============================

def extract_json_array(text: str) -> list:
    """
    Robustly extract a JSON array from LLM output.
    Handles common LLM issues like markdown fences, extra text, etc.
    """
    text = text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        # Remove closing fence
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
    
    # Try to find JSON array in the text
    # Look for [...] pattern
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        json_str = match.group()
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    
    # If direct parsing fails, try to extract strings manually
    # Look for quoted strings
    strings = re.findall(r'"([^"]*)"', text)
    if strings:
        # Filter out very short or empty strings
        return [s.strip() for s in strings if len(s.strip()) > 3]
    
    # Last resort: split by newlines and clean up
    lines = text.split('\n')
    queries = []
    for line in lines:
        line = line.strip()
        # Remove list markers like "1.", "-", "*"
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        # Remove quotes
        line = line.strip('"\'')
        if len(line) > 5 and not line.startswith('[') and not line.startswith(']'):
            queries.append(line)
    
    return queries[:5]  # Max 5 queries


# ============================
# LangGraph Scout Node (SYNC)
# ============================

def scout_node(state: AgentState) -> AgentState:
    """
    Synchronous LangGraph Scout node.

    HARD CONTRACT:
    - LLM MUST return a JSON array of query strings
    - Any violation triggers fallback parsing
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
            # Robust JSON parsing with fallback
            # --------------------------------------------------
            try:
                # First try: strict JSON parsing
                if raw_output.startswith("[") and raw_output.endswith("]"):
                    expanded_queries = json.loads(raw_output)
                else:
                    raise ValueError("Output doesn't start/end with brackets")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "SCOUT_JSON_PARSE_FALLBACK",
                    extra={
                        "iteration": state.iteration_count,
                        "error": str(e),
                        "raw_output_preview": raw_output[:200],
                    },
                )
                # Fallback: robust extraction
                expanded_queries = extract_json_array(raw_output)
            
            # Validate we have queries
            if not expanded_queries:
                logger.warning(
                    "SCOUT_NO_QUERIES_EXTRACTED",
                    extra={"iteration": state.iteration_count},
                )
                # Ultimate fallback: use original query
                expanded_queries = [state.query]
            
            # Ensure all items are strings
            expanded_queries = [str(q) for q in expanded_queries if q]
            
            # Cap at 5 queries
            expanded_queries = expanded_queries[:5]

            logger.info(
                "SCOUT_QUERY_EXPANSION_END",
                extra={
                    "iteration": state.iteration_count,
                    "num_queries": len(expanded_queries),
                    "queries": expanded_queries,
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
                        "expanded_query": q[:100],
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