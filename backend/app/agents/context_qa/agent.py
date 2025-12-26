"""
Context Q&A Agent - Answers follow-up questions using cached papers only.
Route C: Cheapest path, no retrieval needed.
"""

from typing import Optional

from app.schemas.session import SessionContext
from app.logging import logger


CONTEXT_QA_SYSTEM_PROMPT = """You are a medical research assistant answering follow-up questions about previously retrieved literature.

You have access to papers from a previous search. Answer the user's question using ONLY the information in these papers.

## Guidelines

- Be specific and cite paper numbers or PMIDs (e.g., "Paper 2 found that..." or "According to PMID 12345678...")
- If the papers don't contain enough information to answer, say so clearly
- Do NOT hallucinate or add information not in the papers
- Keep responses concise but thorough
- Use appropriate medical terminology
- Note any limitations or conflicts between papers

## Response Structure

For analytical questions: Provide a structured analysis with evidence from specific papers.
For clarification questions: Give a direct, focused answer with citations.
For comparison questions: Create a clear comparison referencing specific papers.
"""


CONTEXT_QA_USER_TEMPLATE = """## Original Research Question
{original_query}

## Retrieved Papers
{papers_context}

## Previous Synthesis Summary
{synthesis_summary}

---

## Follow-up Question
{current_query}

Answer the follow-up question using ONLY the information from the papers above."""


class ContextQAAgent:
    """
    Handles Route C - Direct Q&A using cached context.
    No retrieval, just LLM inference over cached papers.
    """
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: ChatBedrock instance (Haiku for cost efficiency)
        """
        self.llm = llm_client
    
    def answer(
        self,
        current_query: str,
        session_context: SessionContext,
    ) -> str:
        """
        Answer follow-up question using cached papers.
        
        SYNC implementation.
        """
        if not session_context or not session_context.retrieved_papers:
            return (
                "I don't have any papers from a previous search to reference. "
                "Please ask a new research question to start a fresh literature search."
            )
        
        user_message = CONTEXT_QA_USER_TEMPLATE.format(
            original_query=session_context.original_query,
            papers_context=session_context.get_papers_context(max_papers=10),
            synthesis_summary=session_context.synthesis_summary or "No summary available.",
            current_query=current_query,
        )
        
        messages = [
            {"role": "system", "content": CONTEXT_QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info(
                "CONTEXT_QA_SUCCESS",
                extra={
                    "query": current_query[:50],
                    "papers_used": len(session_context.retrieved_papers),
                    "answer_length": len(answer),
                },
            )
            
            return answer
            
        except Exception as e:
            logger.error(
                "CONTEXT_QA_ERROR",
                extra={"error": str(e)},
            )
            return (
                "I encountered an error answering your follow-up question. "
                "Please try rephrasing or start a new search."
            )