"""
Utility Agent - Reformats and transforms existing output.
"""

from typing import Optional

from app.schemas.session import SessionContext
from app.logging import logger


UTILITY_SYSTEM_PROMPT = """You are a formatting assistant for AESOP, a biomedical literature review system.

Your job is to reformat or transform existing research summaries based on user requests.

## Common Tasks

1. **Shorten/Summarize**: Create a more concise version
2. **Bullet Points**: Convert to bullet point format
3. **Key Points Only**: Extract just the main findings
4. **Conclusion Only**: Provide just the conclusion section
5. **Simplify**: Use simpler language for non-experts
6. **Table Format**: Organize information in a table

## Guidelines

- Preserve all factual information and citations (PMIDs)
- Maintain scientific accuracy
- Keep the same evidence-based tone
- Don't add new information not in the original
- Clearly organize the reformatted content

## Output

Provide the reformatted content directly without preamble like "Here's the reformatted version".
"""


UTILITY_USER_TEMPLATE = """## Original Research Summary
{original_summary}

## User Request
{request}

Reformat the summary according to the user's request."""


class UtilityAgent:
    """
    Transforms existing output based on user requests.
    - Shorten/summarize
    - Convert to bullet points
    - Extract specific sections
    - Simplify language
    """
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: ChatBedrock instance
        """
        self.llm = llm_client
    
    def transform(
        self,
        request: str,
        session_context: SessionContext,
    ) -> str:
        """
        Transform existing output based on user request.
        
        Args:
            request: User's transformation request
            session_context: Session with previous synthesis output
            
        Returns:
            Transformed content
        """
        if not session_context or not session_context.synthesis_summary:
            return (
                "I don't have any previous research summary to reformat. "
                "Please ask a research question first, then I can help transform the results."
            )
        
        user_message = UTILITY_USER_TEMPLATE.format(
            original_summary=session_context.synthesis_summary,
            request=request,
        )
        
        messages = [
            {"role": "system", "content": UTILITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            logger.info(
                "UTILITY_TRANSFORM",
                extra={
                    "request": request[:30],
                    "original_length": len(session_context.synthesis_summary),
                    "result_length": len(content),
                },
            )
            
            return content
            
        except Exception as e:
            logger.error(
                "UTILITY_ERROR",
                extra={"error": str(e), "request": request[:30]},
            )
            return (
                "I encountered an error reformatting the content. "
                "Please try again or ask in a different way."
            )