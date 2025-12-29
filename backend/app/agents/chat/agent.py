"""
Chat Agent - Handles general conversation and system questions.
"""

from typing import Optional

from app.schemas.session import SessionContext
from app.agents.chat.prompts import (
    CHAT_SYSTEM_PROMPT,
    CHAT_USER_TEMPLATE,
    get_canned_response,
)
from app.logging import logger


class ChatAgent:
    """
    Handles non-research conversations:
    - Greetings and farewells
    - System capability questions
    - General conversation
    - Thanks and acknowledgments
    
    Uses canned responses for common cases, LLM for nuanced conversation.
    """
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: ChatBedrock instance (Haiku for cost efficiency)
        """
        self.llm = llm_client
    
    def respond(
        self,
        message: str,
        session_context: Optional[SessionContext] = None,
    ) -> str:
        """
        Generate a chat response.
        
        Args:
            message: User's message
            session_context: Optional session for context awareness
            
        Returns:
            Response string
        """
        message = message.strip()
        
        # Try canned response first (fast, no LLM cost)
        canned = get_canned_response(message)
        if canned:
            logger.info(
                "CHAT_CANNED_RESPONSE",
                extra={"user_input": message[:30]},
            )
            return canned
        
        # Use LLM for nuanced conversation
        return self._llm_respond(message, session_context)
    
    def _llm_respond(
        self,
        message: str,
        session_context: Optional[SessionContext],
    ) -> str:
        """
        Generate response using LLM.
        """
        # Format context
        has_session = session_context is not None
        previous_topic = ""
        if session_context:
            previous_topic = session_context.original_query[:100]
        
        user_message = CHAT_USER_TEMPLATE.format(
            has_session="Yes" if has_session else "No",
            previous_topic=previous_topic or "None",
            message=message,
        )
        
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            logger.info(
                "CHAT_LLM_RESPONSE",
                extra={
                    "user_input": message[:30],
                    "response_length": len(content),
                },
            )
            
            return content
            
        except Exception as e:
            logger.error(
                "CHAT_LLM_ERROR",
                extra={"error": str(e), "user_input": message[:30]},
            )
            return (
                "I apologize, but I encountered an issue processing your message. "
                "If you have a medical research question, feel free to ask and I'll search the literature for you!"
            )