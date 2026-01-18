"""
Title Generation Agent.

Generates concise, descriptive titles for conversations based on the first
user message. Uses Claude Haiku for cost efficiency.

Strategy:
- Called after the second message (not blocking first response)
- Generates a 3-6 word title summarizing the conversation topic
- Falls back to truncated first message if LLM fails
"""

import re
from typing import Optional

from langchain_aws import ChatBedrock

from app.logging import logger


# ============================================================================
# Prompts
# ============================================================================

TITLE_SYSTEM_PROMPT = """You are a title generator for a medical research assistant.

Given the first message of a conversation, generate a short, descriptive title.

Rules:
1. Title must be 3-6 words
2. Capture the main topic or intent
3. Use sentence case (capitalize first word only)
4. No quotes, periods, or special characters
5. Be specific but concise

Examples:
- "What are treatments for diabetes?" → "Diabetes treatment options"
- "Hello, how are you?" → "Greeting and introduction"
- "Find studies on COVID vaccine efficacy" → "COVID vaccine efficacy studies"
- "Can you explain the metformin results?" → "Metformin research follow-up"
- "Make it shorter please" → "Request for summary"

Output ONLY the title, nothing else."""

TITLE_USER_TEMPLATE = """First message: "{message}"

Generate a title:"""


# ============================================================================
# Title Generator Agent
# ============================================================================

class TitleGenerator:
    """
    Generates conversation titles from the first user message.
    """
    
    def __init__(self, llm_client: Optional[ChatBedrock] = None):
        """
        Initialize with optional LLM client.
        If not provided, creates a Haiku instance.
        """
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = ChatBedrock(
                model="anthropic.claude-3-haiku-20240307-v1:0",
                region_name="us-east-1",
            )
    
    def generate(self, first_message: str) -> str:
        """
        Generate a title from the first user message.
        
        Args:
            first_message: The user's first message in the conversation
            
        Returns:
            A 3-6 word title string
        """
        # Clean up the message
        message = first_message.strip()[:500]  # Limit input length
        
        try:
            # Call LLM
            messages = [
                {"role": "system", "content": TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": TITLE_USER_TEMPLATE.format(message=message)},
            ]
            
            response = self.llm.invoke(messages)
            title = response.content.strip()
            
            # Validate and clean title
            title = self._clean_title(title)
            
            logger.info("TITLE_GENERATED", extra={
                "title": title,
                "first_message": message[:50],
            })
            
            return title
            
        except Exception as e:
            logger.warning("TITLE_GENERATION_FAILED", extra={
                "error": str(e),
                "first_message": message[:50],
            })
            # Fallback to truncated first message
            return self._fallback_title(message)
    
    def _clean_title(self, title: str) -> str:
        """Clean and validate generated title."""
        # Remove quotes
        title = title.strip('"\'')
        
        # Remove trailing punctuation
        title = re.sub(r'[.!?]+$', '', title)
        
        # Limit length
        words = title.split()
        if len(words) > 8:
            title = ' '.join(words[:6])
        
        # Ensure minimum length
        if len(title) < 3:
            return "New conversation"
        
        # Limit character length
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title
    
    def _fallback_title(self, message: str) -> str:
        """Generate fallback title from message."""
        # Take first 50 chars, find word boundary
        if len(message) <= 50:
            return self._clean_title(message)
        
        truncated = message[:50]
        last_space = truncated.rfind(' ')
        if last_space > 20:
            truncated = truncated[:last_space]
        
        return truncated + "..."


# ============================================================================
# Module-level singleton
# ============================================================================

_title_generator: Optional[TitleGenerator] = None


def get_title_generator() -> TitleGenerator:
    """Get or create TitleGenerator singleton."""
    global _title_generator
    if _title_generator is None:
        _title_generator = TitleGenerator()
    return _title_generator