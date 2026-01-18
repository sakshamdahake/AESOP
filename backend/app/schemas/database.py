"""
Database record schemas for PostgreSQL persistence.
These are INTERNAL models - not used in API responses.

Bridges between:
- PostgreSQL rows (from database.py)
- SessionContext (from session.py, used in Redis + API)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    """Database record for sessions table."""
    id: str
    anonymous_id: Optional[str] = None
    title: Optional[str] = None
    original_query: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    turn_count: int = 0
    message_count: int = 0
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None
    
    # Optional: populated if include_messages=True
    messages: List["MessageRecord"] = Field(default_factory=list)


class MessageRecord(BaseModel):
    """Database record for messages table."""
    id: str
    session_id: str
    role: str  # 'user' or 'assistant'
    content: Optional[str] = None  # User messages
    answer: Optional[Dict[str, Any]] = None  # Assistant: StructuredAnswer as dict
    metadata: Optional[Dict[str, Any]] = None  # Assistant: MessageMetadata as dict
    created_at: datetime


class ResearchContextRecord(BaseModel):
    """Database record for research_contexts table."""
    id: str
    session_id: str
    research_query: str
    query_embedding: Optional[List[float]] = None
    synthesis_summary: Optional[str] = None
    route_taken: Optional[str] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    papers_count: int = 0
    critic_decision: Optional[str] = None
    avg_quality: Optional[float] = None
    created_at: datetime


class ResearchPaperRecord(BaseModel):
    """Database record for research_papers table."""
    id: str
    research_context_id: str
    pmid: str
    title: str
    abstract: Optional[str] = None
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    relevance_score: Optional[float] = None
    methodology_score: Optional[float] = None
    quality_score: Optional[float] = None
    recommendation: Optional[str] = None
    study_type: Optional[str] = None
    created_at: datetime