"""
Conversation schemas for persistent multi-turn chat.

These models define the structure for:
- Conversations (metadata, user linking)
- Messages (user and assistant turns)
- Research contexts (query state, synthesis, papers)
- API request/response formats
"""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import json


# ============================================================================
# Core Database Models
# ============================================================================

class MessageRecord(BaseModel):
    """
    Single message in a conversation.
    Maps directly to the `messages` table.
    """
    id: Optional[str] = None
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    response_type: Optional[str] = None  # 'chat', 'research', 'utility', 'context_qa'
    sequence_num: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class ResearchPaperRecord(BaseModel):
    """
    Paper retrieved during research.
    Maps to the `research_papers` table.
    """
    id: Optional[str] = None
    research_context_id: str
    pmid: str
    title: str
    abstract: Optional[str] = None
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    relevance_score: Optional[float] = None
    methodology_score: Optional[float] = None
    quality_score: Optional[float] = None
    recommendation: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class ResearchContextRecord(BaseModel):
    """
    Research query context for a single research interaction.
    Maps to the `research_contexts` table.
    """
    id: Optional[str] = None
    conversation_id: str
    query: str
    query_embedding: Optional[List[float]] = None
    synthesis_summary: Optional[str] = None
    route_taken: Optional[str] = None
    intent: Optional[str] = None
    papers_count: int = 0
    critic_decision: Optional[str] = None
    avg_quality: Optional[float] = None
    discard_ratio: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Nested papers (populated on fetch)
    papers: List[ResearchPaperRecord] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ConversationRecord(BaseModel):
    """
    Conversation metadata.
    Maps to the `conversations` table.
    """
    id: Optional[str] = None
    anonymous_id: Optional[str] = None
    title: str = "New Conversation"
    title_generated: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = None
    
    # Nested data (populated on fetch)
    messages: List[MessageRecord] = Field(default_factory=list)
    research_contexts: List[ResearchContextRecord] = Field(default_factory=list)
    
    # Computed fields for list view
    message_count: Optional[int] = None
    last_message_preview: Optional[str] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Redis Cache Models
# ============================================================================

class CachedConversation(BaseModel):
    """
    Lightweight conversation cache stored in Redis.
    Contains only recent messages for fast LLM context injection.
    """
    conversation_id: str
    anonymous_id: Optional[str] = None
    title: str = "New Conversation"
    title_generated: bool = False
    
    # Last N message pairs (user + assistant)
    recent_messages: List[MessageRecord] = Field(default_factory=list)
    
    # Quick access to latest research context
    last_research_context_id: Optional[str] = None
    last_synthesis_summary: Optional[str] = None
    
    # Metadata
    message_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_redis(self) -> str:
        """Serialize for Redis storage."""
        return self.model_dump_json()
    
    @classmethod
    def from_redis(cls, data: str) -> "CachedConversation":
        """Deserialize from Redis."""
        return cls.model_validate_json(data)
    
    def get_messages_for_llm(self, limit: int = 10) -> List[dict]:
        """
        Format recent messages for LLM context injection.
        Returns list of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        messages = []
        for msg in self.recent_messages[-limit:]:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        return messages
    
    def has_conversation_started(self) -> bool:
        """Check if conversation has any messages (to avoid repeated greetings)."""
        return self.message_count > 0 or len(self.recent_messages) > 0


# ============================================================================
# API Request Models
# ============================================================================

class ConversationChatRequest(BaseModel):
    """
    Request body for POST /conversations/chat
    """
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID, or null to create new")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "message": "What are the treatments for Type 2 diabetes?",
                    "conversation_id": None
                },
                {
                    "message": "Tell me more about metformin",
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000"
                }
            ]
        }


class ConversationTitleUpdateRequest(BaseModel):
    """
    Request body for PATCH /conversations/{id}/title
    """
    title: str = Field(..., min_length=1, max_length=255, description="New conversation title")


# ============================================================================
# API Response Models
# ============================================================================

class ConversationChatResponse(BaseModel):
    """
    Response for POST /conversations/chat
    """
    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID for follow-ups")
    
    # Routing info
    route_taken: str = Field(..., description="Execution route: chat, utility, full_graph, augmented_context, context_qa")
    intent: Optional[str] = Field(None, description="Classified intent")
    intent_confidence: Optional[float] = Field(None, description="Intent classification confidence")
    
    # Research metrics (if applicable)
    papers_count: int = Field(0, description="Number of papers in context")
    critic_decision: Optional[str] = Field(None, description="CRAG decision")
    avg_quality: Optional[float] = Field(None, description="Average evidence quality")
    
    # Conversation metadata
    title: str = Field(..., description="Conversation title")
    is_new_conversation: bool = Field(..., description="Whether this created a new conversation")


class ConversationListItem(BaseModel):
    """
    Single item in conversation list response.
    """
    id: str
    title: str
    updated_at: datetime
    created_at: datetime
    message_count: int
    last_message_preview: Optional[str] = None


class ConversationListResponse(BaseModel):
    """
    Response for GET /conversations
    """
    conversations: List[ConversationListItem]
    total: int
    limit: int
    offset: int


class ConversationDetailResponse(BaseModel):
    """
    Response for GET /conversations/{id}
    Full conversation with messages and optionally research contexts.
    """
    id: str
    title: str
    anonymous_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    messages: List[MessageRecord]
    message_count: int
    
    # Optional research data
    research_contexts: Optional[List[ResearchContextRecord]] = None


class ConversationDeleteResponse(BaseModel):
    """
    Response for DELETE /conversations/{id}
    """
    status: str = "deleted"
    conversation_id: str