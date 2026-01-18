"""
API Request/Response schemas for AESOP endpoints.
Implements REST-compliant session and message handling.

# backend/app/schemas/api.py
"""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from app.schemas.session import StructuredAnswer, AnswerSection, SessionMessage


# --- Session Endpoints ---

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    initial_message: Optional[str] = Field(
        None,
        min_length=1,
        max_length=2000,
        description="Optional initial message to start the conversation",
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"initial_message": "What are treatments for Type 2 diabetes?"},
                {},  # Empty request creates session without message
            ]
        }


class CreateSessionResponse(BaseModel):
    """Response after creating a new session."""
    session_id: str
    created_at: datetime
    title: Optional[str] = None
    # If initial_message was provided, include the response
    initial_response: Optional["MessageResponse"] = None


class SessionSummary(BaseModel):
    """Summary of a session for list view (sidebar)."""
    session_id: str
    title: str
    updated_at: datetime
    message_count: int = 0


class ListSessionsResponse(BaseModel):
    """Response containing list of sessions."""
    sessions: List[SessionSummary]


class SessionDetailResponse(BaseModel):
    """Detailed session response with full message history."""
    session_id: str
    title: str
    messages: List[SessionMessage]
    papers_count: int = 0
    created_at: datetime
    updated_at: datetime


# --- Message Endpoints ---

class SendMessageRequest(BaseModel):
    """Request to send a message to a session."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message",
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"message": "What sample sizes did these studies use?"},
                {"message": "Summarize the key findings"},
            ]
        }


class MessageMetadata(BaseModel):
    """Metadata about message processing."""
    intent: Optional[str] = Field(None, description="Classified intent: research, followup_research, chat, utility")
    intent_confidence: Optional[float] = Field(None, description="Intent classification confidence")
    processing_route: str = Field(..., description="Execution route: chat, utility, full_graph, augmented_context, context_qa")
    papers_count: int = Field(0, description="Number of papers in context")
    review_outcome: Optional[str] = Field(None, description="CRAG decision: sufficient, retrieve_more")
    evidence_score: Optional[float] = Field(None, description="Average evidence quality score")


class MessageResponse(BaseModel):
    """Response after sending a message."""
    answer: StructuredAnswer
    metadata: MessageMetadata
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": {
                    "sections": [
                        {
                            "type": "summary",
                            "content": "The studies enrolled between 3,000 and 15,000 participants."
                        }
                    ]
                },
                "metadata": {
                    "intent": "follow_up",
                    "intent_confidence": 0.91,
                    "processing_route": "context_qa",
                    "papers_count": 12,
                    "review_outcome": None,
                    "evidence_score": None
                }
            }
        }


# --- Streaming Events ---

class StreamEvent(BaseModel):
    """Base class for streaming events."""
    event: Literal["section_start", "token", "section_end", "metadata", "error"]


class SectionStartEvent(StreamEvent):
    """Event when a new section starts."""
    event: Literal["section_start"] = "section_start" # type: ignore
    type: Literal["summary", "evidence", "methodology", "limitations", "recommendations"]


class TokenEvent(StreamEvent):
    """Event for a content token."""
    event: Literal["token"] = "token" # type: ignore
    content: str


class SectionEndEvent(StreamEvent):
    """Event when a section ends."""
    event: Literal["section_end"] = "section_end" # type: ignore


class MetadataEvent(StreamEvent):
    """Event containing response metadata."""
    event: Literal["metadata"] = "metadata" # type: ignore
    data: MessageMetadata


class ErrorEvent(StreamEvent):
    """Event for errors during streaming."""
    event: Literal["error"] = "error" # type: ignore
    message: str
    code: Optional[str] = None


# --- Legacy Endpoints (deprecated but maintained) ---

class ChatRequest(BaseModel):
    """
    DEPRECATED: Use POST /sessions/{session_id}/messages instead.
    Request model for legacy /chat endpoint.
    """
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for follow-ups")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "message": "What are the treatments for Type 2 diabetes?",
                    "session_id": None
                },
                {
                    "message": "What sample sizes did these studies use?",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000"
                },
            ]
        }


class ChatResponse(BaseModel):
    """
    DEPRECATED: Use MessageResponse instead.
    Response model for legacy /chat endpoint.
    """
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID for follow-ups")
    route_taken: str = Field(..., description="Execution route")
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    papers_count: int = 0
    critic_decision: Optional[str] = None
    avg_quality: Optional[float] = None
    
    # New: Include structured version for gradual migration
    structured_answer: Optional[StructuredAnswer] = None


class ReviewRequest(BaseModel):
    """Legacy request model (backward compatible)."""
    query: str = Field(..., min_length=5, max_length=2000)
    session_id: Optional[str] = None


class ReviewResponse(BaseModel):
    """Legacy response model (backward compatible)."""
    response: str
    session_id: str
    route_taken: str
    papers_count: int
    critic_decision: Optional[str] = None
    avg_quality: Optional[float] = None


# Resolve forward references
CreateSessionResponse.model_rebuild()
