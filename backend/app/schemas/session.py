"""
Session schemas for multi-turn conversation orchestration.
Updated to support both Redis (cache) and PostgreSQL (persistence).

backend/app/schemas/session.py
"""

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
import json


class CachedPaper(BaseModel):
    """Paper cached in session context."""
    pmid: str
    title: str
    abstract: str
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    relevance_score: Optional[float] = None
    methodology_score: Optional[float] = None
    quality_score: Optional[float] = None
    recommendation: Optional[str] = None


class AnswerSection(BaseModel):
    """A single section of a structured answer."""
    type: Literal["summary", "evidence", "methodology", "limitations", "recommendations"]
    content: str


class StructuredAnswer(BaseModel):
    """Structured answer with typed sections."""
    sections: List[AnswerSection] = Field(default_factory=list)
    
    @classmethod
    def from_flat_response(cls, response: str) -> "StructuredAnswer":
        """Convert a flat response string into a structured answer (single summary section)."""
        return cls(sections=[AnswerSection(type="summary", content=response)])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON storage."""
        return {"sections": [{"type": s.type, "content": s.content} for s in self.sections]}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredAnswer":
        """Create from dict (from database JSONB)."""
        if not data or "sections" not in data:
            return cls(sections=[])
        return cls(sections=[AnswerSection(**s) for s in data["sections"]])


class SessionMessage(BaseModel):
    """A single message in the session history."""
    role: Literal["user", "assistant"]
    content: Optional[str] = None  # For user messages
    answer: Optional[StructuredAnswer] = None  # For assistant messages
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None  # For intent, confidence, route, etc.
    
    # 🆕 Optional database fields
    id: Optional[str] = None  # Database message ID
    
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dict for database storage."""
        return {
            "role": self.role,
            "content": self.content,
            "answer": self.answer.to_dict() if self.answer else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        """Create from database dict."""
        # Handle both 'answer' and 'structured_answer' column names
        answer_data = data.get("structured_answer") or data.get("answer")
        
        return cls(
            id=data.get("id"),
            role=data["role"],
            content=data.get("content"),
            answer=StructuredAnswer.from_dict(answer_data) if answer_data else None,
            metadata=data.get("metadata"),
            timestamp=data.get("created_at", datetime.utcnow()),
        )



class SessionContext(BaseModel):
    """
    Session context for multi-turn conversations.
    
    Supports both:
    - Redis (cache layer, fast access)
    - PostgreSQL (persistence layer, permanent storage)
    
    All new fields are Optional to maintain backward compatibility.
    """
    session_id: str
    original_query: str
    query_embedding: List[float] = Field(default_factory=list)
    
    retrieved_papers: List[CachedPaper] = Field(default_factory=list)
    synthesis_summary: str = ""
    
    title: Optional[str] = None
    
    messages: List[SessionMessage] = Field(default_factory=list)
    
    turn_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # 🆕 Optional database fields (for PostgreSQL persistence)
    anonymous_id: Optional[str] = None  # User tracking (optional)
    deleted_at: Optional[datetime] = None  # Soft delete timestamp
    persisted: bool = False  # Flag: has this been saved to DB?
    
    def to_redis(self) -> str:
        """Serialize for Redis storage."""
        return self.model_dump_json()
    
    @classmethod
    def from_redis(cls, data: str) -> "SessionContext":
        """Deserialize from Redis with validation."""
        try:
            parsed = json.loads(data)
            
            # Ensure query_embedding is a list of floats
            if "query_embedding" in parsed and parsed["query_embedding"]:
                parsed["query_embedding"] = [float(x) for x in parsed["query_embedding"]]
            
            return cls.model_validate(parsed)
        except Exception as e:
            raise ValueError(f"Failed to deserialize SessionContext: {e}")
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(SessionMessage(role="user", content=content))
    
    def add_assistant_message(
        self,
        answer: StructuredAnswer,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an assistant message to history."""
        self.messages.append(SessionMessage(
            role="assistant",
            answer=answer,
            metadata=metadata,
        ))
    
    def get_papers_context(self, max_papers: int = 10) -> str:
        """Format papers for LLM context injection."""
        if not self.retrieved_papers:
            return "No papers available from previous search."
        
        blocks = []
        for i, paper in enumerate(self.retrieved_papers[:max_papers], 1):
            blocks.append(
                f"[Paper {i}]\n"
                f"PMID: {paper.pmid}\n"
                f"Title: {paper.title}\n"
                f"Quality Score: {paper.quality_score or 'N/A'}\n"
                f"Abstract: {paper.abstract[:600]}..."
            )
        return "\n\n---\n\n".join(blocks)
    
    def generate_title(self) -> str:
        """Generate title from original query (truncated)."""
        if self.title:
            return self.title
        if len(self.original_query) > 50:
            return self.original_query[:47] + "..."
        return self.original_query
    
    def get_recent_messages_for_context(self, limit: int = 6) -> str:
        """
        Format recent messages for LLM context injection.
        Used by Intent Classifier and Chat Agent.
        
        Args:
            limit: Number of recent messages to include (default 6)
        
        Returns:
            Formatted string of recent messages
        """
        if not self.messages:
            return "No previous messages."
        
        recent = self.messages[-limit:] if len(self.messages) > limit else self.messages
        
        formatted = []
        for msg in recent:
            if msg.role == "user":
                formatted.append(f"User: {msg.content}")
            else:
                # For assistant, show first section content (summary)
                if msg.answer and msg.answer.sections:
                    content = msg.answer.sections[0].content[:200]
                    formatted.append(f"Assistant: {content}...")
                else:
                    formatted.append("Assistant: [No content]")
        
        return "\n".join(formatted)


class RouterDecision(BaseModel):
    """Output from Router Agent."""
    route: Literal["full_graph", "augmented_context", "context_qa"]
    reasoning: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    follow_up_focus: Optional[str] = None
    is_new_session: bool = False