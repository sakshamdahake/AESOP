"""
Session schemas for multi-turn conversation orchestration.
"""

from datetime import datetime
from typing import List, Optional, Literal
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


class SessionContext(BaseModel):
    """
    Cached context from previous query in the session.
    Stored in Redis with TTL of 60 minutes.
    """
    session_id: str
    original_query: str
    query_embedding: List[float] = Field(default_factory=list)
    
    retrieved_papers: List[CachedPaper] = Field(default_factory=list)
    synthesis_summary: str = ""
    
    turn_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
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


class RouterDecision(BaseModel):
    """Output from Router Agent."""
    route: Literal["full_graph", "augmented_context", "context_qa"]
    reasoning: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    follow_up_focus: Optional[str] = None
    is_new_session: bool = False