"""
Agent state models for AESOP system.
Includes both original AgentState and OrchestratorState with intent/chat support.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from app.agents.critic.schemas import PaperGrade
from app.schemas.session import SessionContext, RouterDecision, CachedPaper


class Paper(BaseModel):
    """Paper retrieved from PubMed."""
    pmid: str
    title: str
    abstract: str
    publication_year: Optional[int] = None
    journal: Optional[str] = None


class AgentState(BaseModel):
    """
    Original state for the basic CRAG graph (Scout → Critic → Synthesizer).
    Kept for backward compatibility.
    """
    # User input
    query: str

    # Scout output
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_papers: List[Paper] = Field(default_factory=list)

    # Critic output
    grades: List[PaperGrade] = Field(default_factory=list)
    critic_decision: Optional[str] = None  # "sufficient" | "retrieve_more"
    critic_explanation: Optional[str] = None
    avg_quality: Optional[float] = None
    discard_ratio: Optional[float] = None

    # Synthesizer output
    synthesis_output: Optional[str] = None

    # Control
    iteration_count: int = 0
    max_iterations: int = 1


# Intent type definition
IntentType = Literal["research", "followup_research", "chat", "utility"]


class OrchestratorState(BaseModel):
    """
    Extended state for the session-aware orchestrator graph.
    Includes intent classification, routing, and chat support.
    """
    # =====================
    # User input
    # =====================
    query: str
    session_id: str = ""  # Empty string for new sessions
    
    # =====================
    # Intent classification (NEW)
    # =====================
    intent: Optional[IntentType] = None
    intent_confidence: Optional[float] = None
    intent_reasoning: Optional[str] = None
    
    # =====================
    # Chat response (NEW)
    # =====================
    chat_response: Optional[str] = None
    
    # =====================
    # Utility response (NEW)
    # =====================
    utility_response: Optional[str] = None
    
    # =====================
    # Router output
    # =====================
    router_decision: Optional[RouterDecision] = None
    session_context: Optional[SessionContext] = None
    route_taken: Optional[str] = None  # "full_graph" | "augmented_context" | "context_qa" | "chat" | "utility"
    
    # =====================
    # Scout output (Route A & B)
    # =====================
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_papers: List[Paper] = Field(default_factory=list)
    
    # =====================
    # Critic output (Route A only)
    # =====================
    grades: List[PaperGrade] = Field(default_factory=list)
    critic_decision: Optional[str] = None
    critic_explanation: Optional[str] = None
    avg_quality: Optional[float] = None
    discard_ratio: Optional[float] = None
    
    # =====================
    # Route B: Merged papers
    # =====================
    merged_papers: List[CachedPaper] = Field(default_factory=list)
    
    # =====================
    # Synthesizer output (research routes)
    # =====================
    synthesis_output: Optional[str] = None
    
    # =====================
    # Control
    # =====================
    iteration_count: int = 0
    max_iterations: int = 1
    
    def get_response(self) -> str:
        """
        Get the appropriate response based on route taken.
        """
        if self.route_taken == "chat":
            return self.chat_response or ""
        elif self.route_taken == "utility":
            return self.utility_response or ""
        else:
            return self.synthesis_output or ""