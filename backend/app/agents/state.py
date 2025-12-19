from typing import List, Optional
from pydantic import BaseModel, Field

from app.agents.critic.schemas import PaperGrade


class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
    publication_year: Optional[int] = None
    journal: Optional[str] = None


class AgentState(BaseModel):
    # User input
    query: str

    # Scout output
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_papers: List[Paper] = Field(default_factory=list)

    # Critic output
    grades: List[PaperGrade] = Field(default_factory=list)
    critic_decision: Optional[str] = None  # "sufficient" | "retrieve_more"

    # Synthesizer output
    synthesis_output: Optional[str] = None

    # Control
    iteration_count: int = 0
    max_iterations: int = 3
