from typing import List, Dict, Optional
from pydantic import BaseModel, Field


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
    grades: Dict[str, float] = Field(default_factory=dict)

    # Synthesizer output
    synthesis_output: Optional[str] = None

    # Control
    iteration_count: int = 0
    max_iterations: int = 3
