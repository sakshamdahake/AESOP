from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"
    NEEDS_MORE = "needs_more"


class PaperGrade(BaseModel):
    """
    Structured evaluation of a single paper abstract.
    """

    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Topical relevance to the research question"
    )
    methodology_score: float = Field(
        ..., ge=0.0, le=1.0, description="Methodological rigor"
    )
    sample_size_adequate: bool = Field(
        ..., description="Whether sample size is adequate for the study type"
    )
    study_type: Optional[str] = Field(
        None, description="RCT, Cohort, Case-Control, Case Study, Review, etc."
    )
    recommendation: Recommendation = Field(
        ..., description="keep | discard | needs_more"
    )
