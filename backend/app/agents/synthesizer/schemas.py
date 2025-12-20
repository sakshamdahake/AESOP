from pydantic import BaseModel
from typing import List


class GradedPaper(BaseModel):
    pmid: str
    title: str
    abstract: str
    score: float


class SynthesisInput(BaseModel):
    query: str
    papers: List[GradedPaper]


class SynthesisOutput(BaseModel):
    review_text: str
