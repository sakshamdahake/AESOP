from typing import List

from app.agents.state import Paper
from app.agents.synthesizer.schemas import GradedPaper


def build_graded_papers(
    papers: List[Paper],
    grades: dict[str, float],
) -> List[GradedPaper]:
    graded = []

    for paper in papers:
        score = grades.get(paper.pmid)
        if score is None:
            continue

        graded.append(
            GradedPaper(
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                score=score,
            )
        )

    return graded


def format_papers_for_prompt(graded_papers: List[GradedPaper]) -> str:
    """
    Convert graded papers into a strict, LLM-friendly text block.
    """
    blocks = []

    for paper in graded_papers:
        blocks.append(
            f"""
PMID: {paper.pmid}
Quality Score: {paper.score}
Title: {paper.title}
Abstract: {paper.abstract}
""".strip()
        )

    return "\n\n".join(blocks)
