from typing import List

from app.agents.state import Paper
from app.agents.synthesizer.schemas import GradedPaper


from app.agents.critic.schemas import PaperGrade


def build_graded_papers(
    papers: List[Paper],
    grades: List[PaperGrade],
) -> List[GradedPaper]:
    graded = []

    grade_map: dict[str, float] = {}

    for g in grades:
        score = (
            (g.relevance_score + g.methodology_score) / 2
        )

        if g.recommendation == "discard":
            continue

        if not g.sample_size_adequate:
            score *= 0.7

        grade_map[g.pmid] = score

    for paper in papers:
        score = grade_map.get(paper.pmid)
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
