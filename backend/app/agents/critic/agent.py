from typing import List, Dict, Any
from collections import Counter, defaultdict
import json

from pydantic import ValidationError

from app.logging import logger

from .schemas import PaperGrade, Recommendation
from .rubric import (
    STUDY_TYPE_PRIORS,
    MIN_AVG_QUALITY_FOR_SUFFICIENT,
    MAX_DISCARD_RATIO,
    CONFIDENCE_DECAY_RATE,
    MIN_CONFIDENCE_FLOOR,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def parse_strict_json(text: str) -> dict:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("LLM output violated JSON-only contract")
    return json.loads(text)


def clamp_score(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


ACCEPTANCE_MEMORY = defaultdict(list)


class CriticAgent:
    """
    CRAG-enabled Critic Agent (SYNC).
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    # -----------------------------
    # Single abstract grading
    # -----------------------------

    def grade_abstract(
        self,
        research_question: str,
        abstract: str,
    ) -> PaperGrade:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    research_question=research_question,
                    abstract=abstract,
                ),
            },
        ]

        response = self.llm.invoke(messages)

        try:
            parsed = parse_strict_json(response.content)

            # 🔒 Never trust LLM with identifiers
            parsed.pop("pmid", None)

            parsed["relevance_score"] = clamp_score(
                parsed.get("relevance_score", 0.0)
            )
            parsed["methodology_score"] = clamp_score(
                parsed.get("methodology_score", 0.0)
            )

            grade = PaperGrade.model_validate(parsed)

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(
                "CriticAgent failed strict JSON or schema validation.\n"
                f"Error: {e}\n"
                f"Raw output:\n{response.content}"
            ) from e

        if grade.study_type:
            prior = STUDY_TYPE_PRIORS.get(
                grade.study_type.lower(), 0.20
            )
            grade.methodology_score = max(
                grade.methodology_score,
                prior,
            )

        return grade

    # -----------------------------
    # Batch grading (papers, not abstracts)
    # -----------------------------

    def grade_batch(
        self,
        research_question: str,
        papers: List[Any],
        iteration: int = 0,
    ) -> Dict[str, Any]:

        grades: List[PaperGrade] = []

        for paper in papers:
            grade = self.grade_abstract(
                research_question=research_question,
                abstract=paper.abstract,
            )

            # 🔒 Inject trusted pmid
            grade.pmid = paper.pmid

            grades.append(grade)

        decision = self._make_global_decision(
            grades=grades,
            iteration=iteration,
        )

        if decision == "sufficient":
            self._record_acceptance(grades, iteration)

        return {
            "grades": grades,
            "decision": decision,
        }

    # -----------------------------
    # Global CRAG decision logic
    # -----------------------------

    def _make_global_decision(
        self,
        grades: List[PaperGrade],
        iteration: int,
    ) -> str:

        if not grades:
            return "retrieve_more"

        counts = Counter(g.recommendation for g in grades)
        total = len(grades)

        keep_ratio = counts[Recommendation.KEEP] / total
        discard_ratio = counts[Recommendation.DISCARD] / total
        needs_more_ratio = counts[Recommendation.NEEDS_MORE] / total

        avg_quality = sum(
            (g.relevance_score + g.methodology_score) / 2
            for g in grades
        ) / total

        effective_threshold = max(
            MIN_CONFIDENCE_FLOOR,
            MIN_AVG_QUALITY_FOR_SUFFICIENT
            - (iteration * CONFIDENCE_DECAY_RATE),
        )

        logger.info(
            "CRITIC_CRAG_METRICS",
            extra={
                "iteration": iteration,
                "num_papers": total,
                "keep_ratio": round(keep_ratio, 3),
                "discard_ratio": round(discard_ratio, 3),
                "needs_more_ratio": round(needs_more_ratio, 3),
                "avg_quality": round(avg_quality, 3),
                "effective_threshold": round(effective_threshold, 3),
            },
        )

        if keep_ratio >= 0.40:
            return "sufficient"

        if discard_ratio >= 0.40:
            return "retrieve_more"

        if avg_quality >= effective_threshold and discard_ratio <= MAX_DISCARD_RATIO:
            return "sufficient"

        return "retrieve_more"

    # -----------------------------
    # Learning store
    # -----------------------------

    def _record_acceptance(
        self,
        grades: List[PaperGrade],
        iteration: int,
    ) -> None:
        for g in grades:
            if g.recommendation == Recommendation.KEEP:
                ACCEPTANCE_MEMORY[g.study_type].append(
                    {
                        "quality": (g.relevance_score + g.methodology_score) / 2,
                        "iteration": iteration,
                    }
                )
