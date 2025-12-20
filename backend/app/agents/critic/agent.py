from typing import List, Dict, Any
from collections import Counter, defaultdict

from pydantic import ValidationError

from .schemas import PaperGrade, Recommendation
from .rubric import (
    STUDY_TYPE_PRIORS,
    MIN_AVG_QUALITY_FOR_SUFFICIENT,
    MAX_DISCARD_RATIO,
    CONFIDENCE_DECAY_RATE,
    MIN_CONFIDENCE_FLOOR,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# -------------------------------------------------
# Lightweight in-memory learning store (sync-safe)
# -------------------------------------------------

ACCEPTANCE_MEMORY = defaultdict(list)


class CriticAgent:
    """
    CR์AG-enabled Critic Agent (SYNCHRONOUS VERSION).

    This version is intentionally sync to match the
    current Scout + Synthesizer agents and reduce complexity.
    """

    def __init__(self, llm_client):
        """
        llm_client must support `invoke(messages)`
        """
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
            grade = PaperGrade.model_validate_json(response.content)
        except ValidationError as e:
            raise RuntimeError(
                f"CriticAgent schema validation failed: {e}"
            ) from e

        # -----------------------------
        # Apply study-type prior (EBM)
        # -----------------------------
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
    # Batch grading (CRAG step)
    # -----------------------------

    def grade_batch(
        self,
        research_question: str,
        abstracts: List[str],
        iteration: int = 0,
    ) -> Dict[str, Any]:
        grades: List[PaperGrade] = []

        for abstract in abstracts:
            grade = self.grade_abstract(
                research_question, abstract
            )
            grades.append(grade)

        decision = self._make_global_decision(
            grades=grades,
            iteration=iteration,
        )

        # -----------------------------
        # Learning from acceptances
        # -----------------------------
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

        # Disagreement-aware CRAG
        counts = Counter(g.recommendation for g in grades)
        total = len(grades)

        keep_ratio = counts[Recommendation.KEEP] / total
        discard_ratio = counts[Recommendation.DISCARD] / total

        if keep_ratio >= 0.60:
            return "sufficient"

        if discard_ratio >= 0.50:
            return "retrieve_more"

        # Confidence decay
        effective_threshold = max(
            MIN_CONFIDENCE_FLOOR,
            MIN_AVG_QUALITY_FOR_SUFFICIENT
            - (iteration * CONFIDENCE_DECAY_RATE),
        )

        avg_quality = sum(
            (g.relevance_score + g.methodology_score) / 2
            for g in grades
        ) / total

        if (
            avg_quality >= effective_threshold
            and discard_ratio <= MAX_DISCARD_RATIO
        ):
            return "sufficient"

        return "retrieve_more"

    # -----------------------------
    # Learning store (local)
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
