from typing import List, Dict, Any

from pydantic import ValidationError

from .schemas import PaperGrade, Recommendation
from .rubric import (
    MIN_AVG_QUALITY_FOR_SUFFICIENT,
    MAX_DISCARD_RATIO,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class CriticAgent:
    """
    Evaluates retrieved abstracts and determines whether
    the evidence base is sufficient or more retrieval is required.
    """

    def __init__(self, llm_client):
        """
        llm_client must expose an async `invoke(messages)` method
        compatible with LangChain / OpenAI / Azure / vLLM.
        """
        self.llm = llm_client

    async def grade_abstract(
        self, research_question: str, abstract: str
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

        response = await self.llm.ainvoke(messages)

        try:
            return PaperGrade.model_validate_json(response.content)
        except ValidationError as e:
            raise RuntimeError(
                f"CriticAgent schema validation failed: {e}"
            ) from e

    async def grade_batch(
        self, research_question: str, abstracts: List[str]
    ) -> Dict[str, Any]:
        grades: List[PaperGrade] = []

        for abstract in abstracts:
            grade = await self.grade_abstract(research_question, abstract)
            grades.append(grade)

        decision = self._make_global_decision(grades)

        return {
            "grades": grades,
            "decision": decision,
        }

    def _make_global_decision(self, grades: List[PaperGrade]) -> str:
        if not grades:
            return "retrieve_more"

        discard_count = sum(
            1 for g in grades if g.recommendation == Recommendation.DISCARD
        )

        avg_quality = sum(
            (g.relevance_score + g.methodology_score) / 2
            for g in grades
        ) / len(grades)

        discard_ratio = discard_count / len(grades)

        if (
            avg_quality >= MIN_AVG_QUALITY_FOR_SUFFICIENT
            and discard_ratio <= MAX_DISCARD_RATIO
        ):
            return "sufficient"

        return "retrieve_more"
