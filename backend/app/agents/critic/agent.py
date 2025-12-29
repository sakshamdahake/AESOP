from typing import List, Dict, Any
from collections import Counter
import json

from pydantic import ValidationError

from app.logging import logger
from app.embeddings.bedrock import embed_query
from app.agents.critic.memory import CriticMemoryStore

from .schemas import PaperGrade, Recommendation
from .rubric import (
    STUDY_TYPE_PRIORS,
    MIN_AVG_QUALITY_FOR_SUFFICIENT,
    MAX_DISCARD_RATIO,
    CONFIDENCE_DECAY_RATE,
    MIN_CONFIDENCE_FLOOR,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

import psycopg2

DATABASE_URL = "postgresql://aesop:aesop_pass@postgres:5432/aesop_db"


# -----------------------------
# Utilities
# -----------------------------

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


# -----------------------------
# Critic Agent
# -----------------------------

class CriticAgent:
    """
    CRAG-enabled Critic Agent (SYNC, memory-aware).
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.memory_store = CriticMemoryStore()

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

            if parsed.get("sample_size_adequate") is None:
                parsed["sample_size_adequate"] = False

            grade = PaperGrade.model_validate(parsed)

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(
                "CriticAgent failed strict JSON or schema validation.\n"
                f"Error: {e}\n"
                f"Raw output:\n{response.content}"
            ) from e

        # Evidence hierarchy prior (soft boost, never override)
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
    # Batch grading
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
            research_question=research_question,
            grades=grades,
            iteration=iteration,
        )

        if decision == "sufficient":
            self._record_acceptance(
                grades=grades,
                iteration=iteration,
                research_query=research_question,
            )

        return {
            "grades": grades,
            "decision": decision,
        }

    # -----------------------------
    # Global CRAG decision logic (MEMORY-AWARE)
    # -----------------------------

    def _make_global_decision(
        self,
        research_question: str,
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

        # 🔑 Memory bias (bounded, safe)
        memory_boost = self.memory_store.fetch_memory_bias(
            research_question
        )

        effective_threshold = max(
            MIN_CONFIDENCE_FLOOR,
            MIN_AVG_QUALITY_FOR_SUFFICIENT
            - (iteration * CONFIDENCE_DECAY_RATE)
            - memory_boost,
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
                "memory_boost": round(memory_boost, 3),
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
    # Persistent learning store
    # -----------------------------

    def _record_acceptance(
        self,
        grades: List[PaperGrade],
        iteration: int,
        research_query: str,
    ) -> None:

        embedding = embed_query(research_query)

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        for g in grades:
            if g.recommendation == Recommendation.KEEP:
                cur.execute(
                    """
                    INSERT INTO critic_acceptance_memory (
                        research_query,
                        query_embedding,
                        pmid,
                        study_type,
                        relevance_score,
                        methodology_score,
                        quality_score,
                        iteration
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        research_query,
                        embedding,
                        g.pmid,
                        g.study_type,
                        g.relevance_score,
                        g.methodology_score,
                        (g.relevance_score + g.methodology_score) / 2,
                        iteration,
                    ),
                )

        conn.commit()
        cur.close()
        conn.close()
