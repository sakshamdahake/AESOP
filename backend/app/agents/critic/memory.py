import psycopg2
import math
from datetime import datetime, timezone

from app.embeddings.bedrock import embed_query

DATABASE_URL = "postgresql://aesop:aesop_pass@postgres:5432/aesop_db"


class CriticMemoryStore:
    MAX_BOOST = 0.15
    SIMILARITY_THRESHOLD = 0.75
    DECAY_LAMBDA = 0.01

    def fetch_memory_bias(self, query: str) -> float:
        query_hash = self._hash(query)

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # 1️⃣ Exact-match fast path
        cur.execute(
            """
            SELECT quality_score, accepted_at, 1.0 AS similarity
            FROM critic_acceptance_memory
            WHERE query_hash = %s
            ORDER BY accepted_at DESC
            LIMIT 10
            """,
            (query_hash,),
        )
        rows = cur.fetchall()

        # 2️⃣ Fallback to vector search
        if not rows:
            embedding = embed_query(query)
            cur.execute(
                """
                SELECT
                    quality_score,
                    accepted_at,
                    1 - (query_embedding <=> %s::vector) AS similarity
                FROM critic_acceptance_memory
                WHERE (1 - (query_embedding <=> %s::vector)) >= %s
                ORDER BY similarity DESC
                LIMIT 10
                """,
                (embedding, embedding, self.SIMILARITY_THRESHOLD),
            )

            rows = cur.fetchall()

        cur.close()
        conn.close()

        if not rows:
            return 0.0

        now = datetime.now(timezone.utc)
        weighted = []

        for quality, accepted_at, similarity in rows:
            if accepted_at.tzinfo is None:
                accepted_at = accepted_at.replace(tzinfo=timezone.utc)
            age_days = (now - accepted_at).days
            recency = math.exp(-self.DECAY_LAMBDA * age_days)
            quality = float(quality)
            similarity = float(similarity)
            recency = float(recency)

            weighted.append(quality * similarity * recency)

        score = sum(weighted) / len(weighted)
        return min(score, self.MAX_BOOST)

    @staticmethod
    def _hash(query: str) -> str:
        import hashlib
        return hashlib.md5(query.strip().lower().encode()).hexdigest()
