CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS critic_acceptance_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query context
    research_query TEXT NOT NULL,
    query_hash TEXT GENERATED ALWAYS AS (
        md5(lower(trim(research_query)))
    ) STORED,

    query_embedding VECTOR(1536) NOT NULL,

    -- Paper identity
    pmid TEXT NOT NULL,
    study_type TEXT,
    publication_year INT,

    -- Critic-derived scores (trusted)
    relevance_score FLOAT CHECK (relevance_score BETWEEN 0 AND 1),
    methodology_score FLOAT CHECK (methodology_score BETWEEN 0 AND 1),
    quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1),

    -- CRAG context
    iteration INT NOT NULL,
    accepted_at TIMESTAMP DEFAULT now()
);

-- Fast exact-match lookup
CREATE INDEX IF NOT EXISTS idx_query_hash
ON critic_acceptance_memory (query_hash);

-- Vector similarity index (cosine)
CREATE INDEX IF NOT EXISTS idx_query_embedding
ON critic_acceptance_memory
USING ivfflat (query_embedding vector_cosine_ops)
WITH (lists = 100);
