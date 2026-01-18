-- ============================================================================
-- AESOP Session Persistence Schema - CLEAN INSTALL
-- Migration: 002_clean_and_create_session_tables.sql
-- 
-- WARNING: This will DROP existing tables. Only run if you don't need old data.
-- ============================================================================

-- ============================================================================
-- 0. CLEANUP - Drop existing tables and functions
-- ============================================================================

-- Drop triggers first (they depend on functions)
DROP TRIGGER IF EXISTS trigger_check_conversation_limit ON conversations;
DROP TRIGGER IF EXISTS trigger_check_message_limit ON messages;
DROP TRIGGER IF EXISTS trigger_update_conversation_timestamp ON messages;
DROP TRIGGER IF EXISTS trigger_check_session_limit ON sessions;
DROP TRIGGER IF EXISTS trigger_update_session_timestamp ON messages;

-- Drop tables in correct order (children first due to foreign keys)
DROP TABLE IF EXISTS research_papers CASCADE;
DROP TABLE IF EXISTS research_contexts CASCADE;
DROP TABLE IF EXISTS session_papers CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS check_conversation_limit();
DROP FUNCTION IF EXISTS check_session_limit();
DROP FUNCTION IF EXISTS check_message_limit();
DROP FUNCTION IF EXISTS update_conversation_timestamp();
DROP FUNCTION IF EXISTS update_session_timestamp();
DROP FUNCTION IF EXISTS get_next_sequence_num(UUID);
DROP FUNCTION IF EXISTS get_next_message_sequence(UUID);

-- Ensure vector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. SESSIONS TABLE
-- Stores session metadata and links to anonymous users
-- ============================================================================

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- User identification (optional, for linking sessions)
    anonymous_id VARCHAR(255) NULL,
    
    -- Session metadata
    title VARCHAR(255) NULL,
    original_query TEXT NULL,
    
    -- Query embedding for semantic search
    query_embedding VECTOR(1536) NULL,
    
    -- Synthesis summary (cached for quick access)
    synthesis_summary TEXT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Soft delete
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Index for listing user's sessions (sorted by recent)
CREATE INDEX idx_sessions_anonymous_id_updated 
ON sessions (anonymous_id, updated_at DESC) 
WHERE deleted_at IS NULL;

-- Index for fetching single session
CREATE INDEX idx_sessions_id_not_deleted
ON sessions (id) 
WHERE deleted_at IS NULL;

-- ============================================================================
-- 2. MESSAGES TABLE
-- Stores all messages in a session with ordering
-- ============================================================================

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Parent session
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Message content
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NULL,  -- For user messages (plain text)
    
    -- Structured answer for assistant messages (JSON)
    structured_answer JSONB NULL,
    
    -- Message metadata (intent, route, etc.)
    metadata JSONB NULL,
    
    -- Ordering (monotonically increasing per session)
    sequence_num INTEGER NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Ensure unique ordering within session
    CONSTRAINT unique_session_sequence UNIQUE (session_id, sequence_num)
);

-- Index for fetching messages in order
CREATE INDEX idx_messages_session_sequence 
ON messages (session_id, sequence_num DESC);

-- Index for counting messages per session
CREATE INDEX idx_messages_session_id 
ON messages (session_id);

-- ============================================================================
-- 3. SESSION PAPERS TABLE (cached papers per session)
-- ============================================================================

CREATE TABLE session_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Parent session
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Paper identity
    pmid VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NULL,
    journal VARCHAR(500) NULL,
    publication_year INTEGER NULL,
    
    -- Scores
    relevance_score FLOAT NULL,
    methodology_score FLOAT NULL,
    quality_score FLOAT NULL,
    recommendation VARCHAR(20) NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Unique constraint per session
    CONSTRAINT unique_session_paper UNIQUE (session_id, pmid)
);

-- Index for fetching papers by session
CREATE INDEX idx_session_papers_session 
ON session_papers (session_id);

-- ============================================================================
-- 4. RESEARCH CONTEXTS TABLE
-- Stores research query state for each research interaction
-- ============================================================================

CREATE TABLE research_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Parent session
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Research query
    query TEXT NOT NULL,
    query_embedding VECTOR(1536) NULL,
    
    -- Synthesis output
    synthesis_summary TEXT NULL,
    
    -- Routing and metrics
    route_taken VARCHAR(50) NULL,
    intent VARCHAR(50) NULL,
    papers_count INTEGER NOT NULL DEFAULT 0,
    
    -- Quality metrics from Critic
    critic_decision VARCHAR(50) NULL,
    avg_quality FLOAT NULL,
    discard_ratio FLOAT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for finding research contexts by session
CREATE INDEX idx_research_contexts_session 
ON research_contexts (session_id, created_at DESC);

-- ============================================================================
-- 5. RESEARCH PAPERS TABLE
-- Stores papers retrieved for each research context
-- ============================================================================

CREATE TABLE research_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Parent research context
    research_context_id UUID NOT NULL REFERENCES research_contexts(id) ON DELETE CASCADE,
    
    -- Paper identity
    pmid VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NULL,
    journal VARCHAR(500) NULL,
    publication_year INTEGER NULL,
    
    -- Critic grades
    relevance_score FLOAT NULL CHECK (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1)),
    methodology_score FLOAT NULL CHECK (methodology_score IS NULL OR (methodology_score >= 0 AND methodology_score <= 1)),
    quality_score FLOAT NULL CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)),
    recommendation VARCHAR(20) NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for fetching papers by research context
CREATE INDEX idx_research_papers_context 
ON research_papers (research_context_id);

-- Index for finding papers by PMID
CREATE INDEX idx_research_papers_pmid 
ON research_papers (pmid);

-- ============================================================================
-- 6. RATE LIMITING FUNCTIONS
-- ============================================================================

-- Function to check session limit per anonymous_id
CREATE OR REPLACE FUNCTION check_session_limit()
RETURNS TRIGGER AS $$
DECLARE
    session_count INTEGER;
    max_sessions INTEGER := 100;
BEGIN
    IF NEW.anonymous_id IS NULL THEN
        RETURN NEW;
    END IF;
    
    SELECT COUNT(*) INTO session_count
    FROM sessions
    WHERE anonymous_id = NEW.anonymous_id
      AND deleted_at IS NULL;
    
    IF session_count >= max_sessions THEN
        RAISE EXCEPTION 'Session limit reached. Maximum % sessions per user.', max_sessions;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_check_session_limit
    BEFORE INSERT ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION check_session_limit();

-- Function to check message limit per session
CREATE OR REPLACE FUNCTION check_message_limit()
RETURNS TRIGGER AS $$
DECLARE
    message_count INTEGER;
    max_messages INTEGER := 500;
BEGIN
    SELECT COUNT(*) INTO message_count
    FROM messages
    WHERE session_id = NEW.session_id;
    
    IF message_count >= max_messages THEN
        RAISE EXCEPTION 'Message limit reached. Maximum % messages per session.', max_messages;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_check_message_limit
    BEFORE INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION check_message_limit();

-- ============================================================================
-- 7. AUTO-UPDATE TIMESTAMP FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION update_session_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE sessions 
    SET updated_at = NOW() 
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_session_timestamp
    AFTER INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_session_timestamp();

-- ============================================================================
-- 8. HELPER FUNCTION: Get next sequence number
-- ============================================================================

CREATE OR REPLACE FUNCTION get_next_message_sequence(sess_id UUID)
RETURNS INTEGER AS $$
DECLARE
    next_num INTEGER;
BEGIN
    SELECT COALESCE(MAX(sequence_num), 0) + 1 INTO next_num
    FROM messages
    WHERE session_id = sess_id;
    
    RETURN next_num;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 9. VERIFY INSTALLATION
-- ============================================================================

-- List all created tables
DO $$
BEGIN
    RAISE NOTICE 'Migration completed successfully!';
    RAISE NOTICE 'Tables created: sessions, messages, session_papers, research_contexts, research_papers';
END $$;