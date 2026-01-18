"""
PostgreSQL connection pool for session persistence.
Matches schema from 002_create_session_tables.sql

Tables:
- sessions (not conversations)
- messages (with JSONB answer/metadata)
- research_contexts
- research_papers
"""

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import os
import uuid
from datetime import datetime

from app.logging import logger

# ============================================================================
# Configuration
# ============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aesop:aesop_pass@postgres:5432/aesop_db"
)

MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 10

# ============================================================================
# Connection Pool Singleton
# ============================================================================

class DatabaseService:
    """
    PostgreSQL database service with connection pooling.
    Thread-safe singleton pattern.
    
    Matches schema from 002_create_session_tables.sql
    """
    
    _instance: Optional["DatabaseService"] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Initialize connection pool. Call on app startup."""
        if self._pool is not None:
            logger.warning("DATABASE_POOL_ALREADY_INITIALIZED")
            return
        
        try:
            self._pool = pool.ThreadedConnectionPool(
                MIN_CONNECTIONS,
                MAX_CONNECTIONS,
                DATABASE_URL,
            )
            
            logger.info(
                "DATABASE_POOL_INITIALIZED",
                extra={
                    "min_connections": MIN_CONNECTIONS,
                    "max_connections": MAX_CONNECTIONS,
                }
            )
            
            # Health check
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            
            logger.info("DATABASE_POOL_HEALTH_CHECK_PASSED")
            
        except Exception as e:
            logger.error(
                "DATABASE_POOL_INIT_FAILED",
                extra={"error": str(e)}
            )
            raise
    
    def shutdown(self):
        """Close all connections. Call on app shutdown."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("DATABASE_POOL_SHUTDOWN")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Automatically commits on success, rolls back on error.
        """
        if self._pool is None:
            raise RuntimeError(
                "Database pool not initialized. Call initialize() first."
            )
        
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(
                "DATABASE_CONNECTION_ERROR",
                extra={"error": str(e)}
            )
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    # ========================================================================
    # SESSIONS CRUD (matches 002_create_session_tables.sql)
    # ========================================================================
    
    def create_session(
        self,
        session_id: str,
        anonymous_id: Optional[str] = None,
        title: Optional[str] = None,
        original_query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new session in PostgreSQL.
        
        Returns: Session record as dict
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO sessions (
                            id, anonymous_id, title, original_query, 
                            query_embedding, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id, anonymous_id, title, original_query,
                                  synthesis_summary, created_at, updated_at
                        """,
                        (session_id, anonymous_id, title, original_query, query_embedding)
                    )
                    row = cur.fetchone()
                    
                    logger.info(
                        "SESSION_CREATED_DB",
                        extra={
                            "session_id": session_id,
                            "anonymous_id": anonymous_id,
                        }
                    )
                    
                    return dict(row)
                    
                except psycopg2.errors.RaiseException as e:
                    if "Session limit exceeded" in str(e):
                        logger.warning(
                            "SESSION_LIMIT_REACHED",
                            extra={"anonymous_id": anonymous_id}
                        )
                        raise ValueError(
                            "Session limit exceeded. Maximum 100 active sessions allowed."
                        )
                    raise
    
    def get_session(
        self,
        session_id: str,
        include_messages: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get session by ID.
        Returns None if not found or soft-deleted.
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, anonymous_id, title, original_query, query_embedding,
                           turn_count, message_count, created_at, updated_at, deleted_at
                    FROM sessions
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (session_id,)
                )
                row = cur.fetchone()
                
                if not row:
                    return None
                
                session = dict(row)
                
                # Fetch messages if requested
                if include_messages:
                    cur.execute(
                        """
                        SELECT id, session_id, role, content, answer, metadata, created_at
                        FROM messages
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                        """,
                        (session_id,)
                    )
                    session["messages"] = [dict(msg) for msg in cur.fetchall()]
                
                return session
    
    def list_sessions(
        self,
        anonymous_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List sessions, optionally filtered by anonymous_id.
        Returns list of session summaries sorted by updated_at DESC.
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if anonymous_id:
                    cur.execute(
                        """
                        SELECT id, title, updated_at, created_at, message_count
                        FROM sessions
                        WHERE anonymous_id = %s AND deleted_at IS NULL
                        ORDER BY updated_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (anonymous_id, limit, offset)
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, title, updated_at, created_at, message_count
                        FROM sessions
                        WHERE deleted_at IS NULL
                        ORDER BY updated_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (limit, offset)
                    )
                
                return [dict(row) for row in cur.fetchall()]
    
    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        original_query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update session fields.
        Returns True if updated, False if not found.
        """
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = %s")
            params.append(title)
        
        if original_query is not None:
            updates.append("original_query = %s")
            params.append(original_query)
        
        if query_embedding is not None:
            updates.append("query_embedding = %s")
            params.append(query_embedding)
        
        if not updates:
            return False
        
        updates.append("updated_at = NOW()")
        params.append(session_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    UPDATE sessions
                    SET {', '.join(updates)}
                    WHERE id = %s AND deleted_at IS NULL
                """
                cur.execute(query, params)
                return cur.rowcount > 0
    
    def delete_session(self, session_id: str) -> bool:
        """
        Soft delete session.
        Returns True if deleted, False if not found.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions
                    SET deleted_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (session_id,)
                )
                deleted = cur.rowcount > 0
                
                if deleted:
                    logger.info(
                        "SESSION_SOFT_DELETED",
                        extra={"session_id": session_id}
                    )
                
                return deleted
    
    # ========================================================================
    # MESSAGES CRUD (with JSONB answer and metadata)
    # ========================================================================
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        answer: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to a session.
        
        Args:
            session_id: Session ID
            role: 'user' or 'assistant'
            content: User message content (for user messages)
            answer: Structured answer dict (for assistant messages)
            metadata: Metadata dict (for assistant messages)
        
        Returns: Message record as dict
        """
        message_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO messages (
                            id, session_id, role, content, answer, metadata, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        RETURNING id, session_id, role, content, answer, metadata, created_at
                        """,
                        (
                            message_id,
                            session_id,
                            role,
                            content,
                            Json(answer) if answer else None,
                            Json(metadata) if metadata else None,
                        )
                    )
                    row = cur.fetchone()
                    
                    logger.debug(
                        "MESSAGE_ADDED_DB",
                        extra={
                            "session_id": session_id,
                            "role": role,
                            "message_id": message_id,
                        }
                    )
                    
                    return dict(row)
                    
                except psycopg2.errors.RaiseException as e:
                    if "Message limit exceeded" in str(e):
                        logger.warning(
                            "MESSAGE_LIMIT_REACHED",
                            extra={"session_id": session_id}
                        )
                        raise ValueError(
                            "Message limit exceeded. Maximum 500 messages per session."
                        )
                    raise
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a session, ordered chronologically.
        
        Args:
            session_id: Session ID
            limit: Optional limit (e.g., last N messages)
        
        Returns: List of message dicts
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if limit:
                    # Get last N messages
                    cur.execute(
                        """
                        SELECT id, session_id, role, content, answer, metadata, created_at
                        FROM messages
                        WHERE session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (session_id, limit)
                    )
                    # Reverse to get chronological order
                    return [dict(row) for row in reversed(cur.fetchall())]
                else:
                    # Get all messages
                    cur.execute(
                        """
                        SELECT id, session_id, role, content, answer, metadata, created_at
                        FROM messages
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                        """,
                        (session_id,)
                    )
                    return [dict(row) for row in cur.fetchall()]
    
    # ========================================================================
    # RESEARCH CONTEXTS CRUD
    # ========================================================================
    
    def create_research_context(
        self,
        session_id: str,
        research_query: str,
        query_embedding: Optional[List[float]] = None,
        synthesis_summary: Optional[str] = None,
        route_taken: Optional[str] = None,
        intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        papers_count: int = 0,
        critic_decision: Optional[str] = None,
        avg_quality: Optional[float] = None,
    ) -> str:
        """
        Create a research context.
        Returns: research_context_id
        """
        context_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_contexts (
                        id, session_id, research_query, query_embedding,
                        synthesis_summary, route_taken, intent, intent_confidence,
                        papers_count, critic_decision, avg_quality, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (
                        context_id, session_id, research_query, query_embedding,
                        synthesis_summary, route_taken, intent, intent_confidence,
                        papers_count, critic_decision, avg_quality
                    )
                )
                
                logger.info(
                    "RESEARCH_CONTEXT_CREATED_DB",
                    extra={
                        "context_id": context_id,
                        "session_id": session_id,
                        "route_taken": route_taken,
                    }
                )
                
                return context_id
    
    def add_research_papers(
        self,
        research_context_id: str,
        papers: List[Dict[str, Any]],
    ) -> int:
        """
        Add papers to a research context.
        Returns: Number of papers added
        """
        if not papers:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for paper in papers:
                    paper_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO research_papers (
                            id, research_context_id, pmid, title, abstract,
                            publication_year, journal, relevance_score,
                            methodology_score, quality_score, recommendation,
                            study_type, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (research_context_id, pmid) DO NOTHING
                        """,
                        (
                            paper_id,
                            research_context_id,
                            paper.get("pmid"),
                            paper.get("title"),
                            paper.get("abstract"),
                            paper.get("publication_year"),
                            paper.get("journal"),
                            paper.get("relevance_score"),
                            paper.get("methodology_score"),
                            paper.get("quality_score"),
                            paper.get("recommendation"),
                            paper.get("study_type"),
                        )
                    )
                
                logger.debug(
                    "RESEARCH_PAPERS_ADDED_DB",
                    extra={
                        "research_context_id": research_context_id,
                        "count": len(papers),
                    }
                )
                
                return len(papers)


# ============================================================================
# Module-level singleton
# ============================================================================

_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """Get or create DatabaseService singleton."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service