"""
Hybrid Session Service - Redis (cache) + PostgreSQL (persistence).

Architecture:
- Redis: Fast cache for active sessions (60min TTL)
- PostgreSQL: Permanent storage (forever)
- Write-through: Every change goes to both

Strategy:
- get_session(): Try Redis → If miss, load from DB → Cache in Redis
- save_session(): Write to Redis + Write to PostgreSQL
- create_session(): Write to both immediately
- delete_session(): Soft delete in DB + Remove from Redis

backend/app/services/session.py
"""

import redis
from typing import Optional, List, Dict, Any
from datetime import datetime
import traceback

from app.schemas.session import (
    SessionContext,
    SessionMessage,
    CachedPaper,
    StructuredAnswer,
)
from app.services.database import get_database_service
from app.embeddings.bedrock import embed_query
from app.logging import logger

# Configuration
SESSION_TTL_SECONDS = 60 * 60  # 60 minutes
REDIS_KEY_PREFIX = "aesop:session:"
REDIS_SESSION_LIST_KEY = "aesop:sessions"
REDIS_URL = "redis://redis:6379/0"


class SessionService:
    """
    Hybrid session management: Redis (cache) + PostgreSQL (persistence).
    
    SYNC implementation to match existing codebase.
    """
    
    def __init__(self, redis_url: str = REDIS_URL):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._db = get_database_service()
    
    def _redis_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{REDIS_KEY_PREFIX}{session_id}"
    
    # ========================================================================
    # GET SESSION (with cache-aside pattern)
    # ========================================================================
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Get session with cache-aside pattern.
        
        Flow:
        1. Try Redis (fast)
        2. If miss, try PostgreSQL
        3. If found in DB, cache in Redis
        4. Return session or None
        """
        # 1. Try Redis first (fast path)
        try:
            redis_data = self._redis.get(self._redis_key(session_id))
            if redis_data:
                context = SessionContext.from_redis(redis_data)
                logger.debug(
                    "SESSION_CACHE_HIT",
                    extra={"session_id": session_id}
                )
                return context
        except Exception as e:
            logger.warning(
                "SESSION_REDIS_GET_ERROR",
                extra={"session_id": session_id, "error": str(e)}
            )
            # Continue to DB lookup
        
        # 2. Redis miss - try PostgreSQL
        try:
            db_session = self._db.get_session(session_id, include_messages=True)
            if not db_session:
                logger.debug(
                    "SESSION_NOT_FOUND",
                    extra={"session_id": session_id}
                )
                return None
            
            # 3. Convert DB record to SessionContext
            context = self._db_record_to_session_context(db_session)
            
            # 4. Cache in Redis for future requests
            try:
                self._redis.setex(
                    self._redis_key(session_id),
                    SESSION_TTL_SECONDS,
                    context.to_redis(),
                )
                logger.debug(
                    "SESSION_CACHED_FROM_DB",
                    extra={"session_id": session_id}
                )
            except Exception as e:
                logger.warning(
                    "SESSION_REDIS_CACHE_ERROR",
                    extra={"session_id": session_id, "error": str(e)}
                )
            
            logger.info(
                "SESSION_LOADED_FROM_DB",
                extra={
                    "session_id": session_id,
                    "message_count": len(context.messages),
                }
            )
            
            return context
            
        except Exception as e:
            logger.error(
                "SESSION_DB_GET_ERROR",
                extra={"session_id": session_id, "error": str(e)}
            )
            return None
    
    # ========================================================================
    # SAVE SESSION (write-through to both Redis and PostgreSQL)
    # ========================================================================
    
    def save_session(self, context: SessionContext) -> bool:
        """
        Save session with write-through to both Redis and PostgreSQL.
        
        Flow:
        1. Write to Redis (cache)
        2. Write to PostgreSQL (persistence)
        3. If this is first save (not persisted), create DB record
        4. Otherwise, update DB record
        
        FIXED: Better handling of existing sessions
        """
        context.updated_at = datetime.utcnow()
        
        # Generate title if not set
        if not context.title:
            context.title = context.generate_title()
        
        success = True
        
        # 1. Write to Redis (cache layer)
        try:
            self._redis.setex(
                self._redis_key(context.session_id),
                SESSION_TTL_SECONDS,
                context.to_redis(),
            )
            
            # Update session list (sorted set)
            self._redis.zadd(
                REDIS_SESSION_LIST_KEY,
                {context.session_id: context.updated_at.timestamp()},
            )
            
            logger.debug(
                "SESSION_SAVED_REDIS",
                extra={"session_id": context.session_id}
            )
        except Exception as e:
            logger.error(
                "SESSION_REDIS_SAVE_ERROR",
                extra={"session_id": context.session_id, "error": str(e)}
            )
            success = False
        
        # 2. Write to PostgreSQL (persistence layer)
        try:
            # ✅ FIXED: Check if session exists in DB using direct query
            with self._db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM sessions WHERE id = %s AND deleted_at IS NULL",
                        (context.session_id,)
                    )
                    session_exists = cur.fetchone() is not None
            
            logger.debug(
                "SESSION_DB_CHECK",
                extra={
                    "session_id": context.session_id,
                    "exists": session_exists,
                    "message_count": len(context.messages),
                }
            )
            
            if not session_exists:
                # Session doesn't exist - create it
                self._create_session_in_db(context)
                logger.info(
                    "SESSION_CREATED_IN_DB",
                    extra={
                        "session_id": context.session_id,
                        "message_count": len(context.messages),
                    }
                )
            else:
                # Session exists - update it
                self._update_session_in_db(context)
                logger.debug(
                    "SESSION_UPDATED_IN_DB",
                    extra={
                        "session_id": context.session_id,
                        "message_count": len(context.messages),
                    }
                )
            
        except Exception as e:
            logger.error(
                "SESSION_DB_SAVE_ERROR",
                extra={
                    "session_id": context.session_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            success = False

    
    # ========================================================================
    # CREATE SESSION (write to both immediately)
    # ========================================================================
    
    def create_session(
        self,
        session_id: str,
        initial_message: Optional[str] = None,
        title: Optional[str] = None,
        anonymous_id: Optional[str] = None,
    ) -> SessionContext:
        """
        Create a new session in both Redis and PostgreSQL.
        
        Args:
            session_id: UUID for the session
            initial_message: Optional first message
            title: Optional custom title
            anonymous_id: Optional user identifier
        
        Returns:
            New SessionContext
        """
        now = datetime.utcnow()
        
        context = SessionContext(
            session_id=session_id,
            original_query=initial_message or "",
            title=title or (initial_message[:47] + "..." if initial_message and len(initial_message) > 50 else initial_message),
            messages=[],
            turn_count=0,
            created_at=now,
            updated_at=now,
            anonymous_id=anonymous_id,
            persisted=False,  # Will be set to True after DB write
        )
        
        # Add initial user message if provided
        if initial_message:
            context.add_user_message(initial_message)
        
        # Save to both Redis and PostgreSQL
        self.save_session(context)
        
        logger.info(
            "SESSION_CREATED",
            extra={
                "session_id": session_id,
                "anonymous_id": anonymous_id,
            }
        )
        
        return context
    
    # ========================================================================
    # LIST SESSIONS (from Redis with DB fallback)
    # ========================================================================
    
    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        anonymous_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List sessions sorted by last update time (newest first).
        
        Uses Redis session list, falls back to DB for expired sessions.
        
        Args:
            limit: Maximum sessions to return
            offset: Offset for pagination
            anonymous_id: Optional filter by user
        
        Returns:
            List of session summaries
        """
        # For anonymous_id filtering, must use database
        if anonymous_id:
            try:
                db_sessions = self._db.list_sessions(
                    anonymous_id=anonymous_id,
                    limit=limit,
                    offset=offset,
                )
                return [
                    {
                        "session_id": s["id"],
                        "title": s["title"],
                        "updated_at": s["updated_at"],
                        "message_count": s["message_count"],
                    }
                    for s in db_sessions
                ]
            except Exception as e:
                logger.error(
                    "SESSION_LIST_DB_ERROR",
                    extra={"error": str(e)}
                )
                return []
        
        # Otherwise, use Redis session list (faster)
        try:
            # Get session IDs from sorted set (newest first)
            session_ids = self._redis.zrevrange(
                REDIS_SESSION_LIST_KEY,
                offset,
                offset + limit - 1,
            )
            
            sessions = []
            expired_ids = []
            
            for session_id in session_ids:
                context = self.get_session(session_id)
                if context:
                    sessions.append({
                        "session_id": context.session_id,
                        "title": context.title or context.generate_title(),
                        "updated_at": context.updated_at,
                        "message_count": len(context.messages),
                    })
                else:
                    # Session expired from Redis, mark for cleanup
                    expired_ids.append(session_id)
            
            # Clean up expired sessions from Redis list
            if expired_ids:
                self._redis.zrem(REDIS_SESSION_LIST_KEY, *expired_ids)
            
            return sessions
            
        except Exception as e:
            logger.error(
                "SESSION_LIST_REDIS_ERROR",
                extra={"error": str(e)}
            )
            return []
    
    # ========================================================================
    # DELETE SESSION (soft delete in DB, remove from Redis)
    # ========================================================================
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session (soft delete in DB, remove from Redis).
        
        Args:
            session_id: Session to delete
        
        Returns:
            True if deleted, False if not found
        """
        success = False
        
        # 1. Soft delete in PostgreSQL
        try:
            if self._db.delete_session(session_id):
                success = True
                logger.info(
                    "SESSION_DELETED_DB",
                    extra={"session_id": session_id}
                )
        except Exception as e:
            logger.error(
                "SESSION_DELETE_DB_ERROR",
                extra={"session_id": session_id, "error": str(e)}
            )
        
        # 2. Remove from Redis
        try:
            self._redis.delete(self._redis_key(session_id))
            self._redis.zrem(REDIS_SESSION_LIST_KEY, session_id)
            logger.debug(
                "SESSION_DELETED_REDIS",
                extra={"session_id": session_id}
            )
        except Exception as e:
            logger.warning(
                "SESSION_DELETE_REDIS_ERROR",
                extra={"session_id": session_id, "error": str(e)}
            )
        
        return success
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def extend_ttl(self, session_id: str) -> bool:
        """Extend Redis TTL without modifying content."""
        try:
            return bool(self._redis.expire(
                self._redis_key(session_id),
                SESSION_TTL_SECONDS
            ))
        except Exception as e:
            logger.error(
                "SESSION_TTL_EXTEND_ERROR",
                extra={"session_id": session_id, "error": str(e)}
            )
            return False
    
    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================
    
    def _create_session_in_db(self, context: SessionContext):
        """Create session in PostgreSQL (internal)."""
        print(f"🔍 DEBUG: _create_session_in_db called for {context.session_id}")
        
        try:
            # Create session record
            result = self._db.create_session(
                session_id=context.session_id,
                anonymous_id=context.anonymous_id,
                title=context.title,
                original_query=context.original_query,
                query_embedding=context.query_embedding if context.query_embedding else None,
            )
            print(f"✅ DEBUG: Session created in DB: {result}")
            
            # Add messages
            for i, msg in enumerate(context.messages):
                print(f"🔍 DEBUG: Adding message {i+1}/{len(context.messages)}")
                self._db.add_message(
                    session_id=context.session_id,
                    role=msg.role,
                    content=msg.content,
                    answer=msg.answer.to_dict() if msg.answer else None,
                    metadata=msg.metadata,
                )
            print(f"✅ DEBUG: {len(context.messages)} messages added")
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in _create_session_in_db: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _update_session_in_db(self, context: SessionContext):
        """Update session in PostgreSQL (internal)."""
        print(f"🔍 DEBUG: _update_session_in_db called for {context.session_id}")
        
        # Update session metadata
        self._db.update_session(
            session_id=context.session_id,
            title=context.title,
            original_query=context.original_query,
            query_embedding=context.query_embedding if context.query_embedding else None,
        )
        
        # Get existing messages from DB
        existing_session = self._db.get_session(context.session_id, include_messages=True)
        if existing_session and existing_session.get("messages"):
            existing_msg_count = len(existing_session["messages"])
        else:
            existing_msg_count = 0
        
        print(f"🔍 DEBUG: Existing messages in DB: {existing_msg_count}, Current messages: {len(context.messages)}")
        
        # Add any new messages (only messages after existing_count)
        new_messages = context.messages[existing_msg_count:]
        for i, msg in enumerate(new_messages):
            print(f"🔍 DEBUG: Adding new message {i+1}/{len(new_messages)} role={msg.role}")
            self._db.add_message(
                session_id=context.session_id,
                role=msg.role,
                content=msg.content,
                answer=msg.answer.to_dict() if msg.answer else None,
                metadata=msg.metadata,
            )
        
        print(f"✅ DEBUG: {len(new_messages)} new messages added to DB")

    
    def _db_record_to_session_context(
        self,
        db_session: Dict[str, Any]
    ) -> SessionContext:
        """Convert database record to SessionContext (internal)."""
        # Convert messages
        messages = []
        for db_msg in db_session.get("messages", []):
            messages.append(SessionMessage.from_db_dict(db_msg))
        
        # Compute turn_count from messages (count user messages)
        turn_count = sum(1 for msg in messages if msg.role == "user")
        
        # Create SessionContext
        return SessionContext(
            session_id=db_session["id"],
            anonymous_id=db_session.get("anonymous_id"),
            title=db_session.get("title"),
            original_query=db_session.get("original_query", ""),
            query_embedding=db_session.get("query_embedding", []),
            messages=messages,
            turn_count=turn_count,  # FIXED: Computed from messages
            created_at=db_session["created_at"],
            updated_at=db_session["updated_at"],
            deleted_at=db_session.get("deleted_at"),
            persisted=True,  # Came from DB, so it's persisted
        )


# ============================================================================
# Module-level singleton
# ============================================================================

_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get or create SessionService singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
