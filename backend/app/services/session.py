"""
Redis-based session management for multi-turn conversations.
Uses sync Redis client to match existing sync architecture.
"""

import redis
from typing import Optional, List
from datetime import datetime

from app.schemas.session import SessionContext, CachedPaper, StructuredAnswer, SessionMessage
from app.logging import logger

# Configuration
SESSION_TTL_SECONDS = 60 * 60  # 60 minutes
REDIS_KEY_PREFIX = "aesop:session:"
REDIS_SESSION_LIST_KEY = "aesop:sessions"
REDIS_URL = "redis://redis:6379/0"


class SessionService:
    """
    Manages session context in Redis for multi-turn conversations.
    SYNC implementation to match existing codebase.
    """
    
    def __init__(self, redis_url: str = REDIS_URL):
        self._client = redis.from_url(redis_url, decode_responses=True)
    
    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{REDIS_KEY_PREFIX}{session_id}"
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Retrieve session context from Redis.
        Returns None if session doesn't exist or has expired.
        """
        try:
            data = self._client.get(self._key(session_id))
            if data is None:
                logger.debug(f"SESSION_NOT_FOUND session_id={session_id}")
                return None
            
            context = SessionContext.from_redis(data)
            logger.info(
                "SESSION_RETRIEVED",
                extra={
                    "session_id": session_id,
                    "turn_count": context.turn_count,
                    "papers_cached": len(context.retrieved_papers),
                },
            )
            return context
            
        except Exception as e:
            logger.error(
                "SESSION_GET_ERROR",
                extra={"session_id": session_id, "error": str(e)},
            )
            return None
    
    def save_session(self, context: SessionContext) -> bool:
        """
        Save or update session context in Redis.
        Automatically sets TTL and updates session list.
        """
        try:
            context.updated_at = datetime.utcnow()
            
            # Generate title if not set
            if not context.title:
                context.title = context.generate_title()
            
            self._client.setex(
                self._key(context.session_id),
                SESSION_TTL_SECONDS,
                context.to_redis(),
            )
            
            # Add to session list (sorted set with timestamp as score)
            self._client.zadd(
                REDIS_SESSION_LIST_KEY,
                {context.session_id: context.updated_at.timestamp()},
            )
            
            logger.info(
                "SESSION_SAVED",
                extra={
                    "session_id": context.session_id,
                    "turn_count": context.turn_count,
                    "papers_cached": len(context.retrieved_papers),
                },
            )
            return True
            
        except Exception as e:
            logger.error(
                "SESSION_SAVE_ERROR",
                extra={"session_id": context.session_id, "error": str(e)},
            )
            return False
    
    def create_session(
        self,
        session_id: str,
        initial_message: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionContext:
        """
        Create a new session.
        
        Args:
            session_id: UUID for the session
            initial_message: Optional first message to store
            title: Optional custom title
            
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
        )
        
        # Add initial user message if provided
        if initial_message:
            context.add_user_message(initial_message)
        
        self.save_session(context)
        return context
    
    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[dict]:
        """
        List all sessions sorted by last update time (newest first).
        
        Returns list of session summaries with id, title, updated_at.
        """
        try:
            # Get session IDs from sorted set (newest first)
            session_ids = self._client.zrevrange(
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
                    # Session expired, mark for cleanup
                    expired_ids.append(session_id)
            
            # Clean up expired sessions from list
            if expired_ids:
                self._client.zrem(REDIS_SESSION_LIST_KEY, *expired_ids)
            
            return sessions
            
        except Exception as e:
            logger.error(
                "SESSION_LIST_ERROR",
                extra={"error": str(e)},
            )
            return []
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update the title of a session."""
        context = self.get_session(session_id)
        if not context:
            return False
        
        context.title = title
        return self.save_session(context)
    
    def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        answer: Optional[StructuredAnswer] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Add a message to session history.
        
        Args:
            session_id: Session to update
            role: 'user' or 'assistant'
            content: Message content (for user messages)
            answer: Structured answer (for assistant messages)
            metadata: Optional metadata (for assistant messages)
        """
        context = self.get_session(session_id)
        if not context:
            return False
        
        if role == "user":
            context.add_user_message(content)
        else:
            context.add_assistant_message(answer, metadata)
        
        context.turn_count += 1
        return self.save_session(context)
    
    def extend_ttl(self, session_id: str) -> bool:
        """Extend TTL without modifying content (for Route C)."""
        try:
            return bool(self._client.expire(self._key(session_id), SESSION_TTL_SECONDS))
        except Exception as e:
            logger.error(
                "SESSION_TTL_EXTEND_ERROR",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Manually invalidate a session."""
        try:
            # Remove from session data
            deleted = self._client.delete(self._key(session_id))
            
            # Remove from session list
            self._client.zrem(REDIS_SESSION_LIST_KEY, session_id)
            
            logger.info(
                "SESSION_DELETED",
                extra={"session_id": session_id, "deleted": bool(deleted)},
            )
            return bool(deleted)
        except Exception as e:
            logger.error(
                "SESSION_DELETE_ERROR",
                extra={"session_id": session_id, "error": str(e)},
            )
            return False


# Module-level singleton
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get or create SessionService singleton."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
