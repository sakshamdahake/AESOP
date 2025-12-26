"""
Redis-based session management for multi-turn conversations.
Uses sync Redis client to match existing sync architecture.
"""

import redis
from typing import Optional
from datetime import datetime

from app.schemas.session import SessionContext, CachedPaper
from app.logging import logger

# Configuration
SESSION_TTL_SECONDS = 60 * 60  # 60 minutes
REDIS_KEY_PREFIX = "aesop:session:"
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
        Automatically sets TTL.
        """
        try:
            context.updated_at = datetime.utcnow()
            self._client.setex(
                self._key(context.session_id),
                SESSION_TTL_SECONDS,
                context.to_redis(),
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
            deleted = self._client.delete(self._key(session_id))
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