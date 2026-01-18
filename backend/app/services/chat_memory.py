"""
Hybrid Chat Memory Service: Redis Cache + PostgreSQL Persistence.

This service provides:
- Fast reads from Redis cache (last 5 message pairs)
- Persistent storage in PostgreSQL (all messages forever)
- Write-through caching (every message goes to both)
- Automatic cache population from database on miss
- LLM-ready message formatting with conversation context
"""

import redis
from typing import Optional, List, Tuple
from datetime import datetime
import uuid

from app.schemas.conversation import (
    CachedConversation,
    MessageRecord,
    ConversationRecord,
)
from app.services.database import get_database_service, DatabaseService
from app.logging import logger


# ============================================================================
# Configuration
# ============================================================================

REDIS_URL = "redis://redis:6379/0"
CACHE_KEY_PREFIX = "aesop:conv:"
CACHE_TTL_SECONDS = 60 * 60  # 60 minutes
MAX_CACHED_MESSAGES = 10  # 5 pairs (user + assistant)


# ============================================================================
# Chat Memory Service
# ============================================================================

class ChatMemoryService:
    """
    Hybrid chat memory with Redis cache and PostgreSQL persistence.
    
    Read Strategy (Cache-First):
        1. Check Redis cache
        2. If miss, load from PostgreSQL
        3. Populate cache from database
        4. Return data
    
    Write Strategy (Write-Through):
        1. Write to Redis cache (sync, fast)
        2. Write to PostgreSQL (sync, durable)
        3. Return success
    """
    
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        db_service: Optional[DatabaseService] = None,
    ):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._db = db_service or get_database_service()
    
    def _cache_key(self, conversation_id: str) -> str:
        """Generate Redis key for conversation cache."""
        return f"{CACHE_KEY_PREFIX}{conversation_id}"
    
    # ========================================================================
    # CACHE OPERATIONS
    # ========================================================================
    
    def _get_from_cache(self, conversation_id: str) -> Optional[CachedConversation]:
        """Get conversation from Redis cache."""
        try:
            data = self._redis.get(self._cache_key(conversation_id))
            if data:
                return CachedConversation.from_redis(data)
            return None
        except Exception as e:
            logger.warning("CACHE_GET_ERROR", extra={
                "conversation_id": conversation_id,
                "error": str(e),
            })
            return None
    
    def _save_to_cache(self, cached: CachedConversation) -> bool:
        """Save conversation to Redis cache with TTL."""
        try:
            cached.cached_at = datetime.utcnow()
            self._redis.setex(
                self._cache_key(cached.conversation_id),
                CACHE_TTL_SECONDS,
                cached.to_redis(),
            )
            return True
        except Exception as e:
            logger.warning("CACHE_SAVE_ERROR", extra={
                "conversation_id": cached.conversation_id,
                "error": str(e),
            })
            return False
    
    def _invalidate_cache(self, conversation_id: str) -> bool:
        """Remove conversation from Redis cache."""
        try:
            self._redis.delete(self._cache_key(conversation_id))
            return True
        except Exception as e:
            logger.warning("CACHE_INVALIDATE_ERROR", extra={
                "conversation_id": conversation_id,
                "error": str(e),
            })
            return False
    
    def _extend_cache_ttl(self, conversation_id: str) -> bool:
        """Extend TTL on cache entry."""
        try:
            return bool(self._redis.expire(
                self._cache_key(conversation_id),
                CACHE_TTL_SECONDS,
            ))
        except Exception as e:
            logger.warning("CACHE_TTL_EXTEND_ERROR", extra={
                "conversation_id": conversation_id,
                "error": str(e),
            })
            return False
    
    # ========================================================================
    # DATABASE → CACHE POPULATION
    # ========================================================================
    
    def _load_and_cache_from_database(
        self,
        conversation_id: str,
    ) -> Optional[CachedConversation]:
        """
        Load conversation from PostgreSQL and populate Redis cache.
        Used on cache miss.
        """
        # Get conversation from database
        conversation = self._db.get_conversation(
            conversation_id,
            include_messages=True,
            include_research=False,
        )
        
        if not conversation:
            return None
        
        # Get latest research context for synthesis summary
        latest_research = self._db.get_latest_research_context(conversation_id)
        
        # Build cache object with recent messages only
        recent_messages = conversation.messages[-MAX_CACHED_MESSAGES:]
        
        cached = CachedConversation(
            conversation_id=conversation.id,
            anonymous_id=conversation.anonymous_id,
            title=conversation.title,
            title_generated=conversation.title_generated,
            recent_messages=recent_messages,
            last_research_context_id=latest_research.id if latest_research else None,
            last_synthesis_summary=latest_research.synthesis_summary if latest_research else None,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
        )
        
        # Save to cache
        self._save_to_cache(cached)
        
        logger.info("CACHE_POPULATED_FROM_DB", extra={
            "conversation_id": conversation_id,
            "message_count": cached.message_count,
            "cached_messages": len(recent_messages),
        })
        
        return cached
    
    # ========================================================================
    # PUBLIC API: Conversation Management
    # ========================================================================
    
    def get_or_create_conversation(
        self,
        conversation_id: Optional[str] = None,
        anonymous_id: Optional[str] = None,
    ) -> Tuple[CachedConversation, bool]:
        """
        Get existing conversation or create new one.
        
        Args:
            conversation_id: Existing conversation ID (optional)
            anonymous_id: User identifier for linking (optional)
        
        Returns:
            Tuple of (CachedConversation, is_new)
        """
        # If conversation_id provided, try to get existing
        if conversation_id:
            cached = self.get_conversation(conversation_id)
            if cached:
                return cached, False
            # Conversation not found - could be deleted or invalid ID
            logger.warning("CONVERSATION_NOT_FOUND", extra={
                "conversation_id": conversation_id,
            })
        
        # Create new conversation
        conversation = self._db.create_conversation(
            anonymous_id=anonymous_id,
            title="New Conversation",
        )
        
        # Create cache entry
        cached = CachedConversation(
            conversation_id=conversation.id,
            anonymous_id=conversation.anonymous_id,
            title=conversation.title,
            title_generated=False,
            recent_messages=[],
            message_count=0,
            created_at=conversation.created_at,
        )
        
        self._save_to_cache(cached)
        
        logger.info("CONVERSATION_CREATED", extra={
            "conversation_id": conversation.id,
            "anonymous_id": anonymous_id,
        })
        
        return cached, True
    
    def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[CachedConversation]:
        """
        Get conversation with cache-first strategy.
        
        1. Check Redis cache
        2. On miss, load from PostgreSQL and cache
        3. Return None if not found or deleted
        """
        # Try cache first
        cached = self._get_from_cache(conversation_id)
        if cached:
            logger.debug("CACHE_HIT", extra={"conversation_id": conversation_id})
            self._extend_cache_ttl(conversation_id)
            return cached
        
        # Cache miss - load from database
        logger.debug("CACHE_MISS", extra={"conversation_id": conversation_id})
        return self._load_and_cache_from_database(conversation_id)
    
    def add_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        response_type: str = "chat",
    ) -> Tuple[MessageRecord, MessageRecord]:
        """
        Add a user-assistant turn to the conversation.
        
        Write-through strategy:
        1. Write both messages to PostgreSQL
        2. Update Redis cache with new messages
        
        Returns:
            Tuple of (user_message_record, assistant_message_record)
        """
        # Write to database (durable)
        user_record = self._db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            response_type=None,  # User messages don't have response_type
        )
        
        assistant_record = self._db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            response_type=response_type,
        )
        
        # Update cache
        cached = self._get_from_cache(conversation_id)
        if cached:
            # Add new messages
            cached.recent_messages.append(user_record)
            cached.recent_messages.append(assistant_record)
            
            # Trim to max cached messages
            if len(cached.recent_messages) > MAX_CACHED_MESSAGES:
                cached.recent_messages = cached.recent_messages[-MAX_CACHED_MESSAGES:]
            
            cached.message_count += 2
            self._save_to_cache(cached)
        else:
            # Cache was empty, reload from database
            self._load_and_cache_from_database(conversation_id)
        
        logger.info("TURN_ADDED", extra={
            "conversation_id": conversation_id,
            "response_type": response_type,
        })
        
        return user_record, assistant_record
    
    def update_research_context(
        self,
        conversation_id: str,
        research_context_id: str,
        synthesis_summary: Optional[str] = None,
    ):
        """
        Update cache with latest research context info.
        Called after research is completed.
        """
        cached = self._get_from_cache(conversation_id)
        if cached:
            cached.last_research_context_id = research_context_id
            cached.last_synthesis_summary = synthesis_summary
            self._save_to_cache(cached)
    
    # ========================================================================
    # PUBLIC API: LLM Context
    # ========================================================================
    
    def get_messages_for_llm(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Get recent messages formatted for LLM context injection.
        
        Returns list of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        cached = self.get_conversation(conversation_id)
        if not cached:
            return []
        
        return cached.get_messages_for_llm(limit=limit)
    
    def has_conversation_started(self, conversation_id: str) -> bool:
        """
        Check if conversation has any messages.
        Used to avoid repeated greetings.
        """
        cached = self.get_conversation(conversation_id)
        if not cached:
            return False
        return cached.has_conversation_started()
    
    def get_last_synthesis(self, conversation_id: str) -> Optional[str]:
        """Get the last research synthesis summary for utility transformations."""
        cached = self.get_conversation(conversation_id)
        if cached:
            return cached.last_synthesis_summary
        
        # Fallback to database
        latest = self._db.get_latest_research_context(conversation_id)
        if latest:
            return latest.synthesis_summary
        
        return None
    
    # ========================================================================
    # PUBLIC API: Conversation Metadata
    # ========================================================================
    
    def update_title(
        self,
        conversation_id: str,
        title: str,
        generated: bool = False,
    ) -> bool:
        """
        Update conversation title in both cache and database.
        """
        # Update database
        success = self._db.update_conversation_title(
            conversation_id=conversation_id,
            title=title,
            title_generated=generated,
        )
        
        if not success:
            return False
        
        # Update cache
        cached = self._get_from_cache(conversation_id)
        if cached:
            cached.title = title
            cached.title_generated = generated
            self._save_to_cache(cached)
        
        return True
    
    def should_generate_title(self, conversation_id: str) -> bool:
        """
        Check if we should generate a title for this conversation.
        Returns True if: title not yet generated AND at least 2 messages exist.
        """
        cached = self.get_conversation(conversation_id)
        if not cached:
            return False
        
        return (
            not cached.title_generated
            and cached.message_count >= 2
            and cached.title == "New Conversation"
        )
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Soft delete conversation and invalidate cache.
        """
        # Invalidate cache first
        self._invalidate_cache(conversation_id)
        
        # Soft delete in database
        return self._db.delete_conversation(conversation_id)


# ============================================================================
# Module-level singleton
# ============================================================================

_chat_memory_service: Optional[ChatMemoryService] = None


def get_chat_memory_service() -> ChatMemoryService:
    """Get or create ChatMemoryService singleton."""
    global _chat_memory_service
    if _chat_memory_service is None:
        _chat_memory_service = ChatMemoryService()
    return _chat_memory_service