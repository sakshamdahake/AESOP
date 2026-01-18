"""
Conversation API endpoints.

Provides:
- POST /conversations/chat - Main chat endpoint with persistence
- GET /conversations - List user's conversations
- GET /conversations/{id} - Get full conversation
- DELETE /conversations/{id} - Soft delete conversation
- PATCH /conversations/{id}/title - Update title
"""

from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional
import uuid

from app.schemas.conversation import (
    ConversationChatRequest,
    ConversationChatResponse,
    ConversationListResponse,
    ConversationDetailResponse,
    ConversationDeleteResponse,
    ConversationTitleUpdateRequest,
)
from app.services.chat_memory import get_chat_memory_service
from app.services.database import get_database_service
from app.tasks import run_orchestrated_review_v2
from app.agents.title import get_title_generator
from app.logging import logger


router = APIRouter(prefix="/conversations", tags=["Conversations"])


# ============================================================================
# Main Chat Endpoint
# ============================================================================

@router.post("/chat", response_model=ConversationChatResponse)
def chat(
    request: ConversationChatRequest,
    x_anonymous_id: Optional[str] = Header(None, alias="X-Anonymous-Id"),
):
    """
    Main chat endpoint with persistent conversation support.
    
    - Auto-creates conversation if `conversation_id` not provided
    - Stores all messages in PostgreSQL permanently
    - Caches recent messages in Redis for fast access
    - Generates title after second message
    
    ## Headers
    - `X-Anonymous-Id`: Optional user identifier for linking conversations
    
    ## Examples
    
    **New conversation:**
```json
    {"message": "What are treatments for diabetes?"}
```
    
    **Continue conversation:**
```json
    {"message": "Tell me more about metformin", "conversation_id": "uuid-here"}
```
    """
    chat_memory = get_chat_memory_service()
    
    # Generate anonymous_id if not provided
    anonymous_id = x_anonymous_id or str(uuid.uuid4())
    
    # Get or create conversation
    try:
        cached_conv, is_new = chat_memory.get_or_create_conversation(
            conversation_id=request.conversation_id,
            anonymous_id=anonymous_id,
        )
    except ValueError as e:
        # Rate limit hit
        raise HTTPException(status_code=429, detail=str(e))
    
    conversation_id = cached_conv.conversation_id
    
    # Run the orchestrator graph
    try:
        result = run_orchestrated_review_v2(
            query=request.message,
            conversation_id=conversation_id,
            anonymous_id=anonymous_id,
        )
    except ValueError as e:
        # Message limit hit
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error("CHAT_ERROR", extra={
            "conversation_id": conversation_id,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail="An error occurred processing your message")
    
    # Check if we should generate title
    if chat_memory.should_generate_title(conversation_id):
        try:
            # Get first user message for title generation
            db = get_database_service()
            messages = db.get_recent_messages(conversation_id, limit=2)
            if messages:
                first_message = messages[0].content
                title_gen = get_title_generator()
                new_title = title_gen.generate(first_message)
                chat_memory.update_title(conversation_id, new_title, generated=True)
                result["title"] = new_title
        except Exception as e:
            logger.warning("TITLE_GENERATION_FAILED", extra={
                "conversation_id": conversation_id,
                "error": str(e),
            })
    
    # Get current title
    current_conv = chat_memory.get_conversation(conversation_id)
    title = current_conv.title if current_conv else "New Conversation"
    
    return ConversationChatResponse(
        response=result["response"],
        conversation_id=conversation_id,
        route_taken=result["route_taken"],
        intent=result.get("intent"),
        intent_confidence=result.get("intent_confidence"),
        papers_count=result["papers_count"],
        critic_decision=result.get("critic_decision"),
        avg_quality=result.get("avg_quality"),
        title=title,
        is_new_conversation=is_new,
    )


# ============================================================================
# Conversation Management Endpoints
# ============================================================================

@router.get("", response_model=ConversationListResponse)
def list_conversations(
    x_anonymous_id: str = Header(..., alias="X-Anonymous-Id"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List all conversations for a user.
    
    Returns conversations sorted by most recent activity.
    
    ## Headers
    - `X-Anonymous-Id`: Required user identifier
    
    ## Query Parameters
    - `limit`: Max conversations to return (default 20, max 100)
    - `offset`: Pagination offset (default 0)
    """
    db = get_database_service()
    
    conversations, total = db.list_conversations(
        anonymous_id=x_anonymous_id,
        limit=limit,
        offset=offset,
    )
    
    return ConversationListResponse(
        conversations=conversations,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    include_research: bool = Query(False),
    x_anonymous_id: Optional[str] = Header(None, alias="X-Anonymous-Id"),
):
    """
    Get full conversation with all messages.
    
    Optionally includes research contexts with papers.
    
    ## Query Parameters
    - `include_research`: Include research contexts and papers (default false)
    """
    db = get_database_service()
    chat_memory = get_chat_memory_service()
    
    conversation = db.get_conversation(
        conversation_id=conversation_id,
        include_messages=True,
        include_research=include_research,
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Verify ownership if anonymous_id provided
    if x_anonymous_id and conversation.anonymous_id and conversation.anonymous_id != x_anonymous_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Populate cache for fast subsequent access
    chat_memory.get_conversation(conversation_id)
    
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        anonymous_id=conversation.anonymous_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=conversation.messages,
        message_count=len(conversation.messages),
        research_contexts=conversation.research_contexts if include_research else None,
    )


@router.delete("/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(
    conversation_id: str,
    x_anonymous_id: str = Header(..., alias="X-Anonymous-Id"),
):
    """
    Soft delete a conversation.
    
    The conversation will be hidden from lists and inaccessible.
    
    ## Headers
    - `X-Anonymous-Id`: Required for ownership verification
    """
    db = get_database_service()
    chat_memory = get_chat_memory_service()
    
    # Verify ownership
    conversation = db.get_conversation(conversation_id, include_messages=False)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.anonymous_id != x_anonymous_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete
    success = chat_memory.delete_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
    
    return ConversationDeleteResponse(
        status="deleted",
        conversation_id=conversation_id,
    )


@router.patch("/{conversation_id}/title")
def update_title(
    conversation_id: str,
    request: ConversationTitleUpdateRequest,
    x_anonymous_id: str = Header(..., alias="X-Anonymous-Id"),
):
    """
    Update conversation title manually.
    
    ## Headers
    - `X-Anonymous-Id`: Required for ownership verification
    """
    db = get_database_service()
    chat_memory = get_chat_memory_service()
    
    # Verify ownership
    conversation = db.get_conversation(conversation_id, include_messages=False)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.anonymous_id != x_anonymous_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Update title
    success = chat_memory.update_title(
        conversation_id=conversation_id,
        title=request.title,
        generated=False,  # Manual update
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update title")
    
    return {"status": "updated", "title": request.title}