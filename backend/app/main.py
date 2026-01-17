"""
AESOP FastAPI Application

API v2.0 with RESTful session and message handling.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional, Generator
import os
import uuid
import json
import asyncpg
from neo4j import GraphDatabase
import redis.asyncio as aioredis

from app.tasks import run_review, run_orchestrated_review, create_metadata_dict
from app.services.session import get_session_service
from app.schemas.session import StructuredAnswer, AnswerSection
from app.schemas.api import (
    # New session endpoints
    CreateSessionRequest,
    CreateSessionResponse,
    SessionSummary,
    ListSessionsResponse,
    SessionDetailResponse,
    # New message endpoints
    SendMessageRequest,
    MessageResponse,
    MessageMetadata,
    # Streaming events
    SectionStartEvent,
    TokenEvent,
    SectionEndEvent,
    MetadataEvent,
    ErrorEvent,
    # Legacy endpoints (deprecated)
    ChatRequest,
    ChatResponse,
    ReviewRequest,
    ReviewResponse,
)

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aesop:aesop_pass@postgres:5432/aesop_db")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "aesop_graph_pass"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AESOP: System Starting Up...")
    
    # Check Postgres
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.close()
        print("Postgres (with pgvector) connected.")
    except Exception as e:
        print(f"Postgres Failed: {e}")

    # Check Redis
    try:
        r = aioredis.from_url(REDIS_URL)
        await r.ping()
        print("Redis connected.")
    except Exception as e:
        print(f"Redis Failed: {e}")

    # Check Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        print("Neo4j connected.")
        driver.close()
    except Exception as e:
        print(f"Neo4j Failed: {e}")

    yield
    print("AESOP: Shutting down...")


app = FastAPI(
    title="AESOP API",
    description="Agentic Evidence Synthesis & Orchestration Platform",
    version="2.1.0",
    lifespan=lifespan,
)

# =============================================================================
# CORS MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",  # Vite dev (127.0.0.1)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "system": "AESOP",
        "version": "2.1.0",
    }


# =============================================================================
# SESSION ENDPOINTS (NEW)
# =============================================================================

@app.post("/sessions", response_model=CreateSessionResponse, tags=["Sessions"])
def create_session(request: CreateSessionRequest = None):
    """
    Create a new chat session.
    
    Optionally include an initial message to start the conversation.
    If initial_message is provided, the response will include the AI's reply.
    
    ## Example
    
    ```json
    POST /sessions
    {
      "initial_message": "What are treatments for Type 2 diabetes?"
    }
    ```
    
    Returns session_id for subsequent message requests.
    """
    session_service = get_session_service()
    session_id = str(uuid.uuid4())
    
    # Handle empty request body
    if request is None:
        request = CreateSessionRequest()
    
    initial_response = None
    
    if request.initial_message:
        # Process the initial message
        result = run_orchestrated_review(
            query=request.initial_message,
            session_id=session_id,
        )
        
        # Get or create session context
        context = session_service.get_session(session_id)
        if not context:
            context = session_service.create_session(
                session_id=session_id,
                initial_message=request.initial_message,
            )
        
        # Add messages to history
        context.add_user_message(request.initial_message)
        context.add_assistant_message(
            answer=result["structured_answer"],
            metadata=create_metadata_dict(
                route_taken=result["route_taken"],
                intent=result.get("intent"),
                intent_confidence=result.get("intent_confidence"),
                papers_count=result["papers_count"],
                critic_decision=result.get("critic_decision"),
                avg_quality=result.get("avg_quality"),
            ),
        )
        session_service.save_session(context)
        
        initial_response = MessageResponse(
            answer=result["structured_answer"],
            metadata=MessageMetadata(
                intent=result.get("intent"),
                intent_confidence=result.get("intent_confidence"),
                processing_route=result["route_taken"],
                papers_count=result["papers_count"],
                review_outcome=result.get("critic_decision"),
                evidence_score=result.get("avg_quality"),
            ),
        )
        
        return CreateSessionResponse(
            session_id=session_id,
            created_at=context.created_at,
            title=context.title,
            initial_response=initial_response,
        )
    else:
        # Create empty session
        context = session_service.create_session(session_id=session_id)
        
        return CreateSessionResponse(
            session_id=session_id,
            created_at=context.created_at,
            title=None,
        )


@app.get("/sessions", response_model=ListSessionsResponse, tags=["Sessions"])
def list_sessions(
    limit: int = Query(50, ge=1, le=100, description="Maximum sessions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    List all chat sessions for sidebar history.
    
    Returns sessions sorted by last update time (newest first).
    
    ## Response
    
    ```json
    {
      "sessions": [
        {
          "session_id": "abc123",
          "title": "Treatments for Type 2 diabetes",
          "updated_at": "2026-01-11T12:45:00Z",
          "message_count": 4
        }
      ]
    }
    ```
    """
    session_service = get_session_service()
    sessions = session_service.list_sessions(limit=limit, offset=offset)
    
    return ListSessionsResponse(
        sessions=[
            SessionSummary(
                session_id=s["session_id"],
                title=s["title"],
                updated_at=s["updated_at"],
                message_count=s.get("message_count", 0),
            )
            for s in sessions
        ]
    )


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse, tags=["Sessions"])
def get_session_detail(session_id: str):
    """
    Get detailed information about a session including full message history.
    
    ## Response
    
    ```json
    {
      "session_id": "abc123",
      "title": "Treatments for Type 2 diabetes",
      "messages": [
        {
          "role": "user",
          "content": "What are treatments for Type 2 diabetes?"
        },
        {
          "role": "assistant",
          "answer": {
            "sections": [
              {"type": "summary", "content": "Type 2 diabetes is managed with..."}
            ]
          }
        }
      ],
      "papers_count": 12,
      "created_at": "2026-01-11T12:30:00Z",
      "updated_at": "2026-01-11T12:45:00Z"
    }
    ```
    """
    session_service = get_session_service()
    session = session_service.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return SessionDetailResponse(
        session_id=session.session_id,
        title=session.title or session.generate_title(),
        messages=session.messages,
        papers_count=len(session.retrieved_papers),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    """
    Delete a session and all its data.
    """
    session_service = get_session_service()
    success = session_service.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already expired")
    
    return {"status": "deleted", "session_id": session_id}


# =============================================================================
# MESSAGE ENDPOINTS (NEW)
# =============================================================================

@app.post(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse,
    tags=["Messages"],
)
def send_message(session_id: str, request: SendMessageRequest):
    """
    Send a message to an existing session.
    
    The session_id in the URL is the source of truth - no need to include it in the body.
    
    ## Example
    
    ```json
    POST /sessions/abc123/messages
    {
      "message": "What sample sizes did these studies use?"
    }
    ```
    
    ## Response
    
    ```json
    {
      "answer": {
        "sections": [
          {
            "type": "summary",
            "content": "The studies enrolled between 3,000 and 15,000 participants."
          }
        ]
      },
      "metadata": {
        "intent": "follow_up",
        "intent_confidence": 0.91,
        "processing_route": "context_qa",
        "papers_count": 12,
        "review_outcome": null,
        "evidence_score": null
      }
    }
    ```
    """
    session_service = get_session_service()
    
    # Verify session exists
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        # Process the message
        result = run_orchestrated_review(
            query=request.message,
            session_id=session_id,
        )
        
        # Refresh session (may have been updated by orchestrator)
        session = session_service.get_session(session_id)
        
        # Add messages to history
        session.add_user_message(request.message)
        session.add_assistant_message(
            answer=result["structured_answer"],
            metadata=create_metadata_dict(
                route_taken=result["route_taken"],
                intent=result.get("intent"),
                intent_confidence=result.get("intent_confidence"),
                papers_count=result["papers_count"],
                critic_decision=result.get("critic_decision"),
                avg_quality=result.get("avg_quality"),
            ),
        )
        
        # Update original query if this is first real message
        if not session.original_query:
            session.original_query = request.message
            session.title = session.generate_title()
        
        session_service.save_session(session)
        
        return MessageResponse(
            answer=result["structured_answer"],
            metadata=MessageMetadata(
                intent=result.get("intent"),
                intent_confidence=result.get("intent_confidence"),
                processing_route=result["route_taken"],
                papers_count=result["papers_count"],
                review_outcome=result.get("critic_decision"),
                evidence_score=result.get("avg_quality"),
            ),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/messages/stream",
    tags=["Messages"],
)
def send_message_stream(session_id: str, request: SendMessageRequest):
    """
    Send a message and receive a streaming response.
    
    Returns Server-Sent Events (SSE) with structured section events.
    
    ## Event Types
    
    - `section_start`: New section beginning (includes type)
    - `token`: Content token
    - `section_end`: Section complete
    - `metadata`: Final metadata object
    - `error`: Error during processing
    
    ## Example Events
    
    ```
    {"event": "section_start", "type": "summary"}
    {"event": "token", "content": "The studies enrolled "}
    {"event": "token", "content": "between 3,000 and 15,000 participants."}
    {"event": "section_end"}
    {"event": "metadata", "data": {...}}
    ```
    """
    session_service = get_session_service()
    
    # Verify session exists
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    def generate_events() -> Generator[str, None, None]:
        try:
            # Process the message (currently not streaming internally)
            result = run_orchestrated_review(
                query=request.message,
                session_id=session_id,
            )
            
            structured_answer = result["structured_answer"]
            
            # Emit section events
            for section in structured_answer.sections:
                # Section start
                yield json.dumps(SectionStartEvent(type=section.type).model_dump()) + "\n"
                
                # Simulate token streaming (split content into chunks)
                # In a real implementation, this would stream from the LLM
                words = section.content.split(" ")
                chunk_size = 5
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    if i + chunk_size < len(words):
                        chunk += " "
                    yield json.dumps(TokenEvent(content=chunk).model_dump()) + "\n"
                
                # Section end
                yield json.dumps(SectionEndEvent().model_dump()) + "\n"
            
            # Emit metadata
            metadata = MessageMetadata(
                intent=result.get("intent"),
                intent_confidence=result.get("intent_confidence"),
                processing_route=result["route_taken"],
                papers_count=result["papers_count"],
                review_outcome=result.get("critic_decision"),
                evidence_score=result.get("avg_quality"),
            )
            yield json.dumps(MetadataEvent(data=metadata).model_dump()) + "\n"
            
            # Update session
            session_ctx = session_service.get_session(session_id)
            if session_ctx:
                session_ctx.add_user_message(request.message)
                session_ctx.add_assistant_message(
                    answer=structured_answer,
                    metadata=create_metadata_dict(
                        route_taken=result["route_taken"],
                        intent=result.get("intent"),
                        intent_confidence=result.get("intent_confidence"),
                        papers_count=result["papers_count"],
                        critic_decision=result.get("critic_decision"),
                        avg_quality=result.get("avg_quality"),
                    ),
                )
                session_service.save_session(session_ctx)
            
        except Exception as e:
            yield json.dumps(ErrorEvent(message=str(e)).model_dump()) + "\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="application/x-ndjson",
    )


# =============================================================================
# LEGACY ENDPOINTS (DEPRECATED - MAINTAINED FOR BACKWARD COMPATIBILITY)
# =============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Legacy"], deprecated=True)
def chat(request: ChatRequest):
    """
    DEPRECATED: Use POST /sessions and POST /sessions/{id}/messages instead.
    
    Main chat endpoint with intent classification.
    Maintained for backward compatibility.
    """
    try:
        # Internally route to new implementation
        session_id = request.session_id
        
        if not session_id:
            # Create new session
            session_id = str(uuid.uuid4())
        
        result = run_orchestrated_review(
            query=request.message,
            session_id=session_id,
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            route_taken=result["route_taken"],
            intent=result.get("intent"),
            intent_confidence=result.get("intent_confidence"),
            papers_count=result["papers_count"],
            critic_decision=result.get("critic_decision"),
            avg_quality=result.get("avg_quality"),
            structured_answer=result.get("structured_answer"),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review", response_model=ReviewResponse, tags=["Legacy"], deprecated=True)
def review(request: ReviewRequest):
    """
    DEPRECATED: Use POST /sessions/{id}/messages instead.
    
    Legacy review endpoint (backward compatible).
    """
    try:
        result = run_orchestrated_review(
            query=request.query,
            session_id=request.session_id,
        )
        
        return ReviewResponse(
            response=result["response"],
            session_id=result["session_id"],
            route_taken=result["route_taken"],
            papers_count=result["papers_count"],
            critic_decision=result.get("critic_decision"),
            avg_quality=result.get("avg_quality"),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review/simple", tags=["Utilities"])
def simple_review(query: str):
    """
    Stateless single-turn review (no session support).
    
    Runs the basic CRAG graph without intent classification.
    Useful for one-off analysis and testing.
    """
    result = run_review(query)
    return result


# =============================================================================
# LEGACY SESSION ENDPOINT (redirect to new path)
# =============================================================================

@app.get("/session/{session_id}", tags=["Legacy"], deprecated=True)
def get_session_info_legacy(session_id: str):
    """
    DEPRECATED: Use GET /sessions/{session_id} instead.
    
    Maintained for backward compatibility.
    """
    return get_session_detail(session_id)


@app.delete("/session/{session_id}", tags=["Legacy"], deprecated=True)
def delete_session_legacy(session_id: str):
    """
    DEPRECATED: Use DELETE /sessions/{session_id} instead.
    
    Maintained for backward compatibility.
    """
    return delete_session(session_id)
