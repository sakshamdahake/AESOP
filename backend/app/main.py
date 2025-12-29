"""
AESOP FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
import os
import asyncpg
from neo4j import GraphDatabase
import redis.asyncio as aioredis

from app.tasks import run_review, run_orchestrated_review
from app.services.session import get_session_service

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aesop:aesop_pass@postgres:5432/aesop_db")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "aesop_graph_pass"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AESOP: System Starting Up...")
    
    # Check Postgres
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.close()
        print("✅ Postgres (with pgvector) connected.")
    except Exception as e:
        print(f"❌ Postgres Failed: {e}")

    # Check Redis
    try:
        r = aioredis.from_url(REDIS_URL)
        await r.ping()
        print("✅ Redis connected.")
    except Exception as e:
        print(f"❌ Redis Failed: {e}")

    # Check Neo4j
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        print("✅ Neo4j connected.")
        driver.close()
    except Exception as e:
        print(f"❌ Neo4j Failed: {e}")

    yield
    print("🛑 AESOP: Shutting down...")


app = FastAPI(
    title="AESOP API",
    description="Agentic Evidence Synthesis & Orchestration Platform",
    version="2.0.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    """Request model for chat/review endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for follow-ups")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "message": "What are the treatments for Type 2 diabetes?",
                    "session_id": None
                },
                {
                    "message": "What sample sizes did these studies use?",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000"
                },
                {
                    "message": "Hello, what can you do?",
                    "session_id": None
                }
            ]
        }


class ChatResponse(BaseModel):
    """Response model for chat/review endpoint."""
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID for follow-ups")
    route_taken: str = Field(..., description="Execution route: chat, utility, full_graph, augmented_context, context_qa")
    intent: Optional[str] = Field(None, description="Classified intent: research, followup_research, chat, utility")
    intent_confidence: Optional[float] = Field(None, description="Intent classification confidence")
    papers_count: int = Field(0, description="Number of papers in context")
    critic_decision: Optional[str] = Field(None, description="CRAG decision: sufficient, retrieve_more")
    avg_quality: Optional[float] = Field(None, description="Average evidence quality score")


class ReviewRequest(BaseModel):
    """Legacy request model (backward compatible)."""
    query: str = Field(..., min_length=5, max_length=2000)
    session_id: Optional[str] = None


class ReviewResponse(BaseModel):
    """Legacy response model (backward compatible)."""
    response: str
    session_id: str
    route_taken: str
    papers_count: int
    critic_decision: Optional[str] = None
    avg_quality: Optional[float] = None


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "system": "AESOP",
        "version": "2.0.0",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint with intent classification.
    
    Handles all types of interactions:
    - **Chat**: Greetings, thanks, system questions
    - **Research**: Biomedical literature queries
    - **Follow-up**: Questions about previous results
    - **Utility**: Reformat existing output
    
    The system automatically classifies intent and routes accordingly.
    
    ## Examples
    
    **Chat:**
    ```json
    {"message": "Hello!", "session_id": null}
    {"message": "What can you do?", "session_id": null}
    ```
    
    **Research:**
    ```json
    {"message": "What are treatments for Type 2 diabetes?", "session_id": null}
    ```
    
    **Follow-up (requires session_id):**
    ```json
    {"message": "What sample sizes did these studies use?", "session_id": "<session_id>"}
    ```
    """
    try:
        result = run_orchestrated_review(
            query=request.message,
            session_id=request.session_id,
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
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest):
    """
    Legacy review endpoint (backward compatible).
    
    Use `/chat` for the full intent-aware experience.
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


@app.post("/review/simple")
def simple_review(query: str):
    """
    Original single-turn review (no session support).
    
    Runs the basic CRAG graph without intent classification.
    """
    result = run_review(query)
    return result


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """
    Manually invalidate a session.
    
    Use this to clear session context and start fresh.
    """
    session_service = get_session_service()
    success = session_service.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already expired")
    
    return {"status": "deleted", "session_id": session_id}


@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """
    Get information about a session.
    
    Returns session context including query history and cached papers.
    """
    session_service = get_session_service()
    session = session_service.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "session_id": session.session_id,
        "original_query": session.original_query,
        "turn_count": session.turn_count,
        "papers_count": len(session.retrieved_papers),
        "papers": [
            {
                "pmid": p.pmid,
                "title": p.title[:100],
                "quality_score": p.quality_score,
            }
            for p in session.retrieved_papers[:10]
        ],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }