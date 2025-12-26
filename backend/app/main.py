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


app = FastAPI(title="AESOP Agentic API", lifespan=lifespan)


# --- Request/Response Models ---

class ReviewRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    session_id: Optional[str] = None


class ReviewResponse(BaseModel):
    response: str
    session_id: str
    route_taken: str
    papers_count: int
    critic_decision: Optional[str] = None
    avg_quality: Optional[float] = None


# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "AESOP v2"}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest):
    """
    Multi-turn literature review with session support.
    
    - First query: Creates new session, runs full graph
    - Follow-up queries: Routes intelligently based on similarity
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
    Original single-turn review (backward compatible).
    """
    result = run_review(query)
    return result


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Manually invalidate a session."""
    session_service = get_session_service()
    success = session_service.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already expired")
    
    return {"status": "deleted", "session_id": session_id}