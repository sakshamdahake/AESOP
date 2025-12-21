from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import asyncpg
from neo4j import GraphDatabase
import redis.asyncio as redis

from app.tasks import run_review

# --- Configuration (Load from Env) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aesop:aesop_pass@postgres:5432/aesop_db")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "aesop_graph_pass"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# --- Lifespan: Manage Connection Pools ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Check Connections
    print("🚀 AESOP: System Starting Up...")
    
    # Check Postgres
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;") # Ensure pgvector is on
        await conn.close()
        print("✅ Postgres (with pgvector) connected.")
    except Exception as e:
        print(f"❌ Postgres Failed: {e}")

    # Check Redis
    try:
        r = redis.from_url(REDIS_URL)
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

app = FastAPI(title="Aesop Agentic API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Aesop MVP"}

@app.post("/review")
def review(query: str):
    result = run_review(query)
    return result