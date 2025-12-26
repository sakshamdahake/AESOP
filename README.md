# AESOP - Agentic Evidence Synthesis & Orchestration Platform

<p align="center">
  <img src="docs/assets/aesop-logo.png" alt="AESOP Logo" width="200" />
</p>

<p align="center">
  <strong>A Multi-Agent System for Automated Biomedical Literature Review</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#agents">Agents</a> •
  <a href="#memory-system">Memory System</a>
</p>

---

## Overview

AESOP is an advanced **Corrective Retrieval-Augmented Generation (CRAG)** system designed for biomedical literature review. It employs a multi-agent architecture where specialized AI agents collaborate to search, evaluate, and synthesize scientific evidence from PubMed.

Unlike traditional RAG systems that retrieve once and generate, AESOP implements **iterative refinement loops** where a Critic agent evaluates evidence quality and can trigger additional retrieval cycles until sufficient evidence is gathered.

### Key Differentiators

| Feature | Traditional RAG | AESOP |
|---------|----------------|-------|
| Retrieval | Single-pass | Iterative with CRAG loops |
| Quality Control | None | Critic agent with rubric-based evaluation |
| Memory | Stateless | Persistent pgvector + Redis session cache |
| Multi-turn | Not supported | Session-aware with intelligent routing |
| Evidence Grading | None | GRADE-inspired methodology scoring |

---

## Features

- **🔬 Biomedical Focus**: Specialized prompts and evaluation rubrics for scientific literature
- **🔄 CRAG Loop**: Corrective retrieval until evidence quality threshold is met
- **🧠 Persistent Memory**: pgvector-backed long-term memory influences future evaluations
- **💬 Multi-Turn Sessions**: Intelligent routing for follow-up queries (3-route model)
- **📊 Evidence Grading**: GRADE-inspired methodology and relevance scoring
- **🏗️ LangGraph Architecture**: Robust state machine with conditional edges
- **☁️ AWS Bedrock**: Claude Haiku, Nova Pro, and Titan embeddings
- **🐳 Fully Dockerized**: One-command deployment with all dependencies

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AESOP SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐     ┌──────────────────────────────────────────────┐    │
│    │   FastAPI    │     │           ORCHESTRATOR GRAPH                 │    │
│    │   /review    │────▶│  ┌────────┐                                  │    │
│    └──────────────┘     │  │ Router │──┬─▶ Route A: Full Graph         │    │
│                         │  └────────┘  │   (Scout→Critic→Synthestic)   │    │
│                         │              ├─▶ Route B: Augmented Context   │    │
│                         │              │   (Scout→Merge→Synthesizer)   │    │
│                         │              └─▶ Route C: Context Q&A         │    │
│                         │                  (Direct LLM Answer)          │    │
│                         └──────────────────────────────────────────────┘    │
│                                           │                                  │
│    ┌──────────────────────────────────────┼──────────────────────────────┐  │
│    │                         DATA LAYER   │                              │  │
│    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │  │
│    │  │  PostgreSQL │  │    Redis    │  │   Neo4j     │                 │  │
│    │  │  + pgvector │  │   Session   │  │  (Future)   │                 │  │
│    │  │   Memory    │  │    Cache    │  │  Citations  │                 │  │
│    │  └─────────────┘  └─────────────┘  └─────────────┘                 │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Flow (Route A - Full Graph)

```
                    ┌─────────────────────────────────────────┐
                    │              USER QUERY                  │
                    │  "What are treatments for Type 2        │
                    │   diabetes?"                            │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │            ROUTER AGENT                  │
                    │  • Checks Redis for session context     │
                    │  • Analyzes query patterns              │
                    │  • Routes to appropriate path           │
                    └─────────────────┬───────────────────────┘
                                      │ (New Session → Route A)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CRAG LOOP                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │   ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐    │    │
│  │   │   SCOUT     │      │   CRITIC    │      │   Decision:     │    │    │
│  │   │             │      │             │      │                 │    │    │
│  │   │ • Expand    │─────▶│ • Grade     │─────▶│ sufficient?     │    │    │
│  │   │   query     │      │   papers    │      │                 │    │    │
│  │   │ • Search    │      │ • CRAG      │      │ YES → Synthesize│    │    │
│  │   │   PubMed    │      │   decision  │      │ NO  → Loop back │    │    │
│  │   │ • Fetch     │      │ • Memory    │      │                 │    │    │
│  │   │   abstracts │      │   update    │      └────────┬────────┘    │    │
│  │   └─────────────┘      └─────────────┘               │             │    │
│  │         ▲                                            │             │    │
│  │         └────────────── retrieve_more ───────────────┘             │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │ sufficient                            │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │          SYNTHESIZER AGENT              │
                    │  • Formats evidence by quality          │
                    │  • Generates structured review          │
                    │  • Cites PMIDs                          │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │          SAVE SESSION                    │
                    │  • Cache to Redis (60 min TTL)          │
                    │  • Store query embedding                │
                    │  • Enable follow-up queries             │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │           STRUCTURED REVIEW             │
                    │  1. Background                          │
                    │  2. High-Quality Evidence Summary       │
                    │  3. Lower-Quality Evidence              │
                    │  4. Limitations                         │
                    │  5. Conclusion                          │
                    └─────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- AWS Account with Bedrock access (Claude, Nova, Titan)
- AWS credentials configured

### 1. Clone Repository

```bash
git clone https://github.com/your-org/aesop.git
cd aesop
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your AWS credentials:

```env
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=aesop-dev
```

### 3. Start Services

```bash
docker-compose up -d
```

This starts:
- **Backend API**: http://localhost:8000
- **PostgreSQL + pgvector**: localhost:5432
- **Redis**: localhost:6379
- **Neo4j**: http://localhost:7474 (future use)

### 4. Initialize Database

```bash
docker exec -it aesop_backend python -c "
import asyncpg
import asyncio

async def init():
    conn = await asyncpg.connect('postgresql://aesop:aesop_pass@postgres:5432/aesop_db')
    
    # Create pgvector extension
    await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    # Create critic memory table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS critic_acceptance_memory (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            research_query TEXT NOT NULL,
            query_hash TEXT GENERATED ALWAYS AS (md5(lower(trim(research_query)))) STORED,
            query_embedding VECTOR(1536) NOT NULL,
            pmid TEXT NOT NULL,
            study_type TEXT,
            publication_year INT,
            relevance_score FLOAT CHECK (relevance_score BETWEEN 0 AND 1),
            methodology_score FLOAT CHECK (methodology_score BETWEEN 0 AND 1),
            quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1),
            iteration INT NOT NULL,
            accepted_at TIMESTAMP DEFAULT now()
        );
        
        CREATE INDEX IF NOT EXISTS idx_query_hash ON critic_acceptance_memory (query_hash);
        CREATE INDEX IF NOT EXISTS idx_query_embedding ON critic_acceptance_memory 
            USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);
    ''')
    
    await conn.close()
    print('✅ Database initialized')

asyncio.run(init())
"
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Run a literature review
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the treatments for Type 2 diabetes?"}'
```

---

## API Reference

### Endpoints

#### `POST /review`

Execute a literature review query with session support.

**Request:**
```json
{
  "query": "What are the latest treatments for Type 2 diabetes?",
  "session_id": null  // Optional: for follow-up queries
}
```

**Response:**
```json
{
  "response": "## Background\n\nType 2 diabetes mellitus...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "route_taken": "full_graph",
  "papers_count": 12,
  "critic_decision": "sufficient",
  "avg_quality": 0.72
}
```

#### `POST /review` (Follow-up)

```json
{
  "query": "What sample sizes did these studies use?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "response": "Based on the retrieved studies...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "route_taken": "context_qa",
  "papers_count": 12,
  "critic_decision": null,
  "avg_quality": null
}
```

#### `DELETE /session/{session_id}`

Manually invalidate a session.

**Response:**
```json
{
  "status": "deleted",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### `GET /health`

Health check endpoint.

---

## Agents

### 1. Router Agent

**Purpose:** Intelligent query classification for multi-turn conversations.

**Model:** Claude 3 Haiku (fast, cost-effective)

**Location:** `backend/app/agents/router/`

#### Routing Logic

The Router uses a **multi-signal approach** to classify queries:

| Signal | Description | Detection Method |
|--------|-------------|------------------|
| **Deictic Markers** | "these studies", "those results" | Regex patterns |
| **Explicit References** | "first paper", "PMID 12345" | Regex patterns |
| **Keyword Overlap** | Shared medical terms | Jaccard similarity |
| **Query Type** | Clarification vs. new question | Pattern matching |
| **Embedding Similarity** | Semantic relatedness | Cosine similarity (secondary) |

#### Routes

| Route | Trigger | Execution | Cost |
|-------|---------|-----------|------|
| **Route A: Full Graph** | New topic, low similarity | Scout → Critic → Synthesizer | High |
| **Route B: Augmented Context** | Related topic, needs new evidence | Scout → Merge → Synthesizer | Medium |
| **Route C: Context Q&A** | Question about existing results | Direct LLM with cached papers | Low |

#### Example Classifications

```python
# Route C (Context Q&A)
"What sample sizes did these studies use?"  # Deictic "these studies"
"Explain the methodology of paper 1"         # Explicit reference
"Compare the RCTs in the results"            # Comparison of existing

# Route B (Augmented Context)  
"What about metformin side effects?"         # Related but new focus

# Route A (Full Graph)
"What causes Alzheimer's disease?"           # Completely new topic
```

---

### 2. Scout Agent

**Purpose:** Query expansion and literature retrieval from PubMed.

**Model:** Claude 3 Haiku

**Location:** `backend/app/agents/scout/`

#### Workflow

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         QUERY EXPANSION              │
│  LLM generates 3-5 PubMed queries   │
│                                      │
│  Input: "diabetes treatments"        │
│  Output:                             │
│    - "type 2 diabetes treatment"     │
│    - "diabetes mellitus therapy"     │
│    - "antidiabetic medications"      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         PUBMED SEARCH                │
│  ESearch API for each query         │
│  Returns PMIDs (max 10 per query)   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         PUBMED FETCH                 │
│  EFetch API for abstracts           │
│  Batched requests (3 PMIDs/batch)   │
│  Fault-tolerant (partial results)   │
└─────────────────┬───────────────────┘
                  │
                  ▼
            List[Paper]
```

#### Output Format

```python
class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
    publication_year: Optional[int]
    journal: Optional[str]
```

#### Error Handling

- Network failures return empty results (never crash)
- Invalid PMIDs are skipped
- Partial results are accepted

---

### 3. Critic Agent

**Purpose:** Evidence evaluation and CRAG decision-making.

**Model:** Amazon Nova Pro (better reasoning)

**Location:** `backend/app/agents/critic/`

#### Grading Rubric

Each paper is evaluated on:

| Metric | Range | Description |
|--------|-------|-------------|
| `relevance_score` | 0.0 - 1.0 | Topical relevance to research question |
| `methodology_score` | 0.0 - 1.0 | Methodological rigor |
| `sample_size_adequate` | bool | Sufficient sample for study type |
| `study_type` | string | RCT, Cohort, Case-Control, etc. |
| `recommendation` | enum | KEEP, DISCARD, NEEDS_MORE |

#### Evidence Hierarchy Priors (GRADE-Inspired)

```python
STUDY_TYPE_PRIORS = {
    "meta-analysis": 0.85,
    "systematic review": 0.80,
    "randomized controlled trial": 0.70,
    "rct": 0.70,
    "cohort study": 0.55,
    "case-control study": 0.50,
    "cross-sectional study": 0.45,
    "case series": 0.30,
    "case study": 0.25,
    "expert opinion": 0.20,
}
```

#### CRAG Decision Logic

```python
def _make_global_decision(grades, iteration, memory_boost):
    # Compute aggregate metrics
    keep_ratio = count(KEEP) / total
    discard_ratio = count(DISCARD) / total
    avg_quality = mean((relevance + methodology) / 2)
    
    # Apply memory-influenced threshold
    effective_threshold = max(
        MIN_CONFIDENCE_FLOOR,  # 0.45
        MIN_AVG_QUALITY_FOR_SUFFICIENT  # 0.60
        - (iteration * CONFIDENCE_DECAY_RATE)  # 0.07 per iteration
        - memory_boost,  # 0.0 - 0.15
    )
    
    # Decision rules
    if keep_ratio >= 0.40:
        return "sufficient"
    if discard_ratio >= 0.40:
        return "retrieve_more"
    if avg_quality >= effective_threshold:
        return "sufficient"
    return "retrieve_more"
```

---

### 4. Synthesizer Agent

**Purpose:** Generate structured literature review from graded evidence.

**Model:** Amazon Nova Pro

**Location:** `backend/app/agents/synthesizer/`

#### Output Structure

```markdown
## 1. Background
[Context and importance of the research question]

## 2. Summary of High-Quality Evidence
[Papers with quality score ≥ 0.7, cited by PMID]

## 3. Summary of Lower-Quality or Conflicting Evidence
[Papers with score < 0.7 or conflicting findings]

## 4. Limitations of Current Evidence
[Gaps, biases, methodological concerns]

## 5. Conclusion
[Evidence-based answer to the research question]
```

#### Paper Filtering

```python
def build_graded_papers(papers, grades):
    for paper, grade in zip(papers, grades):
        # Skip discarded papers
        if grade.recommendation == "discard":
            continue
        
        # Compute quality score
        score = (relevance + methodology) / 2
        
        # Penalize inadequate sample size
        if not grade.sample_size_adequate:
            score *= 0.7
        
        yield GradedPaper(pmid, title, abstract, score)
```

---

### 5. Context Q&A Agent

**Purpose:** Answer follow-up questions using cached papers (no retrieval).

**Model:** Claude 3 Haiku

**Location:** `backend/app/agents/context_qa/`

#### When Used

- Route C queries (high similarity or explicit references)
- Questions about existing search results
- Clarifications, comparisons, explanations

#### Context Injection

```python
def get_papers_context(session_context, max_papers=10):
    """Format cached papers for LLM context."""
    for i, paper in enumerate(papers[:max_papers]):
        yield f"""
[Paper {i}]
PMID: {paper.pmid}
Title: {paper.title}
Quality Score: {paper.quality_score}
Abstract: {paper.abstract[:600]}...
"""
```

---

## Memory System

AESOP implements a **two-layer memory architecture**:

### Layer 1: Session Cache (Redis)

**Purpose:** Enable multi-turn conversations within a session.

**TTL:** 60 minutes

**Storage:**
```python
class SessionContext(BaseModel):
    session_id: str
    original_query: str
    query_embedding: List[float]  # 1536-dim Titan
    retrieved_papers: List[CachedPaper]
    synthesis_summary: str
    turn_count: int
    created_at: datetime
    updated_at: datetime
```

**Key Format:** `aesop:session:{session_id}`

**Usage:**
- Router fetches session to determine query relatedness
- Context Q&A uses cached papers for direct answers
- Augmented Context merges cached + new papers

---

### Layer 2: Long-Term Memory (PostgreSQL + pgvector)

**Purpose:** Persistent learning across sessions. High-confidence evidence influences future evaluations.

#### Schema

```sql
CREATE TABLE critic_acceptance_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Query context
    research_query TEXT NOT NULL,
    query_hash TEXT GENERATED ALWAYS AS (md5(lower(trim(research_query)))) STORED,
    query_embedding VECTOR(1536) NOT NULL,
    
    -- Paper identity
    pmid TEXT NOT NULL,
    study_type TEXT,
    publication_year INT,
    
    -- Critic-derived scores
    relevance_score FLOAT CHECK (relevance_score BETWEEN 0 AND 1),
    methodology_score FLOAT CHECK (methodology_score BETWEEN 0 AND 1),
    quality_score FLOAT CHECK (quality_score BETWEEN 0 AND 1),
    
    -- CRAG context
    iteration INT NOT NULL,
    accepted_at TIMESTAMP DEFAULT now()
);

-- Indexes
CREATE INDEX idx_query_hash ON critic_acceptance_memory (query_hash);
CREATE INDEX idx_query_embedding ON critic_acceptance_memory 
    USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);
```

#### Memory Retrieval

```python
class CriticMemoryStore:
    MAX_BOOST = 0.15           # Maximum threshold reduction
    SIMILARITY_THRESHOLD = 0.75  # Minimum similarity for retrieval
    DECAY_LAMBDA = 0.01         # Recency decay factor
    
    def fetch_memory_bias(self, query: str) -> float:
        # 1. Try exact match (fast path)
        rows = db.query("""
            SELECT quality_score, accepted_at, 1.0 AS similarity
            FROM critic_acceptance_memory
            WHERE query_hash = md5(lower(trim(%s)))
        """, query)
        
        # 2. Fall back to vector similarity
        if not rows:
            embedding = embed_query(query)
            rows = db.query("""
                SELECT quality_score, accepted_at,
                       1 - (query_embedding <=> %s::vector) AS similarity
                FROM critic_acceptance_memory
                WHERE (1 - (query_embedding <=> %s::vector)) >= %s
                ORDER BY similarity DESC LIMIT 10
            """, embedding, embedding, SIMILARITY_THRESHOLD)
        
        # 3. Compute weighted score with recency decay
        weighted_scores = []
        for quality, accepted_at, similarity in rows:
            age_days = (now - accepted_at).days
            recency = exp(-DECAY_LAMBDA * age_days)
            weighted_scores.append(quality * similarity * recency)
        
        # 4. Return bounded average
        return min(mean(weighted_scores), MAX_BOOST)
```

#### Memory Influence (Safe, Bounded)

```
Memory DOES:
  ✓ Slightly lower quality threshold (max 0.15 reduction)
  ✓ Enable faster convergence for seen queries
  ✓ Remember high-quality evidence

Memory DOES NOT:
  ✗ Skip paper evaluation
  ✗ Force "sufficient" decision
  ✗ Change individual paper grades
  ✗ Reuse old papers without re-grading
```

**Safety Principle:** Memory influences thresholds, never decisions directly.

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://aesop:aesop_pass@postgres:5432/aesop_db` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `NEO4J_URI` | Neo4j bolt URI | `bolt://neo4j:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `aesop_graph_pass` |
| `AWS_ACCESS_KEY_ID` | AWS credentials | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | Optional |
| `LANGCHAIN_PROJECT` | LangSmith project | `aesop-dev` |

### CRAG Tuning Parameters

Located in `backend/app/agents/critic/rubric.py`:

```python
# Evidence thresholds
MIN_RELEVANCE_TO_KEEP = 0.45
MIN_METHODOLOGY_TO_KEEP = 0.50

# CRAG convergence
MIN_AVG_QUALITY_FOR_SUFFICIENT = 0.60
MAX_DISCARD_RATIO = 0.55

# Iteration decay
CONFIDENCE_DECAY_RATE = 0.07  # Per iteration
MIN_CONFIDENCE_FLOOR = 0.45

# Memory bounds
MAX_MEMORY_BOOST = 0.15
```

---

## Project Structure

```
aesop/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── critic/
│   │   │   │   ├── agent.py        # CriticAgent with CRAG logic
│   │   │   │   ├── memory.py       # pgvector memory store
│   │   │   │   ├── node.py         # LangGraph node
│   │   │   │   ├── prompts.py      # Evaluation prompts
│   │   │   │   ├── rubric.py       # GRADE-inspired thresholds
│   │   │   │   └── schemas.py      # PaperGrade, Recommendation
│   │   │   ├── scout/
│   │   │   │   ├── agent.py        # Scout node with query expansion
│   │   │   │   ├── prompts.py      # Expansion prompts
│   │   │   │   └── tools.py        # PubMed API integration
│   │   │   ├── synthesizer/
│   │   │   │   ├── agent.py        # Synthesis generation
│   │   │   │   ├── prompts.py      # Review structure prompts
│   │   │   │   ├── schemas.py      # GradedPaper
│   │   │   │   └── utils.py        # Paper formatting
│   │   │   ├── router/
│   │   │   │   ├── agent.py        # Multi-signal router
│   │   │   │   ├── node.py         # Router LangGraph node
│   │   │   │   └── prompts.py      # Classification prompts
│   │   │   ├── context_qa/
│   │   │   │   ├── agent.py        # Direct Q&A from context
│   │   │   │   └── node.py         # Context Q&A node
│   │   │   ├── graph.py            # Original CRAG graph
│   │   │   ├── orchestrator_graph.py  # Session-aware graph
│   │   │   └── state.py            # AgentState, OrchestratorState
│   │   ├── schemas/
│   │   │   └── session.py          # SessionContext, RouterDecision
│   │   ├── services/
│   │   │   └── session.py          # Redis SessionService
│   │   ├── embeddings/
│   │   │   └── bedrock.py          # Titan embeddings + cosine sim
│   │   ├── main.py                 # FastAPI application
│   │   ├── tasks.py                # Review task runners
│   │   └── logging.py              # Structured logging
│   ├── db/
│   │   └── migrations/
│   │       └── 001_create_critic_memory.sql
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Development

### Running Tests

```bash
# Enter backend container
docker exec -it aesop_backend bash

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_router.py -v
```

### Local Development

```bash
# Install dependencies
cd backend
uv sync

# Run locally (requires services running)
uvicorn app.main:app --reload --port 8000
```

### Adding a New Agent

1. Create directory: `backend/app/agents/your_agent/`
2. Implement:
   - `agent.py` - Core logic
   - `node.py` - LangGraph node wrapper
   - `prompts.py` - LLM prompts
   - `schemas.py` - Pydantic models (if needed)
3. Add node to graph in `orchestrator_graph.py`
4. Update state in `state.py` if needed

### Debugging

```bash
# View logs
docker-compose logs -f backend

# Check Redis sessions
docker exec -it aesop_redis redis-cli
> KEYS aesop:session:*
> GET aesop:session:<session_id>

# Check PostgreSQL memory
docker exec -it aesop_postgres psql -U aesop -d aesop_db
> SELECT COUNT(*) FROM critic_acceptance_memory;
> SELECT research_query, quality_score FROM critic_acceptance_memory ORDER BY accepted_at DESC LIMIT 10;
```

---

## Performance

### Latency Breakdown (Route A)

| Stage | Typical Duration |
|-------|------------------|
| Router | 200-500ms |
| Scout (query expansion) | 1-3s |
| Scout (PubMed fetch) | 5-15s |
| Critic (per paper) | 1-2s |
| Synthesizer | 5-20s |
| **Total** | **15-45s** |

### Cost Estimation

| Route | LLM Calls | Est. Cost |
|-------|-----------|-----------|
| Route A (Full) | Router + Scout + Critic×N + Synthesizer | $0.05-0.10 |
| Route B (Augmented) | Router + Scout + Synthesizer | $0.03-0.06 |
| Route C (Context Q&A) | Router + Context QA | $0.01-0.02 |

---

## Roadmap

### Completed ✅

- [x] Multi-agent CRAG architecture
- [x] pgvector long-term memory
- [x] Multi-turn session support
- [x] Intelligent 3-route routing
- [x] AWS Bedrock integration

### In Progress 🚧

- [ ] Neo4j citation graph integration
- [ ] Streaming responses
- [ ] Batch processing API

### Planned 📋

- [ ] Full-text PDF retrieval
- [ ] User feedback loop
- [ ] Memory pruning/aging policies
- [ ] Evaluation benchmarks
- [ ] Web UI

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - State machine framework
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) - Literature API
- [GRADE Working Group](https://www.gradeworkinggroup.org/) - Evidence grading methodology

---

<p align="center">
  <sub>Built with ❤️ for evidence-based medicine</sub>
</p>