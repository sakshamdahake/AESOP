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
| Intent Understanding | None | Hybrid classifier (pattern + LLM) |
| Conversation | Single query | Natural chat with context |

---

## Features

- **🔬 Biomedical Focus**: Specialized prompts and evaluation rubrics for scientific literature
- **🔄 CRAG Loop**: Corrective retrieval until evidence quality threshold is met
- **🧠 Persistent Memory**: pgvector-backed long-term memory influences future evaluations
- **💬 Multi-Turn Sessions**: Intelligent routing for follow-up queries (4-route model)
- **🎯 Intent Classification**: Hybrid pattern + LLM classifier for smart query understanding
- **🗣️ Natural Chat**: Conversational interface with greetings, thanks, and system questions
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
│    │   /chat      │────▶│                                              │    │
│    │   /review    │     │  ┌────────┐                                  │    │
│    └──────────────┘     │  │ Intent │──┬─▶ Chat (General conversation) │    │
│                         │  └────────┘  ├─▶ Utility (Reformat output)   │    │
│                         │              └─▶ Router (Research queries)   │    │
│                         │                    │                         │    │
│                         │              ┌─────┴─────┐                   │    │
│                         │              ▼           ▼                   │    │
│                         │         ┌────────┐  ┌────────┐               │    │
│                         │         │ Route A│  │Route B │               │    │
│                         │         │ (Full) │  │(Augment│               │    │
│                         │         └────────┘  └────────┘               │    │
│                         │              │           │                   │    │
│                         │              ▼           ▼                   │    │
│                         │         ┌────────────────────┐               │    │
│                         │         │     Route C        │               │    │
│                         │         │   (Context Q&A)    │               │    │
│                         │         └────────────────────┘               │    │
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

### Complete Agent Flow

```
                    ┌─────────────────────────────────────────┐
                    │              USER MESSAGE                │
                    │  "Hello!" / "What causes diabetes?" /   │
                    │  "Compare these studies"                │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         INTENT CLASSIFIER                │
                    │                                          │
                    │  Stage 1: Fast-path (regex patterns)    │
                    │  Stage 2: Keyword analysis              │
                    │  Stage 3: LLM classification            │
                    │  Stage 4: Context validation            │
                    └─────────────────┬───────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │     CHAT      │        │    UTILITY    │        │    ROUTER     │
    │               │        │               │        │               │
    │ • Greetings   │        │ • Shorten     │        │ • Analyze     │
    │ • Thanks      │        │ • Bullets     │        │   query       │
    │ • System Q&A  │        │ • Simplify    │        │ • Check       │
    │ • Small talk  │        │ • Key points  │        │   session     │
    └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
            │                         │                         │
            │                         │           ┌─────────────┼─────────────┐
            │                         │           │             │             │
            │                         │           ▼             ▼             ▼
            │                         │    ┌───────────┐ ┌───────────┐ ┌───────────┐
            │                         │    │  ROUTE A  │ │  ROUTE B  │ │  ROUTE C  │
            │                         │    │Full Graph │ │ Augmented │ │Context QA │
            │                         │    │           │ │  Context  │ │           │
            │                         │    │Scout→     │ │Scout→     │ │ Direct    │
            │                         │    │Critic→    │ │Merge→     │ │ LLM with  │
            │                         │    │Synthestic │ │Synthesizer│ │ cache     │
            │                         │    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
            │                         │          │             │             │
            └─────────────────────────┴──────────┴─────────────┴─────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────┐
                                    │        SAVE SESSION         │
                                    │  • Update Redis cache       │
                                    │  • Extend TTL               │
                                    │  • Store embeddings         │
                                    └─────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────┐
                                    │          RESPONSE           │
                                    └─────────────────────────────┘
```

### Intent Classification Flow (Hybrid Approach)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID INTENT CLASSIFIER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: Fast-Path (Regex) ─────────────────────────────── ~50% of msgs    │
│  ─────────────────────────────                                               │
│  Patterns: hi, hello, thanks, bye, ok, yes, no, cool, great...              │
│  Result: Instant "chat" classification (no LLM cost)                        │
│                                                                              │
│  STAGE 2: Keyword Analysis ──────────────────────────────── ~30% of msgs    │
│  ─────────────────────────────                                               │
│  • MEDICAL_KEYWORDS (100+ terms): diabetes, cancer, treatment...            │
│  • SYSTEM_KEYWORDS: "who are you", "what can you do"...                     │
│  • FOLLOWUP_KEYWORDS: "these studies", "compare them"...                    │
│  • UTILITY_KEYWORDS: "make it shorter", "bullet points"...                  │
│                                                                              │
│  STAGE 3: LLM Classification ────────────────────────────── ~20% of msgs    │
│  ─────────────────────────────                                               │
│  Claude Haiku with detailed prompt for ambiguous cases                      │
│                                                                              │
│  STAGE 4: Context Validation                                                │
│  ─────────────────────────────                                               │
│  • followup_research without session → research                             │
│  • utility without output → chat                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
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

# Chat (general conversation)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Research query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the treatments for Type 2 diabetes?"}'

# Follow-up query (use session_id from previous response)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What sample sizes did these studies use?", "session_id": "<session_id>"}'
```

---

## API Reference

### Endpoints

#### `POST /chat` (Recommended)

Main endpoint with full intent classification support.

**Request:**
```json
{
  "message": "What are the latest treatments for Type 2 diabetes?",
  "session_id": null
}
```

**Response:**
```json
{
  "response": "## Background\n\nType 2 diabetes mellitus...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "route_taken": "full_graph",
  "intent": "research",
  "intent_confidence": 0.92,
  "papers_count": 12,
  "critic_decision": "sufficient",
  "avg_quality": 0.72
}
```

#### Intent-Based Responses

| Message Type | Intent | Route | Response |
|--------------|--------|-------|----------|
| "Hello!" | `chat` | `chat` | Friendly greeting |
| "What can you do?" | `chat` | `chat` | Capability explanation |
| "What causes diabetes?" | `research` | `full_graph` | Literature review |
| "What sample sizes?" (with session) | `followup_research` | `context_qa` | Answer from cached papers |
| "Make it shorter" (with session) | `utility` | `utility` | Reformatted output |

#### `POST /review` (Legacy)

Backward-compatible endpoint.

```json
{
  "query": "What are the treatments for Type 2 diabetes?",
  "session_id": null
}
```

#### `GET /session/{session_id}`

Get session information.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "original_query": "What are treatments for diabetes?",
  "turn_count": 3,
  "papers_count": 12,
  "papers": [
    {"pmid": "12345678", "title": "Study on diabetes...", "quality_score": 0.85}
  ],
  "created_at": "2024-12-29T10:00:00Z",
  "updated_at": "2024-12-29T10:15:00Z"
}
```

#### `DELETE /session/{session_id}`

Manually invalidate a session.

#### `GET /health`

Health check endpoint.

---

## Agents

### 1. Intent Classifier Agent

**Purpose:** Classify user intent using hybrid pattern + LLM approach.

**Model:** Claude 3 Haiku (for LLM stage only)

**Location:** `backend/app/agents/intent/`

#### Classification Pipeline

| Stage | Method | Coverage | Cost |
|-------|--------|----------|------|
| 1. Fast-path | Regex patterns | ~50% | Free |
| 2. Keyword | Set matching | ~30% | Free |
| 3. LLM | Claude Haiku | ~20% | ~$0.001 |
| 4. Validation | Rule-based | 100% | Free |

#### Intent Types

| Intent | Description | Example |
|--------|-------------|---------|
| `chat` | General conversation | "Hello!", "What can you do?" |
| `research` | Medical literature query | "What causes diabetes?" |
| `followup_research` | Question about prior results | "Compare these studies" |
| `utility` | Reformat existing output | "Make it shorter" |

#### Keyword Sets

```python
MEDICAL_KEYWORDS = {
    "diabetes", "cancer", "treatment", "drug", "symptom",
    "disease", "therapy", "medication", "clinical", "trial",
    # ... 100+ medical/scientific terms
}

SYSTEM_KEYWORDS = {
    "who are you", "what can you do", "how does this work",
    "are you a bot", "can i chat", "your name",
    # ... system/meta questions
}

FOLLOWUP_KEYWORDS = {
    "these studies", "those papers", "compare them",
    "first study", "tell me more", "which one",
    # ... reference indicators
}

UTILITY_KEYWORDS = {
    "make it shorter", "bullet points", "simplify",
    "key points only", "summarize it",
    # ... reformatting requests
}
```

---

### 2. Chat Agent

**Purpose:** Handle general conversation and system questions.

**Model:** Claude 3 Haiku

**Location:** `backend/app/agents/chat/`

#### Features

- **Canned Responses**: Instant replies for greetings, thanks, capability questions
- **LLM Fallback**: Nuanced conversation for complex chat messages
- **Personality**: Friendly, helpful AESOP persona

#### Canned Response Examples

| User Message | Bot Response |
|--------------|--------------|
| "Hello!" | "Hello! I'm AESOP, your biomedical literature review assistant..." |
| "Thanks!" | "You're welcome! Let me know if you have any other research questions." |
| "What can you do?" | "I'm AESOP, an AI-powered biomedical literature review assistant..." |
| "Goodbye!" | "Goodbye! Feel free to come back anytime..." |

---

### 3. Utility Agent

**Purpose:** Transform and reformat existing output.

**Model:** Claude 3 Haiku

**Location:** `backend/app/agents/utility/`

#### Supported Transformations

| Request | Action |
|---------|--------|
| "Make it shorter" | Condense to key points |
| "Bullet points" | Convert to bulleted list |
| "Simplify" | Use simpler language |
| "Just the conclusion" | Extract conclusion only |
| "Table format" | Organize as table |

---

### 4. Router Agent

**Purpose:** Intelligent query classification for research queries.

**Model:** Claude 3 Haiku (for ambiguous cases)

**Location:** `backend/app/agents/router/`

#### Multi-Signal Routing

| Signal | Description | Detection |
|--------|-------------|-----------|
| Deictic Markers | "these studies", "those results" | Regex |
| Explicit References | "first paper", "PMID 12345" | Regex |
| Keyword Overlap | Shared medical terms | Jaccard similarity |
| Query Type | Clarification vs new question | Pattern matching |

#### Routes

| Route | Trigger | Execution | Cost |
|-------|---------|-----------|------|
| **Route A: Full Graph** | New topic | Scout → Critic → Synthesizer | High |
| **Route B: Augmented** | Related topic, needs evidence | Scout → Merge → Synthesizer | Medium |
| **Route C: Context Q&A** | Question about existing results | Direct LLM with cache | Low |

---

### 5. Scout Agent

**Purpose:** Query expansion and literature retrieval from PubMed.

**Model:** Claude 3 Haiku

**Location:** `backend/app/agents/scout/`

#### Workflow

```
User Query → Query Expansion (LLM) → PubMed Search → Fetch Abstracts → Papers
```

#### Features

- **Robust JSON Parsing**: Handles malformed LLM output gracefully
- **Fallback Extraction**: Multiple strategies to extract queries
- **Error Tolerance**: Partial results accepted, never crashes

---

### 6. Critic Agent

**Purpose:** Evidence evaluation and CRAG decision-making.

**Model:** Amazon Nova Pro

**Location:** `backend/app/agents/critic/`

#### Grading Rubric

| Metric | Range | Description |
|--------|-------|-------------|
| `relevance_score` | 0.0 - 1.0 | Topical relevance |
| `methodology_score` | 0.0 - 1.0 | Methodological rigor |
| `sample_size_adequate` | bool | Sufficient sample |
| `study_type` | string | RCT, Cohort, etc. |
| `recommendation` | enum | KEEP, DISCARD, NEEDS_MORE |

#### GRADE-Inspired Study Priors

```python
STUDY_TYPE_PRIORS = {
    "meta-analysis": 0.85,
    "systematic review": 0.80,
    "randomized controlled trial": 0.70,
    "cohort study": 0.55,
    "case-control study": 0.50,
    "case series": 0.30,
    "expert opinion": 0.20,
}
```

#### Throttling Protection

- **Retry with Backoff**: Exponential backoff on AWS throttling
- **Inter-paper Delay**: 500ms between paper evaluations
- **Max Retries**: 5 attempts before failure

---

### 7. Synthesizer Agent

**Purpose:** Generate structured literature review.

**Model:** Amazon Nova Pro

**Location:** `backend/app/agents/synthesizer/`

#### Output Structure

```markdown
## 1. Background
[Context and importance]

## 2. Summary of High-Quality Evidence
[Papers with quality ≥ 0.7]

## 3. Summary of Lower-Quality Evidence
[Papers with quality < 0.7]

## 4. Limitations
[Gaps, biases, concerns]

## 5. Conclusion
[Evidence-based answer]
```

---

### 8. Context Q&A Agent

**Purpose:** Answer questions using cached papers.

**Model:** Claude 3 Haiku

**Location:** `backend/app/agents/context_qa/`

#### When Used

- Route C queries (followup_research intent)
- Questions about existing search results
- Comparisons, clarifications, explanations

---

## Memory System

AESOP implements a **two-layer memory architecture**:

### Layer 1: Session Cache (Redis)

**Purpose:** Enable multi-turn conversations within a session.

**TTL:** 60 minutes

**Key Format:** `aesop:session:{session_id}`

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

**Usage:**
- Intent classifier checks for existing session
- Router uses session for similarity comparison
- Context Q&A uses cached papers
- Utility transforms cached synthesis

---

### Layer 2: Long-Term Memory (PostgreSQL + pgvector)

**Purpose:** Persistent learning across sessions.

#### Schema

```sql
CREATE TABLE critic_acceptance_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    research_query TEXT NOT NULL,
    query_hash TEXT GENERATED ALWAYS AS (md5(lower(trim(research_query)))) STORED,
    query_embedding VECTOR(1536) NOT NULL,
    pmid TEXT NOT NULL,
    study_type TEXT,
    publication_year INT,
    relevance_score FLOAT,
    methodology_score FLOAT,
    quality_score FLOAT,
    iteration INT NOT NULL,
    accepted_at TIMESTAMP DEFAULT now()
);
```

#### Memory Influence

```
Memory DOES:
  ✓ Lower quality threshold (max 0.15 reduction)
  ✓ Enable faster convergence
  ✓ Remember high-quality evidence

Memory DOES NOT:
  ✗ Skip paper evaluation
  ✗ Force decisions
  ✗ Change individual grades
```

---

## Project Structure

```
aesop/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── intent/                 # Intent classification
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Hybrid classifier
│   │   │   │   ├── node.py             # LangGraph node
│   │   │   │   └── prompts.py          # LLM prompts
│   │   │   ├── chat/                   # Chat agent
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Chat handler
│   │   │   │   ├── node.py             # LangGraph node
│   │   │   │   └── prompts.py          # Canned responses
│   │   │   ├── utility/                # Utility agent
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py            # Reformatter
│   │   │   │   └── node.py             # LangGraph node
│   │   │   ├── router/
│   │   │   │   ├── agent.py            # Multi-signal router
│   │   │   │   ├── node.py
│   │   │   │   └── prompts.py
│   │   │   ├── scout/
│   │   │   │   ├── agent.py            # Query expansion + PubMed
│   │   │   │   ├── prompts.py
│   │   │   │   └── tools.py            # PubMed API
│   │   │   ├── critic/
│   │   │   │   ├── agent.py            # CRAG logic + retry
│   │   │   │   ├── memory.py           # pgvector store
│   │   │   │   ├── node.py
│   │   │   │   ├── prompts.py
│   │   │   │   ├── rubric.py           # GRADE thresholds
│   │   │   │   └── schemas.py
│   │   │   ├── synthesizer/
│   │   │   │   ├── agent.py
│   │   │   │   ├── prompts.py
│   │   │   │   └── schemas.py
│   │   │   ├── context_qa/
│   │   │   │   ├── agent.py
│   │   │   │   └── node.py
│   │   │   ├── graph.py                # Original CRAG graph
│   │   │   ├── orchestrator_graph.py   # Full orchestrator with intent
│   │   │   └── state.py                # AgentState, OrchestratorState
│   │   ├── schemas/
│   │   │   └── session.py              # SessionContext, RouterDecision
│   │   ├── services/
│   │   │   └── session.py              # Redis SessionService
│   │   ├── embeddings/
│   │   │   └── bedrock.py              # Titan embeddings
│   │   ├── main.py                     # FastAPI with /chat endpoint
│   │   ├── tasks.py                    # Task runners
│   │   └── logging.py
│   ├── tests/
│   │   └── test_intent_classifier.py   # Intent classifier tests
│   ├── db/
│   │   └── migrations/
│   ├── Dockerfile
│   └── pyproject.toml
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://aesop:aesop_pass@postgres:5432/aesop_db` |
| `REDIS_URL` | Redis connection | `redis://redis:6379/0` |
| `AWS_ACCESS_KEY_ID` | AWS credentials | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith | `false` |

### CRAG Tuning Parameters

```python
# Evidence thresholds
MIN_RELEVANCE_TO_KEEP = 0.45
MIN_METHODOLOGY_TO_KEEP = 0.50
MIN_AVG_QUALITY_FOR_SUFFICIENT = 0.60

# Iteration decay
CONFIDENCE_DECAY_RATE = 0.07
MIN_CONFIDENCE_FLOOR = 0.45

# Memory bounds
MAX_MEMORY_BOOST = 0.15
```

---

## Performance

### Latency by Route

| Route | Typical Duration | LLM Calls |
|-------|------------------|-----------|
| Chat | 50-500ms | 0-1 |
| Utility | 500-2000ms | 1 |
| Context Q&A | 1-3s | 2 |
| Augmented | 10-25s | 3-5 |
| Full Graph | 15-45s | 5-25 |

### Cost by Route

| Route | Est. Cost |
|-------|-----------|
| Chat (canned) | $0.00 |
| Chat (LLM) | $0.001 |
| Utility | $0.002 |
| Context Q&A | $0.01-0.02 |
| Augmented | $0.03-0.06 |
| Full Graph | $0.05-0.10 |

---

## Roadmap

### Completed ✅

- [x] Multi-agent CRAG architecture
- [x] pgvector long-term memory
- [x] Multi-turn session support
- [x] Intelligent 3-route routing
- [x] AWS Bedrock integration
- [x] Intent classification (hybrid pattern + LLM)
- [x] Chat agent with canned responses
- [x] Utility agent for reformatting
- [x] Throttling protection with retry

### In Progress 🚧

- [ ] Chat memory (conversation history)
- [ ] Neo4j citation graph integration
- [ ] Streaming responses

### Planned 📋

- [ ] Full-text PDF retrieval
- [ ] User feedback loop
- [ ] Memory pruning policies
- [ ] Evaluation benchmarks
- [ ] Web UI

---

## Development

### Running Tests

```bash
docker exec -it aesop_backend pytest tests/ -v
```

### View Logs

```bash
docker-compose logs -f backend
```

### Debug Redis Sessions

```bash
docker exec -it aesop_redis redis-cli
> KEYS aesop:session:*
> GET aesop:session:<session_id>
```

### Debug PostgreSQL Memory

```bash
docker exec -it aesop_postgres psql -U aesop -d aesop_db
> SELECT COUNT(*) FROM critic_acceptance_memory;
```

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