
# Critic Agent — CRAG + pgvector Memory

This folder contains the **Critic agent** of the AESOP system — a **CRAG-based (Corrective Retrieval-Augmented Generation) agent** designed for **biomedical evidence evaluation** with **persistent vector memory** using **PostgreSQL + pgvector**.

The Critic is the **decision-making authority** in the multi-agent system.  
It evaluates retrieved scientific papers, determines whether the evidence is sufficient, and controls whether the system should **retrieve more evidence or converge**.

This document explains **what was built**, **why it was built this way**, and **how it works in detail**.

---

## 1. High-Level Role of the Critic Agent

The Critic agent acts as a **senior biomedical reviewer** in the agentic pipeline.

It is responsible for:

- Evaluating **scientific abstracts** for:
  - Relevance to the research question
  - Methodological rigor
- Applying **CRAG logic** to decide:
  - `"sufficient"` → stop retrieval and synthesize
  - `"retrieve_more"` → request more evidence
- Persisting **high-confidence evidence** into a **long-term vector memory**
- Using **past accepted evidence** as a *bounded prior* in future decisions

The Critic **never retrieves papers** and **never generates answers**.  
It only **judges evidence and controls flow**.

---

## 2. Why CRAG (Corrective RAG) Is Used

Traditional RAG systems retrieve documents once and generate an answer immediately.  
This is **unsafe for biomedical domains**, where evidence quality varies widely.

CRAG introduces **explicit correction loops**:

1. Retrieve evidence
2. Critically evaluate evidence
3. Decide whether evidence is sufficient
4. If not sufficient → retrieve more
5. Repeat until convergence or stopping condition

In AESOP:

- The **Critic** implements CRAG logic
- The **Scout** retrieves evidence
- The **Synthesizer** generates the final answer

This separation prevents hallucination and enforces **evidence-driven reasoning**.

---

## 3. Critic Agent Architecture

### Files in This Folder

| File | Purpose |
|---|---|
| `agent.py` | Main Critic agent logic (grading, CRAG decision, memory writes) |
| `node.py` | LangGraph node wrapper (state-safe execution) |
| `memory.py` | pgvector-backed long-term memory |
| `schemas.py` | Strict Pydantic schemas for LLM outputs |
| `rubric.py` | Evidence thresholds and CRAG parameters |
| `prompts.py` | Strict biomedical evaluation prompts |

---

## 4. Evidence Grading (LLM + Guardrails)

### What the Critic Grades

The Critic grades **abstracts only**, not full papers.

For each paper it assigns:

- `relevance_score` ∈ [0,1]
- `methodology_score` ∈ [0,1]
- `study_type` (RCT, cohort, review, etc.)
- `recommendation`:
  - `KEEP`
  - `NEEDS_MORE`
  - `DISCARD`

### Why Strict JSON Enforcement Is Used

Biomedical LLM outputs **must be machine-verifiable**.

The Critic enforces:

- JSON-only output
- No reasoning text
- Pydantic validation
- Score clamping
- Never trusting identifiers from the LLM

If any rule is violated → the grading fails loudly.

This prevents:
- Prompt leakage
- Hallucinated metadata
- Silent corruption

---

## 5. CRAG Global Decision Logic

After grading all papers, the Critic computes **aggregate metrics**:

- Keep ratio
- Discard ratio
- Needs-more ratio
- Average quality score

It then applies CRAG rules:

- If too many papers are weak → `retrieve_more`
- If quality is high enough → `sufficient`
- If uncertainty remains → `retrieve_more`

CRAG thresholds **decay slightly across iterations**, allowing convergence without premature acceptance.

---

## 6. Why We Added Long-Term Memory

Without memory, CRAG systems are **stateless**:
- The same query must re-discover the same evidence
- Convergence cost repeats
- No learning across runs

To solve this, we introduced **persistent Critic memory**.

### Key Design Principle

> **Memory influences thresholds, never decisions.**

Memory is used as a **prior**, not a shortcut.

---

## 7. pgvector-Backed Memory Design

### Storage Technology

- PostgreSQL 16
- pgvector extension
- Amazon Titan embeddings (1536-dimensional)

### What Is Stored

Only **high-confidence accepted evidence** is stored:

- Research query
- Query embedding
- Paper identifiers
- Quality scores
- Timestamp

`DISCARD` papers are never stored.  
`NEEDS_MORE` papers are intentionally excluded from acceptance memory.

This prevents memory poisoning.

---

## 8. Memory Schema (Conceptual)

```sql
critic_acceptance_memory
├── research_query
├── query_hash           -- exact-match fast path
├── query_embedding      -- vector(1536)
├── pmid
├── study_type
├── quality_score
├── iteration
├── accepted_at
````

### Indexes

* `query_hash` → fast exact match
* `ivfflat` cosine index → scalable vector search

---

## 9. How Memory Is Queried

When a new query arrives:

1. **Exact-match fast path**

   * If the same query was seen before → reuse memory rows
2. **Vector similarity fallback**

   * Cosine similarity ≥ 0.75
   * Top-K capped results

A **Memory Confidence Score (MCS)** is computed using:

* Similarity
* Quality
* Time decay

This score is **hard-capped (≤ 0.15)**.

---

## 10. How Memory Influences CRAG (Safely)

Memory **does NOT**:

* Change paper grades
* Force `"sufficient"`
* Skip evaluation
* Reuse old papers

Memory **only**:

* Slightly lowers the effective quality threshold

This means:

* Strong past evidence → faster convergence
* Weak or unrelated queries → no effect

Medical safety is preserved.

---

## 11. Why Many Bugs Were Encountered (And Why That’s Good)

During development, several real-world issues surfaced:

* pgvector type casting (`vector` vs `numeric[]`)
* Timezone-aware vs naive timestamps
* Decimal vs float arithmetic
* PubMed API strictness
* Network failures inside Docker
* Partial data corruption

Each issue was **fixed explicitly**, resulting in:

* Fully defensive Scout tools
* Fault-tolerant retrieval
* Safe memory math
* Production-grade robustness

These issues validate that the system was tested under **realistic conditions**, not idealized demos.

---

## 12. What This System Achieves

This Critic agent implements:

* True CRAG (not pseudo-RAG)
* Evidence-based biomedical evaluation
* Persistent learning across runs
* Safe, bounded memory influence
* Deterministic, inspectable behavior

It is suitable for:

* Biomedical research assistance
* Literature review automation
* Clinical decision support (non-diagnostic)
* Agentic AI research portfolios

---

## 13. Key Takeaway

This Critic agent is **not a text generator**.

It is:

> A **scientific reasoning controller** that decides *when enough evidence is enough* — and remembers that decision responsibly.

---

## 14. Next Extensions (Planned / Possible)

* Exploratory memory for `NEEDS_MORE` papers
* Memory aging / pruning policies
* Scout query expansion using memory
* Citation graph integration (Neo4j)
* Formal evaluation benchmarks

---

