# Project Design Document: Aesop
**The Autonomous Evidence Synthesis & Observation Platform**

---

## 1. Executive Summary
**Aesop** is a distributed, multi-agent system designed to automate the creation of "Living Systematic Reviews" for medical professionals. Unlike static literature reviews which are outdated shortly after publication, Aesop dynamically scrapes, grades, analyzes, and synthesizes medical literature in real-time. It utilizes a **Self-Improving Agentic RAG** architecture to refine its search and grading logic based on user feedback and internal consistency checks.

## 2. Detailed Problem Statement

### The Core Conflict: The Velocity of Knowledge vs. Human Capacity
* **Knowledge Decay:** Medical knowledge is estimated to double every 73 days. A systematic review published in 2023 is effectively "stale" by 2024.
* **The "UpToDate" Gap:** Tools like *UpToDate* rely on manual curation by experts. This limits their coverage to common conditions. Rare diseases, novel drug interactions, or niche genomic markers often lack synthesized evidence.
* **The Hallucination Risk:** Generic LLMs (ChatGPT) cannot be trusted for medical synthesis because they prioritize *plausibility* over *accuracy* and often cite non-existent papers.

### The Solution Gap
Current RAG systems retrieve the "top k" chunks based on semantic similarity. In medicine, this is dangerous. A paper might be "semantically similar" but **methodologically flawed** (e.g., sample size n=5). We need a system that judges **quality**, not just relevance.

## 3. Methodology: Self-Improving Agentic RAG
Aesop implements the **CRAG (Corrective RAG)** and **Self-RAG** patterns.

### The "Think-Act-Observe" Loop
1.  **Retrieval (Act):** Agents fetch abstracts from trusted sources.
2.  **Grading (Think):** A dedicated "Critic Agent" evaluates the *methodology* of the paper (not just the content).
3.  **Correction (Observe):**
    * *If retrieval is poor:* The system rewrites the search query (e.g., expands acronyms, changes "cancer" to "carcinoma").
    * *If evidence is contradictory:* The system triggers a "Conflict Resolution" workflow to find meta-analyses.
4.  **Learning:** User feedback ("This paper is irrelevant") is stored in a **Preference Database (DPO)** to fine-tune the grading agent's prompt for future queries.

## 4. Data Sources
The system will connect to the following APIs via a distributed scraping layer:
1.  **PubMed / MEDLINE (NCBI E-utilities API):** Primary source for peer-reviewed literature.
2.  **ClinicalTrials.gov API:** For ongoing or recently completed trials (often unpublished).
3.  **bioRxiv / medRxiv (RSS/API):** For pre-print (cutting edge, but requires stricter grading).
4.  **Semantic Scholar API:** For citation graphs (checking if a paper has been retracted or heavily refuted).

## 5. End-to-End System Architecture

### High-Level Mental Model
`[User Query]` -> `[Orchestrator Agent]` -> `[Distributed Search Swarm]` -> `[Parallel Grading Agents]` -> `[Synthesis Agent]` -> `[Validation Agent]` -> `[Final Report]`

### Component 1: The Distributed Ingestion Layer (The "Distributed" Constraint)
* **Why Distributed?** Scraping and processing thousands of abstracts, and then parsing hundreds of full-text PDFs, is CPU-intensive and hits API rate limits.
* **Implementation:**
    * **Message Queue:** Redis.
    * **Workers:** Python workers (Celery) running on **AWS Lambda** (for bursty scraping) or **EC2 Spot Instances** (for heavy PDF parsing).
    * **Function:** Workers autonomously manage their own IP rotation and rate limiting.

### Component 2: The Multi-Agent Core (The "Agentic" Constraint)
We utilize a **LangGraph** framework to orchestrate specific roles.

| Agent Role | Model Size | Responsibility |
| :--- | :--- | :--- |
| **Scout (Searcher)** | Small (Llama-3-8B) | Generates keyword permutations. Parses API responses. |
| **Gatekeeper (Triage)** | Small (Llama-3-8B) | "Is this relevant?" Reads abstracts. Discards junk immediately to save cost. |
| **The Critic (Grader)** | Medium (GPT-4o-mini) | "Is this rigorous?" Extracts P-values, Sample Size (N), Conflict of Interest. Scores paper 0-10. |
| **Synthesizer** | Large (Claude 3.5 Sonnet) | Writes the review. Groups findings by theme. Highlights consensus vs. conflict. |
| **The Auditor** | Medium (GPT-4o) | **Hallucination Check.** Verifies every claim in the Synthesizer's draft against the raw text chunks. |

### Component 3: Storage & Memory
* **Vector Database (Pinecone/Weaviate/pgvector):** Stores embeddings of the *useful* chunks (not the junk).
* **Graph Database (Neo4j):** Stores relationships. `(Paper A) --[REFUTES]--> (Paper B)`. This allows the system to detect scientific debates.
* **PostgreSQL:** Stores user feedback and "Grading History" for the self-improvement loop.

### Component 4: Cloud Infrastructure (The "Cloud-Hosted" Constraint)
* **Compute:** Kubernetes Cluster (EKS) or ECS.
    * *Node Pool A:* General purpose (API, Orchestrator).
    * *Node Pool B:* GPU-enabled (Local LLMs for privacy/speed).
* **Frontend:** Vercel / React (The "End-to-End" user interface).

## 6. Implementation Plan

### Phase 1: The Foundation (Weeks 1-4)
* **Goal:** A script that takes a query, hits PubMed, and summarizes 5 abstracts.
* **Tasks:** Set up the **Scout Agent** and the **Synthesizer Agent**.
* **Tech:** Python, LangChain, OpenAI API.

### Phase 2: The Critic & The Pipeline (Weeks 5-8)
* **Goal:** Filter out bad papers.
* **Tasks:** Build the **Critic Agent** with strict prompt engineering ("Ignore papers with N < 30"). Implement the **Distributed Queue** (Celery) to handle 100+ papers.
* **Tech:** Redis, Celery, Docker.

### Phase 3: The "Self-Improving" Loop (Weeks 9-12)
* **Goal:** The system fixes its own search queries.
* **Tasks:** Implement **Corrective RAG**. If the Searcher finds 0 relevant papers, it triggers a "Query Rewrite" function and searches again.
* **Tech:** LangGraph (Conditional Edges).

### Phase 4: The Interface & Deployment (Weeks 13-16)
* **Goal:** A usable product.
* **Tasks:** Build the React UI. Display the "Reasoning Trace" (Show users *why* a paper was rejected). Deploy to AWS.
* **Outcome:** MVP Release.

## 7. Risk Analysis & Mitigation

**Risk 1: The "Abstract Trap"**
* *Critique:* Agents often only read abstracts because PDFs are paywalled. Abstracts are marketing; real flaws are in the Methods section.
* *Mitigation:* The system attempts PDF ingestion. If full text is unavailable, the "Confidence Score" of that citation is penalized by 50%.

**Risk 2: Infinite Loops**
* *Critique:* In a self-correcting system, the agent might rewrite queries forever if no data exists.
* *Mitigation:* Strict **recursion limits** (max 3 retries). If no data is found, return a "Knowledge Gap" report.

**Risk 3: Cost Explosion**
* *Critique:* Running GPT-4 on 500 papers per query is cost-prohibitive.
* *Mitigation:* **Tiered Processing.** Use Llama-3-8B for triage/discarding junk. Use GPT-4/Claude only for final synthesis of the top 10-20 papers.