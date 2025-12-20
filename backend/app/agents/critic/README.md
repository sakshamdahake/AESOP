# 🧠 Critic Agent — CRAG-Based Scientific Evidence Evaluator

## Overview

The **Critic Agent** is a core component of the **Aesop** system, responsible for **evaluating the quality of retrieved scientific literature** and deciding whether the current evidence is **sufficient** or whether **additional retrieval is required**.

This agent implements a **Corrective Retrieval-Augmented Generation (CRAG)** pattern tailored for **medical and scientific systematic reviews**, where *rigor, conservatism, and transparency* are critical.

The Critic Agent is **not a summarizer**.
It acts as a **methodological reviewer**, similar to a human expert conducting a systematic review.

---

## Why a Critic Agent Is Necessary

In scientific domains (especially medicine):

* Not all papers are equal
* Abstracts are often incomplete
* High-quality study design matters more than keywords
* Evidence must converge before synthesis

A naive RAG system will:

* hallucinate certainty
* over-trust weak evidence
* fail silently

The Critic Agent prevents this by **grading**, **rejecting**, and **forcing re-retrieval** until evidence quality is acceptable.

---

## Theoretical Foundation

### 1. Corrective RAG (CRAG)

CRAG introduces a feedback loop:

```
Retrieve → Evaluate → Decide
           ↑           ↓
        Re-retrieve if insufficient
```

The Critic Agent is the **decision-maker** in this loop.

---

### 2. Evidence-Based Medicine (EBM)

The Critic Agent encodes core EBM principles:

* Study design hierarchy (GRADE)
* Sample size adequacy
* Methodological transparency
* Conservative decision-making

This prevents the system from being fooled by:

* buzzwords
* underpowered studies
* anecdotal evidence

---

### 3. Reviewer Psychology (Human-Inspired)

Real reviewers:

* Start strict
* Relax slightly if evidence converges
* Look for consensus
* Learn from prior acceptances

These behaviors are explicitly modeled.

---

## High-Level Decision Criteria

For each retrieved abstract, the Critic Agent produces a structured evaluation:

| Field                  | Meaning                                          |
| ---------------------- | ------------------------------------------------ |
| `relevance_score`      | How well the paper matches the research question |
| `methodology_score`    | Rigor of study design and reporting              |
| `sample_size_adequate` | Whether the study is statistically plausible     |
| `study_type`           | RCT, cohort, case series, etc.                   |
| `recommendation`       | `keep`, `discard`, or `needs_more`               |

The **global decision** is either:

* `sufficient`
* `retrieve_more`

---

## Scientific Enhancements Implemented

### 1️⃣ Evidence-Based Study Priors

Different study designs have different *baseline credibility*.

Example:

* RCTs should never score as poorly as case series
* Meta-analyses should start strong

**Implementation:**

```python
STUDY_TYPE_PRIORS = {
    "randomized controlled trial": 0.65,
    "cohort study": 0.45,
    "case series": 0.20,
}
```

The Critic applies:

```python
methodology_score = max(llm_score, prior)
```

---

### 2️⃣ Reviewer Confidence Decay

The agent relaxes strictness slightly over iterations.

| Iteration | Required Avg Quality |
| --------- | -------------------- |
| 0         | 0.70                 |
| 1         | 0.65                 |
| 2         | 0.60                 |
| 3         | 0.55                 |

This models realistic reviewer behavior without sacrificing rigor.

---

### 3️⃣ Disagreement-Aware CRAG

Instead of averaging blindly, the agent checks **consensus**.

Rules:

* ≥60% `KEEP` → sufficient
* ≥50% `DISCARD` → retrieve more
* Mixed signals → retrieve more

This prevents:

* One weak paper vetoing strong evidence
* One strong paper overriding poor consensus

---

### 4️⃣ Learning From Past Acceptances

The Critic Agent maintains a lightweight **acceptance memory**:

* Which study types were accepted
* At what quality
* At which iteration

This allows future decisions to converge faster.

> ⚠️ Currently in-memory only (MVP).
> Designed to be persisted later.

---

## Code Structure

```
backend/app/agents/critic/
├── agent.py      # Core CRAG logic
├── rubric.py     # Scientific thresholds & priors
├── schemas.py    # Pydantic output contracts
├── prompts.py    # LLM instructions
├── learning.py   # Lightweight learning memory
└── README.md     # This document
```

---

## How Theory Maps to Code

### `rubric.py`

Defines:

* Study-type priors
* Thresholds
* Confidence decay parameters

This file contains **no LLM logic** — only science and policy.

---

### `agent.py`

Key responsibilities:

1. Call LLM (`ainvoke`)
2. Enforce strict JSON schema
3. Apply scientific priors
4. Aggregate grades
5. Decide CRAG action
6. Learn from acceptances

The Critic Agent is **async-safe** and **LangGraph-compatible**.

---

### `learning.py`

Stores acceptance history:

```python
ACCEPTANCE_MEMORY[study_type] → [{quality, iteration}]
```

Used to adapt behavior over time.

---

## Running the Critic Agent (Standalone)

Inside the Docker container:

```bash
python -m app.agents.test_run
```

This runs:

* isolated Critic evaluation
* CRAG loop simulation
* no LangGraph
* no PubMed dependency

---

## What “Correct Behavior” Looks Like

✔ Weak studies are rejected
✔ Borderline studies trigger re-retrieval
✔ Strong evidence converges
✔ RCTs are not unfairly penalized
✔ Case series do not dominate decisions

If the system **refuses to accept evidence too easily**, that is a *feature*, not a bug.

---

## Future Extensions

Planned enhancements:

* Persist learning to Postgres
* Visualization of CRAG convergence
* Reviewer drift detection
* Multi-reviewer ensembles
* Citation anchoring

---

## Design Philosophy

> *In medicine, uncertainty should trigger more evidence — not confidence.*

The Critic Agent enforces this principle throughout the system.

---

## Summary

The Critic Agent is:

* Conservative by design
* Scientifically grounded
* Self-correcting
* Self-improving
* Production-ready

It transforms a simple RAG pipeline into a **trustworthy autonomous reviewer**.

