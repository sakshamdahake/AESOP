"""
Balanced grading rubric for CRAG-based evidence evaluation.

Goal:
- Maintain scientific rigor
- Allow CRAG to converge with realistic evidence
"""

# -----------------------------
# Evidence hierarchy priors
# (Softened GRADE-style)
# -----------------------------

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

# -----------------------------
# Sample size expectations
# (unchanged — still reasonable)
# -----------------------------

SAMPLE_SIZE_THRESHOLDS = {
    "meta-analysis": 0,
    "systematic review": 0,
    "randomized controlled trial": 80,   # lowered from 100
    "rct": 80,
    "cohort study": 200,                 # lowered from 300
    "case-control study": 150,
    "cross-sectional study": 200,
    "case series": 15,
    "case study": 8,
}

# -----------------------------
# Base decision thresholds
# -----------------------------

# Allow papers to survive initial screening
MIN_RELEVANCE_TO_KEEP = 0.45      # was 0.60
MIN_METHODOLOGY_TO_KEEP = 0.50    # was 0.60

# -----------------------------
# CRAG-level thresholds
# -----------------------------

# Allow convergence with good-but-not-perfect evidence
MIN_AVG_QUALITY_FOR_SUFFICIENT = 0.60   # was 0.70
MAX_DISCARD_RATIO = 0.55                # was 0.40

# -----------------------------
# Confidence decay (per iteration)
# -----------------------------

CONFIDENCE_DECAY_RATE = 0.07      # slightly faster decay
MIN_CONFIDENCE_FLOOR = 0.45       # was 0.50
