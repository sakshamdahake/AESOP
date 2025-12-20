"""
Scientifically grounded grading rubric for CRAG-based evidence evaluation.
"""

# -----------------------------
# Evidence hierarchy priors
# (GRADE-style)
# -----------------------------

STUDY_TYPE_PRIORS = {
    "meta-analysis": 0.75,
    "systematic review": 0.70,
    "randomized controlled trial": 0.65,
    "rct": 0.65,
    "cohort study": 0.45,
    "case-control study": 0.40,
    "cross-sectional study": 0.35,
    "case series": 0.20,
    "case study": 0.15,
    "expert opinion": 0.10,
}

# -----------------------------
# Sample size expectations
# -----------------------------

SAMPLE_SIZE_THRESHOLDS = {
    "meta-analysis": 0,
    "systematic review": 0,
    "randomized controlled trial": 100,
    "rct": 100,
    "cohort study": 300,
    "case-control study": 200,
    "cross-sectional study": 300,
    "case series": 20,
    "case study": 10,
}

# -----------------------------
# Base decision thresholds
# -----------------------------

MIN_RELEVANCE_TO_KEEP = 0.60
MIN_METHODOLOGY_TO_KEEP = 0.60

# CRAG-level thresholds
MIN_AVG_QUALITY_FOR_SUFFICIENT = 0.70
MAX_DISCARD_RATIO = 0.40

# Confidence decay (per iteration)
CONFIDENCE_DECAY_RATE = 0.05
MIN_CONFIDENCE_FLOOR = 0.50
