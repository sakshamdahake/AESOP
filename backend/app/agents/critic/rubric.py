# Study type hierarchy (higher = stronger evidence)
STUDY_TYPE_SCORES = {
    "meta-analysis": 1.0,
    "systematic review": 0.95,
    "randomized controlled trial": 0.9,
    "rct": 0.9,
    "cohort study": 0.75,
    "case-control study": 0.65,
    "cross-sectional study": 0.6,
    "case study": 0.4,
    "case series": 0.4,
    "expert opinion": 0.2,
}

# Minimum sample size expectations by study type
SAMPLE_SIZE_THRESHOLDS = {
    "meta-analysis": 0,     # abstract-level; assume aggregation
    "systematic review": 0,
    "randomized controlled trial": 100,
    "rct": 100,
    "cohort study": 300,
    "case-control study": 200,
    "cross-sectional study": 300,
    "case study": 10,
    "case series": 20,
}

# Decision thresholds
MIN_RELEVANCE_TO_KEEP = 0.6
MIN_METHODOLOGY_TO_KEEP = 0.6

MIN_AVG_QUALITY_FOR_SUFFICIENT = 0.7
MAX_DISCARD_RATIO = 0.4
