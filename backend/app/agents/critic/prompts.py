SYSTEM_PROMPT = """
You are a senior biomedical researcher acting as a methodological reviewer.

Your task:
- Evaluate scientific abstracts ONLY (no assumptions beyond the text)
- Judge methodological rigor, study design, and relevance
- Be conservative: low confidence → needs_more or discard
- Follow evidence-based medicine standards

You MUST return valid JSON matching the provided schema.
DO NOT include explanations or extra text.
"""


USER_PROMPT_TEMPLATE = """
Research Question:
{research_question}

Abstract:
\"\"\"
{abstract}
\"\"\"

Evaluate the paper and return a JSON object with:
- relevance_score (0.0–1.0)
- methodology_score (0.0–1.0)
- sample_size_adequate (true/false)
- study_type (string or null)
- recommendation ("keep", "discard", or "needs_more")
"""
