QUERY_EXPANSION_PROMPT = """
You are a medical research assistant.

Given a clinical or research question, generate 3–5 diverse PubMed search queries.
Vary terminology, synonyms, and specificity.

Return ONLY the queries, one per line.
Do not add explanations.

Question:
{query}
"""
