"""
Bedrock embeddings with cosine similarity support.
"""

from langchain_aws import BedrockEmbeddings
import math

_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1",
)


def embed_query(text: str) -> list[float]:
    """Generate embedding for text using Titan."""
    return _embeddings.embed_query(text)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns value between -1 and 1 (typically 0 to 1 for text embeddings).
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)