from langchain_aws import BedrockEmbeddings

_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1",
)

def embed_query(text: str) -> list[float]:
    return _embeddings.embed_query(text)
