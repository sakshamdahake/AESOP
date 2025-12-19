import requests
from typing import List

from app.agents.state import Paper

# PUBMED api base ref: https://www.ncbi.nlm.nih.gov/books/NBK25497/
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search(query: str, retmax: int = 10) -> List[str]:
    response = requests.get(
        f"{PUBMED_BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["esearchresult"]["idlist"]


def pubmed_fetch(pmids: List[str]) -> List[Paper]:
    if not pmids:
        return []

    response = requests.get(
        f"{PUBMED_BASE}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
        timeout=10,
    )
    response.raise_for_status()

    # MVP placeholder parsing
    papers: List[Paper] = []
    for pmid in pmids:
        papers.append(
            Paper(
                pmid=pmid,
                title="Title parsing TODO",
                abstract="Abstract parsing TODO",
            )
        )

    return papers
