import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Iterable

from requests.exceptions import RequestException

from app.agents.state import Paper
from app.logging import logger

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# -----------------------------
# Utilities
# -----------------------------

def _safe_find_text(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is not None and elem.text:
        return elem.text.strip()
    return None


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    """Yield successive chunks of size `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# -----------------------------
# PubMed Search (ESearch)
# -----------------------------

def pubmed_search(query: str) -> List[str]:
    try:
        response = requests.get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 10,
                "tool": "aesop",
                "email": "dev@example.com",
            },
            timeout=10,
        )
        response.raise_for_status()

        return response.json().get("esearchresult", {}).get("idlist", [])

    except RequestException as e:
        logger.error(
            "PUBMED_SEARCH_FAILED",
            extra={"query": query, "error": str(e)},
        )
        return []  # ⬅️ NEVER crash Scout


# -----------------------------
# PubMed Fetch (EFetch) — HARDENED
# -----------------------------

def pubmed_fetch(pmids: List[str]) -> List[Paper]:
    """
    Fetch PubMed records safely.

    Design guarantees:
    - Chunked requests (EFetch is strict)
    - One bad PMID never crashes Scout
    - Partial results are allowed
    """

    if not pmids:
        return []

    papers: List[Paper] = []

    # 🔑 NCBI-safe batch size
    for batch in _chunked(pmids, size=3):
        try:
            response = requests.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "xml",
                    "tool": "aesop",
                    "email": "dev@example.com",
                },
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(
                    "PUBMED_EFETCH_BATCH_FAILED",
                    extra={
                        "pmids": batch,
                        "status": response.status_code,
                        "response": response.text[:200],
                    },
                )
                continue  # ⬅️ skip bad batch only

            root = ET.fromstring(response.text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    citation = article.find("MedlineCitation")
                    article_node = (
                        citation.find("Article") if citation is not None else None
                    )

                    pmid = (
                        _safe_find_text(citation.find("PMID"))
                        if citation is not None
                        else None
                    )

                    title = (
                        _safe_find_text(article_node.find("ArticleTitle"))
                        if article_node is not None
                        else None
                    )

                    # Abstract may have multiple parts
                    abstract_parts = []
                    if article_node is not None:
                        abstract_node = article_node.find("Abstract")
                        if abstract_node is not None:
                            for part in abstract_node.findall("AbstractText"):
                                if part.text:
                                    abstract_parts.append(part.text.strip())

                    abstract = " ".join(abstract_parts) if abstract_parts else None

                    journal = None
                    year = None

                    if article_node is not None:
                        journal_node = article_node.find("Journal")
                        if journal_node is not None:
                            journal = _safe_find_text(journal_node.find("Title"))

                    date_node = (
                        citation.find("DateCompleted")
                        if citation is not None
                        else None
                    )
                    if date_node is not None:
                        year_text = _safe_find_text(date_node.find("Year"))
                        if year_text and year_text.isdigit():
                            year = int(year_text)

                    # Only accept papers with real abstracts
                    if pmid and title and abstract:
                        papers.append(
                            Paper(
                                pmid=pmid,
                                title=title,
                                abstract=abstract,
                                publication_year=year,
                                journal=journal,
                            )
                        )

                except Exception as e:
                    logger.warning(
                        "PUBMED_ARTICLE_PARSE_FAILED",
                        extra={"pmid": pmid, "error": str(e)},
                    )
                    continue  # ⬅️ never crash Scout

        except RequestException as e:
            logger.error(
                "PUBMED_EFETCH_REQUEST_FAILED",
                extra={"pmids": batch, "error": str(e)},
            )
            continue  # ⬅️ never crash Scout

    return papers
