import requests
import xml.etree.ElementTree as ET
from typing import List, Optional

from app.agents.state import Paper

# PubMed API base
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


def _safe_find_text(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is not None and elem.text:
        return elem.text.strip()
    return None


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

    root = ET.fromstring(response.text)
    papers: List[Paper] = []

    for article in root.findall(".//PubmedArticle"):
        try:
            citation = article.find("MedlineCitation")
            article_node = citation.find("Article") if citation is not None else None

            pmid = _safe_find_text(citation.find("PMID")) if citation is not None else None
            title = _safe_find_text(article_node.find("ArticleTitle")) if article_node is not None else None

            # Abstract may have multiple AbstractText nodes
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

            date_node = citation.find("DateCompleted") if citation is not None else None
            if date_node is not None:
                year_text = _safe_find_text(date_node.find("Year"))
                if year_text and year_text.isdigit():
                    year = int(year_text)

            # Only add papers with a real abstract
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

        except Exception:
            # Skip malformed articles but never crash Scout
            continue

    return papers
