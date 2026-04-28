from __future__ import annotations

import datetime as dt
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NAMESPACES = {"atom": ATOM_NS, "arxiv": ARXIV_NS}
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_REQUEST_DELAY_SECONDS = 3.2


@dataclass(frozen=True)
class ArxivQuery:
    category: str
    max_results: int


def fetch_category(query: ArxivQuery) -> str:
    params = {
        "search_query": f"cat:{query.category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": "0",
        "max_results": str(query.max_results),
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "arxiv-daily-server/0.1 (+https://github.com/UNIkeEN/arxiv-daily-server)"
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8")


def fetch_categories(categories: Iterable[str], max_results: int) -> list[dict]:
    papers: dict[str, dict] = {}
    for index, category in enumerate(categories):
        if index > 0:
            time.sleep(ARXIV_REQUEST_DELAY_SECONDS)
        xml_text = fetch_category(ArxivQuery(category=category, max_results=max_results))
        for paper in parse_arxiv_atom(xml_text):
            papers[paper["arxivId"]] = paper
    return sort_papers(papers.values())


def parse_arxiv_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", NAMESPACES)
    return [parse_entry(entry) for entry in entries]


def parse_entry(entry: ET.Element) -> dict:
    raw_id = text_of(entry, "atom:id")
    arxiv_id = normalize_arxiv_id(raw_id.rsplit("/", 1)[-1])
    title = normalize_space(text_of(entry, "atom:title"))
    abstract = normalize_space(text_of(entry, "atom:summary"))
    published_at = date_only(text_of(entry, "atom:published"))
    updated_at = date_only(text_of(entry, "atom:updated"))
    primary_category = primary_category_of(entry)
    categories = [
        category.attrib["term"]
        for category in entry.findall("atom:category", NAMESPACES)
        if category.attrib.get("term")
    ]

    return {
        "id": arxiv_id,
        "arxivId": arxiv_id,
        "title": title,
        "briefSummary": make_brief_summary(abstract),
        "coverImageUrl": "",
        "primaryCategory": primary_category,
        "publishedAt": published_at,
        "updatedAt": updated_at,
        "abstract": abstract,
        "authors": authors_of(entry),
        "categories": categories,
        "arxivUrl": f"https://arxiv.org/abs/{arxiv_id}",
        "pdfUrl": pdf_url_of(entry, arxiv_id),
        "htmlUrl": f"https://arxiv.org/html/{arxiv_id}",
        **optional_arxiv_fields(entry),
    }


def filter_primary_categories(papers: Iterable[dict], whitelist: set[str]) -> list[dict]:
    deduped: dict[str, dict] = {}
    for paper in papers:
        if paper["primaryCategory"] not in whitelist:
            continue
        deduped[paper["arxivId"]] = paper
    return sort_papers(deduped.values())


def sort_papers(papers: Iterable[dict]) -> list[dict]:
    return sorted(
        papers,
        key=lambda paper: (
            paper.get("publishedAt", ""),
            paper.get("updatedAt", ""),
            paper.get("arxivId", ""),
        ),
        reverse=True,
    )


def normalize_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.strip())


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def date_only(value: str) -> str:
    return value.strip()[:10]


def text_of(entry: ET.Element, path: str) -> str:
    element = entry.find(path, NAMESPACES)
    return "" if element is None or element.text is None else element.text


def primary_category_of(entry: ET.Element) -> str:
    primary = entry.find("arxiv:primary_category", NAMESPACES)
    if primary is not None and primary.attrib.get("term"):
        return primary.attrib["term"]
    first_category = entry.find("atom:category", NAMESPACES)
    if first_category is not None and first_category.attrib.get("term"):
        return first_category.attrib["term"]
    return ""


def authors_of(entry: ET.Element) -> list[str]:
    authors = []
    for author in entry.findall("atom:author", NAMESPACES):
        name = author.find("atom:name", NAMESPACES)
        if name is not None and name.text:
            authors.append(normalize_space(name.text))
    return authors


def pdf_url_of(entry: ET.Element, arxiv_id: str) -> str:
    for link in entry.findall("atom:link", NAMESPACES):
        if link.attrib.get("title") == "pdf" and link.attrib.get("href"):
            return link.attrib["href"]
    return f"https://arxiv.org/pdf/{arxiv_id}"


def optional_arxiv_fields(entry: ET.Element) -> dict[str, str]:
    fields = {
        "doi": text_of(entry, "arxiv:doi"),
        "journalRef": text_of(entry, "arxiv:journal_ref"),
        "comment": text_of(entry, "arxiv:comment"),
    }
    return {key: normalize_space(value) for key, value in fields.items() if value.strip()}


def make_brief_summary(abstract: str, max_length: int = 220) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", abstract, maxsplit=1)[0].strip()
    brief = first_sentence or abstract
    if len(brief) <= max_length:
        return brief
    return f"{brief[: max_length - 1].rstrip()}..."


def recent_papers(papers: Iterable[dict], fetch_days: int, today: dt.date | None = None) -> list[dict]:
    if fetch_days <= 0:
        return sort_papers(papers)
    today = today or dt.datetime.now(dt.UTC).date()
    since = today - dt.timedelta(days=fetch_days)
    filtered = []
    for paper in papers:
        published_at = dt.date.fromisoformat(paper["publishedAt"])
        if published_at >= since:
            filtered.append(paper)
    return sort_papers(filtered)
