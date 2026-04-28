#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from arxiv_daily.data_store import find_monthly_ai_files, find_monthly_paper_files, read_json

REQUIRED_PAPER_FIELDS = {
    "id",
    "arxivId",
    "title",
    "briefSummary",
    "coverImageUrl",
    "primaryCategory",
    "publishedAt",
    "updatedAt",
    "abstract",
    "authors",
    "categories",
    "arxivUrl",
    "pdfUrl",
    "htmlUrl",
}

REQUIRED_AI_FIELDS = {
    "arxivId",
    "aiSummary",
    "keywordsZh",
    "keywordsEn",
    "semanticQueryZh",
    "generatedAt",
    "model",
    "promptVersion",
}

REQUIRED_AI_SUMMARY_FIELDS = {"tldr", "motivation", "method", "result", "conclusion", "markdown"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated arXiv JSON data.")
    parser.add_argument("--root", default="data", type=Path)
    args = parser.parse_args()

    errors = validate_root(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Validated data root: {args.root}")
    return 0


def validate_root(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.exists():
        errors.append(f"Data root does not exist: {root}")
        return errors

    index = read_json(root / "index.json", {})
    categories = index.get("categories", [])
    if not isinstance(categories, list):
        errors.append("index.json categories must be an array")
        return errors

    for category_entry in categories:
        category = category_entry.get("id")
        if not isinstance(category, str) or not category:
            errors.append("index.json category entry is missing id")
            continue
        errors.extend(validate_category(root, category))
    return errors


def validate_category(root: Path, category: str) -> list[str]:
    errors: list[str] = []
    seen_arxiv_ids: set[str] = set()
    for path in find_monthly_paper_files(root, category):
        payload = read_json(path, {})
        papers = payload.get("papers")
        if payload.get("category") != category:
            errors.append(f"{path}: category mismatch")
        if payload.get("yearMonth") != path.stem:
            errors.append(f"{path}: yearMonth mismatch")
        if not isinstance(papers, list):
            errors.append(f"{path}: papers must be an array")
            continue
        for paper in papers:
            errors.extend(validate_paper(path, paper, category, seen_arxiv_ids))

    for path in find_monthly_ai_files(root, category):
        payload = read_json(path, {})
        entries = payload.get("papers")
        if payload.get("category") != category:
            errors.append(f"{path}: category mismatch")
        if not isinstance(entries, list):
            errors.append(f"{path}: papers must be an array")
            continue
        for entry in entries:
            errors.extend(validate_ai_entry(path, entry))
    return errors


def validate_paper(path: Path, paper: dict, category: str, seen_arxiv_ids: set[str]) -> list[str]:
    errors = []
    missing = REQUIRED_PAPER_FIELDS - paper.keys()
    if missing:
        errors.append(f"{path}: paper missing fields {sorted(missing)}")
    arxiv_id = paper.get("arxivId")
    if arxiv_id in seen_arxiv_ids:
        errors.append(f"{path}: duplicate arxivId {arxiv_id}")
    if isinstance(arxiv_id, str):
        seen_arxiv_ids.add(arxiv_id)
    if paper.get("id") != arxiv_id:
        errors.append(f"{path}: id must equal arxivId for {arxiv_id}")
    if paper.get("primaryCategory") != category:
        errors.append(f"{path}: {arxiv_id} primaryCategory is not {category}")
    if not isinstance(paper.get("authors"), list):
        errors.append(f"{path}: {arxiv_id} authors must be an array")
    if not isinstance(paper.get("categories"), list):
        errors.append(f"{path}: {arxiv_id} categories must be an array")
    return errors


def validate_ai_entry(path: Path, entry: dict) -> list[str]:
    errors = []
    missing = REQUIRED_AI_FIELDS - entry.keys()
    if missing:
        errors.append(f"{path}: AI entry missing fields {sorted(missing)}")
    ai_summary = entry.get("aiSummary")
    if not isinstance(ai_summary, dict):
        errors.append(f"{path}: {entry.get('arxivId')} aiSummary must be an object")
        return errors
    missing_summary = REQUIRED_AI_SUMMARY_FIELDS - ai_summary.keys()
    if missing_summary:
        errors.append(f"{path}: {entry.get('arxivId')} aiSummary missing {sorted(missing_summary)}")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
