from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from arxiv_daily import SCHEMA_VERSION
from arxiv_daily.arxiv import sort_papers

LATEST_LIMIT = 100


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def category_month_path(data_root: Path, category: str, year_month: str) -> Path:
    year = year_month[:4]
    return data_root / category / year / f"{year_month}.json"


def ai_month_path(data_root: Path, category: str, year_month: str) -> Path:
    year = year_month[:4]
    return data_root / category / year / f"{year_month}-ai-summary.json"


def latest_path(data_root: Path, category: str) -> Path:
    return data_root / category / "latest.json"


def index_path(data_root: Path) -> Path:
    return data_root / "index.json"


def group_by_primary_category_and_month(papers: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for paper in papers:
        year_month = paper["publishedAt"][:7]
        grouped[(paper["primaryCategory"], year_month)].append(paper)
    return dict(grouped)


def merge_monthly_papers(data_root: Path, papers: Iterable[dict]) -> list[Path]:
    changed_months: list[Path] = []
    grouped = group_by_primary_category_and_month(papers)
    for (category, year_month), month_papers in grouped.items():
        path = category_month_path(data_root, category, year_month)
        existing_payload = read_json(path, {})
        existing_papers = {
            paper["arxivId"]: paper
            for paper in existing_payload.get("papers", [])
            if paper.get("arxivId")
        }
        for paper in month_papers:
            existing_papers[paper["arxivId"]] = paper
        merged = sort_papers(existing_papers.values())
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "generatedAt": utc_now_iso(),
            "category": category,
            "yearMonth": year_month,
            "paperCount": len(merged),
            "papers": merged,
        }
        write_json(path, payload)
        changed_months.append(path)
    return changed_months


def rebuild_latest_and_index(data_root: Path, categories: Iterable[str]) -> None:
    category_entries = []
    for category in sorted(categories):
        monthly_files = find_monthly_paper_files(data_root, category)
        all_papers = []
        month_entries = []
        for path in monthly_files:
            payload = read_json(path, {})
            papers = payload.get("papers", [])
            all_papers.extend(papers)
            year_month = payload.get("yearMonth") or path.stem
            ai_path = ai_month_path(data_root, category, year_month)
            ai_payload = read_json(ai_path, {})
            month_entries.append(
                {
                    "yearMonth": year_month,
                    "papersPath": relative_data_path(path, data_root),
                    "aiSummaryPath": relative_data_path(ai_path, data_root),
                    "paperCount": len(papers),
                    "aiSummaryCount": len(ai_payload.get("papers", [])),
                }
            )

        latest_papers = sort_papers(dedupe_papers(all_papers).values())[:LATEST_LIMIT]
        latest_file = latest_path(data_root, category)
        write_json(
            latest_file,
            {
                "schemaVersion": SCHEMA_VERSION,
                "generatedAt": utc_now_iso(),
                "category": category,
                "paperCount": len(latest_papers),
                "papers": latest_papers,
            },
        )

        category_entries.append(
            {
                "id": category,
                "latestPath": relative_data_path(latest_file, data_root),
                "months": sorted(month_entries, key=lambda item: item["yearMonth"], reverse=True),
            }
        )

    write_json(
        index_path(data_root),
        {
            "schemaVersion": SCHEMA_VERSION,
            "generatedAt": utc_now_iso(),
            "categories": category_entries,
        },
    )


def find_monthly_paper_files(data_root: Path, category: str) -> list[Path]:
    category_dir = data_root / category
    if not category_dir.exists():
        return []
    return sorted(
        [
            path
            for path in category_dir.glob("*/*.json")
            if not path.name.endswith("-ai-summary.json")
        ],
        reverse=True,
    )


def find_monthly_ai_files(data_root: Path, category: str) -> list[Path]:
    category_dir = data_root / category
    if not category_dir.exists():
        return []
    return sorted(category_dir.glob("*/*-ai-summary.json"), reverse=True)


def dedupe_papers(papers: Iterable[dict]) -> dict[str, dict]:
    deduped = {}
    for paper in papers:
        if paper.get("arxivId"):
            deduped[paper["arxivId"]] = paper
    return deduped


def relative_data_path(path: Path, data_root: Path) -> str:
    return str(path.relative_to(data_root).as_posix())


def load_primary_categories(config_path: Path, env_override: str | None = None) -> list[str]:
    if env_override:
        categories = [item.strip() for item in env_override.split(",") if item.strip()]
        if categories:
            return categories
    payload = read_json(config_path, {})
    categories = payload.get("primaryCategories", [])
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        raise ValueError(f"Invalid primaryCategories in {config_path}")
    return categories
