#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from arxiv_daily.arxiv import fetch_categories, filter_primary_categories, recent_papers
from arxiv_daily.data_store import (
    load_primary_categories,
    merge_monthly_papers,
    rebuild_latest_and_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch recent arXiv papers into static JSON files.")
    parser.add_argument("--config", default="config/categories.json", type=Path)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--fetch-days", default=env_int("FETCH_DAYS", 3), type=int)
    parser.add_argument(
        "--max-results-per-category",
        default=env_int("MAX_RESULTS_PER_CATEGORY", 100),
        type=int,
    )
    args = parser.parse_args()

    categories = load_primary_categories(args.config, os.getenv("ARXIV_CATEGORIES"))
    if not categories:
        raise SystemExit("No arXiv primary categories configured")

    fetched = fetch_categories(categories, args.max_results_per_category)
    primary_only = filter_primary_categories(fetched, set(categories))
    recent = recent_papers(primary_only, args.fetch_days)
    merge_monthly_papers(args.data_root, recent)
    rebuild_latest_and_index(args.data_root, categories)

    print(
        f"Fetched {len(fetched)} records, kept {len(recent)} recent primary-category records "
        f"for {', '.join(categories)}."
    )
    return 0


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


if __name__ == "__main__":
    raise SystemExit(main())
