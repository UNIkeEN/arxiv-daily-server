#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from arxiv_daily.arxiv import sort_papers
from arxiv_daily.data_store import (
    ai_month_path,
    find_monthly_paper_files,
    load_primary_categories,
    read_json,
    rebuild_latest_and_index,
    utc_now_iso,
    write_json,
)
from arxiv_daily.openai_client import (
    OpenAIClientConfig,
    create_chat_completion,
    extract_json_object,
)
from arxiv_daily.prompts import PromptBundle, load_prompt_bundle, render_user_prompt

REQUIRED_AI_SUMMARY_FIELDS = ("tldr", "motivation", "method", "result", "conclusion", "markdown")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate missing AI sidecar JSON for arXiv papers.")
    parser.add_argument("--config", default="config/categories.json", type=Path)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--prompt-dir", default="prompts/paper-ai-summary", type=Path)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not base_url or not model:
        raise SystemExit("OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL are required")

    categories = load_primary_categories(args.config, os.getenv("ARXIV_CATEGORIES"))
    prompt_bundle = load_prompt_bundle(args.prompt_dir)
    client_config = OpenAIClientConfig(api_key=api_key, base_url=base_url, model=model)

    generated = 0
    failed = 0
    for category in categories:
        for raw_path in find_monthly_paper_files(args.data_root, category):
            result = enhance_month(raw_path, args.data_root, prompt_bundle, client_config)
            generated += result["generated"]
            failed += result["failed"]

    rebuild_latest_and_index(args.data_root, categories)
    print(f"Generated {generated} AI summaries. Failed {failed}.")
    return 0


def enhance_month(
    raw_path: Path,
    data_root: Path,
    prompt_bundle: PromptBundle,
    client_config: OpenAIClientConfig,
) -> dict[str, int]:
    raw_payload = read_json(raw_path, {})
    category = raw_payload["category"]
    year_month = raw_payload["yearMonth"]
    papers = raw_payload.get("papers", [])
    sidecar_path = ai_month_path(data_root, category, year_month)
    sidecar_payload = read_json(sidecar_path, {})
    existing_entries = {
        entry["arxivId"]: entry
        for entry in sidecar_payload.get("papers", [])
        if entry.get("arxivId")
    }

    generated = 0
    failed = 0
    for paper in papers:
        arxiv_id = paper["arxivId"]
        if arxiv_id in existing_entries:
            continue
        try:
            existing_entries[arxiv_id] = generate_ai_entry(paper, prompt_bundle, client_config)
            generated += 1
        except Exception as error:  # noqa: BLE001 - keep raw data useful even if one AI call fails.
            failed += 1
            print(f"Warning: failed to generate AI summary for {arxiv_id}: {error}")

    ordered_entries = [
        existing_entries[paper["arxivId"]]
        for paper in sort_papers(papers)
        if paper["arxivId"] in existing_entries
    ]
    write_json(
        sidecar_path,
        {
            "schemaVersion": raw_payload.get("schemaVersion", 1),
            "generatedAt": utc_now_iso(),
            "category": category,
            "yearMonth": year_month,
            "promptVersion": prompt_bundle.version,
            "model": client_config.model,
            "paperCount": len(ordered_entries),
            "papers": ordered_entries,
        },
    )
    return {"generated": generated, "failed": failed}


def generate_ai_entry(
    paper: dict,
    prompt_bundle: PromptBundle,
    client_config: OpenAIClientConfig,
) -> dict:
    messages = [
        {"role": "system", "content": prompt_bundle.system},
        {"role": "user", "content": render_user_prompt(prompt_bundle, paper)},
    ]
    response = create_chat_completion(client_config, messages)
    payload = normalize_ai_payload(extract_json_object(response))
    return {
        "arxivId": paper["arxivId"],
        **payload,
        "generatedAt": utc_now_iso(),
        "model": client_config.model,
        "promptVersion": prompt_bundle.version,
    }


def normalize_ai_payload(payload: dict) -> dict:
    ai_summary = payload.get("aiSummary")
    if not isinstance(ai_summary, dict):
        raise ValueError("Missing aiSummary object")
    missing_summary_fields = [
        field for field in REQUIRED_AI_SUMMARY_FIELDS if not isinstance(ai_summary.get(field), str)
    ]
    if missing_summary_fields:
        raise ValueError(f"Missing aiSummary fields: {missing_summary_fields}")

    keywords_zh = payload.get("keywordsZh")
    keywords_en = payload.get("keywordsEn")
    semantic_query = payload.get("semanticQueryZh")
    if not isinstance(keywords_zh, list) or not all(isinstance(item, str) for item in keywords_zh):
        raise ValueError("keywordsZh must be a string array")
    if not isinstance(keywords_en, list) or not all(isinstance(item, str) for item in keywords_en):
        raise ValueError("keywordsEn must be a string array")
    if not isinstance(semantic_query, str) or not semantic_query.strip():
        raise ValueError("semanticQueryZh must be a non-empty string")

    return {
        "aiSummary": {field: ai_summary[field].strip() for field in REQUIRED_AI_SUMMARY_FIELDS},
        "keywordsZh": [item.strip() for item in keywords_zh if item.strip()],
        "keywordsEn": [item.strip() for item in keywords_en if item.strip()],
        "semanticQueryZh": semantic_query.strip(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
