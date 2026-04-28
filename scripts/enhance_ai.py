#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate missing AI sidecar JSON for arXiv papers.")
    parser.add_argument("--config", default="config/categories.json", type=Path)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--max-papers", default=env_int_optional("AI_MAX_PAPERS_PER_RUN"), type=int)
    parser.add_argument("--prompt-dir", default="prompts/paper-ai-summary", type=Path)
    parser.add_argument("--concurrency", default=env_int("AI_CONCURRENCY", 4), type=int)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not base_url or not model:
        raise SystemExit("OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL are required")

    categories = load_primary_categories(args.config, os.getenv("ARXIV_CATEGORIES"))
    max_papers = args.max_papers
    if max_papers is None:
        max_papers = env_int("MAX_RESULTS_PER_CATEGORY", 100) * len(categories)
    if max_papers < 0:
        raise SystemExit("--max-papers must be >= 0")
    concurrency = max(args.concurrency, 1)

    prompt_bundle = load_prompt_bundle(args.prompt_dir)
    client_config = OpenAIClientConfig(api_key=api_key, base_url=base_url, model=model)
    log(
        "Starting AI enhancement: "
        f"categories={','.join(categories)} model={model} promptVersion={prompt_bundle.version} "
        f"maxPapers={max_papers} concurrency={concurrency}"
    )

    generated = 0
    failed = 0
    skipped = 0
    deferred = 0
    month_count = 0
    remaining_budget = max_papers
    for category in categories:
        raw_paths = find_monthly_paper_files(args.data_root, category)
        log(f"[{category}] Found {len(raw_paths)} monthly metadata files.")
        for raw_path in raw_paths:
            month_count += 1
            result = enhance_month(
                raw_path,
                args.data_root,
                prompt_bundle,
                client_config,
                max_generate=remaining_budget,
                concurrency=concurrency,
            )
            generated += result["generated"]
            failed += result["failed"]
            skipped += result["skipped"]
            deferred += result["deferred"]
            remaining_budget = max(remaining_budget - result["generated"] - result["failed"], 0)

    rebuild_latest_and_index(args.data_root, categories)
    log(
        "Finished AI enhancement: "
        f"months={month_count} generated={generated} skipped={skipped} "
        f"failed={failed} deferred={deferred} remainingBudget={remaining_budget}."
    )
    return 0


def enhance_month(
    raw_path: Path,
    data_root: Path,
    prompt_bundle: PromptBundle,
    client_config: OpenAIClientConfig,
    max_generate: int | None = None,
    concurrency: int = 1,
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
    log(
        f"[{category} {year_month}] Loaded {len(papers)} papers; "
        f"{len(existing_entries)} existing AI summaries."
    )

    generated = 0
    failed = 0
    skipped = 0
    missing_papers = [paper for paper in papers if paper["arxivId"] not in existing_entries]
    skipped = len(papers) - len(missing_papers)
    generate_limit = len(missing_papers) if max_generate is None else min(max_generate, len(missing_papers))
    selected_papers = missing_papers[:generate_limit]
    deferred = len(missing_papers) - len(selected_papers)
    if not missing_papers:
        log(f"[{category} {year_month}] Nothing to generate; all papers already have AI summaries.")
    elif not selected_papers:
        log(
            f"[{category} {year_month}] Generation budget is exhausted; "
            f"deferring {deferred} missing AI summaries."
        )
    elif deferred:
        log(
            f"[{category} {year_month}] Generation budget allows {len(selected_papers)} "
            f"of {len(missing_papers)} missing AI summaries; deferring {deferred}."
        )

    workers = min(max(concurrency, 1), len(selected_papers))
    if selected_papers:
        log(f"[{category} {year_month}] Generating with concurrency={workers}.")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {}
            for index, paper in enumerate(selected_papers, start=1):
                arxiv_id = paper["arxivId"]
                log(f"[{category} {year_month}] ({index}/{len(selected_papers)}) Queued {arxiv_id}.")
                future = executor.submit(generate_ai_entry, paper, prompt_bundle, client_config)
                future_to_task[future] = (index, paper, time.monotonic())

            for future in as_completed(future_to_task):
                index, paper, started_at = future_to_task[future]
                arxiv_id = paper["arxivId"]
                try:
                    existing_entries[arxiv_id] = future.result()
                    generated += 1
                    elapsed = time.monotonic() - started_at
                    log(
                        f"[{category} {year_month}] ({index}/{len(selected_papers)}) "
                        f"Done {arxiv_id} in {elapsed:.1f}s."
                    )
                except Exception as error:  # noqa: BLE001 - keep raw data useful even if one AI call fails.
                    failed += 1
                    elapsed = time.monotonic() - started_at
                    log(
                        f"[{category} {year_month}] ({index}/{len(selected_papers)}) "
                        f"Failed {arxiv_id} in {elapsed:.1f}s: {error}"
                    )

        if failed:
            log(
                f"[{category} {year_month}] {failed} requests failed and still count toward "
                "this run's generation budget."
            )

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
    log(
        f"[{category} {year_month}] Wrote {len(ordered_entries)} AI summaries "
        f"to {sidecar_path}; generated={generated} skipped={skipped} "
        f"failed={failed} deferred={deferred}."
    )
    return {"generated": generated, "skipped": skipped, "failed": failed, "deferred": deferred}


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


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def env_int_optional(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


if __name__ == "__main__":
    raise SystemExit(main())
