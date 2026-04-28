# arxiv-daily-server

Static JSON data service for the arXiv App.

The `main` branch contains scripts, prompts, and GitHub Actions workflow files.
The `data` branch contains generated JSON files only. The app can read those
files directly through `raw.githubusercontent.com`.

## Data Scope

The first version tracks papers whose arXiv `primaryCategory` is one of:

- `cs.AI`
- `cs.CL`
- `cs.CV`
- `cs.GR`

The fetch step may query arXiv by category, but storage is always based on the
paper's returned `primaryCategory`. A paper with multiple categories is stored
only under its primary category, which avoids duplicate feed entries.

## Data Layout

Generated files live in the `data` branch:

```text
data/
  index.json
  cs.AI/
    latest.json
    2026/
      2026-04.json
      2026-04-ai-summary.json
```

Monthly paper files contain arXiv metadata aligned with the app's `Paper`
model. Monthly AI sidecar files contain summaries and search keywords keyed by
`arxivId`.

## GitHub Actions

Two workflows keep the raw metadata path independent from the AI path:

- `Fetch arXiv Metadata`: runs daily at UTC `02:30` and can be triggered
  manually. It updates raw monthly JSON, `latest.json`, and `index.json`.
- `Enhance arXiv AI Summaries`: runs automatically after a successful metadata
  fetch and can also be triggered manually. It only fills missing AI sidecar
  entries.

If AI credentials or model calls fail, raw arXiv metadata can still be updated
by the fetch workflow.

## GitHub Configuration

Repository secrets:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Repository variables:

- `OPENAI_MODEL`: OpenAI-compatible chat model name.
- `AI_CONCURRENCY`: parallel AI requests, default `4`.
- `AI_MAX_PAPERS_PER_RUN`: max AI generations per workflow run. If unset,
  defaults to `MAX_RESULTS_PER_CATEGORY * category count`.
- `FETCH_DAYS`: number of recent days to refresh, default `3`.
- `MAX_RESULTS_PER_CATEGORY`: arXiv API max results per category, default `100`.
- `ARXIV_CATEGORIES`: optional comma-separated override for primary-category
  whitelist.
- `ARXIV_REQUEST_DELAY_SECONDS`: delay between category requests, default `8`.
- `ARXIV_MAX_RETRIES`: retry count for arXiv 429/5xx responses, default `5`.
- `ARXIV_RETRY_BASE_DELAY_SECONDS`: exponential retry base delay, default `30`.

## Local Commands

Fetch arXiv metadata:

```bash
python scripts/fetch_arxiv.py --data-root data
```

Generate missing AI summaries:

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... \
  python scripts/enhance_ai.py --data-root data
```

Validate generated JSON:

```bash
python scripts/validate_data.py --root data
```

Run tests:

```bash
python -m unittest
```

## Raw URL Example

```text
https://raw.githubusercontent.com/UNIkeEN/arxiv-daily-server/data/data/cs.AI/latest.json
```
