# Paper AI Summary Prompt

Prompt version: `paper-ai-summary/v1`

This prompt produces a compact Chinese explanation for one arXiv paper. It is
designed for a mobile-first paper feed and local semantic search.

## Template Variables

- `paper_json`: compact JSON for one paper.
- `output_schema`: expected JSON schema.

The model must return a single JSON object and no Markdown wrapper.
