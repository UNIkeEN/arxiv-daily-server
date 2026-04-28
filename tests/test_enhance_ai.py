from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import enhance_ai
from arxiv_daily.data_store import merge_monthly_papers, read_json
from arxiv_daily.openai_client import OpenAIClientConfig
from arxiv_daily.prompts import PromptBundle


def paper() -> dict:
    return {
        "id": "2604.00001",
        "arxivId": "2604.00001",
        "title": "Paper",
        "briefSummary": "Short summary.",
        "coverImageUrl": "",
        "primaryCategory": "cs.AI",
        "publishedAt": "2026-04-27",
        "updatedAt": "2026-04-27",
        "abstract": "Abstract.",
        "authors": ["Ada Lovelace"],
        "categories": ["cs.AI"],
        "arxivUrl": "https://arxiv.org/abs/2604.00001",
        "pdfUrl": "https://arxiv.org/pdf/2604.00001",
        "htmlUrl": "https://arxiv.org/html/2604.00001",
    }


class EnhanceAiTests(unittest.TestCase):
    def test_normalize_ai_payload_requires_complete_json(self) -> None:
        with self.assertRaises(ValueError):
            enhance_ai.normalize_ai_payload({"aiSummary": {}})

    def test_ai_failure_does_not_remove_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            merge_monthly_papers(data_root, [paper()])
            raw_path = data_root / "cs.AI" / "2026" / "2026-04.json"
            prompt = PromptBundle(
                version="paper-ai-summary/v1",
                system="system",
                user_template="{paper_json}\n{output_schema}",
                schema={},
            )
            config = OpenAIClientConfig(api_key="key", base_url="https://example.com/v1", model="model")

            with patch.object(enhance_ai, "create_chat_completion", side_effect=RuntimeError("boom")):
                result = enhance_ai.enhance_month(raw_path, data_root, prompt, config)

            raw_payload = read_json(raw_path, {})
            sidecar_payload = read_json(data_root / "cs.AI" / "2026" / "2026-04-ai-summary.json", {})
            self.assertEqual(result, {"generated": 0, "skipped": 0, "failed": 1})
            self.assertEqual(raw_payload["paperCount"], 1)
            self.assertEqual(sidecar_payload["paperCount"], 0)

    def test_existing_ai_summary_is_logged_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            merge_monthly_papers(data_root, [paper()])
            sidecar_path = data_root / "cs.AI" / "2026" / "2026-04-ai-summary.json"
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(
                """{
  "schemaVersion": 1,
  "generatedAt": "2026-04-28T00:00:00Z",
  "category": "cs.AI",
  "yearMonth": "2026-04",
  "promptVersion": "paper-ai-summary/v1",
  "model": "model",
  "paperCount": 1,
  "papers": [
    {
      "arxivId": "2604.00001",
      "aiSummary": {
        "tldr": "摘要",
        "motivation": "动机",
        "method": "方法",
        "result": "效果",
        "conclusion": "结论",
        "markdown": "## TL;DR\\n摘要"
      },
      "keywordsZh": ["关键词"],
      "keywordsEn": ["keyword"],
      "semanticQueryZh": "查询",
      "generatedAt": "2026-04-28T00:00:00Z",
      "model": "model",
      "promptVersion": "paper-ai-summary/v1"
    }
  ]
}
""",
                encoding="utf-8",
            )
            prompt = PromptBundle(
                version="paper-ai-summary/v1",
                system="system",
                user_template="{paper_json}\n{output_schema}",
                schema={},
            )
            config = OpenAIClientConfig(api_key="key", base_url="https://example.com/v1", model="model")

            result = enhance_ai.enhance_month(
                data_root / "cs.AI" / "2026" / "2026-04.json",
                data_root,
                prompt,
                config,
            )

            self.assertEqual(result, {"generated": 0, "skipped": 1, "failed": 0})


if __name__ == "__main__":
    unittest.main()
