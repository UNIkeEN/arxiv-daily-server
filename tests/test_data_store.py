from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from arxiv_daily.data_store import merge_monthly_papers, read_json, rebuild_latest_and_index


def paper(arxiv_id: str, primary_category: str, published_at: str) -> dict:
    return {
        "id": arxiv_id,
        "arxivId": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "briefSummary": "Short summary.",
        "coverImageUrl": "",
        "primaryCategory": primary_category,
        "publishedAt": published_at,
        "updatedAt": published_at,
        "abstract": "Abstract.",
        "authors": ["Ada Lovelace"],
        "categories": [primary_category],
        "arxivUrl": f"https://arxiv.org/abs/{arxiv_id}",
        "pdfUrl": f"https://arxiv.org/pdf/{arxiv_id}",
        "htmlUrl": f"https://arxiv.org/html/{arxiv_id}",
    }


class DataStoreTests(unittest.TestCase):
    def test_merge_monthly_papers_dedupes_by_arxiv_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            first = paper("2604.00001", "cs.AI", "2026-04-27")
            second = {**first, "title": "Updated title"}

            merge_monthly_papers(data_root, [first])
            merge_monthly_papers(data_root, [second])

            payload = read_json(data_root / "cs.AI" / "2026" / "2026-04.json", {})
            self.assertEqual(payload["paperCount"], 1)
            self.assertEqual(payload["papers"][0]["title"], "Updated title")

    def test_rebuild_latest_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            merge_monthly_papers(
                data_root,
                [
                    paper("2604.00001", "cs.AI", "2026-04-27"),
                    paper("2603.00001", "cs.AI", "2026-03-27"),
                ],
            )

            rebuild_latest_and_index(data_root, ["cs.AI"])

            latest = read_json(data_root / "cs.AI" / "latest.json", {})
            index = read_json(data_root / "index.json", {})
            self.assertEqual(latest["paperCount"], 2)
            self.assertEqual(index["categories"][0]["id"], "cs.AI")
            self.assertEqual(index["categories"][0]["months"][0]["yearMonth"], "2026-04")


if __name__ == "__main__":
    unittest.main()
