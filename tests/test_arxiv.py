from __future__ import annotations

import datetime as dt
import sys
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from arxiv_daily.arxiv import (
    ArxivFetchOptions,
    ArxivQuery,
    fetch_category,
    filter_primary_categories,
    normalize_arxiv_id,
    parse_arxiv_atom,
    recent_papers,
)


class ArxivParsingTests(unittest.TestCase):
    def test_parse_primary_category_and_app_fields(self) -> None:
        xml_text = Path("tests/fixtures/arxiv_atom.xml").read_text(encoding="utf-8")
        papers = parse_arxiv_atom(xml_text)

        self.assertEqual(len(papers), 2)
        paper = papers[0]
        self.assertEqual(paper["id"], "2604.00001")
        self.assertEqual(paper["arxivId"], "2604.00001")
        self.assertEqual(paper["primaryCategory"], "cs.AI")
        self.assertEqual(paper["categories"], ["cs.AI", "cs.CL"])
        self.assertEqual(paper["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(paper["comment"], "Accepted mock fixture.")

    def test_primary_category_filter_avoids_secondary_duplicates(self) -> None:
        xml_text = Path("tests/fixtures/arxiv_atom.xml").read_text(encoding="utf-8")
        papers = parse_arxiv_atom(xml_text)
        duplicated = papers + [papers[0]]

        filtered = filter_primary_categories(duplicated, {"cs.AI", "cs.CL", "cs.CV", "cs.GR"})

        self.assertEqual([paper["arxivId"] for paper in filtered], ["2604.00001"])

    def test_recent_papers_filters_by_published_date(self) -> None:
        xml_text = Path("tests/fixtures/arxiv_atom.xml").read_text(encoding="utf-8")
        papers = parse_arxiv_atom(xml_text)

        filtered = recent_papers(papers, fetch_days=1, today=dt.date(2026, 4, 28))

        self.assertEqual([paper["arxivId"] for paper in filtered], ["2604.00002", "2604.00001"])

    def test_normalize_arxiv_id_strips_version(self) -> None:
        self.assertEqual(normalize_arxiv_id("2604.00001v12"), "2604.00001")

    def test_fetch_category_retries_429(self) -> None:
        xml_text = Path("tests/fixtures/arxiv_atom.xml").read_text(encoding="utf-8")
        error = urllib.error.HTTPError(
            url="https://export.arxiv.org/api/query",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "1"},
            fp=None,
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return xml_text.encode("utf-8")

        with (
            patch("arxiv_daily.arxiv.urllib.request.urlopen", side_effect=[error, Response()]),
            patch("arxiv_daily.arxiv.time.sleep") as sleep,
        ):
            result = fetch_category(
                ArxivQuery(category="cs.AI", max_results=1),
                ArxivFetchOptions(max_retries=1, retry_base_delay_seconds=0),
            )

        self.assertEqual(result, xml_text)
        sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
