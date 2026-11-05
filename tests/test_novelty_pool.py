"""Tests for offline novelty scoring against the account alpha pool."""

import json
import tempfile
import unittest
from pathlib import Path

from alpha_agent.novelty_pool import (
    NoveltyPool,
    PoolEntry,
    jaccard,
    token_set,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pool_payload() -> list[dict]:
    return [
        {
            "id": "A1",
            "status": "UNSUBMITTED",
            "settings": {"region": "USA", "universe": "TOP3000"},
            "regular": {"code": "ts_mean(-scl12_buzz, 9) + rank((ts_mean(close, 26) - close) / close)"},
            "is": {"sharpe": 1.4},
        },
        {
            "id": "A2",
            "status": "SUBMITTED",
            "dateSubmitted": "2026-01-01",
            "settings": {"region": "USA", "universe": "TOP3000"},
            "regular": {"code": "rank(ts_delta(close, 5))"},
            "is": {"sharpe": 1.1},
        },
        {
            "id": "A3",
            "status": "UNSUBMITTED",
            "settings": {"region": "EUR", "universe": "TOP1200"},
            "regular": {"code": "ts_mean(-scl12_buzz, 9) + rank((ts_mean(close, 26) - close) / close)"},
            "is": {"sharpe": 0.9},
        },
        {"id": "A4", "status": "UNSUBMITTED", "regular": {}},  # no expression -> dropped
    ]


class PoolParsingTests(unittest.TestCase):
    def test_payload_parsing_filters_incomplete_entries(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        self.assertEqual(len(pool), 3)
        self.assertNotIn("A4", [entry.alpha_id for entry in pool.entries])

    def test_stats_report_coverage_against_reported_total(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload(), total_reported=10)
        stats = pool.stats()
        self.assertEqual(stats["pool_size"], 3)
        self.assertEqual(stats["total_reported"], 10)
        self.assertAlmostEqual(stats["coverage"], 0.3, places=6)

    def test_coverage_is_capped_at_one(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload(), total_reported=2)
        self.assertAlmostEqual(pool.stats()["coverage"], 1.0, places=6)

    def test_submitted_flag_uses_status_or_date(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        by_id = {entry.alpha_id: entry for entry in pool.entries}
        self.assertTrue(by_id["A2"].submitted)
        self.assertFalse(by_id["A1"].submitted)
        self.assertEqual(pool.stats()["submitted"], 1)

    def test_token_set_matches_research_logic(self) -> None:
        from alpha_agent.research_logic import tokenize_expression

        samples = [
            "ts_mean(-scl12_buzz, 9) + rank((ts_mean(close, 26) - close) / close)",
            "group_rank(ts_delta(close, 5), industry)",
        ]
        for sample in samples:
            self.assertEqual(token_set(sample), tokenize_expression(sample))


class SimilarityTests(unittest.TestCase):
    def test_identical_expression_has_zero_novelty(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        expression = "ts_mean(-scl12_buzz, 9) + rank((ts_mean(close, 26) - close) / close)"
        self.assertAlmostEqual(pool.max_similarity(expression), 1.0, places=6)
        self.assertAlmostEqual(pool.novelty(expression), 0.0, places=6)

    def test_unrelated_expression_has_high_novelty(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        novelty = pool.novelty("group_zscore(ts_std_dev(returns, 60), industry)")
        self.assertGreater(novelty, 0.5)

    def test_empty_pool_yields_full_novelty(self) -> None:
        pool = NoveltyPool()
        self.assertEqual(pool.novelty("rank(close)"), 1.0)
        self.assertEqual(pool.max_similarity("rank(close)"), 0.0)
        self.assertEqual(pool.neighbors("rank(close)"), [])

    def test_region_filter_avoids_cross_region_matches(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        expression = "ts_mean(-scl12_buzz, 9) + rank((ts_mean(close, 26) - close) / close)"
        usa = pool.max_similarity(expression, {"region": "USA"})
        eur = pool.max_similarity(expression, {"region": "EUR"})
        self.assertAlmostEqual(usa, 1.0, places=6)
        self.assertAlmostEqual(eur, 1.0, places=6)
        # A region with no pool entries simply has nothing to compare against.
        self.assertEqual(pool.max_similarity(expression, {"region": "CHN"}), 0.0)

    def test_excluding_own_alpha_prevents_self_match(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        expression = "rank(ts_delta(close, 5))"
        self.assertAlmostEqual(pool.max_similarity(expression), 1.0, places=6)
        without_self = pool.max_similarity(expression, exclude_alpha_ids={"A2"})
        self.assertLess(without_self, 0.5)
        neighbors = pool.neighbors(expression, k=1, exclude_alpha_ids={"A2"})
        self.assertEqual(neighbors[0][1].alpha_id, "A1")

    def test_unparseable_expression_falls_back_to_tokens(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        novelty = pool.novelty("rank(close")
        self.assertGreater(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)

    def test_neighbors_are_sorted_and_bounded(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        neighbors = pool.neighbors("rank(ts_delta(close, 5))", k=2)
        self.assertEqual(len(neighbors), 2)
        self.assertGreaterEqual(neighbors[0][0], neighbors[1][0])
        self.assertEqual(neighbors[0][1].alpha_id, "A2")

    def test_jaccard_edge_cases(self) -> None:
        self.assertEqual(jaccard(set(), {"a"}), 0.0)
        self.assertEqual(jaccard({"a"}, {"a"}), 1.0)
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class CacheTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alpha_pool.json"
            pool.save(path)
            self.assertTrue(path.exists())
            reloaded = NoveltyPool.load(path)
            self.assertEqual(len(reloaded), 3)
            self.assertEqual(
                {entry.alpha_id for entry in reloaded.entries},
                {"A1", "A2", "A3"},
            )
            expression = "rank(ts_delta(close, 5))"
            self.assertAlmostEqual(
                pool.max_similarity(expression), reloaded.max_similarity(expression), places=9
            )

    def test_missing_or_corrupt_cache_returns_empty_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(NoveltyPool.load(Path(tmp) / "nope.json")), 0)
            corrupt = Path(tmp) / "bad.json"
            corrupt.write_text("{not json", encoding="utf-8")
            self.assertEqual(len(NoveltyPool.load(corrupt)), 0)

    def test_load_or_fetch_without_client_uses_cache(self) -> None:
        pool = NoveltyPool.from_payload(pool_payload())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alpha_pool.json"
            pool.save(path)
            loaded = NoveltyPool.load_or_fetch(path, client_factory=None)
            self.assertEqual(len(loaded), 3)

    def test_fetch_failure_does_not_raise(self) -> None:
        def broken_client():
            raise RuntimeError("network down")

        with tempfile.TemporaryDirectory() as tmp:
            loaded = NoveltyPool.load_or_fetch(
                Path(tmp) / "alpha_pool.json", client_factory=broken_client, refresh=True
            )
            self.assertEqual(len(loaded), 0)

    def test_fetch_paginates_and_respects_limit(self) -> None:
        pages = {
            0: {"results": [pool_payload()[0]], "next": "/users/self/alphas?limit=1&offset=1"},
            1: {"results": [pool_payload()[1]], "next": None},
        }

        class FakeResponse:
            def __init__(self, data):
                self.data = data
                self.status = 200

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def _request(self, method, url, accept=None):
                page = pages[self.calls]
                self.calls += 1
                return FakeResponse(page)

        pool = NoveltyPool.fetch(client_factory=lambda: FakeClient(), limit=10)
        self.assertEqual(len(pool), 2)


class EntryTests(unittest.TestCase):
    def test_region_property_reads_settings(self) -> None:
        entry = PoolEntry(alpha_id="X", expression="rank(close)", settings={"region": "usa"})
        self.assertEqual(entry.region, "USA")
        self.assertIsNone(PoolEntry(alpha_id="Y", expression="rank(close)").region)


if __name__ == "__main__":
    unittest.main()
