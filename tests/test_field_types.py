"""VECTOR fields must be reduced before they reach an ordinary operator."""

import json
import tempfile
import unittest
from pathlib import Path

from alpha_agent.catalog import Catalog, load_catalog
from alpha_agent.expr_ast import try_parse
from alpha_agent.field_types import vector_violations
from alpha_agent.refine_llm import ProposalExpression, filter_candidates

TYPES = {
    "nws18_relevance": "VECTOR",
    "eps_estimate_value": "VECTOR",
    "close": "MATRIX",
    "scl12_buzz": "MATRIX",
    "industry": "GROUP",
}


class VectorViolationTests(unittest.TestCase):
    def violations(self, expression, types=TYPES):
        node = try_parse(expression)
        self.assertIsNotNone(node)
        return vector_violations(node, types)

    def test_event_field_in_a_time_series_operator_is_rejected(self) -> None:
        problems = self.violations("ts_decay_linear(nws18_relevance, 10)")
        self.assertEqual(len(problems), 1)
        self.assertIn("nws18_relevance", problems[0])
        self.assertIn("vec_avg", problems[0], "the message must say how to fix it")

    def test_the_comparison_that_cost_a_real_simulation(self) -> None:
        """``Operator less does not support event inputs`` from a live run."""
        self.assertTrue(self.violations("trade_when(anl4_ady_numest < 20, rank(close), -1)", {
            **TYPES, "anl4_ady_numest": "VECTOR",
        }))

    def test_reducing_with_vec_first_is_fine(self) -> None:
        self.assertEqual(self.violations("rank(vec_avg(nws18_relevance, 20))"), [])

    def test_matrix_fields_are_not_judged(self) -> None:
        self.assertEqual(self.violations("ts_mean(-scl12_buzz, 8) + rank(close)"), [])

    def test_unknown_field_is_not_a_violation(self) -> None:
        self.assertEqual(self.violations("ts_mean(mystery_field, 8)"), [], "we do not know it is a vector")

    def test_no_types_means_no_judgement(self) -> None:
        self.assertEqual(self.violations("ts_decay_linear(nws18_relevance, 10)", types={}), [])


class FilterIntegrationTests(unittest.TestCase):
    def test_filter_candidates_counts_the_rejection(self) -> None:
        catalog = Catalog(
            operators=frozenset({"ts_decay_linear", "rank", "close"}),
            fields=frozenset(TYPES),
            field_types=dict(TYPES),
        )
        outcome = filter_candidates(
            [
                ProposalExpression(expression="ts_decay_linear(nws18_relevance, 10)"),
                ProposalExpression(expression="rank(close)"),
            ],
            settings={},
            catalog=catalog,
            limit=4,
        )
        self.assertEqual(outcome.counts.get("vector_field_misuse"), 1)
        self.assertEqual([item.expression for item in outcome.accepted], ["rank(close)"])

    def test_a_catalog_without_types_behaves_as_before(self) -> None:
        catalog = Catalog(operators=frozenset({"ts_decay_linear"}), fields=frozenset({"nws18_relevance"}))
        outcome = filter_candidates(
            [ProposalExpression(expression="ts_decay_linear(nws18_relevance, 10)")],
            settings={},
            catalog=catalog,
            limit=4,
        )
        self.assertEqual(len(outcome.accepted), 1, "fail-open when the types are unknown")


class FieldTypeCacheTests(unittest.TestCase):
    def _cache(self, directory: Path, arena: dict) -> Path:
        path = directory / "data_fields.json"
        path.write_text(
            json.dumps({"arena": arena, "count": 3, "types": TYPES}),
            encoding="utf-8",
        )
        return path

    def test_cache_is_used_for_the_arena_that_wrote_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            arena = {"region": "USA", "universe": "TOP3000", "delay": 1}
            catalog = load_catalog(
                fields_summary=None,
                operators_cache=directory / "operators.json",
                client_factory=None,
                arena=arena,
                field_types_cache=self._cache(directory, arena),
            )
        self.assertEqual(catalog.field_types["nws18_relevance"], "VECTOR")
        self.assertTrue(catalog.has_field_types)
        self.assertEqual(catalog.matrix_fields(["close", "nws18_relevance"]), ["close"])

    def test_cache_from_another_arena_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stale = {"region": "USA", "universe": "TOP3000", "delay": 1}
            catalog = load_catalog(
                fields_summary=None,
                operators_cache=directory / "operators.json",
                client_factory=None,
                arena={"region": "EUR", "universe": "TOP1200", "delay": 1},
                field_types_cache=self._cache(directory, stale),
            )
        self.assertEqual(catalog.field_types, {}, "types are per arena")


if __name__ == "__main__":
    unittest.main()
