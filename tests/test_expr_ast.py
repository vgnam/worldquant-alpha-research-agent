"""Tests for the FASTEXPR parser and offline expression analysis.

The most important test here is the parser coverage test: every expression the
repository ships in its two idea libraries must parse, otherwise the offline
filters silently degrade to "unknown shape" for real seeds.
"""

import json
import unittest
from pathlib import Path

from alpha_agent import expr_ast
from alpha_agent.expr_ast import (
    ExprNode,
    ExpressionParseError,
    ast_bigrams,
    behavioral_signature,
    canonical_form,
    collect_call_operators,
    collect_identifiers,
    cross_sectional_profile,
    expression_identity,
    max_window,
    node_depth,
    node_size,
    parse_expression,
    soft_nesting_count,
    strip_redundant_wrappers,
    subtree_hashes,
    try_parse,
    validate_against,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def library_expressions() -> list[str]:
    """Every seed and formatted template expression shipped in the repo."""
    expressions: list[str] = []
    for name in ("alpha_pipeline_ideas.json", "alpha101_ideas.json"):
        payload = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        for item in payload.get("manual_seeds") or []:
            expression = item.get("expression")
            if isinstance(expression, str) and expression.strip():
                expressions.append(expression)
        for family in payload.get("families") or []:
            for template in family.get("templates") or []:
                raw = template.get("expression")
                if not isinstance(raw, str):
                    continue
                try:
                    expressions.append(
                        raw.format(field="close", signed_field="-close", window=5)
                    )
                except (KeyError, IndexError):
                    continue
    return expressions


class ParserCoverageTests(unittest.TestCase):
    def test_every_shipped_expression_parses(self) -> None:
        expressions = library_expressions()
        self.assertGreaterEqual(len(expressions), 100)
        failures = []
        for expression in expressions:
            try:
                parse_expression(expression)
            except ExpressionParseError as exc:
                failures.append((expression[:70], str(exc)))
        self.assertEqual(failures, [], f"{len(failures)} expressions failed to parse")

    def test_roundtrip_is_stable(self) -> None:
        for expression in library_expressions()[:40]:
            node = parse_expression(expression)
            rendered = canonical_form(node)
            self.assertEqual(canonical_form(parse_expression(rendered)), rendered)

    def test_parses_comparison_and_if_else(self) -> None:
        node = parse_expression(
            "if_else(0.25 < ((ts_delay(close, 20) - close) / 10), -1, if_else(close < 0, 1, -1 * (close - ts_delay(close, 1))))"
        )
        operators = collect_call_operators(node)
        self.assertIn("if_else", operators)
        self.assertIn("ts_delay", operators)
        self.assertEqual(node.op, "if_else")

    def test_rejects_garbage_with_a_clear_error(self) -> None:
        for bad in ["", "   ", "rank(", "rank(close))", "close +", "close $ 1"]:
            with self.assertRaises(ExpressionParseError):
                parse_expression(bad)

    def test_try_parse_returns_none_instead_of_raising(self) -> None:
        self.assertIsNone(try_parse("rank("))
        self.assertIsNotNone(try_parse("rank(close)"))


class StructuralTests(unittest.TestCase):
    def test_idempotent_rank_is_collapsed(self) -> None:
        wrapped = parse_expression("rank(rank(close))")
        plain = parse_expression("rank(close)")
        self.assertEqual(canonical_form(strip_redundant_wrappers(wrapped)), canonical_form(plain))

    def test_group_neutralize_is_idempotent_for_same_group(self) -> None:
        nested = parse_expression("group_neutralize(group_neutralize(close, industry), industry)")
        single = parse_expression("group_neutralize(close, industry)")
        self.assertEqual(canonical_form(strip_redundant_wrappers(nested)), canonical_form(single))

    def test_group_neutralize_is_not_collapsed_for_different_groups(self) -> None:
        nested = parse_expression("group_neutralize(group_neutralize(close, industry), subindustry)")
        stripped = canonical_form(strip_redundant_wrappers(nested))
        self.assertNotEqual(stripped, canonical_form(parse_expression("group_neutralize(close, industry)")))
        self.assertIn("subindustry", stripped)

    def test_time_series_nesting_is_preserved(self) -> None:
        node = parse_expression("ts_mean(ts_mean(close, 5), 5)")
        self.assertEqual(canonical_form(strip_redundant_wrappers(node)), canonical_form(node))
        self.assertEqual(soft_nesting_count(node), 1)

    def test_identity_differs_for_different_base_signals(self) -> None:
        a = expression_identity("rank(ts_delta(close, 5))")
        b = expression_identity("rank(ts_delta(close, 10))")
        self.assertNotEqual(a, b)

    def test_size_and_depth(self) -> None:
        node = parse_expression("rank(ts_sum(close, 5))")
        self.assertEqual(node_size(node), 4)  # rank, ts_sum, close, 5
        self.assertEqual(node_depth(node), 2)


class CrossSectionalProfileTests(unittest.TestCase):
    def test_profile_peels_chain_and_keeps_base(self) -> None:
        node = parse_expression("group_rank(rank(ts_delta(close, 5)), industry)")
        ops, base = cross_sectional_profile(node)
        self.assertEqual(canonical_form(base), canonical_form(parse_expression("ts_delta(close, 5)")))
        self.assertIn("group_rank(industry)", ops)
        self.assertIn("rank", ops)

    def test_profile_is_empty_for_plain_signal(self) -> None:
        node = parse_expression("ts_delta(close, 5)")
        ops, base = cross_sectional_profile(node)
        self.assertEqual(ops, frozenset())
        self.assertEqual(canonical_form(base), canonical_form(node))

    def test_collapses_chain_inside_addition(self) -> None:
        a = "ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))"
        b = "ts_mean(-scl12_buzz, 8) + zscore(rank(rank(ts_delta(close, 5))))"
        # rank(x) vs zscore(rank(x)) are different op sets -> must stay distinct
        self.assertNotEqual(behavioral_signature(a), behavioral_signature(b))

        c = "ts_mean(-scl12_buzz, 8) + rank(rank(ts_delta(close, 5)))"
        self.assertEqual(behavioral_signature(a), behavioral_signature(c))


class SubtreeTests(unittest.TestCase):
    def test_subtree_hashes_are_stable_and_distinct(self) -> None:
        a = subtree_hashes(parse_expression("rank(ts_delta(close, 5))"))
        b = subtree_hashes(parse_expression("rank(ts_delta(close, 5))"))
        c = subtree_hashes(parse_expression("rank(ts_delta(close, 10))"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_min_size_filters_leaf_subtrees(self) -> None:
        hashes = subtree_hashes(parse_expression("rank(close)"), min_size=2)
        self.assertEqual(len(hashes), 1)

    def test_bigrams_capture_operator_context(self) -> None:
        bigrams = ast_bigrams(parse_expression("rank(ts_delta(close, 5))"))
        self.assertIn(("rank", "ts_delta"), bigrams)
        self.assertIn(("ts_delta", "#leaf"), bigrams)

    def test_max_window_uses_ts_arguments(self) -> None:
        self.assertEqual(max_window(parse_expression("ts_mean(close, 20) + ts_delta(close, 5)")), 20)
        self.assertIsNone(max_window(parse_expression("rank(close)")))


class ValidationTests(unittest.TestCase):
    def test_unknown_operator_is_reported(self) -> None:
        node = parse_expression("totally_made_up_op(close)")
        ok, errors = validate_against(node, operators={"rank", "ts_mean"})
        self.assertFalse(ok)
        self.assertEqual(errors, ["unknown_operator:totally_made_up_op"])

    def test_unknown_field_is_reported(self) -> None:
        node = parse_expression("rank(mystery_field_xyz)")
        ok, errors = validate_against(node, fields={"close", "open"})
        self.assertFalse(ok)
        self.assertEqual(errors, ["unknown_field:mystery_field_xyz"])

    def test_builtin_group_identifiers_are_allowed(self) -> None:
        node = parse_expression("group_rank(close, industry)")
        ok, errors = validate_against(node, operators={"group_rank"}, fields={"close"})
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_none_catalog_skips_validation(self) -> None:
        node = parse_expression("whatever_op(mystery)")
        ok, errors = validate_against(node)
        self.assertTrue(ok)
        self.assertEqual(errors, [])


class DuplicateWrapperTests(unittest.TestCase):
    def test_wrapper_permutations_share_one_behavioral_signature(self) -> None:
        payload = json.loads((FIXTURES / "duplicate_wrapper_family.json").read_text(encoding="utf-8"))
        signatures = {
            behavioral_signature(expression, payload["settings"])
            for expression in payload["expressions"]
        }
        self.assertEqual(
            len(signatures),
            1,
            f"expected one signature, got {len(signatures)} for {payload['alpha_ids']}",
        )

    def test_identity_signatures_are_not_collapsed(self) -> None:
        """Exact identity must stay strict: these are textually different."""
        payload = json.loads((FIXTURES / "duplicate_wrapper_family.json").read_text(encoding="utf-8"))
        identities = {expression_identity(expression) for expression in payload["expressions"]}
        self.assertGreater(len(identities), 1)

    def test_settings_change_the_signature(self) -> None:
        payload = json.loads((FIXTURES / "duplicate_wrapper_family.json").read_text(encoding="utf-8"))
        expression = payload["expressions"][0]
        other = dict(payload["settings"])
        other["neutralization"] = "SUBINDUSTRY"
        self.assertNotEqual(
            behavioral_signature(expression, payload["settings"]),
            behavioral_signature(expression, other),
        )

    def test_genuinely_different_bet_keeps_its_own_signature(self) -> None:
        payload = json.loads((FIXTURES / "duplicate_wrapper_family.json").read_text(encoding="utf-8"))
        other = "ts_mean(-scl12_buzz, 8) + rank((ts_mean(close, 26) - close) / close)"
        self.assertNotEqual(
            behavioral_signature(payload["expressions"][0], payload["settings"]),
            behavioral_signature(other, payload["settings"]),
        )


class NodeShapeTests(unittest.TestCase):
    def test_leaf_helpers(self) -> None:
        number = ExprNode(leaf="5")
        name = ExprNode(leaf="close")
        self.assertTrue(number.is_number)
        self.assertFalse(name.is_number)
        self.assertTrue(name.is_leaf)

    def test_module_exports_are_real(self) -> None:
        for name in expr_ast.__all__:
            self.assertTrue(hasattr(expr_ast, name), f"{name} is missing")


if __name__ == "__main__":
    unittest.main()
