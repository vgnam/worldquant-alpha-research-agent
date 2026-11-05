"""Tests for gate-margin reward, UCT tree and Pareto fronts."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from alpha_agent.families import data_groups, lineage_family
from alpha_agent.pareto import (
    DEFAULT_OBJECTIVES,
    ParetoFront,
    dominates,
    hypervolume,
    is_feasible,
    normalize_objectives,
)
from alpha_agent.reward import (
    ALL_DIMENSIONS,
    assess_record,
    clamp,
    gate_margins,
    percentile,
    scalar_reward,
    weakest_dimension,
)
from alpha_agent.search_tree import EVALUATED, PENDING, SearchTree

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeCandidate:
    """Minimal stand-in for ``pipeline.Candidate``."""

    def __init__(self, expression: str, family: str = "fam", idea_name: str = "idea",
                 settings: dict | None = None) -> None:
        self.expression = expression
        self.family = family
        self.idea_name = idea_name
        self.settings = settings or {"region": "USA", "universe": "TOP3000"}
        self.parent_key = None


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------


class GateMarginTests(unittest.TestCase):
    def test_margins_are_signed_relative_distances(self) -> None:
        record = load_fixture("record_quality_fail.json")
        margins = gate_margins(record)
        # LOW_SHARPE 1.31 vs 1.58 -> negative
        self.assertLess(margins["LOW_SHARPE"], 0)
        self.assertAlmostEqual(margins["LOW_SHARPE"], (1.31 - 1.58) / 1.58, places=4)
        # PROD_CORRELATION is lower-is-better: 0.9954 vs 0.7 -> negative
        self.assertLess(margins["PROD_CORRELATION"], 0)
        self.assertAlmostEqual(margins["PROD_CORRELATION"], (0.7 - 0.9954) / 0.7, places=4)

    def test_pending_checks_have_no_margin(self) -> None:
        record = load_fixture("record_quality_corr_blocked.json")
        margins = gate_margins(record)
        self.assertNotIn("PROD_CORRELATION", margins)
        self.assertIn("SELF_CORRELATION", margins)
        self.assertLess(margins["SELF_CORRELATION"], 0)

    def test_margins_are_clamped(self) -> None:
        record = {
            "check_diagnostics": [
                {"name": "LOW_SHARPE", "result": "PASS", "limit": 0.5, "value": 50.0},
            ]
        }
        self.assertLessEqual(gate_margins(record)["LOW_SHARPE"], 1.0)

    def test_missing_limit_or_value_is_skipped(self) -> None:
        record = {
            "check_diagnostics": [
                {"name": "CONCENTRATED_WEIGHT", "result": "PASS", "limit": None, "value": None},
                {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 1.7},
            ]
        }
        margins = gate_margins(record)
        self.assertNotIn("CONCENTRATED_WEIGHT", margins)
        self.assertIn("LOW_SHARPE", margins)


class AssessmentTests(unittest.TestCase):
    def test_correlation_blocked_beats_quality_failure(self) -> None:
        blocked = assess_record(load_fixture("record_quality_corr_blocked.json"))
        failed = assess_record(load_fixture("record_quality_fail.json"))
        self.assertGreater(blocked.scalar, failed.scalar)
        self.assertTrue(blocked.quality_ready)
        self.assertFalse(blocked.submit_ready)
        self.assertFalse(failed.quality_ready)

    def test_pending_is_counted_and_penalized(self) -> None:
        record = load_fixture("record_quality_corr_blocked.json")
        assessment = assess_record(record)
        self.assertIn("PROD_CORRELATION", assessment.pending)
        without_pending = json.loads(json.dumps(record))
        for check in without_pending["check_diagnostics"]:
            if check["name"] == "PROD_CORRELATION":
                check["result"] = "PASS"
                check["value"] = 0.5
        higher = assess_record(without_pending)
        self.assertGreater(higher.scalar, assessment.scalar)

    def test_submit_ready_gets_the_big_bonus(self) -> None:
        record = load_fixture("record_quality_corr_blocked.json")
        ready = json.loads(json.dumps(record))
        ready["precheck_submit_ready"] = True
        for check in ready["check_diagnostics"]:
            if check["name"] == "SELF_CORRELATION":
                check["result"] = "PASS"
                check["value"] = 0.5
            if check["name"] == "PROD_CORRELATION":
                check["result"] = "PASS"
                check["value"] = 0.5
        base = assess_record(record)
        boosted = assess_record(ready)
        self.assertTrue(boosted.submit_ready)
        self.assertGreater(boosted.scalar, base.scalar + 900)

    def test_gate_fraction_is_between_zero_and_one(self) -> None:
        for name in ("record_quality_corr_blocked.json", "record_quality_fail.json"):
            assessment = assess_record(load_fixture(name))
            self.assertGreaterEqual(assessment.gate_fraction, 0.0)
            self.assertLessEqual(assessment.gate_fraction, 1.0)
            self.assertGreater(assessment.applicable, 0)

    def test_scalar_reward_matches_assessment(self) -> None:
        record = load_fixture("record_quality_fail.json")
        self.assertAlmostEqual(scalar_reward(record), assess_record(record).scalar, places=9)

    def test_assessment_is_json_serialisable(self) -> None:
        payload = assess_record(load_fixture("record_quality_fail.json")).to_dict()
        json.dumps(payload)
        self.assertIn("margins", payload)
        self.assertIn("quality_margin", payload)


class DimensionTargetingTests(unittest.TestCase):
    def test_weakest_dimension_is_the_lowest_margin(self) -> None:
        margins = {"LOW_SHARPE": 0.5, "SELF_CORRELATION": -0.4}
        picks = {
            weakest_dimension(margins, temperature=0.05, rng=random.Random(seed))
            for seed in range(20)
        }
        self.assertEqual(picks, {"SELF_CORRELATION"})

    def test_softmax_still_explores(self) -> None:
        margins = {"A": 0.9, "B": 0.95}
        picks = {
            weakest_dimension(margins, temperature=1.0, rng=random.Random(seed))
            for seed in range(50)
        }
        self.assertEqual(picks, {"A", "B"})

    def test_preferred_filter_restricts_candidates(self) -> None:
        margins = {"LOW_SHARPE": -0.9, "SELF_CORRELATION": 0.1}
        pick = weakest_dimension(margins, preferred=["SELF_CORRELATION"], rng=random.Random(0))
        self.assertEqual(pick, "SELF_CORRELATION")

    def test_empty_margins_return_none(self) -> None:
        self.assertIsNone(weakest_dimension({}))

    def test_percentile_and_clamp(self) -> None:
        self.assertEqual(percentile(5, [1, 2, 3]), 1.0)
        self.assertEqual(percentile(0, [1, 2, 3]), 0.0)
        self.assertEqual(percentile(1, []), 0.5)
        self.assertEqual(clamp(5), 1.0)
        self.assertEqual(clamp(-5), -1.0)
        self.assertEqual(clamp(0.25), 0.25)


class StructuralDistanceTests(unittest.TestCase):
    def test_identical_expressions_have_zero_distance(self) -> None:
        from alpha_agent.expr_ast import parse_expression, structural_distance

        node = parse_expression("ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))")
        self.assertEqual(structural_distance(node, node), 0)

    def test_window_change_is_a_minimal_edit(self) -> None:
        from alpha_agent.expr_ast import parse_expression, structural_distance

        left = parse_expression("ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))")
        right = parse_expression("ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 10))")
        self.assertEqual(structural_distance(left, right), 1)

    def test_rewrite_is_a_large_edit(self) -> None:
        from alpha_agent.expr_ast import parse_expression, structural_distance

        left = parse_expression("ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))")
        right = parse_expression("ts_rank(ts_backfill(-scl12_buzz, 20), 22) + group_zscore(low, industry)")
        self.assertGreater(structural_distance(left, right), 4)

    def test_filter_prefers_minimal_edits(self) -> None:
        from alpha_agent.catalog import Catalog
        from alpha_agent.refine_llm import ProposalExpression, filter_candidates

        parent = "ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))"
        catalog = Catalog(
            operators=frozenset({"rank", "ts_mean", "ts_delta", "ts_rank", "group_zscore"}),
            fields=frozenset({"close", "scl12_buzz", "low"}),
        )
        outcome = filter_candidates(
            [
                ProposalExpression(expression="ts_rank(ts_mean(-scl12_buzz, 8), 22) + group_zscore(low, industry)"),
                ProposalExpression(expression="ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 10))"),
            ],
            settings={},
            catalog=catalog,
            parent_expression=parent,
            limit=4,
        )
        self.assertEqual(len(outcome.accepted), 2)
        self.assertIn("ts_delta(close,10)", outcome.accepted[0].expression)
        self.assertEqual(outcome.accepted[0].metadata.get("edit_distance"), 1)


# ---------------------------------------------------------------------------
# search tree
# ---------------------------------------------------------------------------


class TreeConstructionTests(unittest.TestCase):
    def test_root_and_child_linking(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        child = tree.add_child(root.node_id, FakeCandidate("rank(ts_delta(close, 5))"))
        self.assertIsNotNone(child)
        self.assertEqual(child.parent_id, root.node_id)
        self.assertEqual(root.children, [child.node_id])
        self.assertEqual(child.depth, 1)

    def test_children_are_capped(self) -> None:
        tree = SearchTree(max_children=2)
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.add_child(root.node_id, FakeCandidate("rank(high)"))
        self.assertIsNone(tree.add_child(root.node_id, FakeCandidate("rank(low)")))

    def test_depth_is_capped(self) -> None:
        tree = SearchTree(max_depth=1)
        root = tree.add_root(FakeCandidate("rank(close)"))
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        self.assertIsNotNone(child)
        self.assertIsNone(tree.add_child(child.node_id, FakeCandidate("rank(high)")))

    def test_behavioral_signature_is_attached(self) -> None:
        tree = SearchTree()
        node = tree.add_root(FakeCandidate("rank(rank(close))"))
        self.assertIsNotNone(node.behavioral_sig)
        plain = tree.add_root(FakeCandidate("rank(close)"))
        self.assertEqual(node.behavioral_sig, plain.behavioral_sig)


class TreeSelectionTests(unittest.TestCase):
    def test_pending_children_are_selected_first(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=100.0)
        pending = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        selected = tree.select(count=1)
        self.assertEqual([node.node_id for node in selected], [pending.node_id])

    def test_uct_prefers_higher_reward_nodes(self) -> None:
        tree = SearchTree()
        good = tree.add_root(FakeCandidate("rank(close)"))
        bad = tree.add_root(FakeCandidate("rank(open)"))
        tree.record(good.node_id, reward=900.0)
        tree.record(bad.node_id, reward=-500.0)
        for _ in range(5):
            tree.record(bad.node_id, reward=-500.0)
        selected = tree.select(count=1, uct_c=0.1)
        self.assertEqual(selected[0].node_id, good.node_id)

    def test_selection_does_not_repeat_within_a_batch(self) -> None:
        tree = SearchTree()
        nodes = [tree.add_root(FakeCandidate(f"rank(ts_mean(close, {n}))")) for n in (5, 10, 20)]
        for node in nodes:
            tree.record(node.node_id, reward=100.0)
        selected = tree.select(count=3)
        self.assertEqual(len({node.node_id for node in selected}), 3)

    def test_pending_children_win_even_when_parent_is_full(self) -> None:
        """A proposed-but-unevaluated child is always cheaper than proposing more."""
        tree = SearchTree(max_children=1)
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=10.0)
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        selected = tree.select(count=1)
        self.assertEqual([node.node_id for node in selected], [child.node_id])

    def test_fully_expanded_parents_are_not_reselected(self) -> None:
        tree = SearchTree(max_children=1)
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=10.0)
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(child.node_id, reward=20.0)
        selected = {node.node_id for node in tree.select(count=5)}
        self.assertNotIn(root.node_id, selected)

    def test_select_zero_returns_empty(self) -> None:
        tree = SearchTree()
        tree.add_root(FakeCandidate("rank(close)"))
        self.assertEqual(tree.select(count=0), [])


class TreeBackpropTests(unittest.TestCase):
    def test_reward_propagates_to_ancestors(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        grandchild = tree.add_child(child.node_id, FakeCandidate("rank(high)"))
        tree.record(grandchild.node_id, reward=750.0)
        self.assertEqual(tree.nodes[child.node_id].q_value, 750.0)
        self.assertEqual(tree.nodes[root.node_id].q_value, 750.0)
        self.assertGreater(tree.nodes[root.node_id].visits, 1)

    def test_backprop_keeps_the_maximum_not_the_mean(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        first = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        second = tree.add_child(root.node_id, FakeCandidate("rank(high)"))
        tree.record(first.node_id, reward=800.0)
        tree.record(second.node_id, reward=-200.0)
        self.assertEqual(tree.nodes[root.node_id].q_value, 800.0)

    def test_record_sets_metrics_and_state(self) -> None:
        tree = SearchTree()
        node = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(
            node.node_id,
            reward=42.0,
            margins={"LOW_SHARPE": 0.1},
            e_vector={"LOW_SHARPE": 0.1},
            alpha_id="ABC123",
        )
        stored = tree.nodes[node.node_id]
        self.assertEqual(stored.state, EVALUATED)
        self.assertEqual(stored.alpha_id, "ABC123")
        self.assertEqual(stored.margins, {"LOW_SHARPE": 0.1})
        self.assertEqual(stored.reward, 42.0)

    def test_best_ignores_unevaluated_nodes(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(root.node_id, reward=10.0)
        self.assertEqual(tree.best().node_id, root.node_id)


class TreePersistenceTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(child.node_id, reward=321.0, alpha_id="X1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search_tree.json"
            tree.save(path)
            reloaded = SearchTree.load(path, objectives=("quality", "divergence"))
        self.assertEqual(len(reloaded.nodes), 2)
        self.assertEqual(reloaded.roots, tree.roots)
        self.assertEqual(reloaded.nodes[child.node_id].reward, 321.0)
        self.assertEqual(reloaded.nodes[root.node_id].q_value, 321.0)

    def test_stats_shape(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=5.0)
        stats = tree.stats()
        for key in ("nodes", "evaluated", "pending", "max_depth", "best_reward"):
            self.assertIn(key, stats)
        self.assertEqual(stats["evaluated"], 1)

    def test_pruning_bounds_node_count(self) -> None:
        tree = SearchTree(max_nodes=8, max_children=4)
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        for index in range(30):
            tree.add_child(root.node_id, FakeCandidate(f"rank(ts_mean(close, {index + 2}))"))
        self.assertLessEqual(len(tree.nodes), 12)

    def test_known_signatures_collects_evaluated_and_pending(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        signatures = tree.known_signatures()
        self.assertEqual(len(signatures), 2)


# ---------------------------------------------------------------------------
# families and Pareto fronts
# ---------------------------------------------------------------------------


class FamilyTests(unittest.TestCase):
    def test_lineage_family_combines_groups(self) -> None:
        self.assertEqual(lineage_family("ts_mean(-scl12_buzz, 9) + group_rank(close, industry)"), "pv+scl12")
        self.assertEqual(lineage_family("rank(implied_volatility_mean_10)"), "vol")

    def test_group_keys_are_not_data_and_price_variants_share_a_group(self) -> None:
        self.assertEqual(data_groups("group_rank(ts_mean(close, 20), industry)"), ("pv",))
        self.assertEqual(data_groups("rank(open)"), data_groups("rank(close)"))
        self.assertEqual(data_groups("rank("), ())


class HypervolumeTests(unittest.TestCase):
    def test_two_dimensions_match_hand_computation(self) -> None:
        # Step function: [0,1] height 2, [1,2] height 1 -> 3.0
        self.assertAlmostEqual(hypervolume([(1.0, 2.0), (2.0, 1.0)]), 3.0, places=9)

    def test_three_dimensions_match_inclusion_exclusion(self) -> None:
        a, b = (1.0, 0.5, 0.5), (0.5, 1.0, 1.0)
        overlap = 0.5 * 0.5 * 0.5
        self.assertAlmostEqual(hypervolume([a, b]), 0.25 + 0.5 - overlap, places=9)

    def test_dominated_and_non_positive_points_add_nothing(self) -> None:
        self.assertAlmostEqual(hypervolume([(0.5, 0.5), (0.2, 0.2)]), 0.25, places=9)
        self.assertEqual(hypervolume([(0.0, 0.5)]), 0.0)
        self.assertEqual(hypervolume([]), 0.0)

    def test_normalisation_uses_fixed_bounds(self) -> None:
        self.assertEqual(normalize_objectives({"quality": -1.0, "divergence": 1.0}, ("quality", "divergence")), (0.0, 1.0))
        self.assertEqual(normalize_objectives({"quality": 3.0, "divergence": 0.5}, ("quality", "divergence")), (1.0, 0.5))
        self.assertIsNone(normalize_objectives({"quality": 0.0}, ("quality", "divergence")))

    def test_raw_metrics_scale_without_thresholds(self) -> None:
        names = ("sharpe", "fitness", "turnover")
        mid = normalize_objectives({"sharpe": 1.0, "fitness": 1.5, "turnover": 1.5}, names)
        self.assertEqual(len(mid), 3)
        self.assertTrue(all(0.0 < value < 1.0 for value in mid), "possible values stay inside the box")
        # Turnover is lower-is-better: less turnover, larger coordinate.
        low, high = (normalize_objectives({"sharpe": 0, "fitness": 0, "turnover": t}, names)[2] for t in (0.1, 0.6))
        self.assertGreater(low, high)
        # Clamping is the permissive default; strict mode rejects instead, so
        # an out-of-range metric can never pass for a legitimate Pareto point.
        self.assertEqual(normalize_objectives({"sharpe": -5, "fitness": 9, "turnover": 4}, names), (0.0, 1.0, 0.0))
        self.assertIsNone(normalize_objectives({"sharpe": -5, "fitness": 0, "turnover": 0.2}, names, strict=True))
        self.assertIsNone(normalize_objectives({"sharpe": 0, "fitness": 0, "turnover": 4}, names, strict=True))
        self.assertTrue(dominates((0.5, 0.5), (0.5, 0.4)))
        self.assertFalse(dominates((0.5, 0.5), (0.5, 0.5)))

    def test_turnover_above_one_stays_visible(self) -> None:
        """5.3% of live results have turnover >= 1.0; the old bound hid them all."""
        names = ("sharpe", "turnover")
        vector = normalize_objectives({"sharpe": 0.5, "turnover": 1.5}, names, strict=True)
        self.assertIsNotNone(vector)
        self.assertTrue(is_feasible(vector), "still inside the reference box")
        self.assertGreater(vector[1], 0.0, "it is inside the box, not on the reference corner")

    def test_default_objectives_are_not_collinear(self) -> None:
        names = set(DEFAULT_OBJECTIVES)
        self.assertNotIn("fitness", names, "r=0.94 with sharpe: a redundant axis")
        self.assertNotIn("returns", names, "r=0.94 with sharpe: a redundant axis")
        self.assertIn("margin", names, "the most independent quality axis (r=0.37)")


class ParetoFrontTests(unittest.TestCase):
    def test_insert_keeps_only_non_dominated(self) -> None:
        front = ParetoFront()
        self.assertTrue(front.insert("a", (0.5, 0.2)))
        self.assertTrue(front.insert("b", (0.2, 0.5)))
        self.assertFalse(front.insert("c", (0.1, 0.1)))
        self.assertTrue(front.insert("d", (0.6, 0.6)))
        self.assertEqual(front.keys(), {"d"})

    def test_epsilon_rejects_near_duplicates(self) -> None:
        front = ParetoFront(epsilon=0.02)
        front.insert("a", (0.50, 0.50))
        self.assertFalse(front.insert("b", (0.51, 0.49)), "a variant of the same trade-off")
        self.assertTrue(front.insert("c", (0.60, 0.30)))
        self.assertEqual(front.keys(), {"a", "c"})

    def test_capacity_drops_the_smallest_contributor(self) -> None:
        front = ParetoFront(capacity=2)
        front.insert("low_corner", (0.9, 0.1))
        front.insert("other_corner", (0.1, 0.9))
        front.insert("middle", (0.5, 0.5))
        # Exclusive volumes: corners 0.04 each, middle 0.16.
        self.assertEqual(len(front), 2)
        self.assertIn("middle", front.keys())

    def test_reinserting_a_key_replaces_it(self) -> None:
        front = ParetoFront()
        front.insert("a", (0.5, 0.5))
        front.insert("a", (0.4, 0.4))
        self.assertEqual(front.vectors(), [(0.4, 0.4)])

    def test_exclusive_volume(self) -> None:
        front = ParetoFront()
        front.insert("a", (1.0, 0.5))
        front.insert("b", (0.5, 1.0))
        self.assertAlmostEqual(front.exclusive_volume({"a"}), 0.25, places=9)
        self.assertAlmostEqual(front.exclusive_volume({"a", "b"}), front.hypervolume(), places=9)
        self.assertEqual(front.exclusive_volume({"zzz"}), 0.0)


# ---------------------------------------------------------------------------
# MCTS-AHD selection: normalised UCT, descent, progressive widening
# ---------------------------------------------------------------------------


class NormalisedUctTests(unittest.TestCase):
    def test_exploration_is_comparable_to_normalised_q(self) -> None:
        """Raw rewards in the hundreds used to drown the exploration term."""
        tree = SearchTree()
        best = tree.add_root(FakeCandidate("rank(close)"))
        rarely_seen = tree.add_root(FakeCandidate("rank(scl12_buzz)"))
        tree.record(best.node_id, reward=900.0)
        tree.record(rarely_seen.node_id, reward=850.0)
        best.attempts = 50
        tree.root_visits = 51
        # With lambda=1.0 the barely tried, slightly worse node wins.
        winner = max((best, rarely_seen), key=lambda node: tree.uct_score(node, 1.0))
        self.assertEqual(winner.node_id, rarely_seen.node_id)
        self.assertAlmostEqual(tree.normalized_q(best), 1.0)
        self.assertAlmostEqual(tree.normalized_q(rarely_seen), 0.0)

    def test_optimism_decays_with_attempts_not_with_visits(self) -> None:
        """A branch whose proposals all fail is tried, not visited: beta must fall."""
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        tree.root_visits = 40
        before = tree.exploration_width(root, 1.0)
        root.attempts = 9          # nine expansions produced nothing evaluable
        after = tree.exploration_width(root, 1.0)
        self.assertEqual(root.visits, 1, "visits never moved: that was the bug")
        self.assertLess(after, before / 2.0)

    def test_errors_do_not_set_the_q_floor(self) -> None:
        tree = SearchTree()
        ok = tree.add_root(FakeCandidate("rank(close)"))
        bad = tree.add_root(FakeCandidate("rank(open)"))
        tree.record(ok.node_id, reward=-300.0)
        tree.record(bad.node_id, reward=-10_000.0, state="error")
        self.assertEqual(tree.q_bounds(), (-300.0, -300.0))

    def test_error_child_does_not_reach_parent_q(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=-500.0)
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(child.node_id, reward=-10_000.0, state="error")
        self.assertEqual(tree.nodes[root.node_id].q_value, -500.0)


class PlanTests(unittest.TestCase):
    def test_empty_tree_widens_the_virtual_root(self) -> None:
        self.assertEqual(SearchTree().plan(), [(None, "widen")])

    def test_descends_to_best_leaf(self) -> None:
        tree = SearchTree()
        good = tree.add_root(FakeCandidate("rank(close)"))
        weak = tree.add_root(FakeCandidate("rank(scl12_buzz)"))
        tree.record(good.node_id, reward=500.0)
        tree.record(weak.node_id, reward=-500.0)
        child = tree.add_child(good.node_id, FakeCandidate("rank(ts_delta(close, 5))"))
        tree.record(child.node_id, reward=700.0)
        jobs = tree.plan(uct_c=0.0, widen_alpha=0.0, value="scalar")
        self.assertEqual(jobs[-1], (tree.nodes[child.node_id], "leaf"))

    def test_progressive_widening_at_root(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        tree.root_visits = 9  # floor(9 ** 0.5) = 3 > 1 root
        jobs = tree.plan(uct_c=0.0, widen_alpha=0.5, value="scalar")
        self.assertEqual(jobs[0], (None, "widen"))
        self.assertEqual(jobs[-1][1], "leaf")

    def test_max_roots_caps_widening(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        tree.root_visits = 100
        jobs = tree.plan(widen_alpha=0.5, max_roots=1, value="scalar")
        self.assertNotIn((None, "widen"), jobs)

    def test_bonus_steers_descent(self) -> None:
        tree = SearchTree()
        a = tree.add_root(FakeCandidate("rank(close)"))
        b = tree.add_root(FakeCandidate("rank(scl12_buzz)"))
        tree.record(a.node_id, reward=10.0)
        tree.record(b.node_id, reward=9.0)
        jobs = tree.plan(uct_c=0.0, widen_alpha=0.0, bonus=lambda node: 5.0 if node is b else 0.0, value="scalar")
        self.assertEqual(jobs[-1][0].node_id, b.node_id)

    def test_full_node_is_skipped_for_an_expandable_descendant(self) -> None:
        tree = SearchTree(max_children=1)
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(child.node_id, reward=2.0)
        jobs = tree.plan(uct_c=0.0, widen_alpha=0.0, value="scalar")
        self.assertEqual(jobs[-1], (tree.nodes[child.node_id], "leaf"))


class PersistenceTests(unittest.TestCase):
    def test_root_visits_roundtrip_and_legacy_default(self) -> None:
        tree = SearchTree()
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.json"
            tree.save(path)
            self.assertEqual(SearchTree.load(path).root_visits, tree.root_visits)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("root_visits")
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(SearchTree.load(path).root_visits, tree.nodes[root.node_id].visits)


class EngineHelperTests(unittest.TestCase):
    def test_child_family_inherits_only_within_the_same_data_groups(self) -> None:
        from alpha_agent.engine import AlphaResearchAgent

        parent = FakeCandidate("ts_mean(-scl12_buzz, 9) + group_rank(close, industry)", family="social_price_repair")
        same = AlphaResearchAgent._child_family(parent, "ts_mean(-scl12_buzz, 12) + rank(open)")
        other = AlphaResearchAgent._child_family(parent, "rank(ebit / enterprise_value)")
        self.assertEqual(same, "social_price_repair")
        self.assertEqual(other, "fundamental")

    def test_backfill_objectives_puts_old_nodes_on_the_front(self) -> None:
        from alpha_agent.engine import AlphaResearchAgent
        from alpha_agent.novelty_pool import NoveltyPool

        tree = SearchTree()
        old = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(old.node_id, reward=1.0, alpha_id="A1",
                    margins={"LOW_SHARPE": -0.2, "LOW_FITNESS": 0.1, "SELF_CORRELATION": -0.3})
        blind = tree.add_root(FakeCandidate("rank(open)"))
        tree.record(blind.node_id, reward=1.0)
        records = [{"status": "ok", "alpha_id": "A1", "expression": "rank(close)",
                    "metrics": {"sharpe": 1.1, "fitness": 0.7, "turnover": 0.2, "margin": 0.001}}]
        self.assertEqual(len(tree.front), 0)
        self.assertEqual(AlphaResearchAgent._backfill_objectives(tree, NoveltyPool(), records), 2)
        self.assertEqual(
            old.objectives,
            {"sharpe": 1.1, "fitness": 0.7, "turnover": 0.2, "margin": 0.001,
             "quality": -0.2, "correlation": -0.3, "divergence": 1.0},
        )
        self.assertEqual(blind.objectives, {"divergence": 1.0}, "no record: metrics stay unknown")
        self.assertEqual(tree.front.keys(), {old.node_id})
        coverage = tree.objective_coverage()
        self.assertEqual((coverage["ready"], coverage["incomplete"]), (1, 1))
        self.assertEqual(AlphaResearchAgent._backfill_objectives(tree, NoveltyPool(), records), 0)

    def test_metric_objectives_skip_absent_metrics(self) -> None:
        from alpha_agent.engine import AlphaResearchAgent

        record = {"metrics": {"sharpe": 1.2, "fitness": None, "turnover": "0.3", "returns": 0.1}}
        self.assertEqual(
            AlphaResearchAgent._metric_objectives(record),
            {"sharpe": 1.2, "turnover": 0.3, "returns": 0.1},
        )

    def test_leaf_operators_are_the_full_mcts_ahd_set(self) -> None:
        """The paper expands a leaf with every operator, whatever blocks it."""
        from alpha_agent.refine_llm import leaf_operators

        for e_vector in ({"SELF_CORRELATION": -0.2, "LOW_SHARPE": 0.1}, {"LOW_SHARPE": -0.2}):
            self.assertEqual(leaf_operators(e_vector), ("m1", "m2", "e2", "s1"))
        self.assertNotIn("e1", leaf_operators({"LOW_SHARPE": -0.2}), "e1 belongs to the virtual root")


class MultiObjectiveTreeTests(unittest.TestCase):
    def _root(self, tree, expression, reward, quality, divergence):
        node = tree.add_root(FakeCandidate(expression))
        tree.record(node.node_id, reward=reward, objectives={"quality": quality, "divergence": divergence})
        return node

    def test_front_backprops_to_every_ancestor_and_the_root(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        root = self._root(tree, "rank(close)", 1.0, 0.0, 0.2)
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(child.node_id, reward=2.0, objectives={"quality": 0.5, "divergence": 0.2})
        grandchild = tree.add_child(child.node_id, FakeCandidate("rank(vwap)"))
        tree.record(grandchild.node_id, reward=0.5, objectives={"quality": -0.5, "divergence": 0.9})
        self.assertEqual(tree.fronts[root.node_id].keys(), {child.node_id, grandchild.node_id})
        self.assertEqual(tree.fronts[child.node_id].keys(), {child.node_id, grandchild.node_id})
        self.assertEqual(tree.fronts[grandchild.node_id].keys(), {grandchild.node_id})
        self.assertEqual(tree.front.keys(), {child.node_id, grandchild.node_id})
        self.assertEqual(tree.front_nodes()[0].node_id, child.node_id)

    def test_ohvi_without_optimism_is_the_exclusive_contribution(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"))
        a = self._root(tree, "rank(close)", 10.0, 0.5, 0.2)
        b = self._root(tree, "rank(scl12_buzz)", 5.0, 0.1, 0.9)
        for node in (a, b):
            expected = tree.front.exclusive_volume(tree.fronts[node.node_id].keys())
            self.assertGreater(expected, 0.0)
            self.assertAlmostEqual(tree.ohvi(node, beta=0.0), expected, places=9)

    def test_optimism_reaches_near_dominated_subtrees_only(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        self._root(tree, "rank(close)", 10.0, 0.5, 0.5)
        near = self._root(tree, "rank(open)", 9.0, 0.45, 0.45)
        far = self._root(tree, "rank(anl4_eps_mean)", 1.0, -0.4, 0.1)
        self.assertEqual(tree.ohvi(near, beta=0.0), 0.0)
        self.assertGreater(tree.ohvi(near, beta=0.1), 0.0)
        self.assertEqual(tree.ohvi(far, beta=0.1), 0.0)

    def test_no_threshold_every_trade_off_counts(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"))
        good = self._root(tree, "rank(close)", 10.0, 0.2, 0.3)
        volume = tree.front.hypervolume()
        unusual = self._root(tree, "rank(anl4_eps_mean)", 1.0, -0.9, 0.95)
        self.assertGreater(tree.front.hypervolume(), volume, "a weak but unique alpha still adds volume")
        self.assertEqual({node.node_id for node in tree.front_nodes()}, {good.node_id, unusual.node_id})
        corner = self._root(tree, "rank(vwap)", 1.0, -1.0, 0.99)
        self.assertNotIn(corner.node_id, {node.node_id for node in tree.front_nodes()}, "on the reference: no volume")

    def test_front_gap_is_the_shift_needed_to_escape_domination(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        a = self._root(tree, "rank(close)", 1.0, 0.5, 0.5)      # (0.75, 0.5)
        near = self._root(tree, "rank(open)", 1.0, 0.45, 0.45)  # (0.725, 0.45)
        far = self._root(tree, "rank(vwap)", 1.0, -0.4, 0.1)    # (0.3, 0.1)
        self.assertAlmostEqual(tree.front_gap(a), 0.0, places=9)
        self.assertAlmostEqual(tree.front_gap(near), 0.025, places=9)
        self.assertAlmostEqual(tree.front_gap(far), 0.4, places=9)
        self.assertIsNone(tree.front_gap(tree.add_root(FakeCandidate("rank(low)"))))

    def test_plan_falls_back_to_the_least_tried_when_nothing_has_a_front(self) -> None:
        """No objective vectors at all: still a sane descent, and it says so."""
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        first = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(first.node_id, reward=1.0)          # no objectives recorded
        second = tree.add_root(FakeCandidate("rank(open)"))
        tree.record(second.node_id, reward=1.0)
        first.attempts = 9
        jobs = tree.plan(uct_c=0.0)
        self.assertEqual(jobs[-1], (second, "leaf"), "the untried branch, not the node id order")
        self.assertEqual(tree.objective_coverage()["ready"], 0)
        self.assertEqual(tree.objective_coverage()["incomplete"], 2)

    def test_pareto_plan_prefers_the_unique_trade_off_over_the_scalar_best(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"))
        diverse = self._root(tree, "rank(scl12_buzz)", 1.0, 0.1, 0.9)
        best_scalar = self._root(tree, "rank(close)", 100.0, 0.12, 0.1)
        jobs = tree.plan(uct_c=0.0, widen_alpha=0.0)
        self.assertEqual(jobs[-1][0].node_id, diverse.node_id)
        jobs = tree.plan(uct_c=0.0, widen_alpha=0.0, value="scalar")
        self.assertEqual(jobs[-1][0].node_id, best_scalar.node_id)

    def test_three_objectives_and_unknown_names(self) -> None:
        tree = SearchTree(objectives=("quality", "correlation", "divergence"))
        node = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(node.node_id, reward=1.0, objectives={"quality": 0.0, "correlation": 0.0, "divergence": 0.5})
        self.assertAlmostEqual(tree.front.hypervolume(), 0.5 * 0.5 * 0.5, places=9)
        with self.assertRaises(ValueError):
            SearchTree(objectives=("alpha_score",))

    def test_fronts_are_rebuilt_on_load_and_legacy_fields_migrate(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"))
        a = self._root(tree, "rank(close)", 1.0, 0.5, 0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.json"
            tree.save(path)
            reloaded = SearchTree.load(path, objectives=("quality", "divergence"))
            self.assertEqual(reloaded.front.keys(), {a.node_id})
            payload = json.loads(path.read_text(encoding="utf-8"))
            for raw in payload["nodes"]:
                objectives = raw.pop("objectives")
                raw["quality"], raw["divergence"] = objectives["quality"], objectives["divergence"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            migrated = SearchTree.load(path, objectives=("quality", "divergence"))
        self.assertEqual(migrated.nodes[a.node_id].objectives, {"quality": 0.5, "divergence": 0.2})
        self.assertAlmostEqual(migrated.front.hypervolume(), reloaded.front.hypervolume(), places=9)

    def test_error_and_objective_free_nodes_stay_off_the_front(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"))
        plain = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(plain.node_id, reward=1.0)
        bad = tree.add_root(FakeCandidate("rank(open)"))
        tree.record(bad.node_id, reward=-10_000.0, state="error", objectives={"quality": 1.0, "divergence": 1.0})
        self.assertEqual(len(tree.front), 0)
        self.assertEqual(tree.plan(uct_c=0.0, widen_alpha=0.0)[-1], (plain, "leaf"))


class MoMctsRegressionTests(unittest.TestCase):
    """The four defects that made ``tree_value="ohvi"`` degenerate in practice."""

    def _root(self, tree, expression, *objectives, reward=1.0):
        node = tree.add_root(FakeCandidate(expression))
        tree.record(
            node.node_id,
            reward=reward,
            objectives={"quality": objectives[0], "divergence": objectives[1]},
        )
        return node

    def test_scores_use_the_same_reference_set_for_every_child(self) -> None:
        """The front owner used to be scored against (almost) nothing.

        ``G_-c`` removed only the scored subtree's own points, so a branch
        holding 14 of 15 front points was measured against 1 baseline point
        while every rival was measured against 14 — the owner won every
        descent. Both children now share one baseline: the points the
        competing branches own.
        """
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        owner = self._root(tree, "rank(close)", 0.8, 0.2)     # (0.90, 0.20)
        rival = self._root(tree, "rank(open)", 0.34, 0.67)    # (0.67, 0.67)
        shared = set(tree.fronts[owner.node_id].keys()) | set(tree.fronts[rival.node_id].keys())
        beta = 0.1
        self.assertAlmostEqual(
            tree.ohvi(owner, beta=beta, reference=shared),
            hypervolume([(min(1.0, 0.90 + beta), min(1.0, 0.20 + beta))]),
            places=9,
        )
        self.assertAlmostEqual(
            tree.ohvi(rival, beta=beta, reference=shared),
            hypervolume([(0.67 + beta, 0.67 + beta)]),
            places=9,
        )

    def test_the_front_owner_does_not_win_every_descent(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        owner = self._root(tree, "rank(close)", 0.8, 0.2)
        for index, (quality, divergence) in enumerate(
            ((0.4, 0.4), (0.0, 0.5), (-0.4, 0.7), (-0.8, 0.9))
        ):
            child = tree.add_child(owner.node_id, FakeCandidate(f"rank(ts_mean(close, {index + 2}))"))
            tree.record(child.node_id, reward=1.0, objectives={"quality": quality, "divergence": divergence})
        rival = self._root(tree, "rank(open)", 0.34, 0.67)    # one point, comparable volume
        self.assertGreater(
            len(tree.fronts[owner.node_id]), len(tree.fronts[rival.node_id]),
            "the owner really does own most of the front",
        )
        winners = set()
        for _ in range(6):
            jobs = tree.plan(uct_c=0.1)
            leaf = jobs[-1][0]
            winners.add(tree.root_of(leaf.node_id).node_id)
        self.assertIn(rival.node_id, winners, "the never-tried branch must get the descent")
        self.assertIn(owner.node_id, winners, "and the owner is not abandoned either")

    def test_error_children_do_not_consume_child_slots(self) -> None:
        """Four failed refinements used to brick a parent for the whole run."""
        tree = SearchTree(max_children=1)
        root = tree.add_root(FakeCandidate("rank(close)"))
        tree.record(root.node_id, reward=1.0)
        for index in range(3):
            child = tree.add_child(root.node_id, FakeCandidate(f"rank(ts_delta(close, {index + 2}))"))
            tree.record(child.node_id, reward=-10_000.0, state="error")
        self.assertTrue(tree.has_room(root))
        survivor = tree.add_child(root.node_id, FakeCandidate("rank(vwap)"))
        self.assertIsNotNone(survivor)
        tree.record(survivor.node_id, reward=1.0)
        self.assertFalse(tree.has_room(root), "a productive child does take the slot")

    def test_inner_node_widens_only_after_its_front_stalls(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0, widen_patience=2)
        root = self._root(tree, "rank(close)", 0.8, 0.2)
        child = tree.add_child(root.node_id, FakeCandidate("rank(open)"))
        tree.record(child.node_id, reward=1.0, objectives={"quality": 0.4, "divergence": 0.4})
        root.visits = 9  # floor(9 ** 0.5) = 3 > 1 child: the birth rate allows it
        self.assertNotIn((root, "widen"), tree.plan(uct_c=0.0, max_roots=1, value="scalar"))
        tree.stalls[root.node_id] = 2
        self.assertIn((root, "widen"), tree.plan(uct_c=0.0, max_roots=1, value="scalar"))

    def test_the_internal_front_is_the_reported_front(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        on_the_box = self._root(tree, "rank(close)", -1.0, 0.5)   # quality -1 -> 0.0
        self.assertEqual(len(tree.front), 0)
        self.assertEqual(tree.front_nodes(), [])
        self.assertEqual(tree.objective_coverage()["out_of_range"], 0, "0.0 is in range, not rejected")
        good = self._root(tree, "rank(open)", 0.5, 0.5)
        self.assertEqual(len(tree.front), len(tree.front_nodes()))
        self.assertNotIn(on_the_box.node_id, tree.front.keys())
        self.assertIn(good.node_id, tree.front.keys())

    def test_out_of_range_objective_is_rejected_and_counted(self) -> None:
        tree = SearchTree(objectives=("quality", "divergence"), epsilon=0.0)
        node = self._root(tree, "rank(close)", 9.0, 0.5)          # quality 9 is impossible
        self.assertEqual(tree.objective_coverage()["out_of_range"], 1)
        self.assertEqual(len(tree.front), 0)
        self.assertIsNone(tree.objective_vector(node))


class ResumedTreeDepthTests(unittest.TestCase):
    def test_a_resumed_tree_takes_the_configured_max_depth(self) -> None:
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "tree.json"
        SearchTree(path, max_depth=5).save(path)
        self.assertEqual(SearchTree.load(path).max_depth, 5, "file value is the fallback")
        self.assertEqual(SearchTree.load(path, max_depth=3).max_depth, 3)


class DiverseSeedingTests(unittest.TestCase):
    def test_picks_spread_seeds_and_requeues_the_rest(self) -> None:
        import alpha_research_pipeline as pipeline
        from alpha_agent.engine import AlphaResearchAgent

        seeds = [
            pipeline.Candidate(expression=expr, settings={}, family="alpha101", idea_name=f"a{i}")
            for i, expr in enumerate([
                "rank(ts_delta(close, 5))",
                "rank(ts_delta(close, 6))",
                "ts_corr(rank(volume), rank(vwap), 10)",
                "-1 * ts_rank(ts_std_dev(returns, 20), 5)",
            ])
        ]

        class Toolbox:
            seed_queue = list(seeds)

        agent = AlphaResearchAgent.__new__(AlphaResearchAgent)
        agent.toolbox = Toolbox()
        tree = SearchTree()
        added = agent._seed_hybrid_roots(tree, count=3, diverse=True)
        self.assertEqual(added, 3)
        expressions = {node.expression for node in tree.children_of(None)}
        # The two near-duplicate delta seeds are not both chosen.
        self.assertFalse({"rank(ts_delta(close, 5))", "rank(ts_delta(close, 6))"} <= expressions)
        self.assertEqual(len(agent.toolbox.seed_queue), 1)
        self.assertEqual({node.family for node in tree.children_of(None)} & {"alpha101"}, set())

    def _seed_roots(self, seeds, count, pool=None):
        from alpha_agent.engine import AlphaResearchAgent

        class Toolbox:
            seed_queue = list(seeds)

        agent = AlphaResearchAgent.__new__(AlphaResearchAgent)
        agent.toolbox = Toolbox()
        tree = SearchTree()
        agent._seed_hybrid_roots(tree, count=count, pool=pool, diverse=True)
        return tree, agent.toolbox

    def test_field_swaps_are_not_roots_and_stay_queued(self) -> None:
        import alpha_research_pipeline as pipeline

        original = pipeline.Candidate(expression="rank(vwap - close)", settings={}, family="alpha101",
                                      idea_name="alpha101_042", stage="manual_seed")
        swap = pipeline.Candidate(expression="rank(vwap - cap)", settings={}, family="alpha101",
                                  idea_name="alpha101_042.swap", stage="field_swap")
        tree, toolbox = self._seed_roots([swap, original], count=2)
        self.assertEqual([node.expression for node in tree.children_of(None)], ["rank(vwap - close)"])
        self.assertEqual(toolbox.seed_queue, [swap])

    def test_distance_ties_go_to_the_seed_the_pool_covers_least(self) -> None:
        import alpha_research_pipeline as pipeline

        seeds = [
            pipeline.Candidate(expression=expr, settings={}, family="alpha101", idea_name=name)
            for name, expr in [
                ("a_first", "rank(ts_delta(close, 5))"),
                ("z_crowded", "ts_corr(volume, vwap, 10)"),
                ("b_fresh", "ts_corr(volume, vwap, 20)"),
            ]
        ]

        class Pool:
            def novelty(self, expression, settings=None):
                return {"rank(ts_delta(close, 5))": 0.9, "ts_corr(volume, vwap, 10)": 0.1}.get(expression, 0.8)

        tree, _ = self._seed_roots(seeds, count=2, pool=Pool())
        names = [node.idea_name for node in tree.children_of(None)]
        # The two ts_corr seeds are equally far from the first pick; the name
        # order would take z_crowded, the pool says b_fresh.
        self.assertEqual(names, ["a_first", "b_fresh"])


if __name__ == "__main__":
    unittest.main()
