"""Integration tests for the hybrid search loop.

These run without a network: the toolbox evaluator is replaced by a fake that
returns records built from the real captured check payloads, and the account
pool / operator catalog are disabled via ``allow_offline_fetch=False``.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import alpha_research_pipeline as pipeline
from alpha_agent.config import AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig
from alpha_agent.engine import AlphaResearchAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeToolbox:
    """Stands in for ``ResearchToolbox``: no API, canned records."""

    def __init__(self, agent: "AlphaResearchAgent", frames: list[dict]) -> None:
        self.agent = agent
        real = agent.toolbox
        self.results = real.results
        self.submissions = real.submissions
        self.evaluated_signatures = real.evaluated_signatures
        self.seed_queue = real.seed_queue
        self.all_ordered_seeds = real.all_ordered_seeds
        self.agent_runs_dir = real.agent_runs_dir
        self.state_path = real.state_path
        self._frames = frames or [load_fixture("record_quality_corr_blocked.json")]
        self._index = 0
        self.evaluated: list[dict] = []
        self.popped_seeds = 0
        # Keep the arena plumbing real: the run configures region/universe/delay
        # and every seed must be forced into it.
        self.arena_settings = real.arena_settings

    # -- used by the engine ------------------------------------------------
    def leaderboard(self, *, limit: int = 10) -> list[dict]:
        return pipeline.build_leaderboard(self.results, family_filter=set(), limit=limit)

    def pop_seed_candidates(self, count: int) -> list:
        batch = self.seed_queue[:count]
        self.seed_queue = self.seed_queue[count:]
        self.popped_seeds += len(batch)
        return batch

    def evaluate_candidates(self, candidates, progress_callback=None) -> list[dict]:
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        records = []
        for index, candidate in enumerate(candidates):
            record = {
                **candidate.to_record(),
                "status": "ok",
                "alpha_id": f"FAKE{self._index:02d}{index}",
                "score": frame.get("score", 0.0),
                "metrics": frame.get("metrics", {}),
                "failed_checks": frame.get("failed_checks", []),
                "failed_blocking_checks": frame.get("failed_blocking_checks", []),
                "failed_correlation_checks": frame.get("failed_correlation_checks", []),
                "failed_submission_checks": frame.get("failed_submission_checks", []),
                "quality_checks_ready": frame.get("quality_checks_ready", False),
                "precheck_submit_ready": frame.get("precheck_submit_ready", False),
                "check_diagnostics": frame.get("check_diagnostics", []),
                "summary": frame.get("summary", {}),
            }
            self.results.append(record)
            self.evaluated.append(record)
            records.append(record)
            if progress_callback:
                progress_callback({"type": "candidate_completed", "record": record})
        return records

    def write_state(self) -> dict:
        return pipeline.write_state(self.state_path, self.results, self.submissions, self.leaderboard(limit=10))

    def best_submittable_candidate(self, *, target_alpha_id=None):
        return None

    def submit_record(self, record):  # pragma: no cover - not expected to run
        raise AssertionError("submit must not be attempted in these tests")


def build_runtime(workdir: Path, **overrides) -> AgentRuntimeConfig:
    agent = AgentConfig(
        budget=4,
        max_iterations=3,
        workdir=workdir,
        idea_library=REPO_ROOT / "alpha_pipeline_ideas.json",
        fields_summary=REPO_ROOT / "wqb_data_fields_summary.json",
        search_mode="hybrid",
        tree_batch_size=2,
        refine_proposals=4,
        refine_top_k=1,
        llm_refine_enabled=False,
        allow_offline_fetch=False,
        sleep_between=0.0,
        submission_mode="disabled",
    )
    if overrides:
        agent = replace(agent, **overrides)
    return AgentRuntimeConfig(
        auth=AuthConfig(email=None, password=None, cookie_header=None),
        model=ModelConfig(provider="heuristic"),
        agent=agent,
    )


class HybridRunTests(unittest.TestCase):
    def _run(self, frames=None, **overrides):
        tmp = tempfile.mkdtemp()
        workdir = Path(tmp)
        runtime = build_runtime(workdir, **overrides)
        agent = AlphaResearchAgent(runtime)
        agent.toolbox = FakeToolbox(agent, frames or [])
        result = agent.run()
        return result, workdir, agent

    def test_report_contains_search_block(self) -> None:
        result, workdir, _ = self._run()
        report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
        self.assertIn("search", report)
        search = report["search"]
        for key in ("mode", "tree", "pareto", "filters", "novelty_pool", "catalog", "llm"):
            self.assertIn(key, search)
        self.assertEqual(search["mode"], "hybrid")
        self.assertEqual(search["pareto"]["objectives"], ["sharpe", "turnover", "margin", "divergence"])
        self.assertIn("hypervolume", search["pareto"])
        self.assertEqual(len(search["pareto"]["front"]), search["pareto"]["front_size"])
        # The run must say which value it actually used and how many nodes could
        # take part; a silent loss of the objectives used to be invisible.
        self.assertEqual(search["pareto"]["value"], "ohvi")
        self.assertEqual(search["pareto"]["value_effective"], "ohvi")
        self.assertIsNone(search["pareto"]["disabled_reason"])
        self.assertGreater(search["pareto"]["coverage"]["ready"], 0)
        self.assertEqual(search["tree"]["front_entries"], search["pareto"]["front_size"])

    def test_a_run_without_metrics_falls_back_and_says_so(self) -> None:
        """No metric means no objective vector: OHVI is off, and the run admits it."""
        result, _, agent = self._run(frames=[{"score": 0.0, "metrics": {}}])
        self.assertEqual(result.summary["pareto_value_effective"], "scalar")
        self.assertIsNotNone(result.summary["pareto_disabled_reason"])
        self.assertIn("falling back", result.summary["pareto_disabled_reason"])
        self.assertEqual(result.summary["pareto_ready"], 0)
        blind = [event for event in result.events if event.get("action") == "pareto_blind"]
        self.assertEqual(len(blind), 1, "warned once, not once per iteration")
        self.assertEqual(blind[0]["details"]["coverage"]["ready"], 0)

    def test_summary_reports_search_mode_and_budget(self) -> None:
        result, _, _ = self._run(budget=4)
        self.assertEqual(result.summary["search_mode"], "hybrid")
        self.assertLessEqual(result.summary["evaluated_this_run"], 4)
        self.assertGreater(result.summary["evaluated_this_run"], 0)

    def test_state_files_are_written(self) -> None:
        _, workdir, _ = self._run()
        self.assertTrue((workdir / "search_tree.json").exists())
        tree = json.loads((workdir / "search_tree.json").read_text(encoding="utf-8"))
        self.assertGreater(len(tree["nodes"]), 0)

    def test_filters_are_counted_before_spending_budget(self) -> None:
        result, _, _ = self._run()
        filters = result.summary.get("search_mode") and json.loads(
            Path(result.report_path).read_text(encoding="utf-8")
        )["search"]["filters"]
        self.assertIn("evaluated", filters)
        self.assertGreaterEqual(filters["evaluated"], 1)

    def test_hybrid_never_submits_when_nothing_is_ready(self) -> None:
        result, _, _ = self._run()
        self.assertEqual(result.summary["submission_attempts_this_run"], 0)
        self.assertEqual(result.submissions, [])

    def test_events_keep_the_legacy_shape(self) -> None:
        result, _, _ = self._run()
        self.assertTrue(result.events)
        for event in result.events:
            for key in ("timestamp", "iteration", "stage", "action", "rationale", "details"):
                self.assertIn(key, event)

    def test_budget_is_respected(self) -> None:
        result, _, _ = self._run(budget=2, tree_batch_size=2, max_iterations=10)
        self.assertLessEqual(result.summary["evaluated_this_run"], 2)

    def test_auth_failure_stops_the_run_without_eating_budget(self) -> None:
        """Regression: a 100-sim run once spent its whole budget on 401 errors."""
        tmp = tempfile.mkdtemp()
        workdir = Path(tmp)
        runtime = build_runtime(workdir, budget=50, max_iterations=20)
        agent = AlphaResearchAgent(runtime)

        class AuthFailToolbox(FakeToolbox):
            def evaluate_candidates(self, candidates, progress_callback=None):
                return [
                    {
                        **candidate.to_record(),
                        "status": "error",
                        "error": {
                            "message": "Incorrect authentication credentials.",
                            "status": 401,
                            "url": "/simulations",
                            "payload": {"detail": "Incorrect authentication credentials."},
                        },
                    }
                    for candidate in candidates
                ]

        agent.toolbox = AuthFailToolbox(agent, [])
        result = agent.run()
        self.assertIn("Authentication rejected", result.summary["stop_reason"])
        self.assertEqual(result.summary["evaluated_this_run"], 0)
        self.assertLess(result.summary["iterations_executed"], 5)

    def test_all_error_batches_stop_after_three_tries(self) -> None:
        tmp = tempfile.mkdtemp()
        workdir = Path(tmp)
        runtime = build_runtime(workdir, budget=100, max_iterations=30, tree_batch_size=2)
        agent = AlphaResearchAgent(runtime)

        class BadExpressionToolbox(FakeToolbox):
            def evaluate_candidates(self, candidates, progress_callback=None):
                return [
                    {
                        **candidate.to_record(),
                        "status": "error",
                        "error": {"message": "bad expression", "status": 400},
                    }
                    for candidate in candidates
                ]

        agent.toolbox = BadExpressionToolbox(agent, [])
        result = agent.run()
        self.assertIn("consecutive batches failed", result.summary["stop_reason"])
        self.assertEqual(result.summary["evaluated_this_run"], 0)

    def test_quality_blocked_frame_marks_candidates_not_submit_ready(self) -> None:
        result, _, _ = self._run(frames=[load_fixture("record_quality_corr_blocked.json")])
        self.assertEqual(result.summary["submittable_count"], 0)
        self.assertGreaterEqual(result.summary["quality_ready_count"], 1)

    def test_legacy_mode_dispatches_to_the_legacy_loop(self) -> None:
        """``search_mode=legacy`` must not enter the hybrid loop or write its files."""
        tmp = tempfile.mkdtemp()
        workdir = Path(tmp)
        runtime = build_runtime(workdir, search_mode="legacy")
        agent = AlphaResearchAgent(runtime)

        called = {"legacy": False}

        def fake_legacy():
            called["legacy"] = True
            raise RuntimeError("legacy-loop-entered")

        agent._run_legacy = fake_legacy  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            agent.run()
        self.assertTrue(called["legacy"])
        self.assertFalse((workdir / "search_tree.json").exists())


class DeterministicFilterTests(unittest.TestCase):
    def test_filter_rejects_duplicates_and_unknowns(self) -> None:
        from alpha_agent.catalog import Catalog
        from alpha_agent.refine_llm import (
            DeterministicRefinementProposer,
            ProposalExpression,
            filter_candidates,
        )

        catalog = Catalog(
            operators=frozenset({"rank", "ts_mean", "ts_delta", "group_neutralize", "group_rank"}),
            fields=frozenset({"close", "open", "vwap", "industry"}),
        )
        proposals = [
            ProposalExpression(expression="rank(close)"),
            ProposalExpression(expression="rank(close)"),  # duplicate
            ProposalExpression(expression="bogus_op(close)"),  # unknown operator
            ProposalExpression(expression="rank(not_a_field)"),  # unknown field
            ProposalExpression(expression="rank("),  # unparseable
            ProposalExpression(expression="ts_mean(open, 10)"),  # fine
        ]
        outcome = filter_candidates(proposals, settings={}, catalog=catalog, limit=10)
        self.assertEqual(len(outcome.accepted), 2)
        self.assertEqual(outcome.counts.get("duplicate"), 1)
        self.assertGreaterEqual(outcome.counts.get("invalid_operator_or_field", 0), 2)
        self.assertEqual(outcome.counts.get("parse"), 1)

    def test_deterministic_proposer_mutates_structure_before_wrapping(self) -> None:
        from alpha_agent.refine_llm import DeterministicRefinementProposer

        node = type(
            "Node",
            (),
            {"expression": "ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))", "settings": {}},
        )()
        proposals = DeterministicRefinementProposer().propose(
            node=node, e_vector={"SELF_CORRELATION": -0.2}, k=6
        )
        self.assertTrue(proposals)
        expressions = [p.expression for p in proposals]
        # Canonical rendering drops the redundant parentheses/spacing, so match
        # on the mutated sub-expressions rather than the source text.
        self.assertTrue(
            any("ts_delta(close,10)" in item for item in expressions)
            or any("ts_mean((-scl12_buzz),16)" in item for item in expressions),
            f"expected a window mutation, got {expressions}",
        )
        # No proposal may be a pure monotone wrapper of the whole formula.
        for expression in expressions:
            self.assertFalse(
                expression.startswith("rank(") and "+" in expression,
                f"unexpected whole-formula wrapper: {expression}",
            )

    def test_parent_subtrees_are_exempt_from_the_blacklist(self) -> None:
        """Regression: banning the parent's own motifs rejected 36/36 mutations."""
        from alpha_agent.catalog import Catalog
        from alpha_agent.expr_ast import subtree_hashes, try_parse
        from alpha_agent.refine_llm import ProposalExpression, filter_candidates

        parent = "ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 5))"
        parent_hashes = list(subtree_hashes(try_parse(parent), min_size=3))
        catalog = Catalog(
            operators=frozenset({"rank", "ts_mean", "ts_delta"}),
            fields=frozenset({"close", "scl12_buzz"}),
        )
        outcome = filter_candidates(
            [ProposalExpression(expression="ts_mean(-scl12_buzz, 8) + rank(ts_delta(close, 10))")],
            settings={},
            catalog=catalog,
            blacklist_subtrees=parent_hashes,
            parent_expression=parent,
            limit=4,
        )
        self.assertEqual(len(outcome.accepted), 1, f"rejected: {outcome.counts}")

    def test_unrelated_blacklisted_subtree_still_rejects(self) -> None:
        from alpha_agent.catalog import Catalog
        from alpha_agent.expr_ast import subtree_hashes, try_parse
        from alpha_agent.refine_llm import ProposalExpression, filter_candidates

        catalog = Catalog(
            operators=frozenset({"rank", "ts_mean", "ts_delta", "ts_std_dev"}),
            fields=frozenset({"close", "volume"}),
        )
        banned = list(subtree_hashes(try_parse("ts_std_dev(volume, 20)"), min_size=3))
        outcome = filter_candidates(
            [
                ProposalExpression(
                    expression="ts_mean(-ts_std_dev(volume, 20), 8) + rank(ts_delta(close, 5))"
                )
            ],
            settings={},
            catalog=catalog,
            blacklist_subtrees=banned,
            parent_expression="rank(ts_delta(close, 5))",
            max_subtree_hits=1,
            limit=4,
        )
        self.assertEqual(len(outcome.accepted), 0)
        self.assertEqual(outcome.counts.get("frequent_subtree"), 1)


class SubmittingToolbox(FakeToolbox):
    """Fake toolbox that can actually report and record a submission."""

    def __init__(self, agent: "AlphaResearchAgent", frames: list[dict]) -> None:
        super().__init__(agent, frames)
        self.submitted_alpha_ids: set[str] = set()

    def best_submittable_candidate(self, *, target_alpha_id=None):
        for record in reversed(self.results):
            alpha_id = record.get("alpha_id")
            if not isinstance(alpha_id, str) or alpha_id in self.submitted_alpha_ids:
                continue
            if target_alpha_id and alpha_id != target_alpha_id:
                continue
            if pipeline.record_is_submit_ready(record):
                return record
        return None

    def submit_record(self, record):
        payload = {
            "alpha_id": record["alpha_id"],
            "result": "ok",
            "submitted_at": pipeline.iso_now(),
            "response": {"ok": True},
        }
        self.submissions.append(payload)
        self.submitted_alpha_ids.add(record["alpha_id"])
        return payload


def submit_ready_frame() -> dict:
    """A check payload where nothing blocks a submission."""
    return {
        "status": "ok",
        "score": 420.0,
        "metrics": {
            "sharpe": 1.72,
            "fitness": 1.21,
            "turnover": 0.31,
            "returns": 0.12,
            "drawdown": 0.11,
            "margin": 0.0009,
        },
        "failed_checks": [],
        "failed_blocking_checks": [],
        "failed_correlation_checks": [],
        "failed_submission_checks": [],
        "quality_checks_ready": True,
        "precheck_submit_ready": True,
        "check_diagnostics": [],
        "summary": {},
    }


class SubmissionWiringTests(unittest.TestCase):
    def _run(self, *, frames=None, **overrides):
        workdir = Path(tempfile.mkdtemp())
        runtime = build_runtime(workdir, **overrides)
        agent = AlphaResearchAgent(runtime)
        agent.toolbox = SubmittingToolbox(agent, frames or [submit_ready_frame()])
        result = agent.run()
        return result

    def test_auto_submits_a_strictly_ready_alpha(self) -> None:
        """The hybrid loop has no robustness stage, so nothing ever submitted."""
        result = self._run(submission_mode="auto_approved", max_submissions_per_run=1)
        self.assertEqual(len(result.submissions), 1)
        self.assertEqual(result.submissions[0]["result"], "ok")
        self.assertTrue(result.submissions[0]["alpha_id"].startswith("FAKE"))
        events = [event for event in result.events if event.get("action") == "submit_best"]
        self.assertTrue(any(event["details"].get("result") == "ok" for event in events))

    def test_one_submission_per_run_by_default(self) -> None:
        result = self._run(submission_mode="auto_approved")
        self.assertEqual(len(result.submissions), 1, "BRAIN submissions are final; no spray")

    def test_a_capped_run_submits_nothing(self) -> None:
        result = self._run(submission_mode="auto_approved", max_submissions_per_run=0)
        self.assertEqual(len(result.submissions), 0)

    def test_disabled_mode_never_submits(self) -> None:
        result = self._run(submission_mode="disabled")
        self.assertEqual(len(result.submissions), 0)

    def test_an_alpha_blocked_by_correlation_is_not_submitted(self) -> None:
        """``failed_submission_checks`` is authoritative when the pipeline sets it."""
        frame = {
            **submit_ready_frame(),
            "failed_correlation_checks": ["SELF_CORRELATION"],
            "failed_submission_checks": ["SELF_CORRELATION"],
        }
        result = self._run(frames=[frame], submission_mode="auto_approved")
        self.assertEqual(len(result.submissions), 0)


class OperatorSweepTests(unittest.TestCase):
    """MCTS-AHD expands a selected leaf with every refinement operator."""

    def test_one_leaf_expansion_asks_the_model_for_all_four_operators(self) -> None:
        import alpha_agent.refine_llm as refine

        real = refine.DeterministicRefinementProposer
        asked: list[tuple] = []

        class Spy(real):  # type: ignore[misc,valid-type]
            def propose(self, **kwargs):
                asked.append((kwargs.get("operator"), kwargs.get("node") is None))
                return super().propose(**kwargs)

        refine.DeterministicRefinementProposer = Spy
        try:
            workdir = Path(tempfile.mkdtemp())
            runtime = build_runtime(workdir, budget=8, max_iterations=2)
            agent = AlphaResearchAgent(runtime)
            agent.toolbox = FakeToolbox(agent, [submit_ready_frame()])
            agent.run()
        finally:
            refine.DeterministicRefinementProposer = real

        operators = {operator for operator, _ in asked}
        self.assertTrue(asked, "the run never asked the proposer for anything")
        self.assertLessEqual(
            {"m1", "m2", "e2", "s1"},
            operators,
            f"a leaf must be expanded by every operator, got {asked}",
        )
        for operator, is_root in asked:
            if operator == "e1":
                self.assertTrue(is_root, "e1 belongs to the virtual root, not to a leaf")


class ArenaPromptTests(unittest.TestCase):
    """The proposer must know which region/universe it is searching."""

    def setUp(self) -> None:
        import os

        os.environ["FAKE_LLM_KEY"] = "test"
        self._added = True

    def tearDown(self) -> None:
        import os

        os.environ.pop("FAKE_LLM_KEY", None)

    def _capture_prompt(self, *, node=None, operator="e1", arena=None) -> dict:
        from alpha_agent.config import ModelConfig
        from alpha_agent.refine_llm import LLMRefinementProposer

        proposer = LLMRefinementProposer(
            ModelConfig(provider="openai", model="m", api_key_env="FAKE_LLM_KEY"), max_calls=1
        )
        captured: dict = {}

        def fake_completion(*, body, api_key):
            captured["prompt"] = json.loads(body["messages"][1]["content"])
            return {"choices": [{"message": {"content": json.dumps({"expressions": []})}}]}

        proposer._chat_completion = fake_completion  # type: ignore[assignment]
        proposer._propose_via_model(
            node=node,
            e_vector={},
            blacklist=(),
            operators=None,
            context={"arena": arena} if arena else {},
            fields=None,
            operator=operator,
        )
        return captured.get("prompt") or {}

    def test_a_new_idea_is_told_the_arena(self) -> None:
        prompt = self._capture_prompt(operator="e1", arena={"region": "USA", "universe": "TOP3000", "delay": 1})
        self.assertEqual(prompt["arena"], {"region": "USA", "universe": "TOP3000", "delay": 1})
        self.assertIn("USA/TOP3000", prompt["arena_note"])
        self.assertIn("submittable in that arena", prompt["arena_note"])
        self.assertEqual(prompt["run_context"]["arena"]["universe"], "TOP3000")

    def test_another_arena_reaches_the_prompt(self) -> None:
        prompt = self._capture_prompt(operator="e1", arena={"region": "EUR", "universe": "TOP1200", "delay": 0})
        self.assertIn("EUR/TOP1200", prompt["arena_note"])
        self.assertEqual(prompt["arena"]["delay"], 0)

    def test_arena_note_defines_universes_and_the_sub_universe_gate(self) -> None:
        prompt = self._capture_prompt(operator="e1", arena={"region": "USA", "universe": "TOP3000", "delay": 1})
        self.assertIn("TOPN is the N most liquid stocks", prompt["arena_note"])
        self.assertIn("MINVOLxM", prompt["arena_note"])
        self.assertIn("LOW_SUB_UNIVERSE_SHARPE", prompt["arena_note"])

    def test_without_an_arena_the_prompt_is_unchanged(self) -> None:
        prompt = self._capture_prompt(operator="e1")
        self.assertNotIn("arena_note", prompt)


class CorrelationRecipeTests(unittest.TestCase):
    """The wrappers that measurably beat SELF_CORRELATION reach the prompt."""

    def _instructions(self) -> dict:
        import alpha_agent.refine_llm as refine

        return refine._OPERATOR_INSTRUCTIONS

    def test_e2_and_m1_carry_the_measured_wrapper_evidence(self) -> None:
        instructions = self._instructions()
        for operator in ("e2", "m1"):
            task = instructions[operator]
            for marker in ("0.768", "0.525", "0.600", "group_neutralize", "group_rank", "industry"):
                self.assertIn(marker, task, f"{operator} prompt lost {marker!r}")

    def test_the_caveat_that_distance_does_not_predict_correlation(self) -> None:
        for operator in ("e2", "m1"):
            self.assertIn("does not predict correlation", self._instructions()[operator])

    def test_the_other_operators_are_left_alone(self) -> None:
        instructions = self._instructions()
        self.assertNotIn("group_neutralize", instructions["e1"])
        self.assertNotIn("group_neutralize", instructions["s1"])

    def test_the_recipe_reaches_the_actual_prompt_payload(self) -> None:
        """``task`` in the JSON sent to the model is the instruction verbatim."""
        import inspect

        import alpha_agent.refine_llm as refine

        source = inspect.getsource(refine.LLMRefinementProposer._propose_via_model)
        self.assertIn('"task": _OPERATOR_INSTRUCTIONS[operator]', source)


class LlmRootInitialisationTests(unittest.TestCase):
    """MCTS-AHD init: one i1 root from scratch, then e1 roots; no library seeds."""

    EXPRESSIONS = ["rank(ts_delta(close, 5))", "ts_corr(volume, vwap, 10)", "rank(high - low)", "ts_rank(returns, 20)"]

    def _run(self, **overrides):
        import os
        from unittest import mock

        from alpha_agent.refine_llm import LLMRefinementProposer

        os.environ["FAKE_LLM_KEY"] = "test"
        self.addCleanup(os.environ.pop, "FAKE_LLM_KEY", None)
        calls: list = []

        def fake_completion(proposer, *, body, api_key):
            payload = json.loads(body["messages"][1]["content"])
            calls.append(payload)
            expression = self.EXPRESSIONS[(len(calls) - 1) % len(self.EXPRESSIONS)]
            return {"choices": [{"message": {"content": json.dumps({"expressions": [{"thought": "t", "expression": expression}]})}}]}

        workdir = Path(tempfile.mkdtemp())
        runtime = build_runtime(workdir, llm_refine_enabled=True, root_source="llm", **overrides)
        runtime = replace(runtime, model=ModelConfig(provider="openai", model="m", api_key_env="FAKE_LLM_KEY"))
        agent = AlphaResearchAgent(runtime)
        agent.toolbox = FakeToolbox(agent, [])
        with mock.patch.object(LLMRefinementProposer, "_chat_completion", fake_completion):
            agent.run()
        return calls, agent, workdir

    def test_an_empty_tree_starts_with_i1_then_e1_and_no_library_seed(self) -> None:
        calls, agent, workdir = self._run(initial_seed_count=3, budget=3, max_iterations=1)
        self.assertEqual([call["operator"] for call in calls[:3]], ["i1", "e1", "e1"])
        self.assertEqual(agent.toolbox.popped_seeds, 0, "no idea-library seed may become a root")
        tree = json.loads((workdir / "search_tree.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tree["roots"]), 3)

    def test_i1_sees_no_existing_alpha(self) -> None:
        calls, _, _ = self._run(initial_seed_count=2, budget=2, max_iterations=1)
        first = calls[0]
        self.assertEqual(first["operator"], "i1")
        self.assertNotIn("parent", first)
        self.assertNotIn("references", first)
        for key in ("elites", "recent_failed_attempts", "failed_children_of_parent"):
            self.assertEqual(first[key], [], key)
        self.assertTrue(first["field_menu"], "i1 still gets the data it may use")

    def test_the_second_root_is_told_to_differ_from_the_first(self) -> None:
        calls, _, _ = self._run(initial_seed_count=2, budget=2, max_iterations=1)
        self.assertEqual(calls[1]["operator"], "e1")
        references = [ref["expression"].replace(" ", "") for ref in calls[1]["references"]]
        self.assertEqual(references, [self.EXPRESSIONS[0].replace(" ", "")])


class SubmittedAlphaSkipTests(unittest.TestCase):
    def test_an_already_submitted_alpha_does_not_block_the_next_one(self) -> None:
        from types import SimpleNamespace

        from alpha_agent.engine import ResearchToolbox

        toolbox = ResearchToolbox.__new__(ResearchToolbox)
        toolbox.agent_cfg = SimpleNamespace(allow_pending_checks=False)
        toolbox.leaderboard = lambda limit=200: [  # type: ignore[assignment]
            {"alpha_id": "A1", "status": "ok"},
            {"alpha_id": "A2", "status": "ok"},
        ]
        toolbox.submitted_alpha_ids = set()
        self.assertEqual(toolbox.best_submittable_candidate()["alpha_id"], "A1")
        toolbox.submitted_alpha_ids = {"A1"}
        self.assertEqual(toolbox.best_submittable_candidate()["alpha_id"], "A2")
        toolbox.submitted_alpha_ids = {"A1", "A2"}
        self.assertIsNone(toolbox.best_submittable_candidate())


class SystemPromptTests(unittest.TestCase):
    def test_patchable_settings_come_from_one_list(self) -> None:
        import alpha_agent.refine_llm as refine

        for key in refine._PATCHABLE_SETTINGS:
            self.assertIn(key, refine._SYSTEM_PROMPT)

    def test_every_patchable_setting_is_explained(self) -> None:
        import alpha_agent.refine_llm as refine

        self.assertEqual(set(refine._PROMPT_SETTINGS), set(refine._PATCHABLE_SETTINGS))

    def test_paper_names_stay_out_of_model_text(self) -> None:
        import alpha_agent.refine_llm as refine

        texts = [refine._SYSTEM_PROMPT, *refine._OPERATOR_INSTRUCTIONS.values()]
        for text in texts:
            for name in ("EoH", "MCTS", "AHD"):
                self.assertNotIn(name, text)

    def test_hard_rules_do_not_forbid_the_correlation_recipe(self) -> None:
        """The recipe says to wrap a bare formula in group_neutralize(..., industry);
        a rule calling that a no-op monotone wrapper would cancel it."""
        import alpha_agent.refine_llm as refine

        wrapper_rule = next(rule for rule in refine._PROMPT_HARD_RULES if "wrap the whole formula" in rule)
        forbidden_part = wrapper_rule.split(":")[0]
        self.assertNotIn("group_neutralize", forbidden_part)
        self.assertNotIn("group_rank", forbidden_part)
        self.assertIn("FIRST industry normalisation", wrapper_rule)

    def test_every_payload_key_is_explained(self) -> None:
        import os
        from types import SimpleNamespace

        import alpha_agent.refine_llm as refine
        from alpha_agent.config import ModelConfig

        os.environ["FAKE_LLM_KEY"] = "test"
        self.addCleanup(os.environ.pop, "FAKE_LLM_KEY", None)
        proposer = refine.LLMRefinementProposer(
            ModelConfig(provider="openai", model="m", api_key_env="FAKE_LLM_KEY"), max_calls=1
        )
        captured: dict = {}

        def fake_completion(*, body, api_key):
            captured["prompt"] = json.loads(body["messages"][1]["content"])
            return {"choices": [{"message": {"content": json.dumps({"expressions": []})}}]}

        proposer._chat_completion = fake_completion  # type: ignore[assignment]
        proposer._propose_via_model(
            node=SimpleNamespace(expression="rank(close)", settings={}, family="price"),
            e_vector={"SHARPE": -0.1},
            blacklist=(),
            operators=None,
            context={"arena": {"region": "USA", "universe": "TOP3000", "delay": 1}, "iteration": 1},
            operator="s1",
        )
        documented = " ".join(refine._PROMPT_INPUTS)
        for key in captured["prompt"]:
            if key in {"quality_note", "budget_note"}:
                continue  # self-describing prose
            self.assertIn(key, documented, f"payload key {key!r} missing from _PROMPT_INPUTS")


if __name__ == "__main__":
    unittest.main()
