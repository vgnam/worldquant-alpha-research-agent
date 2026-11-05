"""mo-hifo: MEoH selection, HiFo insight pool / navigator, and the engine loop."""

import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import alpha_research_pipeline as pipeline
from alpha_agent import mo_hifo as mh
from alpha_agent.config import ModelConfig
from alpha_agent.engine import AlphaResearchAgent
from test_hybrid_engine import FakeToolbox, build_runtime


def member(
    key, sharpe, returns, turnover, expression=None, *, failed_checks=(), self_correlation=None, prod_correlation=None, **extra
) -> mh.Individual:
    return mh.Individual(
        key=key,
        expression=expression or f"rank(ts_delta(close, {abs(hash(key)) % 50 + 2}))",
        metrics={"sharpe": sharpe, "returns": returns, "turnover": turnover, **extra},
        failed_checks=list(failed_checks),
        self_correlation=self_correlation,
        prod_correlation=prod_correlation,
    )


class ObjectiveTests(unittest.TestCase):
    def test_turnover_below_the_fitness_floor_earns_nothing(self) -> None:
        slow = member("a", 1.5, 0.05, 0.02)
        medium = member("b", 1.5, 0.05, 0.10)
        fast = member("c", 1.5, 0.05, 0.20)
        self.assertFalse(mh.constrained_dominates(slow, medium))
        self.assertFalse(mh.constrained_dominates(medium, slow))
        self.assertTrue(mh.constrained_dominates(slow, fast))

    def test_the_turnover_band_is_a_constraint(self) -> None:
        crowded = member("hi", 2.2, 0.10, 1.29)
        weak = member("ok", 0.5, 0.01, 0.20)
        self.assertTrue(mh.constrained_dominates(weak, crowded))
        self.assertFalse(mh.constrained_dominates(crowded, weak))
        dead = member("lo", 1.0, 0.02, 0.005)
        self.assertTrue(mh.constrained_dominates(weak, dead))

    def test_the_band_comes_from_the_check_payload_when_present(self) -> None:
        loose = member("x", 1.0, 0.02, 0.80, high_turnover_limit=0.9)
        self.assertEqual(mh.turnover_violation(loose.metrics), 0.0)
        self.assertGreater(mh.turnover_violation(member("y", 1.0, 0.02, 0.80).metrics), 0.0)

    def test_a_reported_correlation_fail_is_a_constraint(self) -> None:
        crowded = member("c", 1.67, 0.08, 0.25, failed_checks=["LOW_FITNESS", "PROD_CORRELATION"], prod_correlation=0.877)
        weak = member("w", 0.5, 0.01, 0.20)
        self.assertAlmostEqual(mh.correlation_violation(crowded), 0.177)
        self.assertTrue(mh.constrained_dominates(weak, crowded))
        self.assertFalse(mh.constrained_dominates(crowded, weak))
        unvalued = member("u", 1.67, 0.08, 0.25, failed_checks=["SELF_CORRELATION"])
        self.assertEqual(mh.correlation_violation(unvalued), mh.MIN_CORRELATION_VIOLATION)

    def test_pending_or_passed_correlation_is_not_penalised(self) -> None:
        pending = member("p", 1.5, 0.05, 0.2, failed_checks=["LOW_SHARPE"])
        passed = member("q", 1.5, 0.05, 0.2, failed_checks=["LOW_SHARPE"], self_correlation=0.55, prod_correlation=0.65)
        self.assertEqual(mh.violation(pending), 0.0)
        self.assertEqual(mh.violation(passed), 0.0)
        self.assertFalse(pending.correlation_blocked)

    def test_dominance_never_lowers_brain_fitness(self) -> None:
        rng = random.Random(3)

        def fitness(m):
            return m["sharpe"] * math.sqrt(abs(m["returns"]) / max(m["turnover"], 0.125))

        for _ in range(500):
            a = member("a", rng.uniform(0, 3), rng.uniform(0.0, 0.2), rng.uniform(0.01, 0.7))
            b = member("b", rng.uniform(0, 3), rng.uniform(0.0, 0.2), rng.uniform(0.01, 0.7))
            if mh.constrained_dominates(a, b):
                self.assertGreaterEqual(fitness(a.metrics) + 1e-12, fitness(b.metrics))


class SelectionTests(unittest.TestCase):
    def test_non_dominated_scores_zero_and_similar_dominators_penalise(self) -> None:
        strong = member("s", 2.0, 0.10, 0.20, "rank(ts_delta(close, 5))")
        twin = member("t", 1.0, 0.05, 0.30, "rank(ts_delta(close, 5)) + rank(volume)")
        stranger = member("u", 1.0, 0.05, 0.30, "group_rank(ts_mean(sales, 20), industry)")
        scores = mh.dominance_dissimilarity([strong, twin, stranger])
        self.assertEqual(scores[0], 0.0)
        self.assertLess(scores[1], scores[2])
        self.assertEqual(scores[2], 0.0)

    def test_survival_keeps_n_and_puts_infeasible_last(self) -> None:
        pool = [member(f"m{i}", 1.0 + i * 0.1, 0.05, 0.2, f"rank(ts_mean(sales, {i + 3}))") for i in range(5)]
        pool.append(member("bad", 3.0, 0.2, 1.5, "rank(volume)"))
        kept = mh.survive(pool, 4)
        self.assertEqual(len(kept), 4)
        self.assertNotIn("bad", [m.key for m in kept])

    def test_survival_and_parents_pass_over_correlation_blocked_alphas(self) -> None:
        pool = [member(f"m{i}", 1.0 + i * 0.1, 0.05, 0.2, f"rank(ts_mean(sales, {i + 3}))") for i in range(4)]
        crowded = member(
            "crowded", 1.8, 0.09, 0.2, "group_rank(reverse(ts_zscore(returns, 60)), industry)",
            failed_checks=["PROD_CORRELATION"], prod_correlation=0.85,
        )
        kept = mh.survive([*pool, crowded], 4)
        self.assertNotIn("crowded", [m.key for m in kept])
        picks = [mh.select_parents([*pool, crowded], 1, random.Random(seed))[0].key for seed in range(300)]
        self.assertLess(picks.count("crowded"), 300 / 5)

    def test_survival_drops_duplicates(self) -> None:
        a = member("a", 1.0, 0.05, 0.2)
        self.assertEqual(len(mh.survive([a, a, a], 3)), 1)

    def test_parents_are_distinct_and_favour_the_front(self) -> None:
        front = member("f", 2.0, 0.10, 0.20, "rank(ts_delta(close, 5))")
        dominated = [
            member(f"d{i}", 0.5, 0.02, 0.4, f"rank(ts_delta(close, 5)) + rank(ts_mean(volume, {i + 2}))")
            for i in range(4)
        ]
        population = [front, *dominated]
        rng = random.Random(1)
        picks = [mh.select_parents(population, 1, rng)[0].key for _ in range(400)]
        self.assertGreater(picks.count("f"), 400 / len(population))
        pair = mh.select_parents(population, 3, random.Random(2))
        self.assertEqual(len({m.key for m in pair}), 3)


class CreditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.population = [
            member("a", 1.0, 0.05, 0.20, "rank(ts_mean(sales, 5))"),
            member("b", 1.5, 0.03, 0.30, "rank(ts_mean(assets, 5))"),
        ]

    def test_tiers(self) -> None:
        better = member("c", 2.0, 0.08, 0.15, "rank(cap)")
        joins = member("d", 0.8, 0.09, 0.40, "rank(volume)")
        worse = member("e", 0.5, 0.01, 0.50, "rank(open)")
        self.assertGreaterEqual(mh.pareto_credit(better, self.population), 0.8)
        self.assertTrue(0.2 <= mh.pareto_credit(joins, self.population) < 0.8)
        self.assertLess(mh.pareto_credit(worse, self.population), 0.2)
        self.assertEqual(mh.pareto_credit(None, self.population), -0.5)


class InsightPoolTests(unittest.TestCase):
    def test_seeds_and_jaccard_admission(self) -> None:
        pool = mh.InsightPool()
        self.assertEqual(len(pool.items), len(mh.SEED_INSIGHTS))
        self.assertFalse(pool.add(mh.SEED_INSIGHTS[0] + " again"))
        self.assertTrue(pool.add("Use analyst revision breadth as a slow leg against fast price reversal."))

    def test_one_slot_is_reserved_for_an_avoid_insight(self) -> None:
        pool = mh.InsightPool()
        pool.add("Avoid adding a five-day price reversal leg; it ties the alpha to crowded reversal books.", kind="avoid")
        chosen = pool.select(3)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(pool.items[chosen[0]]["kind"], "avoid")

    def test_credit_is_an_ema_and_effects_are_means(self) -> None:
        pool = mh.InsightPool()
        text = mh.SEED_INSIGHTS[0]
        pool.credit([text], 1.0, {"sharpe": 0.4})
        pool.credit([text], 1.0, {"sharpe": 0.2})
        self.assertAlmostEqual(pool.items[text]["effectiveness"], 0.3 + 0.7 * 0.3)
        self.assertAlmostEqual(pool.items[text]["effects"]["sharpe"], 0.3)
        described = pool.describe([text])[0]
        self.assertEqual(described["n"], 2)

    def test_eviction_spares_insights_on_probation(self) -> None:
        pool = mh.InsightPool(max_size=6)
        veteran = mh.SEED_INSIGHTS[1]
        pool.items[veteran]["used_count"] = 5
        pool.items[veteran]["effectiveness"] = -0.9
        pool.add("Condition momentum on low dispersion of analyst estimates for the same quarter.")
        pool.add("Scale earnings surprise legs by the inverse of their historical volatility.")
        self.assertNotIn(veteran, pool.items)
        self.assertLessEqual(len(pool.items), 6)

    def test_round_trip(self) -> None:
        pool = mh.InsightPool()
        pool.credit([mh.SEED_INSIGHTS[2]], 0.5, {"returns": 0.01})
        again = mh.InsightPool.from_dict(json.loads(json.dumps(pool.to_dict())))
        self.assertEqual(again.items, pool.items)


class NavigatorTests(unittest.TestCase):
    DIVERSE = {"dissimilarity": 0.8, "top_group_share": 0.3}

    def _update(self, nav, hv, diversity=None, blocked=False, size=10):
        return nav.update(
            hypervolume_value=hv,
            diversity=diversity or self.DIVERSE,
            correlation_blocked=blocked,
            population_size=size,
            generation=0,
        )

    def test_correlation_block_switches_to_decorrelate(self) -> None:
        nav = mh.Navigator()
        self.assertEqual(self._update(nav, 0.1, blocked=True), "decorrelate")
        self.assertIn("correlation", nav.guidance()["regime_note"])

    def test_stagnation_and_low_diversity_explore(self) -> None:
        nav = mh.Navigator()
        for _ in range(4):
            regime = self._update(nav, 0.1)
        self.assertEqual(regime, "exploration")
        nav = mh.Navigator()
        self.assertEqual(self._update(nav, 0.1, {"dissimilarity": 0.9, "top_group_share": 0.8}), "exploration")

    def test_progress_exploits(self) -> None:
        nav = mh.Navigator()
        for hv in (0.1, 0.2, 0.3):
            regime = self._update(nav, hv)
        self.assertEqual(regime, "exploitation")


def fake_record(candidate: pipeline.Candidate, metrics: dict, **extra) -> dict:
    return {
        **candidate.to_record(),
        "status": "ok",
        "alpha_id": f"A{candidate.signature()[:6]}",
        "metrics": metrics,
        "failed_checks": extra.get("failed_checks", ["LOW_SHARPE"]),
        "failed_blocking_checks": extra.get("failed_blocking_checks", ["LOW_SHARPE"]),
        "failed_correlation_checks": extra.get("failed_correlation_checks", []),
        "check_diagnostics": [],
    }


class SearchLoopTests(unittest.TestCase):
    def _search(self, path: Path, *, budget: int, population: int = 4):
        calls = []
        rng = random.Random(5)
        state = mh.MoHifoState.load_or_new(path)

        def propose(operator, parents, guidance):
            calls.append((operator, [p.key for p in parents], guidance))
            expression = f"rank(ts_mean(sales, {len(calls) + 2}))"
            candidate = pipeline.Candidate(expression=expression, settings={}, family="", idea_name="t")
            return [
                replace(candidate, metadata=mh.candidate_metadata(candidate, operator=operator, parents=parents, guidance=guidance))
            ]

        def evaluate(batch):
            return [
                fake_record(c, {"sharpe": rng.uniform(0, 2), "returns": rng.uniform(0, 0.1), "turnover": rng.uniform(0.05, 0.5)})
                for c in batch
            ], None

        search = mh.MoHifoSearch(
            state=state, propose=propose, evaluate=evaluate, margins_of=lambda r: {},
            population_size=population, batch_size=2,
        )
        reason = search.run(budget, max_batches=100)
        return search, calls, reason

    def test_initialisation_is_i1_then_e1_against_everything_written(self) -> None:
        path = Path(tempfile.mkdtemp()) / "pop.json"
        search, calls, reason = self._search(path, budget=4)
        self.assertEqual(reason, "Budget exhausted.")
        self.assertEqual([c[0] for c in calls[:4]], ["i1", "e1", "e1", "e1"])
        self.assertEqual(len(calls[1][1]), 1, "the second alpha sees the first, not yet simulated")
        self.assertEqual(len(search.state.population), 4)

    def test_generations_survive_to_n_and_resume_from_disk(self) -> None:
        path = Path(tempfile.mkdtemp()) / "pop.json"
        search, calls, _ = self._search(path, budget=12)
        self.assertGreaterEqual(search.state.generation, 2)
        self.assertEqual(len(search.state.population), 4)
        self.assertTrue(all(c[0] in {"e1", "e2", "m1", "m2", "m3"} for c in calls[4:]))
        loaded = mh.MoHifoState.load_or_new(path)
        self.assertEqual(loaded.state_id, search.state.state_id)
        self.assertEqual([m.key for m in loaded.population], [m.key for m in search.state.population])
        self.assertTrue(any(stats["credit_count"] for stats in loaded.insights.items.values()))

    def test_a_killed_run_recovers_its_evaluated_offspring(self) -> None:
        path = Path(tempfile.mkdtemp()) / "pop.json"
        state = mh.MoHifoState(path=path)
        state.initialised = True
        candidate = pipeline.Candidate(expression="rank(ts_mean(cashflow, 5))", settings={}, family="", idea_name="t")
        meta = mh.candidate_metadata(candidate, operator="m1", parents=[], guidance={"state_id": state.state_id, "generation": 3})
        record = fake_record(replace(candidate, metadata=meta), {"sharpe": 1.0, "returns": 0.05, "turnover": 0.2})
        foreign = fake_record(candidate, {"sharpe": 1.0, "returns": 0.05, "turnover": 0.2})
        self.assertEqual(state.recover([record, foreign], lambda r: {}), 1)
        self.assertEqual(len(state.offspring), 1)
        self.assertEqual(state.recover([record], lambda r: {}), 0)

    def test_a_block_below_the_quality_gates_marks_the_alpha_and_decorrelates(self) -> None:
        path = Path(tempfile.mkdtemp()) / "pop.json"
        state = mh.MoHifoState(path=path)
        state.initialised = True
        state.population = [member(f"m{i}", 1.0, 0.05, 0.2, f"rank(ts_mean(sales, {i + 3}))") for i in range(3)]
        candidate = pipeline.Candidate(expression="rank(ts_mean(cashflow, 5))", settings={}, family="", idea_name="t")
        record = fake_record(
            candidate,
            {"sharpe": 1.67, "returns": 0.08, "turnover": 0.25},
            failed_checks=["LOW_FITNESS", "PROD_CORRELATION"],
            failed_blocking_checks=["LOW_FITNESS"],
            failed_correlation_checks=["PROD_CORRELATION"],
        )
        child = mh.individual_from_record(record)
        self.assertFalse(child.quality_ready)
        self.assertTrue(child.correlation_blocked)
        state.offspring = [child]
        search = mh.MoHifoSearch(
            state=state, propose=lambda *a: [], evaluate=lambda b: ([], None), margins_of=lambda r: {}, population_size=3
        )
        search._close_generation()
        self.assertEqual(state.navigator.regime, "decorrelate")
        self.assertNotIn(child.key, [m.key for m in state.population])

    def test_a_population_saved_before_the_constraint_is_rechecked_on_load(self) -> None:
        path = Path(tempfile.mkdtemp()) / "pop.json"
        state = mh.MoHifoState(path=path)
        state.initialised = True
        saved = member("old", 1.48, 0.07, 0.20, failed_checks=["LOW_SHARPE", "PROD_CORRELATION"], prod_correlation=0.8303)
        saved.correlation_blocked = False  # what the earlier version wrote
        state.population = [saved]
        state.save()
        loaded = mh.MoHifoState.load_or_new(path).population[0]
        self.assertTrue(loaded.correlation_blocked)
        self.assertAlmostEqual(mh.violation(loaded), 0.1303)


class EngineTests(unittest.TestCase):
    EXPRESSIONS = [
        "rank(ts_delta(close, 5))",
        "ts_corr(volume, vwap, 10)",
        "rank(high - low)",
        "ts_rank(returns, 20)",
        "rank(ts_mean(volume, 15))",
        "rank(ts_std_dev(returns, 30))",
        "rank(vwap - close)",
        "ts_rank(volume, 40)",
    ]

    def _run(self, prior_records=(), **overrides):
        import os

        from alpha_agent.refine_llm import LLMRefinementProposer

        os.environ["FAKE_LLM_KEY"] = "test"
        self.addCleanup(os.environ.pop, "FAKE_LLM_KEY", None)
        calls: list = []
        counter = [0]

        def fake_completion(proposer, *, body, api_key):
            payload = json.loads(body["messages"][1]["content"])
            calls.append(payload)
            if "strong_alphas" in payload:
                content = {"principles": ["Blend a volume-based leg with a range-based leg normalised within industry."], "avoid": []}
            else:
                counter[0] += 1
                content = {"expressions": [{"thought": "t", "expression": self.EXPRESSIONS[(counter[0] - 1) % len(self.EXPRESSIONS)]}]}
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

        workdir = Path(tempfile.mkdtemp())
        runtime = build_runtime(workdir, llm_refine_enabled=True, search_mode="mo-hifo", session_ping=False, **overrides)
        runtime = replace(runtime, model=ModelConfig(provider="openai", model="m", api_key_env="FAKE_LLM_KEY"))
        agent = AlphaResearchAgent(runtime)
        agent.toolbox = FakeToolbox(agent, [])
        agent.toolbox.results.extend(prior_records)
        with mock.patch.object(LLMRefinementProposer, "_chat_completion", fake_completion):
            result = agent.run()
        return calls, agent, workdir, result

    def test_failures_from_another_arena_stay_out_of_the_prompt(self) -> None:
        usa = {
            "status": "ok",
            "expression": "rank(pcr_oi_20)",
            "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
            "metrics": {"sharpe": 0.1},
            "failed_checks": ["LOW_SHARPE"],
        }
        eur = {**usa, "expression": "rank(ts_mean(sales, 5))", "settings": {"region": "EUR", "universe": "TOP2500", "delay": 1}}
        calls, _, _, _ = self._run(
            prior_records=[usa, eur], population_size=2, budget=4, max_iterations=10, region="EUR", universe="TOP2500"
        )
        shown = [
            failure["expression"].replace(" ", "")
            for call in calls
            for failure in call.get("recent_failed_attempts") or []
        ]
        self.assertNotIn("rank(pcr_oi_20)", shown)
        self.assertIn("rank(ts_mean(sales,5))", shown, "this arena's failures are still shown")

    def test_population_run_end_to_end(self) -> None:
        calls, agent, workdir, result = self._run(population_size=3, budget=6, max_iterations=10)
        operators = [call.get("operator") for call in calls if "operator" in call]
        self.assertEqual(operators[:3], ["i1", "e1", "e1"])
        self.assertEqual(result.summary["search_mode"], "mo_hifo")
        self.assertEqual(result.summary["evaluated_this_run"], 6)
        state = json.loads((workdir / "mo_hifo_population.json").read_text(encoding="utf-8"))
        self.assertEqual(len(state["population"]), 3)
        self.assertEqual(state["objectives"], ["sharpe", "returns", "turnover"])
        self.assertEqual(agent.submission_attempts, [])
        records = agent.toolbox.evaluated
        self.assertTrue(all(record["stage"] == "mo_hifo" for record in records))
        self.assertTrue(all(record["metadata"]["mo_hifo_state"] == state["state_id"] for record in records))

    def test_prompts_carry_insights_directive_and_regime(self) -> None:
        calls, _, _, _ = self._run(population_size=3, budget=5, max_iterations=10)
        proposals = [call for call in calls if "operator" in call]
        self.assertTrue(all("design_insights" in call for call in proposals))
        self.assertTrue(all("design_directive" in call and "search_regime" in call for call in proposals))
        self.assertTrue(any("strong_alphas" in call for call in calls), "one insight extraction per generation")

    def test_without_a_model_nothing_is_simulated(self) -> None:
        workdir = Path(tempfile.mkdtemp())
        runtime = build_runtime(workdir, search_mode="mo-hifo", session_ping=False, budget=4)
        agent = AlphaResearchAgent(runtime)
        agent.toolbox = FakeToolbox(agent, [])
        result = agent.run()
        self.assertIn("needs an LLM proposer", result.summary["stop_reason"])
        self.assertEqual(agent.toolbox.evaluated, [])


class PromptTests(unittest.TestCase):
    def test_m3_exists_and_names_no_paper(self) -> None:
        import alpha_agent.refine_llm as refine

        self.assertIn("m3", refine._OPERATOR_INSTRUCTIONS)
        for name in ("HiFo", "MEoH", "EoH", "MCTS"):
            self.assertNotIn(name, refine._OPERATOR_INSTRUCTIONS["m3"])
            self.assertNotIn(name, mh.REGIME_NOTES["decorrelate"])

    def test_guidance_keys_are_documented(self) -> None:
        import os
        from types import SimpleNamespace

        import alpha_agent.refine_llm as refine

        os.environ["FAKE_LLM_KEY"] = "test"
        self.addCleanup(os.environ.pop, "FAKE_LLM_KEY", None)
        proposer = refine.LLMRefinementProposer(ModelConfig(provider="openai", model="m", api_key_env="FAKE_LLM_KEY"), max_calls=1)
        captured: dict = {}

        def fake_completion(*, body, api_key):
            captured["prompt"] = json.loads(body["messages"][1]["content"])
            return {"choices": [{"message": {"content": json.dumps({"expressions": []})}}]}

        proposer._chat_completion = fake_completion  # type: ignore[assignment]
        guidance = {"insights": [{"insight": "x", "kind": "positive"}], "directive": "d", "regime_note": "r"}
        proposer._propose_via_model(
            node=SimpleNamespace(expression="rank(close)", settings={}, family="price"),
            e_vector={},
            blacklist=(),
            operators=None,
            context={"guidance": guidance},
            operator="m3",
        )
        documented = " ".join(refine._PROMPT_INPUTS)
        for key in ("design_insights", "design_directive", "search_regime"):
            self.assertIn(key, captured["prompt"])
            self.assertIn(key, documented)


class CliTests(unittest.TestCase):
    def test_flags_reach_the_runtime(self) -> None:
        import sys

        import alpha_research_agent as cli

        argv = [
            "alpha_research_agent.py", "run", "--search-mode", "mo-hifo", "--population-size", "12",
            "--crossover-parents", "4", "--no-insight-extraction", "--population-state-path", "p.json",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = cli.parse_args()
        runtime = cli.build_runtime(args)
        agent = runtime.agent
        self.assertEqual(agent.search_mode, "mo-hifo")
        self.assertEqual(agent.population_size, 12)
        self.assertEqual(agent.crossover_parents, 4)
        self.assertFalse(agent.extract_insights)
        self.assertEqual(agent.mo_hifo_state_path, Path("p.json"))


if __name__ == "__main__":
    unittest.main()
