"""Tests for pool-aware seeding and the arena override."""

import tempfile
import unittest
from pathlib import Path

import alpha_research_pipeline as pipeline
from alpha_agent.config import AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig
from alpha_agent.engine import ResearchToolbox
from alpha_agent.pool_fields import (
    collision_risk,
    expression_fields,
    field_swaps,
    field_usage,
    order_seeds_by_collision,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class ExpressionFieldsTests(unittest.TestCase):
    def test_reads_data_fields_and_drops_grouping_builtins(self) -> None:
        fields = expression_fields("ts_mean(-scl12_buzz, 8) + group_rank(rank(-ts_delta(close, 5)), industry)")
        self.assertEqual(fields, {"scl12_buzz", "close"})

    def test_operators_are_not_fields(self) -> None:
        fields = expression_fields("rank(ts_mean(volume, 20))")
        self.assertEqual(fields, {"volume"}, "rank/ts_mean are calls, not data")

    def test_unparsable_expression_has_no_fields(self) -> None:
        self.assertEqual(expression_fields("rank("), set())
        self.assertEqual(expression_fields(""), set())


class CollisionRiskTests(unittest.TestCase):
    def test_risk_is_the_worst_field_share(self) -> None:
        usage, total = field_usage(
            ["rank(scl12_buzz)"] * 8 + ["rank(close)"] * 2
        )
        self.assertEqual(total, 10)
        # One saturated field is enough to sink SELF_CORRELATION.
        self.assertAlmostEqual(collision_risk("ts_mean(scl12_buzz, 8)", usage, total), 0.8)
        self.assertAlmostEqual(collision_risk("rank(close)", usage, total), 0.2)

    def test_field_the_pool_never_used_is_free(self) -> None:
        usage, total = field_usage(["rank(scl12_buzz)"] * 5)
        self.assertEqual(collision_risk("rank(anl4_ebit_value)", usage, total), 0.0)

    def test_similarity_can_only_raise_the_score(self) -> None:
        usage, total = field_usage(["rank(close)"] * 4 + ["rank(open)"] * 4)
        self.assertAlmostEqual(collision_risk("rank(vwap)", usage, total), 0.0)
        self.assertAlmostEqual(collision_risk("rank(vwap)", usage, total, similarity=0.9), 0.9)
        self.assertAlmostEqual(
            collision_risk("rank(close)", usage, total, similarity=0.1), 0.5, msg="worst of the two"
        )

    def test_unknown_seed_is_mid_pack_and_an_empty_pool_is_not_a_signal(self) -> None:
        usage, total = field_usage(["rank(close)"])
        self.assertEqual(collision_risk("rank(", usage, total), 0.5)
        self.assertEqual(collision_risk("rank(close)", usage, 0), 0.5)

    def test_order_is_round_robin_over_families_then_by_risk(self) -> None:
        class Seed:
            def __init__(self, family, risk):
                self.family = family
                self.risk = risk

        seeds = [
            Seed("a", 0.9), Seed("a", 0.1), Seed("b", 0.8), Seed("b", 0.2), Seed("c", 0.7),
        ]
        ordered = order_seeds_by_collision(seeds, lambda seed: seed.risk)
        # One seed per family first, so no family floods the queue ...
        self.assertEqual([(seed.family, seed.risk) for seed in ordered],
                         [("a", 0.1), ("b", 0.2), ("c", 0.7), ("b", 0.8), ("a", 0.9)])
        # ... and the least colliding one represents each family.
        self.assertEqual(ordered[0].risk, 0.1)


class FieldSwapTests(unittest.TestCase):
    TEMPLATE = "ts_mean(-scl12_buzz, 8) + rank((ts_mean(close, 24) - close) / close)"

    def test_swaps_the_most_covered_field_for_a_proven_but_rare_one(self) -> None:
        usage, total = field_usage(
            ["rank(close)"] * 9 + ["rank(scl12_buzz)"] * 4 + ["rank(high)"] * 2 + ["rank(low)"] * 3
        )
        variants = field_swaps(
            self.TEMPLATE,
            usage=usage,
            total=total,
            candidates=["high", "low", "scl12_buzz", "scl12_sentiment_fast_d1"],
            limit=2,
        )
        # ``close`` is the worst offender (9/18) and appears three times in the
        # formula; only price/volume fields the pool already uses are allowed.
        self.assertEqual(len(variants), 2)
        self.assertNotIn("close", variants[0])
        self.assertTrue(variants[0].startswith("ts_mean(-scl12_buzz, 8)"), "structure kept")
        self.assertIn("high", variants[0], "rarest usable field first")
        self.assertIn("low", variants[1])

    def test_a_field_the_pool_never_touched_is_not_a_candidate(self) -> None:
        """Sparse fields look free but produce nonsense like (x - split) / split."""
        usage, total = field_usage(["rank(close)"] * 40)
        self.assertEqual(
            field_swaps(self.TEMPLATE, usage=usage, total=total, candidates=["dividend", "split"]),
            [],
        )

    def test_only_offers_fields_from_the_same_data_group(self) -> None:
        usage, total = field_usage(["rank(scl12_buzz)"] * 5 + ["rank(scl12_sentiment_fast_d1)"])
        variants = field_swaps(
            "rank(scl12_buzz)",
            usage=usage,
            total=total,
            candidates=["close", "fnd6_assets", "scl12_sentiment_fast_d1"],
            limit=3,
        )
        self.assertEqual(len(variants), 1, "close/fnd6 are different data groups")
        self.assertEqual(variants[0], "rank(scl12_sentiment_fast_d1)")

    def test_a_price_is_only_swapped_for_a_price(self) -> None:
        """Same data group is not enough: pv holds market cap next to close."""
        usage, total = field_usage(
            ["rank(close)"] * 10 + ["rank(cap)"] * 2 + ["rank(volume)"] * 2 + ["rank(returns)"] * 2 + ["rank(open)"] * 3
        )
        variants = field_swaps(
            "(close - open) / ((high - low) + 0.001)",
            usage=usage,
            total=total,
            candidates=["cap", "volume", "returns", "vwap", "open"],
            limit=4,
        )
        self.assertEqual(variants, [], "vwap is unused by the pool, the rest are not prices")
        usage["vwap"] = 2
        variants = field_swaps(
            "(close - open) / ((high - low) + 0.001)",
            usage=usage,
            total=total,
            candidates=["cap", "volume", "returns", "vwap"],
            limit=4,
        )
        self.assertEqual(variants, ["(vwap - open) / ((high - low) + 0.001)"])

    def test_volume_is_only_swapped_for_volume(self) -> None:
        usage, total = field_usage(["rank(volume)"] * 10 + ["rank(cap)"] * 2 + ["rank(adv20)"] * 2 + ["rank(close)"] * 2)
        self.assertEqual(
            field_swaps("ts_corr(close, volume, 10)", usage=usage, total=total, candidates=["cap", "adv20", "close"]),
            [],
            "adv20 is an average, cap a dollar amount; neither is daily volume",
        )

    def test_a_field_of_unknown_unit_is_not_swapped(self) -> None:
        usage, total = field_usage(["rank(assets)"] * 10 + ["rank(roe)"] * 2)
        self.assertEqual(field_swaps("rank(assets)", usage=usage, total=total, candidates=["roe"]), [])

    def test_no_variant_without_a_pool_or_a_parseable_expression(self) -> None:
        usage, total = field_usage(["rank(close)"])
        self.assertEqual(field_swaps("rank(", usage=usage, total=total, candidates=["open"]), [])
        self.assertEqual(field_swaps("rank(close)", usage=usage, total=0, candidates=["open"]), [])


class FakeEntry:
    def __init__(self, expression, region="USA"):
        self.expression = expression
        self.region = region


class FakePool:
    def __init__(self, entries):
        self.entries = entries

    def max_similarity(self, expression, settings=None):
        return 0.0


class ArenaAndSeedOrderingTests(unittest.TestCase):
    def _toolbox(self, **overrides) -> ResearchToolbox:
        workdir = Path(tempfile.mkdtemp())
        agent = AgentConfig(
            workdir=workdir,
            idea_library=REPO_ROOT / "alpha_pipeline_ideas.json",
            fields_summary=REPO_ROOT / "wqb_data_fields_summary.json",
            allow_offline_fetch=False,
            **overrides,
        )
        runtime = AgentRuntimeConfig(
            auth=AuthConfig(email=None, password=None, cookie_header=None),
            model=ModelConfig(provider="heuristic"),
            agent=agent,
        )
        return ResearchToolbox(runtime)

    def test_every_seed_is_forced_into_the_configured_arena(self) -> None:
        toolbox = self._toolbox(region="EUR", universe="TOP1200", delay=0)
        self.assertTrue(toolbox.seed_queue, "the idea library produces seeds")
        for candidate in toolbox.seed_queue[:50]:
            self.assertEqual(candidate.settings.get("region"), "EUR")
            self.assertEqual(candidate.settings.get("universe"), "TOP1200")
            self.assertEqual(candidate.settings.get("delay"), 0)

    def test_arena_override_survives_an_idea_library_settings_grid(self) -> None:
        toolbox = self._toolbox(region="CHN", universe="TOP2000U", delay=1)
        settings = toolbox.arena_settings({"region": "USA", "universe": "TOP3000", "delay": 1})
        self.assertEqual((settings["region"], settings["universe"], settings["delay"]), ("CHN", "TOP2000U", 1))

    def test_seed_queue_moves_away_from_fields_the_pool_owns(self) -> None:
        toolbox = self._toolbox()

        def seed(expression: str, family: str) -> pipeline.Candidate:
            return pipeline.Candidate(expression=expression, settings={}, family=family, idea_name=family)

        # Same fields as the pool, different formulas: not duplicates, but the
        # fields they read are what the collision risk is about.
        toolbox.seed_queue = [
            seed("ts_mean(scl12_buzz, 8)", "social"),
            seed("zscore(anl4_eps_mean)", "analyst"),
            seed("rank(mdl175_assets)", "model"),
        ]
        pool = FakePool([FakeEntry("rank(scl12_buzz)")] * 6 + [FakeEntry("rank(anl4_eps_mean)")] * 4)
        report = toolbox.order_seeds_against_pool(pool)
        self.assertEqual(report["pool_entries"], 10)
        self.assertEqual(report["pool_scanned"], 10)
        self.assertEqual(report["pool_fields"], 2)
        self.assertEqual(toolbox.seed_queue[0].expression, "rank(mdl175_assets)")
        self.assertEqual(report["next_seeds"][0]["risk"], 0.0, "a field the pool never used leads")
        self.assertGreater(report["mean_risk_before"], 0.0)

    def test_seeds_already_in_the_pool_are_dropped(self) -> None:
        """Measured: 26 of 324 evaluated alphas were exact copies of a pool formula."""
        toolbox = self._toolbox()

        def seed(expression: str, name: str) -> pipeline.Candidate:
            return pipeline.Candidate(expression=expression, settings={}, family=name, idea_name=name)

        toolbox.seed_queue = [
            seed("ts_mean(-scl12_buzz, 9)", "social"),
            seed("rank(fnd6_fopo)", "fundamental"),
            seed("zscore(cash_st / debt_lt)", "quality"),
            seed("rank(vwap - close)", "pv"),
            seed("rank(ts_delta(close, 5))", "pv2"),
            seed("-ts_mean(scl12_buzz, 5)", "social2"),
        ]
        pool = FakePool([FakeEntry("ts_mean(-scl12_buzz, 9)"), FakeEntry("rank(fnd6_fopo)")])
        report = toolbox.order_seeds_against_pool(pool)
        self.assertEqual(report["pool_duplicate_seeds"], 2)
        expressions = {candidate.expression for candidate in toolbox.seed_queue}
        self.assertNotIn("ts_mean(-scl12_buzz, 9)", expressions)
        self.assertIn("rank(vwap - close)", expressions)

    def test_an_all_duplicate_queue_is_kept_rather_than_emptied(self) -> None:
        toolbox = self._toolbox()
        toolbox.seed_queue = [
            pipeline.Candidate(expression="rank(close)", settings={}, family="pv", idea_name="pv"),
            pipeline.Candidate(expression="rank(open)", settings={}, family="pv", idea_name="pv2"),
        ]
        pool = FakePool([FakeEntry("rank(close)")] * 30 + [FakeEntry("rank(open)")] * 30)
        report = toolbox.order_seeds_against_pool(pool)
        self.assertEqual(report["pool_duplicate_seeds"], 0, "nothing else left to search")
        self.assertEqual(len(toolbox.seed_queue), 2)

    def test_pool_scoped_to_the_arena_when_it_has_enough_alphas(self) -> None:
        toolbox = self._toolbox(region="EUR")
        toolbox.seed_queue = [
            pipeline.Candidate(expression="rank(close)", settings={}, family="pv", idea_name="pv")
        ]
        other = [FakeEntry("rank(scl12_buzz)", region="USA")] * 30
        report = toolbox.order_seeds_against_pool(FakePool(other))
        # EUR has no pool alphas, so the USA pool is all there is to compare with.
        self.assertEqual(report["pool_entries"], 30)
        self.assertEqual(toolbox.seed_queue[0].expression, "rank(close)")

    def test_a_pool_with_fewer_than_twenty_entries_falls_back_to_all_of_it(self) -> None:
        toolbox = self._toolbox(region="EUR")
        toolbox.seed_queue = [
            pipeline.Candidate(expression="rank(close)", settings={}, family="pv", idea_name="pv")
        ]
        mixed = [FakeEntry("rank(close)", region="USA")] * 3 + [FakeEntry("rank(open)", region="EUR")] * 2
        report = toolbox.order_seeds_against_pool(FakePool(mixed))
        self.assertEqual(report["pool_entries"], 5, "too thin to scope, so everything is used")


if __name__ == "__main__":
    unittest.main()
