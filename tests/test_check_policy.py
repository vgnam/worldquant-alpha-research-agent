"""Regression tests for the submission-check policy.

These exist because a live run showed the platform returning checks that the
repository did not know about. `LOW_2Y_SHARPE` in particular is a hard quality
gate (limit 1.58 on the observed USA/TOP3000 configuration), so a candidate
failing it must never be reported as quality-ready or submit-ready.
"""

import unittest

import alpha_research_pipeline as pipeline


def failing_check(name: str, *, limit: float, value: float) -> dict:
    return {"name": name, "result": "FAIL", "limit": limit, "value": value}


class CheckPolicyTests(unittest.TestCase):
    def test_low_2y_sharpe_is_a_blocking_check(self) -> None:
        self.assertIn("LOW_2Y_SHARPE", pipeline.BLOCKING_CHECKS)
        self.assertIn("LOW_2Y_SHARPE", pipeline.SUBMISSION_BLOCKING_CHECKS)
        self.assertIn("LOW_2Y_SHARPE", pipeline.CHECK_SEVERITY_WEIGHTS)

    def test_is_ladder_sharpe_is_a_blocking_check(self) -> None:
        self.assertIn("IS_LADDER_SHARPE", pipeline.BLOCKING_CHECKS)
        self.assertIn("IS_LADDER_SHARPE", pipeline.SUBMISSION_BLOCKING_CHECKS)
        record = {
            "status": "ok",
            "failed_checks": ["IS_LADDER_SHARPE"],
            "summary": {"failed": ["IS_LADDER_SHARPE"], "pending": []},
        }
        self.assertFalse(pipeline.record_quality_ready(record))
        self.assertFalse(pipeline.record_is_submit_ready(record))

    def test_low_2y_sharpe_failure_is_not_quality_ready(self) -> None:
        record = {
            "status": "ok",
            "failed_blocking_checks": ["LOW_2Y_SHARPE"],
            "failed_checks": ["LOW_2Y_SHARPE"],
            "summary": {"failed": ["LOW_2Y_SHARPE"], "pending": []},
        }
        self.assertFalse(pipeline.record_quality_ready(record))

    def test_low_2y_sharpe_failure_is_not_submit_ready(self) -> None:
        record = {
            "status": "ok",
            "failed_blocking_checks": ["LOW_2Y_SHARPE"],
            "failed_submission_checks": ["LOW_2Y_SHARPE"],
            "failed_checks": ["LOW_2Y_SHARPE"],
            "summary": {"failed": ["LOW_2Y_SHARPE"], "pending": []},
        }
        self.assertFalse(pipeline.record_is_submit_ready(record))

    def test_low_2y_sharpe_failure_only_is_still_flagged(self) -> None:
        """The legacy fallback path (no explicit field) must catch it too."""
        record = {
            "status": "ok",
            "failed_checks": ["LOW_2Y_SHARPE"],
            "summary": {"failed": ["LOW_2Y_SHARPE"], "pending": []},
        }
        self.assertFalse(pipeline.record_quality_ready(record))
        self.assertEqual(pipeline.record_submission_failures(record), ["LOW_2Y_SHARPE"])

    def test_metrics_expose_low_2y_sharpe_limit_and_value(self) -> None:
        payload = {"is": {"checks": [failing_check("LOW_2Y_SHARPE", limit=1.58, value=1.06)]}}
        metrics = pipeline.extract_metrics({}, payload)
        self.assertAlmostEqual(metrics["low_2y_sharpe_limit"], 1.58)
        self.assertAlmostEqual(metrics["low_2y_sharpe_value"], 1.06)

    def test_score_penalizes_low_2y_sharpe_failure(self) -> None:
        detail = {"is": {"sharpe": 1.29, "fitness": 0.61}}
        with_failure = pipeline.score_result(
            detail, {"is": {"checks": [failing_check("LOW_2Y_SHARPE", limit=1.58, value=1.21)]}}
        )
        without_failure = pipeline.score_result(
            detail, {"is": {"checks": [{"name": "LOW_2Y_SHARPE", "result": "PASS", "limit": 1.58, "value": 1.61}]}}
        )
        self.assertLess(with_failure, without_failure)

    def test_unseen_check_failure_still_uses_generic_penalty(self) -> None:
        """Unknown future checks must not silently score as a pass."""
        detail = {"is": {"sharpe": 1.5, "fitness": 1.1}}
        baseline = pipeline.score_result(detail, {"is": {"checks": []}})
        unknown_fail = pipeline.score_result(
            detail, {"is": {"checks": [failing_check("SOME_FUTURE_CHECK", limit=1.0, value=0.5)]}}
        )
        self.assertLess(unknown_fail, baseline)


if __name__ == "__main__":
    unittest.main()
