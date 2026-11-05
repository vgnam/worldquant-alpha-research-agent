"""The 404 retry around reading a freshly simulated alpha.

The platform intermittently answers ``404`` for an alpha it created seconds
ago. Retrying the whole candidate costs a second simulation, so the retry has
to sit on the read calls only.
"""

import unittest

import alpha_research_pipeline as pipeline
from worldquant_brain_cli import BrainApiError

KENNEDY_CHECKS = {
    "is": {
        "checks": [
            {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 1.9},
            {"name": "SELF_CORRELATION", "result": "PASS", "limit": 0.7, "value": 0.4},
        ]
    }
}


class FlakyClient:
    """Simulates once, then 404s on the follow-up reads a few times."""

    def __init__(
        self,
        *,
        detail_404: int = 0,
        check_404: int = 0,
        detail_status: int = 404,
        simulation: dict | None = None,
    ) -> None:
        self.detail_404 = detail_404
        self.check_404 = check_404
        self.detail_status = detail_status
        self.simulation = simulation or {"id": "ALPHA1"}
        self.simulations = 0
        self.detail_reads = 0
        self.check_reads = 0

    def simulate(self, payload, *, max_wait, poll_interval):
        self.simulations += 1
        return self.simulation

    def fetch_alpha_detail(self, alpha_id):
        self.detail_reads += 1
        if self.detail_404 > 0:
            self.detail_404 -= 1
            raise BrainApiError("Not found.", status=self.detail_status, url=f"/alphas/{alpha_id}")
        return {"id": alpha_id, "is": {"sharpe": 1.9, "fitness": 1.2, "turnover": 0.3}}

    def check_alpha(self, alpha_id, *, max_wait, poll_interval):
        self.check_reads += 1
        if self.check_404 > 0:
            self.check_404 -= 1
            raise BrainApiError("Not found.", status=404, url=f"/alphas/{alpha_id}/check")
        return KENNEDY_CHECKS


def candidate():
    return pipeline.Candidate(
        expression="rank(ts_delta(close, 5))",
        settings={},
        family="pv",
        idea_name="test",
    )


class AlphaReadRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._delay = pipeline.ALPHA_READ_DELAY
        pipeline.ALPHA_READ_DELAY = 0.0

    def tearDown(self) -> None:
        pipeline.ALPHA_READ_DELAY = self._delay

    def _evaluate(self, client):
        return pipeline.evaluate_candidate(
            client=client, candidate=candidate(), max_wait=5.0, poll_interval=0.0, retries=0
        )

    def test_a_transient_404_costs_no_second_simulation(self) -> None:
        client = FlakyClient(detail_404=2)
        record = self._evaluate(client)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(client.simulations, 1, "the simulation must not be paid for twice")
        self.assertEqual(client.detail_reads, 3)
        self.assertEqual(record["alpha_id"], "ALPHA1")

    def test_the_check_endpoint_gets_the_same_treatment(self) -> None:
        client = FlakyClient(check_404=1)
        record = self._evaluate(client)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(client.simulations, 1)
        self.assertEqual(client.check_reads, 2)

    def test_a_persistent_404_still_fails_after_the_budget(self) -> None:
        client = FlakyClient(detail_404=99)
        record = self._evaluate(client)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["status"], 404)
        self.assertEqual(client.detail_reads, pipeline.ALPHA_READ_RETRIES)
        self.assertEqual(client.simulations, 1)

    def test_other_statuses_are_not_retried(self) -> None:
        client = FlakyClient(detail_404=99, detail_status=403)
        record = self._evaluate(client)
        self.assertEqual(record["status"], "error")
        self.assertEqual(client.detail_reads, 1, "a 403 is an answer, not a race")


class RejectedSimulationTests(unittest.TestCase):
    """A failed simulation must not be mistaken for a missing alpha."""

    def _evaluate(self, client):
        return pipeline.evaluate_candidate(
            client=client, candidate=candidate(), max_wait=5.0, poll_interval=0.0, retries=2
        )

    def test_a_rejected_formula_reports_the_platform_reason(self) -> None:
        client = FlakyClient(
            simulation={
                "id": "SIMULATION_ID",
                "status": "ERROR",
                "alpha": None,
                "message": "Operator ts_decay_linear does not support event inputs.",
            }
        )
        record = self._evaluate(client)
        self.assertEqual(record["status"], "error")
        self.assertIn("does not support event inputs", record["error"]["message"])
        self.assertNotEqual(
            record["error"].get("status"), 404, "the old code reported a bogus 404 here"
        )
        self.assertEqual(client.simulations, 1, "a rejected formula is not retried")
        self.assertEqual(client.detail_reads, 0, "and the alpha endpoint is never queried")

    def test_a_completed_simulation_still_reads_the_alpha(self) -> None:
        client = FlakyClient(simulation={"id": "SIMULATION_ID", "status": "COMPLETE", "alpha": "ALPHA1"})
        record = self._evaluate(client)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["alpha_id"], "ALPHA1")
        self.assertEqual(client.detail_reads, 1)

    def test_the_reason_survives_into_the_negative_examples(self) -> None:
        """The proposer learns from ``failed_attempts``: 404 taught it nothing."""
        from alpha_agent.engine import AlphaResearchAgent

        record = {
            "status": "error",
            "expression": "ts_decay_linear(nws18_relevance, 10)",
            "error": {"message": "Simulation rejected: Operator ts_decay_linear does not support event inputs."},
        }
        summaries = AlphaResearchAgent._failure_summaries([record])
        self.assertEqual(len(summaries), 1)
        self.assertIn("does not support event inputs", summaries[0]["error"])


class RateLimitTests(unittest.TestCase):
    """429 must wait for the platform, not for a fixed 5s backoff."""

    def test_retry_after_is_honoured(self) -> None:
        slept: list[float] = []

        class LimitedClient(FlakyClient):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def simulate(self, payload, *, max_wait, poll_interval):
                self.calls += 1
                raise BrainApiError(
                    "Too many requests.",
                    status=429,
                    headers={"Retry-After": "45"},
                )

        original = pipeline.time.sleep
        pipeline.time.sleep = slept.append
        try:
            record = pipeline.evaluate_candidate(
                client=LimitedClient(), candidate=candidate(), max_wait=5.0, poll_interval=0.0, retries=1
            )
        finally:
            pipeline.time.sleep = original
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"]["status"], 429)
        self.assertEqual(slept, [46.0], "waited Retry-After + 1s instead of 5s")

    def test_a_missing_retry_after_uses_the_default_wait(self) -> None:
        slept: list[float] = []

        class LimitedClient(FlakyClient):
            def simulate(self, payload, *, max_wait, poll_interval):
                raise BrainApiError("Too many requests.", status=429)

        original = pipeline.time.sleep
        pipeline.time.sleep = slept.append
        try:
            pipeline.evaluate_candidate(
                client=LimitedClient(), candidate=candidate(), max_wait=5.0, poll_interval=0.0, retries=0
            )
        finally:
            pipeline.time.sleep = original
        self.assertEqual(slept, [], "no retry budget left, so no sleep")


if __name__ == "__main__":
    unittest.main()
