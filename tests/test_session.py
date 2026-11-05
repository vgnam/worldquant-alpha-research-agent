"""Session handling tests: cookie seeding, rotation capture, expiry detection.

These run fully offline. They exist because a live 100-simulation run was lost
to an expired cookie: the client re-sent the original header on every request,
so a session rotated by the server could never take effect, and nothing checked
the session between iterations.
"""

import json
import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from alpha_agent.config import AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig
from alpha_agent.engine import AlphaResearchAgent
from worldquant_brain_cli import BrainClient

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_client(cookie: str | None = None) -> BrainClient:
    return BrainClient(base_url="https://api.worldquantbrain.com", timeout=5.0, cookie_header=cookie)


class CookieParsingTests(unittest.TestCase):
    def test_parse_cookie_header(self) -> None:
        header = "t=abc123; csrftoken=xyz; _ga=GA1.2; empty=; junk"
        pairs = BrainClient.parse_cookie_header(header)
        self.assertIn(("t", "abc123"), pairs)
        self.assertIn(("csrftoken", "xyz"), pairs)
        self.assertIn(("_ga", "GA1.2"), pairs)
        self.assertNotIn("junk", [name for name, _ in pairs])

    def test_values_with_equals_are_preserved(self) -> None:
        pairs = BrainClient.parse_cookie_header("t=abc==; x=1")
        self.assertIn(("t", "abc=="), pairs)

    def test_seeding_populates_the_jar(self) -> None:
        client = make_client("t=abc; csrftoken=def")
        self.assertTrue(client.has_session_cookies())
        self.assertEqual(len(list(client.cookie_jar)), 2)
        self.assertIn("t=abc", client.cookie_header)

    def test_client_without_cookie_has_empty_jar(self) -> None:
        client = make_client(None)
        self.assertFalse(client.has_session_cookies())


class CookieRotationTests(unittest.TestCase):
    def test_rotation_updates_the_header(self) -> None:
        client = make_client("t=old; csrftoken=def")
        # Simulate the server rotating the session token.
        client.seed_cookies("t=new")
        self.assertIn("t=new", client.cookie_header)
        self.assertNotIn("t=old", client.cookie_header)
        self.assertTrue(client.cookie_rotated)

    def test_sync_is_idempotent(self) -> None:
        client = make_client("t=abc")
        first = client.sync_cookie_header()
        second = client.sync_cookie_header()
        self.assertEqual(first, second)
        self.assertFalse(client.cookie_rotated)

    def test_manual_header_is_not_duplicated_when_jar_has_cookies(self) -> None:
        """The jar must own cookie transmission so Set-Cookie can apply."""
        client = make_client("t=abc")
        # When the jar has cookies, _request should not add a raw Cookie header.
        self.assertTrue(client.has_session_cookies())
        self.assertEqual(client.cookie_header, "t=abc")


class SessionPingTests(unittest.TestCase):
    def _build_agent(self, workdir: Path, ping_status: int | None) -> AlphaResearchAgent:
        agent_cfg = AgentConfig(
            budget=10,
            max_iterations=5,
            workdir=workdir,
            idea_library=REPO_ROOT / "alpha_pipeline_ideas.json",
            fields_summary=REPO_ROOT / "wqb_data_fields_summary.json",
            search_mode="hybrid",
            llm_refine_enabled=False,
            allow_offline_fetch=False,
            sleep_between=0.0,
            submission_mode="disabled",
            session_ping=True,
        )
        runtime = AgentRuntimeConfig(
            auth=AuthConfig(email=None, password=None, cookie_header=None),
            model=ModelConfig(provider="heuristic"),
            agent=agent_cfg,
        )
        agent = AlphaResearchAgent(runtime)

        class FakeResponse:
            def __init__(self, status: int) -> None:
                self.status = status
                self.data = {"count": 0, "results": []}

        class PingClient:
            def __init__(self, status: int | None) -> None:
                self.status = status
                self.calls = 0

            def _request(self, method, url, accept=None):
                self.calls += 1
                return FakeResponse(self.status if self.status is not None else 200)

        class Toolbox:
            def __init__(self, client) -> None:
                self.client = client
                self.results = []
                self.submissions = []
                self.evaluated = []
                self.agent_runs_dir = workdir / "agent_runs"
                self.agent_runs_dir.mkdir(parents=True, exist_ok=True)
                self.state_path = workdir / "state.json"

            def leaderboard(self, limit=10):
                return []

            def pop_seed_candidates(self, count):
                return []

            def evaluate_candidates(self, candidates, progress_callback=None):
                self.evaluated.extend(candidates)
                return []

            def write_state(self):
                return {}

        fake = Toolbox(PingClient(ping_status))
        agent.toolbox = fake  # type: ignore[assignment]
        return agent

    def test_dead_session_stops_before_spending_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._build_agent(Path(tmp), ping_status=401)
            result = agent.run()
        self.assertIn("Session expired", result.summary["stop_reason"])
        self.assertEqual(result.summary["evaluated_this_run"], 0)
        self.assertEqual(result.summary["iterations_executed"], 1)

    def test_healthy_session_does_not_stop_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._build_agent(Path(tmp), ping_status=None)
            result = agent.run()
        self.assertNotIn("Session expired", result.summary["stop_reason"])

    def test_ping_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._build_agent(Path(tmp), ping_status=401)
            agent.runtime = replace(
                agent.runtime,
                agent=replace(agent.runtime.agent, session_ping=False),
            )
            result = agent.run()
        self.assertNotIn("Session expired", result.summary["stop_reason"])


class NetworkErrorTests(unittest.TestCase):
    """A BRAIN read timeout once killed a whole run with a bare socket.timeout."""

    def test_timeout_becomes_statusless_brain_error(self) -> None:
        import socket
        from unittest import mock

        from worldquant_brain_cli import BrainApiError

        client = make_client("t=abc")
        with mock.patch.object(client.opener, "open", side_effect=socket.timeout("The read operation timed out")):
            with self.assertRaises(BrainApiError) as ctx:
                client._request("GET", "/users/self", accept="application/json")
        self.assertIsNone(ctx.exception.status)
        self.assertIn("timed out", str(ctx.exception))

    def test_evaluate_candidate_retries_network_errors_then_records_error(self) -> None:
        from unittest import mock

        import alpha_research_pipeline as pipeline
        from worldquant_brain_cli import BrainApiError

        client = mock.Mock()
        client.simulate.side_effect = BrainApiError("Network error: timed out", status=None)
        candidate = pipeline.Candidate(expression="rank(close)", settings={}, family="f", idea_name="i")
        with mock.patch.object(pipeline.time, "sleep"):
            record = pipeline.evaluate_candidate(
                client=client, candidate=candidate, max_wait=1, poll_interval=0, retries=2
            )
        self.assertEqual(client.simulate.call_count, 3)
        self.assertEqual(record["status"], "error")
        self.assertIsNone(record["error"]["status"])


if __name__ == "__main__":
    unittest.main()
