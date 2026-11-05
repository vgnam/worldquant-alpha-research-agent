"""A stale cookie must not poison a run that also has valid credentials.

Measured: a 93-simulation run died at simulation 21 with
``Authentication rejected (401). Refresh WQB_COOKIE_HEADER`` even though
email/password were valid, because a present cookie header short-circuited the
login entirely.
"""

import tempfile
import types
import unittest
from pathlib import Path

from alpha_agent import engine
from alpha_agent.config import AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig
from worldquant_brain_cli import BrainApiError

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeBrainClient:
    """Records whether it was asked to log in; a pre-set cookie is stale."""

    #: Whether a pre-set cookie is rejected on the validation GET.
    stale_cookie = True
    #: Whether the validation GET is rate limited (which proves nothing).
    rate_limited = False

    def __init__(self, *, base_url=None, timeout=None, cookie_header=None) -> None:
        self.base_url = base_url
        self.cookie_header = cookie_header
        self.opener = types.SimpleNamespace(addheaders=[])
        self.login_calls = 0

    def _request(self, method, path, accept=None):
        if self.cookie_header and not self.login_calls:
            if FakeBrainClient.rate_limited:
                return types.SimpleNamespace(status=429, data=None, headers={"Retry-After": "45"})
            if FakeBrainClient.stale_cookie:
                return types.SimpleNamespace(status=401, data=None, headers={})
        return types.SimpleNamespace(status=200, data={"results": []}, headers={})

    def login(self, email, password):
        self.login_calls += 1
        self.cookie_header = "fresh"


class StaleCookieTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real = engine.BrainClient
        engine.BrainClient = FakeBrainClient
        FakeBrainClient.stale_cookie = True
        FakeBrainClient.rate_limited = False

    def tearDown(self) -> None:
        engine.BrainClient = self._real
        FakeBrainClient.stale_cookie = True
        FakeBrainClient.rate_limited = False

    def _toolbox(self, **auth) -> "engine.ResearchToolbox":
        runtime = AgentRuntimeConfig(
            auth=AuthConfig(
                email=auth.get("email"),
                password=auth.get("password"),
                cookie_header=auth.get("cookie_header"),
            ),
            model=ModelConfig(provider="heuristic"),
            agent=AgentConfig(
                workdir=Path(tempfile.mkdtemp()),
                idea_library=REPO_ROOT / "alpha_pipeline_ideas.json",
                fields_summary=REPO_ROOT / "wqb_data_fields_summary.json",
                allow_offline_fetch=False,
            ),
        )
        return engine.ResearchToolbox(runtime)

    def test_stale_cookie_falls_back_to_the_password_login(self) -> None:
        toolbox = self._toolbox(cookie_header="stale", email="a@b.c", password="pw")
        client = toolbox._build_client()
        self.assertEqual(client.login_calls, 1, "the run must not die on a stale cookie")
        self.assertEqual(client.cookie_header, "fresh")

    def test_a_live_cookie_is_used_without_a_login(self) -> None:
        FakeBrainClient.stale_cookie = False
        toolbox = self._toolbox(cookie_header="good", email="a@b.c", password="pw")
        client = toolbox._build_client()
        self.assertEqual(client.login_calls, 0)

    def test_a_rate_limited_cookie_is_not_trusted(self) -> None:
        """A 429 hides whether the cookie is still valid, so log in instead."""
        FakeBrainClient.rate_limited = True
        toolbox = self._toolbox(cookie_header="maybe", email="a@b.c", password="pw")
        client = toolbox._build_client()
        self.assertEqual(client.login_calls, 1)

    def test_a_stale_cookie_without_credentials_still_fails_loudly(self) -> None:
        toolbox = self._toolbox(cookie_header="stale")
        with self.assertRaises(BrainApiError):
            toolbox._build_client()

    def test_no_cookie_means_a_plain_login(self) -> None:
        toolbox = self._toolbox(email="a@b.c", password="pw")
        client = toolbox._build_client()
        self.assertEqual(client.login_calls, 1)


if __name__ == "__main__":
    unittest.main()
