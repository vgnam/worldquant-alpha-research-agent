import io
import json
import os
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from alpha_agent.config import ModelConfig
from alpha_agent.llm_http import LLMRequestError, post_chat_completion
from alpha_agent.refine_llm import LLMRefinementProposer


def _http_error(code: int, body: str = "{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok(payload: dict) -> _Response:
    return _Response(json.dumps(payload).encode())


class PostChatCompletionTests(unittest.TestCase):
    def test_retries_transient_503_then_succeeds(self):
        sleeps = []
        side_effect = [_http_error(503, '{"code":"maintenance"}'), _ok({"choices": []})]
        with mock.patch("urllib.request.urlopen", side_effect=side_effect) as urlopen:
            result = post_chat_completion(
                base_url="http://x/v1", body={}, api_key="k", timeout=1, sleep=sleeps.append
            )
        self.assertEqual(result, {"choices": []})
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleeps, [2.0])

    def test_retries_socket_read_timeout(self):
        import socket

        side_effect = [socket.timeout("The read operation timed out"), _ok({"choices": []})]
        with mock.patch("urllib.request.urlopen", side_effect=side_effect) as urlopen:
            post_chat_completion(base_url="http://x", body={}, api_key="k", timeout=1, sleep=lambda _: None)
        self.assertEqual(urlopen.call_count, 2)

    def test_non_retryable_status_fails_immediately(self):
        with mock.patch("urllib.request.urlopen", side_effect=[_http_error(401)]) as urlopen:
            with self.assertRaises(LLMRequestError) as ctx:
                post_chat_completion(base_url="http://x", body={}, api_key="k", timeout=1, sleep=lambda _: None)
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(urlopen.call_count, 1)

    def test_gives_up_after_retries(self):
        errors = [_http_error(503) for _ in range(4)]
        with mock.patch("urllib.request.urlopen", side_effect=errors) as urlopen:
            with self.assertRaises(LLMRequestError):
                post_chat_completion(
                    base_url="http://x", body={}, api_key="k", timeout=1, retries=3, sleep=lambda _: None
                )
        self.assertEqual(urlopen.call_count, 4)


class ProposerAccountingTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"TEST_LLM_KEY": "k"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.proposer = LLMRefinementProposer(ModelConfig(provider="openai", api_key_env="TEST_LLM_KEY"))
        self.node = SimpleNamespace(expression="rank(close)", settings={}, family="f")

    def _propose(self):
        return self.proposer._propose_via_model(
            node=self.node, e_vector={"SHARPE": -0.1}, blacklist=[], operators=["rank"], context={}
        )

    def test_failures_do_not_consume_call_cap(self):
        with mock.patch(
            "alpha_agent.refine_llm.post_chat_completion", side_effect=LLMRequestError("HTTP 503: down", 503)
        ):
            self.assertIsNone(self._propose())
            self.assertIsNone(self._propose())
        self.assertEqual(self.proposer.calls, 0)
        self.assertEqual(self.proposer.failures, 2)
        self.assertEqual(self.proposer.consecutive_failures, 2)
        self.assertIn("503", self.proposer.last_error)

    def test_success_resets_consecutive_failures(self):
        content = json.dumps({"expressions": [{"expression": "rank(open)"}]})
        completion = {"choices": [{"message": {"content": content}}]}
        with mock.patch(
            "alpha_agent.refine_llm.post_chat_completion",
            side_effect=[LLMRequestError("HTTP 503: down", 503), completion],
        ):
            self.assertIsNone(self._propose())
            proposal = self._propose()
        self.assertEqual([item.expression for item in proposal.expressions], ["rank(open)"])
        self.assertEqual(self.proposer.calls, 1)
        self.assertEqual(self.proposer.consecutive_failures, 0)


    def test_prompt_gives_metric_directions_not_numeric_targets(self):
        content = json.dumps({"expressions": [{"expression": "rank(open)"}]})
        completion = {"choices": [{"message": {"content": content}}]}
        with mock.patch("alpha_agent.refine_llm.post_chat_completion", return_value=completion) as post:
            self._propose()
        prompt = json.loads(post.call_args.kwargs["body"]["messages"][1]["content"])
        self.assertIn("Sharpe", prompt["metric_directions"]["higher_is_better"])
        self.assertTrue(any("SELF_CORRELATION" in item for item in prompt["metric_directions"]["lower_is_better"]))
        self.assertNotRegex(prompt["quality_note"], r"\d")
        self.assertNotIn("1.25", json.dumps(prompt))


class SettingsPatchTests(unittest.TestCase):
    def test_drops_unknown_keys_and_casts(self):
        from alpha_agent.refine_llm import _clean_settings_patch

        patch = {"grouping_field": {"from": "sector"}, "decay": "4", "neutralization": "sector", "truncation": "x"}
        self.assertEqual(_clean_settings_patch(patch), {"decay": 4, "neutralization": "SECTOR"})
        self.assertEqual(_clean_settings_patch(None), {})


class FailureSummaryTests(unittest.TestCase):
    def test_newest_first_skips_passing_and_filters_parent(self):
        from alpha_agent.engine import AlphaResearchAgent

        records = [
            {"status": "ok", "expression": "a", "parent_key": "p", "quality_checks_ready": False,
             "metrics": {"sharpe": 0.4}, "failed_checks": ["LOW_SHARPE"], "metadata": {"rationale": "[e1] x"}},
            {"status": "ok", "expression": "b", "parent_key": "p", "quality_checks_ready": True},
            {"status": "error", "expression": "c", "parent_key": "q", "error": {"message": "Not found."}},
        ]
        everything = AlphaResearchAgent._failure_summaries(records)
        self.assertEqual([item["expression"] for item in everything], ["c", "a"])
        self.assertEqual(everything[0]["error"], "Not found.")
        self.assertEqual(everything[1]["operator"], "e1")
        children = AlphaResearchAgent._failure_summaries(records, parent_key="p")
        self.assertEqual([item["expression"] for item in children], ["a"])


if __name__ == "__main__":
    unittest.main()
