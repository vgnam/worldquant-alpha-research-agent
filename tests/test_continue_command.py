import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import alpha_research_agent as cli


def _args(workdir: Path, **overrides) -> argparse.Namespace:
    values = {
        "email": "me@example.com",
        "password": "secret",
        "cookie_header": "old-cookie",
        "base_url": "https://api.example",
        "timeout": 30.0,
        "pretty": False,
        "workdir": str(workdir),
        "budget": 100,
        "search_mode": "hybrid",
        "submission_mode": "auto_approved",
        "max_submissions": 100,
        "tree_state_path": str(workdir / "tree.json"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ResumeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = Path(tempfile.mkdtemp())

    def test_saved_state_keeps_settings_but_never_credentials(self) -> None:
        path = cli.save_resume_state(_args(self.workdir), budget=100, evaluated=42, stop_reason="401")
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["remaining"], 58)
        self.assertEqual(saved["settings"]["max_submissions"], 100)
        text = path.read_text(encoding="utf-8")
        for secret in ("secret", "old-cookie", "me@example.com"):
            self.assertNotIn(secret, text)

    def test_continue_runs_the_remaining_budget_with_the_current_cookie(self) -> None:
        cli.save_resume_state(_args(self.workdir), budget=100, evaluated=42, stop_reason="401")
        current = _args(self.workdir, cookie_header="new-cookie", budget=None)
        with mock.patch.object(cli, "command_run", return_value={"ok": True}) as run:
            cli.command_continue(current)
        resumed = run.call_args.args[0]
        self.assertEqual(resumed.budget, 58)
        self.assertEqual(resumed.cookie_header, "new-cookie")
        self.assertEqual(resumed.search_mode, "hybrid")
        self.assertEqual(resumed.tree_state_path, str(self.workdir / "tree.json"))

    def test_an_explicit_budget_overrides_the_remainder(self) -> None:
        cli.save_resume_state(_args(self.workdir), budget=100, evaluated=42, stop_reason="401")
        with mock.patch.object(cli, "command_run", return_value={}) as run:
            cli.command_continue(_args(self.workdir, budget=10))
        self.assertEqual(run.call_args.args[0].budget, 10)

    def test_a_flag_missing_from_the_saved_run_takes_todays_default(self) -> None:
        cli.save_resume_state(_args(self.workdir), budget=100, evaluated=42, stop_reason="401")
        with mock.patch.dict("os.environ", {"ALPHA_AGENT_TREE_MAX_DEPTH": "3"}), \
             mock.patch.object(cli, "command_run", return_value={}) as run:
            cli.command_continue(_args(self.workdir, budget=None))
        resumed = run.call_args.args[0]
        self.assertEqual(resumed.tree_max_depth, 3)
        self.assertEqual(resumed.max_submissions, 100, "saved values still win")

    def test_a_finished_run_has_nothing_to_continue(self) -> None:
        cli.save_resume_state(_args(self.workdir), budget=100, evaluated=100, stop_reason="Budget exhausted.")
        with mock.patch.object(cli, "command_run") as run:
            result = cli.command_continue(_args(self.workdir, budget=None))
        run.assert_not_called()
        self.assertEqual(result["status"], "nothing_to_continue")

    def test_a_run_that_crashes_still_records_what_it_spent(self) -> None:
        results = self.workdir / "results.jsonl"
        results.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

        class Agent:
            def __init__(self, *args, **kwargs):
                pass

            def run(self):
                with results.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"status": "ok"}) + "\n")
                    handle.write(json.dumps({"status": "error"}) + "\n")
                raise RuntimeError("network down")

        with mock.patch.object(cli, "AlphaResearchAgent", Agent), \
             mock.patch.object(cli, "build_runtime", return_value=mock.Mock()), \
             mock.patch.object(cli, "build_planner", return_value=None):
            with self.assertRaises(RuntimeError):
                cli.command_run(_args(self.workdir, interactive_approval=False))
        saved = json.loads((self.workdir / cli.RESUME_FILE).read_text(encoding="utf-8"))
        self.assertEqual((saved["evaluated"], saved["remaining"]), (1, 99))


if __name__ == "__main__":
    unittest.main()
