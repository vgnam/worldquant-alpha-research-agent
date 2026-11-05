#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import alpha_research_pipeline as pipeline
from alpha_agent.config import AgentConfig, AgentRuntimeConfig, AuthConfig, ModelConfig
from alpha_agent.engine import AlphaResearchAgent, ResearchToolbox
from alpha_agent.evaluation import run_evaluation_suite
from alpha_agent.planner import HeuristicPlanner, OpenAIJsonPlanner
from local_env import load_local_dotenv
from worldquant_brain_cli import API_BASE, BrainApiError, DEFAULT_MAX_WAIT, DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT


DEFAULT_AGENT_WORKDIR = Path(".alpha_agent")
DEFAULT_EVAL_CASES = Path("docs") / "eval_cases.json"

load_local_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha Research Agent with planner-driven orchestration, tool use, and guarded submission flow."
    )
    add_auth_args(parser)
    add_common_agent_args(parser)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full alpha research agent loop.")
    add_run_args(run_parser)
    run_parser.set_defaults(func=command_run)

    continue_parser = subparsers.add_parser(
        "continue",
        help="Resume the last run (e.g. after refreshing WQB_COOKIE_HEADER) with its settings and remaining budget.",
    )
    continue_parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Simulations for this continuation (default: what the last run had left).",
    )
    continue_parser.set_defaults(func=command_continue)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show current leaderboard from workdir.")
    leaderboard_parser.add_argument("--limit", type=int, default=10)
    leaderboard_parser.set_defaults(func=command_leaderboard)

    eval_parser = subparsers.add_parser("evaluate", help="Run baseline vs agent evaluation suite.")
    eval_parser.add_argument(
        "--cases",
        default=str(DEFAULT_EVAL_CASES),
        help="Path to evaluation JSON cases list.",
    )
    eval_parser.add_argument(
        "--output",
        default="",
        help="Optional output report path. Defaults to <workdir>/evaluation/report.json",
    )
    eval_parser.set_defaults(func=command_evaluate)

    return parser.parse_args()


def add_auth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", default=os.getenv("WQB_EMAIL"), help="BRAIN account email.")
    parser.add_argument("--password", default=os.getenv("WQB_PASSWORD"), help="BRAIN account password.")
    parser.add_argument(
        "--cookie-header",
        default=os.getenv("WQB_COOKIE_HEADER"),
        help="Optional raw Cookie header. If set, email/password login is skipped.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("WQB_API_BASE", API_BASE),
        help=f"API base URL. Defaults to {API_BASE}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("WQB_TIMEOUT", DEFAULT_TIMEOUT)),
        help="Single request timeout in seconds.",
    )


def add_common_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idea-library",
        default=os.getenv("ALPHA_AGENT_IDEA_LIBRARY") or str(pipeline.DEFAULT_IDEA_LIBRARY),
        help="Path to idea library JSON.",
    )
    parser.add_argument(
        "--fields-summary",
        default=os.getenv("ALPHA_AGENT_FIELDS_SUMMARY") or str(pipeline.DEFAULT_FIELDS_SUMMARY),
        help="Path to cached fields summary JSON.",
    )
    parser.add_argument(
        "--workdir",
        default=os.getenv("ALPHA_AGENT_WORKDIR") or str(DEFAULT_AGENT_WORKDIR),
        help="Agent working directory for results/state/logs.",
    )
    parser.add_argument(
        "--planner-provider",
        choices=("heuristic", "openai"),
        default=os.getenv("ALPHA_AGENT_PLANNER_PROVIDER", "heuristic"),
        help="Planning backend.",
    )
    parser.add_argument(
        "--planner-model",
        default=os.getenv("ALPHA_AGENT_PLANNER_MODEL", "gpt5.5"),
        help="Model name for planner provider.",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_PLANNER_TEMPERATURE", "0.1")),
        help="Planner sampling temperature.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=os.getenv("ALPHA_AGENT_PLANNER_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL for planner provider.",
    )
    parser.add_argument(
        "--planner-api-key-env",
        default=os.getenv("ALPHA_AGENT_PLANNER_API_KEY_ENV", "OPENAI_API_KEY"),
        help="Environment variable name that stores planner API key.",
    )
    parser.add_argument(
        "--planner-timeout",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_PLANNER_TIMEOUT", "90")),
        help="Planner / refinement LLM request timeout in seconds.",
    )
    parser.add_argument(
        "--planner-reasoning-effort",
        choices=["low", "medium", "high"],
        default=os.getenv("ALPHA_AGENT_PLANNER_REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or None,
        help="reasoning_effort sent to the LLM (omitted when unset).",
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget", type=int, default=int(os.getenv("ALPHA_AGENT_BUDGET", "24")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("ALPHA_AGENT_MAX_ITERATIONS", "12")))
    parser.add_argument("--seed-fraction", type=float, default=float(os.getenv("ALPHA_AGENT_SEED_FRACTION", "0.7")))
    parser.add_argument("--refine-top-k", type=int, default=int(os.getenv("ALPHA_AGENT_REFINE_TOP_K", "8")))
    parser.add_argument(
        "--robustness-top-k",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_ROBUSTNESS_TOP_K", "3")),
    )
    parser.add_argument(
        "--robustness-score-threshold",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_ROBUSTNESS_SCORE_THRESHOLD", "500")),
    )
    parser.add_argument("--family", action="append", default=[], help="Optional family filter (repeatable).")
    parser.add_argument(
        "--max-family-budget-share",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MAX_FAMILY_BUDGET_SHARE", "0.45")),
    )
    parser.add_argument(
        "--min-expression-novelty",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MIN_EXPRESSION_NOVELTY", "0.10")),
    )
    parser.add_argument("--shuffle-seeds", action="store_true", default=True, help="Shuffle generated seeds.")
    parser.add_argument("--no-shuffle-seeds", action="store_false", dest="shuffle_seeds")
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("ALPHA_AGENT_RANDOM_SEED", "7")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("ALPHA_AGENT_RETRIES", "2")))
    parser.add_argument("--sleep-between", type=float, default=float(os.getenv("ALPHA_AGENT_SLEEP_BETWEEN", "1.0")))
    parser.add_argument("--max-wait", type=float, default=float(os.getenv("WQB_MAX_WAIT", str(DEFAULT_MAX_WAIT))))
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("WQB_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))),
    )
    parser.add_argument("--allow-pending-checks", action="store_true")
    parser.add_argument(
        "--submission-mode",
        choices=("disabled", "manual", "auto_approved"),
        default="disabled",
        help="Submission governance mode.",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_MAX_SUBMISSIONS", "1")),
        help="Cap on automatic submissions per run (BRAIN submissions are final).",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("ALPHA_AGENT_REGION", "USA"),
        help="Arena region every seed is simulated in; the quality gates and the correlation pool depend on it.",
    )
    parser.add_argument(
        "--universe",
        default=os.getenv("ALPHA_AGENT_UNIVERSE", "TOP3000"),
        help="Arena universe (TOP3000/TOP1000/TOP500/TOPDIV3000/... per region).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_DELAY", "1")),
        help="Arena delay (0 or 1).",
    )
    parser.add_argument(
        "--pool-aware-seeds",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("ALPHA_AGENT_POOL_AWARE_SEEDS", "1") not in ("0", "false", "False")),
        help="Order the seed queue by how little the account pool covers each seed's data fields.",
    )
    parser.add_argument(
        "--interactive-approval",
        action="store_true",
        help="Prompt approval in terminal when submission-mode=manual.",
    )
    add_hybrid_args(parser)


def add_hybrid_args(parser: argparse.ArgumentParser) -> None:
    """Flags for the hybrid (multi-objective MCTS) search."""
    parser.add_argument(
        "--search-mode",
        choices=("legacy", "hybrid", "mo-hifo"),
        default=os.getenv("ALPHA_AGENT_SEARCH_MODE", "legacy"),
        help="legacy = stage machine (default); hybrid = multi-objective MCTS refinement; "
        "mo-hifo = population search on Sharpe/returns/turnover (MEoH selection, HiFo insights).",
    )
    parser.add_argument(
        "--tree-batch-size",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_TREE_BATCH_SIZE", "2")),
        help="Simulations spent per hybrid iteration.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_TREE_MAX_DEPTH", "5")),
        help="Deepest refinement chain below a root; also applied to a resumed tree.",
    )
    parser.add_argument(
        "--uct-c",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_UCT_C", "0.1")),
        help="UCT exploration weight (MCTS-AHD lambda_0) on normalised Q; decays with remaining budget.",
    )
    parser.add_argument(
        "--widen-alpha",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_WIDEN_ALPHA", "0.5")),
        help="Progressive widening exponent: widen while floor(work**alpha) > children.",
    )
    parser.add_argument(
        "--widen-patience",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_WIDEN_PATIENCE", "3")),
        help="Inner nodes widen only after this many evaluations added nothing to their front (0 = never gate).",
    )
    parser.add_argument(
        "--tree-value",
        choices=("ohvi", "scalar"),
        default=os.getenv("ALPHA_AGENT_TREE_VALUE", "ohvi"),
        help="Child selection: ohvi = optimistic hypervolume of each subtree front; scalar = UCT on max reward.",
    )
    parser.add_argument(
        "--mo-objectives",
        default=os.getenv("ALPHA_AGENT_MO_OBJECTIVES", "sharpe,turnover,margin,divergence"),
        help=(
            "Comma-separated Pareto objectives: sharpe, turnover, margin, divergence, "
            "returns, drawdown, fitness, quality, correlation."
        ),
    )
    parser.add_argument(
        "--mo-front-size",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_MO_FRONT_SIZE", "24")),
        help="Max points per Pareto front; the smallest hypervolume contributor is dropped beyond it.",
    )
    parser.add_argument(
        "--mo-epsilon",
        type=float,
        default=float(os.getenv("ALPHA_AGENT_MO_EPSILON", "0.01")),
        help="Near-duplicate tolerance on normalised objectives.",
    )
    parser.add_argument(
        "--children-per-expansion",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_CHILDREN_PER_EXPANSION", "1")),
        help="Simulated children kept per operator call.",
    )
    parser.add_argument(
        "--initial-seeds",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_INITIAL_SEEDS", "0")),
        help="Seed an empty tree with this many structurally diverse idea-library seeds (0 = old behaviour). "
        "With --root-source llm: the number of LLM-written roots (0 = 4, as in MCTS-AHD).",
    )
    parser.add_argument(
        "--root-source",
        choices=("library", "llm"),
        default=os.getenv("ALPHA_AGENT_ROOT_SOURCE", "library"),
        help="library = idea-library seeds as roots; llm = MCTS-AHD init (one i1 root from scratch, then e1 roots).",
    )
    parser.add_argument(
        "--tree-state-path",
        default=os.getenv("ALPHA_AGENT_TREE_STATE_PATH") or None,
        help="Search tree file (default <workdir>/search_tree.json); a new path starts a fresh tree.",
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_MAX_ROOTS", "0")),
        help="Cap on subtree roots created by widening (0 = unlimited).",
    )
    parser.add_argument(
        "--refine-proposals",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_REFINE_PROPOSALS", "6")),
        help="Candidates requested per expanded node before offline filtering.",
    )
    parser.add_argument(
        "--novelty-refresh",
        action="store_true",
        default=os.getenv("ALPHA_AGENT_NOVELTY_REFRESH", "false").lower() == "true",
        help="Refetch the account alpha pool instead of using the cache.",
    )
    parser.add_argument(
        "--no-llm-refine",
        action="store_false",
        dest="llm_refine_enabled",
        help="Disable the LLM refinement proposer and use deterministic mutations.",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_MAX_LLM_CALLS", "200")),
        help="Hard cap on successful refinement LLM calls per run (one leaf expansion costs up to four).",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        default=os.getenv("ALPHA_AGENT_REQUIRE_LLM", "false").lower() == "true",
        help="Hybrid mode: stop instead of spending simulations on deterministic fallback mutations when the LLM fails.",
    )
    parser.add_argument(
        "--llm-failure-limit",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_LLM_FAILURE_LIMIT", "3")),
        help="Consecutive LLM failures (each already retried) before --require-llm stops the run.",
    )
    parser.add_argument(
        "--population-state-path",
        default=os.getenv("ALPHA_AGENT_POPULATION_STATE_PATH") or None,
        help="mo-hifo population file (default <workdir>/mo_hifo_population.json); a new path starts fresh.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_POPULATION_SIZE", "10")),
        help="mo-hifo: alphas kept by survival and offspring per generation.",
    )
    parser.add_argument(
        "--crossover-parents",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_CROSSOVER_PARENTS", "5")),
        help="mo-hifo: parents drawn for the e1/e2 operators.",
    )
    parser.add_argument(
        "--insights-per-prompt",
        type=int,
        default=int(os.getenv("ALPHA_AGENT_INSIGHTS_PER_PROMPT", "3")),
        help="mo-hifo: insights from the pool shown in each prompt.",
    )
    parser.add_argument(
        "--no-insight-extraction",
        action="store_false",
        dest="extract_insights",
        help="mo-hifo: keep the seed insights only (no extra model call per generation).",
    )


#: Written by ``run`` so ``continue`` can resume with the same settings and only
#: the simulations that were left. BRAIN sessions last about four hours, so a
#: long run outlives its cookie. Credentials are never written: ``continue``
#: takes them from the current flags / .env (i.e. the refreshed cookie).
RESUME_FILE = "last_run.json"
_UNSAVED_ARGS = frozenset({"email", "password", "cookie_header", "base_url", "timeout", "pretty", "func", "command"})


def _ok_result_count(workdir: Path) -> int:
    path = Path(workdir) / "results.jsonl"
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            count += json.loads(line).get("status") == "ok"
        except (ValueError, AttributeError):
            continue
    return count


def save_resume_state(args: argparse.Namespace, *, budget: int, evaluated: int, stop_reason: Optional[str]) -> Path:
    settings = {key: value for key, value in vars(args).items() if key not in _UNSAVED_ARGS}
    payload = {
        "settings": settings,
        "budget": int(budget),
        "evaluated": int(evaluated),
        "remaining": max(0, int(budget) - int(evaluated)),
        "stop_reason": stop_reason,
        "saved_at": pipeline.iso_now(),
    }
    path = Path(args.workdir) / RESUME_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def command_run(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args)
    planner = build_planner(runtime.model)
    approval_callback = None
    if args.submission_mode == "manual" and args.interactive_approval:
        approval_callback = cli_manual_approval
    agent = AlphaResearchAgent(runtime, planner=planner, approval_callback=approval_callback)
    budget = int(args.budget)
    # Spent budget is counted from the results store, so a run that dies with
    # an exception still leaves an accurate "remaining" behind.
    ok_before = _ok_result_count(Path(args.workdir))
    stop_reason: Optional[str] = "Interrupted before the run finished."
    try:
        payload = agent.run().to_dict()
        stop_reason = (payload.get("summary") or payload).get("stop_reason")
        return payload
    finally:
        save_resume_state(
            args,
            budget=budget,
            evaluated=_ok_result_count(Path(args.workdir)) - ok_before,
            stop_reason=stop_reason,
        )


def command_continue(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.workdir) / RESUME_FILE
    if not path.exists():
        raise ValueError(f"No saved run in {path}; start one with `run` first.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    budget = int(args.budget) if args.budget is not None else int(saved.get("remaining") or 0)
    if budget <= 0:
        return {"status": "nothing_to_continue", "resume_file": str(path), **{k: saved.get(k) for k in ("budget", "evaluated", "stop_reason")}}
    # Flags added after the run was saved take today's default (.env); saved
    # values win for everything the run already had.
    run_defaults = argparse.ArgumentParser(add_help=False)
    add_run_args(run_defaults)
    resumed = argparse.Namespace(**{**vars(run_defaults.parse_args([])), **saved["settings"]})
    # Auth and connection come from now (the refreshed cookie), not the saved run.
    for key in ("email", "password", "cookie_header", "base_url", "timeout", "pretty"):
        setattr(resumed, key, getattr(args, key))
    resumed.budget = budget
    return command_run(resumed)


def command_leaderboard(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args, run_overrides={})
    toolbox = ResearchToolbox(runtime)
    leaderboard = [pipeline.compact_record(item) for item in toolbox.leaderboard(limit=args.limit)]
    return {
        "count": len(leaderboard),
        "workdir": str(runtime.agent.workdir),
        "leaderboard": leaderboard,
    }


def command_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = build_runtime(args, run_overrides={})
    case_file = Path(args.cases)
    output_path = Path(args.output) if args.output else Path(runtime.agent.workdir) / "evaluation" / "report.json"
    return run_evaluation_suite(
        runtime=runtime,
        case_file=case_file,
        output_path=output_path,
    )


def build_runtime(args: argparse.Namespace, run_overrides: Optional[Dict[str, Any]] = None) -> AgentRuntimeConfig:
    auth = AuthConfig(
        email=args.email,
        password=args.password,
        cookie_header=args.cookie_header,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    model = ModelConfig(
        provider=args.planner_provider,
        model=args.planner_model,
        temperature=args.planner_temperature,
        base_url=args.planner_base_url,
        api_key_env=args.planner_api_key_env,
        timeout=float(getattr(args, "planner_timeout", 90.0)),
        reasoning_effort=getattr(args, "planner_reasoning_effort", None),
    )
    base_agent = AgentConfig(
        budget=getattr(args, "budget", 24),
        max_iterations=getattr(args, "max_iterations", 12),
        seed_fraction=getattr(args, "seed_fraction", 0.7),
        refine_top_k=getattr(args, "refine_top_k", 8),
        robustness_top_k=getattr(args, "robustness_top_k", 3),
        robustness_score_threshold=getattr(args, "robustness_score_threshold", 500.0),
        family_filter=tuple(getattr(args, "family", []) or ()),
        max_family_budget_share=getattr(args, "max_family_budget_share", 0.45),
        min_expression_novelty=getattr(args, "min_expression_novelty", 0.10),
        shuffle_seeds=getattr(args, "shuffle_seeds", True),
        random_seed=getattr(args, "random_seed", 7),
        retries=getattr(args, "retries", 2),
        sleep_between=getattr(args, "sleep_between", 1.0),
        max_wait=getattr(args, "max_wait", DEFAULT_MAX_WAIT),
        poll_interval=getattr(args, "poll_interval", DEFAULT_POLL_INTERVAL),
        allow_pending_checks=getattr(args, "allow_pending_checks", False),
        submission_mode=getattr(args, "submission_mode", "disabled"),
        max_submissions_per_run=getattr(args, "max_submissions", 1),
        workdir=Path(args.workdir),
        idea_library=Path(args.idea_library),
        fields_summary=Path(args.fields_summary),
        region=getattr(args, "region", "USA"),
        universe=getattr(args, "universe", "TOP3000"),
        delay=getattr(args, "delay", 1),
        pool_aware_seeds=getattr(args, "pool_aware_seeds", True),
        search_mode=getattr(args, "search_mode", "legacy"),
        tree_batch_size=getattr(args, "tree_batch_size", 2),
        tree_max_depth=getattr(args, "tree_max_depth", 5),
        uct_c=getattr(args, "uct_c", 0.1),
        widen_alpha=getattr(args, "widen_alpha", 0.5),
        widen_patience=getattr(args, "widen_patience", 3),
        tree_value=getattr(args, "tree_value", "ohvi"),
        mo_objectives=tuple(
            name.strip()
            for name in str(getattr(args, "mo_objectives", "sharpe,turnover,margin,divergence")).split(",")
            if name.strip()
        ),
        mo_front_size=getattr(args, "mo_front_size", 24),
        mo_epsilon=getattr(args, "mo_epsilon", 0.01),
        children_per_expansion=getattr(args, "children_per_expansion", 1),
        max_roots=getattr(args, "max_roots", 0),
        initial_seed_count=getattr(args, "initial_seeds", 0),
        root_source=getattr(args, "root_source", "library"),
        tree_state_path=Path(args.tree_state_path) if getattr(args, "tree_state_path", None) else None,
        refine_proposals=getattr(args, "refine_proposals", 6),
        novelty_refresh=getattr(args, "novelty_refresh", False),
        llm_refine_enabled=getattr(args, "llm_refine_enabled", True),
        max_llm_calls=getattr(args, "max_llm_calls", 64),
        require_llm=getattr(args, "require_llm", False),
        llm_failure_limit=getattr(args, "llm_failure_limit", 3),
        mo_hifo_state_path=(
            Path(args.population_state_path) if getattr(args, "population_state_path", None) else None
        ),
        population_size=getattr(args, "population_size", 10),
        crossover_parents=getattr(args, "crossover_parents", 5),
        insights_per_prompt=getattr(args, "insights_per_prompt", 3),
        extract_insights=getattr(args, "extract_insights", True),
    )
    if run_overrides:
        base_agent = replace(base_agent, **run_overrides)
    return AgentRuntimeConfig(auth=auth, model=model, agent=base_agent)


def build_planner(model_config: ModelConfig) -> Any:
    if model_config.provider == "openai":
        return OpenAIJsonPlanner(model_config=model_config)
    return HeuristicPlanner()


def cli_manual_approval(candidate: Dict[str, Any]) -> bool:
    alpha_id = candidate.get("alpha_id")
    score = candidate.get("score")
    family = candidate.get("family")
    prompt = f"Approve submit for alpha {alpha_id} (family={family}, score={score})? [y/N]: "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def render_output(payload: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    try:
        payload = args.func(args)
        render_output(payload, pretty=args.pretty)
        return 0
    except BrainApiError as exc:
        message = str(exc)
        if exc.status is not None:
            message = f"{message} [status={exc.status}]"
        if exc.url:
            message = f"{message} [url={exc.url}]"
        print(message, file=sys.stderr)
        if exc.payload is not None:
            print(json.dumps(exc.payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    except Exception as exc:
        import traceback

        # A bare message ("The read operation timed out") hid where a run died.
        traceback.print_exc(file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
