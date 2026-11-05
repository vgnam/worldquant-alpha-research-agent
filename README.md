# WorldQuant Alpha Research Agent

Planner-driven, tool-using alpha research system for WorldQuant BRAIN with:

- stage-aware candidate exploration, exploitation, robustness testing, and harvesting
- controlled simulation/check execution via existing API client
- governance-first submission modes (`disabled`, `manual`, `auto_approved`)
- hypothesis logging and failure-pattern-aware mutation logic
- correlation-aware submission readiness and decorrelation repair candidates
- reproducible JSON reports and baseline-vs-agent evaluation harness
- presentation-ready Streamlit console with a built-in demo case, live controls, visual analytics, and economic logic

Repository: [https://github.com/zeron-G/worldquant-alpha-research-agent](https://github.com/zeron-G/worldquant-alpha-research-agent)

## Project Structure

- `worldquant_brain_cli.py`  
  Low-level API client and CLI (auth, simulate, check, submit, metadata fetch).
- `alpha_research_pipeline.py`  
  Baseline heuristic search pipeline (seed + refine + score + optional submit).
- `alpha_research_agent.py`  
  New planner-driven agent CLI entrypoint.
- `alpha101_ideas.json`  
  WorldQuant BRAIN-compatible transliteration of all 101 formulaic alpha seeds for broad replication/search runs.
- `alpha_agent/`  
  Agent runtime modules:
  - `config.py`: shared runtime dataclasses
  - `planner.py`: heuristic planner + OpenAI JSON planner
  - `research_logic.py`: quant-style stage logic, novelty scoring, check-aware and robustness candidate builders
  - `engine.py`: orchestrator loop, tool execution, submission gating, run reports
  - `evaluation.py`: baseline-vs-agent case-suite runner
- `streamlit_app.py`  
  Presentation console for demo mode, live agent runs, parameter tuning, visual diagnostics, and economic analysis.
- `docs/eval_cases.json`  
  Starter replay case suite.

## Features

### 1) End-to-end Alpha Agent Loop

The agent repeatedly executes:

1. gather frontier context (family performance, failed-check histogram, stage, hypotheses)
2. choose next action (`evaluate_seed`, `evaluate_refine`, `evaluate_diversify`, `evaluate_robustness`, `submit_best`, `stop`)
3. call simulation/check tools
4. update leaderboard, stage, and research notebook
5. log rationale, hypothesis, risk note, and outcomes

All run events are recorded under `<workdir>/agent_runs/*.json`.

### 2) Quant-Style Stage Policy

The agent follows four research stages:

- `explore`: maximize family and expression diversity under budget constraints
- `exploit`: target dominant failure checks with check-aware refinements
- `robustness`: stress-test top candidates across universe/neutralization/truncation perturbations
- `harvest`: attempt controlled submission only when readiness and governance align

Transitions are data-driven by score quality, submit-readiness, and robustness evidence.

### 3) Prompt Contract (OpenAI Planner)

When `--planner-provider openai` is enabled, the planner receives a structured context and must return strict JSON:

```json
{
  "action": "evaluate_refine",
  "batch_size": 3,
  "rationale": "...",
  "hypothesis": "...",
  "focus_family": "news_attention",
  "risk_note": "...",
  "target_alpha_id": null
}
```

This enforces reproducible, auditable planner decisions instead of free-form text.

### 4) Submission Governance

`submission_mode` controls risk:

- `disabled`: never submit
- `manual`: submit only with explicit approval callback
- `auto_approved`: allow unattended submit action when planner selects it

By default, runs are safe (`disabled`).

Submit readiness requires both quality checks and correlation checks to pass. A candidate that clears Sharpe/fitness/turnover but fails `SELF_CORRELATION` or `PROD_CORRELATION` is treated as a repair target, not as submit-ready. The agent also requires harvest stage before any agent-driven submission, so robustness evidence is collected first.

### 5) Pluggable Planning Backends

- `heuristic`: deterministic planner (no model key needed)
- `openai`: OpenAI-compatible JSON planner via `chat/completions`

If OpenAI planner fails or key is missing, behavior falls back safely to heuristic planning.

### 6) Evaluation Harness

Run baseline pipeline and agent on the same case suite and budgets:

- per-case score and latency deltas
- aggregate win rate and average deltas
- JSON report artifact for appendix/demo

## Requirements

- Python 3.10+
- WorldQuant BRAIN account access
- Optional: OpenAI-compatible API key (only if using `--planner-provider openai`)
- Optional: Streamlit for web UI

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Copy from `.env.example` and set locally (never commit secrets):

```powershell
$env:WQB_EMAIL="your_email@example.com"
$env:WQB_PASSWORD="your_password"
# or
$env:WQB_COOKIE_HEADER="t=...; csrftoken=..."

$env:ALPHA_AGENT_PLANNER_PROVIDER="heuristic"
$env:ALPHA_AGENT_PLANNER_MODEL="gpt5.5"
$env:ALPHA_AGENT_PLANNER_BASE_URL="https://api.openai.com/v1"
$env:ALPHA_AGENT_PLANNER_API_KEY_ENV="OPENAI_API_KEY"
$env:OPENAI_API_KEY="..."
```

The CLI and Streamlit app load `.env` automatically on startup. Sidebar fields are prefilled from `.env` when values are present; otherwise they start from safe defaults or blank credential fields.

### Cookie header notes

Password-only login is frequently blocked by the platform's reCAPTCHA/persona challenge: `POST /authentication` then answers `401` with `WWW-Authenticate: persona` and a body like `{"inquiry": "..."}`. When that happens, use `WQB_COOKIE_HEADER` instead.

Copy the **entire** `Cookie` request header from a logged-in browser session (DevTools -> Network -> any request to `api.worldquantbrain.com` -> Request Headers -> `Cookie`). Do not trim it down to a single cookie name.

Observed naming detail: the session cookie on the current platform is `t`, not `sessionid`. Analytics cookies (`_ga`, `_fbp`, `cookieyes-consent`, ...) can be present in the same string and are harmless to include. The client sends the string verbatim, so dropping `t` breaks authentication.

## Quick Start

### Run the agent from CLI

```powershell
python .\alpha_research_agent.py --pretty run --budget 16 --max-iterations 10
```

Quant-style tuning example:

```powershell
python .\alpha_research_agent.py --pretty run --budget 20 --refine-top-k 10 --robustness-top-k 4 --robustness-score-threshold 550 --max-family-budget-share 0.4 --min-expression-novelty 0.12
```

Focus on a family:

```powershell
python .\alpha_research_agent.py --pretty run --family social_buzz --budget 12
```

Start from the Alpha101 replication library:

```powershell
python .\alpha_research_agent.py --pretty --idea-library .\alpha101_ideas.json run --family alpha101 --budget 24 --max-iterations 8
```

Use OpenAI planner:

```powershell
python .\alpha_research_agent.py --pretty --planner-provider openai run --budget 16
```

Manual submit mode (requires terminal approval):

```powershell
python .\alpha_research_agent.py --pretty run --submission-mode manual --interactive-approval
```

### Getting an alpha submitted

The funnel measured on this repo's own runs is the reason the sections below
exist: 284 successful simulations on `USA/TOP3000 delay-1` produced **8** alphas
that cleared the quality gates (`LOW_SHARPE ≥ 1.58`, `LOW_FITNESS ≥ 1.0`), those
8 were **one idea wrapped eight ways**, all 8 failed `SELF_CORRELATION`
(0.743-0.768 against a 0.70 limit) against a 1100-alpha account pool, and the
run submitted **nothing** — the hybrid loop only submits at `harvest`, which its
stage machine can never reach.

- **Arena (`--region --universe --delay`).** The quality gates and the
  correlation pool are both defined per arena, so this is the single biggest
  lever on whether anything is submittable; the limits come back in every check
  payload, so the run measures them instead of assuming. Every seed is forced
  into the configured arena, and the deterministic/LLM proposers cannot move a
  candidate out of it.
- **Pool-aware seeding (`--pool-aware-seeds`, on by default).** Seeds are
  ordered by how little of their data the account pool already owns
  (`alpha_agent.pool_fields`): `SELF_CORRELATION` is computed on the PnL, so
  expression similarity does not predict it (r=-0.14), but sharing a data field
  almost guarantees it. On the real pool the seeds proposed for the first 24
  simulations went from a mean collision risk of 0.520 to 0.358 for the same
  queue.
- **Field swaps of the curated seeds (`seed_field_swaps`, default 1).** The
  curated manual seeds are the ones most likely to collide. For each, the field
  the pool is most saturated in is replaced by a *proven but rare* field of the
  same data group (one the pool uses in 1-25% of its alphas), keeping the
  structure that produced the Sharpe. Proposal only: the offline filters and
  the simulator still decide, and a swap that reads well can still be a formula
  the field does not belong in — check `seed_collision_scan` in the log.
- **Submitting (`--submission-mode auto_approved --max-submissions 1`).**
  Submission no longer waits for the legacy `harvest` label: what BRAIN
  requires is that every blocking and correlation check passes, and that is what
  `record_is_submit_ready` enforces. One submission per run by default, because
  BRAIN submissions are final. `--allow-pending-checks` lets an alpha through
  while `PROD_CORRELATION` is still pending; it is off by default.

### Hybrid search: multi-objective MCTS refinement

`--search-mode hybrid` swaps the stage machine for a tree search, while
reusing the same toolbox, gates, artifacts and submission governance:

- refinement (MCTS, after *Navigating the Alpha Jungle*, arXiv:2505.11122):
  the descent picks a formula node, the weakest gate dimension is targeted, an
  LLM (or deterministic mutator) proposes refinements, and each surviving
  candidate costs exactly one simulation. There is no rollout: the backtest is
  the node value.
- multi-objective value (MO-MCTS, Wang & Sebag 2012; Perez et al. 2013): every
  node keeps the epsilon-Pareto front of the objective vectors evaluated in its
  subtree (default: raw Sharpe, turnover (lower is better), margin and
  expression-space divergence; no gate limit enters the objectives), and
  backprop merges each new point into the fronts on its path. UCB is replaced
  by an optimistic hypervolume score: a subtree's front is shifted toward
  better by `uct_c * sqrt(ln(N_parent + 1) / (attempts + 1))` on every
  objective, and the child whose shifted front covers the most volume **over
  one reference set shared by every child at that step** is descended into.
  Sharing the reference is what makes the scores comparable: scoring each child
  against the front minus only its own points let the branch that owned most of
  the front win every descent (measured: 20/20 sequences on a saved run), so
  the other branches were seeded, evaluated once and abandoned. Optimism now
  decays with `attempts` (times the descent handed the branch work), not with
  `visits` (evaluated descendants), so a branch whose proposals are all
  rejected offline still loses its optimism. A branch that owns a unique
  trade-off survives even when another has the best scalar reward.
  This replaces the former QD archive: the global front is the elite set fed to
  crossover (e2) and reported at the end of a run.
- the multi-objective value can switch itself off, loudly. If fewer than two
  evaluated nodes have a complete objective vector (a resumed tree, a record
  without a metric, or a value outside its scaling range) every hypervolume
  score is zero and the descent would quietly become "least-tried branch
  first". The run then emits a `pareto_blind` event, records the reason in
  `summary.pareto_disabled_reason` and `search.pareto.disabled_reason`, and
  continues with `value=scalar`.

The default objective set is deliberately not collinear. Measured over 284
live results: sharpe/fitness correlate at r=0.94 and sharpe/returns at 0.94
(both opt-in), while sharpe/margin is r=0.37, which is why `margin` is the
default third axis. Scaling ranges are the *possible* range of each metric, not
a quality bar (`turnover` runs 0..3, so the 5.3% of live alphas with turnover
above 1.0 stay on the front instead of being clamped onto the reference box),
and a value outside its range is rejected and counted
(`stats.objective_out_of_range`) instead of silently clamped.

Every free filter runs before a simulation is spent: FASTEXPR parse, operator
and data-field whitelist, behavioural duplicate detection (this is what kills
`rank`/`group_rank` wrapper stacking), frequent-subtree avoidance and
time-series self-nesting.

```powershell
# global flags must come before the subcommand
python .\alpha_research_agent.py --planner-provider openai --planner-model glm-5.3 --planner-base-url https://api.mintrouter.ai/v1 run --search-mode hybrid --budget 24
```

```powershell
# deterministic proposer only, no LLM calls
python .\alpha_research_agent.py run --search-mode hybrid --budget 12 --no-llm-refine
```

Hybrid flags (all optional, defaults keep legacy behaviour when
`--search-mode legacy`):

| Flag | Default | Meaning |
|---|---|---|
| `--search-mode` | `legacy` | `legacy` = stage machine, `hybrid` = multi-objective MCTS, `mo-hifo` = population search (below) |
| `--region` | `USA` | Arena region every seed is simulated in |
| `--universe` | `TOP3000` | Arena universe |
| `--delay` | `1` | Arena delay |
| `--pool-aware-seeds` | on | Order seeds by how little the account pool covers their fields |
| `--submission-mode` | `disabled` | `disabled` / `manual` / `auto_approved` |
| `--max-submissions` | `1` | Cap on automatic submissions per run |
| `--allow-pending-checks` | off | Submit while `PROD_CORRELATION` is still pending |
| `--tree-batch-size` | `2` | simulations per iteration |
| `--uct-c` | `0.1` | UCT exploration constant on a [0, 1] exploitation term |
| `--widen-alpha` | `0.5` | Progressive widening exponent: widen while `floor(work**alpha) > children` |
| `--widen-patience` | `3` | Inner nodes widen only after this many evaluations added nothing to their front; `0` restores the ungated rule |
| `--tree-value` | `ohvi` | `ohvi` = optimistic hypervolume over a shared reference, `scalar` = UCT on max gate-margin reward |
| `--mo-objectives` | `sharpe,turnover,margin,divergence` | Pareto objectives; `fitness` and `returns` (r=0.94 with sharpe) plus the gate margins `quality`/`correlation` are opt-in |
| `--mo-front-size` | `24` | max points per front (smallest hypervolume contributor dropped) |
| `--mo-epsilon` | `0.01` | near-duplicate tolerance on normalised objectives |
| `--refine-proposals` | `6` | candidates requested per expanded node before filtering |
| `--no-llm-refine` | off | force the deterministic proposer |
| `--max-llm-calls` | `64` | hard cap on refinement LLM calls |
| `--novelty-refresh` | off | refetch the account alpha pool cache |

Artifacts added by hybrid runs (workdir):

- `search_tree.json`: nodes (`visits`, `attempts`), rewards, gate margins,
  objective values, and the run settings they were scored with (`objectives`,
  `front_size`, `epsilon`, `widen_patience`), so a resumed tree is not
  reinterpreted under a different objective set without a trace; fronts and
  stall counters are rebuilt from the node values on load
- `alpha_pool.json`: cached account alphas used for the divergence axis
- `operators.json`: cached operator whitelist for offline validation
- `agent_runs/*.json` gains a `search` block with tree stats, the global
  Pareto front with its hypervolume, and per-reason filter counts

Two measured caveats worth knowing before reading hybrid reports:

- **Expression similarity does not predict `SELF_CORRELATION`.** Against 15
  evaluated alphas the rank correlation was -0.14 (full pool) and -0.19
  (submitted-only pool). One alpha was textually identical to a submitted alpha
  yet passed at 0.6206, while a structurally distant one failed at 0.7715. The
  divergence axis is therefore reported as *diversity*, never as a correlation
  forecast; the only trustworthy correlation number is the platform check.
- **`correlation` is a weak objective early on.** Until the platform reports a
  correlation check it counts as margin 0.0 (exactly at the limit), so it only
  separates candidates that already cleared the quality gates; it is opt-in
  via `--mo-objectives`.
- **Correlation is only computed for alphas that clear the quality gates.** In
  a live smoke run, three refinements with Sharpe 0.60-0.92 sat at
  `SELF_CORRELATION = PENDING` indefinitely, so no signal about the binding
  constraint was ever returned. This makes the objective effectively
  lexicographic: a refinement that damages quality cannot teach you anything
  about correlation, which is why `gate_fraction` dominates the hybrid reward.

### Population search: `--search-mode mo-hifo`

A population of alphas instead of a tree (`alpha_agent/mo_hifo.py`):

- **Objectives:** maximise Sharpe and returns, minimise `max(turnover, 0.125)`.
  The floor is the one in BRAIN's fitness, so turnover below 12.5% earns
  nothing and a dominance step never lowers fitness. The LOW/HIGH_TURNOVER
  band (1%..70% unless the check payload says otherwise) is a constraint:
  feasible beats infeasible. A SELF/PROD_CORRELATION FAIL the platform
  reported is a constraint too (violation = value - 0.7), even for alphas
  still below the quality gates; PENDING is not penalised, since formula
  similarity does not predict correlation. Of 58 quality-ready alphas in one
  workdir, 42 failed PROD_CORRELATION (0.66..0.89), so breeding from a blocked
  parent mostly reproduces the block.
- **Selection (MEoH, AAAI 2025):** dominance-dissimilarity score
  `v_j = -sum_{i dominates j} Sim_AST(i, j)`; parents by `softmax(v)`,
  survival keeps the best `--population-size` by `v`.
- **Prompt guidance (HiFo-Prompt, ICLR 2026):** an insight pool with EMA credit
  (here Pareto-tiered: dominates members 0.8..1, joins the front 0.2..0.8,
  dominated -0.3..0.2, rejected -0.5) and the mean Sharpe/returns/turnover
  change each insight produced; one "avoid" insight per prompt, distilled from
  alphas blocked by SELF/PROD correlation (at or below the quality gates). A navigator picks exploration /
  exploitation / balanced from hypervolume stagnation and formula diversity,
  plus `decorrelate` after a correlation block; the regime sets the operator
  mix (`e1 e2 m1 m2 m3`) and the directive in the prompt.
- Initialisation is one `i1` alpha from scratch, then `e1` alphas until the
  population is full. State lives in `mo_hifo_population.json`, so `continue`
  resumes it; offspring simulated after the last save are recovered from
  `results.jsonl`.

```powershell
python .\alpha_research_agent.py --planner-provider openai run --search-mode mo-hifo --budget 200 --population-size 10 --population-state-path .alpha_agent/mo_hifo_eur.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--population-size` | `10` | MEoH N: survivors and offspring per generation |
| `--crossover-parents` | `5` | parents shown to `e1`/`e2` |
| `--insights-per-prompt` | `3` | insights retrieved per prompt |
| `--no-insight-extraction` | off | no extra model call per generation |
| `--population-state-path` | `<workdir>/mo_hifo_population.json` | a new path starts a fresh population |

### Show current leaderboard

```powershell
python .\alpha_research_agent.py --pretty leaderboard --limit 10
```

### Run baseline-vs-agent evaluation

```powershell
python .\alpha_research_agent.py --pretty evaluate --cases .\docs\eval_cases.json
```

Output defaults to:

- `<workdir>/evaluation/report.json`

### Run the Presentation Frontend

For a classroom or pre demo, start with the frontend. It opens with a built-in showcase case, so it works even without a live WorldQuant session:

```powershell
python -m streamlit run .\streamlit_app.py
```

Then open the local URL printed by Streamlit, usually:

- `http://localhost:8501`

Recommended presentation flow:

1. Open the app and keep the built-in showcase loaded.
2. Walk through `Overview` for the executive result, readiness funnel, stage machine, and convergence curve.
3. Use `Agent Trace` to show planner actions, hypotheses, failure pressure, family frontier, and candidate details.
4. Use `Economic Logic` to explain the constrained optimization view and adjust cost/value assumptions live.
5. Use `Architecture` to explain design choices and show every hot-modifiable run parameter.
6. Optionally use `Run Live Agent` from the sidebar only after configuring credentials or a cookie header.

The presentation console provides:

- built-in showcase case for fast, reliable presentation
- idea library presets for the core search library and the Alpha101 replication library
- full live controls for auth, LLM planner, budget, family filters, novelty, robustness, retries, polling, workdir, and submission governance
- visual analytics for best score, quality-ready count, correlation-blocked candidates, submit-ready candidates, stage transitions, failure pressure, and family frontier
- candidate inspector with expression, settings, metrics, and check summary
- economic derivation panel showing alpha discovery as constrained expected utility maximization
- raw JSON artifact viewer and downloadable report

You can also load a saved run report from the sidebar with either:

- `Load Latest Workdir Report`
- `Run report JSON path`

More presentation notes are in [`docs/PRESENTATION_FRONTEND.md`](docs/PRESENTATION_FRONTEND.md).
Alpha101 replication notes are in [`docs/ALPHA101.md`](docs/ALPHA101.md).

## Basic Tests

```powershell
python -m unittest tests\test_planner.py tests\test_research_logic.py tests\test_progress.py tests\test_alpha101_library.py
```

## Reproducibility Artifacts

In agent workdir (`.alpha_agent` by default):

- `results.jsonl`: evaluated candidate records
- `submissions.jsonl`: submit attempts
- `state.json`: rolling summary
- `agent_runs/*.json`: full per-run reports with planner decisions and event logs
- `evaluation/report.json`: baseline-vs-agent comparison report

## Legacy Baseline Commands

Baseline scripts remain available:

```powershell
python .\alpha_research_pipeline.py --pretty search --budget 24 --seed-fraction 0.7
python .\alpha_research_pipeline.py --pretty leaderboard --limit 10
python .\alpha_research_pipeline.py --pretty submit-best
```

## Safety Notes

- Never commit credentials, cookies, keys, or private account data.
- Use `.env` and ignored local files for secrets.
- Keep `submission_mode=disabled` for research unless you intentionally enable stronger modes.
- Prefer manual approval for live demos and classroom evaluation.

## Disclaimer

Use responsibly and only with authorized credentials and permissions. Platform behavior and available endpoints may change over time.

## License

[MIT](LICENSE)
