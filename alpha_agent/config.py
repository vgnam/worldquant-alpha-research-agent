from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from alpha_research_pipeline import DEFAULT_FIELDS_SUMMARY, DEFAULT_IDEA_LIBRARY
from worldquant_brain_cli import API_BASE, DEFAULT_MAX_WAIT, DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT


@dataclass(frozen=True)
class AuthConfig:
    email: Optional[str]
    password: Optional[str]
    cookie_header: Optional[str]
    base_url: str = API_BASE
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "heuristic"
    model: str = "gpt5.5"
    temperature: float = 0.1
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 60.0
    #: Sent as ``reasoning_effort`` (low | medium | high) when set.
    reasoning_effort: Optional[str] = None


@dataclass(frozen=True)
class AgentConfig:
    budget: int = 24
    max_iterations: int = 12
    seed_fraction: float = 0.7
    refine_top_k: int = 8
    robustness_top_k: int = 3
    robustness_score_threshold: float = 500.0
    family_filter: tuple[str, ...] = ()
    shuffle_seeds: bool = True
    random_seed: int = 7
    max_family_budget_share: float = 0.45
    min_expression_novelty: float = 0.10
    retries: int = 2
    sleep_between: float = 1.0
    max_wait: float = DEFAULT_MAX_WAIT
    poll_interval: float = DEFAULT_POLL_INTERVAL
    allow_pending_checks: bool = False
    submission_mode: str = "disabled"  # disabled | manual | auto_approved
    #: Cap on automatic submissions per run. One by default: BRAIN
    #: submissions are final, so a run that finds five ready alphas should not
    #: spend all five slots without a human looking first.
    max_submissions_per_run: int = 1
    workdir: Path = Path(".alpha_agent")
    idea_library: Path = DEFAULT_IDEA_LIBRARY
    fields_summary: Path = DEFAULT_FIELDS_SUMMARY
    #: Arena every seed is simulated in. This is the single biggest lever on
    #: whether an alpha can be submitted at all: the quality gates and the
    #: correlation pool are both defined per region/universe/delay, and the
    #: limits come back in each check payload, so the run measures them.
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    #: Order the seed queue by how little of each seed's data the account pool
    #: already covers (alpha_agent.pool_fields). On 284 live simulations the
    #: seeds whose fields the pool was saturated in produced the only
    #: quality-passing alphas, and every one of them failed SELF_CORRELATION.
    pool_aware_seeds: bool = True
    #: Variants generated per curated seed by swapping the field the pool is
    #: saturated in for a proven-but-rare one of the same data group. Keep it
    #: small: the swap keeps the formula structure, but only a simulator can
    #: say whether the new field makes sense in that position.
    seed_field_swaps: int = 1

    # -- hybrid search (opt-in; legacy behaviour stays the default) ----------
    search_mode: str = "legacy"  # legacy | hybrid | mo_hifo
    tree_batch_size: int = 2
    #: MCTS-AHD lambda_0: exploration weight on a [0, 1]-normalised Q, scaled
    #: down linearly with the remaining budget.
    uct_c: float = 0.1
    #: Progressive widening exponent: a node is widened while
    #: floor(work ** widen_alpha) exceeds its child count (MCTS-AHD alpha).
    widen_alpha: float = 0.5
    #: Inner nodes may only widen after this many evaluations in their subtree
    #: added nothing to their front. 0 disables the gate (old behaviour).
    widen_patience: int = 3
    #: Child selection: "ohvi" = optimistic hypervolume of each subtree's
    #: Pareto front over one shared reference (MO-MCTS); "scalar" = UCT on the
    #: max gate-margin reward (Alpha Jungle).
    tree_value: str = "ohvi"
    #: Objectives kept on the Pareto fronts (see alpha_agent.pareto): raw
    #: sharpe, turnover (lower is better), margin and divergence. ``fitness``
    #: (r=0.94 with sharpe) and ``returns`` (r=0.94) are opt-in; the gate
    #: margins ``quality``/``correlation`` are opt-in too.
    mo_objectives: tuple[str, ...] = ("sharpe", "turnover", "margin", "divergence")
    #: Max points per front; beyond it the smallest hypervolume contributor goes.
    mo_front_size: int = 24
    #: Near-duplicate tolerance on the [0, 1]-normalised objectives.
    mo_epsilon: float = 0.01
    #: Children simulated per operator call (MCTS-AHD: one offspring per call).
    children_per_expansion: int = 1
    #: Cap on subtree roots; 0 means unlimited.
    max_roots: int = 0
    #: When > 0, an empty tree is seeded with this many structurally diverse
    #: seeds from the idea library instead of ``tree_batch_size`` in order.
    initial_seed_count: int = 0
    #: Where subtree roots come from. ``library``: idea-library seeds (and
    #: e1 ideas when widening). ``llm``: MCTS-AHD's own scheme, no seed
    #: formulas at all: an empty tree gets one ``i1`` root written from the task
    #: description alone, then ``e1`` roots that differ from the ones so far,
    #: ``initial_seed_count`` in total (paper: N_I = 4); the virtual root is
    #: widened with e1 only.
    root_source: str = "library"
    tree_max_nodes: int = 512
    tree_max_children: int = 4
    tree_max_depth: int = 5
    tree_state_path: Optional[Path] = None
    novelty_pool_path: Optional[Path] = None
    novelty_refresh: bool = False
    novelty_pool_limit: int = 2000
    refine_proposals: int = 6
    refine_top_k: int = 2
    frequent_subtree_topk: int = 10
    frequent_subtree_penalty: float = 0.4
    #: A subtree must appear in at least this many evaluated candidates before
    #: it is banned. 1 would ban the parent's own motifs and block every
    #: mutation of it (observed: 36/36 proposals rejected in a live smoke run).
    subtree_min_support: int = 3
    subtree_min_size: int = 3
    llm_refine_enabled: bool = True
    #: MCTS-AHD expands a leaf with four operators, so one expansion costs up to
    #: four model calls. 64 was enough when a leaf took one or two operators; it
    #: ran out after ~16 iterations and the run quietly continued on
    #: deterministic mutations.
    max_llm_calls: int = 200
    #: Stop the hybrid run instead of falling back to deterministic mutations
    #: once the LLM proposer fails ``llm_failure_limit`` times in a row.
    require_llm: bool = False
    llm_failure_limit: int = 3
    #: Ping an authenticated endpoint at the start of each iteration so an
    #: expired session stops the run instead of burning the whole budget.
    session_ping: bool = True
    #: When false, the hybrid search never touches the network for the operator
    #: catalog or the alpha pool; it uses whatever cache exists (used by tests).
    allow_offline_fetch: bool = True

    # -- mo_hifo population search (alpha_agent.mo_hifo) ---------------------
    #: Population file (default <workdir>/mo_hifo_population.json); a new path
    #: starts a fresh population, an existing one resumes it.
    mo_hifo_state_path: Optional[Path] = None
    #: MEoH's N: alphas kept by survival, and offspring per generation.
    population_size: int = 10
    #: MEoH's d: parents shown to the crossover operators e1 / e2.
    crossover_parents: int = 5
    #: HiFo's s: insights retrieved per prompt.
    insights_per_prompt: int = 3
    #: One model call per generation distils new insights from the front.
    extract_insights: bool = True


@dataclass(frozen=True)
class AgentRuntimeConfig:
    auth: AuthConfig
    model: ModelConfig
    agent: AgentConfig
