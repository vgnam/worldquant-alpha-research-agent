"""Refinement proposal generators for the hybrid search.

Two implementations behind one shape:

* :class:`LLMRefinementProposer` asks an OpenAI-compatible model for targeted
  formula edits, in the spirit of Alpha Jungle's "dimension-targeted refinement
  suggestion". One call per expanded node, no simulation.
* :class:`DeterministicRefinementProposer` is the fail-safe: no API key, no
  network, or malformed JSON all fall back to structural mutations.

Both are gated by the same offline filters before anything reaches the
evaluator:

1. expression must parse,
2. every operator / data field must be in the live catalog,
3. no duplicate behavioural signature (this is what kills wrapper stacking),
4. no blacklisted frequent subtree,
5. no *soft* time-series self-nesting left over from the legacy generator.

The deterministic proposer deliberately mutates **structure** (window, data
field, two-leg weights) rather than wrapping the whole expression: the live run
proved that `rank`/`group_rank`/`group_neutralize` wrappers around an already
wrapped expression produce identical returns and identical correlation, so they
cost a simulation and buy nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import alpha_research_pipeline as pipeline
from alpha_agent.catalog import Catalog
from alpha_agent.config import ModelConfig
from alpha_agent.llm_http import LLMRequestError, post_chat_completion, warn
from alpha_agent.expr_ast import (
    ExprNode,
    behavioral_signature,
    canonical_form,
    collect_call_operators,
    collect_identifiers,
    soft_nesting_count,
    structural_distance,
    subtree_hashes,
    try_parse,
)

__all__ = [
    "ProposalExpression",
    "RefinementProposal",
    "FilterOutcome",
    "DeterministicRefinementProposer",
    "LLMRefinementProposer",
    "filter_candidates",
]

#: Identifier swaps that change the economic signal rather than its packaging.
FIELD_SWAPS = {
    "close": ("vwap", "open"),
    "vwap": ("close",),
    "open": ("close",),
    "high": ("close",),
    "low": ("close",),
    "returns": ("close",),
    "volume": ("adv20",),
    "adv20": ("volume",),
}

MIN_WINDOW = 2
MAX_WINDOW = 252


@dataclass(frozen=True)
class ProposalExpression:
    expression: str
    settings_patch: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class RefinementProposal:
    target_dimension: str
    suggestion: str
    rationale: str
    expressions: tuple[ProposalExpression, ...] = ()
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterOutcome:
    """What survived the offline filters, plus why the rest did not."""

    accepted: List[pipeline.Candidate] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str, amount: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + amount

    @property
    def proposed(self) -> int:
        return sum(self.counts.values()) + len(self.accepted)


# ---------------------------------------------------------------------------
# Offline filtering
# ---------------------------------------------------------------------------


def _normalize_expression_text(expression: str) -> Optional[ExprNode]:
    return try_parse(expression)


def filter_candidates(
    proposals: Sequence[ProposalExpression],
    *,
    settings: Dict[str, Any],
    catalog: Catalog,
    see_signatures: Iterable[str] = (),
    blacklist_subtrees: Iterable[str] = (),
    max_subtree_hits: int = 2,
    subtree_min_size: int = 3,
    parent_expression: Optional[str] = None,
    allow_soft_nesting: bool = False,
    limit: int = 8,
    prefer_minimal_edits: bool = True,
) -> FilterOutcome:
    """Apply every free check, preserving accounting for reporting.

    ``prefer_minimal_edits=False`` keeps the proposer's order. The operator
    schedule needs that: sorting by edit distance turned every e1/m1 call back
    into "take the smallest tweak".
    """
    outcome = FilterOutcome()
    seen = set(see_signatures)
    blacklist = set(blacklist_subtrees)

    # A refinement inherits most of its parent's subtrees by construction, so
    # those must be exempt or every mutation of a frequently seen formula would
    # be rejected for "repeating a frequent subtree".
    parent_subtrees: set[str] = set()
    if parent_expression:
        parent_node = try_parse(parent_expression)
        if parent_node is not None:
            parent_subtrees = subtree_hashes(parent_node, min_size=subtree_min_size)
    effective_blacklist = blacklist - parent_subtrees

    for proposal in proposals:
        expression = (proposal.expression or "").strip()
        if not expression:
            outcome.reject("empty")
            continue

        node = _normalize_expression_text(expression)
        if node is None:
            outcome.reject("parse")
            continue

        ok, errors = catalog.validate(node)
        if not ok:
            outcome.reject("invalid_operator_or_field", len(errors))
            continue

        # A VECTOR (event) field handed to an ordinary operator is rejected by
        # the platform only at simulation time ("Operator ts_decay_linear does
        # not support event inputs"), which threw away 60% of one run's budget.
        type_problems = catalog.type_violations(node)
        if type_problems:
            outcome.reject("vector_field_misuse", len(type_problems))
            continue

        if not allow_soft_nesting and soft_nesting_count(node) > 0:
            outcome.reject("soft_nesting")
            continue

        if effective_blacklist:
            hits = len(subtree_hashes(node, min_size=subtree_min_size) & effective_blacklist)
            if hits >= max_subtree_hits:
                outcome.reject("frequent_subtree")
                continue

        merged_settings = dict(settings)
        merged_settings.update(proposal.settings_patch or {})
        merged_settings = pipeline.normalize_settings(merged_settings)

        try:
            signature = behavioral_signature(expression, merged_settings)
        except Exception:
            signature = None
        if signature is None:
            outcome.reject("parse")
            continue
        if signature in seen:
            outcome.reject("duplicate")
            continue
        seen.add(signature)

        distance = None
        if parent_expression:
            parent_node = try_parse(parent_expression)
            if parent_node is not None:
                distance = structural_distance(parent_node, node)

        outcome.accepted.append(
            pipeline.Candidate(
                expression=canonical_form(node),
                settings=merged_settings,
                family="",  # filled by the caller from the parent node
                idea_name=f"hybrid.{signature[:8]}",
                stage="hybrid_refine",
                priority=0.0,
                parent_key=None,
                metadata={
                    "hypothesis": proposal.hypothesis,
                    "rationale": proposal.rationale,
                    "behavioral_signature": signature,
                    "edit_distance": distance,
                },
            )
        )
        if len(outcome.accepted) >= limit:
            break

    # Prefer minimal edits: they preserve the parent's quality profile, and the
    # platform only reports correlation for alphas that clear the quality gates.
    if prefer_minimal_edits:
        outcome.accepted.sort(
            key=lambda candidate: (
                candidate.metadata.get("edit_distance")
                if isinstance(candidate.metadata.get("edit_distance"), int)
                else 10**6,
                candidate.idea_name,
            )
        )
    return outcome


# ---------------------------------------------------------------------------
# Deterministic proposer
# ---------------------------------------------------------------------------


def _map_windows(node: ExprNode, transform) -> ExprNode:
    if node.is_leaf:
        return node
    args = list(node.args)
    if (node.op or "").startswith("ts_") and len(args) >= 2:
        window = args[-1]
        if window.is_number:
            try:
                new_window = transform(int(float(window.leaf or "")))
            except ValueError:
                new_window = None
            if new_window is not None and new_window != int(float(window.leaf or "")):
                args[-1] = ExprNode(leaf=str(new_window))
    mapped = tuple(_map_windows(arg, transform) for arg in args)
    return ExprNode(op=node.op, args=mapped)


def _replace_identifier(node: ExprNode, mapping: Dict[str, str]) -> ExprNode:
    if node.is_leaf:
        if node.leaf in mapping:
            return ExprNode(leaf=mapping[node.leaf])
        return node
    args = tuple(_replace_identifier(arg, mapping) for arg in node.args)
    return ExprNode(op=node.op, args=args)


class DeterministicRefinementProposer:
    """Structural mutations that do not need a model.

    Ordering is deliberate: structural edits first, settings second. The
    observed failure was that every budget went into settings/wrapper edits that
    could not move `SELF_CORRELATION`.
    """

    def __init__(self, *, max_windows: int = 2, max_proposals: int = 8) -> None:
        self.max_windows = max(1, max_windows)
        self.max_proposals = max(1, max_proposals)

    def propose(
        self,
        *,
        node: Any,
        e_vector: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        blacklist: Sequence[str] = (),
        operators: Optional[Iterable[str]] = None,
        fields: Optional[Iterable[str]] = None,
        k: int = 2,
        operator: Optional[str] = None,
    ) -> List[ProposalExpression]:
        # Structural mutations only; ``operator`` is accepted for interface parity.
        if node is None:
            return []
        parsed = try_parse(str(getattr(node, "expression", "") or ""))
        if parsed is None:
            return []

        proposals: List[ProposalExpression] = []
        weakest = _weakest(e_vector)

        # 1. Lookback windows on time-series operators.
        windows = sorted(
            {
                int(float(arg.leaf or ""))
                for child in _walk(parsed)
                for arg in child.args
                if arg.is_number and (child.op or "").startswith("ts_")
            }
        )[: self.max_windows]
        for window in windows:
            for factor, label in ((2, "double"), (3, "triple")):
                new_window = window * factor
                if MIN_WINDOW <= new_window <= MAX_WINDOW:
                    mutated = _map_windows(parsed, lambda value, w=window, n=new_window: n if value == w else None)
                    proposals.append(
                        ProposalExpression(
                            expression=canonical_form(mutated),
                            hypothesis=f"Longer lookback ({window} -> {new_window}) changes the horizon of the signal and should reduce overlap with existing alphas.",
                            rationale=f"{label} the {window}-day window",
                        )
                    )
            halved = max(MIN_WINDOW, window // 2)
            if halved != window:
                mutated = _map_windows(parsed, lambda value, w=window, n=halved: n if value == w else None)
                proposals.append(
                    ProposalExpression(
                        expression=canonical_form(mutated),
                        hypothesis=f"Shorter lookback ({window} -> {halved}) reacts faster and may decorrelate from slow-moving inventory.",
                        rationale=f"halve the {window}-day window",
                    )
                )

        # 2. Substituting the data field changes the economic content.
        identifiers = collect_identifiers(parsed)
        for identifier in sorted(identifiers):
            for replacement in FIELD_SWAPS.get(identifier, ()):
                if replacement in identifiers:
                    continue
                mutated = _replace_identifier(parsed, {identifier: replacement})
                proposals.append(
                    ProposalExpression(
                        expression=canonical_form(mutated),
                        hypothesis=f"Swapping {identifier} for {replacement} keeps the structure but changes the underlying signal source.",
                        rationale=f"field swap {identifier} -> {replacement}",
                    )
                )

        # 3. Settings that are known to move correlation more than quality.
        for neutralization in ("INDUSTRY", "SUBINDUSTRY"):
            proposals.append(
                ProposalExpression(
                    expression=canonical_form(parsed),
                    settings_patch={"neutralization": neutralization},
                    hypothesis=f"Finer neutralization ({neutralization}) removes common sector exposure that drives correlation.",
                    rationale=f"neutralization -> {neutralization}",
                )
            )
        for truncation in (0.05, 0.1):
            proposals.append(
                ProposalExpression(
                    expression=canonical_form(parsed),
                    settings_patch={"truncation": truncation},
                    hypothesis=f"Truncation {truncation} reshapes the weight profile and can lower correlation without hurting Sharpe.",
                    rationale=f"truncation -> {truncation}",
                )
            )
        for decay in (2, 4):
            proposals.append(
                ProposalExpression(
                    expression=canonical_form(parsed),
                    settings_patch={"decay": decay},
                    hypothesis=f"Adding decay {decay} smooths turnover and can shift the correlation profile.",
                    rationale=f"decay -> {decay}",
                )
            )

        # Target ordering: when correlation is the binding gate, structural edits
        # must be tried before settings edits.
        correlation_first = weakest in {"SELF_CORRELATION", "PROD_CORRELATION"}
        if not correlation_first:
            structural = [p for p in proposals if not p.settings_patch]
            settings_only = [p for p in proposals if p.settings_patch]
            proposals = structural + settings_only

        # Drop self-identical proposals.
        expressions = {canonical_form(parsed)}
        unique: List[ProposalExpression] = []
        for proposal in proposals:
            key = (proposal.expression, tuple(sorted((proposal.settings_patch or {}).items())))
            if proposal.expression in expressions and not proposal.settings_patch:
                continue
            if any(
                (item.expression, tuple(sorted((item.settings_patch or {}).items()))) == key
                for item in unique
            ):
                continue
            unique.append(proposal)
            if len(unique) >= self.max_proposals:
                break
        return unique[: max(1, k) if k else self.max_proposals]


def _walk(node: ExprNode):
    yield node
    for arg in node.args:
        yield from _walk(arg)


def _weakest(e_vector: Dict[str, float]) -> str:
    if not e_vector:
        return ""
    return min(e_vector.items(), key=lambda item: (item[1], item[0]))[0]


# ---------------------------------------------------------------------------
# LLM proposer
# ---------------------------------------------------------------------------

# The previous single prompt demanded "the SMALLEST edit ... at most two
# primitives changed", which is EoH's m2 (parameter tuning) and nothing else:
# every proposal was a window tweak of the parent, so every child inherited the
# parent's SELF_CORRELATION. Following EoH (Liu et al., 2024) and MCTS-AHD
# (Zheng et al., 2025), each call now runs one of several evolution operators,
# chosen by what blocks the parent, and asks for a one-sentence design idea
# ("thought") before each formula.
#
# Layout: the system prompt holds everything that is the same on every call
# (role, output contract, rules, what each input key means); the user message
# is a JSON payload whose ``task`` is the operator instruction for this call.
# Each section below is one constant, so a rule is edited in one place.

#: Simulation settings a model may patch. Anything else (observed: an invented
#: ``grouping_field``) is rejected by BRAIN with HTTP 400 and burns budget.
_PATCHABLE_SETTINGS = {
    "decay": int,
    "truncation": float,
    "neutralization": str,
    "pasteurization": str,
    "nanHandling": str,
}

_PROMPT_ROLE = (
    "You are a quantitative researcher designing formulaic alphas for WorldQuant BRAIN "
    "(daily cross-sectional equity signals; the platform ranks stocks by the formula value). "
    "An alpha is submittable only if it passes the quality gates (Sharpe, fitness, "
    "sub-universe and ladder Sharpe, turnover) AND is not too correlated with alphas the "
    "account already owns (SELF_CORRELATION) or the production pool."
)

_PROMPT_OUTPUT = (
    'Return strict JSON only, no prose around it:\n'
    '{"backbone": str, "expressions": [{"thought": str, "expression": str, "settings_patch": object}]}\n'
    "- thought: the design idea and its economic rationale in one sentence, written FIRST.\n"
    "- expression: the formula that implements exactly that thought.\n"
    "- settings_patch: {} for none.\n"
    "- backbone: the shared idea behind the elites when the task asks for it, else a "
    "one-line summary of this batch."
)

_PROMPT_HARD_RULES = (
    "Use ONLY operators from allowed_operators and data fields that appear in the inputs "
    "(parent, elites, references, path or field_menu).",
    "Keep every formula under 200 characters.",
    "Never wrap the whole formula in rank, zscore or scale, and never stack a second group "
    "normaliser on one that is already there: both leave the bet unchanged and are dropped "
    "as duplicates before simulation. A FIRST industry normalisation (group_neutralize, "
    "group_rank, group_zscore) on a formula that has none does change the bet; see the task.",
    "Never use a subtree listed in forbidden_subtrees.",
    "settings_patch may only contain " + ", ".join(_PATCHABLE_SETTINGS) + ".",
)

_PROMPT_NOVELTY_RULES = (
    "Candidates must be materially different from each other and from the parent: changing "
    "only a lookback window or swapping close/open is NOT a new idea. Normalising a leg "
    "within industry when the parent has no group normalisation IS a material change.",
    "Never resubmit a formula from failed_children_of_parent or recent_failed_attempts, or a "
    "near-copy of one.",
    "Do not retry an idea or data field that already produced low Sharpe unless the thought "
    "explains what is different this time.",
    "Steer away from the mechanisms behind recurring failures: CONCENTRATED_WEIGHT means the "
    "signal piles weight on few stocks; LOW_SUB_UNIVERSE_SHARPE means it only works in small caps.",
)

#: What each patchable setting does, so a patch is a deliberate lever rather
#: than a guess. Keys must match ``_PATCHABLE_SETTINGS``.
_PROMPT_SETTINGS = {
    "decay": "integer days of linear decay applied to the signal; higher smooths it and "
    "lowers turnover (helps HIGH_TURNOVER, can cost Sharpe on fast signals).",
    "truncation": "maximum weight of one stock, as a fraction of the book (e.g. 0.05); "
    "lower spreads weight (helps CONCENTRATED_WEIGHT).",
    "neutralization": "NONE, MARKET, SECTOR, INDUSTRY or SUBINDUSTRY: the platform removes the "
    "mean of the final weights within that group. It acts on top of any group_* operator in "
    "the formula, so check parent.settings before adding one.",
    "pasteurization": "ON or OFF: ON ignores values for stocks outside the universe.",
    "nanHandling": "ON or OFF: how operators treat missing values.",
}

#: What each key of the user payload means. Keys absent from a call are simply
#: not sent; keep this in step with ``_propose_via_model``.
_PROMPT_INPUTS = {
    "operator / task": "which evolution operator this call runs, and its instruction. Follow task.",
    "parent": "the alpha being refined, with the gate it fails worst (binding_weak_check) and its "
    "gate margins, weakest first. Absent for i1 and e1.",
    "elites": "strong alphas from the Pareto front of quality vs. divergence.",
    "references": "e1 only: one alpha from each existing subtree, to differ from.",
    "path": "s1 only: the lineage from subtree root to parent, each step with its result.",
    "failed_children_of_parent / recent_failed_attempts": "formulas already simulated, with "
    "Sharpe, fitness and failed checks, or the platform error.",
    "field_menu": "data fields not used anywhere else in the payload; draw new fields from here.",
    "allowed_operators / forbidden_subtrees": "see the hard rules.",
    "metric_directions": "which way each metric should move.",
    "arena / arena_note": "the region, universe and delay every candidate is simulated in.",
    "run_context": "iteration, remaining budget and the run's failed-check histogram.",
    "design_insights / design_directive / search_regime": "when present: principles distilled "
    "from earlier results (kind positive or avoid, with the mean change in Sharpe, returns and "
    "turnover they produced over n alphas), what to focus on now, and how bold to be. Use them "
    "as guidance; the hard rules still win.",
}


def _bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


_SYSTEM_PROMPT = "\n\n".join([
    _PROMPT_ROLE,
    "## Output\n" + _PROMPT_OUTPUT,
    "## Hard rules\n" + _bullets(_PROMPT_HARD_RULES),
    "## Novelty and learning from failures\n" + _bullets(_PROMPT_NOVELTY_RULES),
    "## Simulation settings (settings_patch)\n"
    + _bullets(f"{key}: {meaning}" for key, meaning in _PROMPT_SETTINGS.items())
    + "\nRegion, universe and delay are fixed by the arena and cannot be patched. Prefer a "
    "structural change over a settings-only change when correlation blocks the parent.",
    "## Inputs (the user message is JSON)\n"
    + _bullets(f"{key}: {meaning}" for key, meaning in _PROMPT_INPUTS.items()),
])

#: Fixed notes sent in every payload. No digits on purpose (see below).
_QUALITY_NOTE = (
    "Correlation is only measured for alphas that pass the quality gates, so every "
    "new idea still needs a plausible economic reason for high Sharpe and fitness."
)
_BUDGET_NOTE = "Each candidate costs one simulation; propose 3 to 4 candidates."
#: Universe names are BRAIN shorthand the model may not know; without this it
#: cannot tell why an alpha that lives on small caps fails the sub-universe gate.
_UNIVERSE_NOTE = (
    "Universe names: TOPN is the N most liquid stocks of the region by average daily "
    "dollar volume, re-selected periodically, so a large N reaches down into small, "
    "illiquid caps; MINVOLxM is every stock whose average daily volume is at least x "
    "million USD. LOW_SUB_UNIVERSE_SHARPE recomputes Sharpe on a smaller, more liquid "
    "sub-universe of the same region: an alpha that earns only on the smallest, least "
    "liquid names fails it, so the signal must also work on the large caps."
)


def _arena_note(region: str, universe: str, delay: Any) -> str:
    return (
        f"Every candidate is simulated in {region}/{universe} at delay {delay}. "
        "The task is to find alphas that are submittable in that arena: use fields "
        "that exist there, respect the liquidity of that universe, and treat the "
        "quality gates quoted by that arena as the bar (they differ per arena, so "
        "do not assume another region's thresholds). "
        + _UNIVERSE_NOTE
    )

#: Which way each metric should move. Deliberately no numeric targets: limits
#: are account specific, and a quoted number anchors the model on clearing it
#: rather than on improving the signal.
_METRIC_DIRECTIONS = {
    "higher_is_better": [
        "Sharpe",
        "fitness",
        "returns",
        "sub-universe Sharpe (LOW_SUB_UNIVERSE_SHARPE)",
        "recent 2-year Sharpe (LOW_2Y_SHARPE)",
        "IS ladder Sharpe (IS_LADDER_SHARPE)",
    ],
    "lower_is_better": [
        "correlation with the account's alphas (SELF_CORRELATION) and the production pool (PROD_CORRELATION)",
        "weight concentration on few stocks (CONCENTRATED_WEIGHT)",
        "drawdown",
    ],
    "keep_in_band": [
        "turnover: not near zero (LOW_TURNOVER) and not excessive (HIGH_TURNOVER); lower within that band helps fitness",
    ],
}

#: Measured on this account (SELF_CORRELATION limit 0.70), same signal three ways:
#:   ts_mean(-scl12_buzz, 8) + rank(-ts_delta(close, 5))                   0.768 FAIL
#:   group_neutralize(<that expression>, industry)                         0.525 PASS
#:   group_rank(ts_mean(-scl12_buzz, 8), industry) + rank(-ts_delta(...))   0.600 PASS
#: So an industry-level re-normalisation cleared a correlation gate that the
#: bare form failed, without changing the economic mechanism. Two of the eleven
#: alphas whose SELF_CORRELATION actually passed on this account are exactly
#: that pair, and expression distance does not predict correlation at all
#: (rank correlation -0.14 against SELF_CORRELATION), so "looks different" buys
#: nothing on its own.
_CORRELATION_RECIPE = (
    "\nIf SELF_CORRELATION blocks the parent, being different enough is mostly a matter "
    "of industry normalisation, not of a new idea. Measured on this account, one signal, "
    "ts_mean(-scl12_buzz, 8) + rank(-ts_delta(close, 5)):\n"
    "- bare: 0.768 FAIL\n"
    "- whole formula in group_neutralize(..., industry): 0.525 PASS\n"
    "- first leg in group_rank(..., industry): 0.600 PASS\n"
    "So try first: keep the mechanism that earns the Sharpe and normalise the whole formula "
    "or one leg within industry / subindustry / sector (group_neutralize, group_rank, "
    "group_zscore, group_scale), if the parent has no such normalisation yet. "
    "Next: move one leg to a different data family. "
    "Formula distance does not predict correlation, so a candidate that merely looks "
    "different is not worth a simulation."
)

#: Evolution operators after EoH (Liu et al., 2024) and MCTS-AHD (Zheng et al.,
#: 2025), rephrased for alpha formulas. Each is sent as the payload's ``task``:
#: a title line, then what to do. The paper names stay out of the text the model
#: sees. The input keys themselves are explained once, in ``_PROMPT_INPUTS``.
_OPERATOR_INSTRUCTIONS = {
    # MCTS-AHD i1 ("directly generate a heuristic ... from scratch"): the
    # repo's prompt carries only the task description and the I/O format, so
    # this payload carries no existing alpha either.
    "i1": (
        "Initialization: write a NEW alpha from scratch; there is no parent and no existing "
        "alpha to start from.\n"
        "From the task alone, pick one plausible economic mechanism that could earn a high "
        "Sharpe in this arena, choose data fields from field_menu that express it, and write "
        "the formula that implements exactly that idea."
    ),
    "e1": (
        "Exploration: start a NEW subtree; there is no parent.\n"
        "Create alphas whose form is totally different from every alpha in references: a "
        "different economic mechanism, data fields from field_menu that the references do not "
        "use, a different operator structure. Each still needs a plausible economic reason "
        "for high Sharpe and fitness."
    ),
    "e2": (
        "Backbone crossover: recombine what the elites share.\n"
        "First identify the common backbone idea that makes the elites pass quality and write "
        "it in backbone. Then create alphas that keep that backbone but express it in a clearly "
        "different form from the parent: other fields, other transformation, other conditioning."
        + _CORRELATION_RECIPE
    ),
    "m1": (
        "Structural mutation: change the parent's form, keep its edge.\n"
        "Introduce a new mechanism: replace one leg entirely with a signal from another data "
        "family, condition the signal (trade_when, volatility or volume regime), change the "
        "normalisation, or add a new term. Keep the part of the parent that carries its Sharpe."
        + _CORRELATION_RECIPE
    ),
    "m2": (
        "Parameter mutation: retune the parent.\n"
        "Identify its main parameters (lookbacks, leg weights, grouping level, decay, "
        "neutralization) and propose different settings, changing them jointly rather than "
        "one number at a time."
    ),
    # HiFo-Prompt's M3: simplify components that are prone to overfitting.
    "m3": (
        "Simplification: make the parent simpler and more robust.\n"
        "Find the components that add parameters or conditions without a clear economic reason "
        "(extra legs, nested conditions, unusually long or short lookbacks) and remove or merge "
        "them, keeping the mechanism that carries the Sharpe. Simpler alphas tend to hold up in "
        "the recent-period and ladder Sharpe checks."
    ),
    "s1": (
        "Trajectory summary: learn from the lineage in path.\n"
        "List the changes along the path that clearly helped quality and those that hurt, then "
        "design new alphas that combine the helpful changes and should beat every step of the path."
    ),
}

#: Operator mix by what blocks the parent. Correlation blocks need new forms,
#: quality blocks need repair around a working idea. Used when the caller does
#: not pick the operator itself.
_OPERATOR_WEIGHTS = {
    "correlation": {"e1": 0.30, "m1": 0.30, "e2": 0.20, "s1": 0.15, "m2": 0.05},
    "quality": {"m1": 0.30, "m2": 0.25, "e2": 0.20, "s1": 0.15, "e1": 0.10},
}

#: MCTS-AHD expands a selected leaf with *every* refinement operator; ``e1`` is
#: the virtual root's operator because it has no parent to mutate.
#:
#: The set is deliberately not narrowed by what blocks the parent. It used to
#: drop ``m2`` for a correlation block (a parameter tweak inherits the parent's
#: correlation) and drop ``e2`` when the tree had no elites yet, which meant a
#: leaf was usually expanded by one or two operators instead of the paper's
#: full sweep — and the population collapsed onto window tweaks.
LEAF_OPERATORS = ("m1", "m2", "e2", "s1")
#: Widening an inner node uses e2, as in MCTS-AHD's ``operators[1]``.
WIDEN_OPERATOR = "e2"


def blocking_kind(e_vector: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> str:
    weakest = (_weakest(e_vector) or "").upper()
    if "CORRELATION" in weakest:
        return "correlation"
    if not weakest and "SELF_CORRELATION" in ((context or {}).get("failed_check_histogram") or {}):
        return "correlation"
    return "quality"


def leaf_operators(e_vector: Dict[str, float], context: Optional[Dict[str, Any]] = None) -> tuple:
    """Every leaf operator, as MCTS-AHD expands a node with its full set.

    ``e_vector``/``context`` are accepted (and ignored) so callers and the
    operator-mix reporting keep working; the mix is no longer conditioned on
    which gate blocks the parent.
    """
    return LEAF_OPERATORS


def _field_menu(fields: Optional[Iterable[str]], *, exclude: Iterable[str] = (), limit: int = 60, seed: int = 0) -> List[str]:
    """A diverse sample of data field ids, a few per name prefix (dataset proxy)."""
    import random

    pool = sorted(set(fields or ()) - set(exclude))
    if not pool:
        return []
    from alpha_agent.families import data_group

    # Bucket by dataset, then by name prefix within it, so every dataset is offered.
    groups: Dict[str, List[str]] = {}
    for name in pool:
        prefix = re.split(r"[_\d]", name, maxsplit=1)[0] or name
        groups.setdefault(f"{data_group(name)}:{prefix}", []).append(name)
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    menu: List[str] = []
    per_group = max(1, limit // max(1, len(order)))
    for prefix in order:
        members = groups[prefix]
        menu.extend(rng.sample(members, min(per_group, len(members))))
        if len(menu) >= limit:
            break
    return sorted(menu[:limit])


def _clean_settings_patch(patch: Any) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in patch.items():
        cast = _PATCHABLE_SETTINGS.get(key)
        if cast is None or value is None or isinstance(value, (dict, list)):
            continue
        try:
            cleaned[key] = cast(value).upper() if cast is str else cast(value)
        except (TypeError, ValueError):
            continue
    return cleaned


class LLMRefinementProposer:
    """Model-backed proposer with a deterministic fallback."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        fallback: Optional[DeterministicRefinementProposer] = None,
        max_calls: int = 64,
        seed: int = 7,
    ) -> None:
        import random

        self.model_config = model_config
        self.rng = random.Random(seed)
        self.operator_counts: Dict[str, int] = {}
        self.fallback = fallback or DeterministicRefinementProposer()
        self.max_calls = max(1, int(max_calls))
        #: Successful model round-trips; failures do not consume the cap.
        self.calls = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.fallback_count = 0
        self.last_error: Optional[str] = None
        self.last_operator: Optional[str] = None

    @property
    def enabled(self) -> bool:
        import os

        return bool(os.getenv(self.model_config.api_key_env, "").strip())

    def propose(
        self,
        *,
        node: Any,
        e_vector: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        blacklist: Sequence[str] = (),
        operators: Optional[Iterable[str]] = None,
        fields: Optional[Iterable[str]] = None,
        k: int = 2,
        operator: Optional[str] = None,
    ) -> List[ProposalExpression]:
        """``node=None`` with ``operator="e1"`` proposes a new subtree root."""
        proposal = self._propose_via_model(
            node=node,
            e_vector=e_vector,
            blacklist=blacklist,
            operators=operators,
            fields=fields,
            context=context,
            operator=operator,
        )
        if proposal is None or not proposal.expressions:
            self.fallback_count += 1
            if self.enabled and self.calls < self.max_calls:
                warn(
                    "LLM refinement unavailable, using deterministic mutations "
                    f"({self.last_error or 'empty or malformed proposal'})"
                )
            if node is None:
                # Deterministic mutations need a parent; a new root cannot fall back.
                return []
            return self.fallback.propose(
                node=node,
                e_vector=e_vector,
                context=context,
                blacklist=blacklist,
                operators=operators,
                fields=fields,
                k=k,
            )
        return list(proposal.expressions)[: max(1, k) if k else None]

    def choose_operator(self, e_vector: Dict[str, float], context: Optional[Dict[str, Any]]) -> str:
        weights = dict(_OPERATOR_WEIGHTS[blocking_kind(e_vector, context)])
        if not (context or {}).get("elites"):
            # e2 / s1 reason over a population; without one they degrade to m1.
            weights.pop("e2", None)
            weights.pop("s1", None)
        names = sorted(weights)
        return self.rng.choices(names, weights=[weights[name] for name in names], k=1)[0]

    # -- model plumbing -----------------------------------------------------
    def _propose_via_model(
        self,
        *,
        node: Any,
        e_vector: Dict[str, float],
        blacklist: Sequence[str],
        operators: Optional[Iterable[str]],
        context: Optional[Dict[str, Any]],
        fields: Optional[Iterable[str]] = None,
        operator: Optional[str] = None,
    ) -> Optional[RefinementProposal]:
        if not self.enabled or self.calls >= self.max_calls:
            return None
        import os

        api_key = os.getenv(self.model_config.api_key_env, "").strip()
        if operator not in _OPERATOR_INSTRUCTIONS:
            operator = "e1" if node is None else self.choose_operator(e_vector, context)
        weakest = _weakest(e_vector) or "unknown"
        deficits = sorted(e_vector.items(), key=lambda item: (item[1], item[0]))[:6]
        expression = str(getattr(node, "expression", "") or "")
        elites = list((context or {}).get("elites") or [])[:5]
        failed = list((context or {}).get("failed_attempts") or [])
        references = list((context or {}).get("references") or [])[:8]
        path = list((context or {}).get("path") or [])[:8]
        failed_children = list((context or {}).get("failed_children") or [])[:6]
        if operator == "i1":
            # From scratch, as in MCTS-AHD: no existing alpha, good or bad.
            elites, failed, references, path, failed_children = [], [], [], [], []
        # Fields already shown (or already failed) are left out of the menu so it surfaces new ones.
        used = set(
            re.findall(
                r"[A-Za-z_][A-Za-z0-9_]*",
                " ".join(
                    [expression] + [str(e.get("expression")) for e in elites + failed + references]
                ),
            )
        )
        prompt: Dict[str, Any] = {"operator": operator, "task": _OPERATOR_INSTRUCTIONS[operator]}
        if node is not None:
            prompt["parent"] = {
                "expression": expression,
                "settings": dict(getattr(node, "settings", {}) or {}),
                "family": str(getattr(node, "family", "") or ""),
                "binding_weak_check": weakest,
                "gate_margins_weakest_first": [
                    {"check": name, "margin": round(float(value), 4)} for name, value in deficits
                ],
            }
        if operator == "e1":
            prompt["references"] = references or elites
        if operator == "s1" and path:
            prompt["path"] = path
        arena = (context or {}).get("arena") or {}
        if arena:
            region = str(arena.get("region") or "").upper()
            universe = str(arena.get("universe") or "").upper()
            delay = arena.get("delay")
            # ``e1`` has no parent, so without this it never learns which arena
            # it is searching and proposes fields the universe does not carry.
            prompt["arena"] = {"region": region, "universe": universe, "delay": delay}
            prompt["arena_note"] = _arena_note(region, universe, delay)
        prompt.update({
            "elites": elites,
            "metric_directions": _METRIC_DIRECTIONS,
            # Negative examples: what was already simulated and why it failed.
            "failed_children_of_parent": failed_children,
            "recent_failed_attempts": failed[:8],
            "field_menu": _field_menu(fields, exclude=used, seed=self.calls + self.failures),
            "forbidden_subtrees": list(blacklist)[:10],
            "allowed_operators": sorted(set(operators or ()))[:120],
            "quality_note": _QUALITY_NOTE,
            "budget_note": _BUDGET_NOTE,
        })
        guidance = (context or {}).get("guidance") or {}
        if guidance:
            # Population search only: hindsight insights and the current regime.
            if guidance.get("insights"):
                prompt["design_insights"] = list(guidance["insights"])
            if guidance.get("directive"):
                prompt["design_directive"] = str(guidance["directive"])
            if guidance.get("regime_note"):
                prompt["search_regime"] = str(guidance["regime_note"])
        if context:
            prompt["run_context"] = {
                key: context.get(key)
                for key in ("iteration", "remaining_budget", "failed_check_histogram", "arena")
                if key in context
            }
        self.last_operator = operator
        self.operator_counts[operator] = self.operator_counts.get(operator, 0) + 1

        body = {
            "model": self.model_config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": float(self.model_config.temperature),
            "response_format": {"type": "json_object"},
        }
        try:
            completion = self._chat_completion(body=body, api_key=api_key)
            payload = self._extract_json(completion)
        except Exception as exc:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = str(exc) if isinstance(exc, LLMRequestError) else f"{type(exc).__name__}: {exc}"
            return None
        self.calls += 1
        self.consecutive_failures = 0
        return self._parse_proposal(payload)

    def _chat_completion(self, *, body: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        return post_chat_completion(
            base_url=self.model_config.base_url,
            body=body,
            api_key=api_key,
            timeout=self.model_config.timeout,
            reasoning_effort=self.model_config.reasoning_effort,
        )

    def _extract_json(self, completion: Dict[str, Any]) -> Dict[str, Any]:
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("planner response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
            return json.loads(text)
        raise ValueError("unsupported message content")

    def _parse_proposal(self, payload: Dict[str, Any]) -> Optional[RefinementProposal]:
        if not isinstance(payload, dict):
            return None
        raw_expressions = payload.get("expressions")
        if not isinstance(raw_expressions, list):
            return None
        items: List[ProposalExpression] = []
        for item in raw_expressions:
            if not isinstance(item, dict):
                continue
            expression = item.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                continue
            items.append(
                ProposalExpression(
                    expression=expression.strip(),
                    settings_patch=_clean_settings_patch(item.get("settings_patch")),
                    hypothesis=str(item.get("thought") or item.get("hypothesis") or ""),
                    rationale=f"[{self.last_operator}] {item.get('rationale') or payload.get('backbone') or ''}".strip(),
                )
            )
        if not items:
            return None
        return RefinementProposal(
            target_dimension=str(payload.get("target_dimension") or ""),
            suggestion=str(payload.get("suggestion") or ""),
            rationale=str(payload.get("rationale") or ""),
            expressions=tuple(items),
            raw=payload,
        )
