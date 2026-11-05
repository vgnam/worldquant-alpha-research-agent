"""Multi-objective population search with hindsight/foresight prompt guidance.

``--search-mode mo-hifo`` combines two published methods on a population of
alphas instead of the hybrid search tree:

* **Selection and survival** after MEoH (Yao et al., "Multi-objective Evolution
  of Heuristic Using Large Language Model", AAAI 2025). The dominance-
  dissimilarity score ``v_j = -sum_i D[i, j] * Sim(P_i, P_j)`` (``D[i, j] = 1``
  when ``i`` dominates ``j``) ranks the population: a non-dominated alpha scores
  0, a dominated one is penalised by how structurally similar its dominators
  are. Parents are drawn by ``softmax(v)``; survivors are the best ``N`` by
  ``v``. ``Sim`` is MEoH's subtree overlap, ``|H_i & H_j| / |H_j|``.
* **Prompt guidance** after HiFo-Prompt (Chen et al., ICLR 2026; the reference
  implementation is MIT-licensed). An insight pool (hindsight) keeps short
  design principles with an EMA effectiveness, a usage penalty and a recency
  bonus; a navigator (foresight) picks explore / exploit / balanced from
  stagnation and diversity and turns it into a design directive.

What is new here, because both papers are single-purpose:

* Objectives are BRAIN's own metrics: maximise Sharpe and returns, minimise
  ``max(turnover, 0.125)``. The floor is the one in BRAIN's fitness,
  ``sharpe * sqrt(|returns| / max(turnover, 0.125))``, so lowering turnover
  below 12.5% earns nothing, and fitness never decreases along a dominance
  step. The turnover band of the LOW/HIGH_TURNOVER checks (1%..70% unless the
  check payload says otherwise) is a constraint, handled by constrained
  domination: feasible beats infeasible, then smaller violation wins. A
  SELF/PROD_CORRELATION FAIL the platform reported is a second constraint,
  whether or not the alpha cleared the quality gates; PENDING counts as none.
* HiFo's tiered credit is computed from Pareto relations instead of a scalar
  fitness: a child that joins the front and dominates members earns 0.8..1.0,
  a child that only joins it 0.2..0.8 (by its share of the new hypervolume), a
  dominated child -0.3..0.2 (by how few members dominate it), a platform
  rejection -0.5. Each insight also keeps the mean change it produced in every
  objective, which the prompt shows next to it.
* The pool extracts "avoid" insights from alphas blocked by SELF/PROD
  correlation, and the navigator has a fourth regime, ``decorrelate``, for the
  generation after such a block.
* Diversity is measured on formulas (mean pairwise subtree dissimilarity and
  the share of the dominant data group), not on exact text: every alpha's text
  differs, so HiFo's string test would always report full diversity.
* The regime also weights which operators run, since a simulation budget does
  not allow HiFo's full operator sweep per generation.

The model never sees the paper names; see ``refine_llm`` for the prompt text.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import alpha_research_pipeline as pipeline
from alpha_agent.expr_ast import subtree_hashes, try_parse
from alpha_agent.families import lineage_family
from alpha_agent.pareto import hypervolume

__all__ = [
    "OBJECTIVES",
    "Individual",
    "InsightPool",
    "Navigator",
    "InsightExtractor",
    "MoHifoState",
    "MoHifoSearch",
    "individual_from_record",
    "constrained_dominates",
    "dominance_dissimilarity",
    "survive",
    "select_parents",
    "pareto_credit",
    "population_hypervolume",
    "population_diversity",
]

#: Pareto objectives, in this order: maximise Sharpe, maximise returns,
#: minimise turnover (floored, see TURNOVER_FLOOR).
OBJECTIVES = ("sharpe", "returns", "turnover")
#: BRAIN's fitness divides by max(turnover, 0.125).
TURNOVER_FLOOR = 0.125
#: LOW_TURNOVER / HIGH_TURNOVER limits when the check payload does not say.
DEFAULT_TURNOVER_BAND = (0.01, 0.70)
#: Correlation checks whose reported FAIL is a constraint violation.
CORRELATION_CHECKS = ("SELF_CORRELATION", "PROD_CORRELATION")
#: Their limit; a FAIL reported without a value, or at the limit, counts as
#: the minimum violation.
CORRELATION_LIMIT = 0.70
MIN_CORRELATION_VIOLATION = 0.01
#: Hypervolume box per objective, (worst, best); values are clamped into it.
_HV_BOUNDS = {
    "sharpe": (-1.0, 4.0),
    "returns": (-0.10, 0.30),
    "turnover": (0.80, TURNOVER_FLOOR),
}
STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Individuals and objectives
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """One evaluated alpha in the population (or one offspring)."""

    key: str
    expression: str
    settings: Dict[str, Any] = field(default_factory=dict)
    family: str = ""
    alpha_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    #: Gate margins, weakest first in the prompt (the proposer's e_vector).
    margins: Dict[str, float] = field(default_factory=dict)
    failed_checks: List[str] = field(default_factory=list)
    quality_ready: bool = False
    correlation_blocked: bool = False
    self_correlation: Optional[float] = None
    prod_correlation: Optional[float] = None
    hypothesis: str = ""
    operator: str = ""
    parents: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self) -> None:
        # Populations saved earlier only flagged blocks past the quality gates.
        if any(name in self.failed_checks for name in CORRELATION_CHECKS):
            self.correlation_blocked = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Individual":
        known = {name for name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})

    def summary(self) -> Dict[str, Any]:
        """What the model sees of this alpha (elites, references)."""
        return {
            "expression": self.expression,
            "idea": self.hypothesis or None,
            "family": self.family,
            "sharpe": self.metrics.get("sharpe"),
            "returns": self.metrics.get("returns"),
            "turnover": self.metrics.get("turnover"),
            "fitness": self.metrics.get("fitness"),
            "self_correlation": self.self_correlation,
            "prod_correlation": self.prod_correlation,
            "failed_checks": list(self.failed_checks),
        }


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def objective_values(metrics: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """``(sharpe, returns, -max(turnover, floor))``: every coordinate maximised."""
    sharpe = _float(metrics.get("sharpe"))
    returns = _float(metrics.get("returns"))
    turnover = _float(metrics.get("turnover"))
    if sharpe is None or returns is None or turnover is None:
        return None
    return (sharpe, returns, -max(turnover, TURNOVER_FLOOR))


def normalized_objectives(metrics: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    """Objectives scaled into ``[0, 1]`` (1 = best) for the hypervolume."""
    raw = {
        "sharpe": _float(metrics.get("sharpe")),
        "returns": _float(metrics.get("returns")),
        "turnover": _float(metrics.get("turnover")),
    }
    if any(value is None for value in raw.values()):
        return None
    raw["turnover"] = max(float(raw["turnover"]), TURNOVER_FLOOR)  # type: ignore[arg-type]
    vector = []
    for name in OBJECTIVES:
        worst, best = _HV_BOUNDS[name]
        fraction = (float(raw[name]) - worst) / (best - worst)  # type: ignore[arg-type]
        vector.append(min(1.0, max(0.0, fraction)))
    return tuple(vector)


def turnover_violation(metrics: Dict[str, Any]) -> float:
    """Distance outside the LOW/HIGH_TURNOVER band; ``inf`` without metrics."""
    turnover = _float(metrics.get("turnover"))
    if turnover is None or objective_values(metrics) is None:
        return math.inf
    low = _float(metrics.get("low_turnover_limit"))
    high = _float(metrics.get("high_turnover_limit"))
    low = DEFAULT_TURNOVER_BAND[0] if low is None else low
    high = DEFAULT_TURNOVER_BAND[1] if high is None else high
    if turnover < low:
        return low - turnover
    if turnover > high:
        return turnover - high
    return 0.0


def correlation_violation(individual: "Individual") -> float:
    """How far each reported SELF/PROD_CORRELATION FAIL sits above the limit.

    PENDING or unreported correlation is not penalised: the platform computes
    it lazily and formula similarity does not predict it, so only its verdict
    counts.
    """
    values = {"SELF_CORRELATION": individual.self_correlation, "PROD_CORRELATION": individual.prod_correlation}
    total = 0.0
    for name in CORRELATION_CHECKS:
        if name in individual.failed_checks:
            value = values[name]
            excess = value - CORRELATION_LIMIT if value is not None else 0.0
            total += max(excess, MIN_CORRELATION_VIOLATION)
    return total


def violation(individual: "Individual") -> float:
    """Total constraint violation: the turnover band plus reported correlation blocks."""
    return turnover_violation(individual.metrics) + correlation_violation(individual)


def _check_value(record: Dict[str, Any], name: str) -> Optional[float]:
    for item in record.get("check_diagnostics") or []:
        if isinstance(item, dict) and item.get("name") == name:
            return _float(item.get("value"))
    return None


def individual_from_record(
    record: Dict[str, Any],
    *,
    margins: Optional[Dict[str, float]] = None,
    generation: int = 0,
) -> Optional[Individual]:
    """An evaluated record as a population member; None for failed requests."""
    if record.get("status") != "ok":
        return None
    expression = str(record.get("expression") or "")
    if not expression:
        return None
    metadata = record.get("metadata") or {}
    quality_ready = bool(pipeline.record_quality_ready(record))
    return Individual(
        key=str(pipeline.record_signature(record) or record.get("candidate_signature") or expression),
        expression=expression,
        settings=dict(record.get("settings") or {}),
        family=str(record.get("family") or lineage_family(expression)),
        alpha_id=record.get("alpha_id"),
        metrics=dict(record.get("metrics") or {}),
        margins=dict(margins or {}),
        failed_checks=list(record.get("failed_checks") or []),
        quality_ready=quality_ready,
        correlation_blocked=bool(record.get("failed_correlation_checks")),
        self_correlation=_check_value(record, "SELF_CORRELATION"),
        prod_correlation=_check_value(record, "PROD_CORRELATION"),
        hypothesis=str(metadata.get("hypothesis") or ""),
        operator=str(metadata.get("operator") or ""),
        parents=list(metadata.get("parents") or []),
        insights=list(metadata.get("insights") or []),
        generation=int(metadata.get("generation", generation) or generation),
    )


# ---------------------------------------------------------------------------
# Dominance and the MEoH score
# ---------------------------------------------------------------------------


def pareto_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def constrained_dominates(a: Individual, b: Individual) -> bool:
    """Deb's constrained domination over the three objectives."""
    va, vb = violation(a), violation(b)
    if va == 0.0 and vb > 0.0:
        return True
    if va > 0.0 or vb > 0.0:
        return vb > va
    oa, ob = objective_values(a.metrics), objective_values(b.metrics)
    if oa is None or ob is None:
        return False
    return pareto_dominates(oa, ob)


_HASH_CACHE: Dict[str, frozenset] = {}


def _hashes(expression: str) -> frozenset:
    cached = _HASH_CACHE.get(expression)
    if cached is None:
        parsed = try_parse(expression)
        cached = frozenset(subtree_hashes(parsed, min_size=2)) if parsed is not None else frozenset()
        if len(_HASH_CACHE) > 20_000:
            _HASH_CACHE.clear()
        _HASH_CACHE[expression] = cached
    return cached


def ast_similarity(a: str, b: str) -> float:
    """MEoH's ``Sim_AST(a, b)``: share of ``b``'s subtrees that also occur in ``a``."""
    ha, hb = _hashes(a), _hashes(b)
    if not hb:
        return 1.0 if a == b else 0.0
    return len(ha & hb) / len(hb)


def dominance_dissimilarity(population: Sequence[Individual]) -> List[float]:
    """``v_j = -sum over dominators i of Sim(P_i, P_j)``; 0 is best."""
    scores = [0.0] * len(population)
    for j, target in enumerate(population):
        for i, other in enumerate(population):
            if i != j and constrained_dominates(other, target):
                scores[j] -= ast_similarity(other.expression, target.expression)
    return scores


def _feasible(individual: Individual) -> bool:
    return violation(individual) == 0.0


def _dominator_counts(population: Sequence[Individual]) -> List[int]:
    return [
        sum(1 for i, other in enumerate(population) if i != j and constrained_dominates(other, target))
        for j, target in enumerate(population)
    ]


def survive(pool: Sequence[Individual], size: int) -> List[Individual]:
    """Keep the best ``size`` by ``v`` (MEoH population management).

    Ties on ``v`` (a dominated alpha whose dominators share nothing with it
    also scores 0) go to feasible alphas, then to fewer dominators, then to
    higher Sharpe, so a 129%-turnover outlier never outlives a valid peer.
    """
    unique: Dict[str, Individual] = {}
    for individual in pool:
        unique.setdefault(individual.key, individual)
    members = list(unique.values())
    scores = dominance_dissimilarity(members)
    dominators = _dominator_counts(members)
    order = sorted(
        range(len(members)),
        key=lambda index: (
            not _feasible(members[index]),
            -scores[index],
            dominators[index],
            -(_float(members[index].metrics.get("sharpe")) or -math.inf),
            members[index].key,
        ),
    )
    return [members[index] for index in order[: max(1, int(size))]]


def select_parents(
    population: Sequence[Individual], count: int, rng: random.Random, *, infeasible_penalty: float = 1.0
) -> List[Individual]:
    """Draw ``count`` distinct parents by ``softmax(v)`` (MEoH parent selection)."""
    members = list(population)
    if not members:
        return []
    scores = dominance_dissimilarity(members)
    scores = [score - (0.0 if _feasible(member) else infeasible_penalty) for score, member in zip(scores, members)]
    chosen: List[Individual] = []
    pool = list(range(len(members)))
    for _ in range(min(int(count), len(members))):
        top = max(scores[index] for index in pool)
        weights = [math.exp(scores[index] - top) for index in pool]
        pick = rng.choices(pool, weights=weights, k=1)[0]
        chosen.append(members[pick])
        pool.remove(pick)
    return chosen


def non_dominated(population: Sequence[Individual]) -> List[Individual]:
    return [
        member
        for member in population
        if not any(constrained_dominates(other, member) for other in population if other is not member)
    ]


def population_hypervolume(population: Sequence[Individual]) -> float:
    vectors = [
        normalized_objectives(member.metrics)
        for member in population
        if _feasible(member) and normalized_objectives(member.metrics) is not None
    ]
    return hypervolume(vector for vector in vectors if vector is not None)


def population_diversity(population: Sequence[Individual]) -> Dict[str, float]:
    """Mean pairwise subtree dissimilarity and the dominant data group's share."""
    members = list(population)
    if len(members) < 2:
        return {"dissimilarity": 1.0, "top_group_share": 0.0}
    total = 0.0
    pairs = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = _hashes(members[i].expression), _hashes(members[j].expression)
            union = a | b
            total += 1.0 - (len(a & b) / len(union) if union else 1.0)
            pairs += 1
    groups: Dict[str, int] = {}
    for member in members:
        group = member.family or lineage_family(member.expression)
        groups[group] = groups.get(group, 0) + 1
    return {
        "dissimilarity": round(total / pairs, 4),
        "top_group_share": round(max(groups.values()) / len(members), 4),
    }


def pareto_credit(child: Optional[Individual], population: Sequence[Individual]) -> float:
    """HiFo's tiered credit, with Pareto relations in place of a scalar fitness."""
    if child is None:
        return -0.5
    members = [member for member in population if member.key != child.key]
    if not members:
        return 0.1
    dominated_by = sum(1 for member in members if constrained_dominates(member, child))
    dominates = sum(1 for member in members if constrained_dominates(child, member))
    if dominated_by == 0 and dominates > 0:
        credit = 0.8 + 0.2 * dominates / len(members)
    elif dominated_by == 0:
        before = population_hypervolume(non_dominated(members))
        after = population_hypervolume(non_dominated(members + [child]))
        share = (after - before) / after if after > 0 else 0.0
        credit = 0.2 + 0.6 * max(0.0, min(1.0, share))
    else:
        credit = -0.3 + 0.5 * (1.0 - dominated_by / len(members))
    return max(-1.0, min(1.0, credit))


# ---------------------------------------------------------------------------
# Hindsight: the insight pool
# ---------------------------------------------------------------------------

#: Generic starting principles (HiFo seeds its pool with expert tips too).
SEED_INSIGHTS = (
    "Anchor each alpha on one clear economic mechanism (mispricing, risk premium, or a "
    "behavioural over- or under-reaction) and pick fields that measure it directly.",
    "Normalise legs within industry or sector before combining them so the bet is relative "
    "to peers rather than to sector moves.",
    "Combine legs from different data families with complementary horizons, a slow "
    "fundamental leg and a faster market-data leg, weighted so neither dominates.",
    "Smooth noisy fast signals with decay or a short mean so turnover stays moderate "
    "without destroying the edge; returns per unit of turnover drive fitness.",
    "Prefer simple formulas with few parameters; conditions that only fit the sample tend "
    "to fail the recent-period and ladder Sharpe checks.",
)


def _words(text: str) -> set:
    return set(text.lower().split())


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class InsightPool:
    """Bounded pool of design principles with HiFo's utility and eviction rules.

    Differences from the reference implementation: similarity is a real
    Jaccard index (the repo divides by the larger set), every insight keeps
    the mean objective change it produced (``effects``), and ``avoid`` insights
    get one reserved slot in each prompt.
    """

    def __init__(
        self,
        *,
        max_size: int = 30,
        novelty_threshold: float = 0.7,
        alpha: float = 0.3,
        usage_weight: float = 0.1,
        recency_bonus: float = 0.2,
        recency_window: int = 2,
        decay_rate: float = 0.01,
        probation: int = 3,
        seeds: Iterable[str] = SEED_INSIGHTS,
    ) -> None:
        self.max_size = int(max_size)
        self.novelty_threshold = float(novelty_threshold)
        self.alpha = float(alpha)
        self.usage_weight = float(usage_weight)
        self.recency_bonus = float(recency_bonus)
        self.recency_window = int(recency_window)
        self.decay_rate = float(decay_rate)
        self.probation = int(probation)
        self.generation = 0
        self.items: Dict[str, Dict[str, Any]] = {}
        for text in seeds:
            self.add(text, kind="positive", source="seed")

    # -- admission and eviction --------------------------------------------
    def add(self, text: str, *, kind: str = "positive", source: str = "extracted") -> bool:
        text = " ".join(str(text or "").split())
        if len(text) < 12 or text in self.items:
            return False
        if any(jaccard(text, other) > self.novelty_threshold for other in self.items):
            return False
        if len(self.items) >= self.max_size:
            self._evict()
        self.items[text] = {
            "kind": "avoid" if kind == "avoid" else "positive",
            "source": source,
            "effectiveness": 0.0,
            "used_count": 0,
            "credit_count": 0,
            "last_used_generation": self.generation,
            "added_generation": self.generation,
            "effects": {},
        }
        return True

    def _evict(self) -> None:
        candidates = [
            (stats["effectiveness"] - self.decay_rate * (self.generation - stats["last_used_generation"]), text)
            for text, stats in self.items.items()
            if stats["used_count"] >= self.probation
        ]
        if candidates:
            victim = min(candidates)[1]
        else:
            victim = min(self.items, key=lambda text: (self.items[text]["last_used_generation"], text))
        del self.items[victim]

    # -- retrieval ----------------------------------------------------------
    def utility(self, text: str) -> float:
        stats = self.items[text]
        recent = (self.generation - stats["last_used_generation"]) <= self.recency_window and stats["used_count"] > 0
        return (
            stats["effectiveness"]
            - self.usage_weight * math.log(stats["used_count"] + 1)
            + (self.recency_bonus if recent else 0.0)
        )

    def select(self, k: int = 3) -> List[str]:
        """Top ``k`` by utility; one slot goes to the best ``avoid`` insight if any."""
        if not self.items or k <= 0:
            return []
        ranked = sorted(self.items, key=lambda text: (-self.utility(text), text))
        avoid = [text for text in ranked if self.items[text]["kind"] == "avoid"]
        chosen: List[str] = avoid[:1] if avoid else []
        for text in ranked:
            if len(chosen) >= k:
                break
            if text not in chosen and self.items[text]["kind"] == "positive":
                chosen.append(text)
        for text in chosen:
            self.items[text]["used_count"] += 1
            self.items[text]["last_used_generation"] = self.generation
        return chosen

    def describe(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        """Prompt view: each insight with its measured effect and sample count."""
        out: List[Dict[str, Any]] = []
        for text in texts:
            stats = self.items.get(text)
            if stats is None:
                continue
            item: Dict[str, Any] = {"insight": text, "kind": stats["kind"]}
            effects = stats.get("effects") or {}
            count = int(effects.get("n", 0) or 0)
            if count:
                item["observed_effect"] = {
                    name: round(float(effects[name]), 4) for name in OBJECTIVES if name in effects
                }
                item["n"] = count
            out.append(item)
        return out

    # -- credit -------------------------------------------------------------
    def credit(self, texts: Iterable[str], value: float, deltas: Optional[Dict[str, float]] = None) -> None:
        for text in texts:
            stats = self.items.get(text)
            if stats is None:
                continue
            stats["effectiveness"] = max(
                -1.0, min(1.0, (1 - self.alpha) * stats["effectiveness"] + self.alpha * float(value))
            )
            stats["credit_count"] += 1
            stats["last_used_generation"] = self.generation
            if deltas:
                effects = stats.setdefault("effects", {})
                count = int(effects.get("n", 0) or 0) + 1
                for name, delta in deltas.items():
                    previous = float(effects.get(name, 0.0))
                    effects[name] = previous + (float(delta) - previous) / count
                effects["n"] = count

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"generation": self.generation, "items": self.items}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], **options: Any) -> "InsightPool":
        pool = cls(seeds=(), **options)
        pool.generation = int(payload.get("generation", 0) or 0)
        for text, stats in (payload.get("items") or {}).items():
            if isinstance(stats, dict):
                pool.items[str(text)] = {
                    "kind": stats.get("kind", "positive"),
                    "source": stats.get("source", "extracted"),
                    "effectiveness": float(stats.get("effectiveness", 0.0)),
                    "used_count": int(stats.get("used_count", 0)),
                    "credit_count": int(stats.get("credit_count", 0)),
                    "last_used_generation": int(stats.get("last_used_generation", 0)),
                    "added_generation": int(stats.get("added_generation", 0)),
                    "effects": dict(stats.get("effects") or {}),
                }
        return pool


# ---------------------------------------------------------------------------
# Foresight: the navigator
# ---------------------------------------------------------------------------

REGIMES = ("exploration", "exploitation", "balanced", "decorrelate")

#: Directive pools, after HiFo's per-regime pools, rewritten for alphas.
DESIGN_DIRECTIVES = {
    "balanced": (
        "the economic reason the signal should earn a premium",
        "how each leg behaves in the large, liquid names of the universe",
        "returns per unit of turnover, which drives fitness",
        "robustness of the signal across time periods",
    ),
    "exploitation": (
        "fine-tuning lookbacks, leg weights and decay of the strongest pattern",
        "removing components that add turnover without adding Sharpe",
        "raising returns of the best trade-offs without raising turnover",
        "the normalisation and grouping level of the strongest legs",
    ),
    "exploration": (
        "an economic mechanism the population does not use yet",
        "data families the population has not drawn from",
        "a different signal horizon than the population's dominant one",
        "a different construction: conditioning, regime filters or cross-sectional grouping",
    ),
    "decorrelate": (
        "keeping the leg that carries the Sharpe and replacing the crowded leg with one from another data family",
        "an industry-level normalisation of the crowded leg",
        "replacing generic price-reversal components with a signal fewer alphas are likely to own",
    ),
}

REGIME_NOTES = {
    "exploration": "Explore an approach clearly different from the parents and from conventional formulas.",
    "exploitation": "Refine and optimise the most effective patterns of the parents.",
    "balanced": "Balance new ideas with patterns that are already proven in the parents.",
    "decorrelate": (
        "Recent alphas were blocked by correlation with alphas that already exist: keep the "
        "mechanism that earns the Sharpe and change the part that is likely owned by the "
        "account or the production pool."
    ),
}

#: Operator mix per regime. HiFo runs every operator each generation; a
#: simulation budget cannot afford that, so the regime sets the mix instead.
REGIME_OPERATOR_WEIGHTS = {
    "exploration": {"e1": 0.35, "e2": 0.25, "m1": 0.25, "m2": 0.05, "m3": 0.10},
    "exploitation": {"e1": 0.05, "e2": 0.20, "m1": 0.25, "m2": 0.35, "m3": 0.15},
    "balanced": {"e1": 0.20, "e2": 0.20, "m1": 0.20, "m2": 0.20, "m3": 0.20},
    "decorrelate": {"e1": 0.30, "e2": 0.20, "m1": 0.40, "m2": 0.00, "m3": 0.10},
}


class Navigator:
    """Rule-based regime policy (HiFo's Eq. 6) on population hypervolume."""

    def __init__(
        self,
        *,
        stagnation_limit: int = 3,
        progress_limit: int = 2,
        diversity_floor: float = 0.3,
        group_share_ceiling: float = 0.6,
        hv_epsilon: float = 1e-4,
        seed: int = 7,
    ) -> None:
        self.stagnation_limit = int(stagnation_limit)
        self.progress_limit = int(progress_limit)
        self.diversity_floor = float(diversity_floor)
        self.group_share_ceiling = float(group_share_ceiling)
        self.hv_epsilon = float(hv_epsilon)
        self.rng = random.Random(seed)
        self.stagnation = 0
        self.progress = 0
        self.last_hv: Optional[float] = None
        self.regime = "balanced"
        self.directive = DESIGN_DIRECTIVES["balanced"][0]
        self.history: List[Dict[str, Any]] = []

    def update(
        self,
        *,
        hypervolume_value: float,
        diversity: Dict[str, float],
        correlation_blocked: bool,
        population_size: int,
        generation: int,
    ) -> str:
        if self.last_hv is not None:
            if hypervolume_value - self.last_hv > self.hv_epsilon:
                self.progress += 1
                self.stagnation = 0
            else:
                self.stagnation += 1
                self.progress = 0
        self.last_hv = float(hypervolume_value)
        low_diversity = population_size >= 4 and (
            diversity.get("dissimilarity", 1.0) < self.diversity_floor
            or diversity.get("top_group_share", 0.0) > self.group_share_ceiling
        )
        if correlation_blocked:
            regime = "decorrelate"
        elif self.stagnation >= self.stagnation_limit or low_diversity:
            regime = "exploration"
        elif self.progress >= self.progress_limit:
            regime = "exploitation"
        else:
            regime = self.rng.choices(
                ["exploration", "exploitation", "balanced"], weights=[0.25, 0.25, 0.5], k=1
            )[0]
        self.regime = regime
        self.directive = self.rng.choice(DESIGN_DIRECTIVES[regime])
        self.history.append(
            {
                "generation": generation,
                "regime": regime,
                "hypervolume": round(float(hypervolume_value), 6),
                "stagnation": self.stagnation,
                "progress": self.progress,
                "diversity": dict(diversity),
                "correlation_blocked": bool(correlation_blocked),
            }
        )
        return regime

    def sample_operator(self, rng: random.Random) -> str:
        weights = {name: value for name, value in REGIME_OPERATOR_WEIGHTS[self.regime].items() if value > 0}
        names = sorted(weights)
        return rng.choices(names, weights=[weights[name] for name in names], k=1)[0]

    def guidance(self) -> Dict[str, str]:
        return {
            "regime": self.regime,
            "directive": f"For this step, pay special attention to {self.directive}.",
            "regime_note": REGIME_NOTES[self.regime],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stagnation": self.stagnation,
            "progress": self.progress,
            "last_hv": self.last_hv,
            "regime": self.regime,
            "directive": self.directive,
            "history": self.history[-200:],
        }

    def load(self, payload: Dict[str, Any]) -> None:
        self.stagnation = int(payload.get("stagnation", 0) or 0)
        self.progress = int(payload.get("progress", 0) or 0)
        self.last_hv = _float(payload.get("last_hv"))
        regime = str(payload.get("regime") or "balanced")
        self.regime = regime if regime in REGIMES else "balanced"
        self.directive = str(payload.get("directive") or DESIGN_DIRECTIVES[self.regime][0])
        self.history = list(payload.get("history") or [])


# ---------------------------------------------------------------------------
# Insight extraction (one model call per generation)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You distil reusable design principles for formulaic equity alphas on WorldQuant BRAIN. "
    "You get the strongest recent alphas (each a trade-off of Sharpe, returns and turnover) "
    "and alphas the platform blocked by correlation with existing alphas. Return strict JSON only: "
    '{"principles": [str], "avoid": [str]}. '
    "principles: 1 or 2 concise, generic, performance-positive design principles drawn from "
    "both the ideas and the formulas of the strong alphas. avoid: at most 1 principle saying "
    "which construction to avoid because it leads to correlation blocks, only if the blocked "
    "alphas show one. Each item is one sentence about strategy, not about a specific window "
    "or number, and must not repeat an existing insight."
)


class InsightExtractor:
    """Asks the model for new insights; ``chat`` is injectable for tests."""

    def __init__(self, chat: Callable[[Dict[str, Any]], Dict[str, Any]], *, model: str, temperature: float) -> None:
        self.chat = chat
        self.model = model
        self.temperature = float(temperature)
        self.calls = 0
        self.failures = 0
        self.last_error: Optional[str] = None

    def extract(
        self,
        *,
        strong: Sequence[Individual],
        blocked: Sequence[Individual],
        existing: Sequence[str],
    ) -> Tuple[List[str], List[str]]:
        if not strong and not blocked:
            return [], []
        payload = {
            "strong_alphas": [member.summary() for member in strong][:6],
            "correlation_blocked_alphas": [member.summary() for member in blocked][:4],
            "existing_insights": list(existing)[:30],
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            completion = self.chat(body)
            content = completion["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else {}
        except Exception as exc:  # extraction is optional: never stop the run for it
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            return [], []
        self.calls += 1
        principles = [str(item) for item in (parsed.get("principles") or []) if isinstance(item, str)][:2]
        avoid = [str(item) for item in (parsed.get("avoid") or []) if isinstance(item, str)][:1]
        if not blocked:
            avoid = []
        return principles, avoid


# ---------------------------------------------------------------------------
# State and the search loop
# ---------------------------------------------------------------------------


@dataclass
class MoHifoState:
    """Everything ``continue`` needs: population, insights, navigator."""

    path: Path
    state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    initialised: bool = False
    population: List[Individual] = field(default_factory=list)
    #: Offspring evaluated in the current generation, not yet through survival.
    offspring: List[Individual] = field(default_factory=list)
    seen_keys: List[str] = field(default_factory=list)
    insights: InsightPool = field(default_factory=InsightPool)
    navigator: Navigator = field(default_factory=Navigator)

    def save(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "state_id": self.state_id,
            "generation": self.generation,
            "initialised": self.initialised,
            "objectives": list(OBJECTIVES),
            "population": [member.to_dict() for member in self.population],
            "offspring": [member.to_dict() for member in self.offspring],
            "seen_keys": self.seen_keys[-5000:],
            "insights": self.insights.to_dict(),
            "navigator": self.navigator.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    @classmethod
    def load_or_new(cls, path: Path, *, seed: int = 7) -> "MoHifoState":
        state = cls(path=Path(path), navigator=Navigator(seed=seed))
        if not state.path.exists():
            return state
        try:
            payload = json.loads(state.path.read_text(encoding="utf-8"))
        except Exception:
            return state
        state.state_id = str(payload.get("state_id") or state.state_id)
        state.generation = int(payload.get("generation", 0) or 0)
        state.initialised = bool(payload.get("initialised", False))
        state.population = [Individual.from_dict(item) for item in payload.get("population") or []]
        state.offspring = [Individual.from_dict(item) for item in payload.get("offspring") or []]
        state.seen_keys = [str(key) for key in payload.get("seen_keys") or []]
        state.insights = InsightPool.from_dict(payload.get("insights") or {})
        state.navigator.load(payload.get("navigator") or {})
        return state

    def recover(self, records: Iterable[Dict[str, Any]], margins_of: Callable[[Dict[str, Any]], Dict[str, float]]) -> int:
        """Re-attach offspring evaluated after the last save (a killed run)."""
        seen = set(self.seen_keys)
        recovered = 0
        for record in records:
            metadata = record.get("metadata") or {}
            if metadata.get("mo_hifo_state") != self.state_id:
                continue
            key = str(pipeline.record_signature(record) or "")
            if not key or key in seen:
                continue
            seen.add(key)
            self.seen_keys.append(key)
            child = individual_from_record(record, margins=margins_of(record), generation=self.generation)
            if child is None:
                continue
            (self.offspring if self.initialised else self.population).append(child)
            recovered += 1
        return recovered


#: ``propose(operator, parents, guidance)`` -> accepted candidates, or None
#: when the model is down (the run stops instead of spending on fallbacks).
ProposeFn = Callable[[str, List[Individual], Dict[str, Any]], Optional[List[pipeline.Candidate]]]
#: ``evaluate(candidates)`` -> (records, stop_reason or None).
EvaluateFn = Callable[[List[pipeline.Candidate]], Tuple[List[Dict[str, Any]], Optional[str]]]


class MoHifoSearch:
    """Generation loop: initialise, breed in batches, survive, guide."""

    def __init__(
        self,
        *,
        state: MoHifoState,
        propose: ProposeFn,
        evaluate: EvaluateFn,
        margins_of: Callable[[Dict[str, Any]], Dict[str, float]],
        extractor: Optional[InsightExtractor] = None,
        population_size: int = 10,
        crossover_parents: int = 5,
        batch_size: int = 2,
        insights_per_prompt: int = 3,
        max_empty_batches: int = 3,
        seed: int = 7,
        should_stop: Optional[Callable[[], Optional[str]]] = None,
        on_batch: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.state = state
        self.propose = propose
        self.evaluate = evaluate
        self.margins_of = margins_of
        self.extractor = extractor
        self.population_size = max(2, int(population_size))
        self.crossover_parents = max(2, int(crossover_parents))
        self.batch_size = max(1, int(batch_size))
        self.insights_per_prompt = int(insights_per_prompt)
        self.max_empty_batches = max(1, int(max_empty_batches))
        self.rng = random.Random(seed)
        self.should_stop = should_stop
        self.on_batch = on_batch
        self.spent = 0
        self.batches = 0
        self.extractions: List[Dict[str, Any]] = []

    # -- one operator call ---------------------------------------------------
    def _plan(self, pending: Sequence[Individual] = ()) -> Tuple[str, List[Individual]]:
        state = self.state
        if not state.initialised:
            # One i1 from scratch, then e1 alphas that must differ from every
            # one so far, including those written for this batch but not yet
            # simulated.
            known = state.population + state.offspring + list(pending)
            return ("i1" if not known else "e1"), list(known)
        operator = state.navigator.sample_operator(self.rng)
        if operator in {"e1", "e2"}:
            parents = select_parents(state.population, self.crossover_parents, self.rng)
        else:
            parents = select_parents(state.population, 1, self.rng)
        if operator == "e2" and len(parents) < 2:
            operator = "m1"
        return operator, parents

    def _guidance(self) -> Dict[str, Any]:
        state = self.state
        state.insights.generation = state.generation
        chosen = state.insights.select(self.insights_per_prompt)
        guidance = dict(state.navigator.guidance())
        guidance["insights"] = state.insights.describe(chosen)
        guidance["insight_texts"] = chosen
        guidance["generation"] = state.generation
        guidance["state_id"] = state.state_id
        guidance["elites"] = [member.summary() for member in non_dominated(state.population)][:5]
        return guidance

    def _deltas(self, child: Individual, parents: Sequence[Individual]) -> Dict[str, float]:
        reference = parents[:1] or self.state.population
        deltas: Dict[str, float] = {}
        for name in OBJECTIVES:
            value = _float(child.metrics.get(name))
            base = [_float(member.metrics.get(name)) for member in reference]
            base = [item for item in base if item is not None]
            if value is not None and base:
                deltas[name] = value - sum(base) / len(base)
        return deltas

    # -- generation bookkeeping ------------------------------------------------
    def _close_generation(self) -> None:
        state = self.state
        blocked = [member for member in state.offspring if member.correlation_blocked]
        if not state.initialised:
            state.initialised = True
        else:
            state.population = survive(state.population + state.offspring, self.population_size)
        state.offspring = []
        state.generation += 1
        state.insights.generation = state.generation
        diversity = population_diversity(state.population)
        state.navigator.update(
            hypervolume_value=population_hypervolume(state.population),
            diversity=diversity,
            correlation_blocked=bool(blocked),
            population_size=len(state.population),
            generation=state.generation,
        )
        if self.extractor is not None:
            front = [member for member in non_dominated(state.population) if _feasible(member)]
            strong = front[: max(1, int(math.ceil(len(state.population) * 0.3)))]
            blocked_all = blocked or [member for member in state.population if member.correlation_blocked]
            positives, avoids = self.extractor.extract(
                strong=strong, blocked=blocked_all, existing=list(state.insights.items)
            )
            added = [text for text in positives if state.insights.add(text, kind="positive")]
            added += [text for text in avoids if state.insights.add(text, kind="avoid")]
            self.extractions.append({"generation": state.generation, "proposed": positives + avoids, "added": added})

    def _generation_full(self) -> bool:
        state = self.state
        if not state.initialised:
            return len(state.population) >= self.population_size
        return len(state.offspring) >= self.population_size

    # -- main loop ---------------------------------------------------------------
    def run(self, budget: int, *, max_batches: int) -> str:
        state = self.state
        empty_batches = 0
        init_calls = 0
        while self.spent < budget:
            if self.batches >= max_batches:
                return "Iteration cap reached."
            if self.should_stop is not None:
                reason = self.should_stop()
                if reason:
                    return reason
            if self._generation_full():
                self._close_generation()
                state.save()
                continue
            batch: List[pipeline.Candidate] = []
            pending: List[Individual] = []
            meta: Dict[str, Tuple[str, List[Individual], List[str]]] = {}
            room = min(self.batch_size, budget - self.spent)
            while len(batch) < room:
                operator, parents = self._plan(pending)
                if not state.initialised:
                    init_calls += 1
                guidance = self._guidance()
                proposals = self.propose(operator, parents, guidance)
                if proposals is None:
                    return "LLM proposer is down; the population resumes from disk."
                for candidate in proposals:
                    if candidate.signature() in meta:
                        continue
                    batch.append(candidate)
                    pending.append(
                        Individual(key=candidate.signature(), expression=candidate.expression, family=candidate.family)
                    )
                    meta[candidate.signature()] = (operator, parents, list(guidance.get("insight_texts") or []))
                    break
                else:
                    break
                if not state.initialised and len(state.population) + len(batch) >= self.population_size:
                    break
            if not batch:
                empty_batches += 1
                if empty_batches >= self.max_empty_batches:
                    if not state.initialised and state.population:
                        self._close_generation()  # start breeding with what initialisation produced
                        state.save()
                        empty_batches = 0
                        continue
                    return "No candidate passed the offline filters."
                continue
            empty_batches = 0
            self.batches += 1
            records, stop_reason = self.evaluate(batch)
            by_key = {str(pipeline.record_signature(record) or ""): record for record in records}
            credited: List[Dict[str, Any]] = []
            for candidate in batch:
                key = candidate.signature()
                record = by_key.get(key)
                operator, parents, insights = meta[key]
                if record is None:
                    continue
                if key not in state.seen_keys:
                    state.seen_keys.append(key)
                child = individual_from_record(record, margins=self.margins_of(record), generation=state.generation)
                reference = state.population if state.initialised else state.population + state.offspring
                credit = pareto_credit(child, reference)
                deltas = self._deltas(child, parents) if child is not None else None
                if child is not None or not stop_reason:
                    # A batch rejected for credentials says nothing about its ideas.
                    state.insights.credit(insights, credit, deltas)
                if child is None:
                    credited.append({"operator": operator, "credit": credit, "status": record.get("status")})
                    continue
                self.spent += 1
                if state.initialised:
                    state.offspring.append(child)
                else:
                    state.population.append(child)
                credited.append(
                    {
                        "operator": operator,
                        "alpha_id": child.alpha_id,
                        "credit": round(credit, 4),
                        "sharpe": child.metrics.get("sharpe"),
                        "returns": child.metrics.get("returns"),
                        "turnover": child.metrics.get("turnover"),
                    }
                )
            state.save()
            if self.on_batch is not None:
                self.on_batch(
                    {
                        "generation": state.generation,
                        "regime": state.navigator.regime,
                        "children": credited,
                        "population": len(state.population),
                        "offspring": len(state.offspring),
                        "hypervolume": round(population_hypervolume(state.population), 6),
                    }
                )
            if stop_reason:
                return stop_reason
            if not state.initialised and init_calls >= 3 * self.population_size and state.population:
                self._close_generation()
                state.save()
        return "Budget exhausted."


def candidate_metadata(
    candidate: pipeline.Candidate, *, operator: str, parents: Sequence[Individual], guidance: Dict[str, Any]
) -> Dict[str, Any]:
    """Metadata stored with each record so a killed run can be recovered."""
    return {
        **dict(candidate.metadata or {}),
        "operator": operator,
        "parents": [member.key for member in parents][:5],
        "insights": list(guidance.get("insight_texts") or []),
        "regime": guidance.get("regime"),
        "generation": guidance.get("generation"),
        "mo_hifo_state": guidance.get("state_id"),
    }
