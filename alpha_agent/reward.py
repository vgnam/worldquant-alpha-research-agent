"""Gate-margin reward for the hybrid search.

Why margins instead of the legacy score
---------------------------------------
The legacy pipeline collapses everything into one heuristic number
(`score_result`) plus hand-tuned bonuses. That number is not comparable across
configurations: the platform's limits change per account and universe (LOW_
SHARPE was observed at 1.58), and a single scalar hides *which* gate is the
binding constraint. The Alpha Jungle paper instead evaluates a candidate along
several dimensions and targets the weakest one; this module produces exactly
that vector.

Three quantities are produced per record:

* ``gate_margins`` - signed relative distance to each check's limit
  (positive = passing with room, negative = failing by that fraction).
* ``scalar_reward`` - bounded comparison key for UCT/Q values.
* ``weakest_dimension`` - softmax-sampled dimension to attack next.

Honesty rules baked in
----------------------
* A ``PENDING`` check is never treated as a pass and never contributes a
  margin; it is tracked separately because the platform blocks submission on it.
* A check whose ``limit``/``value`` are absent contributes no margin, but still
  counts toward the pass fraction.
* Weights here are only for ranking. They carry no absolute meaning and must
  not be reported as a quality metric.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "QUALITY_DIMENSIONS",
    "NOVELTY_DIMENSIONS",
    "SOFT_DIMENSIONS",
    "LOWER_IS_BETTER",
    "GateAssessment",
    "assess_record",
    "gate_margins",
    "scalar_reward",
    "weakest_dimension",
    "percentile",
    "clamp",
]

QUALITY_DIMENSIONS = (
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "LOW_2Y_SHARPE",
    "IS_LADDER_SHARPE",
)

NOVELTY_DIMENSIONS = ("SELF_CORRELATION", "PROD_CORRELATION")

SOFT_DIMENSIONS = ("CLUSTER_TEST", "POWER_POOL_CORRELATION")

#: Checks where a *lower* value is better.
LOWER_IS_BETTER = frozenset(
    {"HIGH_TURNOVER", "CONCENTRATED_WEIGHT", "SELF_CORRELATION", "PROD_CORRELATION"}
    | set(SOFT_DIMENSIONS)
)

ALL_DIMENSIONS = QUALITY_DIMENSIONS + NOVELTY_DIMENSIONS + SOFT_DIMENSIONS


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def percentile(value: float, reference: Sequence[float]) -> float:
    """Fraction of ``reference`` strictly below ``value`` (paper eq. 6/7)."""
    if not reference:
        return 0.5
    below = sum(1 for item in reference if item < value)
    return below / len(reference)


@dataclass(frozen=True)
class GateAssessment:
    """Everything the tree and the archive need to know about one evaluation."""

    margins: dict[str, float] = field(default_factory=dict)
    pending: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    passed: tuple[str, ...] = ()
    applicable: int = 0
    passed_count: int = 0
    gate_fraction: float = 0.0
    scalar: float = 0.0
    quality_ready: bool = False
    submit_ready: bool = False

    @property
    def quality_margin(self) -> float:
        values = [self.margins[name] for name in QUALITY_DIMENSIONS if name in self.margins]
        return min(values) if values else 0.0

    @property
    def novelty_margin(self) -> float:
        values = [self.margins[name] for name in NOVELTY_DIMENSIONS if name in self.margins]
        return min(values) if values else 0.0

    def to_dict(self) -> dict:
        return {
            "margins": {name: round(value, 6) for name, value in sorted(self.margins.items())},
            "pending": list(self.pending),
            "failed": list(self.failed),
            "gate_fraction": round(self.gate_fraction, 6),
            "quality_margin": round(self.quality_margin, 6),
            "novelty_margin": round(self.novelty_margin, 6),
            "scalar": round(self.scalar, 6),
            "quality_ready": self.quality_ready,
            "submit_ready": self.submit_ready,
        }


def _margin_for(check: dict) -> Optional[float]:
    name = str(check.get("name") or "")
    limit = check.get("limit")
    value = check.get("value")
    if limit is None or value is None:
        return None
    try:
        limit_value = float(limit)
        actual = float(value)
    except (TypeError, ValueError):
        return None
    scale = abs(limit_value)
    if scale < 1e-12:
        return None
    if name in LOWER_IS_BETTER:
        return clamp((limit_value - actual) / scale)
    return clamp((actual - limit_value) / scale)


def gate_margins(record: dict, *, include_soft: bool = True) -> dict[str, float]:
    """Signed relative distance to each check limit, in ``[-1, 1]``."""
    diagnostics = record.get("check_diagnostics")
    margins: dict[str, float] = {}
    if not isinstance(diagnostics, list):
        return margins
    allowed = set(ALL_DIMENSIONS) if include_soft else set(QUALITY_DIMENSIONS) | set(NOVELTY_DIMENSIONS)
    for check in diagnostics:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        if name not in allowed:
            continue
        if str(check.get("result") or "") not in {"PASS", "FAIL"}:
            continue
        margin = _margin_for(check)
        if margin is not None:
            margins[name] = margin
    return margins


def _applicable_diagnostics(record: dict) -> list[dict]:
    diagnostics = record.get("check_diagnostics")
    if not isinstance(diagnostics, list):
        return []
    return [
        check
        for check in diagnostics
        if isinstance(check, dict) and str(check.get("name") or "") in ALL_DIMENSIONS
    ]


def assess_record(record: dict, *, include_soft: bool = True) -> GateAssessment:
    """Turn one evaluation record into margins, gate fraction and reward."""
    margins = gate_margins(record, include_soft=include_soft)
    checks = _applicable_diagnostics(record)

    passed: list[str] = []
    failed: list[str] = []
    pending: list[str] = []
    for check in checks:
        name = str(check.get("name") or "")
        result = str(check.get("result") or "").upper()
        if result == "PASS":
            passed.append(name)
        elif result == "FAIL":
            failed.append(name)
        elif result == "PENDING":
            pending.append(name)

    # Applicability: prefer the live diagnostics; fall back to the recorded
    # pass/fail lists so hand-made records still produce a sensible fraction.
    applicable = len(checks) if checks else len(
        set(record.get("failed_checks") or []) | set(record.get("failed_blocking_checks") or [])
    )
    passed_count = len(passed) if checks else max(0, applicable - len(set(record.get("failed_checks") or [])))

    gate_fraction = (passed_count / applicable) if applicable else 0.0

    quality_values = [margins[name] for name in QUALITY_DIMENSIONS if name in margins]
    novelty_values = [margins[name] for name in NOVELTY_DIMENSIONS if name in margins]
    quality_min = min(quality_values) if quality_values else 0.0
    novelty_min = min(novelty_values) if novelty_values else 0.0

    quality_ready = bool(record.get("quality_checks_ready"))
    submit_ready = bool(record.get("precheck_submit_ready"))

    scalar = (
        1000.0 * gate_fraction
        + 500.0 * quality_min
        + 800.0 * novelty_min
        + (1000.0 if submit_ready else 0.0)
        - (50.0 * len(pending))
    )

    return GateAssessment(
        margins=margins,
        pending=tuple(pending),
        failed=tuple(failed),
        passed=tuple(passed),
        applicable=applicable,
        passed_count=passed_count,
        gate_fraction=gate_fraction,
        scalar=scalar,
        quality_ready=quality_ready,
        submit_ready=submit_ready,
    )


def scalar_reward(record: dict, *, include_soft: bool = True) -> float:
    """Convenience wrapper used by the tree."""
    return assess_record(record, include_soft=include_soft).scalar


def weakest_dimension(
    margins: dict[str, float],
    *,
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
    preferred: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Softmax-sample the dimension with the least headroom.

    Mirrors Alpha Jungle eq. 3: sample proportional to ``softmax((1 - e)/T)``
    so weak dimensions are attacked most often without being deterministic,
    which would collapse the search onto one axis.
    """
    candidates = {
        name: value
        for name, value in margins.items()
        if preferred is None or name in set(preferred)
    }
    if not candidates:
        return None
    temperature = max(1e-3, float(temperature))
    names = sorted(candidates)
    weights = [math.exp((1.0 - clamp(candidates[name])) / temperature) for name in names]
    total = sum(weights)
    if total <= 0:
        return names[0]
    picker = rng or random.Random()
    threshold = picker.random() * total
    cumulative = 0.0
    for name, weight in zip(names, weights):
        cumulative += weight
        if threshold <= cumulative:
            return name
    return names[-1]
