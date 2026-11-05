"""Pareto-front primitives for multi-objective MCTS.

Design choices
--------------
* **Raw metrics, no gate thresholds.** The default objectives are the
  simulation's own metrics (Sharpe, turnover, margin) plus expression
  divergence from the account's alphas. Nothing is measured against a
  platform limit: the search looks for alphas no other alpha beats on every
  objective at once, not for alphas that clear a bar.
* **Fixed scaling ranges, fixed reference.** Each objective is mapped
  linearly from ``(worst, best)`` onto ``[0, 1]``, and the hypervolume
  reference point is the origin. The ranges only set the scale; a
  data-dependent nadir would make hypervolume incomparable across runs.
* **Out of range is rejected, not clamped** (``strict=True``, what the tree
  uses). A clamped coordinate silently turns into a fake value that is then
  dropped from every front, which is how ``turnover`` above 1.0 used to make
  a fifth of the population invisible. The ranges below are therefore the
  physically possible one for each metric, not a quality gate, and anything
  outside them is counted as a data problem instead of being folded in.
* **Exact hypervolume by slicing.** Fronts are capped at a few dozen points
  and two to four objectives, so the recursive slicing algorithm (HSO) is
  exact and cheap (about 1 ms for 24 points in 4-D); no Monte-Carlo estimate
  is needed.
* **Epsilon archive, truncated by contribution.** A candidate within
  ``epsilon`` of an existing point on every objective is a near-duplicate and
  is rejected; that is what stops ten ``ts_mean`` window variants of one idea
  from filling the front. When the front is still over capacity, the point
  with the smallest exclusive hypervolume contribution is dropped (SMS-EMOA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "OBJECTIVE_BOUNDS",
    "DEFAULT_OBJECTIVES",
    "FrontEntry",
    "ParetoFront",
    "dominates",
    "hypervolume",
    "is_feasible",
    "normalize_objectives",
]

#: ``(worst, best)`` scaling range of each objective; ``worst > best`` means
#: lower is better. These are the *possible* ranges of the metrics, deliberately
#: much wider than the platform gates, so a real alpha is never squeezed onto a
#: boundary (which would drop it from every front).
#: ``sharpe``, ``fitness``, ``turnover``, ``returns``, ``drawdown``, ``margin``:
#: simulation metrics. ``margin`` (profit per dollar traded) is the most
#: independent quality axis: measured over 284 live results, sharpe/fitness
#: correlate at r=0.94 and sharpe/returns at 0.94, while sharpe/margin is 0.37.
#: ``divergence``: 1 - max expression similarity to the account's alphas.
#: ``quality``: worst quality-gate margin; ``correlation``: worst SELF/PROD
#: correlation margin (0.0 when not reported). Both are relative to platform
#: limits and kept only as opt-in objectives.
OBJECTIVE_BOUNDS: dict[str, tuple[float, float]] = {
    "sharpe": (-2.0, 4.0),
    "fitness": (-1.0, 4.0),
    "turnover": (3.0, 0.0),
    "returns": (-0.30, 0.30),
    "drawdown": (2.0, 0.0),
    "margin": (-0.02, 0.02),
    "divergence": (0.0, 1.0),
    "quality": (-1.0, 1.0),
    "correlation": (-1.0, 1.0),
}

#: No redundant pair: ``fitness`` (r=0.94 with sharpe) and ``returns``
#: (r=0.94) are opt-in, not defaults.
DEFAULT_OBJECTIVES = ("sharpe", "turnover", "margin", "divergence")


def normalize_objectives(
    values: Mapping[str, float],
    names: Sequence[str],
    *,
    strict: bool = False,
) -> Optional[tuple[float, ...]]:
    """Objective vector in ``[0, 1]^d``, ``1`` at the best end of each range.

    None when any objective is missing. With ``strict`` a value outside its
    ``OBJECTIVE_BOUNDS`` also yields None, so the caller can count it as a data
    problem instead of silently clamping it onto the boundary (where it would
    contribute no hypervolume and look like a legitimate Pareto point).
    """
    vector: list[float] = []
    for name in names:
        raw = values.get(name)
        if raw is None:
            return None
        worst, best = OBJECTIVE_BOUNDS[name]
        fraction = (float(raw) - worst) / (best - worst)
        if strict:
            if not 0.0 <= fraction <= 1.0:
                return None
        else:
            fraction = max(0.0, min(1.0, fraction))
        vector.append(fraction)
    return tuple(vector)


def is_feasible(vector: Sequence[float]) -> bool:
    """Strictly inside the reference box, i.e. it adds hypervolume."""
    return all(value > 0.0 for value in vector)


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """``a`` is at least as good everywhere and strictly better somewhere."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _weakly_dominates(a: Sequence[float], b: Sequence[float], epsilon: float = 0.0) -> bool:
    return all(x + epsilon >= y for x, y in zip(a, b))


def hypervolume(points: Iterable[Sequence[float]]) -> float:
    """Volume dominated by ``points`` above the origin."""
    usable = [tuple(point) for point in points if point and all(value > 0.0 for value in point)]
    if not usable:
        return 0.0
    return _slice_volume(usable)


def _slice_volume(points: list[tuple[float, ...]]) -> float:
    dims = len(points[0])
    if dims == 1:
        return max(point[0] for point in points)
    if dims == 2:
        volume = 0.0
        best_y = 0.0
        for x, y in sorted(points, key=lambda point: (-point[0], -point[1])):
            if y > best_y:
                volume += x * (y - best_y)
                best_y = y
        return volume
    # Slice along the last axis: between consecutive heights, the cross-section
    # is the (d-1)-dimensional volume of every point at least that high.
    ordered = sorted(points, key=lambda point: -point[-1])
    volume = 0.0
    for index, point in enumerate(ordered):
        below = ordered[index + 1][-1] if index + 1 < len(ordered) else 0.0
        height = point[-1] - below
        if height > 0.0:
            volume += height * _slice_volume([other[:-1] for other in ordered[: index + 1]])
    return volume


@dataclass(frozen=True)
class FrontEntry:
    key: str
    vector: tuple[float, ...]


class ParetoFront:
    """Bounded epsilon-Pareto archive of ``(key, normalised vector)`` entries."""

    def __init__(self, *, capacity: int = 24, epsilon: float = 0.0) -> None:
        self.capacity = max(1, int(capacity))
        self.epsilon = max(0.0, float(epsilon))
        self.entries: list[FrontEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> set[str]:
        return {entry.key for entry in self.entries}

    def vectors(self) -> list[tuple[float, ...]]:
        return [entry.vector for entry in self.entries]

    def insert(self, key: str, vector: Sequence[float]) -> bool:
        """Add ``vector`` unless it is dominated, a near-duplicate, or on the box.

        Re-inserting an existing key replaces its vector. Returns whether the
        entry is on the front afterwards. A point with a coordinate on (or
        outside) the reference box adds no hypervolume, so it is not stored at
        all: keeping it would make ``len(front)`` disagree with the front the
        run reports and feeds to crossover.
        """
        vector = tuple(float(value) for value in vector)
        self.entries = [entry for entry in self.entries if entry.key != key]
        if not all(value > 0.0 for value in vector):
            return False
        for entry in self.entries:
            if _weakly_dominates(entry.vector, vector):
                return False
            if not dominates(vector, entry.vector) and _weakly_dominates(entry.vector, vector, self.epsilon):
                return False
        self.entries = [entry for entry in self.entries if not dominates(vector, entry.vector)]
        self.entries.append(FrontEntry(key, vector))
        while len(self.entries) > self.capacity:
            contributions = self.contributions()
            victim = min(self.entries, key=lambda entry: (contributions[entry.key], entry.key))
            self.entries.remove(victim)
        return any(entry.key == key for entry in self.entries)

    def hypervolume(self) -> float:
        return hypervolume(self.vectors())

    def contributions(self) -> dict[str, float]:
        """Exclusive hypervolume each entry adds to the rest of the front."""
        total = self.hypervolume()
        return {
            entry.key: total - hypervolume(other.vector for other in self.entries if other is not entry)
            for entry in self.entries
        }

    def exclusive_volume(self, keys: Iterable[str]) -> float:
        """Hypervolume lost if every entry in ``keys`` were removed."""
        owned = set(keys)
        if not owned & self.keys():
            return 0.0
        rest = [entry.vector for entry in self.entries if entry.key not in owned]
        return self.hypervolume() - hypervolume(rest)
