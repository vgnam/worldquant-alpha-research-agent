"""Field-level collision risk of a candidate against the account's alpha pool.

Why this exists
---------------
``SELF_CORRELATION`` is computed on the PnL series and is not predictable from
expression similarity: measured over 15 evaluated alphas the rank correlation
between the two was -0.14. What *does* survive the numeric comparison is the
data: two alphas built on the same field family (``scl12_buzz`` vs
``scl12_buzz``) move the same way almost by construction, however differently
the expression is wrapped.

The live evidence: 284 simulations on USA/TOP3000 produced exactly one alpha
family good enough to clear the quality gates, and all eight wrappings of it
failed ``SELF_CORRELATION`` at 0.743-0.768 against a 0.70 limit — against an
account pool of 1100 alphas whose largest field groups were ``anl4`` (376) and
``scl12`` (127), the exact groups those seeds were built from.

So before a seed is simulated, this module answers: *how much of this seed's
data does the account already own?* A seed whose fields appear in many pool
alphas is a likely duplicate of something already submitted; a seed whose
fields appear in none is the cheapest source of a genuinely uncorrelated
alpha.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable, Optional, Sequence

from alpha_agent.expr_ast import BUILTIN_IDENTIFIERS, collect_identifiers, try_parse
from alpha_agent.families import field_unit

__all__ = [
    "collision_risk",
    "expression_fields",
    "field_swaps",
    "field_usage",
    "order_seeds_by_collision",
]

#: Risk assigned to a seed whose fields cannot be read (unparsable expression).
#: Mid-pack: neither promoted nor buried.
UNKNOWN_RISK = 0.5


def expression_fields(expression: str) -> set[str]:
    """Data fields an expression reads, without grouping builtins.

    Operators are calls, not leaves, so only field names and the grouping
    identifiers (``industry``, ``sector``, ...) come back; the grouping names
    are dropped because every alpha using them would otherwise look like a
    duplicate of every other.
    """
    node = try_parse(str(expression or ""))
    if node is None:
        return set()
    try:
        identifiers = collect_identifiers(node)
    except Exception:  # a shape the collector does not expect
        return set()
    return {name for name in identifiers if name not in BUILTIN_IDENTIFIERS}


def field_usage(expressions: Iterable[str]) -> tuple[Counter[str], int]:
    """How many pool alphas read each field, and how many could be read."""
    usage: Counter[str] = Counter()
    total = 0
    for expression in expressions:
        fields = expression_fields(expression)
        if not fields:
            continue
        total += 1
        for field in fields:
            usage[field] += 1
    return usage, total


def collision_risk(
    expression: str,
    usage: Counter[str],
    total: int,
    *,
    similarity: float = 0.0,
    fields: Optional[set[str]] = None,
) -> float:
    """``0.0`` = no field of this seed is used by the pool, ``1.0`` = all of it.

    The score is the *worst* single field share, not the mean: one field the
    account is saturated in is enough to sink ``SELF_CORRELATION``, so a seed
    that mixes one rare field with one common field is still risky.
    ``similarity`` (expression-space, from ``NoveltyPool``) can only raise the
    score: it is a weaker signal, but an exact re-wrap is still a re-wrap.
    """
    names = expression_fields(expression) if fields is None else set(fields)
    if not names or total <= 0:
        return max(UNKNOWN_RISK, float(similarity))
    worst = max(usage.get(name, 0) for name in names)
    return max(min(1.0, worst / float(total)), min(1.0, float(similarity)))


def field_swaps(
    expression: str,
    *,
    usage: Counter[str],
    total: int,
    candidates: Iterable[str],
    limit: int = 4,
    min_share: float = 0.01,
    max_share: float = 0.25,
) -> list[str]:
    """Same formula, the field the pool is saturated in swapped out.

    This is the cheap way out of the live dead end: the only structure that
    cleared the quality gates (``ts_mean(-scl12_buzz, 8) + rank((ts_mean(close,
    24) - close) / close)``) failed ``SELF_CORRELATION`` because the account
    pool already owns that data. Rewriting the formula from scratch throws the
    Sharpe away; replacing the most-covered field with a *proven but rare*
    field of the same data group keeps the structure and drops the collision.

    The band matters. "Least used field" alone picks sparse corporate-action
    data — the first live run of this rule produced
    ``ts_mean(dividend, 20)`` and ``(x - split) / split``. Requiring that the
    pool already uses the field in ``min_share``..``max_share`` of its alphas
    means the replacement is data another alpha has proven usable, but not data
    the account is already saturated in.

    Only a field of the same unit is offered (:func:`field_unit`): price for
    price, volume for volume. The same *data group* was not enough: ``pv``
    holds prices and market cap alike, and the swap produced
    ``(cap - open) / (high - low)`` from Alpha101 #101. A field whose unit is
    unknown is not swapped at all.
    """
    fields = expression_fields(expression)
    if not fields or total <= 0:
        return []
    worst = max(fields, key=lambda name: (usage.get(name, 0), name))
    unit = field_unit(worst)
    if unit is None:
        return []
    low = max(1, int(round(float(min_share) * total)))
    high = max(low, int(round(float(max_share) * total)))
    pool = [
        name
        for name in candidates
        if name not in fields and field_unit(name) == unit and low <= usage.get(name, 0) <= high
    ]
    pool.sort(key=lambda name: (usage.get(name, 0), name))
    pattern = re.compile(rf"\b{re.escape(worst)}\b")
    variants: list[str] = []
    for replacement in pool:
        variant = pattern.sub(replacement, expression)
        if variant != expression:
            variants.append(variant)
        if len(variants) >= max(0, int(limit)):
            break
    return variants


def order_seeds_by_collision(
    seeds: Sequence[object],
    risk_of: Callable[[object], float],
    *,
    family_of: Optional[Callable[[object], str]] = None,
) -> list:
    """Round-robin over families, least-colliding seed first within each round.

    Family first keeps the queue structurally diverse (twenty window variants
    of one idea is what produced eight identical quality-passing alphas), and
    risk inside the round picks the seed least likely to be a duplicate. Each
    family is sorted by risk *before* the rounds are dealt, so the safest
    member of every family represents it in round 0.
    """
    if family_of is None:
        def family_of(candidate: object) -> str:  # type: ignore[misc]
            return str(getattr(candidate, "family", "") or "")

    risk: dict[int, float] = {id(seed): float(risk_of(seed)) for seed in seeds}
    families: dict[str, list] = {}
    order: list[str] = []
    for seed in seeds:
        family = family_of(seed)
        if family not in families:
            families[family] = []
            order.append(family)
        families[family].append(seed)
    for family in order:
        families[family].sort(key=lambda seed: risk[id(seed)])

    ranked: list = []
    depth = max((len(bucket) for bucket in families.values()), default=0)
    for round_index in range(depth):
        row = [families[family][round_index] for family in order if len(families[family]) > round_index]
        row.sort(key=lambda seed: risk[id(seed)])
        ranked.extend(row)
    return ranked
