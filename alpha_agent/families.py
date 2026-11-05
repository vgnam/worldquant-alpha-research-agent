"""Idea families derived from the data groups an expression reads.

A family label is what the per-family elite cap and the lineage relabelling
use to tell ideas apart. It is the set of coarse datasets an alpha combines
(``pv``, ``scl12``, ``fundamental`` ...), not a hash of its exact fields, so
``close``/``open`` variants of one idea share a family while unrelated
datasets do not.
"""

from __future__ import annotations

import re

from alpha_agent.expr_ast import collect_identifiers, try_parse

__all__ = ["data_group", "data_groups", "field_unit", "lineage_family"]

#: Grouping keys of group_* operators parse as identifiers but are not data.
_GROUP_KEYS = frozenset({"market", "sector", "industry", "subindustry", "country", "exchange", "currency"})
_PRICE_VOLUME = frozenset(
    {"open", "high", "low", "close", "vwap", "volume", "returns", "cap", "sharesout", "dividend", "split"}
)
#: Vendor dataset prefixes look like ``scl12_``, ``anl4_``, ``fnd6_``, ``mdl175_``.
_DATASET_PREFIX = re.compile(r"^([a-z]+\d+)_")
#: Unit of each price-volume field. ``pv`` mixes prices, share counts, dollars
#: and returns, so the data group alone let a swap turn ``vwap - close`` into
#: ``vwap - cap`` (a price minus a market cap). ``advN`` is its own class: an
#: N-day average, not a daily volume.
_PV_UNITS = {
    "open": "price",
    "high": "price",
    "low": "price",
    "close": "price",
    "vwap": "price",
    "volume": "volume",
    "returns": "return",
    "cap": "market_cap",
    "sharesout": "shares",
    "dividend": "dividend",
    "split": "split",
}


def data_group(field_id: str) -> str:
    """Coarse dataset of a field id; the summary file carries no dataset metadata."""
    name = field_id.lower()
    match = _DATASET_PREFIX.match(name)
    if match:
        return match.group(1)
    if name in _PRICE_VOLUME or re.fullmatch(r"adv\d+", name):
        return "pv"
    if "volatility" in name or name.startswith(("implied_", "historical_", "parkinson_")):
        return "vol"
    if name.startswith("news_"):
        return "news"
    if "estimate" in name or "analyst" in name or "revision" in name:
        return "analyst"
    if name.startswith(("beta_", "correlation_", "unsystematic_", "systematic_")):
        return "risk"
    return "fundamental"


def field_unit(field_id: str) -> str | None:
    """Unit class two fields must share to be swapped for one another.

    Price-volume fields have an explicit unit. A vendor dataset (``scl12_``,
    ``anl4_`` ...) is its own class: the id carries no unit, and fields of one
    dataset are the like-for-like replacement the pool-aware swap was built
    for. Anything else (unprefixed fundamentals such as ``assets`` next to
    ``roe``) is unknown, and unknown never matches.
    """
    name = field_id.lower()
    if name in _PV_UNITS:
        return _PV_UNITS[name]
    if re.fullmatch(r"adv\d+", name):
        return "adv"
    match = _DATASET_PREFIX.match(name)
    if match:
        return f"dataset:{match.group(1)}"
    return None


def data_groups(expression: str) -> tuple[str, ...]:
    node = try_parse(expression)
    if node is None:
        return ()
    return tuple(sorted({data_group(name) for name in collect_identifiers(node) if name not in _GROUP_KEYS}))


def lineage_family(expression: str) -> str:
    """Family label for an idea, from the data groups it combines."""
    groups = data_groups(expression)
    return ("+".join(groups) or "none")[:24]
