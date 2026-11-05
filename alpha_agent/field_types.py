"""Data-field types, and the rule they imply.

The platform classifies every data field as ``MATRIX``, ``VECTOR`` or ``GROUP``.
A ``VECTOR`` field (news items, analyst estimates — anything that arrives as
several values per instrument-day) cannot be fed to an ordinary operator: it
must be reduced by a ``vec_*`` aggregation first. The platform says so only at
simulation time:

    Operator ts_decay_linear does not support event inputs
    Operator less does not support event inputs

Measurements from this repo's own runs: 27 of 311 simulations (8.7%) died this
way historically, and in one 100-simulation run 3 of the first 5 did — more than
half of the budget spent to learn something ``/data-fields`` could have said for
free. So the check runs offline, before a simulation is paid for.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from alpha_agent.expr_ast import ExprNode

__all__ = ["VECTOR_REDUCERS", "vector_violations"]

#: Operators that reduce a VECTOR field to a MATRIX. A VECTOR may only appear
#: as a direct argument of one of these.
VECTOR_REDUCERS = frozenset(
    {
        "vec_avg",
        "vec_sum",
        "vec_max",
        "vec_min",
        "vec_count",
        "vec_stddev",
        "vec_ir",
        "vec_skewness",
        "vec_kurtosis",
        "vec_powersum",
        "vec_norm",
    }
)

_KIND = "VECTOR"


def vector_violations(node: ExprNode, field_types: Mapping[str, str]) -> list[str]:
    """Calls that pass a ``VECTOR`` field straight into an ordinary operator.

    ``field_types`` maps field id -> ``type``; unknown fields are not judged,
    because a missing entry means "we do not know", not "it is a matrix".
    """
    if not field_types:
        return []
    problems: list[str] = []

    def walk(current: ExprNode, parent_op: Optional[str]) -> None:
        if current.is_leaf:
            name = current.leaf or ""
            if field_types.get(name) != _KIND:
                return
            if parent_op is not None and parent_op in VECTOR_REDUCERS:
                return
            where = f"inside {parent_op}()" if parent_op else "at the top level"
            problems.append(
                f"{name} is a VECTOR field and cannot be used {where}; "
                f"reduce it first, e.g. vec_avg({name}, 20)"
            )
            return
        for argument in current.args:
            walk(argument, current.op)

    walk(node, None)
    return problems


def filter_matrix_fields(candidates: Iterable[str], field_types: Mapping[str, str]) -> list[str]:
    """Keep only fields that are usable as ordinary numeric data.

    Used when proposing a field substitution: offering a ``VECTOR`` replacement
    would recreate exactly the failure this module exists to prevent.
    """
    names = list(candidates)
    if not field_types:
        return names
    return [name for name in names if field_types.get(name, "MATRIX") != _KIND]
