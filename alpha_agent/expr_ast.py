"""FASTEXPR expression analysis used by the hybrid search.

Why this module exists
----------------------
The legacy pipeline treats an expression as an opaque string and measures
novelty with token overlap. That is exploitable: wrapping a formula in
`rank` / `group_rank` / `group_neutralize` changes every token while leaving
the traded signal unchanged. A live run produced four candidates with
identical Sharpe (1.70), identical fitness (1.06) and identical
SELF_CORRELATION (0.7673) purely because of wrapper stacking, wasting half of
the simulation budget.

This module gives the search three offline checks that never cost a
simulation:

1. `parse_expression` / `try_parse`      - real AST instead of a regex.
2. `behavioral_signature`                - "is this the same bet?" key that
                                           collapses cross-sectional wrapper
                                           permutations.
3. `subtree_hashes` / `collect_operators` - frequent-subtree avoidance and
                                           operator whitelist validation.

Design notes
------------
* The module is dependency free (standard library only). Settings passed to
  `behavioral_signature` must already be normalized by the caller.
* Nothing here rewrites what is sent to the API. Normalization is used only
  for signatures, dedupe and ranking.
* `behavioral_signature` is deliberately *more aggressive* than the exact
  signature: it treats `group_rank(group_neutralize(rank(x), industry),
  industry)` and `group_neutralize(rank(group_rank(rank(x), industry)),
  industry)` as the same behavioral class, because every operator composing
  that chain is a monotone transform of the same base signal. That is the
  right trade-off for skipping expensive evaluations, but it must never be
  used to claim two expressions are mathematically equal.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

__all__ = [
    "ExprNode",
    "ExpressionParseError",
    "parse_expression",
    "try_parse",
    "canonical_form",
    "strip_redundant_wrappers",
    "cross_sectional_profile",
    "behavioral_signature",
    "expression_identity",
    "subtree_hashes",
    "collect_operators",
    "collect_call_operators",
    "collect_identifiers",
    "node_size",
    "node_depth",
    "ast_bigrams",
    "max_window",
    "soft_nesting_count",
    "structural_distance",
    "validate_against",
    "SYMBOL_OPS",
    "CROSS_SECTIONAL_OPS",
]

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

SYMBOL_OPS = frozenset(
    {"+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!=", "&&", "||", "neg"}
)
COMPARISON_OPS = frozenset({"<", ">", "<=", ">=", "==", "!="})
LOGICAL_OPS = frozenset({"&&", "||"})
BINARY_OPS = frozenset({"+", "-", "*", "/"}) | COMPARISON_OPS | LOGICAL_OPS

#: Cross-sectional normalizers: monotone transforms of a signal within a day
#: (optionally within a group). A run of these on the same base is one bet.
CROSS_SECTIONAL_OPS = frozenset(
    {"rank", "zscore", "scale", "sign", "group_rank", "group_neutralize", "group_zscore"}
)

#: Unary operators that are idempotent, so `f(f(x))` can be collapsed for
#: signature purposes. Only applied with arity 1 to stay safe.
_IDEMPOTENT_UNARY = frozenset({"rank", "zscore", "scale", "sign"})

#: Group operators where `g(g(x, k), k) == g(x, k)`.
_IDEMPOTENT_GROUP = frozenset({"group_rank", "group_neutralize", "group_zscore"})

#: Time-series operators that are *not* safe to collapse, but whose nesting
#: with an identical window is a strong homogeneity signal. Used for penalty
#: only, never for dedupe.
_SOFT_NESTED_TS = frozenset(
    {"ts_rank", "ts_decay_linear", "ts_mean", "ts_sum", "ts_std_dev", "ts_zscore"}
)

_NUMBER_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")


class ExpressionParseError(ValueError):
    """Raised when an expression is outside the supported FASTEXPR subset."""


@dataclass(frozen=True)
class ExprNode:
    """A FASTEXPR node.

    Leaves carry ``leaf`` (an identifier or a numeric literal) and no ``op``.
    Internal nodes carry ``op`` and ``args``. Unary minus is stored as
    ``op="neg"`` with a single argument.
    """

    op: Optional[str] = None
    args: tuple["ExprNode", ...] = ()
    leaf: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.op is None

    @property
    def is_number(self) -> bool:
        return self.is_leaf and bool(_NUMBER_RE.match(self.leaf or ""))


# ---------------------------------------------------------------------------
# Tokenizer / parser
# ---------------------------------------------------------------------------

_TWO_CHAR_OPS = ("<=", ">=", "==", "!=", "&&", "||")
_ONE_CHAR = "+-*/(),<>"


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            tokens.append(("NUM", text[i:j]))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("NAME", text[i:j]))
            i = j
            continue
        two = text[i : i + 2]
        if two in _TWO_CHAR_OPS:
            tokens.append(("OP", two))
            i += 2
            continue
        if ch in _ONE_CHAR:
            tokens.append(("OP", ch))
            i += 1
            continue
        raise ExpressionParseError(
            f"Unexpected character {ch!r} at position {i} in {text[:80]!r}"
        )
    tokens.append(("EOF", ""))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.pos = 0

    # -- token helpers ------------------------------------------------------
    def peek(self) -> tuple[str, str]:
        return self.tokens[self.pos]

    def advance(self) -> tuple[str, str]:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, kind: str, value: Optional[str] = None) -> tuple[str, str]:
        token = self.peek()
        if token[0] != kind or (value is not None and token[1] != value):
            raise ExpressionParseError(
                f"Expected {value or kind} but found {token[1]!r} in {self.text[:80]!r}"
            )
        return self.advance()

    def _is_op(self, *values: str) -> bool:
        kind, value = self.peek()
        return kind == "OP" and value in values

    # -- grammar ------------------------------------------------------------
    def parse(self) -> ExprNode:
        node = self.parse_or()
        kind, value = self.peek()
        if kind != "EOF":
            raise ExpressionParseError(
                f"Trailing token {value!r} at position {self.pos} in {self.text[:80]!r}"
            )
        return node

    def parse_or(self) -> ExprNode:
        node = self.parse_and()
        while self._is_op("||"):
            self.advance()
            node = ExprNode(op="||", args=(node, self.parse_and()))
        return node

    def parse_and(self) -> ExprNode:
        node = self.parse_comparison()
        while self._is_op("&&"):
            self.advance()
            node = ExprNode(op="&&", args=(node, self.parse_comparison()))
        return node

    def parse_comparison(self) -> ExprNode:
        node = self.parse_additive()
        while self._is_op(*COMPARISON_OPS):
            _, value = self.advance()
            node = ExprNode(op=value, args=(node, self.parse_additive()))
        return node

    def parse_additive(self) -> ExprNode:
        node = self.parse_multiplicative()
        while self._is_op("+", "-"):
            _, value = self.advance()
            node = ExprNode(op=value, args=(node, self.parse_multiplicative()))
        return node

    def parse_multiplicative(self) -> ExprNode:
        node = self.parse_unary()
        while self._is_op("*", "/"):
            _, value = self.advance()
            node = ExprNode(op=value, args=(node, self.parse_unary()))
        return node

    def parse_unary(self) -> ExprNode:
        if self._is_op("-"):
            self.advance()
            return ExprNode(op="neg", args=(self.parse_unary(),))
        if self._is_op("+"):
            self.advance()
            return self.parse_unary()
        return self.parse_primary()

    def parse_primary(self) -> ExprNode:
        kind, value = self.peek()
        if kind == "NUM":
            self.advance()
            return ExprNode(leaf=value)
        if kind == "NAME":
            self.advance()
            if self._is_op("("):
                self.advance()
                args: list[ExprNode] = []
                if not self._is_op(")"):
                    args.append(self.parse_or())
                    while self._is_op(","):
                        self.advance()
                        args.append(self.parse_or())
                self.expect("OP", ")")
                return ExprNode(op=value, args=tuple(args))
            return ExprNode(leaf=value)
        if kind == "OP" and value == "(":
            self.advance()
            node = self.parse_or()
            self.expect("OP", ")")
            return node
        raise ExpressionParseError(
            f"Unexpected token {value!r} at position {self.pos} in {self.text[:80]!r}"
        )


def parse_expression(text: str) -> ExprNode:
    """Parse a FASTEXPR expression into an :class:`ExprNode`."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionParseError("Expression is empty.")
    return _Parser(text.strip()).parse()


def try_parse(text: str) -> Optional[ExprNode]:
    """Parse, returning ``None`` instead of raising.

    Callers must treat ``None`` as "unknown shape": the candidate can still be
    evaluated, but it is excluded from AST-based filtering.
    """
    try:
        return parse_expression(text)
    except ExpressionParseError:
        return None
    except RecursionError:
        return None


# ---------------------------------------------------------------------------
# Canonical rendering
# ---------------------------------------------------------------------------


def _format_number(raw: str) -> str:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return raw
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def canonical_form(node: ExprNode) -> str:
    """Render a node deterministically, fully parenthesized."""
    if node.is_leaf:
        leaf = node.leaf or ""
        return _format_number(leaf) if _NUMBER_RE.match(leaf) else leaf
    if node.op == "neg" and len(node.args) == 1:
        return f"(-{canonical_form(node.args[0])})"
    if node.op in BINARY_OPS and len(node.args) == 2:
        return f"({canonical_form(node.args[0])} {node.op} {canonical_form(node.args[1])})"
    rendered = ",".join(canonical_form(arg) for arg in node.args)
    return f"{node.op}({rendered})"


# ---------------------------------------------------------------------------
# Structural normalizations
# ---------------------------------------------------------------------------


def _strip_once(node: ExprNode) -> ExprNode:
    if node.is_leaf:
        return node
    rebuilt = ExprNode(op=node.op, args=tuple(_strip_once(arg) for arg in node.args))

    # f(f(x)) -> f(x) for idempotent unary normalizers.
    if rebuilt.op in _IDEMPOTENT_UNARY and len(rebuilt.args) == 1:
        inner = rebuilt.args[0]
        if inner.op == rebuilt.op and len(inner.args) == 1:
            return inner

    # g(g(x, k), k) -> g(x, k) for group normalizers with the same key.
    if rebuilt.op in _IDEMPOTENT_GROUP and len(rebuilt.args) == 2:
        inner, group = rebuilt.args
        if inner.op == rebuilt.op and len(inner.args) == 2 and inner.args[1] == group:
            return inner

    return rebuilt


def strip_redundant_wrappers(node: ExprNode, *, max_passes: int = 8) -> ExprNode:
    """Remove provably redundant nesting (``rank(rank(x))`` etc.).

    Only idempotent cases are removed. Time-series nesting is deliberately
    left alone; it is handled by :func:`soft_nesting_count` as a penalty.
    """
    current = node
    for _ in range(max(1, max_passes)):
        updated = _strip_once(current)
        if updated == current:
            return current
        current = updated
    return current


def cross_sectional_profile(node: ExprNode) -> tuple[frozenset[str], ExprNode]:
    """Split a node into (cross-sectional op chain, underlying base signal).

    Peels leading cross-sectional normalizers, remembering each op and its
    parameters (group key, constants) but *not* their order. Two expressions
    with the same base and the same op multiset are one bet in practice, even
    when the normalizers are permuted, because each step is a monotone
    transform applied to the same underlying signal.
    """
    ops: set[str] = set()
    current = node
    while (not current.is_leaf) and current.op in CROSS_SECTIONAL_OPS:
        first = current.args[0]
        params = ",".join(canonical_form(arg) for arg in current.args[1:])
        ops.add(f"{current.op}({params})" if params else current.op)
        current = first
    return frozenset(ops), current


def _settings_key(settings: Optional[dict]) -> str:
    if not settings:
        return ""
    try:
        return json.dumps(settings, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(sorted(settings.items()))


def _digest(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _canonical_collapsed(node: ExprNode) -> str:
    """Canonical text where every maximal cross-sectional chain is flattened.

    Chains appear anywhere in the tree, not only at the root: in
    ``a + group_rank(rank(x), industry)`` the second operand is a chain. This
    walker therefore collapses at every node it visits.
    """
    ops, base = cross_sectional_profile(node)
    if ops:
        return f"CS[{','.join(sorted(ops))}]({_canonical_collapsed(base)})"

    if node.is_leaf:
        leaf = node.leaf or ""
        return _format_number(leaf) if _NUMBER_RE.match(leaf) else leaf
    if node.op == "neg" and len(node.args) == 1:
        return f"(-{_canonical_collapsed(node.args[0])})"
    if node.op in BINARY_OPS and len(node.args) == 2:
        return (
            f"({_canonical_collapsed(node.args[0])} {node.op} "
            f"{_canonical_collapsed(node.args[1])})"
        )
    rendered = ",".join(_canonical_collapsed(arg) for arg in node.args)
    return f"{node.op}({rendered})"


def expression_identity(node: ExprNode | str) -> str:
    """Exact structural identity, after removing idempotent redundancy."""
    parsed = parse_expression(node) if isinstance(node, str) else node
    return _digest(canonical_form(strip_redundant_wrappers(parsed)))


def behavioral_signature(
    node: ExprNode | str,
    settings: Optional[dict] = None,
) -> str:
    """Signature of the *bet*, not of the text.

    Flattens every cross-sectional chain anywhere in the tree and removes
    idempotent nesting, then folds in the settings. Two candidates sharing this
    signature should be treated as duplicates at proposal time: evaluating the
    second one buys no information.

    This is intentionally more aggressive than :func:`expression_identity`.
    See the module docstring for why, and never use it as a proof of
    mathematical equality.
    """
    parsed = parse_expression(node) if isinstance(node, str) else node
    stripped = strip_redundant_wrappers(parsed)
    payload = json.dumps(
        {"form": _canonical_collapsed(stripped), "settings": _settings_key(settings)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return _digest(payload)


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------


def iter_nodes(node: ExprNode) -> Iterator[ExprNode]:
    yield node
    for arg in node.args:
        yield from iter_nodes(arg)


def node_size(node: ExprNode) -> int:
    return 1 + sum(node_size(arg) for arg in node.args)


def node_depth(node: ExprNode) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(node_depth(arg) for arg in node.args)


def subtree_hashes(node: ExprNode, *, min_size: int = 2) -> set[str]:
    """Hashes of every subtree with at least ``min_size`` nodes."""
    hashes: set[str] = set()
    for child in iter_nodes(node):
        if node_size(child) < min_size:
            continue
        hashes.add(_digest(canonical_form(child)))
    return hashes


def collect_call_operators(node: ExprNode) -> set[str]:
    """Operator *names* used as function calls (excludes symbols)."""
    return {
        child.op
        for child in iter_nodes(node)
        if not child.is_leaf and child.op not in SYMBOL_OPS
    }


def collect_operators(node: ExprNode) -> set[str]:
    """Every operator, including arithmetic and comparison symbols."""
    return {child.op for child in iter_nodes(node) if not child.is_leaf}


def collect_identifiers(node: ExprNode) -> set[str]:
    """Non-numeric leaf identifiers (data fields and group keys)."""
    return {
        child.leaf
        for child in iter_nodes(node)
        if child.is_leaf and child.leaf and not _NUMBER_RE.match(child.leaf)
    }


def ast_bigrams(node: ExprNode) -> set[tuple[str, str]]:
    """Parent-op -> child-op pairs; leaves collapse to ``#leaf``."""
    pairs: set[tuple[str, str]] = set()
    for child in iter_nodes(node):
        if child.is_leaf:
            continue
        parent = child.op or "?"
        for arg in child.args:
            pairs.add((parent, arg.op if not arg.is_leaf else "#leaf"))
    return pairs


def max_window(node: ExprNode) -> Optional[int]:
    """Largest integer literal used as a lookback window argument."""
    windows: list[int] = []
    for child in iter_nodes(node):
        if child.is_leaf or not (child.op or "").startswith("ts_"):
            continue
        for arg in child.args[1:]:
            if arg.is_number:
                try:
                    value = float(arg.leaf or "")
                except ValueError:
                    continue
                if value == int(value) and value > 0:
                    windows.append(int(value))
    return max(windows) if windows else None


def soft_nesting_count(node: ExprNode) -> int:
    """How many times a time-series op is nested inside itself."""
    count = 0
    for child in iter_nodes(node):
        if child.is_leaf or child.op not in _SOFT_NESTED_TS:
            continue
        for arg in child.args:
            if not arg.is_leaf and arg.op == child.op:
                count += 1
    return count


def structural_distance(left: ExprNode, right: ExprNode) -> int:
    """How many tree nodes differ between two expressions.

    Used to prefer *minimal* refinements: a candidate that changes one lookback
    window usually keeps the quality profile of its parent, while a candidate
    that rewrites the whole tree can silently destroy Sharpe. In a live smoke
    run, large rewrites dropped Sharpe from 1.70 to 0.60-1.35, and because the
    platform only computes correlation for alphas that pass the quality gates,
    those simulations taught nothing.
    """
    if left.is_leaf and right.is_leaf:
        return 0 if (left.leaf or "") == (right.leaf or "") else 1
    if left.is_leaf != right.is_leaf:
        return node_size(left) + node_size(right)
    if left.op != right.op:
        return node_size(left) + node_size(right)
    distance = 0
    shared = min(len(left.args), len(right.args))
    for index in range(shared):
        distance += structural_distance(left.args[index], right.args[index])
    for extra in left.args[shared:]:
        distance += node_size(extra)
    for extra in right.args[shared:]:
        distance += node_size(extra)
    return distance


# ---------------------------------------------------------------------------
# Validation against a live catalog
# ---------------------------------------------------------------------------

#: Identifiers the platform accepts that may be absent from a cached field list.
BUILTIN_IDENTIFIERS = frozenset(
    {
        "industry",
        "subindustry",
        "sector",
        "market",
        "country",
        "exchange",
        "true",
        "false",
    }
)


def validate_against(
    node: ExprNode,
    *,
    operators: Optional[Iterable[str]] = None,
    fields: Optional[Iterable[str]] = None,
    extra_identifiers: Iterable[str] = (),
) -> tuple[bool, list[str]]:
    """Check operator names and data fields before spending a simulation.

    Returns ``(ok, errors)`` where errors are stable strings such as
    ``unknown_operator:foo`` or ``unknown_field:bar``.
    """
    errors: list[str] = []

    if operators is not None:
        allowed_ops = set(operators)
        for name in sorted(collect_call_operators(node)):
            if name not in allowed_ops:
                errors.append(f"unknown_operator:{name}")

    if fields is not None:
        allowed_fields = set(fields) | BUILTIN_IDENTIFIERS | set(extra_identifiers)
        for name in sorted(collect_identifiers(node)):
            if name not in allowed_fields:
                errors.append(f"unknown_field:{name}")

    return (not errors), errors
