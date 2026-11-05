"""Offline expression-space diversity scoring against the account alpha pool.

What this module is for
-----------------------
The QD outer loop needs to know whether the search is re-sampling the same
expression region (mode collapse). The account pool is a good reference set for
that: it is large, real, and already paid for. Fetching it costs no
simulations; scoring is local.

What this module is NOT for
---------------------------
It does **not** predict `SELF_CORRELATION`, and it must not be used as a gate
that rejects candidates. Measured on this account (n=15 evaluated alphas, both
against the full pool and against the 30 submitted alphas only):

    Spearman(similarity, SELF_CORRELATION) = -0.14 / -0.19

i.e. no signal, slightly negative. Two concrete counter-examples:

* ``omoN8q1J`` is textually identical to an already-submitted alpha
  (similarity 1.0) yet reports SELF_CORRELATION 0.6206 (pass).
* ``vRONw11v`` is textually dissimilar (similarity 0.42) yet reports
  SELF_CORRELATION 0.7715 (fail, the highest observed).

Real self-correlation is a property of daily returns, not of syntax, so the
only trustworthy measurement is the platform check itself. Anti-duplication is
handled by :mod:`alpha_agent.expr_ast` behavioral signatures (which are exact
enough to collapse wrapper permutations); this module only feeds diversity
descriptors and reporting.

Coverage caveat: the platform caps pagination, so a fetch can return fewer
alphas than ``count`` suggests. ``stats()["coverage"]`` reports what was
actually reached.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from alpha_agent.expr_ast import ast_bigrams, try_parse

__all__ = ["PoolEntry", "NoveltyPool", "token_set", "jaccard"]

ClientFactory = Callable[[], Any]

#: Entries older than this are considered stale and refetched on demand.
DEFAULT_TTL_DAYS = 7.0


def token_set(expression: str) -> set[str]:
    """Token set used for the textual similarity component.

    Mirrors ``alpha_agent.research_logic.tokenize_expression`` so the hybrid
    path stays comparable with legacy novelty scoring, but is duplicated here
    to keep this module free of the pipeline import chain.
    """
    import re

    raw = re.split(r"[^a-zA-Z0-9_]+", (expression or "").lower())
    return {token for token in raw if token and not token.isdigit() and len(token) > 1}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _settings_region(settings: Optional[dict]) -> Optional[str]:
    if not isinstance(settings, dict):
        return None
    region = settings.get("region")
    return str(region).upper() if isinstance(region, str) and region else None


@dataclass(frozen=True)
class PoolEntry:
    """One alpha already in the account."""

    alpha_id: str
    expression: str
    settings: dict = field(default_factory=dict)
    status: str = ""
    submitted: bool = False
    sharpe: Optional[float] = None
    created_at: Optional[str] = None

    @property
    def region(self) -> Optional[str]:
        return _settings_region(self.settings)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_from_alpha(item: dict) -> Optional[PoolEntry]:
    if not isinstance(item, dict):
        return None
    alpha_id = item.get("id")
    regular = item.get("regular")
    expression = None
    if isinstance(regular, dict):
        expression = regular.get("code")
    elif isinstance(regular, str):
        expression = regular
    if not isinstance(alpha_id, str) or not alpha_id or not isinstance(expression, str):
        return None
    if not expression.strip():
        return None
    is_data = item.get("is") if isinstance(item.get("is"), dict) else {}
    status = str(item.get("status") or "")
    return PoolEntry(
        alpha_id=alpha_id,
        expression=expression.strip(),
        settings=item.get("settings") if isinstance(item.get("settings"), dict) else {},
        status=status,
        submitted=status.upper() == "SUBMITTED" or bool(item.get("dateSubmitted")),
        sharpe=_to_float(is_data.get("sharpe")),
        created_at=item.get("dateCreated") if isinstance(item.get("dateCreated"), str) else None,
    )


class NoveltyPool:
    """Cached account pool with O(pool) similarity queries.

    Coverage caveat: the platform caps pagination (observed: only the first
    ~1100 of ~2000 alphas are reachable by offset), so ``stats()["coverage"]``
    must be reported alongside any novelty claim. A partial pool can only
    under-estimate similarity, never over-estimate it.
    """

    def __init__(self, entries: Sequence[PoolEntry] = (), *, total_reported: int = 0) -> None:
        self.entries: list[PoolEntry] = list(entries)
        self.total_reported = max(int(total_reported), len(self.entries))
        self._tokens: list[set[str]] = []
        self._bigrams: list[Optional[set[tuple[str, str]]]] = []
        self._regions: list[Optional[str]] = []
        self._index()

    # -- construction -------------------------------------------------------
    def _index(self) -> None:
        self._tokens = [token_set(entry.expression) for entry in self.entries]
        self._bigrams = []
        for entry in self.entries:
            node = try_parse(entry.expression)
            self._bigrams.append(ast_bigrams(node) if node is not None else None)
        self._regions = [entry.region for entry in self.entries]

    @classmethod
    def from_payload(cls, payload: Iterable[dict], *, total_reported: int = 0) -> "NoveltyPool":
        entries = [entry for entry in (_entry_from_alpha(item) for item in payload) if entry]
        return cls(entries, total_reported=total_reported)

    # -- persistence --------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "NoveltyPool":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        raw_entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(raw_entries, list):
            return cls()
        total_reported = 0
        if isinstance(payload, dict):
            for key in ("total_reported", "count"):
                value = payload.get(key)
                if isinstance(value, int) and value > total_reported:
                    total_reported = value
        entries = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            alpha_id = item.get("alpha_id")
            expression = item.get("expression")
            if not isinstance(alpha_id, str) or not isinstance(expression, str):
                continue
            entries.append(
                PoolEntry(
                    alpha_id=alpha_id,
                    expression=expression,
                    settings=item.get("settings") if isinstance(item.get("settings"), dict) else {},
                    status=str(item.get("status") or ""),
                    submitted=bool(item.get("submitted")),
                    sharpe=_to_float(item.get("sharpe")),
                    created_at=item.get("created_at") if isinstance(item.get("created_at"), str) else None,
                )
            )
        return cls(entries, total_reported=total_reported)

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": _iso_now(),
            "count": len(self.entries),
            "total_reported": self.total_reported,
            "entries": [
                {
                    "alpha_id": entry.alpha_id,
                    "expression": entry.expression,
                    "settings": entry.settings,
                    "status": entry.status,
                    "submitted": entry.submitted,
                    "sharpe": entry.sharpe,
                    "created_at": entry.created_at,
                }
                for entry in self.entries
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # -- fetching -----------------------------------------------------------
    @classmethod
    def fetch(
        cls,
        *,
        client_factory: ClientFactory,
        limit: int = 2000,
        max_pages: int = 40,
        page_size: int = 100,
    ) -> "NoveltyPool":
        """Page through ``GET /users/self/alphas``. Costs no simulations."""
        client = client_factory()
        url: Optional[str] = f"/users/self/alphas?limit={page_size}&offset=0"
        collected: list[dict] = []
        total_reported = 0
        pages = 0
        while url and pages < max_pages and len(collected) < limit:
            response = client._request("GET", url, accept="application/json;version=2.0")
            if response.status >= 400:
                break
            payload = response.data if isinstance(response.data, dict) else {}
            reported = payload.get("count")
            if isinstance(reported, int) and reported > total_reported:
                total_reported = reported
            results = payload.get("results")
            if isinstance(results, list):
                collected.extend(item for item in results if isinstance(item, dict))
            pages += 1
            next_url = payload.get("next")
            url = next_url if isinstance(next_url, str) and next_url else None
        return cls.from_payload(collected[:limit], total_reported=total_reported)

    @classmethod
    def load_or_fetch(
        cls,
        path: Path,
        *,
        client_factory: Optional[ClientFactory] = None,
        refresh: bool = False,
        limit: int = 2000,
        ttl_days: float = DEFAULT_TTL_DAYS,
    ) -> "NoveltyPool":
        """Use the cache when fresh, otherwise fetch; never raise."""
        target = Path(path)
        if not refresh and target.exists() and _is_fresh(target, ttl_days):
            pool = cls.load(target)
            if pool.entries:
                return pool
        if client_factory is None:
            return cls.load(target)
        try:
            pool = cls.fetch(client_factory=client_factory, limit=limit)
        except Exception:
            return cls.load(target)
        if pool.entries:
            try:
                pool.save(target)
            except OSError:
                pass
        return pool

    # -- scoring ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def similarity_to(self, index: int, *, expression: str, tokens: set[str],
                      bigrams: Optional[set[tuple[str, str]]], region: Optional[str]) -> float:
        if region is not None and self._regions[index] is not None:
            if self._regions[index] != region:
                return 0.0
        token_score = jaccard(tokens, self._tokens[index])
        pool_bigrams = self._bigrams[index]
        if bigrams is None or pool_bigrams is None:
            return token_score
        return 0.5 * token_score + 0.5 * jaccard(bigrams, pool_bigrams)

    def neighbors(
        self,
        expression: str,
        settings: Optional[dict] = None,
        *,
        k: int = 5,
        exclude_alpha_ids: Optional[Iterable[str]] = None,
    ) -> list[tuple[float, PoolEntry]]:
        """Most similar pool entries, highest first."""
        if not self.entries:
            return []
        excluded = set(exclude_alpha_ids or ())
        tokens = token_set(expression)
        node = try_parse(expression)
        bigrams = ast_bigrams(node) if node is not None else None
        region = _settings_region(settings)
        scored = [
            (self.similarity_to(index, expression=expression, tokens=tokens,
                                bigrams=bigrams, region=region), index)
            for index in range(len(self.entries))
            if self.entries[index].alpha_id not in excluded
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(score, self.entries[index]) for score, index in scored[: max(1, k)]]

    def max_similarity(
        self,
        expression: str,
        settings: Optional[dict] = None,
        *,
        exclude_alpha_ids: Optional[Iterable[str]] = None,
    ) -> float:
        """Highest similarity against any pool entry, in ``[0, 1]``.

        Pass ``exclude_alpha_ids`` when scoring an alpha that is itself in the
        pool, otherwise it matches itself and reports similarity 1.0.
        """
        if not self.entries:
            return 0.0
        excluded = set(exclude_alpha_ids or ())
        tokens = token_set(expression)
        node = try_parse(expression)
        bigrams = ast_bigrams(node) if node is not None else None
        region = _settings_region(settings)
        best = 0.0
        for index in range(len(self.entries)):
            if self.entries[index].alpha_id in excluded:
                continue
            score = self.similarity_to(
                index, expression=expression, tokens=tokens, bigrams=bigrams, region=region
            )
            if score > best:
                best = score
        return best

    def novelty(
        self,
        expression: str,
        settings: Optional[dict] = None,
        *,
        exclude_alpha_ids: Optional[Iterable[str]] = None,
    ) -> float:
        """``1 - max_similarity``; ``1.0`` when the pool is empty or unusable."""
        if not self.entries:
            return 1.0
        return max(
            0.0,
            min(1.0, 1.0 - self.max_similarity(expression, settings, exclude_alpha_ids=exclude_alpha_ids)),
        )

    def stats(self) -> dict:
        submitted = sum(1 for entry in self.entries if entry.submitted)
        total = max(self.total_reported, len(self.entries))
        return {
            "pool_size": len(self.entries),
            "total_reported": self.total_reported,
            "coverage": round(len(self.entries) / total, 4) if total else 0.0,
            "submitted": submitted,
            "unsubmitted": len(self.entries) - submitted,
        }


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _is_fresh(path: Path, ttl_days: float) -> bool:
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
    except OSError:
        return False
    return age_days <= max(0.0, ttl_days)
