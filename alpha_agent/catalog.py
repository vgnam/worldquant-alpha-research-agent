"""Offline catalog of valid operators and data fields.

The platform is the source of truth, but every lookup costs a request and
LLM-proposed formulas that reference unknown operators waste a simulation.
This module caches both lists in the workdir so expression validation runs
for free before any simulation is spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

__all__ = ["Catalog", "load_catalog", "remember_field_types"]

ClientFactory = Callable[[], Any]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _field_types_from_cache(path: Path, arena: Optional[dict] = None) -> dict[str, str]:
    """Cached field types, but only for the arena that produced them."""
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if arena is not None:
        cached_arena = payload.get("arena")
        if isinstance(cached_arena, dict):
            for key, value in arena.items():
                if cached_arena.get(key) != value:
                    return {}
    types = payload.get("types")
    if isinstance(types, dict):
        return {str(name): str(kind) for name, kind in types.items() if isinstance(kind, str)}
    return {}


def _field_types_from_summary(path: Path) -> dict[str, str]:
    """A summary file that happens to carry ``type`` is still useful."""
    if not path.exists():
        return {}
    payload = _read_json(path)
    types: dict[str, str] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("type"):
                types[item["id"]] = str(item["type"])
    return types


def _fetch_field_types(
    client_factory: ClientFactory,
    arena: dict,
    field_ids: Iterable[str],
) -> dict[str, str]:
    """Look up the types of *specific* fields, one request each.

    The listing endpoint caps ``limit`` at 50 and runs to ~10000 records, so
    fetching everything costs ~200 requests (~6 minutes) and still misses the
    point: a run only touches a few dozen fields. Callers must name them.
    """
    names = sorted({str(name) for name in field_ids if name})
    if not names:
        return {}
    client = client_factory()
    types: dict[str, str] = {}
    for name in names:
        try:
            payload = client.fetch_data_fields(
                region=arena.get("region", "USA"),
                universe=arena.get("universe", "TOP3000"),
                delay=int(arena.get("delay", 1)),
                ids=[name],
            )
        except Exception:
            continue
        for item in payload or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("type"):
                types[item["id"]] = str(item["type"])
    return types


def _fields_from_summary(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    payload = _read_json(path)
    names: set[str] = set()
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                names.add(item["id"])
    return frozenset(names)


def _operators_from_cache(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    payload = _read_json(path)
    if isinstance(payload, dict):
        names = payload.get("names")
        if isinstance(names, list):
            return frozenset(str(name) for name in names if isinstance(name, str))
    if isinstance(payload, list):
        collected = {
            str(item["name"])
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        return frozenset(collected)
    return frozenset()


def _fetch_operators(client_factory: ClientFactory) -> frozenset[str]:
    client = client_factory()
    payload = client.fetch_operators()
    if isinstance(payload, list):
        return frozenset(
            str(item["name"])
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        )
    return frozenset()


@dataclass(frozen=True)
class Catalog:
    """Valid operator names and data field ids for the current account."""

    operators: frozenset[str] = frozenset()
    fields: frozenset[str] = frozenset()
    #: Field id -> ``MATRIX`` / ``VECTOR`` / ``GROUP`` for the run's arena.
    field_types: dict = field(default_factory=dict)

    @property
    def has_operators(self) -> bool:
        return bool(self.operators)

    @property
    def has_fields(self) -> bool:
        return bool(self.fields)

    @property
    def has_field_types(self) -> bool:
        return bool(self.field_types)

    def validate(self, node, **kwargs) -> tuple[bool, list[str]]:
        from alpha_agent.expr_ast import validate_against

        return validate_against(
            node,
            operators=self.operators or None,
            fields=self.fields or None,
            **kwargs,
        )

    def type_violations(self, node) -> list[str]:
        """VECTOR fields used outside a ``vec_*`` reduction (see field_types)."""
        from alpha_agent.field_types import vector_violations

        return vector_violations(node, self.field_types)

    def matrix_fields(self, names: Iterable[str]) -> list[str]:
        from alpha_agent.field_types import filter_matrix_fields

        return filter_matrix_fields(names, self.field_types)


def remember_field_types(
    catalog: "Catalog",
    types: dict,
    *,
    cache_path: Optional[Path] = None,
    arena: Optional[dict] = None,
) -> int:
    """Add resolved types to a live catalog (and its cache); returns how many.

    The catalog is frozen but its ``field_types`` dict is not, so a run can keep
    learning the types of fields its proposer invents without rebuilding it.
    """
    added = 0
    for name, kind in (types or {}).items():
        if not name or not isinstance(kind, str):
            continue
        if catalog.field_types.get(name) != kind:
            catalog.field_types[str(name)] = kind
            added += 1
    if added and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {"arena": arena or {}, "count": len(catalog.field_types), "types": catalog.field_types},
                    ensure_ascii=False,
                    indent=0,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
    return added


def load_catalog(
    *,
    fields_summary: Optional[Path] = None,
    operators_cache: Optional[Path] = None,
    client_factory: Optional[ClientFactory] = None,
    refresh: bool = False,
    arena: Optional[dict] = None,
    field_types_cache: Optional[Path] = None,
    field_ids: Optional[Iterable[str]] = None,
) -> Catalog:
    """Load the catalog, fetching operators and field types when needed.

    Never raises for a missing cache: an empty catalog simply disables the
    corresponding validation step, which keeps the hybrid search fail-open.
    """
    operators = frozenset() if refresh else _operators_from_cache(operators_cache or Path("/nonexistent"))

    if not operators and client_factory is not None:
        try:
            operators = _fetch_operators(client_factory)
        except Exception:
            operators = frozenset()
        if operators and operators_cache is not None:
            try:
                operators_cache.parent.mkdir(parents=True, exist_ok=True)
                operators_cache.write_text(
                    json.dumps(
                        {"count": len(operators), "names": sorted(operators)},
                        ensure_ascii=False,
                        indent=1,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

    fields = _fields_from_summary(fields_summary) if fields_summary else frozenset()

    # Field *types* decide whether a candidate is simulable at all: a VECTOR
    # (event) field handed to an ordinary operator is rejected by the platform
    # and the simulation is lost. They are fetched only for the fields the
    # caller names — a bulk listing is ~200 requests and buys nothing.
    field_types: dict[str, str] = {}
    if not refresh and field_types_cache is not None:
        field_types = _field_types_from_cache(field_types_cache, arena)
    if not field_types and fields_summary:
        field_types = _field_types_from_summary(fields_summary)
    wanted = [str(name) for name in (field_ids or ()) if name and name not in field_types]
    if wanted and client_factory is not None:
        fetched = _fetch_field_types(client_factory, arena or {}, wanted[:200])
        if fetched:
            field_types = {**field_types, **fetched}
            if field_types_cache is not None:
                try:
                    field_types_cache.parent.mkdir(parents=True, exist_ok=True)
                    field_types_cache.write_text(
                        json.dumps(
                            {
                                "arena": arena or {},
                                "count": len(field_types),
                                "types": field_types,
                            },
                            ensure_ascii=False,
                            indent=0,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass

    return Catalog(operators=operators, fields=fields, field_types=field_types)
