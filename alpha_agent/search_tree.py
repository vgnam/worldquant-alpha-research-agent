"""MCTS bookkeeping for formula refinement (Alpha Jungle, arXiv:2505.11122).

Differences from textbook MCTS that this module deliberately implements:

* **No rollout.** A node is scored by its real backtest result. With one
  simulation costing ~90 seconds and real quota, a random playout is
  unaffordable and would estimate something we can measure exactly.
* **Virtual expansion action.** Any node, including an already evaluated one,
  can be expanded again. Without this, promising internal nodes could never be
  revisited and the search degenerates to a one-shot beam.
* **Max-in-subtree value.** ``q_value`` is the best reward seen in the subtree,
  matching the paper's ``Q(s,a)``. Averaging would reward nodes that produce
  many mediocre children.

Selection follows MCTS-AHD (Zheng et al., 2025, ``source/mcts.py`` and
``mcts_ahd.py``):

* **Normalised UCT.** ``(Q - q_min) / (q_max - q_min) + lambda * sqrt(ln(N_parent + 1) / N)``
  with ``lambda`` scaled by the remaining budget. The previous score added a
  ~1.0 exploration bonus to raw rewards in the hundreds or thousands, so it
  was pure greedy selection in practice.
* **Descent from a virtual root.** :meth:`plan` walks root -> leaf by
  ``value`` instead of ranking every node globally.
* **Progressive widening, gated by progress.** A node whose
  ``floor(attempts ** alpha)`` exceeds its child count may be widened on the
  way down; at the virtual root that creates a new subtree (MCTS-AHD's
  ``e1``, a new *idea*). An inner node is only widened once its own front has
  stopped growing for ``widen_patience`` evaluations, because a sibling under
  a node whose front still improves only duplicates work — with the ungated
  rule, 42 of 88 evaluations in a saved run landed on leaves already at the
  depth cap.

Multi-objective value (MO-MCTS, Wang & Sebag 2012; Perez et al. 2013):

* **A Pareto front per node.** Besides the scalar max-in-subtree ``Q``, every
  node keeps the epsilon-Pareto front of the objective vectors evaluated in
  its subtree, and the tree keeps the global front at the virtual root.
  Backprop is ``P_s <- ND(P_s | {f})`` along the path, the set-valued
  analogue of ``Q <- max(Q, r)``.
* **Optimistic hypervolume replaces UCB.** In ``plan(value="ohvi")`` a child
  ``c`` scores ``HV(R | P_c + beta_c) - HV(R)``, where ``P_c`` is the
  subtree's front, every point of it is shifted by
  ``beta_c = lambda * sqrt(ln(N_parent + 1) / (attempts_c + 1))`` toward
  better on every objective, and ``R`` is **one reference set shared by every
  child at that step**: the global front minus the points the competing
  branches own. Sharing ``R`` is what makes the scores comparable. Scoring
  each child against the front *without its own points* (the previous
  behaviour) gave the branch that owned most of the front a nearly empty
  reference and every rival a full one, so the owner won every descent and
  the other branches were never revisited; on a saved 96-node tree the
  champion won 20 of 20 replays. With a common ``R`` the number is the volume
  a subtree would cover if it were as good as its visit count still allows,
  ``beta = 0`` reduces it to the volume the subtree already covers, and a
  near-duplicate-heavy branch is not rewarded for owning many points.
* **Distance to the front breaks the remaining ties.** A branch with no front
  at all (no complete objective vector) scores zero, so when no child adds
  volume the descent takes the child whose front needs the smallest further
  shift to stop being dominated (additive epsilon indicator, Zitzler et al.
  2003, minus ``beta``). Selection never falls back to a scalar reward, so no
  gate threshold re-enters through the back door.

Exploration is carried by ``beta`` alone, so ``attempts`` — not ``visits`` —
feeds it. ``visits`` counts evaluated descendants and never moves for a branch
whose proposals all fail the offline filters, which used to leave such a branch
permanently at maximum optimism; ``attempts`` counts times the branch was
actually handed work and decays the optimism every time.

The tree is persisted so a run can resume, and so an archive can be rebuilt
without paying for simulations twice.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from alpha_agent.expr_ast import behavioral_signature, try_parse
from alpha_agent.pareto import (
    DEFAULT_OBJECTIVES,
    OBJECTIVE_BOUNDS,
    ParetoFront,
    hypervolume,
    is_feasible,
    normalize_objectives,
)

__all__ = ["TreeNode", "SearchTree"]

PENDING = "pending"
EVALUATED = "evaluated"
ERROR = "error"
PRUNED = "pruned"


def _new_id(prefix: str = "n") -> str:
    """Unique node id.

    Must be globally unique, not merely unique per process: a persisted tree can
    be reloaded and extended in a later run, and ``id(object())`` reuses memory
    addresses, which silently overwrote nodes during development.
    """
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TreeNode:
    node_id: str
    expression: str
    settings: dict = field(default_factory=dict)
    family: str = ""
    idea_name: str = ""
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    visits: int = 1
    #: Times this node was handed work by the descent. Unlike ``visits`` (which
    #: counts evaluated descendants) this moves even when every proposal from
    #: the node is rejected, so the optimism term can decay on it.
    attempts: int = 0
    q_value: float = float("-inf")
    state: str = PENDING
    reward: Optional[float] = None
    margins: dict = field(default_factory=dict)
    e_vector: dict = field(default_factory=dict)
    alpha_id: Optional[str] = None
    behavioral_sig: Optional[str] = None
    candidate_key: Optional[str] = None
    #: Raw objective values by name (see ``alpha_agent.pareto.OBJECTIVE_BOUNDS``).
    objectives: dict = field(default_factory=dict)

    @property
    def is_evaluated(self) -> bool:
        return self.state == EVALUATED

    @property
    def is_pending(self) -> bool:
        return self.state == PENDING

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "expression": self.expression,
            "settings": self.settings,
            "family": self.family,
            "idea_name": self.idea_name,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "depth": self.depth,
            "visits": self.visits,
            "attempts": self.attempts,
            "q_value": self.q_value,
            "state": self.state,
            "reward": self.reward,
            "margins": self.margins,
            "e_vector": self.e_vector,
            "alpha_id": self.alpha_id,
            "behavioral_sig": self.behavioral_sig,
            "candidate_key": self.candidate_key,
            "objectives": self.objectives,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TreeNode":
        objectives = payload.get("objectives") if isinstance(payload.get("objectives"), dict) else {}
        if not objectives:
            # Trees written by the QD version stored two objectives as fields.
            objectives = {name: payload.get(name) for name in ("quality", "divergence")}
        return cls(
            node_id=str(payload.get("node_id") or _new_id()),
            expression=str(payload.get("expression") or ""),
            settings=payload.get("settings") if isinstance(payload.get("settings"), dict) else {},
            family=str(payload.get("family") or ""),
            idea_name=str(payload.get("idea_name") or ""),
            parent_id=payload.get("parent_id") if isinstance(payload.get("parent_id"), str) else None,
            children=[str(child) for child in payload.get("children") or [] if isinstance(child, str)],
            depth=int(payload.get("depth") or 0),
            visits=max(1, int(payload.get("visits") or 1)),
            attempts=max(0, int(payload.get("attempts") or 0)),
            q_value=_to_float(payload.get("q_value")) if payload.get("q_value") is not None else float("-inf"),
            state=str(payload.get("state") or PENDING),
            reward=_to_float(payload.get("reward")),
            margins=payload.get("margins") if isinstance(payload.get("margins"), dict) else {},
            e_vector=payload.get("e_vector") if isinstance(payload.get("e_vector"), dict) else {},
            alpha_id=payload.get("alpha_id") if isinstance(payload.get("alpha_id"), str) else None,
            behavioral_sig=payload.get("behavioral_sig") if isinstance(payload.get("behavioral_sig"), str) else None,
            candidate_key=payload.get("candidate_key") if isinstance(payload.get("candidate_key"), str) else None,
            objectives={
                str(name): float(value)
                for name, value in objectives.items()
                if _to_float(value) is not None
            },
        )


class SearchTree:
    """UCT tree over formula refinements with a Pareto front per node."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        max_nodes: int = 512,
        max_children: int = 4,
        max_depth: int = 5,
        objectives: Sequence[str] = DEFAULT_OBJECTIVES,
        front_size: int = 24,
        epsilon: float = 0.01,
        widen_patience: int = 3,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_nodes = max(8, int(max_nodes))
        self.max_children = max(1, int(max_children))
        self.max_depth = max(1, int(max_depth))
        unknown = [name for name in objectives if name not in OBJECTIVE_BOUNDS]
        if unknown or not objectives:
            raise ValueError(f"Unknown objectives {unknown}; choose from {sorted(OBJECTIVE_BOUNDS)}.")
        self.objectives = tuple(objectives)
        self.front_size = max(1, int(front_size))
        self.epsilon = max(0.0, float(epsilon))
        #: Evaluations a subtree may go without gaining a front point before an
        #: inner node is allowed to widen (a sibling is then worth paying for).
        self.widen_patience = max(0, int(widen_patience))
        self.nodes: Dict[str, TreeNode] = {}
        self.roots: list[str] = []
        #: Visit count of the virtual root that parents every subtree.
        self.root_visits = 0
        self._rewards: list[float] = []
        self._pruned = 0
        #: Local front of every node with an evaluated descendant. Derived
        #: from ``TreeNode.objectives``, so it is rebuilt on load, not saved.
        self.fronts: Dict[str, ParetoFront] = {}
        #: Global front, i.e. the virtual root's.
        self.front = self._new_front()
        #: Evaluations since each subtree's front last gained a member, keyed by
        #: node id (``""`` is the virtual root). Derived, so not saved.
        self.stalls: Dict[str, int] = {}
        #: Evaluated nodes that could not enter any front, split by cause, so a
        #: run that silently loses its multi-objective value becomes visible.
        self.incomplete_nodes: set[str] = set()
        self.out_of_range_nodes: set[str] = set()

    # -- construction -------------------------------------------------------
    def add_root(self, candidate: Any, *, node_id: Optional[str] = None) -> TreeNode:
        node = self._from_candidate(candidate, parent=None, depth=0, node_id=node_id)
        self.nodes[node.node_id] = node
        self.roots.append(node.node_id)
        return node

    def add_child(self, parent_id: str, candidate: Any, *, node_id: Optional[str] = None) -> Optional[TreeNode]:
        parent = self.nodes.get(parent_id)
        if parent is None:
            return None
        if self.children_used(parent) >= self.max_children or parent.depth + 1 > self.max_depth:
            return None
        node = self._from_candidate(candidate, parent=parent, depth=parent.depth + 1, node_id=node_id)
        # A child of an unevaluated parent still inherits nothing to learn from.
        self.nodes[node.node_id] = node
        parent.children.append(node.node_id)
        if len(self.nodes) > self.max_nodes:
            self._prune()
        return node

    def _from_candidate(
        self,
        candidate: Any,
        *,
        parent: Optional[TreeNode],
        depth: int,
        node_id: Optional[str] = None,
    ) -> TreeNode:
        expression = str(getattr(candidate, "expression", "") or "")
        settings = getattr(candidate, "settings", {}) or {}
        if not isinstance(settings, dict):
            settings = {}
        signature = None
        if try_parse(expression) is not None:
            try:
                signature = behavioral_signature(expression, settings)
            except Exception:
                signature = None
        return TreeNode(
            node_id=node_id or _new_id(),
            expression=expression,
            settings=dict(settings),
            family=str(getattr(candidate, "family", "") or ""),
            idea_name=str(getattr(candidate, "idea_name", "") or ""),
            parent_id=parent.node_id if parent else None,
            depth=depth,
            behavioral_sig=signature,
            candidate_key=getattr(candidate, "parent_key", None),
        )

    # -- selection ----------------------------------------------------------
    def q_bounds(self) -> tuple[float, float]:
        """Min / max Q over evaluated nodes; error rewards are excluded."""
        values = [
            float(node.q_value)
            for node in self.nodes.values()
            if node.is_evaluated and node.q_value not in (None, float("-inf"))
        ]
        if not values:
            return 0.0, 0.0
        return min(values), max(values)

    def normalized_q(self, node: TreeNode, bounds: Optional[tuple[float, float]] = None) -> float:
        if node.q_value in (None, float("-inf")) or not node.is_evaluated:
            return 0.0
        q_min, q_max = bounds or self.q_bounds()
        if q_max - q_min <= 1e-12:
            return 1.0
        return (float(node.q_value) - q_min) / (q_max - q_min)

    def parent_visits(self, node: TreeNode) -> int:
        if node.parent_id and node.parent_id in self.nodes:
            return self.nodes[node.parent_id].visits
        return max(self.root_visits, 1)

    def attempts_of(self, node: TreeNode) -> int:
        """Times the descent handed this branch work (not its descendants')."""
        return max(0, int(node.attempts))

    def children_used(self, node: TreeNode) -> int:
        """Children holding a slot: every child except one that errored.

        Counting *every* child (the previous rule) meant four failed
        refinements permanently bricked a parent: ``has_room`` said no and
        ``_prune`` only ever drops pending nodes, so the whole subtree below
        that parent became unreachable for the rest of the run.
        """
        return sum(
            1
            for child in node.children
            if child in self.nodes and self.nodes[child].state != ERROR
        )

    def ohvi(self, node: TreeNode, *, beta: float, reference: Optional[Iterable[str]] = None) -> float:
        """Optimistic volume of ``node``'s front *over one shared baseline*.

        ``HV(R | P_c + beta) - HV(R)``: the volume the subtree's front,
        shifted by ``beta`` toward better on every objective, covers on top of
        ``R``.

        ``reference`` — the keys of ``R`` — must be identical for every child
        compared in one descent step; :meth:`plan` passes the union of the
        competing branches' points. ``reference=None`` defaults to the node's
        own points, i.e. the subtree's exclusive contribution to the global
        front, which is only meaningful when scoring one subtree in isolation.
        """
        front = self.fronts.get(node.node_id)
        if not node.is_evaluated or front is None or not len(front):
            return 0.0
        excluded = front.keys() if reference is None else set(reference)
        rest = [entry.vector for entry in self.front.entries if entry.key not in excluded]
        shift = max(0.0, float(beta))
        shifted = [tuple(min(1.0, value + shift) for value in vector) for vector in front.vectors()]
        return hypervolume(rest + shifted) - hypervolume(rest)

    def front_gap(self, node: TreeNode) -> Optional[float]:
        """Additive epsilon indicator of ``node``'s subtree to the global front.

        The smallest shift, on every objective at once, that would lift some
        point of the subtree's front out of domination by the rest of the
        global front: ``min_p max(0, max_q min_i (q_i - p_i))``. Zero when the
        subtree already holds a front point; None when it has no front.
        """
        front = self.fronts.get(node.node_id)
        if not node.is_evaluated or front is None or not len(front):
            return None
        owned = front.keys()
        rest = [entry.vector for entry in self.front.entries if entry.key not in owned]
        return min(
            max(0.0, max((min(q - p for p, q in zip(point, other)) for other in rest), default=0.0))
            for point in front.vectors()
        )

    def exploration_width(self, node: TreeNode, uct_c: float) -> float:
        """``beta``: the optimism applied to every objective of the subtree.

        Divided by ``attempts + 1``, not by ``visits``: a branch whose
        proposals are all rejected offline never gains a visit, so the old
        denominator kept it at maximum optimism forever while the descent kept
        picking it.
        """
        return uct_c * math.sqrt(math.log(self.parent_visits(node) + 1) / (self.attempts_of(node) + 1.0))

    def stall(self, node: Optional[TreeNode]) -> int:
        """Evaluations since this subtree's front last gained a member."""
        return self.stalls.get("" if node is None else node.node_id, 0)

    def stalled(self, node: Optional[TreeNode], patience: Optional[int] = None) -> bool:
        """Whether the subtree has stopped improving for ``patience`` evaluations.

        ``patience=0`` disables the gate.
        """
        limit = self.widen_patience if patience is None else max(0, int(patience))
        return limit <= 0 or self.stall(node) >= limit

    def uct_score(
        self,
        node: TreeNode,
        uct_c: float,
        *,
        bounds: Optional[tuple[float, float]] = None,
        bonus: float = 0.0,
        exploitation: Optional[float] = None,
    ) -> float:
        """Exploitation (normalised Q unless given) + exploration + bonus."""
        exploration = self.exploration_width(node, uct_c)
        if exploitation is None:
            exploitation = self.normalized_q(node, bounds)
        return float(exploitation) + exploration + float(bonus)

    def children_of(self, node: Optional[TreeNode]) -> list[TreeNode]:
        ids = self.roots if node is None else node.children
        return [self.nodes[child] for child in ids if child in self.nodes]

    def has_room(self, node: Optional[TreeNode], *, max_roots: Optional[int] = None) -> bool:
        if node is None:
            return max_roots is None or len(self.roots) < max_roots
        return self.children_used(node) < self.max_children and node.depth + 1 <= self.max_depth

    def _descendable(self, node: TreeNode, _seen: Optional[set[str]] = None) -> bool:
        """Evaluated, and either expandable itself or above an expandable node."""
        seen = set() if _seen is None else _seen
        if node.node_id in seen:  # corrupt persisted tree: do not recurse forever
            return False
        seen.add(node.node_id)
        if not node.is_evaluated:
            return False
        if self.has_room(node):
            return True
        return any(self._descendable(child, seen) for child in self.children_of(node))

    def _should_widen(
        self,
        node: Optional[TreeNode],
        children: list[TreeNode],
        *,
        widen_alpha: float,
        widen_patience: int,
        max_roots: Optional[int],
    ) -> bool:
        """Birth-rate rule (MCTS-AHD), plus a progress gate inside the tree.

        At the virtual root widening means "try a new *idea*" (``e1``), so the
        birth rate alone decides. At an inner node a new sibling is only worth
        a simulation once the branch's own front has stopped moving: before
        this gate, half of a saved run's evaluations landed on leaves that
        were already at the depth cap.
        """
        if not self.has_room(node, max_roots=max_roots):
            return False
        work = max(self.root_visits, 1) if node is None else max(1, node.visits)
        if int(work ** float(widen_alpha)) <= len(children):
            return False
        return node is None or self.stalled(node, widen_patience)

    def plan(
        self,
        *,
        uct_c: float = 0.1,
        widen_alpha: float = 0.5,
        bonus: Optional[Any] = None,
        max_roots: Optional[int] = None,
        value: str = "ohvi",
        widen_patience: Optional[int] = None,
    ) -> list[tuple[Optional[TreeNode], str]]:
        """One MCTS-AHD descent: ``[(node_or_None, "widen" | "leaf"), ...]``.

        ``None`` is the virtual root, so ``(None, "widen")`` asks for a new
        subtree. ``value="ohvi"`` scores every child with :meth:`ohvi` over
        **one reference set per step** — the points the competing branches own
        are removed from the global front once, so all children are compared
        on the same baseline — using ``uct_c`` as the optimism scale
        ``lambda``. Where no child adds volume at all (a branch with no
        complete objective vector), it falls back to the smallest remaining
        distance to the front, ``front_gap - beta``; when even that is
        unavailable the least-tried child wins, by ``beta``. ``value="scalar"``
        is Alpha Jungle's max-reward Q with UCT. ``bonus(node)`` adds a
        per-node term. The descent stops at the first node with no descendable
        child, which becomes the leaf to expand, and every node handed a job
        records an attempt.
        """
        bounds = self.q_bounds()
        patience = self.widen_patience if widen_patience is None else max(0, int(widen_patience))

        def pick(candidates: list[TreeNode]) -> TreeNode:
            extra = {child.node_id: bonus(child) if bonus else 0.0 for child in candidates}
            if value != "ohvi":
                return max(
                    candidates,
                    key=lambda child: (
                        self.uct_score(child, uct_c, bounds=bounds, bonus=extra[child.node_id]),
                        -self.attempts_of(child),
                        child.node_id,
                    ),
                )
            # One baseline for every child: without it the branch that owns
            # most of the front is scored against (almost) nothing while its
            # rivals are scored against that whole front, so it wins forever.
            reference: set[str] = set()
            for child in candidates:
                front = self.fronts.get(child.node_id)
                if front is not None:
                    reference |= front.keys()
            betas = {child.node_id: self.exploration_width(child, uct_c) for child in candidates}
            scores = {
                child.node_id: self.ohvi(child, beta=betas[child.node_id], reference=reference)
                + extra[child.node_id]
                for child in candidates
            }
            if max(scores.values()) > 1e-12:
                return max(
                    candidates,
                    key=lambda child: (scores[child.node_id], -self.attempts_of(child), child.node_id),
                )
            gaps = {child.node_id: self.front_gap(child) for child in candidates}
            ranked = [child for child in candidates if gaps[child.node_id] is not None]
            if ranked:
                return min(
                    ranked,
                    key=lambda child: (
                        gaps[child.node_id] - betas[child.node_id] - extra[child.node_id],
                        self.attempts_of(child),
                        child.node_id,
                    ),
                )
            return max(
                candidates,
                key=lambda child: (betas[child.node_id] + extra[child.node_id], -self.attempts_of(child), child.node_id),
            )

        jobs: list[tuple[Optional[TreeNode], str]] = []
        current: Optional[TreeNode] = None
        while True:
            children = self.children_of(current)
            candidates = [child for child in children if self._descendable(child)]
            if not candidates:
                break
            if self._should_widen(
                current,
                children,
                widen_alpha=widen_alpha,
                widen_patience=patience,
                max_roots=max_roots,
            ):
                jobs.append((current, "widen"))
            current = pick(candidates)
        if current is None:
            jobs.append((None, "widen"))
        elif self.has_room(current):
            jobs.append((current, "leaf"))
        # Charge the work along the whole path: the optimism of a subtree
        # answers "how much have I already tried here", so every ancestor's
        # uncertainty drops when a descendant is expanded — not only the
        # expanded node's, which would leave a root permanently at maximum
        # optimism while its descendants absorbed the budget.
        for target, _kind in jobs:
            self._note_attempt(target)
        return jobs

    def _note_attempt(self, node: Optional[TreeNode]) -> None:
        current = node
        seen: set[str] = set()
        while current is not None and current.node_id not in seen:
            seen.add(current.node_id)
            current.attempts += 1
            current = self.nodes.get(current.parent_id) if current.parent_id else None

    def path_to_root(self, node_id: str) -> list[TreeNode]:
        """Lineage ``[subtree root, ..., node]``."""
        path: list[TreeNode] = []
        current = self.nodes.get(node_id)
        seen: set[str] = set()
        while current is not None and current.node_id not in seen:
            seen.add(current.node_id)
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def subtree(self, node_id: str) -> list[TreeNode]:
        out: list[TreeNode] = []
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None or current in seen:  # corrupt file: a child cycle
                continue
            seen.add(current)
            out.append(node)
            stack.extend(node.children)
        return out

    def root_of(self, node_id: str) -> Optional[TreeNode]:
        path = self.path_to_root(node_id)
        return path[0] if path else None

    def select(self, *, count: int = 1, uct_c: float = 1.2) -> list[TreeNode]:
        """Pick ``count`` nodes to act on.

        Pending children win first: they were proposed but not evaluated, and
        evaluating them is strictly cheaper than proposing more. Remaining slots
        are filled by UCT over evaluated nodes that still have expansion room.
        The batch is made unique by construction rather than by bumping
        ``visits``: the previous "virtual loss" added a permanent visit that
        nothing ever gave back, so repeated calls inflated the counts.
        """
        count = max(0, int(count))
        if count == 0:
            return []

        selected: list[TreeNode] = []
        taken: set[str] = set()
        pending = sorted(
            (node for node in self.nodes.values() if node.is_pending),
            key=lambda node: (self._parent_score(node), -node.depth, node.node_id),
            reverse=True,
        )
        for node in pending:
            if len(selected) >= count:
                break
            selected.append(node)
            taken.add(node.node_id)

        if len(selected) < count:
            expandable = [
                node
                for node in self.nodes.values()
                if node.is_evaluated
                and node.node_id not in taken
                and self.has_room(node)
            ]
            bounds = self.q_bounds()
            expandable.sort(key=lambda node: (-self.uct_score(node, uct_c, bounds=bounds), node.node_id))
            selected.extend(expandable[: count - len(selected)])
        return selected

    def _parent_score(self, node: TreeNode) -> float:
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            return 0.0 if parent.q_value == float("-inf") else float(parent.q_value)
        return 0.0 if node.q_value == float("-inf") else float(node.q_value)

    # -- recording ----------------------------------------------------------
    def record(
        self,
        node_id: str,
        *,
        reward: float,
        margins: Optional[dict] = None,
        e_vector: Optional[dict] = None,
        alpha_id: Optional[str] = None,
        state: str = EVALUATED,
        objectives: Optional[dict] = None,
    ) -> Optional[TreeNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return None
        if objectives is not None:
            node.objectives = {str(name): float(value) for name, value in objectives.items()}
        node.state = state if state in {EVALUATED, ERROR} else EVALUATED
        node.reward = float(reward)
        # A re-evaluation must not lower the subtree's Q: the invariant is
        # max-in-subtree everywhere, including at the node just recorded.
        node.q_value = float(reward) if node.q_value is None else max(float(node.q_value), float(reward))
        node.margins = dict(margins or {})
        node.e_vector = dict(e_vector or {})
        if alpha_id:
            node.alpha_id = alpha_id
        if self._rewards is None:
            self._rewards = []
        self._rewards.append(float(reward))
        self._backprop(node)
        self._backprop_front(node)
        return node

    def _backprop(self, node: TreeNode) -> None:
        current = node
        seen: set[str] = set()
        self.root_visits += 1
        while current.parent_id and current.parent_id in self.nodes:
            parent = self.nodes[current.parent_id]
            if parent.node_id in seen:
                break
            seen.add(parent.node_id)
            parent.visits += 1
            # Error rewards (-10000) never reach an ancestor's Q.
            child_best = max(
                (
                    self.nodes[child].q_value
                    for child in parent.children
                    if child in self.nodes and self.nodes[child].state != ERROR
                ),
                default=float("-inf"),
            )
            best = max(child_best, parent.q_value if parent.q_value is not None else float("-inf"))
            parent.q_value = best
            current = parent

    def _new_front(self) -> ParetoFront:
        return ParetoFront(capacity=self.front_size, epsilon=self.epsilon)

    def objective_vector(self, node: TreeNode) -> Optional[tuple[float, ...]]:
        """Normalised vector, or None when a metric is missing or out of range."""
        return normalize_objectives(node.objectives, self.objectives, strict=True)

    def objective_coverage(self) -> dict:
        """How much of the tree can take part in the Pareto value.

        ``ready`` is the number of evaluated nodes with a usable vector. A run
        with ``tree_value="ohvi"`` and ``ready < 2`` has no multi-objective
        signal at all: it must say so and fall back instead of descending on
        zeros.
        """
        evaluated = self.evaluated()
        ready = sum(1 for node in evaluated if self.objective_vector(node) is not None)
        return {
            "evaluated": len(evaluated),
            "ready": ready,
            "incomplete": len(self.incomplete_nodes),
            "out_of_range": len(self.out_of_range_nodes),
            "front": len(self.front),
        }

    def _backprop_front(self, node: TreeNode) -> None:
        """``P_s <- ND(P_s | {f})`` for the node, its ancestors and the root."""
        if not node.is_evaluated:
            return
        vector = self.objective_vector(node)
        if vector is None:
            # Count *why* a point cannot enter a front. Silent losses used to
            # turn the whole multi-objective value off without one log line.
            if all(name in node.objectives for name in self.objectives):
                self.out_of_range_nodes.add(node.node_id)
            else:
                self.incomplete_nodes.add(node.node_id)
            return
        for member in self.path_to_root(node.node_id):
            front = self.fronts.setdefault(member.node_id, self._new_front())
            before = front.keys()
            kept = front.insert(node.node_id, vector)
            self._note_progress(member.node_id, grew=kept and node.node_id not in before)
        before = self.front.keys()
        kept = self.front.insert(node.node_id, vector)
        self._note_progress("", grew=kept and node.node_id not in before)

    def _note_progress(self, key: str, *, grew: bool) -> None:
        """Track consecutive evaluations that added nothing to a subtree's front."""
        self.stalls[key] = 0 if grew else self.stalls.get(key, 0) + 1

    def rebuild_fronts(self) -> None:
        """Recompute every front from node objectives (load, migration)."""
        self.fronts = {}
        self.front = self._new_front()
        self.incomplete_nodes = set()
        self.out_of_range_nodes = set()
        for node in list(self.nodes.values()):
            self._backprop_front(node)
        # Stall counters are derived from evaluation order, which a reload does
        # not preserve; start optimistic rather than inventing a history.
        self.stalls = {}

    # -- queries ------------------------------------------------------------
    def front_nodes(self) -> list[TreeNode]:
        """Global front members that add hypervolume, best on the first
        objective first."""
        members = [
            self.nodes[entry.key]
            for entry in self.front.entries
            if entry.key in self.nodes and is_feasible(entry.vector)
        ]
        return sorted(
            members,
            key=lambda node: (tuple(-value for value in self.objective_vector(node) or ()), node.node_id),
        )

    def evaluated(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.is_evaluated]

    def pending(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.is_pending]

    def best(self) -> Optional[TreeNode]:
        candidates = [node for node in self.evaluated() if node.reward is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda node: (float(node.reward or 0.0), node.node_id))

    def expandable(self) -> list[TreeNode]:
        return [node for node in self.evaluated() if self.has_room(node)]

    def has_work(self) -> bool:
        return bool(self.pending() or self.expandable())

    def known_signatures(self) -> set[str]:
        return {
            node.behavioral_sig
            for node in self.nodes.values()
            if isinstance(node.behavioral_sig, str) and node.behavioral_sig
        }

    def stats(self) -> dict:
        evaluated = self.evaluated()
        depths = [node.depth for node in self.nodes.values()]
        coverage = self.objective_coverage()
        return {
            "nodes": len(self.nodes),
            "roots": len(self.roots),
            "root_visits": self.root_visits,
            "evaluated": len(evaluated),
            "pending": len(self.pending()),
            "error": sum(1 for node in self.nodes.values() if node.state == ERROR),
            "pruned": self._pruned,
            "max_depth": max(depths) if depths else 0,
            "total_visits": sum(node.visits for node in self.nodes.values()),
            "total_attempts": sum(node.attempts for node in self.nodes.values()),
            "best_reward": (max(self._rewards) if self._rewards else None),
            "objectives": list(self.objectives),
            "objective_ready": coverage["ready"],
            "objective_incomplete": coverage["incomplete"],
            "objective_out_of_range": coverage["out_of_range"],
            "front_size": len(self.front_nodes()),
            "front_entries": len(self.front),
            "hypervolume": round(self.front.hypervolume(), 6),
        }

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "max_nodes": self.max_nodes,
            "max_children": self.max_children,
            "max_depth": self.max_depth,
            # Run settings a reload cannot guess; recorded so a resumed tree is
            # not reinterpreted under different objectives without a trace.
            "objectives": list(self.objectives),
            "front_size": self.front_size,
            "epsilon": self.epsilon,
            "widen_patience": self.widen_patience,
            "roots": list(self.roots),
            "root_visits": self.root_visits,
            "pruned": self._pruned,
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }

    def save(self, path: Optional[Path] = None) -> Optional[Path]:
        target = Path(path) if path else self.path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.path = target
        return target

    @classmethod
    def load(cls, path: Path, **options: Any) -> "SearchTree":
        """Load a saved tree.

        ``options`` (objectives, front_size, epsilon, widen_patience, max_depth)
        are run settings rather than state, so they come from the caller; the
        file's own values are only a fallback for older files that predate them.
        Lowering ``max_depth`` keeps deeper nodes already in the file but stops
        expanding them.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        restored: dict = {}
        if isinstance(payload.get("objectives"), list) and payload["objectives"]:
            restored["objectives"] = tuple(str(name) for name in payload["objectives"])
        for key in ("front_size", "epsilon", "widen_patience"):
            if payload.get(key) is not None:
                restored[key] = payload[key]
        options = dict(options)
        max_depth = options.pop("max_depth", None)
        tree = cls(
            path=Path(path),
            max_nodes=int(payload.get("max_nodes") or 512),
            max_children=int(payload.get("max_children") or 4),
            max_depth=int(max_depth or payload.get("max_depth") or 5),
            **{**restored, **options},
        )
        for raw in payload.get("nodes") or []:
            if not isinstance(raw, dict):
                continue
            node = TreeNode.from_dict(raw)
            tree.nodes[node.node_id] = node
        tree.roots = [str(root) for root in payload.get("roots") or [] if str(root) in tree.nodes]
        tree._pruned = int(payload.get("pruned") or 0)
        # Trees saved before the virtual root existed: one visit per subtree visit.
        tree.root_visits = int(
            payload.get("root_visits")
            or sum(tree.nodes[root].visits for root in tree.roots if root in tree.nodes)
        )
        tree._rewards = [
            float(node.reward) for node in tree.nodes.values() if node.reward is not None
        ]
        tree.rebuild_fronts()
        return tree

    # -- pruning ------------------------------------------------------------
    def _prune(self) -> None:
        """Drop the least promising unevaluated nodes to bound memory."""
        pending = sorted(
            self.pending(),
            key=lambda node: (self._parent_score(node), -node.depth, node.node_id),
        )
        while len(self.nodes) > self.max_nodes and pending:
            victim = pending.pop(0)
            self.nodes.pop(victim.node_id, None)
            if victim.parent_id and victim.parent_id in self.nodes:
                parent = self.nodes[victim.parent_id]
                if victim.node_id in parent.children:
                    parent.children.remove(victim.node_id)
            if victim.node_id in self.roots:
                self.roots.remove(victim.node_id)
            self._pruned += 1
