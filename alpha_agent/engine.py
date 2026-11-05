from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import alpha_research_pipeline as pipeline
from alpha_agent.config import AgentRuntimeConfig
from alpha_agent.planner import HeuristicPlanner, PlannerAction
from alpha_agent.research_logic import (
    ResearchNotebook,
    build_check_aware_refinements,
    build_correlation_repair_candidates,
    build_robustness_candidates,
    dedupe_candidates,
    safe_float,
)
from worldquant_brain_cli import BrainApiError, BrainClient


ApprovalCallback = Callable[[Dict[str, Any]], bool]
ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class AgentRunResult:
    run_id: str
    started_at: str
    finished_at: str
    report_path: Path
    summary: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    submissions: List[Dict[str, Any]]
    state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "report_path": str(self.report_path),
            "summary": self.summary,
            "leaderboard": self.leaderboard,
            "events": self.events,
            "submissions": self.submissions,
            "state": self.state,
        }


class ResearchToolbox:
    def __init__(self, runtime: AgentRuntimeConfig) -> None:
        self.runtime = runtime
        self.agent_cfg = runtime.agent
        self.workdir = Path(self.agent_cfg.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.results_store = pipeline.JsonlStore(self.workdir / "results.jsonl")
        self.submissions_store = pipeline.JsonlStore(self.workdir / "submissions.jsonl")
        self.state_path = self.workdir / "state.json"
        self.agent_runs_dir = self.workdir / "agent_runs"
        self.agent_runs_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[BrainClient] = None
        self.library = pipeline.load_json(Path(self.agent_cfg.idea_library))
        self.available_fields = pipeline.load_available_fields(str(self.agent_cfg.fields_summary))

        self.results: List[Dict[str, Any]] = []
        self.submissions: List[Dict[str, Any]] = []
        self.evaluated_signatures: set[str] = set()
        self.submitted_alpha_ids: set[str] = set()
        self.seed_queue: List[pipeline.Candidate] = []
        self.all_ordered_seeds: List[pipeline.Candidate] = []
        self.refresh_records()
        self.refresh_seed_queue()

    def _build_client(self) -> BrainClient:
        auth = self.runtime.auth
        client = BrainClient(
            base_url=auth.base_url,
            timeout=auth.timeout,
            cookie_header=auth.cookie_header,
        )
        client.opener.addheaders = [("User-Agent", pipeline.DEFAULT_USER_AGENT)]
        if auth.cookie_header and self._cookie_is_alive(client):
            return client
        # A stale cookie used to win even when email/password were valid, and the
        # run died mid-way with 401 (a 93-simulation run stopped at 21 that way).
        # Verify the cookie once, then fall back to a password login.
        if not auth.email or not auth.password:
            if auth.cookie_header:
                self._warn_auth("WQB_COOKIE_HEADER is expired and no email/password is set.")
            raise BrainApiError("Missing credentials. Set email/password or cookie header.")
        client.login(auth.email, auth.password)
        return client

    @staticmethod
    def _cookie_is_alive(client: BrainClient) -> bool:
        """One cheap authenticated GET: is this cookie still accepted?

        A ``429`` counts as "not verified", and the caller then prefers a real
        login over trusting a cookie it could not check — a stale cookie plus a
        rate limit was exactly how a run ended up aborting on 401 later.
        """
        try:
            response = client._request(
                "GET", "/users/self/alphas?limit=1", accept="application/json;version=2.0"
            )
        except Exception:
            return True  # fail-open: a transient blip must not force a re-login
        status = getattr(response, "status", None)
        if not isinstance(status, int):
            return True
        return status < 400 and status != 429

    @staticmethod
    def _warn_auth(message: str) -> None:
        print(f"[alpha-agent] {message}", file=sys.stderr, flush=True)

    @property
    def client(self) -> BrainClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def refresh_records(self) -> None:
        self.results = self.results_store.read_all()
        self.submissions = self.submissions_store.read_all()
        self.evaluated_signatures = {
            signature
            for signature in (pipeline.record_signature(record) for record in self.results)
            if signature
        }
        self.submitted_alpha_ids = {
            str(record.get("alpha_id"))
            for record in self.submissions
            if isinstance(record.get("alpha_id"), str)
        }

    def arena_settings(self, settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Force a candidate into the configured arena (region/universe/delay).

        The quality gates and the correlation pool are both defined per arena,
        and neither the idea library nor the model is allowed to move a
        candidate out of the one the run chose.
        """
        merged = dict(settings or {})
        merged["region"] = str(self.agent_cfg.region or merged.get("region") or "USA").upper()
        merged["universe"] = str(self.agent_cfg.universe or merged.get("universe") or "TOP3000").upper()
        try:
            merged["delay"] = int(self.agent_cfg.delay)
        except (TypeError, ValueError):
            merged["delay"] = 1
        return pipeline.normalize_settings(merged)

    def refresh_seed_queue(self) -> int:
        pivot_families = pipeline.correlation_pivot_families(self.results)
        seeds = [
            replace(candidate, settings=self.arena_settings(candidate.settings))
            for candidate in pipeline.generate_seed_candidates(
                library=self.library,
                family_filter=set(self.agent_cfg.family_filter),
                available_fields=self.available_fields,
            )
        ]
        ordered = pipeline.prioritize_seed_candidates(
            seeds=seeds,
            blocked_families=pivot_families,
            shuffle_generated=self.agent_cfg.shuffle_seeds,
            random_seed=self.agent_cfg.random_seed,
        )
        self.all_ordered_seeds = ordered
        self.seed_queue = [
            candidate
            for candidate in ordered
            if candidate.signature() not in self.evaluated_signatures
        ]
        return len(self.seed_queue)

    def order_seeds_against_pool(self, pool: Any, *, catalog: Any = None) -> Dict[str, Any]:
        """Reorder the seed queue by how little the account pool covers its data.

        Putting the pool estimate *before* the simulation is the cheap way to
        avoid SELF_CORRELATION: on 284 live simulations the only seeds that
        cleared the quality gates were built on the field groups the pool was
        already saturated in, and all of them then failed correlation.
        """
        from alpha_agent.pool_fields import collision_risk, field_swaps, field_usage, order_seeds_by_collision

        region = str(self.agent_cfg.region or "USA").upper()
        entries = list(getattr(pool, "entries", []) or [])
        scoped = [
            entry
            for entry in entries
            if getattr(entry, "region", None) in (None, "", region)
        ]
        # A pool with no alphas in this arena says nothing about collisions.
        if len(scoped) >= 20:
            entries = scoped
        # ``SELF_CORRELATION`` is blocked by alphas the account actually holds,
        # and the pool is mostly *drafts*: on this account 1070 of 1100 entries
        # were UNSUBMITTED. Scoring field saturation on all of them punished
        # ``close`` (172 drafts) although only 4 of the 30 submitted alphas use
        # it, and left the real blocker — a submitted social-buzz + intraday
        # reversal alpha in USA/TOP3000 — untouched. Prefer the submitted set when it is big
        # enough to be a signal.
        submitted = [
            entry
            for entry in entries
            if bool(getattr(entry, "submitted", False)) or str(getattr(entry, "status", "")).upper() == "ACTIVE"
        ]
        reference = submitted if len(submitted) >= 5 else entries
        usage, total = field_usage(str(getattr(entry, "expression", "") or "") for entry in reference)
        if not total:
            return {"pool_entries": len(entries), "pool_scanned": 0, "pool_fields": 0, "reordered": 0}

        cache: Dict[str, float] = {}

        def risk_of(candidate: Any) -> float:
            signature = candidate.signature()
            if signature not in cache:
                try:
                    similarity = float(pool.max_similarity(candidate.expression, candidate.settings))
                except Exception:
                    similarity = 0.0
                cache[signature] = collision_risk(candidate.expression, usage, total, similarity=similarity)
            return cache[signature]

        queue = list(self.seed_queue)
        manual = [candidate for candidate in queue if getattr(candidate, "stage", "") == "manual_seed"]
        generated = [candidate for candidate in queue if getattr(candidate, "stage", "") != "manual_seed"]

        # Seeds that are literally already in the account. Measured on this
        # account: 26 of 324 evaluated alphas were exact copies of a pool
        # formula, and they sat at positions 1-8 — so the first simulations of
        # every run re-derived formulas the account already held and could only
        # fail SELF_CORRELATION. Drop them (their field swaps stay: a swap is
        # not a duplicate).
        from alpha_agent.expr_ast import canonical_form, try_parse

        def _canon(expression: Any) -> str:
            text = str(expression or "")
            node = try_parse(text)
            if node is None:
                return " ".join(text.split())
            try:
                return canonical_form(node)
            except Exception:
                return " ".join(text.split())

        known_expressions = {_canon(getattr(entry, "expression", "")) for entry in entries}

        def _is_known(candidate: Any) -> bool:
            return _canon(candidate.expression) in known_expressions

        duplicate_seeds = [c for c in manual + generated if _is_known(c)]
        if len(manual) + len(generated) - len(duplicate_seeds) >= 1:
            manual = [c for c in manual if not _is_known(c)]
            generated = [c for c in generated if not _is_known(c)]
            dropped_duplicates = len(duplicate_seeds)
        else:
            # Everything is a duplicate: dropping them all would leave nothing
            # to search, so keep them and let the report say why.
            dropped_duplicates = 0

        # The curated seeds are the ones most likely to collide: they were
        # written before the pool existed. Keep the structure, swap the field
        # the pool already owns for one it does not.
        swaps: list[Any] = []
        known = set(self.evaluated_signatures) | {candidate.signature() for candidate in queue}
        # A VECTOR field cannot stand in for anything: the platform rejects it
        # before producing a number, so only MATRIX fields are offered.
        matrix_only = bool(catalog is not None and getattr(catalog, "has_field_types", False))
        replacement_fields = (
            catalog.matrix_fields(self.available_fields or ())
            if matrix_only
            else (self.available_fields or ())
        )
        for candidate in manual:
            for variant in field_swaps(
                candidate.expression,
                usage=usage,
                total=total,
                candidates=replacement_fields,
                limit=max(1, int(getattr(self.agent_cfg, "seed_field_swaps", 1))),
            ):
                swapped = replace(
                    candidate,
                    expression=variant,
                    stage="field_swap",
                    idea_name=f"{candidate.idea_name}.swap",
                    metadata={**(candidate.metadata or {}), "field_swap": True},
                )
                if swapped.signature() in known:
                    continue
                known.add(swapped.signature())
                swaps.append(swapped)

        before = [risk_of(candidate) for candidate in generated]
        # Curated seeds and their swaps compete on the same scale: the safest
        # variant of a proven structure should be simulated before the variant
        # whose field the pool already owns. Keeping the manual seeds strictly
        # first re-burns the first simulations of every run on a known collision.
        curated = order_seeds_by_collision(manual + swaps, risk_of)
        ordered = curated + order_seeds_by_collision(generated, risk_of)
        self.seed_queue = ordered
        after = [risk_of(candidate) for candidate in ordered[len(curated):]]
        return {
            "pool_entries": len(entries),
            "pool_submitted": len(submitted),
            "collision_reference": "submitted" if len(submitted) >= 5 else "all",
            "pool_duplicate_seeds": dropped_duplicates,
            "pool_scanned": total,
            "pool_fields": len(usage),
            "reordered": len(generated),
            "field_swaps": len(swaps),
            "mean_risk_before": round(sum(before) / len(before), 4) if before else None,
            "mean_risk_next": round(sum(after[:5]) / len(after[:5]), 4) if after else None,
            "next_seeds": [
                {
                    "expression": candidate.expression,
                    "family": candidate.family,
                    "risk": round(risk_of(candidate), 4),
                }
                for candidate in ordered[len(manual):][:5]
            ],
        }

    def pop_seed_candidates(self, count: int) -> List[pipeline.Candidate]:
        count = max(0, int(count))
        if count == 0:
            return []
        batch = self.seed_queue[:count]
        self.seed_queue = self.seed_queue[count:]
        return batch

    def leaderboard(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        return pipeline.build_leaderboard(
            self.results,
            family_filter=set(self.agent_cfg.family_filter),
            limit=limit,
        )

    def preview_refine_candidates(
        self,
        *,
        frontier_records: Sequence[Dict[str, Any]],
    ) -> Tuple[List[pipeline.Candidate], set[str], List[Dict[str, Any]]]:
        top_records = list(frontier_records[: self.agent_cfg.refine_top_k])
        blocked_families = pipeline.correlation_pivot_families(top_records)
        generic = pipeline.build_refinement_candidates(
            top_records,
            family_filter=set(self.agent_cfg.family_filter),
            blocked_families=blocked_families,
        )
        correlation_repairs = build_correlation_repair_candidates(
            top_records,
            family_filter=set(self.agent_cfg.family_filter),
        )
        check_aware = build_check_aware_refinements(
            top_records,
            family_filter=set(self.agent_cfg.family_filter),
            blocked_families=blocked_families,
        )
        merged = dedupe_candidates(correlation_repairs + check_aware + generic)
        fresh = [
            candidate
            for candidate in merged
            if candidate.signature() not in self.evaluated_signatures
        ]
        return fresh, blocked_families, top_records

    def preview_diversification_candidates(self, blocked_families: set[str]) -> List[pipeline.Candidate]:
        return pipeline.build_diversification_candidates(
            seeds=self.all_ordered_seeds,
            excluded_signatures=self.evaluated_signatures,
            blocked_families=blocked_families,
        )

    def preview_robustness_candidates(
        self,
        *,
        frontier_records: Sequence[Dict[str, Any]],
    ) -> List[pipeline.Candidate]:
        candidates = build_robustness_candidates(
            frontier_records,
            family_filter=set(self.agent_cfg.family_filter),
            top_k=self.agent_cfg.robustness_top_k,
        )
        return [
            candidate
            for candidate in candidates
            if candidate.signature() not in self.evaluated_signatures
        ]

    def evaluate_candidates(
        self,
        candidates: Sequence[pipeline.Candidate],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        submission_attempts: List[Dict[str, Any]] = []

        def emit_progress(payload: Dict[str, Any]) -> None:
            if payload.get("type") == "candidate_completed":
                self.refresh_records()
                payload["leaderboard"] = [
                    pipeline.compact_record(item)
                    for item in self.leaderboard(limit=8)
                ]
                payload["state"] = self.write_state()
            if progress_callback:
                progress_callback(payload)

        evaluated = pipeline.evaluate_batch(
            client=self.client,
            candidates=candidates,
            results_store=self.results_store,
            submissions_store=self.submissions_store,
            submitted_alpha_ids=self.submitted_alpha_ids,
            max_wait=self.agent_cfg.max_wait,
            poll_interval=self.agent_cfg.poll_interval,
            retries=self.agent_cfg.retries,
            sleep_between=self.agent_cfg.sleep_between,
            should_attempt_submit=False,
            allow_pending_checks=self.agent_cfg.allow_pending_checks,
            stop_on_submittable=False,
            submission_attempts=submission_attempts,
            progress_callback=emit_progress,
        )
        self.refresh_records()
        return evaluated

    def best_submittable_candidate(self, *, target_alpha_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        allow_pending = self.agent_cfg.allow_pending_checks
        submitted = set(getattr(self, "submitted_alpha_ids", ()) or ())
        for record in self.leaderboard(limit=200):
            alpha_id = record.get("alpha_id")
            if not isinstance(alpha_id, str):
                continue
            if target_alpha_id and alpha_id != target_alpha_id:
                continue
            # A submitted alpha stays "ready" by its checks; without this skip it
            # kept the top slot and blocked every later submission of the run.
            if alpha_id in submitted:
                continue
            if not pipeline.record_is_submit_ready(record, allow_pending_checks=allow_pending):
                continue
            return record
        return None

    def submit_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        alpha_id = str(record["alpha_id"])
        submission = pipeline.attempt_submit(
            client=self.client,
            alpha_id=alpha_id,
            record=record,
            submissions_store=self.submissions_store,
            allow_pending_checks=self.agent_cfg.allow_pending_checks,
            max_wait=self.agent_cfg.max_wait,
            poll_interval=self.agent_cfg.poll_interval,
        )
        self.refresh_records()
        return submission

    def write_state(self) -> Dict[str, Any]:
        leaderboard = self.leaderboard(limit=10)
        return pipeline.write_state(
            self.state_path,
            self.results,
            self.submissions,
            leaderboard,
        )


class AlphaResearchAgent:
    def __init__(
        self,
        runtime: AgentRuntimeConfig,
        planner: Optional[Any] = None,
        approval_callback: Optional[ApprovalCallback] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.runtime = runtime
        self.planner = planner or HeuristicPlanner()
        self.toolbox = ResearchToolbox(runtime)
        self.approval_callback = approval_callback
        self.progress_callback = progress_callback
        self.events: List[Dict[str, Any]] = []
        self.submission_attempts: List[Dict[str, Any]] = []
        self.seed_evaluated_count = 0
        self._search_report: Optional[Dict[str, Any]] = None
        self.notebook = ResearchNotebook(
            budget=max(1, int(runtime.agent.budget)),
            max_family_budget_share=runtime.agent.max_family_budget_share,
            min_expression_novelty=runtime.agent.min_expression_novelty,
            robustness_score_threshold=runtime.agent.robustness_score_threshold,
        )

    def run(self) -> AgentRunResult:
        """Dispatch on the configured search mode.

        ``legacy`` is the original stage-machine loop and remains the default.
        ``hybrid`` runs the MCTS + quality-diversity search added alongside it.
        Both share the same toolbox, gates, artifacts and submission governance.
        """
        mode = str(getattr(self.runtime.agent, "search_mode", "legacy") or "legacy").lower().replace("-", "_")
        if mode == "hybrid":
            return self._run_hybrid()
        if mode == "mo_hifo":
            return self._run_mo_hifo()
        return self._run_legacy()

    def _run_legacy(self) -> AgentRunResult:
        started_at = pipeline.iso_now()
        run_id = self._build_run_id(started_at)
        budget = max(0, int(self.runtime.agent.budget))
        seed_target = (
            min(budget, max(1, int(math.ceil(budget * float(self.runtime.agent.seed_fraction)))))
            if budget > 0
            else 0
        )
        evaluated_total = 0
        evaluated_this_run = 0
        stop_reason = "Completed."
        iteration_executed = 0
        current_stage = "explore"
        self._emit_progress(
            {
                "type": "run_started",
                "run_id": run_id,
                "started_at": started_at,
                "budget": budget,
                "max_iterations": self.runtime.agent.max_iterations,
                "workdir": str(self.runtime.agent.workdir),
            }
        )

        for iteration in range(1, self.runtime.agent.max_iterations + 1):
            iteration_executed = iteration
            remaining_budget = budget - evaluated_total
            if remaining_budget <= 0:
                stop_reason = "Budget exhausted."
                break

            if bool(getattr(self.runtime.agent, "session_ping", True)):
                session_status = self._session_status()
                if session_status is not None:
                    stop_reason = (
                        f"Session expired before iteration {iteration} ({session_status}). "
                        "Refresh WQB_COOKIE_HEADER and rerun; stored results resume and no "
                        "simulation was spent this iteration."
                    )
                    self._append_event(
                        iteration=iteration,
                        stage="session",
                        action="stop",
                        rationale=stop_reason,
                        details={"auth_status": session_status, "budget_spent": evaluated_total},
                        hypothesis="The session cookie reached the end of its lifetime.",
                        risk_note="Every further request would fail identically.",
                    )
                    break

            self.toolbox.refresh_seed_queue()
            frontier = self.toolbox.leaderboard(limit=max(20, self.runtime.agent.refine_top_k))
            current_stage, stage_reason = self.notebook.determine_stage(
                iteration=iteration,
                evaluated_total=evaluated_total,
                seed_target=seed_target,
                leaderboard=frontier,
                records=self.toolbox.results,
            )
            refine_candidates, blocked_families, _ = self.toolbox.preview_refine_candidates(
                frontier_records=frontier
            )
            diversify_candidates = self.toolbox.preview_diversification_candidates(blocked_families)
            robustness_candidates = self.toolbox.preview_robustness_candidates(frontier_records=frontier)

            ranked_refine = self.notebook.rank_candidates(
                candidates=refine_candidates,
                frontier_records=frontier,
                stage=current_stage,
                blocked_families=blocked_families,
            )
            ranked_diversify = self.notebook.rank_candidates(
                candidates=diversify_candidates,
                frontier_records=frontier,
                stage=current_stage,
                blocked_families=blocked_families,
            )
            ranked_robustness = self.notebook.rank_candidates(
                candidates=robustness_candidates,
                frontier_records=frontier,
                stage="robustness",
                blocked_families=set(),
            )

            context = self._build_context(
                iteration=iteration,
                budget=budget,
                evaluated_total=evaluated_total,
                seed_target=seed_target,
                stage=current_stage,
                stage_reason=stage_reason,
                refine_candidates=ranked_refine,
                diversify_candidates=ranked_diversify,
                robustness_candidates=ranked_robustness,
                frontier=frontier,
            )
            decision = self._decide(context).clamped(remaining_budget)
            self.notebook.record_hypothesis(
                iteration=iteration,
                stage=current_stage,
                action=decision.action,
                hypothesis=decision.hypothesis or "No explicit hypothesis provided.",
                rationale=decision.rationale,
                focus_family=decision.focus_family,
                risk_note=decision.risk_note,
            )
            self._emit_progress(
                {
                    "type": "planner_decision",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": current_stage,
                    "stage_reason": stage_reason,
                    "decision": decision.raw or {},
                    "action": decision.action,
                    "batch_size": decision.batch_size,
                    "rationale": decision.rationale,
                    "hypothesis": decision.hypothesis,
                    "risk_note": decision.risk_note,
                    "context": context,
                    "leaderboard": [pipeline.compact_record(item) for item in frontier[:8]],
                }
            )

            if decision.action == "stop":
                stop_reason = decision.rationale or "Planner requested stop."
                self._append_event(
                    iteration=iteration,
                    stage=current_stage,
                    action="stop",
                    rationale=stop_reason,
                    details={"planner_context": context},
                    hypothesis=decision.hypothesis,
                    risk_note=decision.risk_note,
                )
                break

            if decision.action == "submit_best":
                submission_outcome = self._execute_submit_action(decision, iteration, current_stage)
                if submission_outcome is not None:
                    self.submission_attempts.append(submission_outcome)
                self.toolbox.write_state()
                continue

            batch: List[pipeline.Candidate] = []
            if decision.action == "evaluate_seed":
                raw_seed_batch = self.toolbox.pop_seed_candidates(max(decision.batch_size * 2, decision.batch_size))
                batch = self._select_batch(
                    candidates=raw_seed_batch,
                    batch_size=decision.batch_size,
                    stage=current_stage,
                    frontier=frontier,
                    blocked_families=blocked_families,
                )
            elif decision.action == "evaluate_refine":
                batch = self._select_batch(
                    candidates=ranked_refine,
                    batch_size=decision.batch_size,
                    stage=current_stage,
                    frontier=frontier,
                    blocked_families=blocked_families,
                )
            elif decision.action == "evaluate_diversify":
                batch = self._select_batch(
                    candidates=ranked_diversify,
                    batch_size=decision.batch_size,
                    stage=current_stage,
                    frontier=frontier,
                    blocked_families=blocked_families,
                )
            elif decision.action == "evaluate_robustness":
                batch = self._select_batch(
                    candidates=ranked_robustness,
                    batch_size=decision.batch_size,
                    stage="robustness",
                    frontier=frontier,
                    blocked_families=set(),
                    enforce_family_cap=False,
                )

            if not batch:
                stop_reason = f"No candidates available for {decision.action}."
                self._append_event(
                    iteration=iteration,
                    stage=current_stage,
                    action=decision.action,
                    rationale=decision.rationale,
                    details={"requested_batch": decision.batch_size, "executed_batch": 0},
                    hypothesis=decision.hypothesis,
                    risk_note=decision.risk_note,
                )
                continue

            self._emit_progress(
                {
                    "type": "batch_started",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": current_stage,
                    "action": decision.action,
                    "batch_size": len(batch),
                    "candidates": [candidate.to_record() for candidate in batch],
                    "remaining_budget": remaining_budget,
                }
            )

            records = self.toolbox.evaluate_candidates(
                batch,
                progress_callback=lambda payload, iteration=iteration, stage=current_stage, action=decision.action: self._emit_progress(
                    {
                        **payload,
                        "run_id": run_id,
                        "iteration": iteration,
                        "stage": stage,
                        "action": action,
                        "budget": budget,
                    }
                ),
            )
            self.notebook.observe_records(records)
            failed_records = [record for record in records if record.get("status") != "ok"]
            auth_status = self._auth_error_statuses(records)
            if auth_status is not None:
                stop_reason = (
                    f"Authentication rejected ({auth_status}). Refresh WQB_COOKIE_HEADER; "
                    "failed requests did not consume simulation budget."
                )
                self._append_event(
                    iteration=iteration,
                    stage=current_stage,
                    action="stop",
                    rationale=stop_reason,
                    details={"auth_status": auth_status, "failed_records": len(failed_records)},
                    hypothesis="The session cookie expired mid-run.",
                    risk_note="Without valid auth every further request would fail the same way.",
                )
                break
            successful = len(records) - len(failed_records)
            evaluated_total += successful
            evaluated_this_run += successful
            if decision.action == "evaluate_seed":
                self.seed_evaluated_count += successful

            event_details = {
                "requested_batch": decision.batch_size,
                "executed_batch": len(records),
                "ok_count": sum(1 for record in records if record.get("status") == "ok"),
                "error_count": sum(1 for record in records if record.get("status") == "error"),
                "quality_ready_count": sum(1 for record in records if pipeline.record_quality_ready(record)),
                "submittable_count": sum(1 for record in records if pipeline.record_is_submit_ready(record)),
                "correlation_blocked_count": sum(
                    1
                    for record in records
                    if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
                ),
                "families": sorted({str(record.get("family") or "") for record in records}),
                "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results, top_k=5),
                "best_score_after": self._best_score(),
                "best_alpha_after": self._best_alpha_id(),
            }
            self._append_event(
                iteration=iteration,
                stage=current_stage,
                action=decision.action,
                rationale=decision.rationale,
                details=event_details,
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            self.toolbox.write_state()
            self._emit_progress(
                {
                    "type": "batch_completed",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": current_stage,
                    "action": decision.action,
                    "details": event_details,
                    "leaderboard": [
                        pipeline.compact_record(item)
                        for item in self.toolbox.leaderboard(limit=8)
                    ],
                    "state": self.toolbox.write_state(),
                }
            )

        finished_at = pipeline.iso_now()
        leaderboard = [pipeline.compact_record(item) for item in self.toolbox.leaderboard(limit=10)]
        state = self.toolbox.write_state()
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "budget": budget,
            "evaluated_count": len(self.toolbox.results),
            "evaluated_this_run": evaluated_this_run,
            "seed_evaluated_this_run": self.seed_evaluated_count,
            "submission_attempts_this_run": len(self.submission_attempts),
            "best_score": self._best_score(),
            "best_alpha_id": self._best_alpha_id(),
            "quality_ready_count": sum(1 for record in self.toolbox.results if pipeline.record_quality_ready(record)),
            "submittable_count": sum(
                1 for record in self.toolbox.results if pipeline.record_is_submit_ready(record)
            ),
            "correlation_blocked_count": sum(
                1
                for record in self.toolbox.results
                if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
            ),
            "iterations_executed": iteration_executed,
            "stop_reason": stop_reason,
            "submission_mode": self.runtime.agent.submission_mode,
            "final_stage": current_stage,
            "stage_history": self.notebook.stage_history,
            "family_stats": self.notebook.family_stats(self.toolbox.results),
            "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results),
            "hypothesis_log_tail": self.notebook.recent_hypotheses(limit=8),
        }
        report_path = self._write_run_report(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            leaderboard=leaderboard,
            state=state,
        )
        result = AgentRunResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            report_path=report_path,
            summary=summary,
            leaderboard=leaderboard,
            events=self.events,
            submissions=self.submission_attempts,
            state=state,
        )
        self._emit_progress({"type": "run_finished", "run_id": run_id, "result": result.to_dict()})
        return result

    # ------------------------------------------------------------------
    # Hybrid search: MCTS inner loop + quality-diversity outer loop
    # ------------------------------------------------------------------
    @staticmethod
    def _failure_summaries(
        records: List[Dict[str, Any]], *, parent_key: Optional[str] = None, limit: int = 8
    ) -> List[Dict[str, Any]]:
        """Recent attempts that did not pass quality, newest first, as negative examples.

        With ``parent_key`` only children of that tree node are returned, i.e. what
        was already tried from the parent the model is about to refine.
        """
        failures: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for record in reversed(records):
            if parent_key is not None and record.get("parent_key") != parent_key:
                continue
            if record.get("status") == "ok" and record.get("quality_checks_ready"):
                continue
            expression = str(record.get("expression") or "")
            if not expression or expression in seen:
                continue
            seen.add(expression)
            metadata = record.get("metadata") or {}
            operator = re.match(r"\[(\w+)\]", str(metadata.get("rationale") or ""))
            item: Dict[str, Any] = {
                "expression": expression,
                "idea": metadata.get("hypothesis") or record.get("idea_name"),
                "operator": operator.group(1) if operator else None,
            }
            if record.get("status") == "ok":
                metrics = record.get("metrics") or {}
                item.update(
                    sharpe=metrics.get("sharpe"),
                    fitness=metrics.get("fitness"),
                    turnover=metrics.get("turnover"),
                    failed_checks=record.get("failed_checks") or [],
                )
            else:
                item["error"] = str((record.get("error") or {}).get("message") or record.get("status"))[:160]
            failures.append(item)
            if len(failures) >= limit:
                break
        return failures

    @staticmethod
    def _elite_summaries(frontier: List[Dict[str, Any]], *, limit: int = 5, per_family: int = 2) -> List[Dict[str, Any]]:
        """Top evaluated alphas for the LLM's population-level operators (e2, s1).

        Capped per family so the population is not five window variants of one idea.
        """
        elites: List[Dict[str, Any]] = []
        family_counts: Counter[str] = Counter()
        for record in frontier:
            if record.get("status") != "ok" or not record.get("expression"):
                continue
            family = str(record.get("family") or "")
            if family_counts[family] >= per_family:
                continue
            family_counts[family] += 1
            metrics = record.get("metrics") or {}
            self_corr = next(
                (
                    item.get("value")
                    for item in (record.get("check_diagnostics") or [])
                    if isinstance(item, dict) and item.get("name") == "SELF_CORRELATION"
                ),
                None,
            )
            elites.append(
                {
                    "expression": record.get("expression"),
                    "idea": (record.get("metadata") or {}).get("hypothesis") or record.get("idea_name"),
                    "family": family,
                    "sharpe": metrics.get("sharpe"),
                    "fitness": metrics.get("fitness"),
                    "turnover": metrics.get("turnover"),
                    "self_correlation": self_corr,
                    "failed_checks": record.get("failed_checks") or [],
                }
            )
            if len(elites) >= limit:
                break
        return elites

    # -- hybrid-search helpers (MCTS-AHD descent + Pareto fronts) ---------------
    @staticmethod
    def _record_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Latest record by alpha id and by expression, for node summaries."""
        index: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if record.get("status") != "ok":
                continue
            if record.get("alpha_id"):
                index[f"id:{record['alpha_id']}"] = record
            if record.get("expression"):
                index[f"expr:{record['expression']}"] = record
        return index

    @staticmethod
    def _record_summary(record: Dict[str, Any], *, expression: str = "", family: str = "") -> Dict[str, Any]:
        metrics = record.get("metrics") or {}
        self_corr = next(
            (
                item.get("value")
                for item in (record.get("check_diagnostics") or [])
                if isinstance(item, dict) and item.get("name") == "SELF_CORRELATION"
            ),
            None,
        )
        return {
            "expression": record.get("expression") or expression,
            "idea": (record.get("metadata") or {}).get("hypothesis") or record.get("idea_name"),
            "family": record.get("family") or family,
            "sharpe": metrics.get("sharpe"),
            "fitness": metrics.get("fitness"),
            "turnover": metrics.get("turnover"),
            "self_correlation": self_corr,
            "failed_checks": record.get("failed_checks") or [],
        }

    def _node_summary(self, node: Any, index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        record = index.get(f"id:{node.alpha_id}") or index.get(f"expr:{node.expression}") or {}
        return self._record_summary(record, expression=node.expression, family=node.family)

    def _subtree_references(self, tree: Any, index: Dict[str, Dict[str, Any]], rng: Any) -> List[Dict[str, Any]]:
        """MCTS-AHD e1 input: one random evaluated alpha from every subtree."""
        references: List[Dict[str, Any]] = []
        for root_id in tree.roots:
            members = [member for member in tree.subtree(root_id) if member.is_evaluated]
            if members:
                references.append(self._node_summary(rng.choice(members), index))
        return references[:8]

    def _front_elite_summaries(
        self, tree: Any, index: Dict[str, Dict[str, Any]], *, limit: int = 5, per_family: int = 2
    ) -> List[Dict[str, Any]]:
        """Global Pareto front members, spread over the front, capped per family.

        Taking every ``step``-th member of the front (sorted on the first
        objective) gives e2 parents from different trade-off regions rather
        than the top of one axis.
        """
        members = tree.front_nodes()
        if not members:
            return []
        step = max(1, len(members) // max(1, limit))
        ordered = members[::step] + [node for index_, node in enumerate(members) if index_ % step]
        elites: List[Dict[str, Any]] = []
        family_counts: Counter[str] = Counter()
        for node in ordered:
            if family_counts[node.family] >= per_family:
                continue
            record = index.get(f"id:{node.alpha_id}") or index.get(f"expr:{node.expression}")
            if not record:
                continue
            family_counts[node.family] += 1
            summary = self._record_summary(record, expression=node.expression, family=node.family)
            summary["family"] = node.family
            summary["objectives"] = {name: round(value, 4) for name, value in node.objectives.items()}
            elites.append(summary)
            if len(elites) >= limit:
                break
        return elites

    @staticmethod
    def _child_family(parent: Any, expression: str) -> str:
        """Inherit the parent's family unless the child reads different data groups."""
        from alpha_agent.families import data_groups, lineage_family

        if parent is None:
            return lineage_family(expression)
        if data_groups(expression) == data_groups(str(parent.expression)):
            return str(parent.family or lineage_family(str(parent.expression)))
        return lineage_family(expression)

    def _relabel_tree_families(self, tree: Any) -> None:
        """Re-derive every non-root family from its lineage (roots keep theirs)."""
        from alpha_agent.families import lineage_family

        stack = []
        for root_id in tree.roots:
            root = tree.nodes.get(root_id)
            if root is None:
                continue
            root.family = root.family or lineage_family(root.expression)
            stack.append(root)
        while stack:
            parent = stack.pop()
            for child in tree.children_of(parent):
                child.family = self._child_family(parent, child.expression)
                stack.append(child)

    #: Simulation metrics a Pareto objective can be built from. All of them are
    #: stored on the node when the record has them; ``mo_objectives`` only
    #: selects which are used, so changing the objective set never needs a
    #: re-simulation.
    OBJECTIVE_METRICS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin")

    @classmethod
    def _metric_objectives(cls, record: Dict[str, Any]) -> Dict[str, float]:
        """Raw simulation metrics used as Pareto objectives (absent ones skipped)."""
        metrics = record.get("metrics") or {}
        out: Dict[str, float] = {}
        for name in cls.OBJECTIVE_METRICS:
            try:
                out[name] = float(metrics[name])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @classmethod
    def _backfill_objectives(cls, tree: Any, pool: Any, records: Optional[List[Dict[str, Any]]] = None) -> int:
        """Objectives missing from nodes evaluated before they were recorded.

        Uses the stored gate margins, the matching result record and the
        cached pool, so it costs no simulation. Without it, pre-existing
        subtrees would sit off the Pareto fronts and score zero under
        ``tree_value="ohvi"``. Keys a node already has are never overwritten.
        """
        from alpha_agent.reward import GateAssessment

        index = cls._record_index(records or [])
        filled = 0
        for node in tree.evaluated():
            missing = [name for name in tree.objectives if name not in node.objectives]
            if not missing:
                continue
            derived: Dict[str, float] = {}
            record = index.get(f"id:{node.alpha_id}") or index.get(f"expr:{node.expression}")
            if record:
                derived.update(cls._metric_objectives(record))
            if node.margins:
                assessment = GateAssessment(margins=dict(node.margins))
                derived["quality"] = assessment.quality_margin
                derived["correlation"] = assessment.novelty_margin
            if "divergence" in missing:
                try:
                    derived["divergence"] = float(
                        pool.novelty(
                            node.expression,
                            node.settings,
                            exclude_alpha_ids={node.alpha_id} if node.alpha_id else None,
                        )
                    )
                except Exception:
                    derived["divergence"] = 1.0
            added = {name: value for name, value in derived.items() if name not in node.objectives}
            if added:
                node.objectives = {**node.objectives, **added}
                filled += 1
        if filled:
            tree.rebuild_fronts()
        return filled

    @staticmethod
    def _learn_field_types(
        catalog: Any,
        expressions: Sequence[str],
        *,
        cache_path: Any = None,
        arena: Optional[Dict[str, Any]] = None,
        client_factory: Any = None,
    ) -> int:
        """Resolve the types of fields the proposer used but the cache lacks.

        One request per new field (the listing endpoint has no bulk id filter),
        cached to disk, so the cost is paid once per field per arena. Fail-open:
        an unknown type only means the VECTOR check cannot judge that field.
        """
        from alpha_agent.catalog import remember_field_types
        from alpha_agent.pool_fields import expression_fields

        if catalog is None or not getattr(catalog, "has_field_types", False):
            return 0
        unknown = sorted(
            {
                field
                for expression in expressions
                for field in expression_fields(str(expression or ""))
            }
            - set(catalog.field_types)
        )
        if not unknown or client_factory is None:
            return 0
        try:
            client = client_factory()
            payload = client.fetch_data_fields(
                region=str((arena or {}).get("region") or "USA"),
                universe=str((arena or {}).get("universe") or "TOP3000"),
                delay=int((arena or {}).get("delay") or 1),
                ids=unknown[:20],
            )
        except Exception:
            return 0
        types = {
            str(item["id"]): str(item["type"])
            for item in payload or []
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("type")
        }
        return remember_field_types(catalog, types, cache_path=cache_path, arena=arena)

    def _run_hybrid(self) -> AgentRunResult:
        """Multi-objective tree search over formula refinements.

        Per iteration: descend by hypervolume UCT -> propose refinements ->
        offline filters -> evaluate the survivors -> record reward and
        objectives, backprop scalar Q and Pareto fronts.
        Every free filter runs before a simulation is spent.
        """
        from alpha_agent.catalog import Catalog, load_catalog
        from alpha_agent.novelty_pool import NoveltyPool
        import random

        from alpha_agent.refine_llm import (
            WIDEN_OPERATOR,
            DeterministicRefinementProposer,
            FilterOutcome,
            LLMRefinementProposer,
            filter_candidates,
            leaf_operators,
        )
        from alpha_agent.reward import assess_record
        from alpha_agent.search_tree import EVALUATED, SearchTree
        from alpha_agent.expr_ast import subtree_hashes, try_parse

        cfg = self.runtime.agent
        started_at = pipeline.iso_now()
        run_id = self._build_run_id(started_at)
        budget = max(0, int(cfg.budget))
        workdir = Path(cfg.workdir)

        tree_path = Path(cfg.tree_state_path) if cfg.tree_state_path else workdir / "search_tree.json"
        pool_path = Path(cfg.novelty_pool_path) if cfg.novelty_pool_path else workdir / "alpha_pool.json"
        operators_path = workdir / "operators.json"

        front_options = {
            "objectives": tuple(cfg.mo_objectives),
            "front_size": int(cfg.mo_front_size),
            "epsilon": float(cfg.mo_epsilon),
            "widen_patience": int(cfg.widen_patience),
        }
        tree = SearchTree(
            tree_path,
            max_nodes=cfg.tree_max_nodes,
            max_children=cfg.tree_max_children,
            max_depth=cfg.tree_max_depth,
            **front_options,
        )
        if tree_path.exists():
            try:
                tree = SearchTree.load(tree_path, max_depth=cfg.tree_max_depth, **front_options)
            except Exception:
                pass  # Corrupt file: keep the fresh tree built above.
        # Lineage families. Idempotent, so it also migrates older trees.
        self._relabel_tree_families(tree)
        rng = random.Random(int(cfg.random_seed))
        widen_turn = [0]

        try:
            pool = NoveltyPool.load_or_fetch(
                pool_path,
                client_factory=(lambda: self.toolbox.client) if cfg.allow_offline_fetch else None,
                refresh=cfg.novelty_refresh,
                limit=cfg.novelty_pool_limit,
            )
        except Exception:
            pool = NoveltyPool()

        self._backfill_objectives(tree, pool, self.toolbox.results)

        # Only the fields the seeds actually read: the listing endpoint cannot
        # be paged cheaply, and these are the fields a rejected simulation
        # would have complained about.
        from alpha_agent.pool_fields import expression_fields

        seed_field_ids = sorted(
            {
                field
                for candidate in list(getattr(self.toolbox, "all_ordered_seeds", []))[:400]
                for field in expression_fields(candidate.expression)
            }
        )
        try:
            catalog = load_catalog(
                fields_summary=Path(cfg.fields_summary),
                operators_cache=operators_path,
                client_factory=(lambda: self.toolbox.client) if cfg.allow_offline_fetch else None,
                arena={
                    "region": cfg.region,
                    "universe": cfg.universe,
                    "delay": int(cfg.delay),
                },
                field_types_cache=workdir / "data_fields.json",
                field_ids=seed_field_ids,
            )
        except Exception:
            catalog = Catalog()

        # Seed ordering runs after the catalog so field substitutions can be
        # restricted to MATRIX fields (a VECTOR replacement is a guaranteed
        # rejected simulation).
        seed_order: Dict[str, Any] = {"pool_entries": 0, "pool_fields": 0, "reordered": 0}
        if bool(getattr(cfg, "pool_aware_seeds", True)):
            try:
                seed_order = self.toolbox.order_seeds_against_pool(pool, catalog=catalog)
            except Exception as exc:  # never let seed ordering stop a run
                seed_order = {"error": str(exc)[:160]}
            self._emit_progress(
                {
                    "type": "seed_collision_scan",
                    "run_id": run_id,
                    "stage": "explore",
                    "action": "order_seeds",
                    "arena": {
                        "region": cfg.region,
                        "universe": cfg.universe,
                        "delay": cfg.delay,
                    },
                    "catalog": {
                        "field_types": len(catalog.field_types),
                        "vector_fields": sum(
                            1 for kind in catalog.field_types.values() if kind == "VECTOR"
                        ),
                    },
                    "seed_order": seed_order,
                }
            )

        # Seeds are added straight to the tree (add_root), so the offline filter
        # that guards proposals never sees them: a seeded formula that hands a
        # VECTOR field to ts_decay_linear still costs a simulation. Drop them
        # here, now that the types are known.
        if getattr(catalog, "has_field_types", False):
            from alpha_agent.expr_ast import try_parse as _try_parse

            kept: List[Any] = []
            dropped = 0
            for candidate in list(self.toolbox.seed_queue):
                parsed = _try_parse(candidate.expression)
                if parsed is not None and catalog.type_violations(parsed):
                    dropped += 1
                    continue
                kept.append(candidate)
            self.toolbox.seed_queue = kept
            seed_order["seeds_dropped_vector"] = dropped

        fallback = DeterministicRefinementProposer(max_proposals=max(4, int(cfg.refine_proposals) * 2))
        if cfg.llm_refine_enabled and self.runtime.model.provider == "openai":
            proposer: Any = LLMRefinementProposer(
                self.runtime.model, fallback=fallback, max_calls=int(cfg.max_llm_calls)
            )
        else:
            proposer = fallback

        filters = FilterOutcome()
        subtree_counter: Counter[str] = Counter()
        evaluated_total = 0
        evaluated_this_run = 0
        iteration_executed = 0
        stop_reason = "Completed."
        stage = "explore"
        blacklist: List[str] = []
        consecutive_failures = 0
        #: What the descent actually used; ``ohvi`` unless the tree had no
        #: usable objective vector (see the pareto_blind event below).
        descent_value = str(cfg.tree_value)
        #: Set once if the multi-objective value had to be disabled; reported at
        #: the end so a run cannot silently claim it searched on the Pareto front.
        pareto_blind_reason: Optional[str] = None

        self._emit_progress(
            {
                "type": "run_started",
                "run_id": run_id,
                "started_at": started_at,
                "budget": budget,
                "max_iterations": cfg.max_iterations,
                "workdir": str(cfg.workdir),
                "search_mode": "hybrid",
            }
        )

        for iteration in range(1, int(cfg.max_iterations) + 1):
            iteration_executed = iteration
            remaining = budget - evaluated_total
            if remaining <= 0:
                stop_reason = "Budget exhausted."
                break

            if bool(getattr(cfg, "session_ping", True)):
                session_status = self._session_status()
                if session_status is not None:
                    stop_reason = (
                        f"Session expired before iteration {iteration} ({session_status}). "
                        "Refresh WQB_COOKIE_HEADER and rerun; the tree and results "
                        "resume from disk, and no simulation was spent this iteration."
                    )
                    self._append_event(
                        iteration=iteration,
                        stage="session",
                        action="stop",
                        rationale=stop_reason,
                        details={"auth_status": session_status, "budget_spent": evaluated_total},
                        hypothesis="The session cookie reached the end of its lifetime.",
                        risk_note="Every further request would fail identically.",
                    )
                    break

            if cfg.require_llm and not (
                isinstance(proposer, LLMRefinementProposer) and proposer.enabled
            ):
                stop_reason = (
                    "require_llm is set but no LLM refinement proposer is available "
                    "(check --planner-provider openai, --planner-api-key-env and --no-llm-refine)."
                )
                break

            frontier = self.toolbox.leaderboard(limit=max(20, int(cfg.refine_top_k)))
            stage, stage_reason = self.notebook.determine_stage(
                iteration=iteration,
                evaluated_total=evaluated_total,
                seed_target=max(1, int(math.ceil(budget * float(cfg.seed_fraction)))) if budget else 0,
                leaderboard=frontier,
                records=self.toolbox.results,
            )
            best_record = frontier[0] if frontier else {}
            best_diag = {
                str(item.get("name")): item
                for item in (best_record.get("check_diagnostics") or [])
                if isinstance(item, dict)
            }
            context = {
                "iteration": iteration,
                "remaining_budget": remaining,
                # The proposer must know which arena it is searching, or it
                # invents formulas for fields the universe does not have.
                "arena": {
                    "region": cfg.region,
                    "universe": cfg.universe,
                    "delay": int(cfg.delay),
                },
                "best_score": best_record.get("score"),
                "best_expression": best_record.get("expression"),
                "best_sharpe": (best_record.get("metrics") or {}).get("sharpe"),
                "best_fitness": (best_record.get("metrics") or {}).get("fitness"),
                "best_self_correlation": (best_diag.get("SELF_CORRELATION") or {}).get("value"),
                "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results, top_k=5),
                # Pareto-front members, so e2 sees different trade-offs, not five variants.
                "elites": (
                    self._front_elite_summaries(tree, self._record_index(self.toolbox.results))
                    or self._elite_summaries(frontier)
                ),
                "failed_attempts": self._failure_summaries(self.toolbox.results),
            }

            llm_roots = str(getattr(cfg, "root_source", "library")) == "llm"
            if not tree.nodes and not llm_roots:
                initial = int(cfg.initial_seed_count)
                self._seed_hybrid_roots(
                    tree,
                    count=min(remaining, initial) if initial > 0 else max(1, min(remaining, int(cfg.tree_batch_size))),
                    pool=pool,
                    diverse=initial > 0,
                )
            if not tree.nodes and not isinstance(proposer, LLMRefinementProposer):
                # No seeds and no model to invent a root (e1): nothing to search.
                stop_reason = "Search tree exhausted."
                break

            blacklist = [
                item
                for item, count in subtree_counter.most_common(int(cfg.frequent_subtree_topk) * 4)
                if count >= max(2, int(cfg.subtree_min_support))
            ][: int(cfg.frequent_subtree_topk)]
            batch: List[pipeline.Candidate] = []
            node_for_signature: Dict[str, str] = {}
            known = tree.known_signatures() | set(self.toolbox.evaluated_signatures)
            llm_down = False
            jobs_log: List[Dict[str, Any]] = []
            record_index = self._record_index(self.toolbox.results)

            def queue(node: Any, candidate: pipeline.Candidate) -> None:
                batch.append(candidate)
                node_for_signature[candidate.signature()] = node.node_id
                if node.behavioral_sig:
                    known.add(node.behavioral_sig)

            def expand(node: Any, operator: str) -> Optional[int]:
                """One operator call on ``node`` (None = virtual root). None means LLM down."""
                extra: Dict[str, Any] = {}
                if node is None:
                    references = self._subtree_references(tree, record_index, rng)
                    if not references and operator == "e1" and tree.roots:
                        # Roots written this iteration are not simulated yet, but
                        # e1 still has to differ from them (MCTS-AHD evaluates
                        # each initial root before the next e1 call).
                        references = [self._node_summary(tree.nodes[root], record_index) for root in tree.roots][:8]
                    extra["references"] = references
                else:
                    extra["failed_children"] = self._failure_summaries(
                        self.toolbox.results, parent_key=node.node_id, limit=6
                    )
                    if operator == "s1":
                        extra["path"] = [
                            self._node_summary(member, record_index)
                            for member in tree.path_to_root(node.node_id)
                        ]
                proposals = proposer.propose(
                    node=node,
                    e_vector=(node.e_vector or node.margins) if node is not None else {},
                    context={**context, **extra},
                    blacklist=blacklist,
                    operators=catalog.operators,
                    fields=catalog.fields,
                    k=int(cfg.refine_proposals),
                    operator=operator,
                )
                failures = int(getattr(proposer, "consecutive_failures", 0))
                if cfg.require_llm and failures > 0:
                    # Discard the deterministic fallback: the user asked for LLM edits only.
                    return None if failures >= max(1, int(cfg.llm_failure_limit)) else 0
                # The proposer invents field names; without their type the
                # offline filter cannot tell a matrix from an event field, and
                # the simulation is rejected for it.
                self._learn_field_types(
                    catalog,
                    [proposal.expression for proposal in proposals],
                    cache_path=workdir / "data_fields.json",
                    arena={
                        "region": cfg.region,
                        "universe": cfg.universe,
                        "delay": int(cfg.delay),
                    },
                    client_factory=(lambda: self.toolbox.client) if cfg.allow_offline_fetch else None,
                )
                outcome = filter_candidates(
                    proposals,
                    settings=(
                        self.toolbox.arena_settings(node.settings)
                        if node is not None
                        else self.toolbox.arena_settings({})
                    ),
                    catalog=catalog,
                    see_signatures=known,
                    blacklist_subtrees=blacklist,
                    subtree_min_size=int(cfg.subtree_min_size),
                    parent_expression=str(node.expression) if node is not None else None,
                    limit=max(1, int(cfg.refine_proposals)),
                    prefer_minimal_edits=False,
                )
                for reason, amount in outcome.counts.items():
                    filters.reject(reason, amount)
                queued = 0
                for candidate in outcome.accepted:
                    if queued >= max(1, int(cfg.children_per_expansion)) or len(batch) >= remaining:
                        break
                    patched = replace(
                        candidate,
                        family=self._child_family(node, candidate.expression),
                        parent_key=node.node_id if node is not None else None,
                    )
                    child = tree.add_root(patched) if node is None else tree.add_child(node.node_id, patched)
                    if child is None:
                        break
                    queue(child, patched)
                    queued += 1
                jobs_log.append(
                    {
                        "node": node.node_id if node is not None else "ROOT",
                        "operator": operator,
                        "proposed": len(proposals),
                        "queued": queued,
                    }
                )
                return queued

            def widen_root() -> Optional[int]:
                """New subtree: alternate a library seed and an LLM e1 idea (e1 only for LLM roots)."""
                widen_turn[0] += 1
                if not llm_roots and widen_turn[0] % 2 == 0:
                    seeds = self.toolbox.pop_seed_candidates(1)
                    if seeds:
                        root = tree.add_root(seeds[0])
                        queue(root, self._candidate_from_node(root))
                        jobs_log.append({"node": root.node_id, "action": "library_seed", "family": root.family})
                        return 1
                return expand(None, "e1")

            # Proposed-but-unevaluated nodes (seeds, leftovers) are the cheapest work.
            for node in tree.pending():
                if len(batch) >= remaining:
                    break
                queue(node, self._candidate_from_node(node))
                jobs_log.append({"node": node.node_id, "action": "evaluate_pending"})

            initialised = False
            if llm_roots and not tree.roots and isinstance(proposer, LLMRefinementProposer):
                # MCTS-AHD initialisation (mcts_ahd.py): one i1 root from the
                # task description alone, then e1 roots until N_I of them exist.
                initialised = True
                target_roots = int(cfg.initial_seed_count) or 4
                operator_for_root = "i1"
                while len(tree.roots) < target_roots and len(batch) < remaining:
                    before = len(tree.roots)
                    if expand(None, operator_for_root) is None:
                        llm_down = True
                        break
                    operator_for_root = "e1"
                    if len(tree.roots) == before and before > 0:
                        break  # the model gave nothing usable; the next iteration retries
                if not tree.roots and not llm_down:
                    stop_reason = "LLM root initialisation (i1/e1) produced no usable alpha."
                    break

            jobs: List[tuple] = []
            descent_value = str(cfg.tree_value)
            coverage = tree.objective_coverage()
            if (
                descent_value == "ohvi"
                # One evaluation cannot be compared with anything yet; from the
                # second on, a tree with fewer than two usable vectors has no
                # multi-objective signal at all.
                and coverage["evaluated"] >= 2
                and coverage["ready"] < 2
            ):
                # No usable objective vector means every OHVI score is zero:
                # the descent would silently become "least-tried child first".
                # Say so once and fall back to the scalar UCT it replaced.
                descent_value = "scalar"
                if pareto_blind_reason is None:
                    pareto_blind_reason = (
                        f"tree_value=ohvi disabled for this run: only "
                        f"{coverage['ready']}/{coverage['evaluated']} evaluated node(s) have a "
                        f"complete objective vector for {list(tree.objectives)} "
                        f"({coverage['incomplete']} missing a metric, "
                        f"{coverage['out_of_range']} outside their scaling range); "
                        "falling back to value=scalar."
                    )
                    self._append_event(
                        iteration=iteration,
                        stage="search",
                        action="pareto_blind",
                        rationale=pareto_blind_reason,
                        details={"coverage": coverage, "objectives": list(tree.objectives)},
                        hypothesis="The records or the resumed tree lost a metric the objectives need.",
                        risk_note="Hypervolume selection is off; the run is a scalar UCT search.",
                    )
                    self._emit_progress(
                        {
                            "type": "pareto_blind",
                            "run_id": run_id,
                            "iteration": iteration,
                            "stage": stage,
                            "action": "pareto_blind",
                            "rationale": pareto_blind_reason,
                            "coverage": coverage,
                        }
                    )
            if len(batch) < remaining and not initialised:
                jobs = tree.plan(
                    uct_c=float(cfg.uct_c) * (remaining / budget if budget else 0.0),
                    widen_alpha=float(cfg.widen_alpha),
                    max_roots=int(cfg.max_roots) or None,
                    value=descent_value,
                    widen_patience=int(cfg.widen_patience),
                )
                if not any(kind == "leaf" for _, kind in jobs) and not any(t is None for t, _ in jobs):
                    # The descent found nothing expandable: grow a new subtree instead.
                    jobs.append((None, "widen"))

            for target, kind in jobs:
                if len(batch) >= remaining or llm_down:
                    break
                if target is None:
                    llm_down = widen_root() is None
                    continue
                operators_for_job = (
                    (WIDEN_OPERATOR,)
                    if kind == "widen"
                    else leaf_operators(target.e_vector or target.margins, context)
                )
                for operator in operators_for_job:
                    if len(batch) >= remaining:
                        break
                    if expand(target, operator) is None:
                        llm_down = True
                        break

            if llm_down:
                stop_reason = (
                    f"LLM refinement failed {proposer.consecutive_failures} time(s) in a row "
                    f"({proposer.last_error}). Stopped before spending simulations on fallback "
                    "mutations; rerun once the provider recovers, state resumes from disk."
                )
                self._append_event(
                    iteration=iteration,
                    stage=stage,
                    action="stop",
                    rationale=stop_reason,
                    details={"llm_failures": proposer.failures, "budget_spent": evaluated_total},
                    hypothesis="The LLM provider is down or rejecting requests.",
                    risk_note="No simulation was spent this iteration.",
                )
                break

            if not batch:
                stop_reason = "No candidate passed the offline filters."
                self._append_event(
                    iteration=iteration,
                    stage=stage,
                    action="tree_expand",
                    rationale="UCT selected nodes but every proposal was filtered out before evaluation.",
                    details={"filters": dict(filters.counts), "jobs": jobs_log},
                    hypothesis="The operator library or the subtree blacklist may be too tight.",
                    risk_note="No simulation was spent this iteration.",
                )
                continue

            self._emit_progress(
                {
                    "type": "planner_decision",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": stage,
                    "stage_reason": stage_reason,
                    "action": "tree_expand",
                    "batch_size": len(batch),
                    "rationale": f"MCTS descent planned {len(jobs)} job(s); {len(batch)} candidate(s) passed offline filters.",
                    "jobs": jobs_log,
                    "tree_value": descent_value,
                    "objective_coverage": coverage,
                    "hypothesis": "Refining the weakest gate dimension increases the chance of a submit-ready alpha.",
                    "risk_note": f"{sum(filters.counts.values())} proposal(s) rejected without spending budget.",
                    "leaderboard": [pipeline.compact_record(item) for item in frontier[:8]],
                }
            )
            self._emit_progress(
                {
                    "type": "batch_started",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": stage,
                    "action": "tree_expand",
                    "batch_size": len(batch),
                    "candidates": [candidate.to_record() for candidate in batch],
                    "remaining_budget": remaining,
                }
            )

            records = self.toolbox.evaluate_candidates(
                batch,
                progress_callback=lambda payload, iteration=iteration, stage=stage: self._emit_progress(
                    {**payload, "run_id": run_id, "iteration": iteration, "stage": stage, "budget": budget}
                ),
            )
            self.notebook.observe_records(records)
            failed_records = [record for record in records if record.get("status") != "ok"]
            auth_status = self._auth_error_statuses(records)
            if auth_status is not None:
                stop_reason = (
                    f"Authentication rejected ({auth_status}). Refresh WQB_COOKIE_HEADER; "
                    "failed requests did not consume simulation budget."
                )
                self._append_event(
                    iteration=iteration,
                    stage=stage,
                    action="tree_expand",
                    rationale=stop_reason,
                    details={
                        "auth_status": auth_status,
                        "failed_records": len(failed_records),
                        "budget_spent": evaluated_total,
                    },
                    hypothesis="The session cookie expired mid-run.",
                    risk_note="Without valid auth every further request would fail the same way.",
                )
                break
            if failed_records and len(failed_records) == len(records):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    stop_reason = "Three consecutive batches failed entirely; stopping to protect the budget."
                    self._append_event(
                        iteration=iteration,
                        stage=stage,
                        action="tree_expand",
                        rationale=stop_reason,
                        details={"failed_records": len(failed_records)},
                        hypothesis="Systematic failure, not a per-candidate problem.",
                        risk_note="Continuing would repeat the same failure.",
                    )
                    break
            else:
                consecutive_failures = 0

            # Failed requests never reached the simulator, so they must not eat
            # into the simulation budget.
            successful = len(records) - len(failed_records)
            evaluated_total += successful
            evaluated_this_run += successful
            if any(tree.nodes[n].depth == 0 for n in node_for_signature.values() if n in tree.nodes):
                self.seed_evaluated_count += successful

            for record in records:
                signature = pipeline.record_signature(record) or ""
                node_id = node_for_signature.get(signature)
                if not node_id:
                    continue
                node = tree.nodes.get(node_id)
                if node is None:
                    continue
                if record.get("status") != "ok":
                    tree.record(node_id, reward=-10_000.0, state="error")
                    continue
                assessment = assess_record(record)
                divergence = 1.0
                try:
                    divergence = pool.novelty(
                        str(record.get("expression") or ""),
                        record.get("settings"),
                        exclude_alpha_ids={record.get("alpha_id")} if record.get("alpha_id") else None,
                    )
                except Exception:
                    divergence = 1.0
                tree.record(
                    node_id,
                    reward=assessment.scalar,
                    margins=assessment.margins,
                    e_vector=assessment.margins,
                    alpha_id=record.get("alpha_id"),
                    state=EVALUATED,
                    objectives={
                        **self._metric_objectives(record),
                        "quality": assessment.quality_margin,
                        # 0.0 (at the limit) when the platform has not reported a
                        # correlation check yet, matching the scalar reward.
                        "correlation": assessment.novelty_margin,
                        "divergence": float(divergence),
                    },
                )
                parsed = try_parse(str(record.get("expression") or ""))
                if parsed is not None:
                    subtree_counter.update(
                        subtree_hashes(parsed, min_size=int(cfg.subtree_min_size))
                    )

            details = {
                "requested_batch": len(batch),
                "executed_batch": len(records),
                "ok_count": sum(1 for record in records if record.get("status") == "ok"),
                "error_count": sum(1 for record in records if record.get("status") == "error"),
                "quality_ready_count": sum(1 for record in records if pipeline.record_quality_ready(record)),
                "submittable_count": sum(1 for record in records if pipeline.record_is_submit_ready(record)),
                "correlation_blocked_count": sum(
                    1
                    for record in records
                    if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
                ),
                "tree_nodes": len(tree.nodes),
                "front_size": len(tree.front_nodes()),
                "hypervolume": round(tree.front.hypervolume(), 4),
                "filters": dict(filters.counts),
                "best_score_after": self._best_score(),
                "best_alpha_after": self._best_alpha_id(),
            }
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="tree_expand",
                rationale=f"Ran {len(jobs_log)} tree job(s); {len(records)} candidate(s) evaluated.",
                details={**details, "jobs": jobs_log, "tree": tree.stats()},
                hypothesis="Dimension-targeted refinements move the binding gate.",
                risk_note=f"{sum(filters.counts.values())} proposal(s) rejected offline.",
            )
            self.toolbox.write_state()
            tree.save(tree_path)
            self._emit_progress(
                {
                    "type": "batch_completed",
                    "run_id": run_id,
                    "iteration": iteration,
                    "stage": stage,
                    "action": "tree_expand",
                    "details": details,
                    "leaderboard": [pipeline.compact_record(item) for item in self.toolbox.leaderboard(limit=8)],
                    "state": self.toolbox.write_state(),
                }
            )

            # Submission does not wait for the legacy harvest label: the hybrid
            # loop never runs the robustness stage, so `determine_stage` can
            # never return "harvest" here and the old branch was unreachable —
            # 311 live simulations produced zero submissions that way.
            if cfg.submission_mode != "disabled" and len(self.submission_attempts) < int(
                getattr(cfg, "max_submissions_per_run", 1)
            ):
                outcome = self._submit_if_ready(iteration=iteration, stage=stage)
                if outcome is not None:
                    self.submission_attempts.append(outcome)

        finished_at = pipeline.iso_now()
        tree.save(tree_path)
        leaderboard = [pipeline.compact_record(item) for item in self.toolbox.leaderboard(limit=10)]
        state = self.toolbox.write_state()
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "budget": budget,
            "evaluated_count": len(self.toolbox.results),
            "evaluated_this_run": evaluated_this_run,
            "seed_evaluated_this_run": self.seed_evaluated_count,
            "submission_attempts_this_run": len(self.submission_attempts),
            "best_score": self._best_score(),
            "best_alpha_id": self._best_alpha_id(),
            "quality_ready_count": sum(1 for record in self.toolbox.results if pipeline.record_quality_ready(record)),
            "submittable_count": sum(
                1 for record in self.toolbox.results if pipeline.record_is_submit_ready(record)
            ),
            "correlation_blocked_count": sum(
                1
                for record in self.toolbox.results
                if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
            ),
            "iterations_executed": iteration_executed,
            "stop_reason": stop_reason,
            "submission_mode": cfg.submission_mode,
            "final_stage": stage,
            "search_mode": "hybrid",
            "pareto_ready": tree.objective_coverage()["ready"],
            "pareto_value_effective": descent_value,
            "pareto_disabled_reason": pareto_blind_reason,
            "stage_history": self.notebook.stage_history,
            "family_stats": self.notebook.family_stats(self.toolbox.results),
            "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results),
            "hypothesis_log_tail": self.notebook.recent_hypotheses(limit=8),
        }

        self._search_report = {
            "mode": "hybrid",
            "tree": tree.stats(),
            "pareto": {
                "objectives": list(tree.objectives),
                "value": str(cfg.tree_value),
                "value_effective": descent_value,
                "disabled_reason": pareto_blind_reason,
                "coverage": tree.objective_coverage(),
                "front_size": len(tree.front_nodes()),
                "hypervolume": round(tree.front.hypervolume(), 6),
                "families": len({node.family for node in tree.front_nodes()}),
                "front": [
                    {
                        "node_id": node.node_id,
                        "alpha_id": node.alpha_id,
                        "expression": node.expression,
                        "family": node.family,
                        "reward": node.reward,
                        "objectives": {name: round(value, 6) for name, value in node.objectives.items()},
                    }
                    for node in tree.front_nodes()
                ],
            },
            "filters": {**dict(filters.counts), "evaluated": evaluated_total},
            "novelty_pool": pool.stats(),
            "catalog": {
                "operators": len(catalog.operators),
                "fields": len(catalog.fields),
            },
            "proposer": type(proposer).__name__,
            "llm": {
                "calls": int(getattr(proposer, "calls", 0)),
                "failures": int(getattr(proposer, "failures", 0)),
                "fallbacks": int(getattr(proposer, "fallback_count", 0)),
                "operators": dict(getattr(proposer, "operator_counts", {}) or {}),
                "last_error": getattr(proposer, "last_error", None),
            },
            "frequent_subtrees": blacklist[: int(cfg.frequent_subtree_topk)],
        }

        report_path = self._write_run_report(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            leaderboard=leaderboard,
            state=state,
        )
        result = AgentRunResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            report_path=report_path,
            summary=summary,
            leaderboard=leaderboard,
            events=self.events,
            submissions=self.submission_attempts,
            state=state,
        )
        self._emit_progress({"type": "run_finished", "run_id": run_id, "result": result.to_dict()})
        return result

    def _run_mo_hifo(self) -> AgentRunResult:
        """Population search: MEoH selection with HiFo prompt guidance.

        See ``alpha_agent.mo_hifo``. It shares the toolbox, the offline
        filters, the gates and submission governance with the hybrid search;
        the search tree is not used. Objectives: Sharpe, returns, turnover.
        """
        import os

        from alpha_agent.catalog import Catalog, load_catalog
        from alpha_agent.expr_ast import behavioral_signature
        from alpha_agent.families import lineage_family
        from alpha_agent.mo_hifo import (
            OBJECTIVES,
            InsightExtractor,
            MoHifoSearch,
            MoHifoState,
            candidate_metadata,
            non_dominated,
            population_diversity,
            population_hypervolume,
        )
        from alpha_agent.pool_fields import expression_fields
        from alpha_agent.refine_llm import (
            DeterministicRefinementProposer,
            FilterOutcome,
            LLMRefinementProposer,
            filter_candidates,
        )
        from alpha_agent.reward import assess_record

        cfg = self.runtime.agent
        started_at = pipeline.iso_now()
        run_id = self._build_run_id(started_at)
        budget = max(0, int(cfg.budget))
        workdir = Path(cfg.workdir)
        state_path = (
            Path(cfg.mo_hifo_state_path) if cfg.mo_hifo_state_path else workdir / "mo_hifo_population.json"
        )
        state = MoHifoState.load_or_new(state_path, seed=int(cfg.random_seed))
        arena = {"region": cfg.region, "universe": cfg.universe, "delay": int(cfg.delay)}

        def margins_of(record: Dict[str, Any]) -> Dict[str, float]:
            try:
                return dict(assess_record(record).margins)
            except Exception:
                return {}

        recovered = state.recover(self.toolbox.results, margins_of)

        field_ids = sorted(
            {field for member in state.population + state.offspring for field in expression_fields(member.expression)}
        )
        try:
            catalog = load_catalog(
                fields_summary=Path(cfg.fields_summary),
                operators_cache=workdir / "operators.json",
                client_factory=(lambda: self.toolbox.client) if cfg.allow_offline_fetch else None,
                arena=arena,
                field_types_cache=workdir / "data_fields.json",
                field_ids=field_ids,
            )
        except Exception:
            catalog = Catalog()

        fallback = DeterministicRefinementProposer(max_proposals=max(4, int(cfg.refine_proposals) * 2))
        proposer: Optional[LLMRefinementProposer] = None
        if cfg.llm_refine_enabled and self.runtime.model.provider == "openai":
            proposer = LLMRefinementProposer(self.runtime.model, fallback=fallback, max_calls=int(cfg.max_llm_calls))

        # Every formula already simulated (any search, any run) is a duplicate
        # by behaviour, not only the ones this population has seen.
        known: set[str] = set(self.toolbox.evaluated_signatures)
        for record in self.toolbox.results:
            if record.get("status") != "ok" or not record.get("expression"):
                continue
            try:
                signature = behavioral_signature(
                    str(record["expression"]), pipeline.normalize_settings(record.get("settings") or {})
                )
            except Exception:
                signature = None
            if signature:
                known.add(signature)

        filters = FilterOutcome()
        jobs_log: List[Dict[str, Any]] = []
        search_ref: List[Any] = []

        def arena_results() -> List[Dict[str, Any]]:
            """Records of this arena only: results.jsonl is shared across arenas,
            and another region's failures name fields this one may not have."""
            out = []
            for record in self.toolbox.results:
                settings = record.get("settings") or {}
                if (
                    str(settings.get("region", cfg.region)).upper() == str(cfg.region).upper()
                    and str(settings.get("universe", cfg.universe)).upper() == str(cfg.universe).upper()
                    and int(settings.get("delay", cfg.delay)) == int(cfg.delay)
                ):
                    out.append(record)
            return out

        def propose(operator: str, parents: List[Any], guidance: Dict[str, Any]) -> Optional[List[pipeline.Candidate]]:
            assert proposer is not None
            node = parents[0] if parents and operator not in {"i1", "e1"} else None
            spent = search_ref[0].spent if search_ref else 0
            records_here = arena_results()
            context: Dict[str, Any] = {
                "iteration": state.generation,
                "remaining_budget": budget - spent,
                "arena": arena,
                "failed_check_histogram": self.notebook.failed_check_histogram(records_here, top_k=5),
                # e2 recombines the drawn parents; the other operators see the front.
                "elites": (
                    [member.summary() for member in parents]
                    if operator == "e2"
                    else list(guidance.get("elites") or [])
                ),
                "failed_attempts": self._failure_summaries(records_here),
                "guidance": guidance,
            }
            if operator == "e1":
                context["references"] = [member.summary() for member in parents][:8]
            if node is not None:
                context["failed_children"] = self._failure_summaries(records_here, parent_key=node.key, limit=6)
            proposals = proposer.propose(
                node=node,
                e_vector=dict(node.margins) if node is not None else {},
                context=context,
                blacklist=(),
                operators=catalog.operators,
                fields=catalog.fields,
                k=int(cfg.refine_proposals),
                operator=operator,
            )
            failures = int(getattr(proposer, "consecutive_failures", 0))
            if failures >= max(1, int(cfg.llm_failure_limit)):
                return None
            if cfg.require_llm and failures > 0:
                return []  # discard the deterministic fallback
            self._learn_field_types(
                catalog,
                [proposal.expression for proposal in proposals],
                cache_path=workdir / "data_fields.json",
                arena=arena,
                client_factory=(lambda: self.toolbox.client) if cfg.allow_offline_fetch else None,
            )
            outcome = filter_candidates(
                proposals,
                settings=self.toolbox.arena_settings(node.settings if node is not None else {}),
                catalog=catalog,
                see_signatures=known,
                blacklist_subtrees=(),
                subtree_min_size=int(cfg.subtree_min_size),
                parent_expression=node.expression if node is not None else None,
                limit=max(1, int(cfg.refine_proposals)),
                prefer_minimal_edits=False,
            )
            for reason, amount in outcome.counts.items():
                filters.reject(reason, amount)
            accepted: List[pipeline.Candidate] = []
            for candidate in outcome.accepted[: max(1, int(cfg.children_per_expansion))]:
                signature = candidate.metadata.get("behavioral_signature")
                if signature:
                    known.add(str(signature))
                accepted.append(
                    replace(
                        candidate,
                        family=lineage_family(candidate.expression),
                        stage="mo_hifo",
                        parent_key=node.key if node is not None else None,
                        metadata=candidate_metadata(candidate, operator=operator, parents=parents, guidance=guidance),
                    )
                )
            jobs_log.append(
                {
                    "generation": state.generation,
                    "operator": operator,
                    "regime": guidance.get("regime"),
                    "parents": len(parents),
                    "proposed": len(proposals),
                    "queued": len(accepted),
                }
            )
            return accepted

        consecutive_failures = [0]

        def evaluate(batch: List[pipeline.Candidate]) -> tuple:
            records = self.toolbox.evaluate_candidates(
                batch,
                progress_callback=lambda payload: self._emit_progress(
                    {**payload, "run_id": run_id, "iteration": state.generation, "stage": "mo_hifo", "budget": budget}
                ),
            )
            self.notebook.observe_records(records)
            auth_status = self._auth_error_statuses(records)
            if auth_status is not None:
                return records, (
                    f"Authentication rejected ({auth_status}). Refresh WQB_COOKIE_HEADER and run continue; "
                    "failed requests did not consume simulation budget."
                )
            failed = [record for record in records if record.get("status") != "ok"]
            if failed and len(failed) == len(records):
                consecutive_failures[0] += 1
                if consecutive_failures[0] >= 3:
                    return records, "Three consecutive batches failed entirely; stopping to protect the budget."
            else:
                consecutive_failures[0] = 0
            return records, None

        def should_stop() -> Optional[str]:
            if not bool(getattr(cfg, "session_ping", True)):
                return None
            status = self._session_status()
            if status is None:
                return None
            return (
                f"Session expired ({status}). Refresh WQB_COOKIE_HEADER and run continue; "
                "the population resumes from disk and no simulation was spent."
            )

        def on_batch(details: Dict[str, Any]) -> None:
            self._append_event(
                iteration=int(details.get("generation") or 0),
                stage="mo_hifo",
                action="population_batch",
                rationale=f"Evaluated {len(details.get('children') or [])} offspring in regime {details.get('regime')}.",
                details={**details, "filters": dict(filters.counts)},
                hypothesis="Pareto selection on Sharpe, returns and turnover with distilled insights.",
                risk_note=f"{sum(filters.counts.values())} proposal(s) rejected offline.",
            )
            self._emit_progress(
                {
                    "type": "batch_completed",
                    "run_id": run_id,
                    "iteration": details.get("generation"),
                    "stage": "mo_hifo",
                    "action": "population_batch",
                    "details": details,
                    "leaderboard": [pipeline.compact_record(item) for item in self.toolbox.leaderboard(limit=8)],
                    "state": self.toolbox.write_state(),
                }
            )
            if cfg.submission_mode != "disabled" and len(self.submission_attempts) < int(
                getattr(cfg, "max_submissions_per_run", 1)
            ):
                outcome = self._submit_if_ready(iteration=int(details.get("generation") or 0), stage="mo_hifo")
                if outcome is not None:
                    self.submission_attempts.append(outcome)

        extractor: Optional[InsightExtractor] = None
        if proposer is not None and bool(cfg.extract_insights):
            api_key = os.getenv(self.runtime.model.api_key_env, "").strip()
            extractor = InsightExtractor(
                lambda body: proposer._chat_completion(body=body, api_key=api_key),
                model=self.runtime.model.model,
                temperature=float(self.runtime.model.temperature),
            )

        search = MoHifoSearch(
            state=state,
            propose=propose,
            evaluate=evaluate,
            margins_of=margins_of,
            extractor=extractor,
            population_size=int(cfg.population_size),
            crossover_parents=int(cfg.crossover_parents),
            batch_size=int(cfg.tree_batch_size),
            insights_per_prompt=int(cfg.insights_per_prompt),
            seed=int(cfg.random_seed),
            should_stop=should_stop,
            on_batch=on_batch,
        )
        search_ref.append(search)

        self._emit_progress(
            {
                "type": "run_started",
                "run_id": run_id,
                "started_at": started_at,
                "budget": budget,
                "max_iterations": cfg.max_iterations,
                "workdir": str(cfg.workdir),
                "search_mode": "mo_hifo",
                "population_state": str(state_path),
                "recovered_offspring": recovered,
            }
        )
        if proposer is None or not proposer.enabled:
            stop_reason = (
                "mo-hifo needs an LLM proposer (--planner-provider openai and an API key); nothing was simulated."
            )
        else:
            stop_reason = search.run(budget, max_batches=max(1, int(cfg.max_iterations)))
        state.save()

        finished_at = pipeline.iso_now()
        leaderboard = [pipeline.compact_record(item) for item in self.toolbox.leaderboard(limit=10)]
        toolbox_state = self.toolbox.write_state()
        front = non_dominated(state.population)
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "budget": budget,
            "evaluated_count": len(self.toolbox.results),
            "evaluated_this_run": search.spent,
            "submission_attempts_this_run": len(self.submission_attempts),
            "best_score": self._best_score(),
            "best_alpha_id": self._best_alpha_id(),
            "quality_ready_count": sum(1 for record in self.toolbox.results if pipeline.record_quality_ready(record)),
            "submittable_count": sum(
                1 for record in self.toolbox.results if pipeline.record_is_submit_ready(record)
            ),
            "correlation_blocked_count": sum(
                1
                for record in self.toolbox.results
                if pipeline.record_quality_ready(record) and record.get("failed_correlation_checks")
            ),
            "iterations_executed": search.batches,
            "stop_reason": stop_reason,
            "submission_mode": cfg.submission_mode,
            "final_stage": "mo_hifo",
            "search_mode": "mo_hifo",
            "generation": state.generation,
            "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results),
        }
        self._search_report = {
            "mode": "mo_hifo",
            "objectives": list(OBJECTIVES),
            "state_path": str(state_path),
            "generation": state.generation,
            "population_size": len(state.population),
            "hypervolume": round(population_hypervolume(state.population), 6),
            "diversity": population_diversity(state.population),
            "regime": state.navigator.regime,
            "navigator_history": state.navigator.history[-30:],
            "front": [{**member.summary(), "alpha_id": member.alpha_id} for member in front],
            "insights": [
                {"insight": text, **{k: v for k, v in stats.items() if k != "effects"}, "effects": stats.get("effects")}
                for text, stats in sorted(
                    state.insights.items.items(), key=lambda item: -float(item[1].get("effectiveness", 0.0))
                )
            ],
            "extractions": search.extractions,
            "filters": {**dict(filters.counts), "evaluated": search.spent},
            "jobs": jobs_log[-50:],
            "llm": {
                "calls": int(getattr(proposer, "calls", 0)),
                "failures": int(getattr(proposer, "failures", 0)),
                "fallbacks": int(getattr(proposer, "fallback_count", 0)),
                "operators": dict(getattr(proposer, "operator_counts", {}) or {}),
                "last_error": getattr(proposer, "last_error", None),
                "insight_calls": extractor.calls if extractor else 0,
                "insight_failures": extractor.failures if extractor else 0,
            },
        }
        report_path = self._write_run_report(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            leaderboard=leaderboard,
            state=toolbox_state,
        )
        result = AgentRunResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            report_path=report_path,
            summary=summary,
            leaderboard=leaderboard,
            events=self.events,
            submissions=self.submission_attempts,
            state=toolbox_state,
        )
        self._emit_progress({"type": "run_finished", "run_id": run_id, "result": result.to_dict()})
        return result

    def _seed_hybrid_roots(self, tree: Any, *, count: int, pool: Any = None, diverse: bool = False) -> int:
        """Add seed roots; with ``diverse`` pick a structurally spread subset.

        Diverse mode draws from the whole seed queue (e.g. the 101 WorldQuant
        formulaic alphas) and greedily takes the seed farthest, by Jaccard
        distance over its structure, from those already chosen, starting from
        the one most divergent from the account pool. Unchosen seeds stay queued
        for later root widening. Each seed becomes its own family so the
        per-family elite cap sees them as separate ideas.

        Roots come from the curated seeds themselves, not their pool-driven
        field swaps: a swap rewrites fields mechanically (observed: ``close``
        -> ``cap``), so it is a refinement, not a root idea.

        The distance uses operators, fields, parent->child operator pairs and
        subtrees together. Subtrees alone made nearly every pair of Alpha101
        formulas tie at distance 1, so the pick fell back to the name order.
        """
        if not diverse:
            seeds = self.toolbox.pop_seed_candidates(max(1, int(count)))
            for candidate in seeds:
                tree.add_root(candidate)
            return len(seeds)

        from alpha_agent.expr_ast import (
            ast_bigrams,
            collect_call_operators,
            collect_identifiers,
            subtree_hashes,
            try_parse,
        )

        def features(node: Any) -> set:
            return (
                {f"op:{name}" for name in collect_call_operators(node)}
                | {f"field:{name}" for name in collect_identifiers(node)}
                | {f"pair:{parent}>{child}" for parent, child in ast_bigrams(node)}
                | {f"sub:{digest}" for digest in subtree_hashes(node, min_size=2)}
            )

        queue = list(getattr(self.toolbox, "seed_queue", []) or [])
        originals = [candidate for candidate in queue if getattr(candidate, "stage", "") != "field_swap"]
        parsed = []
        for candidate in originals or queue:
            node = try_parse(candidate.expression)
            if node is not None:
                parsed.append((candidate, features(node)))
        if not parsed:
            return 0

        def divergence(candidate: Any) -> float:
            try:
                return float(pool.novelty(candidate.expression, candidate.settings)) if pool is not None else 1.0
            except Exception:
                return 1.0

        def distance(a: set, b: set) -> float:
            union = a | b
            return 1.0 - (len(a & b) / len(union)) if union else 0.0

        divergences = {id(item[0]): divergence(item[0]) for item in parsed}
        chosen = [max(parsed, key=lambda item: (divergences[id(item[0])], item[0].idea_name))]
        rest = [item for item in parsed if item is not chosen[0]]
        while rest and len(chosen) < max(1, int(count)):
            # Ties on distance go to the seed the pool covers least, not to the name.
            best = max(
                rest,
                key=lambda item: (
                    round(min(distance(item[1], other[1]) for other in chosen), 3),
                    divergences[id(item[0])],
                    item[0].idea_name,
                ),
            )
            chosen.append(best)
            rest.remove(best)
        picked = {id(item[0]) for item in chosen}
        self.toolbox.seed_queue = [candidate for candidate in queue if id(candidate) not in picked]
        for candidate, _ in chosen:
            tree.add_root(replace(candidate, family=str(candidate.idea_name or candidate.family)))
        return len(chosen)

    def _candidate_from_node(self, node: Any) -> pipeline.Candidate:
        return pipeline.Candidate(
            expression=str(node.expression),
            settings=dict(node.settings or {}),
            family=str(node.family or ""),
            idea_name=str(node.idea_name or "hybrid.node"),
            stage="hybrid_node",
            priority=0.0,
            parent_key=node.parent_id,
            metadata={"node_id": node.node_id, "behavioral_signature": node.behavioral_sig},
        )

    def _session_status(self) -> Optional[int]:
        """Ping an authenticated endpoint; return 401/403 if the session died.

        Called at the top of each iteration so a dead cookie stops the run
        before any budget is spent. Network errors return ``None`` (fail-open):
        a transient blip should not abort a multi-hour run.
        """
        try:
            client = self.toolbox.client
            response = client._request(
                "GET", "/users/self/alphas?limit=1", accept="application/json;version=2.0"
            )
        except Exception:
            return None
        if isinstance(getattr(response, "status", None), int) and response.status in {401, 403}:
            return int(response.status)
        return None

    @staticmethod
    def _auth_error_statuses(records: Sequence[Dict[str, Any]]) -> Optional[int]:
        """Return an auth status code if the batch was rejected for credentials.

        A 401/403 during a run almost always means the ``WQB_COOKIE_HEADER``
        session expired. Observed live: a 100-simulation run spent its whole
        budget on 401 errors because nothing checked for this.
        """
        for record in records:
            if record.get("status") == "ok":
                continue
            error = record.get("error") or {}
            status = error.get("status")
            if isinstance(status, int) and status in {401, 403}:
                return status
        return None

    def _decide(self, context: Dict[str, Any]) -> PlannerAction:
        action = self.planner.decide(context)
        if action.action not in {
            "evaluate_seed",
            "evaluate_refine",
            "evaluate_diversify",
            "evaluate_robustness",
            "submit_best",
            "stop",
        }:
            return PlannerAction.stop(f"Invalid planner action: {action.action}")
        return action

    def _submit_if_ready(self, *, iteration: int, stage: str) -> Optional[Dict[str, Any]]:
        """Submit the best strictly-ready alpha, whatever the research stage is.

        What BRAIN requires for a submission is that every blocking *and*
        correlation check passes — ``record_is_submit_ready`` — and that is
        exactly what is required here. The legacy stage machine additionally
        waits for two robustness probes before harvest; that is a governance
        nicety of the stage mode, not a platform rule, and the hybrid loop has
        no robustness stage at all.
        """
        candidate = self.toolbox.best_submittable_candidate()
        if not candidate:
            return None
        alpha_id = str(candidate.get("alpha_id") or "")
        if not alpha_id or alpha_id in self.toolbox.submitted_alpha_ids:
            return None
        mode = self.runtime.agent.submission_mode
        decision = PlannerAction(
            action="submit_best",
            rationale=(
                f"Alpha {alpha_id} passes every blocking and correlation check "
                f"(mode={mode}); submitting before the search drifts away from it."
            ),
            hypothesis="A strictly submit-ready alpha is worth a submission slot.",
            risk_note="BRAIN submissions are final and cannot be withdrawn.",
            target_alpha_id=alpha_id,
        )
        return self._execute_submit_action(decision, iteration, stage, require_stage=False)

    def _execute_submit_action(
        self,
        decision: PlannerAction,
        iteration: int,
        stage: str,
        *,
        require_stage: bool = True,
    ) -> Optional[Dict[str, Any]]:
        mode = self.runtime.agent.submission_mode
        if require_stage and stage != "harvest":
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "blocked",
                    "reason": "Agent submissions require harvest stage after robustness evidence.",
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        candidate = self.toolbox.best_submittable_candidate(target_alpha_id=decision.target_alpha_id)
        if not candidate:
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={"result": "skipped", "reason": "No eligible submittable candidate."},
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        if mode == "disabled":
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "blocked",
                    "reason": "Submission mode is disabled.",
                    "alpha_id": candidate.get("alpha_id"),
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        approved = mode == "auto_approved"
        if mode == "manual":
            approved = self.approval_callback(candidate) if self.approval_callback else False
        if not approved:
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={
                    "result": "blocked",
                    "reason": "Manual approval not granted.",
                    "alpha_id": candidate.get("alpha_id"),
                },
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return None

        try:
            submission = self.toolbox.submit_record(candidate)
            result = {
                "alpha_id": submission.get("alpha_id"),
                "result": submission.get("result", "ok"),
                "submitted_at": submission.get("submitted_at", pipeline.iso_now()),
                "response": submission.get("response"),
            }
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={"result": "ok", "alpha_id": submission.get("alpha_id")},
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return result
        except BrainApiError as exc:
            error_payload = {
                "alpha_id": candidate.get("alpha_id"),
                "result": "error",
                "submitted_at": pipeline.iso_now(),
                "response": {
                    "message": str(exc),
                    "status": exc.status,
                    "url": exc.url,
                    "payload": exc.payload,
                },
            }
            self._append_event(
                iteration=iteration,
                stage=stage,
                action="submit_best",
                rationale=decision.rationale,
                details={"result": "error", "alpha_id": candidate.get("alpha_id"), "message": str(exc)},
                hypothesis=decision.hypothesis,
                risk_note=decision.risk_note,
            )
            return error_payload

    def _select_batch(
        self,
        *,
        candidates: Sequence[pipeline.Candidate],
        batch_size: int,
        stage: str,
        frontier: Sequence[Dict[str, Any]],
        blocked_families: set[str],
        enforce_family_cap: bool = True,
    ) -> List[pipeline.Candidate]:
        if not candidates:
            return []
        ranked = self.notebook.rank_candidates(
            candidates=candidates,
            frontier_records=frontier,
            stage=stage,
            blocked_families=blocked_families,
        )
        if not enforce_family_cap:
            return list(ranked[:batch_size])
        uncapped = [candidate for candidate in ranked if not self.notebook.is_family_capped(candidate.family)]
        if len(uncapped) >= batch_size:
            return uncapped[:batch_size]
        merged = uncapped + [candidate for candidate in ranked if candidate not in uncapped]
        return merged[:batch_size]

    def _append_event(
        self,
        *,
        iteration: int,
        stage: str,
        action: str,
        rationale: str,
        details: Dict[str, Any],
        hypothesis: str,
        risk_note: str,
    ) -> None:
        event = {
                "timestamp": pipeline.iso_now(),
                "iteration": iteration,
                "stage": stage,
                "action": action,
                "rationale": rationale,
                "hypothesis": hypothesis,
                "risk_note": risk_note,
                "details": details,
        }
        self.events.append(event)
        self._emit_progress({"type": "event_appended", "event": event})

    def _emit_progress(self, payload: Dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        payload.setdefault("timestamp", pipeline.iso_now())
        try:
            self.progress_callback(payload)
        except Exception:
            # Progress rendering must not interrupt a live research run.
            pass

    def _build_context(
        self,
        *,
        iteration: int,
        budget: int,
        evaluated_total: int,
        seed_target: int,
        stage: str,
        stage_reason: str,
        refine_candidates: Sequence[pipeline.Candidate],
        diversify_candidates: Sequence[pipeline.Candidate],
        robustness_candidates: Sequence[pipeline.Candidate],
        frontier: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        remaining_budget = max(0, budget - evaluated_total)
        best = frontier[0] if frontier else {}
        best_submittable = self.toolbox.best_submittable_candidate()
        return {
            "iteration": iteration,
            "budget": budget,
            "evaluated_total": evaluated_total,
            "remaining_budget": remaining_budget,
            "seed_target": seed_target,
            "seed_target_remaining": max(0, seed_target - self.seed_evaluated_count),
            "seed_queue_remaining": len(self.toolbox.seed_queue),
            "refine_candidates_available": len(refine_candidates),
            "diversification_candidates_available": len(diversify_candidates),
            "robustness_candidates_available": len(robustness_candidates),
            "leaderboard_count": len(frontier),
            "best_score": best.get("score"),
            "best_alpha_id": best.get("alpha_id"),
            "best_submittable_alpha_id": best_submittable.get("alpha_id") if best_submittable else None,
            "submission_mode": self.runtime.agent.submission_mode,
            "allow_pending_checks": self.runtime.agent.allow_pending_checks,
            "family_filter": list(self.runtime.agent.family_filter),
            "research_stage": stage,
            "stage_reason": stage_reason,
            "family_budget_cap": self.notebook.family_cap(),
            "family_stats": self.notebook.family_stats(self.toolbox.results),
            "failed_check_histogram": self.notebook.failed_check_histogram(self.toolbox.results),
            "recent_hypotheses": self.notebook.recent_hypotheses(limit=5),
            "recent_events": self.events[-4:],
            "frontier_preview": [
                {
                    "alpha_id": item.get("alpha_id"),
                    "family": item.get("family"),
                    "score": item.get("score"),
                    "failed_checks": item.get("failed_checks"),
                    "stage": item.get("stage"),
                }
                for item in frontier[:5]
            ],
        }

    def _best_score(self) -> Optional[float]:
        leaderboard = self.toolbox.leaderboard(limit=1)
        if not leaderboard:
            return None
        return safe_float(leaderboard[0].get("score"))

    def _best_alpha_id(self) -> Optional[str]:
        leaderboard = self.toolbox.leaderboard(limit=1)
        if not leaderboard:
            return None
        alpha_id = leaderboard[0].get("alpha_id")
        return str(alpha_id) if alpha_id else None

    def _build_run_id(self, started_at: str) -> str:
        seed = json.dumps(
            {
                "started_at": started_at,
                "budget": self.runtime.agent.budget,
                "family_filter": list(self.runtime.agent.family_filter),
                "submission_mode": self.runtime.agent.submission_mode,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
        return f"agent_{digest}"

    def _write_run_report(
        self,
        *,
        run_id: str,
        started_at: str,
        finished_at: str,
        summary: Dict[str, Any],
        leaderboard: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Path:
        report = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime": {
                "auth": {
                    "base_url": self.runtime.auth.base_url,
                    "timeout": self.runtime.auth.timeout,
                    "using_cookie_header": bool(self.runtime.auth.cookie_header),
                    "email_configured": bool(self.runtime.auth.email),
                },
                "model": {
                    "provider": self.runtime.model.provider,
                    "model": self.runtime.model.model,
                    "temperature": self.runtime.model.temperature,
                    "base_url": self.runtime.model.base_url,
                },
                "agent": {
                    "budget": self.runtime.agent.budget,
                    "max_iterations": self.runtime.agent.max_iterations,
                    "seed_fraction": self.runtime.agent.seed_fraction,
                    "refine_top_k": self.runtime.agent.refine_top_k,
                    "robustness_top_k": self.runtime.agent.robustness_top_k,
                    "robustness_score_threshold": self.runtime.agent.robustness_score_threshold,
                    "max_family_budget_share": self.runtime.agent.max_family_budget_share,
                    "min_expression_novelty": self.runtime.agent.min_expression_novelty,
                    "family_filter": list(self.runtime.agent.family_filter),
                    "submission_mode": self.runtime.agent.submission_mode,
                    "allow_pending_checks": self.runtime.agent.allow_pending_checks,
                    "workdir": str(self.runtime.agent.workdir),
                    "search_mode": str(getattr(self.runtime.agent, "search_mode", "legacy")),
                },
            },
            "summary": summary,
            "leaderboard": leaderboard,
            "events": self.events,
            "hypothesis_log": self.notebook.hypothesis_log,
            "stage_history": self.notebook.stage_history,
            "submissions": [pipeline.compact_submission(item) for item in self.submission_attempts],
            "state": state,
        }
        if self._search_report is not None:
            report["search"] = self._search_report
        safe_stamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.toolbox.agent_runs_dir / f"{safe_stamp}_{run_id}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path
