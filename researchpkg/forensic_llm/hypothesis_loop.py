"""
hypothesis_orchestrated investigation loop (core_spec v2).

Orientation (``orientation_budget_fraction`` of global; default 10%) → one-shot planning → programmatic dispatch of fixed-plan
hypothesis workers (``ThreadPoolExecutor``; I/O-bound LLM/DB, not OS processes) → deterministic merge of detections & report assembly.

No reflection / cross-scheme LLM reconciliation / synthesis phases. Dynamic
hypothesis injection after tasks is **off by default** (``InvestigatorConfig.
dynamic_hypothesis_injection``); enable only for exploratory runs.

Token budget model (orchestrator pool, no mid-run boost): ``docs/BUDGET_ORCHESTRATOR.md``.
"""
from __future__ import annotations

import copy
import logging
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from researchpkg.forensic_llm.artefacts import (
    ensure_orientation_report_file,
    is_orientation_report_substantive,
    load_hypothesis_result,
    load_orientation_report_text,
    persist_dispatch_queue_snapshot,
    persist_plan_with_queue,
    planning_orientation_report_cap,
    prepare_orientation_report_for_planning,
    report_md_path,
    save_hypothesis_result,
    save_investigation_report,
    save_run_stats,
    task_dir,
    write_orientation_run_metadata,
)
from researchpkg.forensic_llm.dispatch_queue import (
    DispatchQueueManager,
)
from researchpkg.forensic_llm.hypothesis_track import (
    build_hypothesis_track,
    save_hypothesis_track,
)
from researchpkg.forensic_llm.investigation_report import (
    assemble_investigation_report,
    render_report_markdown,
)
from researchpkg.forensic_llm.memory_store import (
    build_shared_context_excerpt,
    flush_global_memory,
)
from researchpkg.forensic_llm.model_tokenizer import (
    count_tokens,
)
from researchpkg.forensic_llm.models import (
    DispatchQueueItem,
    ForensicReport,
    HypothesisResult,
    HypothesisTaskBrief,
    InvestigationPlan,
    RunStats,
)
from researchpkg.forensic_llm.orientation_utils import (
    format_orientation_budget_block,
)
from researchpkg.forensic_llm.plan_utils import (
    apply_worker_budget_pool_to_dispatch_queue,
    build_hypothesis_brief,
    dispatch_queue_sort_key,
    enforce_plan_hypothesis_guardrails,
    hypothesis_wave_level,
)
from researchpkg.forensic_llm.prompts import (
    build_orientation_prompt,
    build_planning_prompt,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)
from researchpkg.forensic_llm.tool_defs import (
    HYPOTHESIS_WORKER_TOOL_NAMES,
)

if TYPE_CHECKING:
    from researchpkg.forensic_llm.agent import (
        ForensicAgent,
    )
    from researchpkg.forensic_llm.config import (
        InvestigatorConfig,
    )

log = logging.getLogger(__name__)


def _refresh_runtime_hypothesis_budgets(
    agent: "ForensicAgent",
    queue: DispatchQueueManager,
    cfg: "InvestigatorConfig",
) -> None:
    """
    Option B: split ``agent._hypothesis_worker_pool_remaining`` across pending tasks
    (min floor + planner weights). Mutates each pending item's ``budget_tokens``.
    """
    if not cfg.use_runtime_hypothesis_budget_pool:
        return
    pending = queue.pending()
    if not pending:
        return
    with agent._hypothesis_pool_lock:
        pool = max(0, int(agent._hypothesis_worker_pool_remaining or 0))
    if pool == 0 and pending:
        floor = max(500, min(50_000, cfg.hypothesis_task_min_tokens // 4))
        for it in pending:
            it.budget_tokens = floor
        log.warning(
            "Hypothesis runtime pool is 0; assigning minimal budget %s to each of %d pending task(s)",
            floor,
            len(pending),
        )
        return
    apply_worker_budget_pool_to_dispatch_queue(
        pending,
        allocation_pool_tokens=pool,
        min_per_task=max(1, cfg.hypothesis_task_min_tokens),
    )


def _queue_snapshot_for_stream(
    queue: DispatchQueueManager, *, limit: int = 300
) -> Dict[str, Any]:
    """Compact queue state for orchestrator NDJSON events."""
    with queue._lock:
        items = sorted(queue.items, key=dispatch_queue_sort_key)
        n_total = len(items)
        rows = []
        for i in items[:limit]:
            rows.append(
                {
                    "task_id": i.task_id,
                    "scheme": i.scheme,
                    "hypothesis_id": i.hypothesis_id,
                    "dispatch_priority": i.dispatch_priority,
                    "hypothesis_wave": hypothesis_wave_level(i.hypothesis_id),
                    "status": i.status,
                    "source": getattr(i, "source", None) or "plan",
                }
            )
    return {
        "tasks": rows,
        "n_total": n_total,
        "truncated": n_total > limit,
    }


def _user_msg(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def run_hypothesis_orchestrated_loop(
    agent: "ForensicAgent",
    report: ForensicReport,
) -> None:
    """Main v2 control flow."""
    cfg = agent.config
    run_dir = agent._run_dir
    t0 = time.time()
    global_budget = agent._global_token_budget
    orient_cap = int(global_budget * cfg.orientation_budget_fraction)

    report.strategy = "hypothesis_orchestrated"
    run_stats = RunStats(
        run_id=agent._run_id,
        max_parallel_workers=cfg.max_parallel_workers,
    )

    (run_dir / "orchestrator").mkdir(parents=True, exist_ok=True)
    (run_dir / "memory").mkdir(parents=True, exist_ok=True)
    (run_dir / "orientation").mkdir(parents=True, exist_ok=True)
    (run_dir / "schemes").mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)

    plan_cap = planning_orientation_report_cap(cfg.text_limits)
    ensure_orientation_report_file(run_dir)

    messages: List[Dict[str, Any]] = []

    # --- Orientation (fraction of global run token budget) ---
    agent.budget.start_phase("orientation")
    agent._orientation_budget_ctx = {
        "cap": orient_cap,
        "planning_input_cap": plan_cap,
    }
    agent._orientation_budget_pacing_sent.clear()
    budget_block = format_orientation_budget_block(
        tokens_used=0,
        tokens_cap=orient_cap,
        planning_input_cap_tokens=plan_cap,
        encourage_deep_until_fraction=cfg.orientation_budget_encourage_deep_until_fraction,
        min_fraction_for_complete=cfg.orientation_budget_min_fraction_for_complete,
    )
    from researchpkg.forensic_llm.prompts.orientation import (
        build_orientation_kickoff_message,
    )

    messages.append(
        _user_msg(build_orientation_kickoff_message(budget_block=budget_block))
    )
    agent._orientation_prompt_message = None
    orientation_steps = 0
    orientation_done = False
    while not orientation_done:
        if agent.budget.should_stop():
            log.warning(
                "Global budget stop during orientation without complete_orientation; "
                "proceeding to planning"
            )
            break
        orient_used = agent.budget.phase_tokens_used("orientation")
        if orient_cap > 0 and orient_used >= orient_cap:
            log.warning(
                "Orientation sub-cap reached (%d >= %d tokens); ending orientation",
                orient_used,
                orient_cap,
            )
            break
        orientation_steps += 1
        _, _, finished = agent._run_one_step(
            report,
            messages,
            allow_finish=False,
            phase_name="orientation",
        )
        if finished:
            orientation_done = True

    run_stats.orientation_tokens = agent.budget.phase_tokens_used("orientation")
    agent._orientation_prompt_message = None
    report_body = load_orientation_report_text(run_dir)
    if not is_orientation_report_substantive(report_body):
        log.warning(
            "Orientation report is thin or empty after orientation (%d chars); "
            "planning will be weak — model should use orientation_report during orientation.",
            len(report_body),
        )
    orientation_for_planning, orient_source = prepare_orientation_report_for_planning(
        run_dir, limits=cfg.text_limits
    )
    try:
        planning_report_tokens = (
            count_tokens(orientation_for_planning) if orientation_for_planning else 0
        )
    except RuntimeError:
        planning_report_tokens = len(orientation_for_planning or "") // 4
    write_orientation_run_metadata(
        run_dir,
        orientation_steps,
        limits=cfg.text_limits,
        orientation_tokens=run_stats.orientation_tokens,
        planning_source=orient_source,
        planning_report_tokens=planning_report_tokens,
    )
    agent._orientation_scratchpad = orientation_for_planning or report_body
    log.info(
        "Orientation complete | %d steps | %d phase tokens | planning input %d tokens "
        "(%s, cap %d, raw report)",
        orientation_steps,
        run_stats.orientation_tokens,
        planning_report_tokens,
        orient_source,
        plan_cap,
    )

    if agent.budget.should_stop():
        run_stats.finish_reason = "global_budget_exhausted"
        _finalize(agent, report, plan=None, run_stats=run_stats, t0=t0)
        return

    # --- Planning (one-shot; raw orientation report, not orientation chat history) ---
    if not orientation_for_planning:
        orientation_for_planning, _ = prepare_orientation_report_for_planning(
            run_dir, limits=cfg.text_limits
        )
    if not orientation_for_planning:
        orientation_for_planning = agent._orientation_scratchpad
        log.warning(
            "orientation/orientation_report.md missing or empty; planning fallback weak"
        )

    agent.budget.start_phase("planning")
    plan_messages: List[Dict[str, Any]] = [
        _user_msg(
            build_planning_prompt(
                orientation_for_planning,
                parallel_mode=False,
                sql_max_per_core=cfg.sql_max_per_core,
                n_schemes=8,
                min_hypotheses_per_scheme=cfg.min_hypotheses_per_scheme,
                max_hypotheses_per_scheme=cfg.max_hypotheses_per_scheme,
                min_dispatch_items=cfg.planning_min_dispatch_items,
                text_limits=cfg.text_limits,
                global_run_token_budget=agent._global_token_budget,
                tokens_remaining_before_planning=agent.budget.remaining_tokens,
                orientation_budget_fraction=cfg.orientation_budget_fraction,
                orchestrator_reserve_budget_fraction=cfg.orchestrator_reserve_budget_fraction,
            )
        )
    ]
    plan_step, _, _ = agent._run_one_step(
        report,
        plan_messages,
        allow_finish=False,
        phase_name="planning",
        response_schema=InvestigationPlan.model_json_schema(),
    )
    plan = agent._parse_plan(plan_step)
    if not plan.phases:
        plan = agent._repair_or_fallback_plan(report, plan_messages, plan)
    if not plan.phases:
        run_stats.finish_reason = "setup_failed"
        report.error_message = "Planning produced no phases"
        _finalize(agent, report, plan=plan, run_stats=run_stats, t0=t0)
        return

    plan, guard_stats = enforce_plan_hypothesis_guardrails(
        plan,
        min_per_scheme=cfg.min_hypotheses_per_scheme,
        max_per_scheme=cfg.max_hypotheses_per_scheme,
    )
    if guard_stats.get("padded") or guard_stats.get("trimmed"):
        log.warning(
            "Plan hypothesis guardrails adjusted counts (min=%d max=%d per scheme): %s",
            cfg.min_hypotheses_per_scheme,
            cfg.max_hypotheses_per_scheme,
            guard_stats,
        )
    if not plan.dispatch_queue:
        run_stats.finish_reason = "setup_failed"
        report.error_message = "No hypotheses in dispatch queue"
        _finalize(agent, report, plan=plan, run_stats=run_stats, t0=t0)
        return

    if cfg.use_planner_weighted_task_budgets or cfg.use_runtime_hypothesis_budget_pool:
        remaining = max(0, agent.budget.remaining_tokens)
        reserve = max(
            50_000,
            int(remaining * cfg.orchestrator_reserve_budget_fraction),
        )
        worker_pool = max(0, remaining - reserve)
        if cfg.use_runtime_hypothesis_budget_pool:
            agent._hypothesis_worker_pool_remaining = worker_pool
            log.info(
                "Runtime hypothesis budget pool: worker_pool=%s tokens (re-split before each batch; min_per_task=%d)",
                f"{worker_pool:,}",
                cfg.hypothesis_task_min_tokens,
            )
        elif cfg.use_planner_weighted_task_budgets:
            assigned = apply_worker_budget_pool_to_dispatch_queue(
                plan.dispatch_queue,
                allocation_pool_tokens=worker_pool,
                min_per_task=max(1, cfg.hypothesis_task_min_tokens),
            )
            log.info(
                "Planner-weighted task budgets: worker_pool=%s assigned_sum=%s across %d tasks",
                f"{worker_pool:,}",
                f"{assigned:,}",
                len(plan.dispatch_queue),
            )

    report.investigation_plan = plan
    agent._update_plan_slot(plan)
    persist_plan_with_queue(run_dir, plan)
    log.info(
        "Investigation plan: %d phases, %d hypothesis tasks",
        len(plan.phases),
        len(plan.dispatch_queue),
    )

    # --- Hypothesis dispatch (queue fixed at plan time; optional injection) ---
    queue = DispatchQueueManager(
        items=list(plan.dispatch_queue),
        max_parallel=max(1, cfg.max_parallel_workers),
    )
    run_stats.tasks_spawned = len(queue.items)
    injected_count = 0
    persist_dispatch_queue_snapshot(run_dir, queue.snapshot())
    _refresh_runtime_hypothesis_budgets(agent, queue, cfg)
    persist_dispatch_queue_snapshot(run_dir, queue.snapshot())
    agent._emit_orchestrator_event(
        "orchestrator_queue_initialized",
        {
            "max_parallel": queue.max_parallel,
            **_queue_snapshot_for_stream(queue),
        },
        echo_terminal=True,
    )

    completed: Dict[str, HypothesisResult] = {}
    errors: Dict[str, str] = {}
    lock = threading.Lock()

    def _fallback_per_task_budget() -> int:
        """Legacy equal split of remaining pool (used for injected tasks or when disabled)."""
        remaining_pool = max(0, agent.budget.remaining_tokens)
        orchestrator_reserve = max(
            50_000,
            int(remaining_pool * cfg.orchestrator_reserve_budget_fraction),
        )
        worker_pool = max(0, remaining_pool - orchestrator_reserve)
        pending_n = len(queue.pending()) or 1
        return max(cfg.hypothesis_task_min_tokens, worker_pool // pending_n)

    def _run_item(item: DispatchQueueItem) -> None:
        if cfg.use_runtime_hypothesis_budget_pool:
            task_budget = max(1, int(item.budget_tokens or 0))
        elif cfg.use_planner_weighted_task_budgets and int(item.budget_tokens or 0) > 0:
            task_budget = max(1, int(item.budget_tokens))
        else:
            task_budget = _fallback_per_task_budget()
            item.budget_tokens = task_budget
        queue.mark_running(item.task_id)
        agent._emit_orchestrator_event(
            "orchestrator_task_dispatch",
            {
                "task_id": item.task_id,
                "scheme": item.scheme,
                "hypothesis_id": item.hypothesis_id,
                "dispatch_priority": item.dispatch_priority,
                "source": item.source,
                "budget_tokens": task_budget,
                "hypothesis_text_preview": truncate_text_to_tokens(
                    item.hypothesis_text or "",
                    agent.config.text_limits.injection_hypothesis_preview,
                    side=TruncationSide.TAIL,
                ),
            },
            echo_terminal=True,
        )
        try:
            result = _execute_hypothesis_task(
                parent=agent,
                plan=plan,
                item=item,
                budget_tokens=task_budget,
                completed_peers=completed,
            )
            with lock:
                completed[item.task_id] = result
            track = build_hypothesis_track(
                run_dir, item, result, limits=agent.config.text_limits
            )
            track_path = save_hypothesis_track(run_dir, track)
            rel_track = str(track_path.relative_to(run_dir))
            extra_paths = [rel_track, f"hypothesis_tracks/{item.task_id}.json"]
            result.artifact_paths = list(
                dict.fromkeys((result.artifact_paths or []) + extra_paths)
            )
            save_hypothesis_result(run_dir, result)
            flush_global_memory(run_dir, result, agent._shared_blackboard)
            agent._merge_detections_from_worker_dir(task_dir(run_dir, item.task_id))
            agent._emit_orchestrator_event(
                "orchestrator_task_finished",
                {
                    "task_id": item.task_id,
                    "scheme": result.scheme,
                    "hypothesis_id": result.hypothesis_id,
                    "result_status": result.status,
                    "finish_reason": result.finish_reason,
                    "n_flagged_documents": len(result.flagged_document_ids or []),
                    "tokens_used": result.tokens_used,
                    "steps": getattr(result, "steps", 0),
                },
            )
            terminal = (
                "completed"
                if result.finish_reason
                in (
                    "completed",
                    "task_budget_95pct",
                    "task_budget_stop_frac",
                    "task_budget_exhausted",
                )
                else "failed"
            )
            queue.mark_terminal(item.task_id, terminal)
            run_stats.tasks_completed += 1
            if result.finish_reason in (
                "task_budget_95pct",
                "task_budget_stop_frac",
                "task_budget_exhausted",
            ):
                run_stats.tasks_stopped_at_95pct += 1
            if terminal == "completed" and cfg.dynamic_hypothesis_injection:
                nonlocal injected_count
                from researchpkg.forensic_llm.hypothesis_injection import (
                    inject_followup_hypotheses,
                )

                n_new = inject_followup_hypotheses(
                    agent,
                    plan,
                    queue,
                    result,
                    injected_count,
                    run_dir,
                )
                injected_count += n_new
                run_stats.tasks_injected += n_new
                run_stats.tasks_spawned += n_new
            plan.dispatch_queue = list(queue.items)
            persist_plan_with_queue(run_dir, plan)
            persist_dispatch_queue_snapshot(run_dir, queue.snapshot())
            if cfg.use_runtime_hypothesis_budget_pool:
                with agent._hypothesis_pool_lock:
                    cur = int(agent._hypothesis_worker_pool_remaining or 0)
                    agent._hypothesis_worker_pool_remaining = max(
                        0, cur - int(result.tokens_used or 0)
                    )
        except Exception as exc:
            log.exception("Hypothesis task %s failed: %s", item.task_id, exc)
            agent._emit_orchestrator_event(
                "orchestrator_task_failed",
                {
                    "task_id": item.task_id,
                    "scheme": item.scheme,
                    "hypothesis_id": item.hypothesis_id,
                    "error": str(exc),
                },
                echo_terminal=True,
            )
            with lock:
                errors[item.task_id] = str(exc)
            queue.mark_terminal(item.task_id, "failed")
            run_stats.tasks_failed += 1
            partial = HypothesisResult(
                scheme=item.scheme,
                hypothesis_id=item.hypothesis_id,
                task_id=item.task_id,
                status="inconclusive",
                finish_reason="worker_error",
                error=str(exc),
            )
            save_hypothesis_result(run_dir, partial)
            try:
                fail_track = build_hypothesis_track(
                    run_dir, item, partial, limits=agent.config.text_limits
                )
                save_hypothesis_track(run_dir, fail_track)
            except Exception:
                log.debug(
                    "Could not build hypothesis track for failed %s", item.task_id
                )
            plan.dispatch_queue = list(queue.items)
            persist_plan_with_queue(run_dir, plan)
            persist_dispatch_queue_snapshot(run_dir, queue.snapshot())

    while not queue.all_terminal() and not agent.budget.should_stop():
        _refresh_runtime_hypothesis_budgets(agent, queue, cfg)
        pending = queue.pending()
        running = queue.running_count()
        slots = queue.max_parallel - running
        if slots <= 0 or not pending:
            time.sleep(0.05)
            continue

        batch = pending[:slots]
        if agent.budget.should_stop():
            break
        if queue.max_parallel == 1:
            for item in batch:
                if agent.budget.should_stop():
                    break
                _run_item(item)
        else:
            # I/O-bound workers (LLM API + DB): threads, not OS processes.
            with ThreadPoolExecutor(
                max_workers=queue.max_parallel,
                thread_name_prefix="hypothesis",
            ) as pool:
                futures = [pool.submit(_run_item, item) for item in batch]
                for fut in as_completed(futures):
                    fut.result()

    plan.dispatch_queue = list(queue.items)
    persist_plan_with_queue(run_dir, plan)
    persist_dispatch_queue_snapshot(run_dir, queue.snapshot())

    run_stats.worker_tokens = sum(r.tokens_used for r in completed.values())
    run_stats.orchestrator_tokens = (
        agent.budget.used_tokens
        - run_stats.orientation_tokens
        - run_stats.worker_tokens
    )

    if agent.budget.should_stop() and not queue.all_terminal():
        run_stats.finish_reason = "global_budget_exhausted"
    elif errors and len(completed) < run_stats.tasks_spawned:
        run_stats.finish_reason = "orchestrator_complete"
    else:
        run_stats.finish_reason = "orchestrator_complete"

    _finalize(agent, report, plan=plan, run_stats=run_stats, t0=t0)


def _execute_hypothesis_task(
    parent: "ForensicAgent",
    plan: InvestigationPlan,
    item: DispatchQueueItem,
    budget_tokens: int,
    completed_peers: Dict[str, HypothesisResult],
) -> HypothesisResult:
    """Run one hypothesis worker (isolated ForensicAgent subdirectory)."""
    from researchpkg.forensic_llm.agent import (
        ForensicAgent,
    )

    run_dir = parent._run_dir
    tdir = task_dir(run_dir, item.task_id)
    tdir.mkdir(parents=True, exist_ok=True)

    peer_results = [r for r in completed_peers.values() if r.scheme == item.scheme]
    orient_text = load_orientation_report_text(run_dir)
    if orient_text and parent.config.text_limits.shared_context_orientation > 0:
        from researchpkg.forensic_llm.text_truncation import (
            TruncationSide,
            truncate_text_to_tokens,
        )

        orient_text = truncate_text_to_tokens(
            orient_text,
            int(parent.config.text_limits.shared_context_orientation),
            side=TruncationSide.HEAD,
        )

    shared_context = build_shared_context_excerpt(
        run_dir,
        item.scheme,
        parent._shared_blackboard,
        peer_results,
        orientation_excerpt=orient_text,
        limits=parent.config.text_limits,
    )
    brief = build_hypothesis_brief(plan, item, budget_tokens, shared_context)

    worker_cfg = copy.deepcopy(parent.config)
    worker_cfg.budget.max_tokens = budget_tokens
    # Align tracker stop with task cap (parent BudgetConfig.stop_threshold defaults to 0.95).
    worker_cfg.budget.stop_threshold = worker_cfg.task_budget_stop_fraction
    worker_cfg.enabled_tools = list(HYPOTHESIS_WORKER_TOOL_NAMES)
    if parent.config.enable_grep and "grep" not in worker_cfg.enabled_tools:
        worker_cfg.enabled_tools.append("grep")
    worker_cfg.stream_trace_terminal = False

    worker = ForensicAgent(worker_cfg, _worker_run_dir=tdir)
    worker._parent_blackboard = parent._shared_blackboard
    if parent.config.stream_trace and parent._stream_trace_file is not None:
        worker._parent_stream_sink = (
            parent._stream_trace_lock,
            run_dir / "audit_trace_stream.ndjson",
        )
    return worker._run_hypothesis_task_worker(brief, parent_run_dir=run_dir)


def _finalize(
    agent: "ForensicAgent",
    report: ForensicReport,
    plan: Optional[InvestigationPlan],
    run_stats: RunStats,
    t0: float,
) -> None:
    """Assemble reports and set termination metadata."""
    run_dir = agent._run_dir
    run_stats.wall_time_seconds = time.time() - t0

    if plan is not None:
        agent._merge_live_detections_into_report(report)
        inv_report = assemble_investigation_report(plan, run_dir, report, run_stats)
        save_investigation_report(run_dir, inv_report)
        md = render_report_markdown(inv_report)
        report_md_path(run_dir).write_text(md, encoding="utf-8")
        report.narrative = md
        report.investigation_plan = plan

    if not report.termination_reason:
        report.termination_reason = run_stats.finish_reason or "completed"

    save_run_stats(run_dir, run_stats)
    log.info(
        "hypothesis_orchestrated run finished | reason=%s | tasks=%d/%d",
        run_stats.finish_reason,
        run_stats.tasks_completed,
        run_stats.tasks_spawned,
    )
