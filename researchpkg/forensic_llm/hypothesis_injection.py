"""Dynamic hypothesis injection after each completed task (core_spec v2)."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List

from researchpkg.forensic_llm.dispatch_queue import (
    DispatchQueueManager,
)
from researchpkg.forensic_llm.models import (
    DispatchQueueItem,
    HypothesisResult,
    InvestigationPlan,
)
from researchpkg.forensic_llm.plan_utils import task_id_for
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)

if TYPE_CHECKING:
    from researchpkg.forensic_llm.agent import (
        ForensicAgent,
    )

log = logging.getLogger(__name__)

_HYP_NUM = re.compile(r"^P(\d+)$", re.IGNORECASE)


def _next_hypothesis_id(scheme: str, queue: DispatchQueueManager) -> str:
    max_n = 0
    with queue._lock:
        for item in queue.items:
            if item.scheme != scheme:
                continue
            m = _HYP_NUM.match(item.hypothesis_id.strip())
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"P{max_n + 1}"


def _clip_hypothesis_text(text: str, agent: "ForensicAgent") -> str:
    lim = agent.config.text_limits
    return truncate_text_to_tokens(
        text,
        lim.hypothesis_text_store,
        side=TruncationSide.TAIL,
    )


def _clip_hypothesis_rationale(text: str, agent: "ForensicAgent") -> str:
    lim = agent.config.text_limits
    return truncate_text_to_tokens(
        text,
        lim.hypothesis_rationale_store,
        side=TruncationSide.TAIL,
    )


def _rule_based_followups(
    result: HypothesisResult,
    queue: DispatchQueueManager,
    max_items: int,
    agent: "ForensicAgent",
) -> List[DispatchQueueItem]:
    """Derive follow-ups from open_questions without an LLM call."""
    items: List[DispatchQueueItem] = []
    priority = queue.min_pending_priority() - 1
    questions = [q.strip() for q in (result.open_questions or []) if q.strip()]
    if result.status != "confirmed":
        questions = questions[: max(0, max_items // 2)]
    else:
        questions = questions[:max_items]

    for q in questions[:max_items]:
        if len(q) < 12:
            continue
        hyp_id = _next_hypothesis_id(result.scheme, queue)
        task_id = task_id_for(result.scheme, hyp_id)
        if queue.contains_task(task_id):
            continue
        text = (
            q
            if q.lower().startswith(("if ", "trace", "verify", "test"))
            else f"Follow up on open question: {q}"
        )
        items.append(
            DispatchQueueItem(
                scheme=result.scheme,
                hypothesis_id=hyp_id,
                dispatch_priority=priority,
                task_id=task_id,
                hypothesis_text=_clip_hypothesis_text(text, agent),
                source="orchestrator",
            )
        )
        priority -= 1
    return items


def _sync_plan_queue(plan: InvestigationPlan, queue: DispatchQueueManager) -> None:
    plan.dispatch_queue = [
        DispatchQueueItem.model_validate(d) for d in queue.snapshot()
    ]


def _emit_hypothesis_injection_event(
    agent: "ForensicAgent",
    last_result: HypothesisResult,
    new_items: List[DispatchQueueItem],
    channels: List[str],
) -> None:
    if not new_items:
        return
    emit = getattr(agent, "_emit_orchestrator_event", None)
    if emit is None:
        return
    lim = agent.config.text_limits
    items_payload = [
        {
            "task_id": it.task_id,
            "scheme": it.scheme,
            "hypothesis_id": it.hypothesis_id,
            "dispatch_priority": it.dispatch_priority,
            "hypothesis_text_preview": truncate_text_to_tokens(
                it.hypothesis_text or "",
                lim.injection_hypothesis_preview,
                side=TruncationSide.TAIL,
            ),
            "source": it.source,
        }
        for it in new_items
    ]
    detail = [
        {"task_id": it.task_id, "channel": ch} for it, ch in zip(new_items, channels)
    ]
    emit(
        "orchestrator_hypothesis_injected",
        {
            "after_task_id": last_result.task_id,
            "after_scheme": last_result.scheme,
            "n_injected": len(new_items),
            "task_ids": [i.task_id for i in new_items],
            "injection_detail": detail,
            "items": items_payload,
        },
        echo_terminal=True,
    )


def inject_followup_hypotheses(
    agent: "ForensicAgent",
    plan: InvestigationPlan,
    queue: DispatchQueueManager,
    last_result: HypothesisResult,
    injected_so_far: int,
    run_dir,
) -> int:
    """Enqueue new hypothesis tasks from findings. Returns count newly injected."""
    from researchpkg.forensic_llm.artefacts import (
        load_global_memory,
        persist_plan_with_queue,
    )
    from researchpkg.forensic_llm.memory_store import (
        render_global_memory_excerpt,
    )
    from researchpkg.forensic_llm.prompts.orchestrator_followup import (
        build_orchestrator_followup_prompt,
        parse_followup_proposals,
    )

    cfg = agent.config
    lim = cfg.text_limits
    if not cfg.dynamic_hypothesis_injection:
        return 0
    if injected_so_far >= cfg.max_dynamic_hypotheses:
        return 0
    if agent.budget.should_stop():
        return 0

    cap = cfg.max_dynamic_hypotheses - injected_so_far
    new_items: List[DispatchQueueItem] = []
    channels: List[str] = []

    for item in _rule_based_followups(last_result, queue, min(2, cap), agent):
        if queue.inject(item):
            new_items.append(item)
            channels.append("rule_based_open_question")

    remaining = cap - len(new_items)
    if remaining <= 0:
        if new_items:
            _sync_plan_queue(plan, queue)
            persist_plan_with_queue(run_dir, plan)
            _emit_hypothesis_injection_event(agent, last_result, new_items, channels)
        return len(new_items)

    if not (
        last_result.open_questions
        or last_result.status == "confirmed"
        or last_result.key_findings
    ):
        if new_items:
            _sync_plan_queue(plan, queue)
            persist_plan_with_queue(run_dir, plan)
            _emit_hypothesis_injection_event(agent, last_result, new_items, channels)
        return len(new_items)

    memory = load_global_memory(run_dir)
    excerpt = render_global_memory_excerpt(memory, last_result.scheme, limits=lim)
    existing_ids = [i.task_id for i in queue.items]

    from researchpkg.forensic_llm.agent import _user_msg
    from researchpkg.forensic_llm.prompts import (
        build_system_prompt,
    )
    from researchpkg.forensic_llm.tools import (
        get_scratchpad_text,
    )

    agent.budget.start_phase("orchestrator_inject")
    prompt = build_orchestrator_followup_prompt(
        last_result,
        excerpt,
        existing_ids,
        max_proposals=min(2, remaining),
        text_limits=lim,
    )
    system = build_system_prompt(
        task=agent.config.task,
        scratchpad_text=truncate_text_to_tokens(
            get_scratchpad_text(),
            lim.injection_scratchpad,
            side=TruncationSide.TAIL,
        ),
        is_worker=False,
    )
    try:
        resp = agent._call_llm([_user_msg(prompt)], system)
        proposals = parse_followup_proposals(resp.content or "")
    except Exception as exc:
        log.warning("Follow-up proposal LLM failed: %s", exc)
        proposals = []

    priority = queue.min_pending_priority() - 1
    for prop in proposals[:remaining]:
        scheme = str(prop.get("scheme") or last_result.scheme).strip()
        text = str(prop.get("hypothesis_text") or "").strip()
        if not text:
            continue
        hyp_id = _next_hypothesis_id(scheme, queue)
        task_id = task_id_for(scheme, hyp_id)
        if queue.contains_task(task_id):
            continue
        prio = int(prop.get("dispatch_priority", priority))
        prop_rationale = str(
            prop.get("hypothesis_rationale") or prop.get("rationale") or ""
        ).strip()
        item = DispatchQueueItem(
            scheme=scheme,
            hypothesis_id=hyp_id,
            dispatch_priority=prio,
            task_id=task_id,
            hypothesis_text=_clip_hypothesis_text(text, agent),
            hypothesis_rationale=_clip_hypothesis_rationale(prop_rationale, agent),
            source="orchestrator",
        )
        if queue.inject(item):
            new_items.append(item)
            priority = prio - 1
            channels.append("llm_followup_proposal")

    if new_items:
        _sync_plan_queue(plan, queue)
        persist_plan_with_queue(run_dir, plan)
        log.info(
            "Orchestrator injected %d follow-up hypothesis task(s): %s",
            len(new_items),
            [i.task_id for i in new_items],
        )
        _emit_hypothesis_injection_event(agent, last_result, new_items, channels)
    return len(new_items)
