from .investigation_discipline import (
    COMPREHENSIVE_INVESTIGATION_GUIDANCE,
    DISCRIMINATIVENESS_BEFORE_FLAG,
    HYPOTHESIS_INVESTIGATION_DEPTH,
    REPORT_SUSPICION_DISCIPLINE,
)


def build_scheme_phase_prompt(
    scheme_type: str,
    hypotheses: list[str],
    worker_mode: bool = False,
    hypothesis_task_mode: bool = False,
    plan_context: str = "",
) -> str:
    """Build a user message for scheme- or hypothesis-scoped investigation."""
    hyp_text = (
        "\n".join(f"  - {h}" for h in hypotheses)
        if hypotheses
        else "  (derive from the scheme card and shared context)"
    )

    worker_header = ""
    if hypothesis_task_mode:
        worker_header = (
            f"**[HYPOTHESIS TASK — {scheme_type}]**\n\n"
            "You are investigating **one hypothesis only** for this scheme. "
            "Do NOT call `finish_investigation`. "
            "You **must** call `report_suspicion` with `je_header.document_id` UUIDs whenever you "
            "confirm suspicious JEs — evaluation reads `detections.json`, not scratchpad alone. "
            "Use `blackboard_write` for entity findings that may link other schemes.\n\n"
        )
    elif worker_mode:
        worker_header = (
            f"**[SCHEME WORKER — {scheme_type.upper().replace('_', ' ')}]**\n\n"
            "Investigate this scheme in isolation. "
            "Do NOT call `finish_investigation`. "
            "Use `report_suspicion` when you identify journal entries worth flagging.\n\n"
        )

    plan_block = ""
    if plan_context.strip():
        plan_block = f"**Planned execution brief:**\n{plan_context.strip()}\n\n"

    scope_line = (
        f"Focus exclusively on **{scheme_type}** and the hypothesis listed below."
        if hypothesis_task_mode
        else f"Focus on detecting **{scheme_type}** fraud."
    )

    end_check = "Record outcome (CONFIRMED / RULED_OUT / INCONCLUSIVE) in the scratchpad, then stop."

    return (
        worker_header
        + f"**INVESTIGATION: {scheme_type.upper().replace('_', ' ')}**\n\n"
        + scope_line
        + "\n\n"
        + plan_block
        + f"**Hypothesis to test:**\n{hyp_text}\n\n"
        + "**Working discipline:**\n"
        "0. Follow the planned query families unless evidence forces a justified deviation (note in scratchpad).\n"
        + HYPOTHESIS_INVESTIGATION_DEPTH
        + "\n"
        + DISCRIMINATIVENESS_BEFORE_FLAG
        + "\n"
        + COMPREHENSIVE_INVESTIGATION_GUIDANCE
        + "\n"
        + REPORT_SUSPICION_DISCIPLINE
        + "\n"
        "Use `report_suspicion` only when SQL evidence supports a clear fraud pattern.\n"
        "**Scratchpad:** update after completing this hypothesis.\n" + end_check
    )
