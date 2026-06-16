"""Deterministic assembly of investigation_report.json and report.md."""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List

from researchpkg.forensic_llm.artefacts import (
    load_all_hypothesis_results,
    load_global_memory,
)
from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.hypothesis_track import (
    load_all_hypothesis_tracks,
)
from researchpkg.forensic_llm.models import (
    ForensicReport,
    InvestigationPlan,
    RunStats,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)

_PREVIEW_TOKENS = TextTruncationLimits().track_tool_code_preview


def assemble_investigation_report(
    plan: InvestigationPlan,
    run_dir: pathlib.Path,
    report: ForensicReport,
    run_stats: RunStats,
) -> Dict[str, Any]:
    """Merge plan, hypothesis results, tracks, global memory, and suspicions."""
    memory = load_global_memory(run_dir)
    hypothesis_results = load_all_hypothesis_results(run_dir)
    hypothesis_tracks = load_all_hypothesis_tracks(run_dir)
    hypothesis_tracks.sort(
        key=lambda t: (
            t.dispatch_priority,
            t.dispatch_sequence,
            t.task_id or t.hypothesis_id,
        )
    )
    suspicions = [
        {
            "document_id": s.document_id,
            "scheme_type": s.scheme_type.value if s.scheme_type else "unknown",
            "confidence": s.confidence,
            "rationale": s.rationale,
        }
        for s in (report.suspicion_list or [])
        if s.document_id
    ]
    return {
        "run_id": report.run_id,
        "task": report.task,
        "model": report.model,
        "finish_reason": run_stats.finish_reason or report.termination_reason,
        "dispatch_queue": [i.model_dump() for i in plan.dispatch_queue],
        "hypothesis_results": [h.model_dump() for h in hypothesis_results],
        "hypothesis_tracks": [t.model_dump() for t in hypothesis_tracks],
        "global_memory": memory.model_dump(),
        "suspicion_count": len(suspicions),
        "suspicions": suspicions,
        "run_stats": run_stats.model_dump(),
        "orientation_risk_summary": plan.orientation_risk_summary,
        "execution_notes": plan.execution_notes,
    }


def _render_track_section(track: Dict[str, Any]) -> List[str]:
    lines = [
        f"### [{track.get('dispatch_priority', '?')}] "
        f"{track.get('scheme')} — {track.get('hypothesis_id')} "
        f"(`{track.get('task_id', '')}`)",
        "",
    ]
    if track.get("hypothesis_text"):
        lines.append(f"**Hypothesis:** {track['hypothesis_text']}")
    if track.get("hypothesis_rationale"):
        lines.append(f"**Planner rationale:** {track['hypothesis_rationale']}")
        lines.append("")
    lines.append(
        f"**Outcome:** {track.get('result_status')} ({track.get('finish_reason')}) · "
        f"queue={track.get('queue_status')} · source={track.get('source', 'plan')}"
    )
    eff = track.get("effective_sql_calls", 0)
    sql = track.get("sql_calls", 0)
    lines.append(
        f"**Usage:** {track.get('steps', 0)} steps · "
        f"{track.get('tokens_used', 0)} tokens · "
        f"sql={sql} · effective_sql={eff}"
    )
    flagged = track.get("flagged_document_ids") or []
    if flagged:
        lines.append(f"**Flagged JEs:** {len(flagged)}")
    lines.append("")

    if track.get("key_findings"):
        lines.append("**Findings**")
        lines.append("")
        lines.append(track["key_findings"])
        lines.append("")

    evidence = track.get("evidence_checks_run") or []
    if evidence:
        lines.append("**Evidence checks**")
        lines.append("")
        for e in evidence[:12]:
            lines.append(f"- {e}")
        if len(evidence) > 12:
            lines.append(f"- … and {len(evidence) - 12} more")
        lines.append("")

    for step in track.get("step_traces") or []:
        sn = step.get("step_number", "?")
        lines.append(f"#### Step {sn}")
        if step.get("reasoning_snippet"):
            lines.append(f"> {step['reasoning_snippet']}")
        for tool in step.get("tools") or []:
            name = tool.get("tool", "")
            args = tool.get("args_summary", "")
            preview = tool.get("result_preview", "")
            lines.append(f"- **{name}**: `{args}`")
            if preview:
                lines.append(
                    f"  - result: {truncate_text_to_tokens(preview, _PREVIEW_TOKENS, side=TruncationSide.TAIL)}"
                )
        lines.append("")

    paths = track.get("artefact_paths") or {}
    if paths:
        lines.append("**Artefacts**")
        lines.append("")
        for k, v in paths.items():
            if v:
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    return lines


def render_report_markdown(payload: Dict[str, Any]) -> str:
    """Markdown report with per-hypothesis investigation detail."""
    lines = [
        "# Forensic Investigation Report",
        "",
        f"**Run ID:** `{payload.get('run_id', '')}`",
        f"**Task:** {payload.get('task', '')}",
        f"**Model:** {payload.get('model', '')}",
        f"**Finish reason:** {payload.get('finish_reason', '')}",
        "",
        "## Summary",
        "",
    ]
    stats = payload.get("run_stats") or {}
    lines.append(
        f"- Tasks completed: {stats.get('tasks_completed', 0)} / "
        f"{stats.get('tasks_spawned', 0)} spawned"
    )
    if stats.get("tasks_injected"):
        lines.append(f"- Dynamic injections: {stats.get('tasks_injected', 0)}")
    lines.append(f"- Suspicion count: {payload.get('suspicion_count', 0)}")
    lines.append("")

    tracks = payload.get("hypothesis_tracks") or []
    if tracks:
        lines.append("## Hypothesis investigations (dispatch order)")
        lines.append("")
        for track in tracks:
            lines.extend(_render_track_section(track))
    else:
        lines.append("## Hypothesis outcomes")
        lines.append("")
        for hr in payload.get("hypothesis_results") or []:
            lines.append(
                f"### {hr.get('scheme')} — {hr.get('hypothesis_id')} "
                f"({hr.get('status')}, {hr.get('finish_reason')})"
            )
            if hr.get("hypothesis_text"):
                lines.append(hr["hypothesis_text"])
            if hr.get("key_findings"):
                lines.append(hr["key_findings"])
            flagged = hr.get("flagged_document_ids") or []
            if flagged:
                lines.append(f"\nFlagged JEs: {len(flagged)}")
            lines.append("")

    gm = payload.get("global_memory") or {}
    if gm.get("salient_findings"):
        lines.append("## Salient findings")
        lines.append("")
        for f in gm["salient_findings"]:
            lines.append(f"- {f}")
        lines.append("")

    if gm.get("open_risks"):
        lines.append("## Open risks")
        lines.append("")
        for r in gm["open_risks"]:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)
