"""Build per-hypothesis investigation tracks from worker task directories."""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, List, Optional

from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.models import (
    DispatchQueueItem,
    HypothesisInvestigationTrack,
    HypothesisResult,
    HypothesisStepTrace,
    ToolActionTrace,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)


def _preview(text: str, max_tokens: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    if max_tokens <= 0:
        return t
    return truncate_text_to_tokens(t, max_tokens, side=TruncationSide.HEAD)


def _tool_args_summary(
    tool: str, args: Dict[str, Any], limits: TextTruncationLimits
) -> str:
    if tool == "sql":
        mode = str(args.get("mode", "preview")).lower()
        q = str(args.get("query", ""))
        q = re.sub(r"\s+", " ", q).strip()
        if mode == "export":
            fn = args.get("filename") or "auto.csv"
            return f"export {fn}: {_preview(q, limits.track_sql_query_preview)}"
        return _preview(q, limits.track_sql_query_preview)
    if tool == "write_csv":
        return (
            f"export {args.get('filename', 'export.csv')}: "
            f"{_preview(str(args.get('query', '')), limits.track_sql_query_preview)}"
        )
    if tool == "code_interpreter":
        return _preview(str(args.get("code", "")), limits.track_tool_code_preview)
    if tool == "scratchpad":
        return _preview(str(args.get("note", "")), limits.track_tool_note_preview)
    if tool == "report_suspicion":
        did = args.get("document_id") or ""
        if did:
            return f"document_id={did}"
        n = len(args.get("suspicions") or [])
        return f"{n} suspicion(s)" if n else "report"
    return _preview(json.dumps(args, ensure_ascii=False), limits.track_json_preview)


def _extract_tool_results_from_request(
    request: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Map tool_call_id -> result text from the next step's request payload."""
    out: Dict[str, str] = {}
    for msg in request or []:
        if msg.get("role") != "tool":
            continue
        tid = str(msg.get("tool_call_id") or "")
        content = msg.get("content", "")
        if isinstance(content, str):
            out[tid] = content
        elif isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            out[tid] = "\n".join(parts)
    return out


def build_hypothesis_track(
    run_dir: pathlib.Path,
    item: DispatchQueueItem,
    result: HypothesisResult,
    *,
    limits: Optional[TextTruncationLimits] = None,
) -> HypothesisInvestigationTrack:
    """Assemble investigation trace from ``tasks/<task_id>/`` artefacts."""
    lim = limits or TextTruncationLimits()
    tdir = run_dir / "tasks" / item.task_id
    calls_dir = tdir / "calls_steps"
    step_traces: List[HypothesisStepTrace] = []
    evidence: List[str] = []

    call_files = sorted(calls_dir.glob("step_*.json")) if calls_dir.is_dir() else []
    for idx, path in enumerate(call_files):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        step_no = int(raw.get("step_number") or idx + 1)
        resp = raw.get("response") or {}
        reasoning = (resp.get("reasoning") or resp.get("content") or "").strip()
        tool_results: Dict[str, str] = {}
        if idx + 1 < len(call_files):
            try:
                nxt = json.loads(call_files[idx + 1].read_text(encoding="utf-8"))
                tool_results = _extract_tool_results_from_request(
                    nxt.get("request") or []
                )
            except (json.JSONDecodeError, OSError):
                pass

        actions: List[ToolActionTrace] = []
        for tc in resp.get("tool_calls") or []:
            name = tc.get("name") or ""
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tid = str(tc.get("id") or "")
            result_preview = _preview(
                tool_results.get(tid, ""), lim.track_tool_code_preview
            )
            arg_dict = args if isinstance(args, dict) else {}
            actions.append(
                ToolActionTrace(
                    tool=name,
                    args_summary=_tool_args_summary(name, arg_dict, lim),
                    result_preview=result_preview,
                )
            )
            if name == "sql":
                evidence.append(
                    f"step {step_no}: sql — {_tool_args_summary('sql', arg_dict, lim)}"
                )
            elif name in ("write_csv", "code_interpreter", "graph_query"):
                evidence.append(f"step {step_no}: {name}")

        step_traces.append(
            HypothesisStepTrace(
                step_number=step_no,
                reasoning_snippet=_preview(reasoning, lim.track_reasoning_preview),
                tools=actions,
                prompt_tokens=int(resp.get("prompt_tokens") or 0),
                completion_tokens=int(resp.get("completion_tokens") or 0),
            )
        )

    scratch = ""
    sp = tdir / "scratchpad.md"
    if sp.is_file():
        scratch = truncate_text_to_tokens(
            sp.read_text(encoding="utf-8"),
            lim.hypothesis_track_scratchpad,
            side=TruncationSide.TAIL,
        )

    rel = (
        lambda p: str(p.relative_to(run_dir))
        if p.is_file()
        else str(p.relative_to(run_dir))
    )
    artefact_paths = {
        "task_dir": rel(tdir),
        "result_json": rel(
            run_dir / "schemes" / result.scheme / f"{result.hypothesis_id.lower()}.json"
        ),
        "scratchpad": rel(sp) if sp.is_file() else "",
        "audit_trace": rel(tdir / "audit_trace_stream.ndjson"),
        "calls_steps_dir": rel(calls_dir) if calls_dir.is_dir() else "",
        "memory": rel(tdir / "memory.md") if (tdir / "memory.md").is_file() else "",
    }
    plots = sorted((tdir / "plots").glob("*")) if (tdir / "plots").is_dir() else []
    if plots:
        artefact_paths["plots"] = ", ".join(rel(p) for p in plots[:8])

    eff = getattr(result, "effective_sql_calls", None)
    if eff is None:
        eff = result.sql_calls

    return HypothesisInvestigationTrack(
        task_id=item.task_id,
        scheme=item.scheme,
        hypothesis_id=item.hypothesis_id,
        hypothesis_text=result.hypothesis_text or item.hypothesis_text,
        hypothesis_rationale=(
            result.hypothesis_rationale
            or getattr(item, "hypothesis_rationale", "")
            or ""
        ),
        dispatch_priority=item.dispatch_priority,
        dispatch_sequence=item.dispatch_sequence,
        source=item.source,
        queue_status=item.status,
        result_status=result.status,
        finish_reason=result.finish_reason,
        tokens_used=result.tokens_used,
        steps=result.steps or len(step_traces),
        sql_calls=result.sql_calls,
        effective_sql_calls=eff,
        flagged_document_ids=list(result.flagged_document_ids or []),
        key_findings=result.key_findings or "",
        open_questions=list(result.open_questions or []),
        evidence_checks_run=result.evidence_checks_run or evidence[:40],
        step_traces=step_traces,
        scratchpad_excerpt=_preview(scratch, 4000),
        artefact_paths=artefact_paths,
    )


def save_hypothesis_track(
    run_dir: pathlib.Path,
    track: HypothesisInvestigationTrack,
) -> pathlib.Path:
    from researchpkg.forensic_llm.artefacts import (
        write_json,
    )

    path = run_dir / "tasks" / track.task_id / "hypothesis_track.json"
    write_json(path, track)
    index_path = run_dir / "hypothesis_tracks" / f"{track.task_id}.json"
    write_json(index_path, track)
    return path


def backfill_hypothesis_tracks(run_dir: pathlib.Path) -> int:
    """
    Build ``hypothesis_tracks/`` from an existing run (tasks/*/calls_steps).

    Returns the number of tracks written.
    """
    import json

    from researchpkg.forensic_llm.artefacts import (
        load_all_hypothesis_results,
    )
    from researchpkg.forensic_llm.models import (
        DispatchQueueItem,
        InvestigationPlan,
    )
    from researchpkg.forensic_llm.plan_utils import (
        task_id_for,
    )

    plan_path = run_dir / "investigation_plan.json"
    if not plan_path.is_file():
        return 0
    plan = InvestigationPlan.model_validate(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    by_task = {i.task_id: i for i in (plan.dispatch_queue or []) if i.task_id}
    results = load_all_hypothesis_results(run_dir)
    written = 0
    for result in results:
        tid = result.task_id or task_id_for(result.scheme, result.hypothesis_id)
        item = by_task.get(tid)
        if item is None:
            item = DispatchQueueItem(
                scheme=result.scheme,
                hypothesis_id=result.hypothesis_id,
                task_id=tid,
                hypothesis_text=result.hypothesis_text,
            )
        save_hypothesis_track(run_dir, build_hypothesis_track(run_dir, item, result))
        written += 1
    return written


def load_all_hypothesis_tracks(
    run_dir: pathlib.Path,
) -> List[HypothesisInvestigationTrack]:
    tracks: List[HypothesisInvestigationTrack] = []
    root = run_dir / "hypothesis_tracks"
    if not root.is_dir():
        return tracks
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            tracks.append(HypothesisInvestigationTrack.model_validate(raw))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return tracks
