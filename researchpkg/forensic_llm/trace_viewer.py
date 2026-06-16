"""
Full detailed and streaming view of the forensic audit trace.

- Format a saved audit_trace.json (or run directory) into a human-readable
  full-detail report (reasoning, tool args, tool results, tokens, timestamps).
- Optional follow mode: tail a live audit_trace_stream.ndjson and print each
  new step as it is appended during a running investigation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Optional: use pydantic models if available for consistent serialisation
try:
    from .models import AgentStep, ForensicReport

    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False

from researchpkg.forensic_llm.config import (
    ContextTokenConfig,
    TextTruncationLimits,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)

# Default max tokens for tool result in detailed view (0 = no limit)
DEFAULT_RESULT_TRUNCATE_TOKENS = ContextTokenConfig().excerpts.trace_tool_result


def _truncate_tokens(s: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not s:
        return s
    return truncate_text_to_tokens(s, max_tokens, side=TruncationSide.TAIL)


def _format_timestamp(ts: Any) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def format_step(
    step: Dict[str, Any],
    result_truncate_tokens: int = DEFAULT_RESULT_TRUNCATE_TOKENS,
) -> str:
    """Format a single agent step with full detail (reasoning, tool calls, tokens)."""
    lines = []
    step_num = step.get("step_number", "?")
    ts = step.get("timestamp")
    lines.append("")
    lines.append("═" * 80)
    lines.append("  STEP %s  %s" % (step_num, _format_timestamp(ts)))
    lines.append("═" * 80)

    # Token counts for this step (budget vs informational)
    tin = step.get("llm_tokens_input", 0)
    tout = step.get("llm_tokens_output", 0)
    treason = step.get("llm_tokens_reasoning", 0)
    tbudget = step.get("llm_tokens_budgeted", tin + treason)
    lines.append(
        "  LLM tokens: input=%s  output=%s  reasoning=%s  budget_counted=%s"
        % (tin, tout, treason, tbudget)
    )
    lines.append(
        "  (completion/output is informational; tool tokens_output below is tool "
        "context size, not billed LLM output)"
    )
    lines.append("")

    # Reasoning (full)
    reasoning = step.get("reasoning") or ""
    if reasoning.strip():
        lines.append("  ── Reasoning ──")
        for line in reasoning.strip().split("\n"):
            lines.append("  %s" % line)
        lines.append("")
    else:
        tool_calls = step.get("tool_calls") or []
        if tool_calls:
            lines.append(
                "  ── Reasoning: (none — model returned only tool calls; "
                "add brief reasoning in your response to see it here)"
            )
        else:
            lines.append("  ── Reasoning: (none)")
        lines.append("")

    # Tool calls
    tool_calls = step.get("tool_calls") or []
    if not tool_calls:
        lines.append("  ── Tool calls: (none)")
    else:
        for i, tc in enumerate(tool_calls, 1):
            lines.append("  ── Tool call %d: %s ──" % (i, tc.get("tool", "?")))
            args = tc.get("args") or {}
            try:
                args_str = json.dumps(args, indent=2)
            except Exception:
                args_str = str(args)
            for line in args_str.split("\n"):
                lines.append("    %s" % line)
            lines.append("    elapsed_ms: %s" % tc.get("elapsed_ms"))
            lines.append("    tool_context_tokens: %s" % tc.get("tokens_output", 0))
            if tc.get("error"):
                lines.append("    error: %s" % tc.get("error"))
            result = tc.get("result")
            if result is not None:
                lines.append("    result:")
                for line in _truncate_tokens(result, result_truncate_tokens).split(
                    "\n"
                ):
                    lines.append("      %s" % line)
            lines.append("")

    return "\n".join(lines)


def format_stream_line(
    obj: Dict[str, Any],
    result_truncate_tokens: int = DEFAULT_RESULT_TRUNCATE_TOKENS,
) -> str:
    """Format one NDJSON line from ``audit_trace_stream.ndjson``."""
    if obj.get("event") == "blackboard":
        return (
            "\n  [legacy stream line: event=blackboard — no longer emitted by "
            "the agent]\n"
        )
    if obj.get("event") == "suspicion_reported":
        lines = [
            "",
            "!" * 80,
            "  SUSPICION REPORTED  %s" % _format_timestamp(obj.get("timestamp")),
        ]
        task_id = obj.get("task_id")
        if task_id:
            lines.append("  task: %s" % task_id)
        lines.append("  source: %s" % obj.get("source", "report_suspicion"))
        lines.append("  total_detections: %s" % obj.get("total_detections", "?"))
        lines.append("!" * 80)
        for i, item in enumerate(obj.get("items") or [], 1):
            lines.append(
                "  %d. %s  document_id=%s  confidence=%.2f"
                % (
                    i,
                    item.get("scheme", "?"),
                    item.get("document_id", "?"),
                    float(item.get("confidence", 0.0) or 0.0),
                )
            )
            rationale = (item.get("rationale") or "").strip()
            if rationale:
                lines.append(
                    "     %s"
                    % _truncate_tokens(
                        rationale,
                        TextTruncationLimits().trace_rationale,
                    )
                )
        lines.append("")
        return "\n".join(lines)
    ev = obj.get("event") or ""
    if isinstance(ev, str) and ev.startswith("orchestrator_"):
        lines = [
            "",
            "─" * 80,
            "  ORCHESTRATOR  %s  %s" % (ev, _format_timestamp(obj.get("timestamp"))),
            "─" * 80,
        ]
        skip = {"event", "timestamp", "step_number", "run_id"}
        for k, v in sorted(obj.items()):
            if k in skip:
                continue
            if k == "tasks" and isinstance(v, list):
                lines.append("  %s: (%d entries)" % (k, len(v)))
                for i, row in enumerate(v[:15], 1):
                    lines.append(
                        "    %d. %s  %s  prio=%s  %s"
                        % (
                            i,
                            row.get("task_id", "?"),
                            row.get("scheme", "?"),
                            row.get("dispatch_priority", "?"),
                            row.get("status", "?"),
                        )
                    )
                if len(v) > 15:
                    lines.append("    … %d more" % (len(v) - 15))
                continue
            if k in ("items", "injection_detail") and isinstance(v, list):
                lines.append("  %s:" % k)
                for row in v[:12]:
                    lines.append(
                        "    %s"
                        % _truncate_tokens(
                            json.dumps(row, default=str),
                            TextTruncationLimits().track_json_preview,
                        )
                    )
                if len(v) > 12:
                    lines.append("    … %d more" % (len(v) - 12))
                continue
            lines.append(
                "  %s: %s"
                % (
                    k,
                    _truncate_tokens(
                        str(v), TextTruncationLimits().auto_scratchpad_tail
                    ),
                )
            )
        lines.append("")
        return "\n".join(lines)
    return format_step(obj, result_truncate_tokens=result_truncate_tokens)


def format_trace(
    report: Dict[str, Any],
    result_truncate_tokens: int = DEFAULT_RESULT_TRUNCATE_TOKENS,
    include_narrative: bool = True,
    narrative_max_tokens: int = TextTruncationLimits().trace_narrative,
) -> str:
    """
    Produce a full detailed text view of the forensic trace.

    report: loaded audit_trace (dict from ForensicReport.model_dump() or JSON).
    result_truncate_tokens: max tokens per tool result (0 = no truncation).
    include_narrative: append narrative and suspicion summary at the end.
    narrative_max_tokens: truncate narrative to this many tokens if > 0.
    """
    lines = []
    lines.append("")
    lines.append("╔" + "═" * 78 + "╗")
    lines.append("║  FORENSIC AUDIT TRACE – FULL DETAIL" + " " * 43 + "║")
    lines.append("╚" + "═" * 78 + "╝")
    lines.append("")
    lines.append("  run_id:        %s" % report.get("run_id", "—"))
    lines.append("  model:         %s" % report.get("model", "—"))
    lines.append("  task:          %s" % report.get("task", "—"))
    lines.append("  started_at:    %s" % _format_timestamp(report.get("started_at")))
    lines.append("  completed_at:  %s" % _format_timestamp(report.get("completed_at")))
    lines.append("  steps_taken:   %s" % report.get("steps_taken", "—"))
    lines.append("  total_tokens_input:  %s" % report.get("total_tokens_input", 0))
    lines.append("  total_tokens_output: %s" % report.get("total_tokens_output", 0))
    lines.append("  budget_exhausted:    %s" % report.get("budget_exhausted", False))
    lines.append("")

    steps = report.get("steps") or []
    for step in steps:
        lines.append(format_step(step, result_truncate_tokens=result_truncate_tokens))

    if include_narrative:
        lines.append("")
        lines.append("═" * 80)
        lines.append(
            "  SUSPICION LIST (%d items)" % len(report.get("suspicion_list") or [])
        )
        lines.append("═" * 80)
        for i, s in enumerate(report.get("suspicion_list") or [], 1):
            lines.append(
                "  %d. %s  confidence=%.2f  severity=%s"
                % (
                    i,
                    s.get("scheme_type", "?"),
                    s.get("confidence", 0),
                    s.get("severity", "—"),
                )
            )
            if s.get("document_id"):
                lines.append("     document_id=%s" % s.get("document_id"))
            if s.get("rationale"):
                lines.append(
                    "     %s"
                    % _truncate_tokens(
                        s.get("rationale", ""),
                        TextTruncationLimits().trace_rationale,
                    )
                )
        lines.append("")
        narrative = report.get("narrative") or ""
        if narrative:
            lines.append("═" * 80)
            lines.append("  NARRATIVE")
            lines.append("═" * 80)
            if narrative_max_tokens > 0:
                narrative = _truncate_tokens(narrative, narrative_max_tokens)
            for line in narrative.split("\n"):
                lines.append("  %s" % line)
        lines.append("")

    return "\n".join(lines)


def load_trace(path: Path) -> Dict[str, Any]:
    """Load audit trace from a path (audit_trace.json or run directory)."""
    p = path.resolve()
    if p.is_dir():
        p = p / "audit_trace.json"
    if not p.exists():
        raise FileNotFoundError("Trace not found: %s" % path)
    text = p.read_text(encoding="utf-8")
    return json.loads(text)


def view_static(
    path: Path, result_truncate_tokens: int = DEFAULT_RESULT_TRUNCATE_TOKENS
) -> None:
    """Load trace from path and print full detailed view to stdout."""
    report = load_trace(path)
    print(format_trace(report, result_truncate_tokens=result_truncate_tokens))


def view_stream(
    run_dir: Path, result_truncate_tokens: int = DEFAULT_RESULT_TRUNCATE_TOKENS
) -> None:
    """
    Follow audit_trace_stream.ndjson in run_dir and print each new step as it arrives.
    If the stream file does not exist yet, wait for it (e.g. investigation starting).
    """
    stream_path = run_dir.resolve() / "audit_trace_stream.ndjson"
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError("Not a run directory: %s" % run_dir)

    print("[trace_viewer] Following %s (Ctrl+C to stop)" % stream_path, file=sys.stderr)
    print(
        "[trace_viewer] Waiting for stream file..." if not stream_path.exists() else "",
        file=sys.stderr,
    )

    while not stream_path.exists():
        time.sleep(0.5)

    with open(stream_path, "r", encoding="utf-8") as f:
        # Read existing lines first
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                step = json.loads(line)
                print(
                    format_stream_line(
                        step, result_truncate_tokens=result_truncate_tokens
                    )
                )
                sys.stdout.flush()
            except json.JSONDecodeError:
                pass
        # Then follow new lines
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                step = json.loads(line)
                print(
                    format_stream_line(
                        step, result_truncate_tokens=result_truncate_tokens
                    )
                )
                sys.stdout.flush()
            except json.JSONDecodeError:
                pass


def main(argv=None) -> int:
    """CLI for viewing trace: view_static(path) or view_stream(run_dir)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Full detailed and streaming view of forensic audit trace."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to run directory or audit_trace.json",
    )
    parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow live stream (run_dir/audit_trace_stream.ndjson)",
    )
    parser.add_argument(
        "--no-truncate", action="store_true", help="Do not truncate long tool results"
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=DEFAULT_RESULT_TRUNCATE_TOKENS,
        help="Max tokens per tool result (default %d, 0=no limit)"
        % DEFAULT_RESULT_TRUNCATE_TOKENS,
    )
    args = parser.parse_args(argv)

    if args.no_truncate:
        result_truncate_tokens = 0
    else:
        result_truncate_tokens = args.truncate

    path = args.path
    if path is None:
        parser.error("path is required (run directory or audit_trace.json)")
    path = path.resolve()

    try:
        if args.follow:
            if path.is_file():
                parser.error("--follow requires a run directory, not a file")
            view_stream(path, result_truncate_tokens=result_truncate_tokens)
        else:
            view_static(path, result_truncate_tokens=result_truncate_tokens)
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
