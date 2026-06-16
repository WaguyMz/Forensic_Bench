from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Dict, List, Optional

from .context import get_scratchpad_live_path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local scratchpad state
#
# Each thread (e.g. a parallel scheme worker) has its own isolated scratchpad
# so concurrent workers cannot corrupt each other's notes.  The helper
# functions below transparently route reads and writes to the calling thread's
# local storage.
# ---------------------------------------------------------------------------

_LOCAL = threading.local()


def _notes() -> List[str]:
    if not hasattr(_LOCAL, "notes"):
        _LOCAL.notes = []
    return _LOCAL.notes


def _structured() -> Optional[str]:
    return getattr(_LOCAL, "structured", None)


def _run_context() -> Optional[Dict[str, Any]]:
    return getattr(_LOCAL, "run_context", None)


def set_scratchpad_run_context(
    step: int,
    phase: str = "",
    sql_calls: int = 0,
    **extra: Any,
) -> None:
    ctx: Dict[str, Any] = {"step": step, "phase": phase, "sql_calls": sql_calls}
    ctx.update(extra)
    _LOCAL.run_context = ctx


def _flush_scratchpad_live() -> None:
    path = get_scratchpad_live_path()
    if not path:
        return
    try:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(get_scratchpad_text(), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write live scratchpad to %s: %s", path, exc)


def scratchpad(note: str = "", mode: str = "append", clear_notes: bool = False) -> str:
    """
    Manage the investigation scratchpad (thread-local).

    ``replace`` updates only the structured ``## Investigation Scratchpad`` block.
    Appended notes (manual reflections and auto-stubs) are preserved unless
    ``clear_notes=True`` (discouraged during orientation).
    """
    if mode == "read":
        return get_scratchpad_text()
    if mode == "replace":
        if clear_notes:
            _notes().clear()
        _LOCAL.structured = (note or "").strip()
        _flush_scratchpad_live()
        n_notes = len(_notes())
        if clear_notes:
            return (
                f"[SCRATCHPAD] Structured block replaced ({len(note or '')} chars); "
                "appended notes cleared."
            )
        return (
            f"[SCRATCHPAD] Structured block replaced ({len(note or '')} chars); "
            f"{n_notes} appended note(s) preserved."
        )

    _notes().append((note or "").strip())
    _flush_scratchpad_live()
    return f"[SCRATCHPAD] Note recorded. Total notes: {len(_notes())}."


def get_scratchpad_text() -> str:
    ctx = _run_context()
    ctx_block = ""
    if ctx:
        lines = [
            "## Run context",
            "",
            f"- **Step:** {ctx.get('step', '?')}",
            f"- **Phase:** {ctx.get('phase') or '(none)'}",
            f"- **SQL calls so far:** {ctx.get('sql_calls', 0)}",
        ]
        if ctx.get("phase") == "orientation":
            used = ctx.get("orientation_tokens_used")
            cap = ctx.get("orientation_tokens_cap")
            if used is not None and cap is not None:
                cap_i = int(cap)
                used_i = int(used)
                pct = (100.0 * used_i / cap_i) if cap_i > 0 else 0.0
                lines.append(
                    f"- **Orientation budget:** **{pct:.1f}%** of cap "
                    f"({used_i:,} / {cap_i:,} input tokens; **{max(0, cap_i - used_i):,} remaining**)"
                )
            pin = ctx.get("orientation_planning_input_cap")
            if pin is not None:
                lines.append(
                    f"- **Planning orientation cap:** {int(pin):,} tokens "
                    "(from `orientation/orientation_report.md` after fit; model tokenizer)"
                )
            enc_until = ctx.get("orientation_encourage_deep_until_fraction")
            min_frac = ctx.get("orientation_min_fraction_for_complete")
            if (
                used is not None
                and cap is not None
                and enc_until is not None
                and min_frac is not None
            ):
                cap_i2 = int(cap)
                used_i2 = int(used)
                pct2 = (100.0 * used_i2 / cap_i2) if cap_i2 > 0 else 0.0
                eu_pct = float(enc_until) * 100.0
                mc_pct = float(min_frac) * 100.0
                min_tok = int(float(min_frac) * cap_i2)
                if pct2 + 1e-6 < float(enc_until) * 100.0:
                    lines.append(
                        f"- **Phase pacing:** **{pct2:.1f}%** used — you are **encouraged to go deeper** "
                        f"(SQL + dense `orientation_report` ## sections). Aim to use most of the cap "
                        f"through ~**{eu_pct:.0f}%** before wrapping up."
                    )
                else:
                    lines.append(
                        f"- **Phase pacing:** **{pct2:.1f}%** used — finish any missing checklist coverage, "
                        f"then call `complete_orientation` only when the report is decision-ready."
                    )
                if pct2 + 1e-6 < mc_pct:
                    lines.append(
                        f"- **Early completion guard:** `complete_orientation` is blocked below **~{mc_pct:.0f}%** "
                        f"usage (about **{min_tok:,}** / {cap_i2:,} input tokens)."
                    )
        t_used = ctx.get("task_tokens_used")
        t_cap = ctx.get("task_tokens_cap")
        if t_used is not None and t_cap is not None:
            tc = int(t_cap)
            tu = int(t_used)
            pct = (100.0 * tu / tc) if tc > 0 else 0.0
            rem = max(0, tc - tu)
            lines.append(
                f"- **Task input budget:** **{pct:.1f}%** used "
                f"({tu:,} / {tc:,} input tokens; **{rem:,} remaining**)"
            )
            tw = ctx.get("task_budget_warn_fraction")
            td = ctx.get("task_budget_report_deadline_fraction")
            ts = ctx.get("task_budget_stop_fraction")
            if tw is not None and td is not None and ts is not None:
                lines.append(
                    f"- **Pacing:** call `report_suspicion` before ~{float(td) * 100.0:.0f}% "
                    f"if evidence is clear; hard stop ~{float(ts) * 100.0:.0f}% of this task cap"
                )
        lines.extend(["", "---", ""])
        ctx_block = "\n".join(lines)
    notes = _notes()
    structured = _structured()
    if structured:
        parts = [ctx_block + "## Investigation Scratchpad\n\n" + structured]
        if notes:
            parts.append(
                "\n\n---\n\n### Appended Notes\n\n"
                + "\n\n---\n\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
            )
        return "".join(parts)
    if not notes:
        return ctx_block + "(scratchpad is empty)"
    return (
        ctx_block
        + "## Investigation Scratchpad\n\n"
        + "\n\n---\n\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
    )


def reset_scratchpad() -> None:
    _notes().clear()
    _LOCAL.structured = None
    _LOCAL.run_context = None
