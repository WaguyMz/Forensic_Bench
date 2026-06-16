"""Orientation scratchpad helpers and budget pacing."""
from __future__ import annotations

import pathlib
import re

from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.model_tokenizer import (
    count_tokens,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)

# Legacy markers (optional in scratchpad; synthesis uses full body, not handoff-only).
_PLANNING_HANDOFF_RE = re.compile(
    r"(?im)^#{1,3}\s*("
    r"planning\s+handoff"
    r"|orientation\s+(?:summary|synthesis|planning\s+memo)"
    r"|summary\s+for\s+planning"
    r")\s*$"
)

_RUN_CONTEXT_RE = re.compile(
    r"(?im)^##\s+Run context\s*$.*?^---\s*$",
    re.MULTILINE | re.DOTALL,
)

_INVESTIGATION_SCRATCHPAD_RE = re.compile(
    r"(?im)^##\s+Investigation Scratchpad\s*\n+"
)

# If live scratchpad is thinner than this, synthesis may merge step snapshots.
ORIENTATION_SYNTHESIS_SCRATCHPAD_MIN_TOKENS = 12_000


def strip_scratchpad_run_context(scratchpad: str) -> str:
    """Remove ephemeral run-context header from scratchpad text."""
    text = (scratchpad or "").strip()
    if not text:
        return ""
    text = _RUN_CONTEXT_RE.sub("", text, count=1).strip()
    text = _INVESTIGATION_SCRATCHPAD_RE.sub("", text, count=1).strip()
    return text


def prepare_orientation_synthesis_input(
    scratchpad: str,
    *,
    limits: TextTruncationLimits | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Scratchpad text fed to orientation summary synthesis (model tokenizer).

    Strips run-context chrome and truncates from the tail so the most recent
    orientation work is preserved up to ``max_tokens`` (default: ``orientation_memo_synthesis_input``).
    """
    lim = limits or TextTruncationLimits()
    cap = max_tokens if max_tokens is not None else lim.orientation_memo_synthesis_input
    body = strip_scratchpad_run_context(scratchpad)
    if not body or cap <= 0:
        return body
    return truncate_text_to_tokens(body, cap, side=TruncationSide.TAIL)


def _sql_export_manifest(run_dir: pathlib.Path, *, max_entries: int = 80) -> str:
    """List SQL export CSV paths under the run (for planning / synthesis)."""
    exports_dir = run_dir / "plots" / "sql_outputs"
    if not exports_dir.is_dir():
        return ""
    rows: list[str] = []
    files = sorted(
        exports_dir.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files[:max_entries]:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rel = path.relative_to(run_dir)
        rows.append(f"- `{rel}` ({size:,} bytes)")
    if len(files) > max_entries:
        rows.append(f"- … and {len(files) - max_entries} more export(s)")
    return "\n".join(rows)


def _richest_scratchpad_step_snapshot(run_dir: pathlib.Path) -> str:
    """Return the largest step snapshot body (safety net if live pad was wiped)."""
    steps_dir = run_dir / "scratchpad_steps"
    if not steps_dir.is_dir():
        return ""
    best = ""
    best_tokens = 0
    for path in sorted(steps_dir.glob("step_*.md")):
        try:
            text = strip_scratchpad_run_context(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        n = _token_len(text)
        if n > best_tokens:
            best_tokens = n
            best = text
    return best


def build_orientation_synthesis_bundle(
    scratchpad: str,
    *,
    run_dir: pathlib.Path | None = None,
    limits: TextTruncationLimits | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Merge scratchpad, SQL export manifest, and (if thin) richest step snapshot
    for orientation summary synthesis.
    """
    lim = limits or TextTruncationLimits()
    cap = max_tokens if max_tokens is not None else lim.orientation_memo_synthesis_input
    body = strip_scratchpad_run_context(scratchpad)
    parts: list[str] = []

    if body.strip():
        parts.append("## Orientation scratchpad (model notes)\n\n" + body.strip())

    if run_dir is not None:
        manifest = _sql_export_manifest(run_dir)
        if manifest:
            parts.append(
                "## SQL export manifest (worker paths)\n\n"
                "Use these paths for follow-up analysis; facts must still appear in notes.\n\n"
                + manifest
            )
        body_tokens = _token_len(body)
        if body_tokens < ORIENTATION_SYNTHESIS_SCRATCHPAD_MIN_TOKENS:
            recovered = _richest_scratchpad_step_snapshot(run_dir)
            if recovered and _token_len(recovered) > body_tokens:
                parts.append(
                    "## Recovered orientation work log (step snapshot)\n\n"
                    "The live scratchpad was thin; this is the richest saved step file.\n\n"
                    + recovered
                )

    if not parts:
        return ""
    combined = "\n\n---\n\n".join(parts)
    return prepare_orientation_synthesis_input(combined, limits=lim, max_tokens=cap)


def _token_len(text: str) -> int:
    if not text:
        return 0
    try:
        return count_tokens(text)
    except RuntimeError:
        return max(1, len(text) // 4)


def orientation_budget_pct(tokens_used: int, tokens_cap: int) -> float:
    """Share of orientation cap consumed (0–100+), using budget-tracker token counts."""
    if tokens_cap <= 0:
        return 0.0
    return 100.0 * float(tokens_used) / float(tokens_cap)


def format_orientation_budget_block(
    *,
    tokens_used: int,
    tokens_cap: int,
    planning_input_cap_tokens: int,
    encourage_deep_until_fraction: float = 0.72,
    min_fraction_for_complete: float = 0.30,
) -> str:
    """Orientation budget snapshot for the orchestrator (informational only)."""
    pct = orientation_budget_pct(tokens_used, tokens_cap)
    remaining = max(0, tokens_cap - tokens_used)
    ed = max(0.0, min(1.0, float(encourage_deep_until_fraction)))
    mc = max(0.0, min(1.0, float(min_fraction_for_complete)))
    return (
        "**Orientation budget (informational — you pace the phase)**\n"
        f"- **{pct:.1f}%** of orientation cap "
        f"({tokens_used:,} / {tokens_cap:,} input tokens; {remaining:,} remaining)\n"
        f"- **Depth target:** use most of this cap through ~**{ed * 100.0:.0f}%** for dense baselines "
        f"(SQL + interpreted `orientation_report` sections).\n"
        f"- **Early exit guard:** `complete_orientation` is rejected below ~**{mc * 100.0:.0f}%** "
        f"cap usage (~{int(mc * max(1, tokens_cap)):,} input tokens) except under global budget stop.\n"
        f"- **Prefix each call:** System → [Current Orientation Report] → [Current Step]\n"
        f"- **Report sections:** Intent · Observations · Conclusion · Next (no raw SQL tables)\n"
        f"- Planning receives the **first {planning_input_cap_tokens:,} tokens** of that report "
        "(model tokenizer; no LLM digest). If the report is longer, the **opening** sections "
        "are kept — put the highest-value baselines and entity-specific findings **early**. "
        "**Planning has NO conversation history** — write self-contained sections."
    )
