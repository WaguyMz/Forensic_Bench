"""Run-directory artefact paths and JSON persistence (core_spec v2)."""
from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional

from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.models import (
    GlobalMemory,
    HypothesisResult,
    InvestigationPlan,
    RunStats,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)


def memory_dir(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "memory"


def global_memory_path(run_dir: pathlib.Path) -> pathlib.Path:
    return memory_dir(run_dir) / "global.json"


def orientation_summary_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "orientation" / "summary.json"


def orientation_report_path(run_dir: pathlib.Path) -> pathlib.Path:
    """Live orientation findings for planning (markdown, written during orientation)."""
    return run_dir / "orientation" / "orientation_report.md"


# Legacy: agent auto-appended stubs when SQL ran without orientation_report (removed from pipeline).
_ORIENTATION_AUTO_STUB_RE = re.compile(
    r"(?ms)^## Step \d+ — synthesize before continuing\n.*?(?=^## |\Z)",
)


def strip_orientation_auto_stub_sections(text: str) -> str:
    """Remove auto-generated 'synthesize before continuing' blocks (not for planning)."""
    body = _ORIENTATION_AUTO_STUB_RE.sub("", text or "")
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _flush_orientation_report_file(path: pathlib.Path) -> None:
    """Ensure orientation_report.md is persisted to disk (live for tail -f / next step)."""
    if not path.is_file():
        return
    try:
        with path.open("r+", encoding="utf-8") as fh:
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        try:
            data = path.read_bytes()
            path.write_bytes(data)
            with path.open("rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass


def orientation_report_stats(text: str) -> Dict[str, int]:
    """Lightweight stats for logging after each orientation step."""
    body = _orientation_report_body_without_boilerplate(text)
    sections = sum(1 for ln in body.splitlines() if ln.strip().startswith("##"))
    return {"chars": len(body), "sections": sections}


def sync_orientation_report_live(run_dir: pathlib.Path) -> Dict[str, int]:
    """
    Re-read ``orientation/orientation_report.md`` from disk after a step.

    Call at the end of every orientation step so the next turn's
    [Current Orientation Report] slot matches the file on disk.
    """
    path = orientation_report_path(run_dir)
    if path.is_file():
        _flush_orientation_report_file(path)
    text = load_orientation_report_text(run_dir)
    return orientation_report_stats(text)


def load_orientation_report_text(run_dir: pathlib.Path) -> str:
    """Return orientation report body for LLM slots and planning (no HTML boilerplate).

    Strips legacy auto-stub sections so planning never ingests SQL export excerpts
    that were only meant as ephemeral nudges.
    """
    path = orientation_report_path(run_dir)
    if not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").strip()
        raw = strip_orientation_auto_stub_sections(raw)
        return _orientation_report_body_without_boilerplate(raw)
    except OSError:
        return ""


ORIENTATION_REPORT_HEADER = (
    "# Orientation report\n\n"
    "<!--\n**Cumulative orientation memory** — append `##` sections as you learn.\n"
    "Planning reads this file directly (first ~100k tokens), NOT chat history.\n"
    "Write like investigator field notes: explanatory prose first, tables as evidence.\n"
    "Required: Vendor/AP, Revenue/COA, O2C×vendor, Planning leads; no TBD; no fraud vocabulary.\n"
    "Avoid table-only sections and raw 1000+ row SQL dumps.\n-->\n"
)

_ORIENTATION_BANNED_WORDS_RE = re.compile(
    r"\b(?:fraud|fraudulent|suspicious|anomal(?:y|ies)|red\s*flags?|"
    r"schemes?|manipulation|window\s+dressing|collusion|fictitious|implausible)\b",
    re.IGNORECASE,
)
_ORIENTATION_INCOMPLETE_MARKERS_RE = re.compile(
    r"\b(?:TBD|TODO)\b|need to verify|need to confirm|to be determined|"
    r"still unknown|unclear whether",
    re.IGNORECASE,
)


def _markdown_table_data_row_count(body: str) -> int:
    n = 0
    for line in (body or "").splitlines():
        s = line.strip()
        if not s.startswith("|") or s.count("|") < 3:
            continue
        if re.match(r"^\|[\s\-:|]+\|$", s):
            continue
        n += 1
    return n


def _orientation_narrative_prose_chars(body: str) -> int:
    """Non-table, non-heading text length — orientation should be prose-rich."""
    n = 0
    for line in (body or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("<!--"):
            continue
        if s in ("---", "***"):
            continue
        n += len(s) + 1
    return n


def _orientation_section_count(body: str) -> int:
    return len(re.findall(r"(?m)^##\s+\S", body or ""))


def orientation_completion_issues(text: str) -> list[str]:
    """
    Gaps that block ``complete_orientation`` until the report is planning-ready.
    """
    issues: list[str] = []
    body = _orientation_report_body_without_boilerplate(text)
    if len(body) < 400:
        issues.append(
            "Report is still too thin — append measured `##` sections via `orientation_report`."
        )
        return issues
    if _ORIENTATION_INCOMPLETE_MARKERS_RE.search(body):
        issues.append(
            "Resolve TBD/TODO/'need to verify' placeholders — run follow-up SQL and record results."
        )
    if _ORIENTATION_BANNED_WORDS_RE.search(body):
        issues.append(
            "Remove fraud/suspicion vocabulary; use neutral structural language until planning."
        )
    if not re.search(r"(?im)^##\s+Planning\s+[Ll]eads\b", body):
        issues.append(
            "Add final `## Planning leads` (neutral test intents only; no scheme conclusions)."
        )
    has_vendor_block = bool(
        re.search(r"(?im)^##\s+.*vendor", body)
        or (
            "vendor_id" in body.lower()
            and "auxiliary_gl" in body.lower()
            and _markdown_table_data_row_count(body) >= 5
        )
    )
    if not has_vendor_block:
        issues.append(
            "Add `## Vendor / AP master` with a full vendor census table (all vendors)."
        )
    has_revenue_block = bool(
        re.search(r"(?im)^##\s+.*revenue", body)
        or (re.search(r"(?im)^##\s+.*coa", body) and re.search(r"\b7\d{5}\b", body))
    )
    if not has_revenue_block:
        issues.append(
            "Add `## Revenue / COA` with COA reconciliation and revenue-by-account for the latest year."
        )
    has_o2c_cross = bool(
        re.search(r"(?im)^##\s+.*o2c", body)
        or (
            re.search(r"O2C", body, re.IGNORECASE)
            and re.search(r"411", body)
            and re.search(r"401", body)
        )
    )
    if not has_o2c_cross:
        issues.append(
            "Add `## O2C × vendor/aux` cross-tab (or document zero hits with the filter used)."
        )
    prose_chars = _orientation_narrative_prose_chars(body)
    if prose_chars < 2500:
        issues.append(
            "Add investigative prose — explain schema, linkages, process behavior, and baselines "
            f"(non-table narrative is only ~{prose_chars:,} chars; aim for rich field notes, not table dumps)."
        )
    if _orientation_section_count(body) < 6:
        issues.append(
            "Add more `##` sections with explanatory narrative (schema, process mix, actors, period-end, controls)."
        )
    table_rows = _markdown_table_data_row_count(body)
    if table_rows >= 30 and prose_chars < table_rows * 40:
        issues.append(
            "Report is table-heavy — add paragraphs under each section explaining what the data means "
            "for planning (tables support the narrative; they do not replace it)."
        )
    if not re.search(r"vendor_id", body, re.IGNORECASE) and not re.search(
        r"\bV-\d{6}\b", body
    ):
        issues.append(
            "Vendor census should list every `vendor_id` (table column or enumerated IDs)."
        )
    return issues


def planning_orientation_report_cap(limits: TextTruncationLimits | None = None) -> int:
    """Token cap for planning input from ``orientation/orientation_report.md``."""
    lim = limits or TextTruncationLimits()
    cap = int(
        lim.planning_orientation_prompt or lim.orientation_summary_store or 100_000
    )
    return max(0, cap)


def prepare_orientation_report_for_planning(
    run_dir: pathlib.Path,
    *,
    limits: TextTruncationLimits | None = None,
) -> tuple[str, str]:
    """
    Load the live orientation report for planning (no LLM digest).

    Returns ``(text, provenance_tag)``. When over cap, keeps the **first** tokens
    (HEAD) so early structural sections are preserved.
    """
    body = load_orientation_report_text(run_dir)
    if not body.strip():
        return "", "orientation_report_empty"
    cap = planning_orientation_report_cap(limits)
    if cap <= 0:
        return body.strip(), "orientation_report_direct"
    try:
        from researchpkg.forensic_llm.text_truncation import (
            count_tokens,
        )

        n_tok = count_tokens(body)
    except RuntimeError:
        n_tok = len(body) // 4
    if n_tok <= cap:
        return body.strip(), "orientation_report_direct"
    clipped = truncate_text_to_tokens(body, cap, side=TruncationSide.HEAD)
    return clipped.strip(), "orientation_report_head_truncated"


def ensure_orientation_report_file(run_dir: pathlib.Path) -> pathlib.Path:
    """Create ``orientation/orientation_report.md`` with the standard header if missing."""
    path = orientation_report_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        path.write_text(ORIENTATION_REPORT_HEADER + "\n", encoding="utf-8")
    return path


def _orientation_report_body_without_boilerplate(text: str) -> str:
    text = strip_orientation_auto_stub_sections(text or "")
    lines: list[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("<!--") and s.endswith("-->"):
            continue
        if s == "# Orientation report":
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def is_orientation_report_substantive(text: str, *, min_chars: int = 400) -> bool:
    """True when the report has real findings beyond the empty template header."""
    body = _orientation_report_body_without_boilerplate(text)
    return len(body) >= min_chars


def validate_orientation_section(text: str) -> dict[str, Any]:
    """Light advisory checks (no rigid Intent/Observations/Conclusion template)."""
    issues: list[str] = []
    body = (text or "").strip()
    has_section = bool(re.search(r"(?m)^##\s+\S", body))
    has_numbers = bool(
        re.search(
            r"\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:%|€|B|M|K|billion|million|thousand)?",
            body,
        )
    )
    has_sql_table = bool(re.search(r"^\|.*\|.*\|", body, re.MULTILINE))
    has_conversation_ref = bool(
        re.search(r"(?i)(as mentioned|see above|see below|as I said)", body)
    )
    banned = _ORIENTATION_BANNED_WORDS_RE.search(body)
    incomplete = _ORIENTATION_INCOMPLETE_MARKERS_RE.search(body)

    if len(body) < 40:
        issues.append("Section is very short — add a bit more measured context")
    if not has_section:
        issues.append("Prefer a `## Topic` heading so planning can scan the report")
    if not has_numbers:
        issues.append("Consider adding counts, %, or date ranges from your screening")
    if has_sql_table:
        issues.append("Avoid pasting raw SQL tables — summarize key numbers in prose")
    if has_conversation_ref:
        issues.append("Write self-contained prose (planning has no chat history)")
    if banned:
        issues.append(
            "Avoid fraud/suspicion vocabulary in orientation — use neutral structural wording"
        )
    if incomplete:
        issues.append(
            "Replace TBD/TODO/'need to verify' with SQL results before completing orientation"
        )
    prose_in_section = _orientation_narrative_prose_chars(body)
    table_rows = len(re.findall(r"(?m)^\|[^\n]+\|[^\n]+\|", body))
    if table_rows >= 3 and prose_in_section < 120:
        issues.append(
            "This section looks table-only — add 2+ sentences explaining what the numbers mean for the ledger"
        )
    elif prose_in_section < 80 and len(body) >= 80:
        issues.append(
            "Expand with investigative prose (how data links, what is normal here) — tables are optional support"
        )

    valid = len(body) >= 40 and not has_sql_table

    feedback_parts: list[str] = []
    if issues:
        feedback_parts.append("Suggestions (optional):")
        for issue in issues:
            feedback_parts.append(f"- {issue}")

    return {
        "valid": valid,
        "has_section": has_section,
        "has_numbers": has_numbers,
        "issues": issues,
        "feedback": "\n".join(feedback_parts)
        if feedback_parts
        else "Section recorded.",
    }


def append_orientation_report_text(
    run_dir: pathlib.Path,
    text: str,
    *,
    max_chars: int = 120_000,
) -> int:
    """
    Append markdown to the live orientation report (flushed immediately).

    Returns the byte size of the file after write.
    """
    chunk = (text or "").strip()
    if not chunk:
        return 0
    if len(chunk) > max_chars:
        chunk = (
            chunk[:max_chars]
            + "\n\n[truncated: single chunk exceeded server character limit]"
        )
    path = ensure_orientation_report_file(run_dir)
    prev = path.read_text(encoding="utf-8")
    sep = "\n\n" if _orientation_report_body_without_boilerplate(prev) else ""
    path.write_text(prev.rstrip() + sep + "\n" + chunk + "\n", encoding="utf-8")
    _flush_orientation_report_file(path)
    try:
        return path.stat().st_size
    except OSError:
        return len(chunk)


def replace_orientation_report_text(
    run_dir: pathlib.Path,
    text: str,
    *,
    max_chars: int = 120_000,
) -> int:
    """Replace the orientation report body (keeps standard header comment block)."""
    body = (text or "").strip()
    if len(body) > max_chars:
        body = (
            body[:max_chars]
            + "\n\n[truncated: single chunk exceeded server character limit]"
        )
    path = ensure_orientation_report_file(run_dir)
    path.write_text(ORIENTATION_REPORT_HEADER + "\n" + body + "\n", encoding="utf-8")
    _flush_orientation_report_file(path)
    try:
        return path.stat().st_size
    except OSError:
        return len(body)


def scheme_dir(run_dir: pathlib.Path, scheme: str) -> pathlib.Path:
    return run_dir / "schemes" / scheme


def hypothesis_result_path(
    run_dir: pathlib.Path, scheme: str, hypothesis_id: str
) -> pathlib.Path:
    hid = hypothesis_id.lower()
    if not hid.startswith("p"):
        hid = f"p{hid}"
    return scheme_dir(run_dir, scheme) / f"{hid}.json"


def task_dir(run_dir: pathlib.Path, task_id: str) -> pathlib.Path:
    return run_dir / "tasks" / task_id


def investigation_report_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "investigation_report.json"


def run_stats_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "run_stats.json"


def report_md_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "report.md"


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_global_memory(run_dir: pathlib.Path) -> GlobalMemory:
    raw = read_json(global_memory_path(run_dir))
    if raw:
        return GlobalMemory.model_validate(raw)
    return GlobalMemory()


def save_global_memory(run_dir: pathlib.Path, memory: GlobalMemory) -> None:
    from datetime import datetime

    memory.updated_at = datetime.utcnow().isoformat()
    write_json(global_memory_path(run_dir), memory)


def save_hypothesis_result(
    run_dir: pathlib.Path, result: HypothesisResult
) -> pathlib.Path:
    path = hypothesis_result_path(run_dir, result.scheme, result.hypothesis_id)
    write_json(path, result)
    return path


def load_hypothesis_result(
    run_dir: pathlib.Path, scheme: str, hypothesis_id: str
) -> Optional[HypothesisResult]:
    path = hypothesis_result_path(run_dir, scheme, hypothesis_id)
    raw = read_json(path)
    if raw:
        return HypothesisResult.model_validate(raw)
    return None


def load_all_hypothesis_results(run_dir: pathlib.Path) -> List[HypothesisResult]:
    results: List[HypothesisResult] = []
    schemes_root = run_dir / "schemes"
    if not schemes_root.is_dir():
        return results
    for scheme_path in sorted(schemes_root.iterdir()):
        if not scheme_path.is_dir():
            continue
        for pfile in sorted(scheme_path.glob("p*.json")):
            raw = read_json(pfile)
            if raw:
                results.append(HypothesisResult.model_validate(raw))
    return results


def save_run_stats(run_dir: pathlib.Path, stats: RunStats) -> None:
    write_json(run_stats_path(run_dir), stats)


def save_investigation_report(run_dir: pathlib.Path, payload: Dict[str, Any]) -> None:
    write_json(investigation_report_path(run_dir), payload)


def persist_plan_with_queue(run_dir: pathlib.Path, plan: InvestigationPlan) -> None:
    write_json(run_dir / "investigation_plan.json", plan)


def persist_dispatch_queue_snapshot(
    run_dir: pathlib.Path, items: List[Dict[str, Any]]
) -> None:
    """Live dispatch queue state (status/priority) during a run."""
    write_json(run_dir / "dispatch_queue.json", {"items": items})


def write_orientation_run_metadata(
    run_dir: pathlib.Path,
    steps: int,
    *,
    limits: TextTruncationLimits | None = None,
    orientation_tokens: int | None = None,
    planning_source: str | None = None,
    planning_report_tokens: int | None = None,
) -> None:
    """
    Persist orientation run metadata only (planning uses ``orientation_report.md``).

    ``summary.json`` is no longer a compressed digest; it records steps/tokens/caps
    for dashboards and backward-compatible tooling.
    """
    planning_cap = planning_orientation_report_cap(limits)
    payload: Dict[str, Any] = {
        "planning_input": "orientation_report.md",
        "planning_source": planning_source or "orientation_report_direct",
        "orientation_report_relpath": "orientation/orientation_report.md",
        "planning_report_token_cap": planning_cap,
        "steps": steps,
    }
    if orientation_tokens is not None:
        payload["orientation_tokens"] = orientation_tokens
    if planning_report_tokens is not None:
        payload["planning_report_tokens"] = planning_report_tokens
    write_json(orientation_summary_path(run_dir), payload)


def write_orientation_summary(
    run_dir: pathlib.Path,
    text: str,
    steps: int,
    *,
    limits: TextTruncationLimits | None = None,
    orientation_tokens: int | None = None,
    summary_text: str | None = None,
    summary_source: str | None = None,
    summary_tokens: int | None = None,
    summary_planning_cap: int | None = None,
) -> None:
    """Backward-compatible wrapper — writes metadata only (no digest in ``summary``)."""
    _ = text, summary_text, summary_source, summary_tokens, summary_planning_cap
    write_orientation_run_metadata(
        run_dir,
        steps,
        limits=limits,
        orientation_tokens=orientation_tokens,
        planning_source=summary_source,
        planning_report_tokens=summary_tokens,
    )


def load_orientation_summary(run_dir: pathlib.Path) -> str:
    """Return orientation report text for planning (raw report, not legacy digest)."""
    text, _ = prepare_orientation_report_for_planning(run_dir)
    if text.strip():
        return text
    path = orientation_summary_path(run_dir)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    summary = data.get("summary") if isinstance(data, dict) else ""
    return summary.strip() if isinstance(summary, str) else ""
