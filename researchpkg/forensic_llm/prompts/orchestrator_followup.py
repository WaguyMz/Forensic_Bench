"""Orchestrator follow-up hypothesis proposal (core_spec v2 dynamic queue)."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.models import (
    HypothesisResult,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)


def build_orchestrator_followup_prompt(
    last_result: HypothesisResult,
    global_excerpt: str,
    existing_task_ids: List[str],
    max_proposals: int = 2,
    text_limits: TextTruncationLimits | None = None,
) -> str:
    """Compact one-shot prompt — no orientation conversation history."""
    lim = text_limits or TextTruncationLimits()
    result_json = truncate_text_to_tokens(
        json.dumps(last_result.model_dump(mode="json"), indent=2, default=str),
        lim.orchestrator_followup_json,
        side=TruncationSide.TAIL,
    )
    memory_excerpt = truncate_text_to_tokens(
        global_excerpt or "",
        lim.orchestrator_followup_memory,
        side=TruncationSide.TAIL,
    )
    return (
        "## Orchestrator: propose follow-up hypothesis tasks\n\n"
        "Based on the **just-completed** hypothesis result and shared memory, "
        f"propose **0–{max_proposals}** new investigation tasks to run next. "
        "Each task must be a **single distinct investigation angle** for one scheme (free-form text, "
        "not ``If … then …`` boilerplate).\n\n"
        f"**Completed task:** `{last_result.task_id}`\n"
        f"```json\n{result_json}\n```\n\n"
        f"**Shared memory excerpt:**\n```\n{memory_excerpt}\n```\n\n"
        f"**Already queued or done (do not duplicate):** {', '.join(existing_task_ids[:40])}\n\n"
        "Respond with **ONLY** a JSON array (no markdown fences):\n"
        "[\n"
        '  {"scheme": "<scheme_type>", "hypothesis_text": "<concise claim>", '
        '"hypothesis_rationale": "<one paragraph: why now, what to test, confirm vs rule-out>", '
        '"dispatch_priority": <int lower runs sooner>}\n'
        "]\n\n"
        "Rules:\n"
        "- Prefer tracing entities, accounts, or JEs surfaced in `open_questions` or `key_findings`.\n"
        "- Use an existing scheme from the fraud catalogue; do not invent scheme names.\n"
        "- Empty array `[]` if no follow-up is justified.\n"
    )


def parse_followup_proposals(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM JSON array of follow-up proposals."""
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if len(lines) > 2 else lines).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict) and row.get("hypothesis_text"):
            out.append(row)
    return out
