"""Prompts for per-hypothesis investigation workers (core_spec v2)."""
from __future__ import annotations

import json

from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.models import (
    HypothesisTaskBrief,
    WorkerBrief,
)
from researchpkg.forensic_llm.prompts.investigation_discipline import (
    COMPREHENSIVE_INVESTIGATION_GUIDANCE,
    HYPOTHESIS_INVESTIGATION_DEPTH,
    REPORT_SUSPICION_DISCIPLINE,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)


def build_hypothesis_worker_prompt(
    brief: HypothesisTaskBrief,
    text_limits: TextTruncationLimits | None = None,
) -> str:
    lim = text_limits or TextTruncationLimits()
    shared = brief.shared_context or {}
    shared_excerpt = truncate_text_to_tokens(
        shared.get("global_memory_excerpt", "") or "",
        lim.hypothesis_worker_shared_memory,
        side=TruncationSide.TAIL,
    )
    peer_block = ""
    peers = shared.get("peer_hypothesis_summaries") or []
    if peers:
        peer_block = "\n**Peer hypotheses (same scheme):**\n" + "\n".join(
            f"- {p.get('hypothesis_id')}: {p.get('status')} — {p.get('key_finding', '')}"
            for p in peers
        )

    return (
        f"## Hypothesis investigation task: `{brief.task_id}`\n\n"
        f"- **Scheme:** `{brief.scheme}`\n"
        f"- **Hypothesis ID:** `{brief.hypothesis_id}`\n"
        f"- **Token budget:** {brief.budget_tokens:,} (the planner allocated this envelope for **your** hypothesis — treat it as a target to work toward through SQL + reasoning, not a ceiling to leave mostly unused)\n\n"
        f"### Hypothesis\n{brief.hypothesis_text}\n\n"
        + (
            f"### Why this task (planner rationale)\n{brief.hypothesis_rationale}\n\n"
            if (brief.hypothesis_rationale or "").strip()
            else ""
        )
        + f"### Exit criteria\n"
        + "\n".join(f"- {c}" for c in brief.exit_criteria)
        + "\n\n"
        f"### Benign rival explanations to falsify\n"
        + "\n".join(f"- {b}" for b in brief.benign_rivals)
        + "\n\n"
        f"### Planned query families\n"
        + "\n".join(f"- {q}" for q in brief.planned_query_families)
        + "\n\n"
        f"### Shared context\n```\n{shared_excerpt}\n```\n" + peer_block + "\n\n"
        "### Operating rules\n"
        "1. Investigate **only** this hypothesis — but investigate it **thoroughly** (screen → drill → "
        "falsify benign rivals → follow material leads). **Spend most of your allocated token budget** "
        "unless you reach a decisive verdict very early with dense SQL evidence.\n"
        "2. Use the **planned query families** unless evidence justifies a documented deviation "
        "(note in scratchpad).\n"
        "3. **`report_suspicion`** whenever you have SQL-backed suspicious `je_header.document_id` UUIDs "
        "— during investigation, not only in the summary.\n"
        "4. Use `blackboard_write` for entity findings relevant to other schemes.\n"
        "5. Do **not** call `finish_investigation`.\n"
        "6. Stop tool use only when the verdict is evidence-backed — a summary step follows.\n"
        "7. **Budget:** the scratchpad *Run context* shows **Task input budget** (% used / "
        "remaining). If SQL-backed evidence confirms fraud, call `report_suspicion` *before* "
        "you exhaust that cap—do not plan heavy exports when the % is already high.\n\n"
        + HYPOTHESIS_INVESTIGATION_DEPTH
        + "\n"
        + COMPREHENSIVE_INVESTIGATION_GUIDANCE
        + "\n"
        + REPORT_SUSPICION_DISCIPLINE
    )


def build_hypothesis_summary_prompt(
    brief: HypothesisTaskBrief,
    text_limits: TextTruncationLimits | None = None,
) -> str:
    lim = text_limits or TextTruncationLimits()
    hyp_text = truncate_text_to_tokens(
        brief.hypothesis_text or "",
        lim.hypothesis_worker_hypothesis_text,
        side=TruncationSide.HEAD,
    ).replace('"', "'")
    hyp_rat = truncate_text_to_tokens(
        brief.hypothesis_rationale or "",
        lim.hypothesis_rationale_store,
        side=TruncationSide.TAIL,
    ).replace('"', "'")
    return (
        f"Produce the structured JSON summary for hypothesis `{brief.hypothesis_id}` "
        f"on scheme `{brief.scheme}`. Respond with ONLY the JSON object.\n\n"
        "**`flagged_document_ids`:** must match every UUID you reported via `report_suspicion` "
        "during this task (same IDs from SQL). If status is `confirmed` and you have exemplar JEs, "
        "this list must be non-empty. Use `[]` only when status is `falsified`.\n\n"
        "```json\n"
        "{\n"
        f'  "scheme": "{brief.scheme}",\n'
        f'  "hypothesis_id": "{brief.hypothesis_id}",\n'
        f'  "hypothesis_text": "{hyp_text}",\n'
        f'  "hypothesis_rationale": "{hyp_rat}",\n'
        '  "status": "<confirmed|falsified|inconclusive>",\n'
        '  "finish_reason": "<completed|task_budget_stop_frac|task_budget_exhausted|worker_error>",\n'
        '  "flagged_document_ids": ["<uuid>", "..."],\n'
        '  "flagged_entities": ["<entity_id>", "..."],\n'
        '  "key_findings": "<2-4 sentences: what you tested, what data showed, verdict rationale>",\n'
        '  "evidence_checks_run": ["<screen>", "<drill>", "<benign rival>", "..."],\n'
        '  "benign_rivals_considered": ["..."],\n'
        '  "open_questions": ["..."],\n'
        '  "error": null\n'
        "}\n"
        "```"
    )


def build_worker_brief_prompt(brief: WorkerBrief) -> str:
    """Legacy autonomous worker brief (deprecated path)."""
    candidate_schemes = (
        ", ".join(brief.candidate_schemes) if brief.candidate_schemes else "unspecified"
    )
    return (
        f"## Worker Brief: {brief.worker_id}\n\n"
        f"- Goal: `{brief.scheme_or_goal}`\n"
        f"- Candidate schemes: {candidate_schemes}\n"
        f"- Budget tokens: {brief.budget_tokens:,}\n\n"
        f"{brief.brief}\n\n"
        "Do not call `finish_investigation`. Use `report_suspicion` for confirmed JEs."
    )


def build_worker_summary_prompt(brief: WorkerBrief) -> str:
    """Legacy autonomous worker summary (deprecated path)."""
    return (
        f"Summarize worker `{brief.worker_id}` as JSON only.\n"
        '{"worker_id": "...", "scheme_or_goal": "...", "status": "completed", '
        '"flagged_document_ids": ["<uuid>"], "key_findings": "..."}'
    )


def parse_hypothesis_summary_json(text: str, brief: HypothesisTaskBrief) -> dict:
    from researchpkg.forensic_llm.json_utils import (
        extract_first_json_object,
    )

    obj = extract_first_json_object(text or "") or {}
    obj.setdefault("scheme", brief.scheme)
    obj.setdefault("hypothesis_id", brief.hypothesis_id)
    obj.setdefault("hypothesis_text", brief.hypothesis_text)
    obj.setdefault("hypothesis_rationale", brief.hypothesis_rationale)
    obj.setdefault("status", "inconclusive")
    if "finish_reason" not in obj:
        obj["finish_reason"] = None
    return obj
