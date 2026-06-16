"""Promote hypothesis worker outputs into global shared memory."""
from __future__ import annotations

import pathlib
from typing import List, Optional

from researchpkg.forensic_llm.artefacts import (
    load_global_memory,
    save_global_memory,
)
from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.models import (
    GlobalMemory,
    HypothesisMemoryEntry,
    HypothesisResult,
    SharedEvidenceBlackboard,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)

_MAX_FLAG_IDS_SAMPLE = 200
_MAX_FLAG_ENTITIES_SAMPLE = 80
_MAX_EVIDENCE_CHECKS_STORE = 40
_MAX_SUMMARY_ENTRIES = 80


def promote_hypothesis_result(
    memory: GlobalMemory,
    result: HypothesisResult,
    blackboard: SharedEvidenceBlackboard,
    *,
    limits: Optional[TextTruncationLimits] = None,
) -> GlobalMemory:
    """Merge one completed hypothesis task into global memory."""
    lim = limits or TextTruncationLimits()
    pointer = {
        "scheme": result.scheme,
        "hypothesis_id": result.hypothesis_id,
        "task_id": result.task_id,
        "path": f"schemes/{result.scheme}/{result.hypothesis_id.lower()}.json",
        "status": result.status,
    }
    if pointer not in memory.hypothesis_pointers:
        memory.hypothesis_pointers.append(pointer)

    tid = (result.task_id or "").strip() or f"{result.scheme}_{result.hypothesis_id}"
    flagged = list(result.flagged_document_ids or [])
    entities = list(result.flagged_entities or [])
    summary = HypothesisMemoryEntry(
        scheme=result.scheme,
        hypothesis_id=result.hypothesis_id,
        task_id=tid,
        status=result.status,
        hypothesis_text=truncate_text_to_tokens(
            result.hypothesis_text or "",
            lim.hypothesis_text_store,
            side=TruncationSide.TAIL,
        ),
        hypothesis_rationale=truncate_text_to_tokens(
            result.hypothesis_rationale or "",
            lim.hypothesis_rationale_store,
            side=TruncationSide.TAIL,
        ),
        key_findings=truncate_text_to_tokens(
            result.key_findings or "",
            lim.key_findings_store,
            side=TruncationSide.TAIL,
        ),
        flagged_document_ids=flagged[:_MAX_FLAG_IDS_SAMPLE],
        total_flagged_documents=len(flagged),
        flagged_entities=entities[:_MAX_FLAG_ENTITIES_SAMPLE],
        evidence_checks_run=list(
            (result.evidence_checks_run or [])[:_MAX_EVIDENCE_CHECKS_STORE]
        ),
    )
    kept = [s for s in memory.hypothesis_summaries if s.task_id != tid]
    kept.append(summary)
    memory.hypothesis_summaries = kept[-_MAX_SUMMARY_ENTRIES:]

    if result.key_findings:
        salient_body = truncate_text_to_tokens(
            result.key_findings or "",
            lim.salient_finding,
            side=TruncationSide.TAIL,
        )
        salient = f"[{result.scheme}/{result.hypothesis_id}] {salient_body}"
        memory.salient_findings.append(salient)
        memory.salient_findings = memory.salient_findings[-40:]

    for q in result.open_questions or []:
        if q and q not in memory.open_risks:
            memory.open_risks.append(q)
    memory.open_risks = memory.open_risks[-30:]

    if result.status == "confirmed":
        memory.scheme_verdicts[result.scheme] = "strong_evidence"
    elif result.scheme not in memory.scheme_verdicts:
        if result.status == "falsified":
            memory.scheme_verdicts[result.scheme] = "no_material_evidence"
        else:
            memory.scheme_verdicts[result.scheme] = "inconclusive"

    cross = blackboard.find_cross_scheme_entities()
    memory.cross_scheme_entities = cross[:25]

    return memory


def flush_global_memory(
    run_dir: pathlib.Path,
    result: HypothesisResult,
    blackboard: SharedEvidenceBlackboard,
    *,
    limits: Optional[TextTruncationLimits] = None,
) -> GlobalMemory:
    memory = load_global_memory(run_dir)
    memory = promote_hypothesis_result(memory, result, blackboard, limits=limits)
    save_global_memory(run_dir, memory)
    return memory


def render_global_memory_excerpt(
    memory: GlobalMemory,
    scheme: str,
    *,
    limits: Optional[TextTruncationLimits] = None,
) -> str:
    """Compact global memory for orchestrator follow-up prompts."""
    lim = limits or TextTruncationLimits()
    summ_lines = []
    for s in memory.hypothesis_summaries[-6:]:
        if s.scheme != scheme:
            continue
        clip = truncate_text_to_tokens(
            s.key_findings or "",
            lim.memory_excerpt_clip,
            side=TruncationSide.TAIL,
        )
        summ_lines.append(f"- **{s.hypothesis_id}** ({s.status}): {clip}")
    recent = "\n".join(summ_lines) if summ_lines else "(none for this scheme)"

    lines = [
        "## Hypothesis summaries (this scheme)",
        recent,
        "\n## Salient findings",
        "\n".join(memory.salient_findings[-8:]) or "(none)",
        "\n## Open risks",
        "\n".join(memory.open_risks[-8:]) or "(none)",
        f"\n## Verdict for {scheme}",
        memory.scheme_verdicts.get(scheme, "unknown"),
    ]
    if memory.cross_scheme_entities:
        brief = []
        for ent in memory.cross_scheme_entities[:12]:
            if isinstance(ent, dict):
                eid = ent.get("entity_id") or ent.get("id") or ent
                schemes = ent.get("schemes")
                if schemes:
                    brief.append(f"{eid} ({','.join(map(str, schemes[:3]))})")
                else:
                    brief.append(str(eid))
            else:
                brief.append(str(ent))
        lines.extend(["\n## Cross-scheme entities", "; ".join(brief)])
    blob = "\n".join(lines)
    return truncate_text_to_tokens(
        blob,
        lim.global_memory_excerpt,
        side=TruncationSide.TAIL,
    )


def build_shared_context_excerpt(
    run_dir: pathlib.Path,
    scheme: str,
    blackboard: SharedEvidenceBlackboard,
    peer_results: List[HypothesisResult],
    orientation_excerpt: str = "",
    *,
    limits: Optional[TextTruncationLimits] = None,
) -> dict:
    """Tier-A context injected at hypothesis worker spawn."""
    lim = limits or TextTruncationLimits()
    memory = load_global_memory(run_dir)
    summ_bits = []
    for s in memory.hypothesis_summaries[-12:]:
        if s.scheme != scheme:
            continue
        summ_bits.append(
            f"[{s.hypothesis_id} {s.status}] "
            + truncate_text_to_tokens(
                s.key_findings or "",
                lim.shared_context_key_finding,
                side=TruncationSide.TAIL,
            )
        )
    summary_block = "\n".join(summ_bits) if summ_bits else "(none yet for this scheme)"
    parts = [
        "## Global memory (hypothesis detail)\n",
        summary_block,
        "\n\n## Global memory (salient lines)\n",
        "\n".join(memory.salient_findings[-8:]) or "(none yet)",
        "\n\n## Blackboard\n",
        truncate_text_to_tokens(
            blackboard.to_context_block(),
            lim.shared_context_blackboard,
            side=TruncationSide.TAIL,
        ),
    ]
    orient_snip = ""
    if orientation_excerpt.strip():
        orient_snip = truncate_text_to_tokens(
            orientation_excerpt.strip(),
            lim.shared_context_orientation,
            side=TruncationSide.TAIL,
        )
        parts.extend(["\n\n## Orientation\n", orient_snip])
    if peer_results:
        lines = ["\n\n## Peer hypotheses (same scheme)\n"]
        for pr in peer_results:
            kf = truncate_text_to_tokens(
                pr.key_findings or "",
                lim.shared_context_peer_finding,
                side=TruncationSide.TAIL,
            )
            lines.append(f"- {pr.hypothesis_id}: {pr.status} — {kf}")
        parts.append("\n".join(lines))
    blob = "".join(parts)
    blob = truncate_text_to_tokens(
        blob,
        lim.shared_context_blob,
        side=TruncationSide.TAIL,
    )
    return {
        "global_memory_excerpt": blob,
        "orientation_excerpt": orient_snip,
        "blackboard_excerpt": truncate_text_to_tokens(
            blackboard.to_context_block(),
            lim.shared_context_blackboard,
            side=TruncationSide.TAIL,
        ),
        "peer_hypothesis_summaries": [
            {
                "hypothesis_id": pr.hypothesis_id,
                "status": pr.status,
                "key_finding": truncate_text_to_tokens(
                    pr.key_findings or "",
                    lim.shared_context_peer_finding,
                    side=TruncationSide.TAIL,
                ),
            }
            for pr in peer_results
        ],
    }
