"""
Pydantic v2 data models for the forensic investigator.

These cover:
- Tool call records (input + result)
- Suspicion items (the agent's findings)
- Investigation plans and scheme reports (Plan-Execute-Reflect architecture)
- Full investigation state / trace
- Offline evaluation results (document-level and scheme-level)
"""
from __future__ import annotations

import math
import threading
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Scheme taxonomy (mirrors the generator's anomaly categories)
# ---------------------------------------------------------------------------


class SchemeType(str, Enum):
    # 8 core schemes (merged from original 10 for distinctiveness)
    FICTITIOUS_AP_DISBURSEMENTS = "fictitious_ap_disbursements"  # Embezzlement + expense laundering: fictitious/inflated AP payments via shell or real vendors
    AP_CONTROL_BYPASS = "ap_control_bypass"  # 3-way match bypass / reused invoice / missing PO-GR-invoice chain
    VENDOR_COLLUSION = "vendor_collusion"  # Kickback + related-party: inflated vendor payments + insider relationship
    SHADOW_PAYROLL = "shadow_payroll"  # Ghost employees / payroll fraud
    PAYROLL_TAX_DIVERSION = (
        "payroll_tax_diversion"  # Suppress tax remittance (negative-signal)
    )
    REVENUE_MANIPULATION = (
        "revenue_manipulation"  # Quarter-end inflate + next-period reversal
    )
    INVENTORY_MANIPULATION = (
        "inventory_manipulation"  # Fictitious receipts + write-off arc
    )
    CIRCULAR_CASH_FLOW = "circular_cash_flow"  # Fake collection via AR/suspense cycle
    # Additional opportunistic / statistical patterns
    FICTITIOUS_VENDOR = "fictitious_vendor"  # Payments to shell / fake vendor
    DUPLICATE_PAYMENT = "duplicate_payment"  # Same invoice paid twice
    ROUND_TRIPPING = "round_tripping"  # Circular payments
    SPLIT_TRANSACTION = "split_transaction"  # Threshold splitting
    LATE_POSTING = "late_posting"  # Posting in wrong period
    SOD_VIOLATION = "sod_violation"  # Segregation-of-duties failure
    BENFORD_VIOLATION = "benford_violation"  # Statistical first-digit anomaly
    UNUSUAL_AMOUNT = "unusual_amount"  # Outlier / round-number amount
    DORMANT_ACCOUNT = "dormant_account"  # Activity on dormant GL/entity
    CIRCULAR_TRANSACTION = "circular_transaction"  # Circular payment chain
    TREND_BREAK = "trend_break"  # Temporal pattern break
    UNKNOWN = "unknown"  # Could not classify


# ---------------------------------------------------------------------------
# Tool call record
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool: str
    args: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    # Informational tool-context size (SQL rows, paths, etc.) — NOT LLM output and
    # NOT counted against the run budget (see llm_tokens_* on AgentStep).
    tokens_input: int = 0
    tokens_output: int = 0
    elapsed_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Single agent step
# ---------------------------------------------------------------------------


class AgentStep(BaseModel):
    step_number: int
    # Thinking/reasoning trace from models that expose a separate reasoning stream
    reasoning: Optional[str] = None
    # Visible text content of the LLM response (separate from reasoning in thinking models)
    content: Optional[str] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    # Token counts for the LLM call that drove this step
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    # Separately reported reasoning tokens for thinking models. This may be a
    # subset of llm_tokens_output depending on provider semantics.
    llm_tokens_reasoning: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def llm_tokens_budgeted(self) -> int:
        """Tokens counted against the phase/run budget (prompt + reasoning)."""
        return self.llm_tokens_input + self.llm_tokens_reasoning

    @property
    def total_tokens(self) -> int:
        """LLM API totals plus informational tool-context tokens (audit only)."""
        tc = sum(c.total_tokens for c in self.tool_calls)
        return self.llm_tokens_input + self.llm_tokens_output + tc


# ---------------------------------------------------------------------------
# One suspicion item
# ---------------------------------------------------------------------------


class SuspicionItem(BaseModel):
    # Primary key of the flagged entry (je_header.document_id)
    document_id: Optional[str] = None

    # Optional: the entity at the centre of the scheme
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None  # "vendor" | "employee" | "customer"

    scheme_type: SchemeType = SchemeType.UNKNOWN

    # Legacy field for evaluator hydration only; LLM runs do not score by confidence.
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # Severity 1-5 (1 = low, 5 = critical)
    severity: int = Field(ge=1, le=5, default=3)

    # One-sentence rationale
    rationale: str = ""

    # Key evidence bullet points
    supporting_evidence: List[str] = Field(default_factory=list)

    # Other related document_ids
    related_document_ids: List[str] = Field(default_factory=list)

    # Estimated monetary exposure
    monetary_impact: Optional[float] = None

    # Fiscal period string, e.g. "2024-Q4"
    period: Optional[str] = None

    # GL accounts involved
    gl_accounts: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MemGPT-style evidence register entry
# ---------------------------------------------------------------------------


class EvidenceEntry(BaseModel):
    """
    Structured evidence record produced when turns are evicted from context.

    Inspired by MemGPT (Packer et al., 2023): instead of lossy text summaries
    the agent maintains a structured archival store of hypothesis outcomes.
    Each entry survives context compaction and is re-injected as a concise
    table on every subsequent LLM call.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    scheme: str
    hypothesis: str
    status: str = "INCONCLUSIVE"  # CONFIRMED | RULED_OUT | INCONCLUSIVE
    key_finding: str = ""
    document_ids: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    amount_exposure: float = 0.0
    step_added: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_table_row(self) -> str:
        from researchpkg.forensic_llm.config import (
            TextTruncationLimits,
        )
        from researchpkg.forensic_llm.text_truncation import (
            TruncationSide,
            truncate_text_to_tokens,
        )

        lim = TextTruncationLimits()
        ids_short = ", ".join(self.document_ids[:3]) + (
            "…" if len(self.document_ids) > 3 else ""
        )
        finding_short = truncate_text_to_tokens(
            self.key_finding or "",
            lim.memory_excerpt_clip,
            side=TruncationSide.TAIL,
        )
        scheme_cell = truncate_text_to_tokens(
            self.scheme or "",
            lim.rationale_short,
            side=TruncationSide.HEAD,
        )
        return (
            f"| {self.entry_id} | {scheme_cell} | {self.status:<12} "
            f"| {finding_short} | {ids_short} |"
        )


# ---------------------------------------------------------------------------
# Shared evidence blackboard (MetaGPT-style inter-worker communication)
# ---------------------------------------------------------------------------


class SharedEvidenceBlackboard:
    """
    Thread-safe evidence blackboard for parallel scheme workers.

    Inspired by MetaGPT's shared message pool and AutoGen's group-chat
    architecture.  Each worker scheme agent writes its discovered entities and
    key findings here; the parent orchestrator reads the blackboard after all
    workers complete to drive cross-scheme analysis.

    Workers write lightweight structured facts, NOT raw messages, to avoid
    flooding the orchestrator context.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entity_findings: Dict[str, List[Dict[str, Any]]] = {}
        self._scheme_signals: Dict[str, Dict[str, Any]] = {}

    def write_entity(
        self,
        scheme: str,
        entity_id: str,
        entity_type: str,
        confidence: float,
        rationale: str,
    ) -> None:
        from researchpkg.forensic_llm.config import (
            TextTruncationLimits,
        )
        from researchpkg.forensic_llm.text_truncation import (
            TruncationSide,
            truncate_text_to_tokens,
        )

        with self._lock:
            if scheme not in self._entity_findings:
                self._entity_findings[scheme] = []
            self._entity_findings[scheme].append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "confidence": confidence,
                    "rationale": truncate_text_to_tokens(
                        rationale or "",
                        TextTruncationLimits().rationale_medium,
                        side=TruncationSide.TAIL,
                    ),
                }
            )

    def write_scheme_signal(self, scheme: str, key: str, value: Any) -> None:
        with self._lock:
            if scheme not in self._scheme_signals:
                self._scheme_signals[scheme] = {}
            self._scheme_signals[scheme][key] = value

    def read_all_entities(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {k: list(v) for k, v in self._entity_findings.items()}

    def find_cross_scheme_entities(self) -> List[Dict[str, Any]]:
        """Return entities that appear in findings across ≥2 different schemes."""
        with self._lock:
            entity_schemes: Dict[str, Set[str]] = {}
            entity_meta: Dict[str, Dict[str, Any]] = {}
            for scheme, findings in self._entity_findings.items():
                for f in findings:
                    eid = f["entity_id"]
                    entity_schemes.setdefault(eid, set()).add(scheme)
                    entity_meta.setdefault(eid, f)
            return [
                {**entity_meta[eid], "schemes": list(schemes)}
                for eid, schemes in entity_schemes.items()
                if len(schemes) >= 2
            ]

    def to_context_block(self) -> str:
        """Render the blackboard as a compact context block for the orchestrator."""
        lines: List[str] = ["## Shared Evidence Blackboard\n"]

        cross = self.find_cross_scheme_entities()
        if cross:
            lines.append(
                f"### Cross-scheme entities ({len(cross)} entity/entities appear in ≥2 schemes)\n"
            )
            for item in cross[:20]:
                lines.append(
                    f"- **{item['entity_id']}** ({item['entity_type']}) "
                    f"→ schemes: {', '.join(item['schemes'])} "
                    f"(conf: {item['confidence']:.2f})"
                )
            lines.append("")

        for scheme, findings in self._entity_findings.items():
            if not findings:
                continue
            lines.append(f"### {scheme} ({len(findings)} flagged entities)\n")
            for f in findings[:10]:
                lines.append(
                    f"- {f['entity_id']} [{f['entity_type']}] "
                    f"conf={f['confidence']:.2f}: {f['rationale']}"
                )
            if len(findings) > 10:
                lines.append(f"  … and {len(findings) - 10} more")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Investigation plan (Plan-Execute-Reflect architecture)
# ---------------------------------------------------------------------------

CORE_SCHEMES = frozenset(
    {
        SchemeType.FICTITIOUS_AP_DISBURSEMENTS,
        SchemeType.VENDOR_COLLUSION,
        SchemeType.SHADOW_PAYROLL,
        SchemeType.REVENUE_MANIPULATION,
        SchemeType.INVENTORY_MANIPULATION,
        SchemeType.AP_CONTROL_BYPASS,
        SchemeType.CIRCULAR_CASH_FLOW,
    }
)


class PhaseSummary(BaseModel):
    """Structured summary produced at the end of a scheme investigation phase."""

    scheme: str
    hypotheses_tested: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_ruled_out: int = 0
    hypotheses_open: int = 0
    flagged_entities: List[str] = Field(default_factory=list)
    flagged_document_ids: List[str] = Field(default_factory=list)
    total_flagged_amount: float = 0.0
    confidence: str = "low"  # low | medium | high
    sql_calls_used: int = 0
    open_questions: List[str] = Field(default_factory=list)
    key_findings: str = ""


class CompetingExplanation(BaseModel):
    """Alternative explanation considered during open-world discovery."""

    explanation_type: str = "fraud"  # fraud | benign | control_gap
    title: str = ""
    rationale: str = ""
    confirming_test: str = ""
    falsifying_test: str = ""


class AnomalyCard(BaseModel):
    """
    Open-world discovery unit produced before scheme-specific planning.

    Each card captures a material anomaly pattern, the competing explanations for
    that pattern, and the next evidence needed before committing to a scheme.
    """

    card_id: str = ""
    title: str = ""
    priority: int = 1
    summary: str = ""
    candidate_scheme: str = "unknown"
    scheme_confidence: str = "low"  # low | medium | high
    suspicious_entities: List[str] = Field(default_factory=list)
    suspicious_document_ids: List[str] = Field(default_factory=list)
    primary_processes: List[str] = Field(default_factory=list)
    gl_accounts: List[str] = Field(default_factory=list)
    evidence_signals: List[str] = Field(default_factory=list)
    materiality: str = "medium"  # low | medium | high | critical
    competing_explanations: List[CompetingExplanation] = Field(default_factory=list)
    confirming_queries: List[str] = Field(default_factory=list)
    falsifying_queries: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)


class OpenWorldPlan(BaseModel):
    """Blind anomaly discovery output used to seed bounded scheme planning."""

    anomaly_cards: List[AnomalyCard] = Field(default_factory=list)
    reflection_after_cards: List[int] = Field(default_factory=lambda: [1, 3])


class PlannedHypothesis(BaseModel):
    """One planned investigation angle (planning / dispatch)."""

    hypothesis_id: str = ""
    hypothesis_text: str = ""
    hypothesis_rationale: str = ""
    # Relative complexity / depth weight for worker token allocation (scaled at runtime).
    budget_tokens: int = 0


class SchemePhase(BaseModel):
    """One phase of a planned investigation — one scheme to investigate."""

    scheme: SchemeType
    priority: int = 1
    budget_sql_calls: int = 15
    budget_tokens: int = 2_000_000
    initial_hypotheses: List[PlannedHypothesis] = Field(default_factory=list)

    @field_validator("initial_hypotheses", mode="before")
    @classmethod
    def _normalize_initial_hypotheses(cls, value: Any) -> Any:
        from researchpkg.forensic_llm.plan_utils import (
            coerce_planned_hypotheses,
        )

        return coerce_planned_hypotheses(value)

    plan_rationale: str = ""
    priority_signals: List[str] = Field(default_factory=list)
    benign_rival_explanations: List[str] = Field(default_factory=list)
    planned_query_sequence: List[str] = Field(default_factory=list)
    grounding_query_templates: List[str] = Field(
        default_factory=list,
        description="SQL-shaped templates to list document_id values if hypotheses confirm",
    )
    exit_criteria: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending | in_progress | completed | skipped
    sql_calls_used: int = 0
    tokens_used: int = 0
    phase_summary: Optional[PhaseSummary] = None


class DispatchQueueItem(BaseModel):
    """One hypothesis investigation task in priority order."""

    scheme: str
    hypothesis_id: str  # P1 … P10
    dispatch_priority: int = 1
    # Order in which the planner listed this row in ``dispatch_queue`` (0 = unset).
    # Tie-breaker after ``dispatch_priority`` so runtime order matches global plan
    # order, not lexicographic ``task_id`` (which clusters by scheme name).
    dispatch_sequence: int = 0
    status: str = "pending"  # pending | running | completed | failed | skipped
    task_id: str = ""
    budget_tokens: int = 0
    # Populated for orchestrator-injected tasks (not in the original plan phases).
    hypothesis_text: str = ""
    hypothesis_rationale: str = ""
    source: str = "plan"  # plan | orchestrator


class ToolActionTrace(BaseModel):
    """One tool invocation within a hypothesis investigation step."""

    tool: str = ""
    args_summary: str = ""
    result_preview: str = ""


class HypothesisStepTrace(BaseModel):
    """LLM step trace for a single hypothesis task."""

    step_number: int = 0
    reasoning_snippet: str = ""
    tools: List[ToolActionTrace] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class HypothesisInvestigationTrack(BaseModel):
    """Full investigation trace for one hypothesis task (tasks/<id>/hypothesis_track.json)."""

    task_id: str = ""
    scheme: str = ""
    hypothesis_id: str = ""
    hypothesis_text: str = ""
    hypothesis_rationale: str = ""
    dispatch_priority: int = 0
    dispatch_sequence: int = 0
    source: str = "plan"
    queue_status: str = ""
    result_status: str = "inconclusive"
    finish_reason: str = ""
    tokens_used: int = 0
    steps: int = 0
    sql_calls: int = 0
    effective_sql_calls: int = 0
    flagged_document_ids: List[str] = Field(default_factory=list)
    key_findings: str = ""
    open_questions: List[str] = Field(default_factory=list)
    evidence_checks_run: List[str] = Field(default_factory=list)
    step_traces: List[HypothesisStepTrace] = Field(default_factory=list)
    scratchpad_excerpt: str = ""
    artefact_paths: Dict[str, str] = Field(default_factory=dict)


class HypothesisResult(BaseModel):
    """Canonical per-hypothesis artefact (schemes/<scheme>/p<n>.json)."""

    scheme: str
    hypothesis_id: str
    hypothesis_text: str = ""
    hypothesis_rationale: str = ""
    status: str = "inconclusive"  # confirmed | falsified | inconclusive
    finish_reason: str = "completed"
    tokens_used: int = 0
    steps: int = 0
    sql_calls: int = 0
    effective_sql_calls: int = 0
    flagged_document_ids: List[str] = Field(default_factory=list)
    flagged_entities: List[str] = Field(default_factory=list)
    key_findings: str = ""
    evidence_checks_run: List[str] = Field(default_factory=list)
    benign_rivals_considered: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    task_id: str = ""
    artifact_paths: List[str] = Field(default_factory=list)


class HypothesisTaskBrief(BaseModel):
    """Spawn payload for a single hypothesis worker."""

    task_id: str
    scheme: str
    hypothesis_id: str
    hypothesis_text: str
    hypothesis_rationale: str = ""
    exit_criteria: List[str] = Field(default_factory=list)
    benign_rivals: List[str] = Field(default_factory=list)
    budget_tokens: int = 1_000_000
    planned_query_families: List[str] = Field(default_factory=list)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    plan_rationale: str = ""
    priority_signals: List[str] = Field(default_factory=list)


class HypothesisMemoryEntry(BaseModel):
    """Per-hypothesis rollup in ``memory/global.json`` (readable shared memory)."""

    scheme: str
    hypothesis_id: str
    task_id: str = ""
    status: str = ""
    hypothesis_text: str = ""
    hypothesis_rationale: str = ""
    key_findings: str = ""
    # Sample of JE UUIDs when the full list is large; see ``total_flagged_documents``.
    flagged_document_ids: List[str] = Field(default_factory=list)
    total_flagged_documents: int = 0
    flagged_entities: List[str] = Field(default_factory=list)
    evidence_checks_run: List[str] = Field(default_factory=list)


class GlobalMemory(BaseModel):
    """Orchestrator-curated run-level memory (memory/global.json)."""

    cross_scheme_entities: List[Dict[str, Any]] = Field(default_factory=list)
    scheme_verdicts: Dict[str, str] = Field(default_factory=dict)
    hypothesis_pointers: List[Dict[str, str]] = Field(default_factory=list)
    # Narrative memory: one structured entry per completed hypothesis task (most recent wins per task_id).
    hypothesis_summaries: List[HypothesisMemoryEntry] = Field(default_factory=list)
    open_risks: List[str] = Field(default_factory=list)
    salient_findings: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RunStats(BaseModel):
    """Global run statistics (run_stats.json)."""

    run_id: str = ""
    finish_reason: str = ""
    orientation_tokens: int = 0
    orchestrator_tokens: int = 0
    worker_tokens: int = 0
    tasks_spawned: int = 0
    tasks_injected: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_stopped_at_95pct: int = 0
    max_parallel_workers: int = 5
    wall_time_seconds: float = 0.0


class InvestigationPlan(BaseModel):
    """Structured plan produced by the planner phase."""

    phases: List[SchemePhase] = Field(default_factory=list)
    dispatch_queue: List[DispatchQueueItem] = Field(default_factory=list)
    total_budget_sql: int = 210
    orientation_complete: bool = False
    orientation_risk_summary: List[str] = Field(default_factory=list)
    execution_notes: List[str] = Field(default_factory=list)
    reflection_after_phases: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Autonomous worker runtime models
# ---------------------------------------------------------------------------


class WorkerBrief(BaseModel):
    """Runtime-only brief for a generic autonomous worker."""

    worker_id: str = ""
    scheme_or_goal: str = ""
    brief: str = ""
    candidate_schemes: List[str] = Field(default_factory=list)
    budget_sql_calls: int = 12
    budget_tokens: int = 1_000_000
    parent_context: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkerMessage(BaseModel):
    """Mailbox message sent from the parent orchestrator to a worker."""

    sender: str = "parent"
    instruction: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    consumed: bool = False


class WorkerStatus(BaseModel):
    """Live status snapshot for a worker known to the parent runtime."""

    worker_id: str
    scheme_or_goal: str = ""
    state: str = "pending"  # pending | running | completed | failed | cancelled
    candidate_schemes: List[str] = Field(default_factory=list)
    mailbox_depth: int = 0
    steps_taken: int = 0
    sql_calls_used: int = 0
    code_interpreter_calls_used: int = 0
    flagged_document_ids: List[str] = Field(default_factory=list)
    latest_summary: str = ""
    run_dir: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class WorkerSummary(BaseModel):
    """Persisted free-form worker result normalized for parent reconciliation."""

    worker_id: str
    scheme_or_goal: str = ""
    status: str = "completed"
    candidate_schemes: List[str] = Field(default_factory=list)
    flagged_document_ids: List[str] = Field(default_factory=list)
    flagged_entities: List[str] = Field(default_factory=list)
    total_flagged_amount: float = 0.0
    key_findings: str = ""
    open_questions: List[str] = Field(default_factory=list)
    evidence_checks_run: List[str] = Field(default_factory=list)
    recommended_scheme_verdicts: Dict[str, str] = Field(default_factory=dict)
    confidence: str = "medium"
    sql_calls_used: int = 0
    code_interpreter_calls_used: int = 0


class CoverageEntry(BaseModel):
    """Parent-side canonical closed-world coverage state for one scheme."""

    scheme: str
    status: str = "uncovered"  # uncovered | in_progress | strong_evidence | no_material_evidence | insufficient_data
    supporting_worker_ids: List[str] = Field(default_factory=list)
    flagged_document_ids: List[str] = Field(default_factory=list)
    confidence: str = ""
    notes: str = ""


class CoverageLedger(BaseModel):
    """Parent-owned scheme coverage ledger for autonomous orchestration."""

    entries: Dict[str, CoverageEntry] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Scheme report — groups SuspicionItems into coherent schemes
# ---------------------------------------------------------------------------

_DEFAULT_W_CONF = 0.35
_DEFAULT_W_SEV = 0.20
_DEFAULT_W_COV = 0.25
_DEFAULT_W_MON = 0.20


class SchemeReport(BaseModel):
    """
    A coherent fraud scheme grouping multiple SuspicionItems by
    (entity_id, scheme_type).  Computed by post-processing the flat
    suspicion_list into scheme-level findings.
    """

    scheme_type: SchemeType
    perpetrator_id: Optional[str] = None
    perpetrator_type: Optional[str] = None
    start_period: Optional[str] = None
    end_period: Optional[str] = None
    document_ids: List[str] = Field(default_factory=list)
    related_entity_ids: List[str] = Field(default_factory=list)
    total_monetary_impact: float = 0.0
    items: List[SuspicionItem] = Field(default_factory=list)

    @property
    def avg_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.confidence for i in self.items) / len(self.items)

    @property
    def max_severity(self) -> int:
        if not self.items:
            return 1
        return max(i.severity for i in self.items)

    def scheme_score(
        self,
        max_impact_in_dataset: float = 1.0,
        w_conf: float = _DEFAULT_W_CONF,
        w_sev: float = _DEFAULT_W_SEV,
        w_cov: float = _DEFAULT_W_COV,
        w_mon: float = _DEFAULT_W_MON,
    ) -> float:
        """
        Composite score: higher = more severe / more urgent.

        scheme_score = w_c * avg_confidence
                     + w_s * (max_severity / 5)
                     + w_v * coverage_ratio    (always 1.0 at agent side)
                     + w_m * log_monetary_normalized
        """
        log_mon = 0.0
        if self.total_monetary_impact > 0 and max_impact_in_dataset > 0:
            log_mon = math.log1p(self.total_monetary_impact) / math.log1p(
                max_impact_in_dataset
            )
        return (
            w_conf * self.avg_confidence
            + w_sev * (self.max_severity / 5.0)
            + w_cov * 1.0
            + w_mon * log_mon
        )

    @staticmethod
    def from_suspicion_list(items: List[SuspicionItem]) -> List["SchemeReport"]:
        """Group a flat suspicion_list into SchemeReports by (entity_id, scheme_type)."""
        groups: Dict[tuple, List[SuspicionItem]] = {}
        for item in items:
            key = (item.entity_id or "_none_", item.scheme_type)
            groups.setdefault(key, []).append(item)

        reports: List["SchemeReport"] = []
        for (eid, stype), group_items in groups.items():
            all_doc_ids = []
            all_related = set()
            total_impact = 0.0
            periods = []
            for it in group_items:
                if it.document_id:
                    all_doc_ids.append(it.document_id)
                all_related.update(it.related_document_ids)
                if it.monetary_impact:
                    total_impact += it.monetary_impact
                if it.period:
                    periods.append(it.period)

            perp_id = eid if eid != "_none_" else None
            perp_type = group_items[0].entity_type if perp_id else None

            sr = SchemeReport(
                scheme_type=stype,
                perpetrator_id=perp_id,
                perpetrator_type=perp_type,
                start_period=min(periods) if periods else None,
                end_period=max(periods) if periods else None,
                document_ids=all_doc_ids,
                related_entity_ids=sorted(all_related - set(all_doc_ids)),
                total_monetary_impact=total_impact,
                items=group_items,
            )
            reports.append(sr)

        reports.sort(key=lambda r: r.scheme_score(), reverse=True)
        return reports


# ---------------------------------------------------------------------------
# Final output of a single run
# ---------------------------------------------------------------------------


class ForensicReport(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    task: str = "full"
    strategy: str = "hypothesis_orchestrated"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Token accounting
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    # Separately reported reasoning tokens. This may already be a subset of
    # total_tokens_output depending on the backend.
    total_tokens_reasoning: int = 0
    total_tool_tokens: int = 0
    budget_exhausted: bool = False
    steps_taken: int = 0

    # LLM reliability accounting (includes recovered retry failures).
    llm_errors_total: int = 0
    llm_errors_unrecovered: int = 0

    # High-level termination reason for this run (for manifests / dashboards).
    # Examples: "completed", "budget_exhausted", "forced_finish_no_finish_investigation".
    termination_reason: str = ""

    # If the run crashes or is interrupted, capture the error for run_manifest.json
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Full trace of every agent step
    steps: List[AgentStep] = Field(default_factory=list)

    # The machine-readable suspicion list
    suspicion_list: List[SuspicionItem] = Field(default_factory=list)

    # Scheme-level reports (grouped from suspicion_list)
    scheme_reports: List[SchemeReport] = Field(default_factory=list)

    # The investigation plan used (Plan-Execute-Reflect only)
    investigation_plan: Optional[InvestigationPlan] = None

    # Optional blind anomaly-discovery output captured before scheme planning.
    open_world_plan: Optional[OpenWorldPlan] = None

    # Runtime-only worker artefacts for autonomous orchestration mode.
    worker_summaries: Dict[str, WorkerSummary] = Field(default_factory=dict)
    coverage_ledger: Optional[CoverageLedger] = None

    # The narrative markdown report written by the agent
    narrative: str = ""

    # Scratchpad notes accumulated by the agent (tool: scratchpad)
    scratchpad: str = ""

    @property
    def total_tokens(self) -> int:
        return (
            self.total_tokens_input + self.total_tokens_output + self.total_tool_tokens
        )

    @property
    def budget_counted_tokens(self) -> int:
        """Tokens that count against the run budget (prompt + reasoning)."""
        return self.total_tokens_input + self.total_tokens_reasoning

    @property
    def budget_fraction(self) -> float:
        """Fraction of the configured budget used (0–1+)."""
        return 0.0  # filled in by the runner using BudgetTracker


# ---------------------------------------------------------------------------
# Offline evaluation against anomaly_labels
# ---------------------------------------------------------------------------


class SchemeMetrics(BaseModel):
    scheme_type: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    n_true: int = 0
    n_flagged: int = 0
    n_correct: int = 0


class SchemeEvalMetrics(BaseModel):
    """Scheme-level evaluation metrics (one ground-truth scheme = perpetrator + type)."""

    scheme_id: str = ""
    scheme_type: str = ""
    perpetrator_id: str = ""
    n_true_docs: int = 0
    n_flagged_docs: int = 0
    coverage_ratio: float = 0.0
    detected: bool = False
    perpetrator_identified: bool = False


class SchemeEvalSummary(BaseModel):
    """Aggregate scheme-level metrics across all ground-truth schemes."""

    n_true_schemes: int = 0
    n_detected: int = 0
    n_predicted_schemes: int = 0
    scheme_detection_rate: float = 0.0
    scheme_precision: float = 0.0
    scheme_f1: float = 0.0
    mean_coverage: float = 0.0
    perpetrator_identification_rate: float = 0.0
    per_scheme: List[SchemeEvalMetrics] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    run_id: str
    model: str
    task: str

    # Entry-level (document_id exact match; unit of evaluation = JE)
    entry_precision: float = 0.0
    entry_recall: float = 0.0
    entry_f1: float = 0.0
    entry_accuracy: float = 0.0

    # ROC / PR area under curve (require confidence scores)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    # Per-scheme breakdown (document-level, by anomaly_type)
    per_scheme: List[SchemeMetrics] = Field(default_factory=list)

    # Scheme-level evaluation (perpetrator + type grouping)
    scheme_eval: Optional[SchemeEvalSummary] = None

    # Aggregate counts
    n_true_anomalies: int = 0
    n_flagged: int = 0
    n_correct: int = 0
    n_steps: int = 0
    total_tokens: int = 0
    budget_fraction: float = 0.0

    # Derived confusion-matrix style counts at the entry level.
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Entry-level confusion matrix between predicted scheme_type and
    # ground-truth scheme_type (unit = journal-entry document_id).
    #
    # This is different from `per_scheme` (which aggregates by predicted
    # scheme buckets) and from `scheme_eval` (which groups scheme instances
    # by perpetrator + type).
    confusion_matrix: Optional[Dict[str, Any]] = None

    # False positive rate = FP / (FP + TP) = 1 - precision (false discovery rate).
    # Fraction of flagged items that are wrong; 0 = perfect, 1 = all flags are wrong.
    false_positive_rate: float = 0.0

    # Human-readable summary of model performance, including a per-scheme
    # breakdown.  This is intended for downstream dashboards or reports.
    performance_summary: str = ""
