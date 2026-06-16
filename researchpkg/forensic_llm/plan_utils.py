"""Utilities for investigation plans and hypothesis dispatch queues."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from researchpkg.forensic_llm.models import (
    DispatchQueueItem,
    HypothesisTaskBrief,
    InvestigationPlan,
    PlannedHypothesis,
    SchemePhase,
    SchemeType,
)

log = logging.getLogger(__name__)

CANONICAL_SCHEME_VALUES: Tuple[str, ...] = (
    "fictitious_ap_disbursements",
    "revenue_manipulation",
    "vendor_collusion",
    "shadow_payroll",
    "inventory_manipulation",
)

_HYPOTHESIS_ID_RE = re.compile(r"^\[?(P\d+)\]?\s*", re.IGNORECASE)
_HYP_WAVE_RE = re.compile(r"^P(\d+)$", re.IGNORECASE)


def parse_hypothesis_line(text: str, fallback_id: str) -> Tuple[str, str]:
    """Return (hypothesis_id, hypothesis_text) from a legacy plan line like '[P1] …'."""
    stripped = (text or "").strip()
    match = _HYPOTHESIS_ID_RE.match(stripped)
    if match:
        return match.group(1).upper(), stripped[match.end() :].strip()
    return fallback_id, stripped


def coerce_planned_hypotheses(raw: Any) -> List[Dict[str, str]]:
    """Normalise planner output (strings or objects) into PlannedHypothesis dicts."""
    if not raw:
        return []
    out: List[Dict[str, str]] = []
    for idx, item in enumerate(raw):
        fallback_id = f"P{idx + 1}"
        if isinstance(item, str):
            hid, text = parse_hypothesis_line(item, fallback_id)
            out.append(
                {
                    "hypothesis_id": hid,
                    "hypothesis_text": text,
                    "hypothesis_rationale": "",
                }
            )
            continue
        if isinstance(item, PlannedHypothesis):
            out.append(item.model_dump())
            continue
        if isinstance(item, dict):
            text = str(item.get("hypothesis_text") or "").strip()
            hid = str(item.get("hypothesis_id") or "").strip().upper()
            if not hid and text:
                hid, text = parse_hypothesis_line(text, fallback_id)
            if not hid:
                hid = fallback_id
            out.append(
                {
                    "hypothesis_id": hid,
                    "hypothesis_text": text,
                    "hypothesis_rationale": str(
                        item.get("hypothesis_rationale") or ""
                    ).strip(),
                    "budget_tokens": int(item.get("budget_tokens") or 0),
                }
            )
    return out


def planned_hypothesis_from_phase(
    phase: SchemePhase, hypothesis_id: str
) -> Optional[PlannedHypothesis]:
    for idx, hyp in enumerate(phase.initial_hypotheses or [], start=1):
        hid = (hyp.hypothesis_id or f"P{idx}").upper()
        if hid == hypothesis_id.upper():
            return hyp
    return None


def task_id_for(scheme: str, hypothesis_id: str) -> str:
    return f"{scheme}_{hypothesis_id}"


def hypothesis_wave_level(hypothesis_id: str) -> int:
    """
    Numeric wave from per-scheme hypothesis id (P1 → 1, P2 → 2).

    Used so the runtime launches **all P1 tasks across schemes**, then all P2, etc.
    """
    hid = (hypothesis_id or "").strip().upper()
    m = _HYP_WAVE_RE.match(hid)
    if m:
        return int(m.group(1))
    return 9999


def dispatch_queue_sort_key(item: DispatchQueueItem) -> Tuple[int, int, str, str]:
    """
    Runtime order: ``dispatch_priority`` (1 = soonest), then ``dispatch_sequence``.

    Planned tasks get priorities from :func:`enrich_dispatch_priorities` (P1 wave
    across all schemes, then P2, …). Injected tasks may use lower priorities to
    jump ahead without re-sorting by scheme.
    """
    return (
        int(item.dispatch_priority or 0),
        int(item.dispatch_sequence or 0),
        item.scheme,
        item.task_id,
    )


def _planned_hypothesis_list(raw: Any) -> List[PlannedHypothesis]:
    rows = coerce_planned_hypotheses(raw)
    return [PlannedHypothesis.model_validate(r) for r in rows]


def _dedupe_hypotheses(hyps: List[PlannedHypothesis]) -> List[PlannedHypothesis]:
    seen: set[str] = set()
    out: List[PlannedHypothesis] = []
    for idx, hyp in enumerate(hyps, start=1):
        hid = (hyp.hypothesis_id or f"P{idx}").upper()
        if hid in seen:
            continue
        seen.add(hid)
        out.append(
            hyp.model_copy(update={"hypothesis_id": hid})
            if hasattr(hyp, "model_copy")
            else PlannedHypothesis(
                hypothesis_id=hid,
                hypothesis_text=hyp.hypothesis_text,
                hypothesis_rationale=hyp.hypothesis_rationale,
                budget_tokens=int(getattr(hyp, "budget_tokens", 0) or 0),
            )
        )
    return out


def _cap_dispatch_items_by_scheme(
    items: List[DispatchQueueItem],
    max_per_scheme: int,
) -> List[DispatchQueueItem]:
    counts: Dict[str, int] = {}
    capped: List[DispatchQueueItem] = []
    for item in sorted(items, key=dispatch_queue_sort_key):
        n = counts.get(item.scheme, 0)
        if n >= max_per_scheme:
            continue
        counts[item.scheme] = n + 1
        capped.append(item)
    return capped


def enforce_plan_hypothesis_guardrails(
    plan: InvestigationPlan,
    *,
    min_per_scheme: int,
    max_per_scheme: int,
) -> Tuple[InvestigationPlan, Dict[str, Any]]:
    """
    Trim/pad per-scheme ``initial_hypotheses`` to ``[min_per_scheme, max_per_scheme]`` and
    rebuild ``dispatch_queue`` from phases (authoritative).
    """
    min_per_scheme = max(1, int(min_per_scheme))
    max_per_scheme = max(min_per_scheme, int(max_per_scheme))
    stats: Dict[str, Any] = {
        "trimmed": {},
        "padded": {},
        "missing_schemes_filled": [],
    }

    # Preserve planner complexity weights across rebuild (dispatch_queue + phases).
    pre_budgets: Dict[Tuple[str, str], int] = {}
    for dq in plan.dispatch_queue or []:
        try:
            sv, hid_u = str(dq.scheme), str(dq.hypothesis_id or "").strip().upper()
        except Exception:
            continue
        b = int(getattr(dq, "budget_tokens", 0) or 0)
        if b > 0:
            pre_budgets[(sv, hid_u)] = b
    for phase in plan.phases:
        try:
            sv = phase.scheme.value
        except Exception:
            continue
        for hyp in phase.initial_hypotheses or []:
            b = int(getattr(hyp, "budget_tokens", 0) or 0)
            if b <= 0:
                continue
            hid_u = str(hyp.hypothesis_id or "").strip().upper()
            k = (sv, hid_u)
            pre_budgets[k] = max(pre_budgets.get(k, 0), b)

    phase_by_scheme = {p.scheme.value: p for p in plan.phases}
    ordered_phases: List[SchemePhase] = []

    for idx, scheme_val in enumerate(CANONICAL_SCHEME_VALUES):
        phase = phase_by_scheme.get(scheme_val)
        if phase is None:
            phase = SchemePhase(
                scheme=SchemeType(scheme_val),
                priority=idx + 1,
                initial_hypotheses=[],
                plan_rationale=(
                    "Phase synthesized by guardrails — planner omitted this scheme."
                ),
                exit_criteria=[
                    "Each hypothesis receives CONFIRMED, RULED_OUT, or INCONCLUSIVE with cited evidence."
                ],
            )
            stats["missing_schemes_filled"].append(scheme_val)

        hyps = _dedupe_hypotheses(_planned_hypothesis_list(phase.initial_hypotheses))
        if len(hyps) > max_per_scheme:
            stats["trimmed"][scheme_val] = len(hyps) - max_per_scheme
            hyps = hyps[:max_per_scheme]

        existing_ids = {h.hypothesis_id.upper() for h in hyps}
        padded_before = len(hyps)
        if len(hyps) < min_per_scheme:
            for fb in fallback_initial_hypotheses_for_scheme(
                scheme_val, max_per_scheme
            ):
                if len(hyps) >= min_per_scheme:
                    break
                hid = fb.hypothesis_id.upper()
                if hid in existing_ids:
                    continue
                hyps.append(fb)
                existing_ids.add(hid)
        if len(hyps) > padded_before:
            stats["padded"][scheme_val] = len(hyps) - padded_before

        phase.initial_hypotheses = hyps
        phase.priority = idx + 1
        ordered_phases.append(phase)

    plan.phases = ordered_phases
    plan.dispatch_queue = build_dispatch_queue(
        plan,
        max_hypotheses_per_scheme=max_per_scheme,
    )
    for dq in plan.dispatch_queue:
        k = (str(dq.scheme), str(dq.hypothesis_id or "").strip().upper())
        if k in pre_budgets:
            dq.budget_tokens = pre_budgets[k]
    return plan, stats


def apply_worker_budget_pool_to_dispatch_queue(
    items: List[DispatchQueueItem],
    *,
    allocation_pool_tokens: int,
    min_per_task: int = 25_000,
) -> int:
    """
    Assign each queue item's ``budget_tokens`` so the sum equals *allocation_pool_tokens*.

    Planner-supplied values are treated as **non-negative weights** (larger ⇒ more
    tokens). Zeros are treated as equal weight after applying the per-task floor.
    """
    n = len(items)
    if n == 0:
        return 0
    pool = max(0, int(allocation_pool_tokens))
    min_t = max(1, int(min_per_task))

    if pool == 0:
        for it in items:
            it.budget_tokens = min_t
        return sum(it.budget_tokens for it in items)

    if pool < n * min_t:
        base = pool // n
        rem = pool - base * n
        for i, it in enumerate(items):
            it.budget_tokens = base + (1 if i < rem else 0)
        return pool

    extra_pool = pool - n * min_t
    raw_w = [max(0, int(getattr(it, "budget_tokens", 0) or 0)) for it in items]
    wsum = sum(raw_w)
    if wsum == 0:
        raw_w = [1] * n
        wsum = n

    floats = [extra_pool * (raw_w[i] / wsum) for i in range(n)]
    ints = [int(x) for x in floats]
    rem = extra_pool - sum(ints)
    fracs = sorted(
        [(floats[i] - ints[i], i) for i in range(n)],
        reverse=True,
        key=lambda t: t[0],
    )
    for k in range(rem):
        ints[fracs[k][1]] += 1

    for i, it in enumerate(items):
        it.budget_tokens = min_t + ints[i]
    return sum(it.budget_tokens for it in items)


def build_dispatch_queue(
    plan: InvestigationPlan,
    max_hypotheses_per_scheme: int = 10,
) -> List[DispatchQueueItem]:
    """
    Build or validate dispatch_queue from plan phases.

    If the planner emitted dispatch_queue, normalise it; otherwise derive from
    phases sorted by scheme priority then hypothesis order.
    """
    if plan.dispatch_queue:
        items: List[DispatchQueueItem] = []
        seq = 0
        for raw in plan.dispatch_queue:
            if isinstance(raw, DispatchQueueItem):
                item = raw
            elif isinstance(raw, dict):
                item = DispatchQueueItem.model_validate(raw)
            else:
                continue
            if not item.task_id:
                item.task_id = task_id_for(item.scheme, item.hypothesis_id)
            if item.dispatch_sequence == 0:
                item.dispatch_sequence = seq
            seq += 1
            phase = find_phase(plan, item.scheme)
            if phase is not None:
                planned = planned_hypothesis_from_phase(phase, item.hypothesis_id)
                if planned is not None:
                    if not (item.hypothesis_text or "").strip():
                        item.hypothesis_text = planned.hypothesis_text
                    if not (item.hypothesis_rationale or "").strip():
                        item.hypothesis_rationale = planned.hypothesis_rationale
                    if not int(getattr(item, "budget_tokens", 0) or 0):
                        pb = int(getattr(planned, "budget_tokens", 0) or 0)
                        if pb > 0:
                            item.budget_tokens = pb
            items.append(item)
        return enrich_dispatch_priorities(
            _cap_dispatch_items_by_scheme(items, max_hypotheses_per_scheme)
        )

    # Fallback: bucket by P-wave so P1 runs across all schemes before P2, …
    by_wave: Dict[int, List[Tuple[SchemePhase, PlannedHypothesis, str]]] = {}
    phases = sorted(plan.phases, key=lambda p: p.priority)
    for phase in phases:
        scheme = phase.scheme.value
        hyps = (phase.initial_hypotheses or [])[:max_hypotheses_per_scheme]
        for idx, hyp in enumerate(hyps, start=1):
            hyp_id = (hyp.hypothesis_id or f"P{idx}").upper()
            wave = hypothesis_wave_level(hyp_id)
            by_wave.setdefault(wave, []).append((phase, hyp, hyp_id))

    items: List[DispatchQueueItem] = []
    seq = 0
    for wave in sorted(by_wave):
        entries = sorted(
            by_wave[wave],
            key=lambda t: (t[0].priority, t[0].scheme.value, t[2]),
        )
        for phase, hyp, hyp_id in entries:
            scheme = phase.scheme.value
            items.append(
                DispatchQueueItem(
                    scheme=scheme,
                    hypothesis_id=hyp_id,
                    dispatch_priority=0,
                    dispatch_sequence=seq,
                    task_id=task_id_for(scheme, hyp_id),
                    hypothesis_text=hyp.hypothesis_text,
                    hypothesis_rationale=hyp.hypothesis_rationale,
                    budget_tokens=max(0, int(getattr(hyp, "budget_tokens", 0) or 0)),
                )
            )
            seq += 1
    return enrich_dispatch_priorities(items)


def _wave_plan_sort_key(item: DispatchQueueItem) -> Tuple[int, int, str, str]:
    """Plan-time ordering: P1 wave across schemes, then P2, …"""
    return (
        hypothesis_wave_level(item.hypothesis_id),
        int(item.dispatch_sequence or 0),
        item.scheme,
        item.task_id,
    )


def enrich_dispatch_priorities(
    items: List[DispatchQueueItem],
) -> List[DispatchQueueItem]:
    """
    Normalize run order: **all P1 tasks (every scheme), then all P2, …**

    ``dispatch_priority`` is reassigned 1…N in that order (planner-supplied
    priorities are not used for scheduling). Ties within a wave use
    ``dispatch_sequence`` (planner / JSON order), then ``scheme``.
    """
    items = sorted(items, key=_wave_plan_sort_key)
    for idx, item in enumerate(items, start=1):
        item.dispatch_priority = idx
    return items


def find_phase(plan: InvestigationPlan, scheme: str) -> Optional[SchemePhase]:
    for phase in plan.phases:
        if phase.scheme.value == scheme:
            return phase
    return None


def hypothesis_text_from_phase(phase: SchemePhase, hypothesis_id: str) -> str:
    for idx, line in enumerate(phase.initial_hypotheses or [], start=1):
        hid, text = parse_hypothesis_line(line, f"P{idx}")
        if hid == hypothesis_id:
            return text
    return ""


def build_hypothesis_brief(
    plan: InvestigationPlan,
    item: DispatchQueueItem,
    budget_tokens: int,
    shared_context: dict,
) -> HypothesisTaskBrief:
    phase = find_phase(plan, item.scheme)
    hyp_text = (item.hypothesis_text or "").strip()
    hyp_rationale = (item.hypothesis_rationale or "").strip()
    exit_criteria: List[str] = []
    benign: List[str] = []
    planned_queries: List[str] = []
    rationale = ""
    signals: List[str] = []
    if phase is not None:
        if not hyp_text:
            hyp_text = hypothesis_text_from_phase(phase, item.hypothesis_id)
        if not hyp_rationale:
            hyp_rationale = hypothesis_rationale_from_phase(phase, item.hypothesis_id)
        exit_criteria = list(phase.exit_criteria or [])
        benign = list(phase.benign_rival_explanations or [])
        planned_queries = list(phase.planned_query_sequence or [])
        rationale = phase.plan_rationale or ""
        signals = list(phase.priority_signals or [])
    return HypothesisTaskBrief(
        task_id=item.task_id,
        scheme=item.scheme,
        hypothesis_id=item.hypothesis_id,
        hypothesis_text=hyp_text,
        hypothesis_rationale=hyp_rationale,
        exit_criteria=exit_criteria,
        benign_rivals=benign,
        budget_tokens=budget_tokens,
        planned_query_families=planned_queries,
        shared_context=shared_context,
        plan_rationale=rationale,
        priority_signals=signals,
    )


# Default hypotheses when the LLM omits `phases` (repair failed). Free-form text + rationale.
_FALLBACK_HYPOTHESES_BY_SCHEME: Dict[str, List[Dict[str, str]]] = {
    "fictitious_ap_disbursements": [
        {
            "hypothesis_id": "P1",
            "hypothesis_text": "Payments concentrate on vendors with thin or missing master-data coverage.",
            "hypothesis_rationale": "Test whether a small vendor set absorbs abnormal disbursement share versus P2P baselines; confirm with vendor master joins and payment JE exemplars.",
        },
        {
            "hypothesis_id": "P2",
            "hypothesis_text": "Duplicate invoice references or (vendor, amount, date) tuples recur above baseline.",
            "hypothesis_rationale": "Recycled vouchers can mask fictitious AP; population-level duplicate rates distinguish process noise from structuring.",
        },
        {
            "hypothesis_id": "P3",
            "hypothesis_text": "AP settlements lack expected goods-receipt or three-way-match linkage.",
            "hypothesis_rationale": "Control bypass often appears as invoices paid without GR/IR fields populated; compare linkage rates to company baselines.",
        },
        {
            "hypothesis_id": "P4",
            "hypothesis_text": "Round-dollar or threshold-adjacent payments cluster on few vendors.",
            "hypothesis_rationale": "Manual structuring leaves amount signatures just below approval limits; export full P2P amounts for distribution tests.",
        },
        {
            "hypothesis_id": "P5",
            "hypothesis_text": "Vendor payment bank details overlap employee payroll accounts.",
            "hypothesis_rationale": "Kickback routing may reuse employee bank tokens on vendor side; cross-match bank identifiers where available.",
        },
        {
            "hypothesis_id": "P6",
            "hypothesis_text": "Period-end AP accruals reverse unusually early in the next period.",
            "hypothesis_rationale": "Earnings management can park expense in accruals then release; trace accrual JEs across adjacent periods.",
        },
        {
            "hypothesis_id": "P7",
            "hypothesis_text": "Dormant vendors reactivate with sudden high-value payment bursts.",
            "hypothesis_rationale": "Dormant-ID reuse is a common shell-vendor pattern; screen vendors with long inactivity then payment spikes.",
        },
        {
            "hypothesis_id": "P8",
            "hypothesis_text": "A single creator or approver dominates suspicious AP journal entries.",
            "hypothesis_rationale": "Fraud concentration often maps to one posting actor; measure creator share on high-risk AP slices.",
        },
        {
            "hypothesis_id": "P9",
            "hypothesis_text": "P2P lines lack auxiliary accounts at rates far below peer processes.",
            "hypothesis_rationale": "Sparse sub-ledger linkage can hide payee identity; compare auxiliary population rates across business processes.",
        },
        {
            "hypothesis_id": "P10",
            "hypothesis_text": "High-value P2P payments cluster on a small set of document types or references.",
            "hypothesis_rationale": "Structuring may reuse document types; profile amount tails by document_type and reference patterns.",
        },
    ],
    "shadow_payroll": [
        {
            "hypothesis_id": "P1",
            "hypothesis_text": "Recently hired employees receive recurring month-end salary accruals.",
            "hypothesis_rationale": "Ghost schemes onboard plausible HR records shortly before pay runs; join payroll auxiliaries to hire_date and flag hire-to-first-pay gaps under ~90 days.",
        },
        {
            "hypothesis_id": "P2",
            "hypothesis_text": "Payments continue after recorded employee termination dates.",
            "hypothesis_rationale": "Zombie payroll persists when termination dates precede payment posting dates; join payroll JEs to HR status timelines.",
        },
        {
            "hypothesis_id": "P3",
            "hypothesis_text": "Payroll accruals use two-line structure (salary/payable) without employer social-charge lines.",
            "hypothesis_rationale": "Legitimate month-end runs often split gross and URSSAF; ghost runs may omit the 645100 debit; compare line-count patterns on H2R payroll documents.",
        },
        {
            "hypothesis_id": "P4",
            "hypothesis_text": "Multiple employees share the same payroll bank institution (routing or bank name).",
            "hypothesis_rationale": "Diversion may reuse bank name/routing while varying account numbers; cluster employees on payroll_routing_code and payroll_bank_name.",
        },
        {
            "hypothesis_id": "P5",
            "hypothesis_text": "Payroll expense is reclassified through non-payroll clearing accounts.",
            "hypothesis_rationale": "Diversion may route gross pay through suspense before settlement; profile account paths on payroll amounts.",
        },
        {
            "hypothesis_id": "P6",
            "hypothesis_text": "Payroll amounts for select employees are statistical outliers versus peers.",
            "hypothesis_rationale": "Inflated payments to colluding insiders show as per-employee amount tails; export employee-level payroll distributions.",
        },
        {
            "hypothesis_id": "P7",
            "hypothesis_text": "Payroll journals are created by users outside the payroll function.",
            "hypothesis_rationale": "Segregation failures let non-HR creators post payroll; compare `created_by` on H2R lines to HR roster roles.",
        },
        {
            "hypothesis_id": "P8",
            "hypothesis_text": "Withholding or benefit lines diverge from gross pay trends for the same employees.",
            "hypothesis_rationale": "Net-pay diversion can leave gross intact but distort withholding; reconcile tax/benefit lines per employee.",
        },
        {
            "hypothesis_id": "P9",
            "hypothesis_text": "H2R postings use created_by actors absent from the employee master at material volume.",
            "hypothesis_rationale": "Orientation user–HR gaps may indicate ghost administrators; measure mismatch rate on payroll-related JEs.",
        },
        {
            "hypothesis_id": "P10",
            "hypothesis_text": "Payroll accrual and payment pairs show abnormal timing gaps versus pay calendar.",
            "hypothesis_rationale": "Delayed or accelerated settlement can mask unauthorized runs; compare accrual-to-payment lags to baseline.",
        },
    ],
    "vendor_collusion": [
        {
            "hypothesis_id": "P1",
            "hypothesis_text": "A single vendor absorbs a disproportionate share of procurement spend.",
            "hypothesis_rationale": "Collusion often concentrates spend; compare vendor-level totals and unit prices to peer vendors.",
        },
        {
            "hypothesis_id": "P2",
            "hypothesis_text": "Invoice amounts cluster just below approval thresholds.",
            "hypothesis_rationale": "Threshold splitting is a classic kickback enabler; test amount histograms near policy limits.",
        },
        {
            "hypothesis_id": "P3",
            "hypothesis_text": "Vendor and employee master data share addresses, phones, or tax identifiers.",
            "hypothesis_rationale": "Related-party ties hide in contact-field overlap; fuzzy-match vendor and employee attributes.",
        },
        {
            "hypothesis_id": "P4",
            "hypothesis_text": "The same approver repeatedly authorizes one vendor's invoices.",
            "hypothesis_rationale": "Approver–vendor dyads with high concentration weaken segregation; rank approval chains by vendor.",
        },
        {
            "hypothesis_id": "P5",
            "hypothesis_text": "Newly registered vendors invoice large amounts within days of onboarding.",
            "hypothesis_rationale": "Shell vendors often invoice immediately after creation; measure days from vendor start to first material invoice.",
        },
        {
            "hypothesis_id": "P6",
            "hypothesis_text": "Invoiced quantities exceed recorded receipts on a recurring basis.",
            "hypothesis_rationale": "Overbilling collusion shows as invoice–receipt gaps; compare billed vs received quantities where data exists.",
        },
        {
            "hypothesis_id": "P7",
            "hypothesis_text": "Consulting or pass-through vendors bill repetitive round amounts.",
            "hypothesis_rationale": "Sham services repeat identical charges; screen round-amount concentration by vendor category.",
        },
        {
            "hypothesis_id": "P8",
            "hypothesis_text": "Circular payment patterns link vendors and customers with minimal goods movement.",
            "hypothesis_rationale": "Round-tripping inflates both sides; trace cash paths and inventory corroboration for linked parties.",
        },
        {
            "hypothesis_id": "P9",
            "hypothesis_text": "Vendor payments lack lettrage or clearing linkage where peers show high match rates.",
            "hypothesis_rationale": "Weak matching controls ease duplicate payment fraud; compare lettrage population on vendor settlements vs baseline.",
        },
        {
            "hypothesis_id": "P10",
            "hypothesis_text": "Intercompany vendor flags coincide with above-median payment velocity.",
            "hypothesis_rationale": "IC vendors can obscure related-party flows; test payment counts and amounts on is_intercompany vendors.",
        },
    ],
    "revenue_manipulation": [
        {
            "hypothesis_id": "P1",
            "hypothesis_text": "Revenue postings cluster in the last days of fiscal periods at elevated rates.",
            "hypothesis_rationale": "Cut-off abuse shows as period-end spikes versus intra-period baselines; use posting-date distributions on revenue accounts.",
        },
        {
            "hypothesis_id": "P2",
            "hypothesis_text": "Expense deferrals (prepaid asset vs class-6 expense) spike near period-end or mid-month.",
            "hypothesis_rationale": "Window dressing shifts costs out of the period via Dr deferred-charges / Cr expense; profile posting-day concentration on that GL pair.",
        },
        {
            "hypothesis_id": "P3",
            "hypothesis_text": "Revenue grows without comparable cash collection or AR movement.",
            "hypothesis_rationale": "Fictitious revenue decouples from cash; compare revenue GL activity to AR/cash timing.",
        },
        {
            "hypothesis_id": "P4",
            "hypothesis_text": "Manual or non-standard journals hit revenue accounts outside billing interfaces.",
            "hypothesis_rationale": "Top-side entries bypass O2C controls; isolate manual SA lines on income accounts.",
        },
        {
            "hypothesis_id": "P5",
            "hypothesis_text": "Provision accounts are debited to release other operating income near close.",
            "hypothesis_rationale": "Cookie-jar releases use Dr provisions / Cr other income; screen that pair separately from O2C revenue spikes.",
        },
        {
            "hypothesis_id": "P6",
            "hypothesis_text": "Early revenue (AR debit, product revenue credit) lacks delivery or COGS corroboration.",
            "hypothesis_rationale": "Premature recognition inflates sales without inventory or cost movement; match revenue JEs to customer-level delivery signals.",
        },
        {
            "hypothesis_id": "P7",
            "hypothesis_text": "Related-party customers contribute material revenue without arm's-length pricing variation.",
            "hypothesis_rationale": "Round-tripping inflates sales to affiliates; flag related-party revenue concentration and pricing bands.",
        },
        {
            "hypothesis_id": "P8",
            "hypothesis_text": "Identical journal narratives repeat across unrelated revenue documents.",
            "hypothesis_rationale": "Scripted fake entries reuse text templates; cluster `line_text` or reference patterns on revenue JEs.",
        },
        {
            "hypothesis_id": "P9",
            "hypothesis_text": "O2C credits lack customer auxiliary linkage at rates below company baseline.",
            "hypothesis_rationale": "Missing customer keys on revenue lines impede validation; measure auxiliary coverage on O2C population.",
        },
        {
            "hypothesis_id": "P10",
            "hypothesis_text": "Large O2C credits post on dates immediately before fiscal period close.",
            "hypothesis_rationale": "Cut-off stuffing concentrates at period end; compare last-3-day share on revenue accounts to annual average.",
        },
    ],
    "inventory_manipulation": [
        {
            "hypothesis_id": "P1",
            "hypothesis_text": "Inventory balances grow faster than COGS or shipments without logistics support.",
            "hypothesis_rationale": "Overstated inventory inflates margins; reconcile inventory GL growth to movement in related accounts.",
        },
        {
            "hypothesis_id": "P2",
            "hypothesis_text": "Write-downs or obsolescence entries cluster only after period close or audit windows.",
            "hypothesis_rationale": "Delayed impairment hides obsolescence until external pressure; time write-off postings vs operational events.",
        },
        {
            "hypothesis_id": "P3",
            "hypothesis_text": "Receipt journals lack matching issues or consumption in the same period.",
            "hypothesis_rationale": "Fictitious receipts inflate stock without downstream usage; pair receipt and issue flows by SKU or site.",
        },
        {
            "hypothesis_id": "P4",
            "hypothesis_text": "Inter-location transfers move inventory without freight, duty, or timing consistency.",
            "hypothesis_rationale": "Phantom inventory circulates on paper transfers; validate transfer JEs against logistics metadata.",
        },
        {
            "hypothesis_id": "P5",
            "hypothesis_text": "Standard cost revaluations lift margins without BOM or purchase price changes.",
            "hypothesis_rationale": "Capitalization via standard cost can boost earnings; compare revaluation JEs to master cost changes.",
        },
        {
            "hypothesis_id": "P6",
            "hypothesis_text": "Cycle-count adjustments repeat in the same locations without root-cause resolution.",
            "hypothesis_rationale": "Systematic count fraud shows as recurring adjustments; track adjustment frequency by warehouse.",
        },
        {
            "hypothesis_id": "P7",
            "hypothesis_text": "Obsolete SKU segments carry rising balances despite low turnover.",
            "hypothesis_rationale": "Obsolete buildup inflates assets; segment inventory by age and turnover metrics.",
        },
        {
            "hypothesis_id": "P8",
            "hypothesis_text": "Consignment or third-party inventory is recognized as owned on the GL.",
            "hypothesis_rationale": "Ownership blur inflates on-books inventory; test consignment flags vs GL recognition.",
        },
        {
            "hypothesis_id": "P9",
            "hypothesis_text": "Inventory receipt and issue activity show different creator concentration patterns.",
            "hypothesis_rationale": "Split-creator collusion may separate receipt vs write-off actors; compare created_by distributions across movement types.",
        },
        {
            "hypothesis_id": "P10",
            "hypothesis_text": "Inventory GL activity lacks references or auxiliary fields more than other asset accounts.",
            "hypothesis_rationale": "Weak metadata on inventory JEs obscures traceability; benchmark reference/auxiliary rates vs other asset families.",
        },
    ],
}


def fallback_initial_hypotheses_for_scheme(
    scheme_value: str, max_n: int
) -> List[PlannedHypothesis]:
    """Return up to ``max_n`` default hypotheses for ``scheme_value`` (repair fallback)."""
    rows = _FALLBACK_HYPOTHESES_BY_SCHEME.get(scheme_value, [])
    if max_n < 1:
        return []
    return [PlannedHypothesis.model_validate(r) for r in rows[:max_n]]
