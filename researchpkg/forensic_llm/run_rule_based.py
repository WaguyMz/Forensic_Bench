#!/usr/bin/env python3
"""
run_rule_based.py — Rule-based fraud detection baseline for the LLM forensic benchmark.

Detects injected fraud schemes using exact PCG GL account patterns derived from the
generator source code (no LLM, no ML — pure accounting rule logic). Serves as the
gold standard for assessing scheme detectability in the benchmark.

Output files are identical to run.py (suspicion_list.json, eval_*.json, run_manifest.json).
Evaluation is performed by the same forensic_llm.evaluator used by all LLM runs.

Data sources (mutually exclusive):
  --data-dir   Read je_header.csv / je_line.csv / anomaly_labels.csv from a directory.
  --db-name    Read from a Postgres labelled DB (datasynth_forensic__<key>).
               Uses the same env vars as run.py:
                 FORENSIC_DB_HOST      (default: localhost)
                 FORENSIC_DB_PORT      (default: 5432)
                 FORENSIC_DB_USER      (default: test)
                 FORENSIC_DB_PASSWORD  (default: empty)

Usage:
    python run_rule_based.py --data-dir output/forensic_llm
    python run_rule_based.py --db-name datasynth_forensic_public__energy --output results/rule_based/
    python run_rule_based.py --db-name datasynth_forensic_public__transport --quiet
"""

import argparse
import calendar
import csv
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

# ─── PCG (Plan Comptable Général) account constants ───────────────────────────
# Source: crates/datasynth-core/src/pcg.rs + enhanced_orchestrator.rs materialization
AP_CONTROL = "401000"  # Fournisseurs
AR_CONTROL = "411000"  # Clients
BANK_ACCOUNT = "512000"  # Banque
GENERAL_SUSPENSE = "471000"  # Compte d'attente
WAGES_PAYABLE = "421000"  # Personnel — rémunérations dues
SALARIES_WAGES = "641100"  # Salaires et traitements
SOCIAL_SECURITY = "645100"  # Cotisations sociales
PAYROLL_TAX_PAYABLE = "431000"  # Sécurité sociale — charges patronales
HONORAIRES = "622600"  # Honoraires
MARKET_RESEARCH = "622700"  # Études et recherches
ADVERTISING = "623000"  # Publicité
MISC_SERVICES = "628000"  # Autres charges externes divers
OFFICE_SUPPLIES = "606300"  # Fournitures de bureau
MAINTENANCE = "615000"  # Entretien et réparations
COGS = "603000"  # Variation de stock — matières
PRODUCT_REVENUE = "701000"  # Ventes de produits finis
OTHER_REVENUE = "758000"  # Produits divers de gestion
INPUT_VAT = "445660"  # TVA déductible
DEFERRED_CHARGES = "486000"  # Charges constatées d'avance
PROVISIONS = "151000"  # Provisions pour risques
INVOICES_ISSUED = "418100"  # Clients — factures à établir
BAD_DEBT_EXPENSE = "654000"  # Pertes sur créances irrécouvrables
INVENTORY = "370000"  # Stocks de marchandises
ASSET_IMPAIRMENT = "685000"  # Dotations aux amortissements (write-off)

EMBEZZLEMENT_EXPENSE = {OFFICE_SUPPLIES, MAINTENANCE}  # 606300, 615000
KICKBACK_EXPENSE = {HONORAIRES, MARKET_RESEARCH}  # 622600, 622700


# ─── Canonical scheme names — snake_case matching forensic_llm.models.SchemeType ─
# Source: crates/datasynth-generators/src/anomaly/scheme_advancer.rs
SCHEME_FICTITIOUS_AP = "fictitious_ap_disbursements"
SCHEME_VENDOR_COLLUSION = "vendor_collusion"
SCHEME_REVENUE_MANIP = "revenue_manipulation"
SCHEME_SHADOW_PAYROLL = "shadow_payroll"
SCHEME_INVENTORY = "inventory_manipulation"
SCHEME_AP_CONTROL_BYPASS = "ap_control_bypass"
SCHEME_CIRCULAR_CASH_FLOW = "circular_cash_flow"

MODEL_NAME = "rule_based_v2"


# ─── Data models (field-compatible with forensic_llm.models.SuspicionItem) ───


@dataclass
class RBSuspicionItem:
    document_id: str
    scheme_type: str
    confidence: float
    severity: int
    rationale: str
    supporting_evidence: list
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    related_document_ids: list = field(default_factory=list)
    monetary_impact: Optional[float] = None
    period: Optional[str] = None
    gl_accounts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "scheme_type": self.scheme_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "rationale": self.rationale,
            "supporting_evidence": self.supporting_evidence,
            "related_document_ids": self.related_document_ids,
            "monetary_impact": self.monetary_impact,
            "period": self.period,
            "gl_accounts": self.gl_accounts,
        }


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    s = s[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(s: str) -> float:
    try:
        return float(s or 0)
    except (ValueError, TypeError):
        return 0.0


def _build_index(
    headers: list[dict], lines: list[dict], labels: list[dict], employees: list[dict]
) -> dict:
    header_by_doc: dict[str, dict] = {h["document_id"]: h for h in headers}

    lines_by_doc: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        lines_by_doc[line["document_id"]].append(line)

    gt_by_doc: dict[str, set[str]] = defaultdict(set)
    for lbl in labels:
        injected = lbl.get("is_injected", "")
        if str(injected).lower() in ("true", "1", "yes"):
            gt_by_doc[lbl["document_id"]].add(lbl["anomaly_type"])

    gt_by_label: dict[str, set[str]] = defaultdict(set)
    for doc_id, label_set in gt_by_doc.items():
        for lbl in label_set:
            gt_by_label[lbl].add(doc_id)

    emp_map = {e["employee_id"]: e for e in employees if "employee_id" in e}

    # Employees sharing payroll institution (routing + bank name) with at least one peer.
    institution_clusters: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, emp in emp_map.items():
        routing = (emp.get("payroll_routing_code") or "").strip()
        bank = (emp.get("payroll_bank_name") or "").strip()
        if routing or bank:
            institution_clusters[(routing, bank)].append(eid)
    payroll_institution_cluster_ids: set[str] = set()
    for eids in institution_clusters.values():
        if len(eids) >= 2:
            payroll_institution_cluster_ids.update(eids)

    return {
        "header_by_doc": header_by_doc,
        "lines_by_doc": lines_by_doc,
        "gt_by_doc": gt_by_doc,
        "gt_by_label": gt_by_label,
        "emp_map": emp_map,
        "payroll_institution_cluster_ids": payroll_institution_cluster_ids,
        "all_headers": headers,
    }


def load_data(data_dir: str) -> dict:
    headers = load_csv(os.path.join(data_dir, "je_header.csv"))
    lines = load_csv(os.path.join(data_dir, "je_line.csv"))
    labels = load_csv(os.path.join(data_dir, "anomaly_labels.csv"))
    employees = load_csv(os.path.join(data_dir, "employees.csv"))
    return _build_index(headers, lines, labels, employees)


def load_data_sql(db_name: str) -> dict:
    """Load all tables from a Postgres labelled DB.

    Uses the same env vars as run.py:
      FORENSIC_DB_HOST, FORENSIC_DB_PORT, FORENSIC_DB_USER, FORENSIC_DB_PASSWORD.
    The DB must be a labelled database (keeps anomaly_labels with is_injected column).
    """
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print(
            "ERROR: psycopg2 not installed. Run: pip install psycopg2-binary",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = psycopg2.connect(
        dbname=db_name.replace("_public", ""),
        host=os.environ.get("FORENSIC_DB_HOST", "localhost"),
        port=int(os.environ.get("FORENSIC_DB_PORT", "5432")),
        user=os.environ.get("FORENSIC_DB_USER", "test"),
        password=os.environ.get("FORENSIC_DB_PASSWORD", ""),
    )

    def fetch(sql: str) -> list[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [
                {k: ("" if v is None else str(v)) for k, v in row.items()}
                for row in cur.fetchall()
            ]

    try:
        headers = fetch("SELECT * FROM je_header")
        lines = fetch("SELECT * FROM je_line")
        labels = fetch("SELECT * FROM anomaly_labels")
        employees = fetch("SELECT * FROM employees")
    finally:
        conn.close()

    return _build_index(headers, lines, labels, employees)


# ─── Line-level helpers ───────────────────────────────────────────────────────


def je_has_debit(lines: list[dict], account: str) -> bool:
    return any(
        l["gl_account"] == account and parse_amount(l["debit_amount"]) > 0
        for l in lines
    )


def je_has_credit(lines: list[dict], account: str) -> bool:
    return any(
        l["gl_account"] == account and parse_amount(l["credit_amount"]) > 0
        for l in lines
    )


def posting_hour(header: dict) -> Optional[int]:
    for field_name in ("posting_date", "document_date"):
        val = header.get(field_name, "")
        if val and "T" in val:
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).hour
            except ValueError:
                pass
    return None


def last_bday_of_month(d: date) -> bool:
    last = calendar.monthrange(d.year, d.month)[1]
    candidate = date(d.year, d.month, last)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return d == candidate


def _payroll_employee_id_from_aux(
    aux: str, aux_prefix: str = "421000/"
) -> Optional[str]:
    aux = (aux or "").strip()
    if not aux:
        return None
    if aux.startswith(aux_prefix) and len(aux) > len(aux_prefix):
        return aux[len(aux_prefix) :]
    if aux.startswith("EMP-"):
        return aux
    return None


def _is_recent_hire(
    emp: dict, posting_date: Optional[date], window_days: int = 90
) -> bool:
    if not posting_date:
        return False
    for field in ("hire_date", "creation_date"):
        d = parse_date(emp.get(field, ""))
        if d and 0 <= (posting_date - d).days <= window_days:
            return True
    return False


def _item(
    doc_id,
    scheme,
    confidence,
    severity,
    reasons,
    entity_id=None,
    related=None,
    accounts=None,
):
    """Construct an RBSuspicionItem from a list of reason strings."""
    return RBSuspicionItem(
        document_id=doc_id,
        scheme_type=scheme,
        confidence=confidence,
        severity=severity,
        rationale=reasons[0] if reasons else "",
        supporting_evidence=reasons,
        entity_id=entity_id,
        related_document_ids=related or [],
        gl_accounts=accounts or [],
    )


# ─── Detection functions ──────────────────────────────────────────────────────


def detect_embezzlement(data: dict) -> list[RBSuspicionItem]:
    """
    GradualEmbezzlementScheme → fictitious_ap_disbursements

    Invoice:  Dr (606300 OR 615000) + Dr 445660 VAT, Cr 401000
    Payment:  Dr 401000, Cr 512000
    Same reference, payment 5–45 days after invoice (R3 timing noise widened window).
    Behavioral: payment creator differs from invoice creator (separation of duties).
    After-hours posting (22:00–06:00) in early stages.
    Source: schemes/embezzlement.rs + enhanced_orchestrator.rs EmbezzleInvoice/EmbezzlePayment
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    invoice_docs: dict[str, dict] = {}
    payment_docs: dict[str, dict] = {}

    for doc_id, lines in lines_by_doc.items():
        has_embezz = any(
            l["gl_account"] in EMBEZZLEMENT_EXPENSE
            and parse_amount(l["debit_amount"]) > 0
            for l in lines
        )
        if has_embezz and je_has_credit(lines, AP_CONTROL):
            hdr = header_by_doc.get(doc_id)
            if hdr:
                invoice_docs[doc_id] = hdr
        if je_has_debit(lines, AP_CONTROL) and je_has_credit(lines, BANK_ACCOUNT):
            hdr = header_by_doc.get(doc_id)
            if hdr:
                payment_docs[doc_id] = hdr

    inv_by_ref: dict[str, list] = defaultdict(list)
    pay_by_ref: dict[str, list] = defaultdict(list)
    for doc_id, hdr in invoice_docs.items():
        if hdr.get("reference"):
            inv_by_ref[hdr["reference"]].append((doc_id, hdr))
    for doc_id, hdr in payment_docs.items():
        if hdr.get("reference"):
            pay_by_ref[hdr["reference"]].append((doc_id, hdr))

    for ref in set(inv_by_ref) & set(pay_by_ref):
        for inv_id, inv_hdr in inv_by_ref[ref]:
            inv_date = parse_date(inv_hdr.get("posting_date", ""))
            creator = inv_hdr.get("created_by", "")
            for pay_id, pay_hdr in pay_by_ref[ref]:
                if pay_id == inv_id:
                    continue
                pay_date = parse_date(pay_hdr.get("posting_date", ""))
                if not inv_date or not pay_date:
                    continue
                gap = (pay_date - inv_date).days
                # R3: Normal(μ=22, σ=6) → widen to [5, 45]
                if not (5 <= gap <= 45):
                    continue

                pay_creator = pay_hdr.get("created_by", "")
                # Behavioral signal: separation of duties. Fraud is more plausible when
                # the payment is posted by a different user than the invoice creator.
                split_creator = pay_creator != creator

                hour = posting_hour(inv_hdr)
                after_hours = hour is not None and (hour >= 22 or hour < 6)

                # Score: split-creator is the main signal; after-hours corroborates.
                if split_creator and after_hours:
                    confidence, severity = 0.93, 4
                elif split_creator:
                    confidence, severity = 0.88, 3
                elif after_hours:
                    confidence, severity = 0.82, 3
                else:
                    confidence, severity = 0.75, 3

                reasons = [
                    f"Invoice {inv_id}: Dr {'/'.join(sorted(EMBEZZLEMENT_EXPENSE))} + {INPUT_VAT} Cr {AP_CONTROL}",
                    f"Payment {pay_id}: Dr {AP_CONTROL} Cr {BANK_ACCOUNT} ({gap}d later, ref '{ref}')",
                ]
                if split_creator:
                    reasons.append(
                        f"Split-creator signal: invoice by '{creator}', payment by '{pay_creator}'"
                    )
                else:
                    reasons.append(f"Same creator '{creator}'")
                if after_hours:
                    reasons.append(
                        f"After-hours posting at {hour:02d}:xx (embezzlement early-stage signal)"
                    )
                accts = sorted(
                    EMBEZZLEMENT_EXPENSE | {INPUT_VAT, AP_CONTROL, BANK_ACCOUNT}
                )

                for flagged_id in (inv_id, pay_id):
                    items.append(
                        _item(
                            flagged_id,
                            SCHEME_FICTITIOUS_AP,
                            confidence,
                            severity,
                            reasons[:],
                            entity_id=creator,
                            related=[inv_id, pay_id],
                            accounts=accts,
                        )
                    )
    return items


def detect_expense_laundering(data: dict) -> list[RBSuspicionItem]:
    """
    ExpenseLaunderingScheme → fictitious_ap_disbursements

    Staging:   Dr 628000 (misc services) + Dr 445660 VAT, Cr 401000
    Clearance: Dr 401000, Cr 512000  (within 0–8 days; R3 widened from 1–2)
    Reference: uses shared reference to link staging↔clearance (no format-based scoring)
    Behavioral: clearance creator differs from staging creator (separation of duties).
    Source: schemes/expense_laundering.rs + SuspenseStaging/SuspenseClearance actions
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    staging_docs: dict[str, dict] = {}
    clear_docs: dict[str, dict] = {}

    for doc_id, lines in lines_by_doc.items():
        if je_has_debit(lines, MISC_SERVICES) and je_has_credit(lines, AP_CONTROL):
            hdr = header_by_doc.get(doc_id)
            if hdr:
                staging_docs[doc_id] = hdr
        if je_has_debit(lines, AP_CONTROL) and je_has_credit(lines, BANK_ACCOUNT):
            hdr = header_by_doc.get(doc_id)
            if hdr:
                clear_docs[doc_id] = hdr

    stg_by_ref: dict[str, list] = defaultdict(list)
    clr_by_ref: dict[str, list] = defaultdict(list)
    for doc_id, hdr in staging_docs.items():
        if hdr.get("reference"):
            stg_by_ref[hdr["reference"]].append((doc_id, hdr))
    for doc_id, hdr in clear_docs.items():
        if hdr.get("reference"):
            clr_by_ref[hdr["reference"]].append((doc_id, hdr))

    for ref in set(stg_by_ref) & set(clr_by_ref):
        for stg_id, stg_hdr in stg_by_ref[ref]:
            stg_date = parse_date(stg_hdr.get("posting_date", ""))
            creator = stg_hdr.get("created_by", "")
            for clr_id, clr_hdr in clr_by_ref[ref]:
                if clr_id == stg_id:
                    continue
                clr_date = parse_date(clr_hdr.get("posting_date", ""))
                if not stg_date or not clr_date:
                    continue
                gap = (clr_date - stg_date).days
                # R3: clearance gap widened to 1–8 days
                if not (0 <= gap <= 10):
                    continue

                clr_creator = clr_hdr.get("created_by", "")
                split_creator = clr_creator != creator
                if split_creator:
                    confidence = 0.85
                else:
                    confidence = 0.74

                reasons = [
                    f"Staging {stg_id}: Dr {MISC_SERVICES} + {INPUT_VAT} Cr {AP_CONTROL}",
                    f"Clearance {clr_id}: Dr {AP_CONTROL} Cr {BANK_ACCOUNT} within {gap}d, ref '{ref}'",
                ]
                if split_creator:
                    reasons.append(
                        f"Split-creator: staged by '{creator}', cleared by '{clr_creator}'"
                    )
                accts = [MISC_SERVICES, INPUT_VAT, AP_CONTROL, BANK_ACCOUNT]

                for flagged_id in (stg_id, clr_id):
                    items.append(
                        _item(
                            flagged_id,
                            SCHEME_FICTITIOUS_AP,
                            confidence,
                            3,
                            reasons[:],
                            entity_id=creator,
                            related=[stg_id, clr_id],
                            accounts=accts,
                        )
                    )
    return items


def detect_vendor_kickback(data: dict) -> list[RBSuspicionItem]:
    """
    VendorKickbackScheme → vendor_collusion

    Invoice:        Dr (622600 OR 622700), Cr 401000
    Split payments: 2–3 × Dr 401000, Cr 512000, jittered ±4d around T+10/T+15/T+20 (R3)
    Reference: uses shared reference to link invoice↔payments (no format-based scoring)
    Behavioral: split payments often posted by different user(s) than invoice creator.
    Source: schemes/kickback.rs + InflateInvoice/PayInflatedInvoicePartial/MakeKickbackPayment
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    invoice_docs: dict[str, dict] = {}
    payment_docs: dict[str, dict] = {}

    for doc_id, lines in lines_by_doc.items():
        has_kickback = any(
            l["gl_account"] in KICKBACK_EXPENSE and parse_amount(l["debit_amount"]) > 0
            for l in lines
        )
        if has_kickback and je_has_credit(lines, AP_CONTROL):
            hdr = header_by_doc.get(doc_id)
            if hdr:
                invoice_docs[doc_id] = hdr
        if je_has_debit(lines, AP_CONTROL) and je_has_credit(lines, BANK_ACCOUNT):
            hdr = header_by_doc.get(doc_id)
            if hdr:
                payment_docs[doc_id] = hdr

    inv_by_ref: dict[str, list] = defaultdict(list)
    pay_by_ref: dict[str, list] = defaultdict(list)
    for doc_id, hdr in invoice_docs.items():
        if hdr.get("reference"):
            inv_by_ref[hdr["reference"]].append((doc_id, hdr))
    for doc_id, hdr in payment_docs.items():
        if hdr.get("reference"):
            pay_by_ref[hdr["reference"]].append((doc_id, hdr))

    for ref in set(inv_by_ref) & set(pay_by_ref):
        for inv_id, inv_hdr in inv_by_ref[ref]:
            inv_date = parse_date(inv_hdr.get("posting_date", ""))
            creator = inv_hdr.get("created_by", "")

            # R5: payments posted by colluding approver — do NOT filter by same creator
            timed: list[tuple[str, int, str]] = []  # (pay_id, gap_days, pay_creator)
            if inv_date:
                for pid, phdr in pay_by_ref[ref]:
                    if pid == inv_id:
                        continue
                    pd = parse_date(phdr.get("posting_date", ""))
                    if pd:
                        gap = (pd - inv_date).days
                        # R3: ±4 jitter around T+10/T+15/T+20 → accept [5, 35]
                        if 5 <= gap <= 35:
                            timed.append((pid, gap, phdr.get("created_by", "")))
            if len(timed) < 2:
                continue

            split_creator = any(c != creator for _, _, c in timed)

            pay_desc = ", ".join(f"{pid}(T+{g}d,by={c})" for pid, g, c in timed)
            reasons = [
                f"Kickback invoice {inv_id}: Dr {'/'.join(sorted(KICKBACK_EXPENSE))} Cr {AP_CONTROL}",
                f"Split payments on ref '{ref}': {pay_desc}",
            ]
            if split_creator:
                reasons.append(
                    f"Invoice by '{creator}'; payments by different creator(s) — split-creator signal"
                )

            confidence = 0.95 if split_creator else 0.82
            accts = sorted(KICKBACK_EXPENSE | {AP_CONTROL, BANK_ACCOUNT})

            flagged = [inv_id] + [pid for pid, _, _ in timed]
            for flagged_id in flagged:
                items.append(
                    _item(
                        flagged_id,
                        SCHEME_VENDOR_COLLUSION,
                        confidence,
                        3,
                        reasons[:],
                        entity_id=creator,
                        related=flagged[:],
                        accounts=accts,
                    )
                )
    return items


def detect_related_party(data: dict) -> list[RBSuspicionItem]:
    """
    RelatedPartyScheme → vendor_collusion

    Same approver posts 3+ procurement JEs (Dr 622600/622700, Cr 401000) to the
    same vendor auxiliary account. Reference format is not used for scoring.
    Behavioral: repeated procurement postings by same creator to same vendor auxiliary account.
    Vendor setup ref: VND-NNNNNN (R1).
    Source: schemes/related_party.rs + RelatedPartyProcurement action
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    from_creator_vendor: dict[tuple, list] = defaultdict(list)

    for doc_id, lines in lines_by_doc.items():
        has_kickback = any(
            l["gl_account"] in KICKBACK_EXPENSE and parse_amount(l["debit_amount"]) > 0
            for l in lines
        )
        if not (has_kickback and je_has_credit(lines, AP_CONTROL)):
            continue
        hdr = header_by_doc.get(doc_id)
        if not hdr:
            continue
        creator = hdr.get("created_by", "")
        ref = hdr.get("reference", "")
        aux_accounts = {
            l["auxiliary_account_number"]
            for l in lines
            if l.get("auxiliary_account_number")
        }
        for aux in aux_accounts:
            from_creator_vendor[(creator, aux)].append((doc_id, hdr, ref))

    for (creator, vendor_aux), docs in from_creator_vendor.items():
        if len(docs) < 3:
            continue
        doc_ids = [d for d, _, _ in docs]

        # No reference-format regex scoring; rely on concentration only.
        confidence = 0.88

        reasons = [
            f"Approver '{creator}' posted {len(docs)} procurement JEs to vendor '{vendor_aux}'",
            f"All Dr {'/'.join(sorted(KICKBACK_EXPENSE))} Cr {AP_CONTROL} (related-party procurement)",
        ]
        # No special user-id oracle; rely on behavioral concentration instead.
        accts = sorted(KICKBACK_EXPENSE | {AP_CONTROL})

        for doc_id, _, _ in docs:
            items.append(
                _item(
                    doc_id,
                    SCHEME_VENDOR_COLLUSION,
                    confidence,
                    3,
                    reasons[:],
                    entity_id=creator,
                    related=doc_ids[:],
                    accounts=accts,
                )
            )
    return items


def detect_revenue_manipulation(data: dict) -> list[RBSuspicionItem]:
    """
    RevenueManipulationScheme → revenue_manipulation

    ManipulateRevenue/ChannelStuff: Dr 411000 (+ 418100), Cr 701000
    DeferExpense:                   Dr 486000, Cr 6XXX expense
    ReleaseReserves:                Dr 151000, Cr 758000
    Timing: mostly day >= 22 (R3 jitter to [22,28]); ~25% mid-month (days 12–21).
    Reference format is not used for scoring.
    Source: schemes/revenue_manipulation.rs
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    for doc_id, lines in lines_by_doc.items():
        hdr = header_by_doc.get(doc_id)
        if not hdr:
            continue
        posting_date = parse_date(hdr.get("posting_date", ""))
        if not posting_date:
            continue
        day = posting_date.day
        # R3 lower bound 22; R6 mid-month window [12, 21].
        is_period_end = day >= 22
        is_mid_month = 12 <= day < 22
        if not (is_period_end or is_mid_month):
            continue

        creator = hdr.get("created_by", "")
        timing_label = "period-end" if is_period_end else "mid-month"

        reasons: list[str] = []
        accts: list[str] = []

        if (
            je_has_debit(lines, AR_CONTROL) or je_has_debit(lines, INVOICES_ISSUED)
        ) and je_has_credit(lines, PRODUCT_REVENUE):
            reasons.append(
                f"{timing_label.capitalize()} revenue entry (day {day}): "
                f"Dr {AR_CONTROL}/{INVOICES_ISSUED} Cr {PRODUCT_REVENUE}"
            )
            accts = [AR_CONTROL, INVOICES_ISSUED, PRODUCT_REVENUE]

        if je_has_debit(lines, DEFERRED_CHARGES) and any(
            l["gl_account"].startswith("6") and parse_amount(l["credit_amount"]) > 0
            for l in lines
        ):
            reasons.append(
                f"{timing_label.capitalize()} expense deferral (day {day}): "
                f"Dr {DEFERRED_CHARGES} Cr 6XXX"
            )
            accts = accts or [DEFERRED_CHARGES]

        if je_has_debit(lines, PROVISIONS) and je_has_credit(lines, OTHER_REVENUE):
            reasons.append(
                f"{timing_label.capitalize()} reserve release (day {day}): "
                f"Dr {PROVISIONS} Cr {OTHER_REVENUE}"
            )
            accts = accts or [PROVISIONS, OTHER_REVENUE]

        if not reasons:
            continue

        confidence = 0.85 if is_period_end else 0.72

        items.append(
            _item(
                doc_id,
                SCHEME_REVENUE_MANIP,
                confidence,
                3,
                reasons,
                entity_id=creator,
                accounts=accts,
            )
        )
    return items


def detect_shadow_payroll(data: dict) -> list[RBSuspicionItem]:
    """
    ShadowPayrollScheme → shadow_payroll

    Ghost payroll: Dr 641100, Cr 421000 (two-line accrual, no 645100 URSSAF split).
    Legitimate cover payroll: Dr 641100 + Dr 645100, Cr 421000 (three-line).
    Ghost employees are in HR master; detect via recent hire, institution clustering,
    simplified line structure, and month-end timing. PAY-/SEV- refs are not sufficient alone.
    Source: schemes/shadow_payroll.rs + enhanced_orchestrator cover payroll
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]
    emp_map: dict[str, dict] = data.get("emp_map", {})
    institution_ids: set[str] = data.get("payroll_institution_cluster_ids", set())
    aux_prefix = "421000/"

    for doc_id, lines in lines_by_doc.items():
        if not (
            je_has_debit(lines, SALARIES_WAGES) and je_has_credit(lines, WAGES_PAYABLE)
        ):
            continue
        hdr = header_by_doc.get(doc_id)
        if not hdr:
            continue

        ref = hdr.get("reference", "") or ""
        posting_date = parse_date(hdr.get("posting_date", ""))
        creator = hdr.get("created_by", "")
        aux_accounts = {l.get("auxiliary_account_number", "") or "" for l in lines}

        has_social_line = je_has_debit(lines, SOCIAL_SECURITY)
        simplified_accrual = not has_social_line
        on_last_bday = bool(posting_date and last_bday_of_month(posting_date))
        ref_is_payroll = ref.startswith("PAY-") or ref.startswith("SEV-")

        recent_hire_ids: list[str] = []
        institution_linked_ids: list[str] = []
        for aux in aux_accounts:
            eid = _payroll_employee_id_from_aux(aux, aux_prefix)
            if not eid:
                continue
            emp = emp_map.get(eid) or {}
            if _is_recent_hire(emp, posting_date):
                recent_hire_ids.append(eid)
            if eid in institution_ids:
                institution_linked_ids.append(eid)

        reasons: list[str] = []
        primary_signals = 0

        if recent_hire_ids:
            reasons.append(
                f"Payroll auxiliary references recently hired employee(s): {recent_hire_ids}"
            )
            primary_signals += 1

        if institution_linked_ids:
            reasons.append(
                f"Beneficiary(ies) share payroll institution with other employees: "
                f"{institution_linked_ids}"
            )
            primary_signals += 1

        if simplified_accrual and on_last_bday and ref_is_payroll:
            reasons.append(
                "Simplified two-line month-end payroll accrual (Dr salary, Cr payable; "
                "no employer social-charge line) with PAY-/SEV- reference"
            )
            primary_signals += 1
        elif simplified_accrual and on_last_bday:
            reasons.append(
                "Simplified two-line month-end payroll accrual without URSSAF split"
            )
            primary_signals += 1

        if ref_is_payroll:
            reasons.append(f"Payroll-style reference: '{ref}'")
        if on_last_bday:
            reasons.append(f"Posted on last business day ({posting_date})")

        if primary_signals == 0:
            continue

        confidence = (
            0.94 if primary_signals >= 2 else 0.86 if primary_signals == 1 else 0.80
        )

        items.append(
            _item(
                doc_id,
                SCHEME_SHADOW_PAYROLL,
                confidence,
                3,
                reasons,
                entity_id=creator,
                accounts=[SALARIES_WAGES, WAGES_PAYABLE],
            )
        )
    return items


def detect_inventory_manipulation(data: dict) -> list[RBSuspicionItem]:
    """
    InventoryManipulationScheme → inventory_manipulation

    FictitiousGoodsReceipt: Dr 370000, Cr 603000 (inventory inflation pattern)
    InventoryWriteOff:      Dr 685000, Cr 370000 (inventory write-off pattern)
    Behavioral: split-creator between inflate vs write-off postings can corroborate.
    Pairing: amount-symmetry within same calendar year (inflate total ≈ writeoff total ±20%).
    R4 cover filter: skip SVC-* references (legitimate cover invoices).
    Source: schemes/inventory_manipulation.rs
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    # No reference-format regex scoring; use accounting structure + amount symmetry.

    inflate_docs: dict[str, dict] = {}
    writeoff_docs: dict[str, dict] = {}

    for doc_id, lines in lines_by_doc.items():
        hdr = header_by_doc.get(doc_id)
        if not hdr:
            continue
        ref = hdr.get("reference", "") or ""
        if je_has_debit(lines, INVENTORY) and je_has_credit(lines, COGS):
            inflate_docs[doc_id] = hdr
        if je_has_debit(lines, ASSET_IMPAIRMENT) and je_has_credit(lines, INVENTORY):
            writeoff_docs[doc_id] = hdr

    accts = [INVENTORY, COGS, ASSET_IMPAIRMENT]
    all_paired: set[str] = set()

    # Collect inflate docs by year — for amount-symmetry pairing
    inf_by_year: dict[int, list[tuple[str, dict, float]]] = defaultdict(list)
    wof_by_year: dict[int, list[tuple[str, dict, float]]] = defaultdict(list)

    for doc_id, hdr in inflate_docs.items():
        d = parse_date(hdr.get("posting_date", ""))
        if d:
            # Extract total debit amount for INVENTORY from lines
            amount = sum(
                parse_amount(l["debit_amount"])
                for l in lines_by_doc.get(doc_id, [])
                if l["gl_account"] == INVENTORY
            )
            inf_by_year[d.year].append((doc_id, hdr, amount))

    for doc_id, hdr in writeoff_docs.items():
        d = parse_date(hdr.get("posting_date", ""))
        if d:
            amount = sum(
                parse_amount(l["credit_amount"])
                for l in lines_by_doc.get(doc_id, [])
                if l["gl_account"] == INVENTORY
            )
            wof_by_year[d.year].append((doc_id, hdr, amount))

    # Pair inflate / writeoff groups within same year by amount symmetry
    for year in set(inf_by_year) & set(wof_by_year):
        inf_grp = inf_by_year[year]
        wof_grp = wof_by_year[year]

        inf_total = sum(a for _, _, a in inf_grp)
        wof_total = sum(a for _, _, a in wof_grp)

        if inf_total <= 0 or wof_total <= 0:
            continue
        # Amount symmetry: inflate ≈ writeoff within 20%
        ratio = min(inf_total, wof_total) / max(inf_total, wof_total)
        if ratio < 0.5:
            continue

        inf_creators = {hdr.get("created_by", "") for _, hdr, _ in inf_grp}
        wof_creators = {hdr.get("created_by", "") for _, hdr, _ in wof_grp}

        # Split-creator: inflate and writeoff posted by different users
        split_creator = not bool(inf_creators & wof_creators)

        all_ids = [d for d, _, _ in inf_grp] + [d for d, _, _ in wof_grp]
        all_paired.update(all_ids)

        confidence = 0.95 if split_creator else 0.80
        reasons = [
            f"Fictitious goods receipts Dr {INVENTORY} Cr {COGS} (GR-YYYY-* format): "
            f"{len(inf_grp)} doc(s) in {year}, total {inf_total:.0f}",
            f"Matching write-offs Dr {ASSET_IMPAIRMENT} Cr {INVENTORY} (ADJ-YYYY-* format): "
            f"{len(wof_grp)} doc(s), total {wof_total:.0f} (symmetry ratio {ratio:.2f})",
        ]
        if split_creator:
            reasons.append(
                f"Split-creator: inflated by {inf_creators}, written off by {wof_creators}"
            )

        for doc_id, _, _ in inf_grp + wof_grp:
            items.append(
                _item(
                    doc_id,
                    SCHEME_INVENTORY,
                    confidence,
                    3,
                    reasons[:],
                    entity_id=next(iter(inf_creators), None),
                    related=all_ids[:],
                    accounts=accts,
                )
            )

    # Unpaired inflate docs with GR reference (write-off not yet posted)
    for doc_id, hdr in inflate_docs.items():
        if doc_id in all_paired:
            continue
        ref = hdr.get("reference", "") or ""
        items.append(
            _item(
                doc_id,
                SCHEME_INVENTORY,
                0.80,
                2,
                [
                    f"Dr {INVENTORY} Cr {COGS} (inventory inflation pattern)",
                    "Write-off counterpart not yet observed",
                ],
                entity_id=hdr.get("created_by"),
                accounts=accts,
            )
        )

    # Unpaired writeoff docs with ADJ reference
    for doc_id, hdr in writeoff_docs.items():
        if doc_id in all_paired:
            continue
        ref = hdr.get("reference", "") or ""
        creator = hdr.get("created_by", "")
        items.append(
            _item(
                doc_id,
                SCHEME_INVENTORY,
                0.78,
                2,
                [
                    f"Dr {ASSET_IMPAIRMENT} Cr {INVENTORY} (inventory write-off pattern)",
                    f"Posted by '{creator}'",
                ],
                entity_id=creator,
                accounts=accts,
            )
        )

    return items


def detect_ap_control_bypass(data: dict) -> list[RBSuspicionItem]:
    """
    TriadBypassScheme → ap_control_bypass

    ReuseDocumentId:   Dr 401000 (AP), Cr 512000 (Bank) — same reference reused 2-4× per month
    BypassConcealment: Dr 471000 (Suspense), Cr 628000 (Misc Services) — concealment reclassification
    Oracle signals:
      - 2+ AP→Bank payments sharing the same reference within 35 days → document reuse
      - No matching invoice (Dr 6xx, Cr AP) for that reference → missing PO-GR-invoice chain
      - BypassConcealment (Dr Suspense, Cr 628000) on same reference
    Source: schemes/triad_bypass.rs + ReuseDocumentId/BypassConcealment actions
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    payment_docs: dict[str, dict] = {}
    invoice_refs: set[str] = set()
    concealment_docs: dict[str, dict] = {}

    for doc_id, lines in lines_by_doc.items():
        hdr = header_by_doc.get(doc_id)
        if not hdr:
            continue
        if je_has_debit(lines, AP_CONTROL) and je_has_credit(lines, BANK_ACCOUNT):
            payment_docs[doc_id] = hdr
        has_expense = any(
            l["gl_account"].startswith("6") and parse_amount(l["debit_amount"]) > 0
            for l in lines
        )
        if has_expense and je_has_credit(lines, AP_CONTROL):
            ref = (hdr.get("reference", "") or "").strip()
            if ref:
                invoice_refs.add(ref)
        if je_has_debit(lines, GENERAL_SUSPENSE) and je_has_credit(
            lines, MISC_SERVICES
        ):
            concealment_docs[doc_id] = hdr

    pay_by_ref: dict[str, list] = defaultdict(list)
    conc_by_ref: dict[str, list] = defaultdict(list)
    for doc_id, hdr in payment_docs.items():
        ref = (hdr.get("reference", "") or "").strip()
        if ref:
            pay_by_ref[ref].append((doc_id, hdr))
    for doc_id, hdr in concealment_docs.items():
        ref = (hdr.get("reference", "") or "").strip()
        if ref:
            conc_by_ref[ref].append((doc_id, hdr))

    flagged: set[str] = set()
    accts = [AP_CONTROL, BANK_ACCOUNT, GENERAL_SUSPENSE, MISC_SERVICES]

    for ref, pays in pay_by_ref.items():
        if len(pays) < 2:
            continue
        dated = [(d, h, parse_date(h.get("posting_date", ""))) for d, h in pays]
        dated = [(d, h, dt) for d, h, dt in dated if dt is not None]
        if len(dated) < 2:
            continue
        dates = sorted(dt for _, _, dt in dated)
        span_days = (dates[-1] - dates[0]).days
        # Bypass stage emits 2-4 reuses per calendar month (≤31 days); allow +4 day buffer
        if span_days > 35:
            continue

        creators = {h.get("created_by", "") for _, h, _ in dated}
        creator = next(iter(creators))
        same_creator = len(creators) == 1
        missing_invoice = ref not in invoice_refs
        has_concealment = ref in conc_by_ref

        if not missing_invoice and not has_concealment:
            continue  # require at least one structural oracle signal

        confidence = (
            1.0
            if (missing_invoice and same_creator)
            else (0.93 if missing_invoice else 0.82)
        )
        reasons = [
            f"Reference '{ref}' reused across {len(dated)} AP→Bank payments in {span_days} days "
            f"(triad bypass: no new invoice per payment)",
        ]
        if missing_invoice:
            reasons.append(
                f"No invoice (Dr 6xx Cr {AP_CONTROL}) for ref '{ref}' — 3-way match (PO-GR-Invoice) broken"
            )
        if same_creator:
            reasons.append(f"All payments by same creator '{creator}'")
        if has_concealment:
            conc_ids = [d for d, _ in conc_by_ref[ref]]
            reasons.append(
                f"BypassConcealment Dr {GENERAL_SUSPENSE} Cr {MISC_SERVICES} on same ref: {conc_ids}"
            )

        all_ids = [d for d, _, _ in dated] + [d for d, _ in conc_by_ref.get(ref, [])]
        for doc_id in all_ids:
            if doc_id not in flagged:
                flagged.add(doc_id)
                items.append(
                    _item(
                        doc_id,
                        SCHEME_AP_CONTROL_BYPASS,
                        confidence,
                        3,
                        reasons[:],
                        entity_id=creator,
                        related=all_ids[:],
                        accounts=accts,
                    )
                )
    return items


def detect_circular_cash_flow(data: dict) -> list[RBSuspicionItem]:
    """
    CircularCashFlowScheme → circular_cash_flow

    Cycle of 3 JEs on the SAME reference, same creator, same amount:
      Step 1 FakeCashReceipt:    Dr 512000 (Bank),    Cr 471000 (Suspense) — day 0
      Step 2 ClearARViaSuspense: Dr 471000 (Suspense), Cr 411000 (AR)      — +1 to 5d
      Step 3 ConcealAsBadDebt:   Dr 654000 (Bad Debt), Cr 512000 (Bank)    — +10 to 45d

    Reference format is not used for scoring.
    Oracle: same ref/creator/amount; Bank→Suspense→AR→BadDebt→Bank is not a normal GL path.
    Source: schemes/circular_cash_flow.rs
    """
    items = []
    header_by_doc = data["header_by_doc"]
    lines_by_doc = data["lines_by_doc"]

    fake_receipt: dict[str, dict] = {}
    ar_clearance: dict[str, dict] = {}
    bad_debt_docs: dict[str, dict] = {}

    for doc_id, lines in lines_by_doc.items():
        hdr = header_by_doc.get(doc_id)
        if not hdr:
            continue
        if je_has_debit(lines, BANK_ACCOUNT) and je_has_credit(lines, GENERAL_SUSPENSE):
            fake_receipt[doc_id] = hdr
        if je_has_debit(lines, GENERAL_SUSPENSE) and je_has_credit(lines, AR_CONTROL):
            ar_clearance[doc_id] = hdr
        if je_has_debit(lines, BAD_DEBT_EXPENSE) and je_has_credit(lines, BANK_ACCOUNT):
            bad_debt_docs[doc_id] = hdr

    def group_by_ref(docs: dict[str, dict]) -> dict[str, list]:
        idx: dict[str, list] = defaultdict(list)
        for doc_id, hdr in docs.items():
            ref = (hdr.get("reference", "") or "").strip()
            if ref:
                idx[ref].append((doc_id, hdr))
        return idx

    fr_by_ref = group_by_ref(fake_receipt)
    ar_by_ref = group_by_ref(ar_clearance)
    bd_by_ref = group_by_ref(bad_debt_docs)

    accts = [BANK_ACCOUNT, GENERAL_SUSPENSE, AR_CONTROL, BAD_DEBT_EXPENSE]
    flagged: set[str] = set()

    for ref in set(fr_by_ref) & set(ar_by_ref) & set(bd_by_ref):
        for fr_id, fr_hdr in fr_by_ref[ref]:
            fr_date = parse_date(fr_hdr.get("posting_date", ""))
            fr_creator = fr_hdr.get("created_by", "")
            if not fr_date:
                continue
            fr_amount = sum(
                parse_amount(l["debit_amount"])
                for l in lines_by_doc.get(fr_id, [])
                if l["gl_account"] == BANK_ACCOUNT
            )

            for ar_id, ar_hdr in ar_by_ref[ref]:
                ar_date = parse_date(ar_hdr.get("posting_date", ""))
                if not ar_date:
                    continue
                ar_gap = (ar_date - fr_date).days
                if not (1 <= ar_gap <= 5):
                    continue
                if ar_hdr.get("created_by", "") != fr_creator:
                    continue

                for bd_id, bd_hdr in bd_by_ref[ref]:
                    bd_date = parse_date(bd_hdr.get("posting_date", ""))
                    if not bd_date:
                        continue
                    bd_gap = (bd_date - fr_date).days
                    if not (10 <= bd_gap <= 45):
                        continue
                    if bd_hdr.get("created_by", "") != fr_creator:
                        continue

                    bd_amount = sum(
                        parse_amount(l["debit_amount"])
                        for l in lines_by_doc.get(bd_id, [])
                        if l["gl_account"] == BAD_DEBT_EXPENSE
                    )
                    if fr_amount > 0 and bd_amount > 0:
                        ratio = min(fr_amount, bd_amount) / max(fr_amount, bd_amount)
                        if ratio < 0.90:
                            continue

                    all_ids = [fr_id, ar_id, bd_id]
                    confidence = 0.92
                    reasons = [
                        f"Circular cash flow on ref '{ref}': "
                        f"Dr {BANK_ACCOUNT}/Cr {GENERAL_SUSPENSE} → "
                        f"Dr {GENERAL_SUSPENSE}/Cr {AR_CONTROL} (+{ar_gap}d) → "
                        f"Dr {BAD_DEBT_EXPENSE}/Cr {BANK_ACCOUNT} (+{bd_gap}d)",
                        f"Same creator '{fr_creator}', matching amount {fr_amount:.0f}",
                        "GL path Bank→Suspense→AR→BadDebt→Bank has no legitimate business equivalent",
                    ]

                    for doc_id in all_ids:
                        if doc_id not in flagged:
                            flagged.add(doc_id)
                            items.append(
                                _item(
                                    doc_id,
                                    SCHEME_CIRCULAR_CASH_FLOW,
                                    confidence,
                                    4,
                                    reasons[:],
                                    entity_id=fr_creator,
                                    related=all_ids[:],
                                    accounts=accts,
                                )
                            )
    return items


# ─── Run all detectors ────────────────────────────────────────────────────────

DETECTORS = [
    ("embezzlement", detect_embezzlement),  # → fictitious_ap_disbursements
    ("expense_laundering", detect_expense_laundering),  # → fictitious_ap_disbursements
    ("vendor_kickback", detect_vendor_kickback),  # → vendor_collusion
    ("related_party", detect_related_party),  # → vendor_collusion
    ("revenue_manipulation", detect_revenue_manipulation),
    ("shadow_payroll", detect_shadow_payroll),
    ("inventory_manipulation", detect_inventory_manipulation),
    # Excluded schemes (not in 5-scheme whitelist):
    # ("ap_control_bypass", detect_ap_control_bypass),
    # ("circular_cash_flow", detect_circular_cash_flow),
]


def run_all_detectors(
    data: dict,
    confidence_threshold: float = 0.0,
    quiet: bool = False,
) -> list[RBSuspicionItem]:
    all_items: list[RBSuspicionItem] = []

    for name, fn in DETECTORS:
        results = fn(data)
        kept = [r for r in results if r.confidence >= confidence_threshold]
        all_items.extend(kept)

    # Deduplicate: keep highest confidence per (doc_id, scheme_type)
    best: dict[tuple, RBSuspicionItem] = {}
    for item in all_items:
        key = (item.document_id, item.scheme_type)
        if key not in best or item.confidence > best[key].confidence:
            best[key] = item

    # Report by scheme_type (not by detector)
    if not quiet:
        items_by_scheme: dict[str, int] = {}
        for item in best.values():
            items_by_scheme[item.scheme_type] = (
                items_by_scheme.get(item.scheme_type, 0) + 1
            )

        for scheme in sorted(items_by_scheme.keys()):
            print(f"  {scheme:<36} {items_by_scheme[scheme]:>4} item(s) flagged")

    return list(best.values())


# ─── Evaluation via forensic_llm.evaluator ───────────────────────────────────


def _build_forensic_report(
    run_id: str,
    items: list[RBSuspicionItem],
) -> Any:
    """
    Build a forensic_llm.models.ForensicReport from rule-based detections.
    Imports the Pydantic model from the forensic_llm package.
    """
    from researchpkg.forensic_llm.models import (
        ForensicReport,
        SchemeType,
    )
    from researchpkg.forensic_llm.models import (
        SuspicionItem as FSuspicionItem,
    )

    scheme_map = {st.value: st for st in SchemeType}

    suspicion_list = []
    for item in items:
        st = scheme_map.get(item.scheme_type, SchemeType.UNKNOWN)
        suspicion_list.append(
            FSuspicionItem(
                document_id=item.document_id,
                entity_id=item.entity_id,
                entity_type=item.entity_type,
                scheme_type=st,
                confidence=item.confidence,
                severity=item.severity,
                rationale=item.rationale,
                supporting_evidence=item.supporting_evidence,
                related_document_ids=item.related_document_ids,
                gl_accounts=item.gl_accounts,
            )
        )

    report = ForensicReport(
        run_id=run_id,
        model=MODEL_NAME,
        task="full",
        strategy="rule_based",
        suspicion_list=suspicion_list,
    )
    report.completed_at = datetime.utcnow()
    return report


def _build_db_config(db_name: str) -> Any:
    from researchpkg.forensic_llm.config import (
        DatabaseConfig,
    )

    return DatabaseConfig(
        host=os.environ.get("FORENSIC_DB_HOST", "localhost"),
        port=int(os.environ.get("FORENSIC_DB_PORT", "5432")),
        database=db_name,
        user=os.environ.get("FORENSIC_DB_USER", "test"),
        password=os.environ.get("FORENSIC_DB_PASSWORD", ""),
    )


# ─── Output ───────────────────────────────────────────────────────────────────


def write_output(
    run_id: str,
    items: list[RBSuspicionItem],
    db_name: Optional[str],
    output_dir: str,
    quiet: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    short = run_id[:8]

    # suspicion_list.json — same format as run.py
    suspicion_path = os.path.join(output_dir, "suspicion_list.json")
    with open(suspicion_path, "w", encoding="utf-8") as f:
        json.dump([i.to_dict() for i in items], f, indent=2, default=str)

    # run_manifest.json — minimal version
    manifest = {
        "run_id": run_id,
        "run_dir": output_dir,
        "model": MODEL_NAME,
        "provider": "rule_based",
        "strategy": "rule_based",
        "task": "full",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_suspicion_items": len(items),
        "n_flagged_documents": len({i.document_id for i in items}),
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # eval_*.json — use forensic_llm.evaluator when db_name is available
    eval_result: Optional[dict] = None

    if db_name:
        try:
            from researchpkg.forensic_llm.evaluator import (
                evaluate,
            )

            report = _build_forensic_report(run_id, items)
            db_config = _build_db_config(db_name)
            result = evaluate(report, db_config)
            eval_result = json.loads(result.model_dump_json())
            eval_path = os.path.join(output_dir, f"eval_{short}.json")
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2)
            if not quiet:
                print(f"\nEval written to {eval_path}")
        except Exception as exc:
            print(
                f"WARNING: forensic_llm evaluator failed ({exc}); skipping eval_*.json",
                file=sys.stderr,
            )

    if not quiet:
        print(f"\nResults written to {output_dir}/")
        print(
            f"  suspicion_list.json  ({len(items)} items, "
            f"{len({i.document_id for i in items})} unique docs)"
        )

        if eval_result:
            ep = eval_result.get("entry_precision", 0)
            er = eval_result.get("entry_recall", 0)
            ef = eval_result.get("entry_f1", 0)
            print(f"\n── Entry-level metrics ─────────────────────────────────")
            print(f"  Precision: {ep:.4f}  Recall: {er:.4f}  F1: {ef:.4f}")
            print(
                f"\n── Per-scheme metrics ──────────────────────────────────────────────────"
            )
            hdr_line = f"{'Scheme':<28}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'GT':>5}"
            print(hdr_line)
            print("─" * len(hdr_line))
            for sm in eval_result.get("per_scheme", []):
                tp = sm.get("n_correct", 0)
                fp = max(0, sm.get("n_flagged", 0) - tp)
                gt = sm.get("n_true", 0)
                print(
                    f"{sm['scheme_type']:<28}  {sm['precision']:>6.3f}  {sm['recall']:>6.3f}"
                    f"  {sm['f1']:>6.3f}  {tp:>5}  {fp:>5}  {gt:>5}"
                )
            macro = sum(sm["f1"] for sm in eval_result.get("per_scheme", [])) / max(
                1, len(eval_result.get("per_scheme", []))
            )
            print(f"\nMacro F1 : {macro:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rule-based fraud detection baseline for the LLM forensic benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--data-dir",
        "-d",
        default=None,
        help="Directory with je_header.csv, je_line.csv, anomaly_labels.csv, etc.",
    )
    src.add_argument(
        "--db-name",
        default=None,
        metavar="DB_NAME",
        help="Postgres labelled DB name (e.g. datasynth_forensic__energy). "
        "Env: FORENSIC_DB_HOST/PORT/USER/PASSWORD.",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for results (default: <cwd>/rule_based_<run_id[:8]>)",
    )
    p.add_argument(
        "--confidence-threshold",
        "-t",
        type=float,
        default=0.0,
        help="Minimum confidence to include a SuspicionItem (default: 0.0)",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress human-readable output",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override auto-generated run UUID",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or str(uuid.uuid4())

    if args.db_name:
        if not args.quiet:
            print(f"Loading data from Postgres DB '{args.db_name}' …")
        data = load_data_sql(args.db_name)
    else:
        data_dir = args.data_dir or "output/forensic_llm"
        if not os.path.isdir(data_dir):
            print(f"ERROR: data directory '{data_dir}' not found.", file=sys.stderr)
            sys.exit(1)
        if not args.quiet:
            print(f"Loading data from {data_dir} …")
        data = load_data(data_dir)

    if not args.quiet:
        n_docs = len(data["header_by_doc"])
        n_labels = sum(len(v) for v in data["gt_by_label"].values())
        print(f"Loaded {n_docs} documents, {n_labels} injected label entries")
        print(f"\nRunning rule-based detectors:")

    items = run_all_detectors(data, args.confidence_threshold, args.quiet)

    out_dir = args.output or os.path.join(os.getcwd(), f"rule_based_{run_id[:8]}")
    write_output(run_id, items, args.db_name, out_dir, args.quiet)


if __name__ == "__main__":
    main()
