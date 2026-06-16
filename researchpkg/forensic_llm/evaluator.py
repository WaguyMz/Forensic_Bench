"""
Offline evaluation of a ForensicReport against the anomaly_labels table.

Evaluation levels
-----------------
1. **Entry-level** – exact document_id match between the suspicion list
   and anomaly_labels.document_id.  Computes precision, recall, F1.

2. **Score-level** – treat `suspicion_item.confidence` as a binary
   classifier score and compute ROC-AUC / PR-AUC.

3. **Per-scheme** – breakdown of metrics by anomaly_type / scheme_type.

4. **Scheme-level detection** – a fraud scheme (which may span multiple
   document_ids) is "detected" if at least one of its constituent
   document_ids appears in the suspicion list with confidence >= threshold.

Notes
-----
- `anomaly_labels.document_id` is TEXT; `je_header.document_id` is UUID.
  We normalise both to lowercase stripped strings for matching.
- Labels with `is_injected = FALSE` are synthetic data-quality issues
  introduced by the generator; they count as true positives only for
  "Process" and "Statistical" anomaly_types.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from .models import (
    EvaluationResult,
    ForensicReport,
    SchemeEvalMetrics,
    SchemeEvalSummary,
    SchemeMetrics,
    SchemeType,
    SuspicionItem,
)

# Finish reasons for which run.py auto-runs evaluate_and_save (--evaluate).
EVAL_ELIGIBLE_FINISH_REASONS = frozenset(
    {"completed", "budget_exhausted", "orchestrator_complete"}
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _fetch_labels(db_config) -> List[Dict[str, Any]]:
    """
    Fetch all rows from anomaly_labels, normalising document_id to a
    consistent string form.

    Connects to the non-public database (derived from db_config.database by
    removing '_public' from the name) to fetch labels. This prevents the agent
    from cheating since the agent connects to the sanitized public database.
    """
    import psycopg2

    # Derive non-public database name by removing '_public' suffix.
    # E.g., 'datasynth_forensic_public_energy' -> 'datasynth_forensic_energy'
    non_public_db = db_config.database.replace("_public", "")

    # Build DSN for the non-public database (same host/port/user/password, different DB)
    use_socket = (
        db_config.host in ("localhost", "127.0.0.1", "::1") and not db_config.password
    )
    if use_socket:
        dsn = f"dbname={non_public_db} user={db_config.user}"
    elif db_config.password:
        dsn = (
            f"host={db_config.host} port={db_config.port} dbname={non_public_db} "
            f"user={db_config.user} password={db_config.password}"
        )
    else:
        dsn = (
            f"host={db_config.host} port={db_config.port} dbname={non_public_db} "
            f"user={db_config.user}"
        )

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            base_cols = [
                "anomaly_id",
                "anomaly_type",
                "document_id",
                "document_type",
                "company_code",
                "anomaly_date",
                "severity",
                "description",
                "monetary_impact",
                "is_injected",
            ]

            def _run_select(
                include_metadata: bool,
            ) -> tuple[list[str], list[tuple[Any, ...]]]:
                cols = list(base_cols)
                if include_metadata:
                    cols.append("metadata")
                cur.execute(f"SELECT {', '.join(cols)} FROM anomaly_labels")
                out_cols = [d[0] for d in cur.description]
                out_rows = cur.fetchall()
                return out_cols, out_rows

            try:
                cols, rows = _run_select(include_metadata=True)
            except psycopg2.Error as exc:
                # Some deployments don't have the `metadata` column on anomaly_labels.
                # PostgreSQL marks the transaction as aborted after any error; we must
                # rollback before issuing the fallback query or psycopg2 raises
                # "current transaction is aborted, commands ignored...".
                conn.rollback()
                msg = str(getattr(exc, "pgerror", None) or exc).lower()
                if "column" in msg and "metadata" in msg and "does not exist" in msg:
                    cols, rows = _run_select(include_metadata=False)
                    cols.append("metadata")
                    rows = [tuple(r) + (None,) for r in rows]
                else:
                    raise
    finally:
        conn.close()

    return [dict(zip(cols, row)) for row in rows]


def _normalise_id(doc_id: Optional[str]) -> str:
    if not doc_id:
        return ""
    return str(doc_id).lower().strip().replace("-", "")


# Canonical 5-scheme taxonomy (matches SchemeType in models.py and the prompts).
# Maps DB anomaly_type (e.g. fraud(ghostemployee), fraud(invoicemanipulation))
# to the canonical scheme used by the model.
ANOMALY_TYPE_TO_CANONICAL_SCHEME: Dict[str, str] = {
    "fraud(ghostemployee)": "shadow_payroll",
    "fraud(invoicemanipulation)": "ap_control_bypass",
    "fraud(revenuemanipulation)": "revenue_manipulation",
    "fraud(kickback)": "vendor_collusion",
    "fraud(inventorytheft)": "inventory_manipulation",
    "fraud(fictitiousvendor)": "fictitious_ap_disbursements",
    "fraud(assetmisappropriation)": "fictitious_ap_disbursements",  # embezzlement + expense laundering merged
    "fraud(suspenseaccountabuse)": "circular_cash_flow",
    "fraud(fictitioustransaction)": "unknown",
}


def _canonical_scheme_from_anomaly_type(anomaly_type: str) -> str:
    """Map DB anomaly_type to the 8-scheme canonical taxonomy for evaluation."""
    at = (anomaly_type or "").replace(" ", "_").lower().strip()
    return ANOMALY_TYPE_TO_CANONICAL_SCHEME.get(at, at)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    report: ForensicReport,
    db_config,
    confidence_threshold: float = 0.5,
    labels: Optional[List[Dict[str, Any]]] = None,
    budget_fraction_hint: Optional[float] = None,
) -> EvaluationResult:
    """
    Score a ForensicReport against ground-truth anomaly labels.

    Parameters
    ----------
    report               : completed ForensicReport
    db_config            : DatabaseConfig
    confidence_threshold : retained for compatibility but ignored when
                           confidence is binary-fixed at 1.0 for all items.
    labels               : pre-fetched label rows (avoids a second DB round-trip
                           when called from evaluate_and_save).

    Returns
    -------
    EvaluationResult with all metrics populated.
    """
    if labels is None:
        try:
            labels = _fetch_labels(db_config)
        except Exception as exc:
            log.error("Could not fetch anomaly labels: %s", exc)
            return EvaluationResult(
                run_id=report.run_id,
                model=report.model,
                task=report.task,
            )

    if not labels:
        log.warning("anomaly_labels table is empty – returning zero metrics")
        return EvaluationResult(
            run_id=report.run_id,
            model=report.model,
            task=report.task,
        )

    def _label_document_ids(lbl: Dict[str, Any]) -> List[str]:
        """
        Expand a label row into the JE document_id(s) it refers to.

        Ground-truth labels typically use je_header.document_id, but some
        generators may store a scheme-level identifier (e.g. "scheme-...") in
        anomaly_labels.document_id and keep the underlying JE IDs in metadata.
        """
        doc = str(lbl.get("document_id") or "").strip()
        meta = lbl.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = None

        # If the label doc_id is already a UUID-like JE id, use it.
        if doc and not doc.lower().startswith("scheme-"):
            return [doc]

        if isinstance(meta, dict):
            for k in ("document_ids", "je_document_ids", "related_document_ids"):
                v = meta.get(k)
                if isinstance(v, list) and v:
                    return [str(x) for x in v if x]
            # Some generators store a single id.
            for k in ("document_id", "je_document_id"):
                v = meta.get(k)
                if v:
                    return [str(v)]

        # Fall back to whatever was in document_id (even if scheme-level).
        return [doc] if doc else []

    # Build ground-truth set of JE ids
    true_doc_ids: Set[str] = set()
    for lbl in labels:
        for did in _label_document_ids(lbl):
            nid = _normalise_id(did)
            if nid:
                true_doc_ids.add(nid)
    # Exclude empty strings
    true_doc_ids.discard("")

    # Build per-scheme ground-truth: canonical scheme_type → set of JE document_ids
    # Map DB anomaly_type (e.g. fraud(ghostemployee)) to the 8-scheme canonical taxonomy
    # so it matches model predictions (fictitious_ap_disbursements, vendor_collusion, etc.)
    scheme_to_true: Dict[str, Set[str]] = {}
    for lbl in labels:
        canonical = _canonical_scheme_from_anomaly_type(
            lbl.get("anomaly_type") or "Unknown"
        )
        for did in _label_document_ids(lbl):
            nid = _normalise_id(did)
            if nid:
                scheme_to_true.setdefault(canonical, set()).add(nid)

    # Build flagged sets from the suspicion list.
    # Binary detection: every suspicion is treated as "flagged".
    flagged_items = list(report.suspicion_list)
    flagged_doc_ids: Set[str] = {
        _normalise_id(s.document_id) for s in flagged_items if s.document_id
    }
    flagged_doc_ids.discard("")

    # Build a score map for AUC: doc_id → max confidence in suspicion list
    score_map: Dict[str, float] = {}
    for s in report.suspicion_list:
        if s.document_id:
            nid = _normalise_id(s.document_id)
            if nid:
                score_map[nid] = max(score_map.get(nid, 0.0), s.confidence)

    # -----------------------------------------------------------------------
    # Entry-level binary metrics
    # -----------------------------------------------------------------------
    n_true = len(true_doc_ids)
    n_flagged = len(flagged_doc_ids)
    n_correct = len(flagged_doc_ids & true_doc_ids)

    precision = n_correct / n_flagged if n_flagged > 0 else 0.0
    recall = n_correct / n_true if n_true > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # Accuracy over the evaluation set (true ∪ flagged); unit = JE.
    eval_set_size = n_true + n_flagged - n_correct
    accuracy = n_correct / eval_set_size if eval_set_size > 0 else 0.0

    # -----------------------------------------------------------------------
    # Score-level AUC (over all unique document_ids appearing in labels or
    # the suspicion list)
    # -----------------------------------------------------------------------
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    all_doc_ids = list(true_doc_ids | set(score_map.keys()))
    if len(all_doc_ids) >= 2 and score_map:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            y_true = [1 if did in true_doc_ids else 0 for did in all_doc_ids]
            y_score = [score_map.get(did, 0.0) for did in all_doc_ids]

            # Guard: degenerate score distribution (all scores identical) yields
            # meaningless AUC values (e.g. PR-AUC = 0.911 with P=R=0).
            # This happens when confidence is hardcoded to 1.0 for all items and
            # there are zero true positives — the curve is undefined.
            if len(set(y_score)) == 1:
                log.warning(
                    "Skipping AUC: all prediction scores are identical (%.4f). "
                    "ROC-AUC and PR-AUC are undefined for a degenerate score "
                    "distribution — likely caused by confidence=1.0 on all items "
                    "with zero true positives.",
                    y_score[0],
                )
            elif sum(y_true) > 0 and sum(y_true) < len(y_true):
                roc_auc = float(roc_auc_score(y_true, y_score))
                pr_auc = float(average_precision_score(y_true, y_score))
        except Exception as exc:
            log.warning("AUC computation failed: %s", exc)

    # -----------------------------------------------------------------------
    # Per-scheme metrics
    # -----------------------------------------------------------------------
    # Map scheme_type → flagged doc_ids
    scheme_to_flagged: Dict[str, Set[str]] = {}
    for s in flagged_items:
        st = s.scheme_type.value if s.scheme_type else "unknown"
        if s.document_id:
            nid = _normalise_id(s.document_id)
            if nid:
                scheme_to_flagged.setdefault(st, set()).add(nid)

    # -----------------------------------------------------------------------
    # Entry-level confusion matrix (predicted scheme_type vs true scheme_type)
    # unit: journal-entry document_id
    # -----------------------------------------------------------------------
    core_label_order: List[str] = [
        SchemeType.SHADOW_PAYROLL.value,
        SchemeType.FICTITIOUS_AP_DISBURSEMENTS.value,
        SchemeType.REVENUE_MANIPULATION.value,
        SchemeType.VENDOR_COLLUSION.value,
        SchemeType.INVENTORY_MANIPULATION.value,
    ]

    # 8 fraud classes only (no negative class in confusion matrix).
    # Filter out the "negative" row/column from the matrix before plotting.
    # "unknown" predictions and docs with no true label both map to "negative".
    # Docs with multiple true labels are assigned to the first matching scheme
    # in label order.
    NEGATIVE = "negative"
    label_order: List[str] = core_label_order + [NEGATIVE]
    label_set = set(label_order)

    # doc_id -> set of true scheme types (after canonicalization)
    doc_to_true_schemes: Dict[str, Set[str]] = {}
    for true_scheme, doc_ids in scheme_to_true.items():
        for did in doc_ids:
            doc_to_true_schemes.setdefault(did, set()).add(true_scheme)

    # doc_id -> (predicted_scheme_type, confidence), deduping by max confidence
    pred_doc_to_scheme: Dict[str, Dict[str, Any]] = {}
    for s in flagged_items:
        if not s.document_id:
            continue
        did = _normalise_id(s.document_id)
        if not did:
            continue
        pred_scheme = s.scheme_type.value if s.scheme_type else NEGATIVE
        if pred_scheme not in label_set:
            pred_scheme = NEGATIVE  # unknown scheme → treat as negative
        conf = float(getattr(s, "confidence", 0.0) or 0.0)
        cur = pred_doc_to_scheme.get(did)
        if cur is None or float(cur.get("confidence", 0.0) or 0.0) < conf:
            pred_doc_to_scheme[did] = {
                "scheme": pred_scheme,
                "confidence": conf,
            }

    row_labels = list(label_order)
    col_labels = list(label_order)
    # Initialize full matrix (rows=pred, cols=true)
    counts: List[List[int]] = [[0 for _ in col_labels] for _ in row_labels]

    # Score all flagged docs + all true fraud docs (to capture false negatives).
    all_true_doc_ids: Set[str] = set(doc_to_true_schemes.keys())
    docs_to_score = all_true_doc_ids | set(pred_doc_to_scheme.keys())
    n_scored_documents = len(docs_to_score)

    for did in docs_to_score:
        pred_info = pred_doc_to_scheme.get(did)
        pred_scheme = pred_info["scheme"] if pred_info is not None else NEGATIVE

        true_schemes = doc_to_true_schemes.get(did, set())
        if not true_schemes:
            true_label = NEGATIVE  # flagged but no true fraud label → FP
        elif len(true_schemes) == 1:
            true_label = next(iter(true_schemes))
        else:
            # multi-label: assign to first scheme in label order
            true_label = next(
                (lbl for lbl in core_label_order if lbl in true_schemes),
                next(iter(true_schemes)),
            )

        if pred_scheme not in label_set:
            pred_scheme = NEGATIVE
        if true_label not in label_set:
            true_label = NEGATIVE

        i = row_labels.index(pred_scheme)
        j = col_labels.index(true_label)
        counts[i][j] += 1

    # Filter out the "negative" class for the confusion matrix plot
    neg_index = row_labels.index(NEGATIVE) if NEGATIVE in row_labels else -1
    filtered_row_labels = [l for l in row_labels if l != NEGATIVE]
    filtered_col_labels = [l for l in col_labels if l != NEGATIVE]
    filtered_counts = [
        [counts[i][j] for j, cl in enumerate(col_labels) if cl != NEGATIVE]
        for i, rl in enumerate(row_labels)
        if rl != NEGATIVE
    ]

    # Convert to a compact "nonzero cells" view for human reading.
    nonzero_cells: List[Dict[str, Any]] = []
    for i, pr in enumerate(row_labels):
        for j, tr in enumerate(col_labels):
            c = counts[i][j]
            if c:
                nonzero_cells.append({"pred": pr, "true": tr, "count": c})

    confusion_matrix: Dict[str, Any] = {
        "row_labels": filtered_row_labels,
        "col_labels": filtered_col_labels,
        "matrix": filtered_counts,
        "n_scored_documents": n_scored_documents,
        "nonzero_cells": nonzero_cells,
    }

    per_scheme: List[SchemeMetrics] = []
    all_scheme_keys = set(scheme_to_true.keys()) | set(scheme_to_flagged.keys())
    for sk in sorted(all_scheme_keys):
        true_set = scheme_to_true.get(sk, set())
        flag_set = scheme_to_flagged.get(sk, set())
        n_t = len(true_set)
        n_f = len(flag_set)
        n_c = len(true_set & flag_set)
        p = n_c / n_f if n_f > 0 else 0.0
        r = n_c / n_t if n_t > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        set_size = n_t + n_f - n_c
        acc_s = n_c / set_size if set_size > 0 else 0.0
        per_scheme.append(
            SchemeMetrics(
                scheme_type=sk,
                precision=round(p, 4),
                recall=round(r, 4),
                f1=round(f, 4),
                accuracy=round(acc_s, 4),
                n_true=n_t,
                n_flagged=n_f,
                n_correct=n_c,
            )
        )

    # -----------------------------------------------------------------------
    # Confusion-style counts
    # -----------------------------------------------------------------------
    true_positives = n_correct
    false_positives = max(0, n_flagged - n_correct)
    false_negatives = max(0, n_true - n_correct)

    # False positive rate = FP / flagged (false discovery rate; 0 = perfect).
    false_positive_rate = round(false_positives / max(1, n_flagged), 4)

    # -----------------------------------------------------------------------
    # Human-readable performance summary
    # -----------------------------------------------------------------------
    summary_lines: List[str] = []
    summary_lines.append(
        "Overall performance (JE-level, binary detection): "
        "TP=%d, FP=%d, FN=%d, FPR=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, accuracy=%.4f, "
        "roc_auc=%s, pr_auc=%s."
        % (
            true_positives,
            false_positives,
            false_negatives,
            false_positive_rate,
            precision,
            recall,
            f1,
            accuracy,
            f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
            f"{pr_auc:.4f}" if pr_auc is not None else "N/A",
        )
    )
    if per_scheme:
        summary_lines.append("Per-scheme breakdown:")
        for sm in per_scheme:
            tp_s = sm.n_correct
            fp_s = max(0, sm.n_flagged - sm.n_correct)
            fn_s = max(0, sm.n_true - sm.n_correct)
            summary_lines.append(
                f"- {sm.scheme_type}: "
                f"TP={tp_s}, FP={fp_s}, FN={fn_s}, "
                f"precision={sm.precision:.4f}, recall={sm.recall:.4f}, "
                f"f1={sm.f1:.4f}, accuracy={sm.accuracy:.4f}, "
                f"n_true={sm.n_true}, n_flagged={sm.n_flagged}."
            )

    # -----------------------------------------------------------------------
    # Scheme-level evaluation
    # -----------------------------------------------------------------------
    scheme_eval = evaluate_schemes(report, labels, confidence_threshold)

    if scheme_eval.n_true_schemes > 0:
        summary_lines.append(
            f"Scheme-level: SDR={scheme_eval.scheme_detection_rate:.4f}, "
            f"Precision={scheme_eval.scheme_precision:.4f}, "
            f"F1={scheme_eval.scheme_f1:.4f}, "
            f"Mean Coverage={scheme_eval.mean_coverage:.4f}, "
            f"Perpetrator ID Rate={scheme_eval.perpetrator_identification_rate:.4f} "
            f"({scheme_eval.n_detected}/{scheme_eval.n_true_schemes} schemes detected)."
        )

    result = EvaluationResult(
        run_id=report.run_id,
        model=report.model,
        task=report.task,
        entry_precision=round(precision, 4),
        entry_recall=round(recall, 4),
        entry_f1=round(f1, 4),
        entry_accuracy=round(accuracy, 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        pr_auc=round(pr_auc, 4) if pr_auc is not None else None,
        per_scheme=per_scheme,
        scheme_eval=scheme_eval,
        confusion_matrix=confusion_matrix,
        n_true_anomalies=n_true,
        n_flagged=n_flagged,
        n_correct=n_correct,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        false_positive_rate=false_positive_rate,
        n_steps=report.steps_taken,
        total_tokens=report.total_tokens,
        budget_fraction=budget_fraction_hint
        if budget_fraction_hint is not None
        else 1.0,
        performance_summary="\n".join(summary_lines),
    )

    # Pretty-print summary
    log.info(
        "Evaluation | P=%.3f R=%.3f F1=%.3f FPR=%.3f | ROC-AUC=%s | PR-AUC=%s | "
        "flagged=%d true=%d correct=%d | TP=%d FP=%d FN=%d",
        precision,
        recall,
        f1,
        false_positive_rate,
        f"{roc_auc:.3f}" if roc_auc is not None else "N/A",
        f"{pr_auc:.3f}" if pr_auc is not None else "N/A",
        n_flagged,
        n_true,
        n_correct,
        true_positives,
        false_positives,
        false_negatives,
    )
    return result


# ---------------------------------------------------------------------------
# Scheme-level evaluation
# ---------------------------------------------------------------------------


def evaluate_schemes(
    report: ForensicReport,
    labels: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
    coverage_threshold: float = 0.3,
) -> SchemeEvalSummary:
    """
    Evaluate at the scheme level: a ground-truth scheme (perpetrator + type)
    is detected if the agent flags >= coverage_threshold of its document_ids
    with confidence >= confidence_threshold.

    Ground-truth schemes are built by grouping anomaly_labels on
    (metadata->perpetrator_id, anomaly_type).
    """
    # Build ground-truth schemes from labels
    gt_schemes: Dict[str, Dict[str, Any]] = {}
    for lbl in labels:
        doc_id = _normalise_id(lbl.get("document_id"))
        if not doc_id:
            continue
        atype = (lbl.get("anomaly_type") or "Unknown").replace(" ", "_").lower()
        canonical = _canonical_scheme_from_anomaly_type(
            lbl.get("anomaly_type") or "Unknown"
        )

        perp = ""
        if isinstance(lbl.get("metadata"), dict):
            perp = lbl["metadata"].get("perpetrator_id", "")
        elif isinstance(lbl.get("metadata"), str):
            try:
                meta = json.loads(lbl["metadata"])
                perp = meta.get("perpetrator_id", "")
            except (json.JSONDecodeError, TypeError):
                pass

        scheme_key = f"{perp or '_none_'}::{atype}"
        if scheme_key not in gt_schemes:
            gt_schemes[scheme_key] = {
                "scheme_type": canonical,
                "perpetrator_id": perp or "",
                "doc_ids": set(),
            }
        gt_schemes[scheme_key]["doc_ids"].add(doc_id)

    if not gt_schemes:
        return SchemeEvalSummary()

    # Build predicted set (JE-level).
    flagged_doc_ids: Set[str] = {
        _normalise_id(s.document_id)
        for s in report.suspicion_list
        if s.document_id and (s.confidence or 0.0) >= confidence_threshold
    }
    flagged_doc_ids.discard("")

    # Build predicted entity → scheme mapping for perpetrator identification
    predicted_perps: Dict[str, Set[str]] = {}
    for s in report.suspicion_list:
        if s.entity_id:
            st = s.scheme_type.value if s.scheme_type else "unknown"
            predicted_perps.setdefault(st, set()).add(s.entity_id.strip().lower())

    per_scheme: List[SchemeEvalMetrics] = []
    n_detected = 0
    n_perp_identified = 0
    total_coverage = 0.0

    for scheme_key, gt in gt_schemes.items():
        gt_docs = gt["doc_ids"]
        flagged_in_scheme = gt_docs & flagged_doc_ids
        coverage = len(flagged_in_scheme) / len(gt_docs) if gt_docs else 0.0
        detected = coverage >= coverage_threshold and len(flagged_in_scheme) >= 1

        perp_found = False
        gt_perp = gt["perpetrator_id"].strip().lower()
        if gt_perp and detected:
            st = gt["scheme_type"]
            if gt_perp in predicted_perps.get(st, set()):
                perp_found = True

        if detected:
            n_detected += 1
            total_coverage += coverage
        if perp_found:
            n_perp_identified += 1

        per_scheme.append(
            SchemeEvalMetrics(
                scheme_id=scheme_key,
                scheme_type=gt["scheme_type"],
                perpetrator_id=gt["perpetrator_id"],
                n_true_docs=len(gt_docs),
                n_flagged_docs=len(flagged_in_scheme),
                coverage_ratio=round(coverage, 4),
                detected=detected,
                perpetrator_identified=perp_found,
            )
        )

    # Scheme precision: of all scheme-clusters the agent reported, how many
    # correspond to a real ground-truth scheme.  We match predicted schemes
    # (from SchemeReport) to ground-truth schemes by (entity_id, scheme_type).
    n_predicted_schemes = len(report.scheme_reports) if report.scheme_reports else 0
    matched_predicted = 0
    if report.scheme_reports:
        gt_perp_types = {
            (gt["perpetrator_id"].strip().lower(), gt["scheme_type"])
            for gt in gt_schemes.values()
        }
        for sr in report.scheme_reports:
            pred_key = (
                (sr.perpetrator_id or "").strip().lower(),
                sr.scheme_type.value if sr.scheme_type else "",
            )
            if pred_key in gt_perp_types:
                matched_predicted += 1

    n_true = len(gt_schemes)
    sdr = n_detected / n_true if n_true > 0 else 0.0
    s_prec = matched_predicted / n_predicted_schemes if n_predicted_schemes > 0 else 0.0
    s_f1 = 2 * sdr * s_prec / (sdr + s_prec) if (sdr + s_prec) > 0 else 0.0
    mean_cov = total_coverage / n_detected if n_detected > 0 else 0.0
    perp_rate = n_perp_identified / n_true if n_true > 0 else 0.0

    return SchemeEvalSummary(
        n_true_schemes=n_true,
        n_detected=n_detected,
        n_predicted_schemes=n_predicted_schemes,
        scheme_detection_rate=round(sdr, 4),
        scheme_precision=round(s_prec, 4),
        scheme_f1=round(s_f1, 4),
        mean_coverage=round(mean_cov, 4),
        perpetrator_identification_rate=round(perp_rate, 4),
        per_scheme=per_scheme,
    )


# ---------------------------------------------------------------------------
# Convenience: evaluate + save
# ---------------------------------------------------------------------------


def _hydrate_suspicions_from_detections_file(
    report: ForensicReport,
    output_dir: "pathlib.Path",
) -> None:
    """If ``report.suspicion_list`` is empty, load JE flags from ``detections.json``."""
    import pathlib

    if report.suspicion_list:
        return
    det_path = pathlib.Path(output_dir) / "detections.json"
    if not det_path.is_file():
        return
    try:
        payload = json.loads(det_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read %s for evaluation hydration: %s", det_path, exc)
        return
    items: List[SuspicionItem] = []
    for det in payload.get("detections", []):
        doc_id = det.get("document_id")
        if not doc_id:
            continue
        scheme_str = str(det.get("scheme_id", "unknown")).strip().lower()
        try:
            scheme_type = SchemeType(scheme_str)
        except ValueError:
            scheme_type = SchemeType.UNKNOWN
        items.append(
            SuspicionItem(
                document_id=str(doc_id),
                scheme_type=scheme_type,
                confidence=1.0,
            )
        )
    if items:
        report.suspicion_list = items
        log.info(
            "Hydrated %d suspicion(s) from %s for evaluation",
            len(items),
            det_path,
        )


def evaluate_and_save(
    report: ForensicReport,
    db_config,
    output_dir: str = "./forensic_output",
    confidence_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate a report, save metrics as JSON, and emit a CSV of detections."""
    import csv
    import pathlib

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _hydrate_suspicions_from_detections_file(report, out)

    # Fetch labels once; reuse for both evaluate() and the detection CSV.
    try:
        labels = _fetch_labels(db_config)
    except Exception as exc:
        log.error("Could not fetch anomaly labels: %s", exc)
        labels = []

    # Read budget_fraction from budget_summary.json.
    # Prefer budget-counted tokens (prompt + separately reported reasoning), then
    # fall back to older billed-token fields for backward compatibility.
    # This is the only place we have access to the configured token cap.
    budget_fraction_hint: Optional[float] = None
    bs_path = pathlib.Path(output_dir) / "budget_summary.json"
    if bs_path.exists():
        try:
            bs = json.loads(bs_path.read_text(encoding="utf-8"))
            budget_counted = (
                bs.get("grand_total_budget_counted_tokens")
                or bs.get("budget_counted_tokens")
                or 0
            )
            grand_billed = (
                bs.get("grand_total_billed_tokens")
                or bs.get("total_billed_tokens")
                or 0
            )
            configured_max = (
                bs.get("configured_global_max_tokens")
                or bs.get("configured_max_tokens")
                or 0
            )
            numerator = budget_counted or grand_billed
            if numerator and configured_max:
                budget_fraction_hint = round(numerator / configured_max, 4)
        except Exception:
            pass

    # Core evaluation (metrics only) — pass pre-fetched labels to avoid a second round-trip.
    result = evaluate(
        report,
        db_config,
        confidence_threshold,
        labels=labels,
        budget_fraction_hint=budget_fraction_hint,
    )

    # Save metrics JSON with the detailed performance summary.
    eval_path = out / f"eval_{report.run_id[:8]}.json"
    eval_path.write_text(
        json.dumps(result.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Evaluation saved to %s", eval_path)

    # -----------------------------------------------------------------------
    # Confusion matrix plot (scheme_type)
    # -----------------------------------------------------------------------
    try:
        if result.confusion_matrix:
            import matplotlib

            matplotlib.use("Agg")  # ensure headless rendering
            import matplotlib.pyplot as plt

            cm = result.confusion_matrix
            row_labels = cm.get("row_labels", [])
            col_labels = cm.get("col_labels", [])
            matrix = cm.get("matrix", [])

            # Defensive: ensure we have a 2D list.
            if (
                isinstance(matrix, list)
                and matrix
                and all(isinstance(r, list) for r in matrix)
            ):
                import numpy as np

                arr = np.array(matrix, dtype=float)
                vmin = 0.0
                vmax = float(np.max(arr)) if np.max(arr) > 0 else 1.0

                fig, ax = plt.subplots(figsize=(12, 10))
                # Use 'viridis' colormap for better perceptual uniformity
                im = ax.imshow(
                    arr, interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax
                )
                ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set(
                    xticks=np.arange(len(col_labels)),
                    yticks=np.arange(len(row_labels)),
                    xticklabels=col_labels,
                    yticklabels=row_labels,
                    ylabel="Predicted scheme_type",
                    xlabel="True scheme_type",
                    title="Scheme-type Confusion Matrix (JE-level document_id)",
                )

                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

                # Annotate only nonzero cells to avoid unreadable clutter.
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        v = arr[i, j]
                        if v != 0:
                            ax.text(j, i, int(v), ha="center", va="center", fontsize=8)

                fig.tight_layout()
                fig_path = out / f"confusion_matrix_scheme_type_{report.run_id[:8]}.png"
                fig.savefig(fig_path, dpi=200)
                plt.close(fig)
                log.info("Confusion matrix plot saved to %s", fig_path)
    except Exception as exc:
        # Plotting must never break evaluation saving.
        log.warning("Could not generate confusion matrix plot: %s", exc)

    # Classification report (scores per scheme + global) for scripts/dashboards
    report_path = out / f"classification_report_{report.run_id[:8]}.txt"
    lines = [
        "# Classification report (unit = JE)",
        "# Ground truth is normalized to the 10-scheme taxonomy (no raw DB types like assetmisappropriation).",
        f"run_id: {report.run_id[:8]}  model: {report.model}  task: {report.task}",
        f"confidence_threshold: {confidence_threshold}  (binary detection: all suspicions treated as flagged)",
        "",
        "## Global (JE-level)",
        f"  accuracy            {result.entry_accuracy:.4f}",
        f"  precision           {result.entry_precision:.4f}",
        f"  recall              {result.entry_recall:.4f}",
        f"  f1                  {result.entry_f1:.4f}",
        f"  false_positive_rate {result.false_positive_rate:.4f}  (FP/flagged; 0=perfect)",
        f"  support    TP={result.n_correct}  FP={result.false_positives}  FN={result.false_negatives}  (n_true={result.n_true_anomalies}, n_flagged={result.n_flagged})",
        "",
        "## Per scheme",
    ]
    for sm in result.per_scheme:
        lines.append(
            f"  {sm.scheme_type}: accuracy={sm.accuracy:.4f} precision={sm.precision:.4f} "
            f"recall={sm.recall:.4f} f1={sm.f1:.4f} (n_true={sm.n_true}, n_flagged={sm.n_flagged}, n_correct={sm.n_correct})"
        )
    if result.scheme_eval and result.scheme_eval.n_true_schemes > 0:
        se = result.scheme_eval
        lines.extend(
            [
                "",
                "## Scheme-level summary (perpetrator+type)",
                f"  n_true_schemes           {se.n_true_schemes}",
                f"  n_detected_schemes       {se.n_detected}",
                f"  n_predicted_schemes      {se.n_predicted_schemes}",
                f"  scheme_detection_rate    {se.scheme_detection_rate:.4f}",
                f"  scheme_precision         {se.scheme_precision:.4f}",
                f"  scheme_f1                {se.scheme_f1:.4f}",
                f"  mean_coverage_detected   {se.mean_coverage:.4f}",
                f"  perpetrator_id_rate      {se.perpetrator_identification_rate:.4f}",
                "",
                "## Scheme-level per-instance metrics",
                "# Each row is a ground-truth scheme (perpetrator + canonical scheme_type).",
                "# coverage_ratio = flagged_docs / true_docs; detected=True iff coverage_ratio >= coverage_threshold",
                "scheme_id,scheme_type,perpetrator_id,n_true_docs,n_flagged_docs,coverage_ratio,detected,perpetrator_identified",
            ]
        )
        for sm in se.per_scheme:
            lines.append(
                f"{sm.scheme_id},{sm.scheme_type},{sm.perpetrator_id},"
                f"{sm.n_true_docs},{sm.n_flagged_docs},{sm.coverage_ratio:.4f},"
                f"{int(sm.detected)},{int(sm.perpetrator_identified)}"
            )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Classification report saved to %s", report_path)

    # -----------------------------------------------------------------------
    # Build and save CSV of detections (reuse labels already fetched above)
    # -----------------------------------------------------------------------
    if not labels:
        log.warning("No anomaly labels available; skipping detection CSV")
        return result

    label_index: Dict[str, Dict[str, Any]] = {}
    for lbl in labels:
        nid = _normalise_id(lbl.get("document_id"))
        if nid:
            label_index[nid] = lbl

    # Convenience sets for FN rows.
    predicted_doc_ids: Set[str] = set()

    detections_path = out / f"detections_{report.run_id[:8]}.csv"
    with detections_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "run_id",
                "model",
                "task",
                "document_id",
                "entity_id",
                "entity_type",
                "predicted_scheme_type",
                "confidence",
                "severity",
                "period",
                "gl_accounts",
                "flagged_binary",
                "true_label_document",
                "true_anomaly_id",
                "true_anomaly_type",
                "true_severity",
                "true_monetary_impact",
                "true_is_injected",
                "is_true_positive",
                "is_false_positive",
                "is_false_negative",
            ]
        )

        # Rows for all suspicion items (including low-confidence ones).
        for s in report.suspicion_list:
            nid = _normalise_id(s.document_id)
            if nid:
                predicted_doc_ids.add(nid)
            lbl = label_index.get(nid)
            # Binary detection: every suspicion row is treated as flagged.
            flagged = 1
            is_true = 1 if lbl is not None else 0
            tp = 1 if flagged and is_true else 0
            fp = 1 if flagged and not is_true else 0

            writer.writerow(
                [
                    report.run_id,
                    report.model,
                    report.task,
                    s.document_id or "",
                    s.entity_id or "",
                    s.entity_type or "",
                    s.scheme_type.value if s.scheme_type else "",
                    s.confidence,
                    s.severity,
                    s.period or "",
                    ";".join(s.gl_accounts),
                    flagged,
                    lbl.get("document_id") if lbl else "",
                    lbl.get("anomaly_id") if lbl else "",
                    lbl.get("anomaly_type") if lbl else "",
                    lbl.get("severity") if lbl else "",
                    lbl.get("monetary_impact") if lbl else "",
                    lbl.get("is_injected") if lbl else "",
                    tp,
                    fp,
                    0,  # FN is defined at label level below
                ]
            )

        # Additional rows for false negatives: labels that were never flagged.
        flagged_doc_ids: Set[str] = {
            _normalise_id(s.document_id) for s in report.suspicion_list if s.document_id
        }
        for nid, lbl in label_index.items():
            if nid in flagged_doc_ids:
                continue
            # Not flagged at all → false negative at document level.
            writer.writerow(
                [
                    report.run_id,
                    report.model,
                    report.task,
                    lbl.get("document_id") or "",
                    "",  # entity_id (unknown here)
                    "",  # entity_type
                    "",  # predicted_scheme_type
                    0.0,  # confidence
                    "",  # severity (prediction)
                    "",  # period
                    "",  # gl_accounts
                    0,  # flagged_binary
                    lbl.get("document_id") or "",
                    lbl.get("anomaly_id") or "",
                    lbl.get("anomaly_type") or "",
                    lbl.get("severity") or "",
                    lbl.get("monetary_impact") or "",
                    lbl.get("is_injected") or "",
                    0,  # is_true_positive
                    0,  # is_false_positive
                    1,  # is_false_negative
                ]
            )

    log.info("Detection CSV saved to %s", detections_path)
    return result


# ---------------------------------------------------------------------------
# Multi-run comparison table
# ---------------------------------------------------------------------------


def compare_runs(results: List[EvaluationResult]) -> str:
    """
    Return a markdown table comparing multiple evaluation results.
    """
    if not results:
        return "(no results to compare)"
    header = (
        "| run_id | model | task | Acc | P | R | F1 | FPR | ROC-AUC | PR-AUC "
        "| SDR | S-Prec | S-F1 | MeanCov | PerpID | Schemes | steps | tokens |"
    )
    sep = (
        "|--------|-------|------|-----|---|---|-----|-----|---------|--------"
        "|-----|--------|------|---------|--------|---------|-------|--------|"
    )
    rows = []
    for r in results:
        se = r.scheme_eval
        sdr = f"{se.scheme_detection_rate:.3f}" if se else "N/A"
        sp = f"{se.scheme_precision:.3f}" if se else "N/A"
        sf1 = f"{se.scheme_f1:.3f}" if se else "N/A"
        mc = f"{se.mean_coverage:.3f}" if se else "N/A"
        pid = f"{se.perpetrator_identification_rate:.3f}" if se else "N/A"
        schemes = (
            f"{se.n_detected}/{se.n_true_schemes}"
            if se and se.n_true_schemes > 0
            else "N/A"
        )
        rows.append(
            f"| {r.run_id[:8]} | {r.model} | {r.task} "
            f"| {r.entry_accuracy:.3f} | {r.entry_precision:.3f} | {r.entry_recall:.3f} | {r.entry_f1:.3f} "
            f"| {r.false_positive_rate:.3f} "
            f"| {r.roc_auc or 'N/A'} | {r.pr_auc or 'N/A'} "
            f"| {sdr} | {sp} | {sf1} | {mc} | {pid} | {schemes} "
            f"| {r.n_steps} | {r.total_tokens:,} |"
        )
    return "\n".join([header, sep] + rows)
