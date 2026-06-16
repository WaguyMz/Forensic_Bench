#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize a generated dataset and write config.json into the dataset dir."
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory (contains forensic_llm/).",
    )
    p.add_argument(
        "--dataset-key",
        default=None,
        help="Optional dataset key (defaults to dataset-dir basename).",
    )
    p.add_argument(
        "--source-config",
        default=None,
        help="Optional path to the YAML config used to generate this dataset.",
    )
    return p.parse_args()


def _read_csv_header(path: Path) -> Tuple[int, Iterable[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, [])
        row_count = 0
        for _ in r:
            row_count += 1
    return row_count, header


def _count_and_group(path: Path, group_col: str) -> Tuple[int, Counter]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        counts: Counter = Counter()
        n = 0
        for row in r:
            n += 1
            k = row.get(group_col)
            if k is None or k == "":
                k = "(null)"
            counts[k] += 1
    return n, counts


def _minmax_date(path: Path, col: str) -> Optional[Tuple[str, str]]:
    mn: Optional[str] = None
    mx: Optional[str] = None
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = (row.get(col) or "").strip()
            if not v:
                continue
            if mn is None or v < mn:
                mn = v
            if mx is None or v > mx:
                mx = v
    if mn is None or mx is None:
        return None
    return mn, mx


def _fraud_breakdown_from_anomaly_labels(path: Path) -> Dict[str, Any]:
    """
    anomaly_labels.csv schema (example):
      anomaly_type = "Fraud(Kickback)" etc.
    We'll parse:
      - anomaly_type raw distribution
      - fraud_scheme distribution if pattern Fraud(<scheme>)
      - injected vs not
    """
    totals = 0
    anomaly_type = Counter()
    fraud_scheme = Counter()
    injected = Counter()
    severity = Counter()

    fraud_re = re.compile(r"^Fraud\\((?P<scheme>[^)]+)\\)$")

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            totals += 1
            at = (row.get("anomaly_type") or "").strip() or "(null)"
            anomaly_type[at] += 1
            m = fraud_re.match(at)
            if m:
                fraud_scheme[m.group("scheme")] += 1
            inj = (row.get("is_injected") or "").strip().lower()
            injected["true" if inj == "true" else "false"] += 1
            sev = (row.get("severity") or "").strip() or "(null)"
            severity[sev] += 1

    return {
        "rows": totals,
        "anomaly_type_counts": dict(anomaly_type),
        "fraud_scheme_counts": dict(fraud_scheme),
        "is_injected_counts": dict(injected),
        "severity_counts": dict(severity),
    }


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_key = args.dataset_key or dataset_dir.name
    source_cfg = Path(args.source_config).resolve() if args.source_config else None

    forensic_dir = dataset_dir / "forensic_llm"
    run_manifest_path = dataset_dir / "run_manifest.json"

    cfg_obj: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "paths": {"dataset_dir": str(dataset_dir)},
        "source_config": str(source_cfg) if source_cfg else None,
        "counts": {},
        "schema": {},
        "date_ranges": {},
        "process_counts": {},
        "fraud_distributions": None,
        "notes": [],
    }

    # Preferred: forensic_llm export (je_header/je_line/master data + anomaly labels).
    if forensic_dir.exists():
        cfg_obj["paths"]["forensic_llm_dir"] = str(forensic_dir)
        paths = {
            "je_header": forensic_dir / "je_header.csv",
            "je_line": forensic_dir / "je_line.csv",
            "employees": forensic_dir / "employees.csv",
            "vendors": forensic_dir / "vendors.csv",
            "customers": forensic_dir / "customers.csv",
            "anomaly_labels": forensic_dir / "anomaly_labels.csv",
        }
        for required in ("je_header", "je_line", "employees", "vendors", "customers"):
            if not paths[required].exists():
                raise SystemExit(f"Missing required file: {paths[required]}")

        je_headers, je_header_cols = _read_csv_header(paths["je_header"])
        je_lines, je_line_cols = _read_csv_header(paths["je_line"])
        employees, _ = _read_csv_header(paths["employees"])
        vendors, _ = _read_csv_header(paths["vendors"])
        customers, _ = _read_csv_header(paths["customers"])

        _, process_counts = _count_and_group(paths["je_header"], "business_process")
        posting_range = _minmax_date(paths["je_header"], "posting_date")
        document_range = _minmax_date(paths["je_header"], "document_date")

        anomaly_summary = None
        if paths["anomaly_labels"].exists():
            anomaly_summary = _fraud_breakdown_from_anomaly_labels(paths["anomaly_labels"])

        cfg_obj["counts"] = {
            "je_header_rows": je_headers,
            "je_line_rows": je_lines,
            "employees_rows": employees,
            "vendors_rows": vendors,
            "customers_rows": customers,
        }
        cfg_obj["schema"] = {
            "je_header_columns": list(je_header_cols),
            "je_line_columns": list(je_line_cols),
        }
        cfg_obj["date_ranges"] = {
            "posting_date": {"min": posting_range[0], "max": posting_range[1]}
            if posting_range
            else None,
            "document_date": {"min": document_range[0], "max": document_range[1]}
            if document_range
            else None,
        }
        cfg_obj["process_counts"] = dict(process_counts)
        cfg_obj["fraud_distributions"] = anomaly_summary

    # Fallback: use run_manifest.json produced by the generator even when forensic_llm export is absent.
    else:
        if not run_manifest_path.exists():
            raise SystemExit(
                f"Missing forensic_llm/ and no run_manifest.json found under dataset dir: {dataset_dir}"
            )

        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        cfg_obj["paths"]["run_manifest"] = str(run_manifest_path)

        stats = manifest.get("statistics") or {}
        cfg_obj["counts"] = {
            "total_entries": stats.get("total_entries"),
            "total_line_items": stats.get("total_line_items"),
            "accounts_count": stats.get("accounts_count"),
            "companies_count": stats.get("companies_count"),
            "period_months": stats.get("period_months"),
            "vendors_rows": stats.get("vendor_count"),
            "customers_rows": stats.get("customer_count"),
            "materials_rows": stats.get("material_count"),
            "fixed_assets_rows": stats.get("asset_count"),
            "employees_rows": stats.get("employee_count"),
            "p2p_chain_count": stats.get("p2p_chain_count"),
            "o2c_chain_count": stats.get("o2c_chain_count"),
        }

        snap = manifest.get("config_snapshot") or {}
        output_cfg = snap.get("output") or {}
        cfg_obj["paths"]["output_directory"] = output_cfg.get("output_directory")

        # Configured (intended) master data sizes (may differ from actual statistics above)
        md = (snap.get("master_data") or {})
        cfg_obj["counts"]["configured_master_data"] = {
            "vendors": (md.get("vendors") or {}).get("count"),
            "customers": (md.get("customers") or {}).get("count"),
            "employees": (md.get("employees") or {}).get("count"),
            "materials": (md.get("materials") or {}).get("count"),
            "fixed_assets": (md.get("fixed_assets") or {}).get("count"),
        }

        # Configured business process weights/rates (not actual observed counts)
        bp = snap.get("business_processes") or {}
        if bp:
            cfg_obj["process_counts"] = {"configured": bp}

        # Configured fraud distributions (not actual injected labels)
        fraud_cfg = snap.get("fraud") or {}
        if fraud_cfg:
            cfg_obj["fraud_distributions"] = {"configured": fraud_cfg}

        cfg_obj["notes"].append(
            "forensic_llm export not found; counts/distributions are derived from run_manifest.json (actual total_entries/total_line_items) and config_snapshot (configured master_data / processes / fraud)."
        )

    out_path = dataset_dir / "config.json"
    out_path.write_text(json.dumps(cfg_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

