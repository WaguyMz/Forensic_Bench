#!/usr/bin/env python3
"""
Generate ForensicBench-6D leaderboard with per-sector union-of-predictions scoring.

For each (sector, model) cell with 5 completed runs, unions ``suspicion_list.json``
from all seeds into one virtual prediction set, re-evaluates via ``evaluator.evaluate``,
then aggregates macro-over-sectors like ``generate_leaderboard.py``.

Existing run artifacts are not modified. Combined eval JSON is written to
``<results_root>/_combined_eval/<dataset>/<model>/eval_combined.json``.

Usage:
  python scripts/generate_leaderboard_combined.py
  python scripts/generate_leaderboard_combined.py experiments/results
  python scripts/generate_leaderboard_combined.py --md-out path/to/LEADERBOARD_COMBINED_5x5.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FORENSIC_ROOT = Path(__file__).resolve().parents[2] / "researchpkg" / "forensic_llm"
DEFAULT_RESULTS_ROOT = FORENSIC_ROOT / "experiments" / "results"

# Reuse leaderboard constants, ranking, and rendering from the per-run script.
from generate_leaderboard import (  # noqa: E402
    CANONICAL_DATASETS,
    MODEL_LABELS,
    RunRow,
    _build_leaderboard,
    _find_eval,
    _load_json,
    _model_excluded,
    _ordered_models,
    _print_leaderboard,
    _run_coverage,
    _sector_consistency,
    _type_f1_from_confusion,
    render_latex,
    render_markdown,
)

from researchpkg.forensic_llm.config import DatabaseConfig
from researchpkg.forensic_llm.evaluator import (
    _fetch_labels,
    evaluate,
)
from researchpkg.forensic_llm.models import (
    ForensicReport,
    SchemeReport,
    SchemeType,
    SuspicionItem,
)


@dataclass(frozen=True)
class CellRuns:
    dataset: str
    model_slug: str
    run_dirs: Tuple[Path, ...]


def _load_suspicions(run_dir: Path) -> List[SuspicionItem]:
    """Load JE-level flags from suspicion_list.json, falling back to detections.json."""
    susp_path = run_dir / "suspicion_list.json"
    if susp_path.is_file():
        raw = json.loads(susp_path.read_text(encoding="utf-8"))
        return [SuspicionItem.model_validate(item) for item in raw]

    det_path = run_dir / "detections.json"
    if not det_path.is_file():
        return []
    payload = json.loads(det_path.read_text(encoding="utf-8"))
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
    return items


def _load_scheme_reports(run_dir: Path) -> List[SchemeReport]:
    path = run_dir / "scheme_reports.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SchemeReport.model_validate(item) for item in raw]


def _union_suspicions(suspicion_lists: Sequence[List[SuspicionItem]]) -> List[SuspicionItem]:
    """Union JE flags across runs; per document_id keep the item with max confidence."""
    best: Dict[str, SuspicionItem] = {}
    for suspicions in suspicion_lists:
        for item in suspicions:
            if not item.document_id:
                continue
            doc_key = str(item.document_id).strip().lower()
            cur = best.get(doc_key)
            if cur is None or float(item.confidence or 0.0) > float(cur.confidence or 0.0):
                best[doc_key] = item
    return list(best.values())


def _union_scheme_reports(report_lists: Sequence[List[SchemeReport]]) -> List[SchemeReport]:
    """Union scheme reports by (scheme_type, perpetrator_id), merging document_ids."""
    merged: Dict[Tuple[str, str], SchemeReport] = {}
    for reports in report_lists:
        for sr in reports:
            perp = (sr.perpetrator_id or "").strip().lower()
            key = (sr.scheme_type.value, perp)
            if key not in merged:
                merged[key] = sr.model_copy(deep=True)
                continue
            existing = merged[key]
            doc_ids = set(existing.document_ids) | set(sr.document_ids)
            existing.document_ids = sorted(doc_ids)
            existing.items = _union_suspicions([existing.items, sr.items])
    return list(merged.values())


def _db_config_from_manifest(run_dir: Path) -> DatabaseConfig:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run_manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    db = manifest.get("config", {}).get("database")
    if not isinstance(db, dict):
        raise ValueError(f"No database config in {manifest_path}")
    return DatabaseConfig(
        host=db.get("host", "localhost"),
        port=int(db.get("port", 5432)),
        database=db.get("database", "datasynth_forensic_public"),
        user=db.get("user", "postgres"),
        password=db.get("password"),
        statement_timeout_ms=int(db.get("statement_timeout_ms", 30_000)),
        default_max_rows=int(db.get("default_max_rows", 2000)),
        hard_max_rows=int(db.get("hard_max_rows", 10_000)),
    )


def _manifest_model(run_dir: Path) -> str:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return str(manifest.get("model") or "")
    return ""


def _discover_cells(
    results_root: Path,
    *,
    exclude_40m: bool,
    exclude_10m: bool,
) -> List[CellRuns]:
    cells: List[CellRuns] = []
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if dataset_dir.name.startswith(("RESULTS", "LEADERBOARD", "_combined_eval")):
            continue
        dataset = dataset_dir.name
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            slug = model_dir.name
            if _model_excluded(slug, exclude_40m, exclude_10m):
                continue
            run_dirs = [
                child
                for child in sorted(model_dir.iterdir())
                if child.is_dir()
                and not child.name.startswith(".")
                and _find_eval(child) is not None
            ]
            if run_dirs:
                cells.append(
                    CellRuns(
                        dataset=dataset,
                        model_slug=slug,
                        run_dirs=tuple(run_dirs),
                    )
                )
    return cells


def _filter_complete_cells(
    cells: Sequence[CellRuns],
    *,
    datasets: Sequence[str],
    runs_per_cell: int,
) -> List[CellRuns]:
    counts: Dict[str, Dict[str, int]] = {}
    by_key: Dict[Tuple[str, str], CellRuns] = {}
    for cell in cells:
        counts.setdefault(cell.model_slug, {})
        counts[cell.model_slug][cell.dataset] = len(cell.run_dirs)
        by_key[(cell.dataset, cell.model_slug)] = cell

    complete_slugs = {
        slug
        for slug, ds_counts in counts.items()
        if all(ds_counts.get(ds, 0) == runs_per_cell for ds in datasets)
    }
    dataset_set = set(datasets)
    return [
        by_key[(ds, slug)]
        for slug in sorted(complete_slugs)
        for ds in datasets
        if (ds, slug) in by_key and ds in dataset_set
    ]


def _combined_eval_for_cell(
    cell: CellRuns,
    *,
    labels: Optional[List[dict]],
    write_eval_path: Optional[Path],
) -> dict:
    suspicion_lists = [_load_suspicions(run_dir) for run_dir in cell.run_dirs]
    scheme_report_lists = [_load_scheme_reports(run_dir) for run_dir in cell.run_dirs]
    union_suspicions = _union_suspicions(suspicion_lists)
    union_scheme_reports = _union_scheme_reports(scheme_report_lists)

    db_config = _db_config_from_manifest(cell.run_dirs[0])
    report = ForensicReport(
        run_id=f"combined-{cell.dataset}-{cell.model_slug}",
        model=_manifest_model(cell.run_dirs[0]),
        task="full",
        suspicion_list=union_suspicions,
        scheme_reports=union_scheme_reports,
    )

    result = evaluate(report, db_config, labels=labels)
    payload = result.model_dump()

    if write_eval_path is not None:
        write_eval_path.parent.mkdir(parents=True, exist_ok=True)
        write_eval_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
    return payload


def _individual_entry_f1_values(cell: CellRuns) -> List[float]:
    values: List[float] = []
    for run_dir in cell.run_dirs:
        eval_path = _find_eval(run_dir)
        if eval_path is None:
            continue
        ev = _load_json(eval_path)
        values.append(float(ev.get("entry_f1") or 0.0))
    return values


def _consistency_per_model_from_cells(
    cells: Sequence[CellRuns],
    models: List[str],
    datasets: List[str],
) -> Dict[str, float]:
    """Consistency from per-seed Entry F1 variance (unchanged from per-run view)."""
    out: Dict[str, float] = {}
    for slug in models:
        per_ds: List[float] = []
        for ds in datasets:
            matching = [c for c in cells if c.model_slug == slug and c.dataset == ds]
            if not matching:
                continue
            f1_vals = _individual_entry_f1_values(matching[0])
            if f1_vals:
                per_ds.append(_sector_consistency(f1_vals))
        out[slug] = sum(per_ds) / len(per_ds) if per_ds else 0.0
    return out


def _build_leaderboard_with_consistency(
    rows: List[RunRow],
    cells: Sequence[CellRuns],
) -> List:
    leaderboard = _build_leaderboard(rows)
    datasets = list(CANONICAL_DATASETS)
    models = _ordered_models({r.model_slug for r in rows})
    consistency = _consistency_per_model_from_cells(cells, models, datasets)
    for entry in leaderboard:
        entry.consistency = consistency.get(entry.model_slug, entry.consistency)
    leaderboard.sort(
        key=lambda r: (
            -r.e_f1,
            -r.type_f1,
            -r.recall,
            -r.precision,
            -r.coverage,
            -r.consistency,
        )
    )
    for i, row in enumerate(leaderboard, 1):
        row.rank = i
    return leaderboard


def _combined_metrics_documentation(*, runs_per_cell: int) -> List[str]:
    n_sectors = len(CANONICAL_DATASETS)
    sectors = ", ".join(CANONICAL_DATASETS)
    return [
        "## Combined-prediction aggregation",
        "",
        "Unlike `LEADERBOARD_5x5.md`, which averages per-seed metrics, this table "
        f"re-scores each sector cell by taking the **union** of JE-level predictions "
        f"across the {runs_per_cell} seeds, then running `evaluator.evaluate()` once "
        "per (sector, model).",
        "",
        "1. **Per sector** — union `suspicion_list.json` from all seeds in the cell "
        "(one flag per `document_id`; highest confidence wins on conflicts).",
        "2. **Per model** — macro mean of the five sector-level combined scores.",
        "",
        f"Sectors: {sectors}.",
        "",
        "**Consistency** is still computed from per-seed Entry F1 variance within each "
        "sector (seed stability), not from the union score.",
        "",
        "Combined eval artifacts: `_combined_eval/<dataset>/<model>/eval_combined.json`.",
        "",
    ]


def render_markdown_combined(
    leaderboard: List,
    results_root: Path,
    *,
    runs_per_cell: int,
) -> str:
    base = render_markdown(leaderboard, results_root, runs_per_cell=runs_per_cell)
    lines = base.splitlines()
    title_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("# ForensicBench-6D Leaderboard")),
        0,
    )
    lines[title_idx] = "# ForensicBench-6D Leaderboard (Combined Predictions)"
    grid_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("**Grid:**")),
        None,
    )
    if grid_idx is not None:
        lines[grid_idx] = (
            f"**Grid:** {len(CANONICAL_DATASETS)} sectors × {runs_per_cell} seeds "
            f"(complete 5×5 only; union-of-predictions scoring)  "
        )
    doc_start = next(
        (i for i, line in enumerate(lines) if line == "## Metric definitions and computation"),
        len(lines),
    )
    combined_doc = _combined_metrics_documentation(runs_per_cell=runs_per_cell)
    return "\n".join(lines[:doc_start] + combined_doc + lines[doc_start:])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        nargs="?",
        default=str(DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--runs-per-cell",
        type=int,
        default=5,
        help="Required completed runs per (sector, model) cell (default: 5).",
    )
    parser.add_argument(
        "--include-40m",
        action="store_true",
        help="Include model folders whose names contain 40M.",
    )
    parser.add_argument(
        "--include-10m",
        action="store_true",
        help="Include qwen397_10M runs.",
    )
    parser.add_argument(
        "--no-write-combined-eval",
        action="store_true",
        help="Skip writing _combined_eval/<dataset>/<model>/eval_combined.json files.",
    )
    parser.add_argument(
        "--md-out",
        default="",
        help="Markdown output (default: <results_root>/LEADERBOARD_COMBINED_5x5.md).",
    )
    parser.add_argument(
        "--tex-out",
        default="",
        help="LaTeX output (default: <results_root>/LEADERBOARD_COMBINED_5x5.tex).",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).resolve()
    if not results_root.is_dir():
        print(f"Not a directory: {results_root}", file=sys.stderr)
        return 1

    all_cells = _discover_cells(
        results_root,
        exclude_40m=not args.include_40m,
        exclude_10m=not args.include_10m,
    )
    cells = _filter_complete_cells(
        all_cells,
        datasets=CANONICAL_DATASETS,
        runs_per_cell=args.runs_per_cell,
    )
    if not cells:
        print(
            "No models with a complete 5×5 grid found.",
            file=sys.stderr,
        )
        return 1

    # Fetch labels once per dataset is not possible with one DB; each cell has its own DB.
    # Cache labels by non-public database name to avoid repeated round-trips.
    labels_cache: Dict[str, List[dict]] = {}

    def _labels_for_cell(cell: CellRuns) -> List[dict]:
        db_config = _db_config_from_manifest(cell.run_dirs[0])
        non_public = db_config.database.replace("_public", "")
        if non_public not in labels_cache:
            labels_cache[non_public] = _fetch_labels(db_config)
        return labels_cache[non_public]

    rows: List[RunRow] = []
    combined_root = results_root / "_combined_eval"
    for cell in cells:
        eval_out = (
            None
            if args.no_write_combined_eval
            else combined_root / cell.dataset / cell.model_slug / "eval_combined.json"
        )
        ev = _combined_eval_for_cell(
            cell,
            labels=_labels_for_cell(cell),
            write_eval_path=eval_out,
        )
        rows.append(
            RunRow(
                dataset=cell.dataset,
                model_slug=cell.model_slug,
                model_label=MODEL_LABELS.get(cell.model_slug, cell.model_slug),
                entry_recall=float(ev.get("entry_recall") or 0.0),
                entry_precision=float(ev.get("entry_precision") or 0.0),
                entry_f1=float(ev.get("entry_f1") or 0.0),
                type_f1=_type_f1_from_confusion(ev),
                coverage=_run_coverage(ev),
            )
        )

    leaderboard = _build_leaderboard_with_consistency(rows, cells)
    md_path = (
        Path(args.md_out)
        if args.md_out
        else results_root / "LEADERBOARD_COMBINED_5x5.md"
    )
    tex_path = (
        Path(args.tex_out)
        if args.tex_out
        else results_root / "LEADERBOARD_COMBINED_5x5.tex"
    )

    md_text = render_markdown_combined(
        leaderboard,
        results_root,
        runs_per_cell=args.runs_per_cell,
    )
    tex_text = render_latex(leaderboard, runs_per_cell=args.runs_per_cell)
    tex_text = tex_text.replace(
        "ForensicBench-6D leaderboard (complete 5$\\times$5)",
        "ForensicBench-6D leaderboard (combined predictions, complete 5$\\times$5)",
    )
    tex_text = tex_text.replace(
        "tab:forensicbench-6d-leaderboard",
        "tab:forensicbench-6d-leaderboard-combined",
    )

    md_path.write_text(md_text, encoding="utf-8")
    tex_path.write_text(tex_text, encoding="utf-8")

    _print_leaderboard(
        leaderboard,
        md_path=md_path,
        tex_path=tex_path,
        n_runs=len(rows),
        runs_per_cell=args.runs_per_cell,
    )
    if not args.no_write_combined_eval:
        print(f"Wrote combined eval artifacts under {combined_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
