#!/usr/bin/env python3
"""
Generate ForensicBench-6D leaderboard (MD + LaTeX) for complete 5×5 models.

Ranking: lexicographic on E-F1 → Type-F1 → Recall → Precision → Coverage → Consistency
(no composite weights).

Usage:
  python scripts/generate_leaderboard.py
  python scripts/generate_leaderboard.py experiments/results
  python scripts/generate_leaderboard.py --md-out path/to/LEADERBOARD.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

FORENSIC_ROOT = Path(__file__).resolve().parents[2] / "researchpkg" / "forensic_llm"
DEFAULT_RESULTS_ROOT = FORENSIC_ROOT / "experiments" / "results"

CANONICAL_DATASETS = (
    "energy",
    "healthcare",
    "luxurygoods",
    "manufacturing",
    "transport",
)

# Paper leaderboard models (12), ranked by E-F1 in the EMNLP 2026 submission.
MODEL_ORDER = [
    "minimax229b",
    "qwen3.5_397B_fp8",
    "qwen122b",
    "qwen3.6_35B_fp8",
    "mistral_128b",
    "gemma4_31b",
    "gemma4e4b",
    "gptoss_120b",
    "mistral_119b",
    "qwen9b",
    "granite_30b",
    "llama33_70b",
]

MODEL_LABELS: Dict[str, str] = {
    "minimax229b": "MiniMax-M2.7",
    "qwen3.5_397B_fp8": "Qwen3.5-397B",
    "qwen122b": "Qwen3.5-122B",
    "qwen3.6_35B_fp8": "Qwen3.6-35B",
    "mistral_128b": "Mistral-Medium-3.5-128B",
    "gemma4_31b": "Gemma-4-31B",
    "gemma4e4b": "Gemma-4-E4B",
    "gptoss_120b": "GPT-OSS-120B",
    "mistral_119b": "Mistral-Small-4-119B",
    "qwen9b": "Qwen3.5-9B",
    "granite_30b": "Granite-30B",
    "llama33_70b": "Llama-3.3-70B",
}

CONSISTENCY_EPS = 0.01

# Canonical scheme types (matches evaluator confusion matrix core labels).
CORE_SCHEME_TYPES = (
    "shadow_payroll",
    "fictitious_ap_disbursements",
    "revenue_manipulation",
    "vendor_collusion",
    "inventory_manipulation",
)


@dataclass(frozen=True)
class RunRow:
    dataset: str
    model_slug: str
    model_label: str
    entry_recall: float
    entry_precision: float
    entry_f1: float
    type_f1: float
    coverage: float


@dataclass
class LeaderboardRow:
    rank: int
    model_slug: str
    model_label: str
    recall: float
    precision: float
    e_f1: float
    type_f1: float
    coverage: float
    consistency: float
    n_runs: int


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_eval(run_dir: Path) -> Optional[Path]:
    matches = sorted(run_dir.glob("eval_*.json"))
    return matches[0] if matches else None


def _model_excluded(slug: str, exclude_40m: bool, exclude_10m: bool) -> bool:
    if exclude_40m and "40M" in slug.upper():
        return True
    if exclude_10m and slug.endswith("_10M"):
        return True
    return False


def _type_f1_from_confusion(ev: dict) -> float:
    """
    Scheme-type F1 using the same denominators as Entry F1.

    TP = diagonal count from ``confusion_matrix.nonzero_cells`` (flagged
    fraud JEs with correct ``scheme_type``).  Always TP_type <= TP_entry.

    Precision = TP_type / n_flagged, Recall = TP_type / n_true — so
    Type-F1 <= Entry-F1 on every run.
    """
    cm = ev.get("confusion_matrix") or {}
    cells: List[dict] = cm.get("nonzero_cells") or []
    n_flagged = int(ev.get("n_flagged") or 0)
    n_true = int(ev.get("n_true_anomalies") or 0)
    if not cells or n_flagged <= 0 or n_true <= 0:
        return 0.0

    core = set(CORE_SCHEME_TYPES)
    tp = sum(
        int(c.get("count") or 0)
        for c in cells
        if c.get("pred") == c.get("true") and c.get("pred") in core
    )
    prec = tp / n_flagged
    rec = tp / n_true
    if prec + rec == 0.0:
        return 0.0
    type_f1 = 2.0 * prec * rec / (prec + rec)
    entry_f1 = float(ev.get("entry_f1") or 0.0)
    # TP_type <= TP_entry always; cap handles 4-decimal rounding in stored entry_f1.
    return min(type_f1, entry_f1)


def _run_coverage(ev: dict) -> float:
    scheme_eval = ev.get("scheme_eval") or {}
    per_scheme = scheme_eval.get("per_scheme") or []
    ratios = [
        float(row.get("coverage_ratio") or 0.0)
        for row in per_scheme
        if row.get("n_true_docs", 0) > 0
    ]
    if not ratios:
        return 0.0
    return statistics.mean(ratios)


def _discover_runs(
    results_root: Path,
    *,
    exclude_40m: bool,
    exclude_10m: bool,
) -> List[RunRow]:
    rows: List[RunRow] = []
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if dataset_dir.name.startswith("RESULTS") or dataset_dir.name.startswith("LEADERBOARD"):
            continue
        dataset = dataset_dir.name
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            slug = model_dir.name
            if _model_excluded(slug, exclude_40m, exclude_10m):
                continue
            label = MODEL_LABELS.get(slug, slug)
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                eval_path = _find_eval(run_dir)
                if eval_path is None:
                    continue
                ev = _load_json(eval_path)
                rows.append(
                    RunRow(
                        dataset=dataset,
                        model_slug=slug,
                        model_label=label,
                        entry_recall=float(ev.get("entry_recall") or 0.0),
                        entry_precision=float(ev.get("entry_precision") or 0.0),
                        entry_f1=float(ev.get("entry_f1") or 0.0),
                        type_f1=_type_f1_from_confusion(ev),
                        coverage=_run_coverage(ev),
                    )
                )
    return rows


def _filter_complete_grid(
    rows: List[RunRow],
    *,
    datasets: Sequence[str],
    runs_per_cell: int,
) -> List[RunRow]:
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        counts.setdefault(r.model_slug, {})
        counts[r.model_slug][r.dataset] = counts[r.model_slug].get(r.dataset, 0) + 1

    complete = {
        slug
        for slug, ds_counts in counts.items()
        if all(ds_counts.get(ds, 0) == runs_per_cell for ds in datasets)
    }
    dataset_set = set(datasets)
    return [
        r for r in rows if r.model_slug in complete and r.dataset in dataset_set
    ]


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _sector_consistency(f1_values: List[float]) -> float:
    """Per-sector consistency in [0, 100]: 100 * max(0, 1 - CV)."""
    if len(f1_values) < 2:
        return 100.0 if f1_values else 0.0
    mu = statistics.mean(f1_values)
    sigma = statistics.stdev(f1_values)
    cv = sigma / (mu + CONSISTENCY_EPS)
    return max(0.0, 1.0 - cv) * 100.0


def _macro_mean_per_dataset(
    rows: List[RunRow],
    models: List[str],
    datasets: List[str],
    attr: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for slug in models:
        per_ds: List[float] = []
        for ds in datasets:
            vals = [getattr(r, attr) for r in rows if r.model_slug == slug and r.dataset == ds]
            if vals:
                per_ds.append(_mean(vals))
        out[slug] = _mean(per_ds)
    return out


def _consistency_per_model(
    rows: List[RunRow],
    models: List[str],
    datasets: List[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for slug in models:
        per_ds: List[float] = []
        for ds in datasets:
            f1_vals = [r.entry_f1 for r in rows if r.model_slug == slug and r.dataset == ds]
            if f1_vals:
                per_ds.append(_sector_consistency(f1_vals))
        out[slug] = _mean(per_ds)
    return out


def _ordered_models(slugs: Iterable[str]) -> List[str]:
    seen = set(slugs)
    ordered = [s for s in MODEL_ORDER if s in seen]
    for s in sorted(seen):
        if s not in ordered:
            ordered.append(s)
    return ordered


def _build_leaderboard(rows: List[RunRow]) -> List[LeaderboardRow]:
    datasets = list(CANONICAL_DATASETS)
    models = _ordered_models({r.model_slug for r in rows})

    recall = _macro_mean_per_dataset(rows, models, datasets, "entry_recall")
    precision = _macro_mean_per_dataset(rows, models, datasets, "entry_precision")
    e_f1 = _macro_mean_per_dataset(rows, models, datasets, "entry_f1")
    type_f1 = _macro_mean_per_dataset(rows, models, datasets, "type_f1")
    coverage = _macro_mean_per_dataset(rows, models, datasets, "coverage")
    consistency = _consistency_per_model(rows, models, datasets)

    entries: List[LeaderboardRow] = []
    for slug in models:
        n_runs = sum(1 for r in rows if r.model_slug == slug)
        entries.append(
            LeaderboardRow(
                rank=0,
                model_slug=slug,
                model_label=MODEL_LABELS.get(slug, slug),
                recall=recall[slug],
                precision=precision[slug],
                e_f1=e_f1[slug],
                type_f1=type_f1[slug],
                coverage=coverage[slug],
                consistency=consistency[slug],
                n_runs=n_runs,
            )
        )

    entries.sort(
        key=lambda r: (
            -r.e_f1,
            -r.type_f1,
            -r.recall,
            -r.precision,
            -r.coverage,
            -r.consistency,
        )
    )
    for i, row in enumerate(entries, 1):
        row.rank = i
    return entries


def _metrics_documentation(*, runs_per_cell: int) -> List[str]:
    n_sectors = len(CANONICAL_DATASETS)
    n_runs = runs_per_cell * n_sectors
    sectors = ", ".join(CANONICAL_DATASETS)
    return [
        "## Metric definitions and computation",
        "",
        "All six axes are computed from `eval_*.json` produced by `evaluator.py` "
        f"after a completed forensic run. Each leaderboard entry aggregates "
        f"**{n_runs} runs** ({n_sectors} sectors × {runs_per_cell} seeds). "
        "A model appears only if every sector cell has exactly "
        f"{runs_per_cell} completed evaluations.",
        "",
        "### Aggregation convention (macro over sectors)",
        "",
        "For Recall, Precision, E-F1, Type-F1, and Coverage:",
        "",
        "1. **Per run** — compute the metric from one `eval_*.json`.",
        "2. **Per sector** — average over the "
        f"{runs_per_cell} seeds in that sector.",
        "3. **Per model** — unweighted mean of the five sector averages "
        "(one value per sector, then macro mean).",
        "",
        f"Sectors: {sectors}.",
        "",
        "**Ranking** — lexicographic sort on: "
        "**E-F1** → Type-F1 → Recall → Precision → Coverage → Consistency "
        "(no weighted composite).",
        "",
        "---",
        "",
        "### 1. E-F1 (Entry F1) — primary ranking key",
        "",
        "**What it measures:** Harmonic balance between finding fraudulent "
        "journal entries and avoiding false alarms, at the JE (`document_id`) level.",
        "",
        "**Per-run computation** (from `eval_*.json`):",
        "",
        "- $T$ = set of ground-truth fraudulent `document_id`s (from `anomaly_labels`)",
        "- $F$ = set of flagged `document_id`s (from `report.suspicion_list`)",
        "- $\\text{TP} = |T \\cap F|$, $\\text{FN} = |T \\setminus F|$, "
        "$\\text{FP} = |F \\setminus T|$",
        "- $\\text{Recall} = \\text{TP}/|T|$, $\\text{Precision} = \\text{TP}/|F|$",
        "- $\\text{E-F1} = 2 \\cdot \\text{Precision} \\cdot \\text{Recall} "
        "/ (\\text{Precision} + \\text{Recall})$",
        "",
        "**Leaderboard value:** macro mean of per-sector E-F1 averages, reported as %.",
        "",
        "---",
        "",
        "### 2. Type-F1 (Scheme-type classification F1)",
        "",
        "**What it measures:** Whether the agent assigns the **correct canonical "
        "scheme type** to journal entries — e.g. `revenue_manipulation` vs "
        "`vendor_collusion`. Distinct from E-F1 (entry detection) and Coverage "
        "(investigation depth per scheme).",
        "",
        "**Per-run computation** (from `confusion_matrix` in `eval_*.json`):",
        "",
        "The evaluator builds a JE-level confusion matrix over five canonical "
        "scheme types (rows = predicted `scheme_type`, columns = true type). "
        "Diagonal counts are read from `confusion_matrix.nonzero_cells`.",
        "",
        "Per-run computation (aligned with Entry F1 denominators):",
        "",
        "- $\\text{TP}_{\\text{type}}$ = $\\sum_c$ count(`pred=$c$`, `true=$c$`) "
        "(flagged fraud with **correct** scheme label; always "
        "$\\leq \\text{TP}_{\\text{entry}}$)",
        "- $\\text{Precision}_{\\text{type}} = \\text{TP}_{\\text{type}} / "
        "n_{\\text{flagged}}$",
        "- $\\text{Recall}_{\\text{type}} = \\text{TP}_{\\text{type}} / "
        "n_{\\text{true}}$",
        "- $\\text{Type-F1} = 2 \\cdot P_{\\text{type}} \\cdot R_{\\text{type}} / "
        "(P_{\\text{type}} + R_{\\text{type}})$",
        "",
        "Using the same $n_{\\text{flagged}}$ and $n_{\\text{true}}$ as Entry F1 "
        "ensures **Type-F1 $\\leq$ Entry-F1** on every run: only a subset of "
        "entry true positives count as type true positives.",
        "",
        "**Leaderboard value:** macro mean of per-sector Type-F1 averages, reported as %.",
        "",
        "---",
        "",
        "### 3. Recall (Entry Recall)",
        "",
        "**What it measures:** Fraction of truly fraudulent journal entries "
        "that the agent flagged (sensitivity / completeness).",
        "",
        "**Per-run computation:** $\\text{Recall} = |T \\cap F| / |T|$ "
        "(field `entry_recall` in `eval_*.json`).",
        "",
        "**Leaderboard value:** macro mean of per-sector recall averages, reported as %.",
        "",
        "---",
        "",
        "### 4. Precision (Entry Precision)",
        "",
        "**What it measures:** Fraction of flagged journal entries that are "
        "truly fraudulent (flag credibility / false-discovery control).",
        "",
        "**Per-run computation:** $\\text{Precision} = |T \\cap F| / |F|$ "
        "(field `entry_precision` in `eval_*.json`).",
        "",
        "**Leaderboard value:** macro mean of per-sector precision averages, reported as %.",
        "",
        "---",
        "",
        "### 5. Coverage (Scheme entry coverage)",
        "",
        "**What it measures:** How deeply the agent investigates each true fraud "
        "scheme — the fraction of a scheme's journal entries that get flagged. "
        "Gives partial credit regardless of whether the scheme type label is correct.",
        "",
        "**Per-run computation:**",
        "",
        "For each ground-truth scheme $s$ in `scheme_eval.per_scheme`:",
        "",
        "$$\\text{coverage}_s = \\frac{\\text{n\\_flagged\\_docs}_s}"
        "{\\text{n\\_true\\_docs}_s}$$",
        "",
        "Per-run Coverage = mean of $\\text{coverage}_s$ over all true schemes "
        "with $\\text{n\\_true\\_docs} > 0$.",
        "",
        "**Leaderboard value:** macro mean of per-sector coverage averages, reported as %.",
        "",
        "---",
        "",
        "### 6. Consistency (Seed stability)",
        "",
        "**What it measures:** How stable Entry F1 is across the "
        f"{runs_per_cell} random seeds within each sector — penalizes "
        "path-dependent luck (planning order, hypothesis sampling).",
        "",
        "**Per-sector computation:**",
        "",
        f"Let $\\{{f_1, \\ldots, f_{{{runs_per_cell}}}\\}}$ be Entry F1 scores "
        "for the seeds in one sector.",
        "",
        "$$\\mu = \\frac{1}{"
        f"{runs_per_cell}"
        "}\\sum_i f_i, \\quad "
        "\\sigma = \\text{stdev}(f_1, \\ldots, f_"
        f"{runs_per_cell}"
        ")$$",
        "",
        "$$\\text{Consistency}_{\\text{sector}} = 100 \\times "
        "\\max\\!\\left(0,\\; 1 - \\frac{\\sigma}{\\mu + \\varepsilon}\\right)$$",
        "",
        f"with $\\varepsilon = {CONSISTENCY_EPS}$ (avoids division by zero when "
        "$\\mu \\approx 0$).",
        "",
        "**Leaderboard value:** unweighted mean of the five per-sector "
        "consistency scores, reported on a 0–100 scale.",
        "",
        "**Note:** a model that consistently scores near zero can still show "
        "high Consistency (low variance, not high quality). Use E-F1 as the "
        "primary quality indicator.",
        "",
    ]


def render_markdown(
    leaderboard: List[LeaderboardRow],
    results_root: Path,
    *,
    runs_per_cell: int,
) -> str:
    n_models = len(leaderboard)
    lines = [
        "# ForensicBench-6D Leaderboard",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Scope:** `{results_root.relative_to(FORENSIC_ROOT)}`  ",
        f"**Grid:** {len(CANONICAL_DATASETS)} sectors × {runs_per_cell} seeds (complete 5×5 only)  ",
        f"**Models:** {n_models}",
        "",
        "**Ranking:** lexicographic on **E-F1** → Type-F1 → Recall → Precision "
        "→ Coverage → Consistency (no weighted composite).",
        "",
        "| Rank | Model | **E-F1** | Type-F1 | Recall | Precision | Coverage | Consistency |",
        "|-----:|-------|---------:|--------:|-------:|----------:|---------:|------------:|",
    ]
    for r in leaderboard:
        lines.append(
            f"| {r.rank} | {r.model_label} | "
            f"**{r.e_f1 * 100:.1f}** | {r.type_f1 * 100:.1f} | "
            f"{r.recall * 100:.1f} | {r.precision * 100:.1f} | "
            f"{r.coverage * 100:.1f} | {r.consistency:.1f} |"
        )
    lines.append("")
    lines.extend(_metrics_documentation(runs_per_cell=runs_per_cell))
    return "\n".join(lines)


def _tex_pct(value: float) -> str:
    return f"{value * 100:.1f}"


def render_latex(
    leaderboard: List[LeaderboardRow],
    *,
    runs_per_cell: int,
) -> str:
    lines = [
        "% Auto-generated — ForensicBench-6D leaderboard (complete 5×5)",
        r"\usepackage{booktabs}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{ForensicBench-6D leaderboard (complete 5$\times$5 grid). "
        r"Ranked lexicographically: E-F1 $\rightarrow$ Type-F1 $\rightarrow$ Recall "
        r"$\rightarrow$ Precision $\rightarrow$ Coverage $\rightarrow$ Consistency. "
        f"Each model: {runs_per_cell * len(CANONICAL_DATASETS)} runs.}}",
        r"\label{tab:forensicbench-6d-leaderboard}",
        r"\small",
        r"\begin{tabular}{r l r r r r r r}",
        r"\toprule",
        r"Rank & Model & \textbf{E-F1} & Type-F1 & Recall & Prec. & Cov. & Consist. \\",
        r"\midrule",
    ]
    for r in leaderboard:
        lines.append(
            f"{r.rank} & {r.model_label} & "
            f"\\textbf{{{_tex_pct(r.e_f1)}}} & {_tex_pct(r.type_f1)} & "
            f"{_tex_pct(r.recall)} & {_tex_pct(r.precision)} & "
            f"{_tex_pct(r.coverage)} & {r.consistency:.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _print_leaderboard(
    leaderboard: List[LeaderboardRow],
    *,
    md_path: Path,
    tex_path: Path,
    n_runs: int,
    runs_per_cell: int,
) -> None:
    n_sectors = len(CANONICAL_DATASETS)
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    print(
        f"  {len(leaderboard)} models — {n_runs} runs "
        f"({n_sectors}×{runs_per_cell} complete)"
    )
    print()
    header = (
        f"{'Rank':>4}  {'Model':<28}  "
        f"{'E-F1':>6}  {'Type-F1':>7}  {'Recall':>6}  {'Prec.':>6}  "
        f"{'Cov.':>6}  {'Consist.':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in leaderboard:
        print(
            f"{row.rank:4d}  {row.model_label:<28}  "
            f"{row.e_f1 * 100:6.1f}  {row.type_f1 * 100:7.1f}  "
            f"{row.recall * 100:6.1f}  {row.precision * 100:6.1f}  "
            f"{row.coverage * 100:6.1f}  {row.consistency:8.1f}"
        )


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
        "--md-out",
        default="",
        help="Markdown output (default: <results_root>/LEADERBOARD_5x5.md).",
    )
    parser.add_argument(
        "--tex-out",
        default="",
        help="LaTeX output (default: <results_root>/LEADERBOARD_5x5.tex).",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).resolve()
    if not results_root.is_dir():
        print(f"Not a directory: {results_root}", file=sys.stderr)
        return 1

    all_rows = _discover_runs(
        results_root,
        exclude_40m=not args.include_40m,
        exclude_10m=not args.include_10m,
    )
    rows = _filter_complete_grid(
        all_rows,
        datasets=CANONICAL_DATASETS,
        runs_per_cell=args.runs_per_cell,
    )
    if not rows:
        print(
            "No models with a complete 5×5 grid found.",
            file=sys.stderr,
        )
        return 1

    leaderboard = _build_leaderboard(rows)
    md_path = Path(args.md_out) if args.md_out else results_root / "LEADERBOARD_5x5.md"
    tex_path = Path(args.tex_out) if args.tex_out else results_root / "LEADERBOARD_5x5.tex"

    md_text = render_markdown(
        leaderboard,
        results_root,
        runs_per_cell=args.runs_per_cell,
    )
    tex_text = render_latex(leaderboard, runs_per_cell=args.runs_per_cell)

    md_path.write_text(md_text, encoding="utf-8")
    tex_path.write_text(tex_text, encoding="utf-8")

    _print_leaderboard(
        leaderboard,
        md_path=md_path,
        tex_path=tex_path,
        n_runs=len(rows),
        runs_per_cell=args.runs_per_cell,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
