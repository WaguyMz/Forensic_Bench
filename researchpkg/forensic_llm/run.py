"""
CLI entry point for the forensic LLM investigator.

Usage examples
--------------
# Full investigation with defaults (local vLLM on port 8020)
python -m forensic_llm.run

# Focused shadow-payroll task, custom model, 10M token budget
python -m forensic_llm.run \\
    --task shadow_payroll \\
    --model openai/gpt-oss-20b \\
    --base-url http://localhost:8020/v1 \\
    --max-tokens 10000000

# Anthropic Claude
python -m forensic_llm.run \\
    --provider anthropic \\
    --api-key <your-api-key> \\
    --task full

# Ensemble of 3 runs with evaluation
python -m forensic_llm.run \\
    --n-agents 3 \\
    --evaluate \\
    --output-dir ./results/ensemble_01

# Dry-run: only list DB schema and exit
python -m forensic_llm.run --dry-run

# Token budget model (orientation / planning / worker pool):
#   researchpkg/forensic_llm/docs/BUDGET_ORCHESTRATOR.md

# Custom tool set (SQL + scratchpad only, no plotting/CSV)
python -m forensic_llm.run \\
    --tools sql scratchpad report_suspicion finish_investigation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace

from researchpkg.config import (
    FORENSIC_OUTPUT_DIR,
    SYNTH_DATA_OUTPUT_DIR,
)

# Build config from args
from researchpkg.forensic_llm.config import (
    BudgetConfig,
    DatabaseConfig,
    InvestigatorConfig,
    LLMConfig,
)

# ---------------------------------------------------------------------------
# Logging setup (must happen before importing other modules)
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s – %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    # Quieten noisy libraries
    for noisy in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forensic_llm",
        description="Agentic LLM forensic investigator over journal-entry databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # LLM backend
    llm = p.add_argument_group("LLM backend")
    llm.add_argument(
        "--base-url",
        default=os.environ.get("FORENSIC_LLM_BASE_URL", "http://localhost:8020/v1"),
        help="OpenAI-compatible API base URL (vLLM / RunPod / OpenAI).",
    )
    llm.add_argument(
        "--api-key",
        default=os.environ.get("FORENSIC_LLM_API_KEY", "dummy"),
        help="API key (any non-empty string for vLLM).",
    )
    llm.add_argument(
        "--model",
        default=os.environ.get("FORENSIC_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        help="Model name as returned by GET /v1/models.",
    )
    llm.add_argument(
        "--tokenizer-model",
        default=os.environ.get("FORENSIC_TOKENIZER_MODEL") or None,
        help=(
            "HuggingFace Hub id for token counting when the API model id is not on HF "
            "(e.g. zai-org/GLM-5.1 for Eden zhipuai/GLM-5.1). "
            "Also set via FORENSIC_TOKENIZER_MODEL."
        ),
    )
    llm.add_argument(
        "--provider",
        choices=["openai_compatible", "anthropic"],
        default="openai_compatible",
        help="LLM provider / SDK to use.",
    )
    llm.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    llm.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling top_p.",
    )
    llm.add_argument(
        "--no-native-tools",
        action="store_true",
        help="Disable native tool calling; use ReAct text parsing instead.",
    )
    llm.add_argument(
        "--enable-thinking",
        action="store_true",
        default=os.environ.get("FORENSIC_LLM_ENABLE_THINKING", "0").strip().lower()
        in ("1", "true", "yes"),
        help=(
            "Best-effort: explicitly enable model thinking/reasoning mode on supported "
            "OpenAI-compatible backends (e.g. vLLM chat templates). "
            "Sends chat_template_kwargs={enable_thinking:true} in extra_body. "
            "Recommended for Qwen3 models; use temperature>=0.6 with this flag."
        ),
    )
    llm.add_argument(
        "--disable-thinking",
        action="store_true",
        default=os.environ.get("FORENSIC_LLM_DISABLE_THINKING", "0").strip().lower()
        in ("1", "true", "yes"),
        help="Best-effort: disable model thinking/reasoning mode on supported OpenAI-compatible backends (e.g. vLLM chat templates).",
    )
    llm.add_argument(
        "--llm-max-retries",
        type=int,
        default=max(
            0,
            int(os.environ.get("FORENSIC_LLM_MAX_RETRIES", "5")),
        ),
        help="Max extra attempts per LLM request after transient errors (429, 5xx, timeouts).",
    )

    # Database
    db = p.add_argument_group("Database")
    db.add_argument(
        "--db-host",
        default=os.environ.get("FORENSIC_DB_HOST", "localhost"),
    )
    db.add_argument(
        "--db-port",
        type=int,
        default=int(os.environ.get("FORENSIC_DB_PORT", 5432)),
    )
    db.add_argument(
        "--db-name",
        default=os.environ.get("FORENSIC_DB_NAME", "datasynth_forensic_public"),
    )
    db.add_argument(
        "--db-user",
        default=os.environ.get("FORENSIC_DB_USER", "test"),
    )
    db.add_argument(
        "--db-password",
        default=os.environ.get("FORENSIC_DB_PASSWORD", ""),
    )

    # Investigation
    inv = p.add_argument_group("Investigation")
    inv.add_argument(
        "--task",
        choices=[
            "full",
            "fictitious_ap_disbursements",
            "revenue_manipulation",
            "vendor_collusion",
            "shadow_payroll",
            "inventory_manipulation",
        ],
        default="full",
        help="Investigation focus / task.",
    )
    inv.add_argument(
        "--max-parallel-workers",
        type=int,
        default=None,
        help=(
            "Max concurrent hypothesis task workers as in-process threads "
            "(default 5; pass 1 for serial). Suited to I/O-bound LLM/DB work."
        ),
    )
    inv.add_argument(
        "--max-tokens",
        type=int,
        default=100_000_000,
        help="Run-wide input token budget (prompt + reasoning; completion not counted).",
    )
    inv.add_argument(
        "--max-steps",
        type=int,
        default=10_000_000_000,
        help="Maximum number of agentic tool-call steps (default: 1e10, effectively unlimited).",
    )
    inv.add_argument(
        "--n-agents",
        type=int,
        default=1,
        help="Number of independent agent runs (ensemble).",
    )
    inv.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for ensemble runs.",
    )
    inv.add_argument(
        "--output-dir",
        default=FORENSIC_OUTPUT_DIR,
        help="Directory for reports, plots, and traces.",
    )
    inv.add_argument(
        "--grep-root",
        default=SYNTH_DATA_OUTPUT_DIR,
        help="Root directory for the grep tool.",
    )
    inv.add_argument(
        "--tools",
        nargs="+",
        metavar="TOOL",
        default=None,
        choices=[
            "sql",
            "grep",
            "read_image",
            "scratchpad",
            "write_csv",
            "code_interpreter",
            "report_suspicion",
            "finish_investigation",
        ],
        help=(
            "Explicit list of tools to expose to the model. "
            "Default (when omitted): sql read_image scratchpad code_interpreter "
            "code_interpreter report_suspicion finish_investigation. "
            "Add 'grep' to extend the default set."
        ),
    )
    inv.add_argument(
        "--sql-min-quota-base",
        type=int,
        default=0,
        help="Deprecated (ignored): former minimum SQL floor for finish_investigation.",
    )
    inv.add_argument(
        "--sql-min-quota-per-core",
        type=int,
        default=0,
        help="Deprecated (ignored): former per-scheme minimum SQL floor.",
    )
    inv.add_argument(
        "--sql-max-per-core",
        type=str,
        default=None,
        help=(
            "Optional plan metadata only: sets budget_sql_calls in each scheme phase "
            "when the planner omits it. Does not force early continuation. "
            "Accepts an integer or 'auto' (max_tokens / 13)."
        ),
    )
    inv.add_argument(
        "--parallel-schemes",
        dest="parallel_scheme_execution",
        action="store_true",
        default=False,
        help=(
            "Legacy planning flag (v2 uses hypothesis_orchestrated + thread workers). "
            "When set, omits reflection_after_phases in the plan JSON only."
        ),
    )
    inv.add_argument(
        "--parallel-workers",
        dest="parallel_scheme_max_workers",
        type=int,
        default=5,
        metavar="K",
        help="Maximum concurrent scheme workers when --parallel-schemes is used (default: 5 = all schemes).",
    )
    inv.add_argument(
        "--parallel-orchestrator-weight",
        dest="parallel_orchestrator_token_slots",
        type=float,
        default=None,
        metavar="W",
        help=(
            "Orchestrator token-slot weight when --parallel-schemes is used (float OK). "
            "Budget unit = max_tokens / (n_schemes + W); orchestrator gets W units, "
            "each subagent gets 1 unit. Default 2.5: with 15M global and 5 schemes, "
            "each subagent gets 2M and orchestrator gets 5M."
        ),
    )

    # Evaluation (on by default)
    ev = p.add_argument_group("Evaluation")
    ev.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Score the run against anomaly_labels after completion (default: on).",
    )
    ev.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Skip evaluation after the run.",
    )
    ev.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for binary entry-level metrics.",
    )

    # Misc
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DB schema and available models, then exit.",
    )
    # Streaming trace is enabled by default; allow explicit disable.
    p.add_argument(
        "--stream-trace",
        dest="stream_trace",
        action="store_true",
        help="Append each step to run_dir/audit_trace_stream.ndjson and print a human-readable stream to stderr (default: on).",
    )
    p.add_argument(
        "--no-stream-trace",
        dest="stream_trace",
        action="store_false",
        help="Disable streaming trace (no ndjson file, no investigation stream in terminal).",
    )
    p.set_defaults(stream_trace=True)

    # Subcommands: default is 'run'; 'view' for trace viewer
    sub = p.add_subparsers(dest="command", help="Command (default: run)")
    view_p = sub.add_parser(
        "view", help="Full detailed and streaming view of audit trace"
    )
    view_p.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Run directory or path to audit_trace.json",
    )
    view_p.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow live stream (audit_trace_stream.ndjson in run dir).",
    )
    view_p.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate long tool results.",
    )
    view_p.add_argument(
        "--truncate",
        type=int,
        default=0,
        metavar="N",
        help="Max tokens per tool result (default from TextTruncationLimits, 0=no limit).",
    )

    return p


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def _dry_run(args: argparse.Namespace) -> None:
    print("\n=== Forensic LLM Investigator – Dry Run ===\n")

    # Test DB connection
    try:
        import psycopg2

        from researchpkg.forensic_llm.config import (
            DatabaseConfig,
        )

        _dry_db = DatabaseConfig(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password,
        )
        conn = psycopg2.connect(_dry_db.dsn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            print(f"Database: {args.db_name}@{args.db_host}:{args.db_port}")
            print(f"Tables: {', '.join(tables)}\n")

            for tbl in tables:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                count = cur.fetchone()[0]
                print(f"  {tbl}: {count:,} rows")
        conn.close()
    except Exception as exc:
        print(f"[DB ERROR] {exc}")

    # Test LLM endpoint
    print(f"\nLLM endpoint: {args.base_url}")
    try:
        import openai

        client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
        models = [m.id for m in client.models.list().data]
        print(f"Available models: {models}")
        print(f"Selected model: {args.model}")
    except Exception as exc:
        print(f"[LLM ERROR] {exc}")

    print("\nDry run complete. No investigation was run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    log = logging.getLogger("forensic_llm.run")

    # View subcommand: full detailed and streaming view of audit trace
    if getattr(args, "command", None) == "view":
        import pathlib

        from researchpkg.forensic_llm.trace_viewer import (
            view_static,
            view_stream,
        )

        path = getattr(args, "path", None)
        if not path:
            parser.error("view requires a path (run directory or audit_trace.json)")
        path = pathlib.Path(path)
        from researchpkg.forensic_llm.trace_viewer import (
            DEFAULT_RESULT_TRUNCATE_TOKENS,
        )

        result_truncate_tokens = (
            0
            if getattr(args, "no_truncate", False)
            else getattr(args, "truncate", DEFAULT_RESULT_TRUNCATE_TOKENS)
        )
        try:
            if getattr(args, "follow", False):
                view_stream(path, result_truncate_tokens=result_truncate_tokens)
            else:
                view_static(path, result_truncate_tokens=result_truncate_tokens)
            return 0
        except FileNotFoundError as e:
            log.error("%s", e)
            return 1

    if args.dry_run:
        _dry_run(args)
        return 0

    _env = InvestigatorConfig.from_env()

    llm_cfg = replace(
        _env.llm,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        provider=args.provider,
        use_native_tools=not args.no_native_tools,
        enable_thinking=bool(getattr(args, "enable_thinking", False)),
        disable_thinking=bool(getattr(args, "disable_thinking", False)),
        max_retries=max(0, int(getattr(args, "llm_max_retries", 5))),
        tokenizer_model=getattr(args, "tokenizer_model", None)
        or _env.llm.tokenizer_model,
    )
    db_cfg = DatabaseConfig(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )
    budget_cfg = BudgetConfig(
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
    )

    sql_max_per_core: int | None = None
    scheme_phase_budget_tokens_default: int | None = None
    orch_slots = getattr(args, "parallel_orchestrator_token_slots", None)
    if orch_slots is None:
        orch_slots = _env.parallel_orchestrator_token_slots
    if args.sql_max_per_core is not None:
        if str(args.sql_max_per_core).strip().lower() == "auto":
            # Fixed formula requested by user:
            # max_sql_per_core = max_tokens / (n_schemes + 5), with n_schemes=8.
            n_schemes = 5
            denom = n_schemes + float(orch_slots)
            denom = max(1, int(denom))
            sql_max_per_core = int(args.max_tokens // denom)
            scheme_phase_budget_tokens_default = int(args.max_tokens // denom)
        else:
            sql_max_per_core = int(args.sql_max_per_core)

    # Start from ``FORENSIC_*``-tuned defaults, then overlay CLI fields.
    config = replace(
        _env,
        llm=llm_cfg,
        database=db_cfg,
        budget=budget_cfg,
        task=args.task,
        output_dir=args.output_dir,
        n_agents=args.n_agents,
        seed=args.seed,
        run_evaluation=args.evaluate,
        enabled_tools=args.tools,
        enable_grep=bool(args.tools and "grep" in args.tools),
        grep_root=args.grep_root,
        stream_trace=getattr(args, "stream_trace", True),
        sql_min_quota_base=args.sql_min_quota_base,
        sql_min_quota_per_core=args.sql_min_quota_per_core,
        sql_max_per_core=sql_max_per_core,
        scheme_phase_budget_tokens_default=scheme_phase_budget_tokens_default,
        parallel_scheme_execution=getattr(args, "parallel_scheme_execution", False),
        parallel_scheme_max_workers=getattr(args, "parallel_scheme_max_workers", 5),
    )
    if getattr(args, "max_parallel_workers", None) is not None:
        config.max_parallel_workers = max(1, args.max_parallel_workers)
    if getattr(args, "parallel_orchestrator_token_slots", None) is not None:
        config.parallel_orchestrator_token_slots = (
            args.parallel_orchestrator_token_slots
        )
    config.apply_token_budgets_to_llm()

    log.info(
        "Config: model=%s provider=%s task=%s budget=%dM n_agents=%d context_window=%d",
        llm_cfg.model,
        llm_cfg.provider,
        config.task,
        config.budget.max_tokens // 1_000_000,
        config.n_agents,
        config.model_context_window,
    )

    # Run investigation(s)
    from researchpkg.forensic_llm.agent import (
        ForensicAgent,
        run_ensemble,
    )

    if config.n_agents == 1:
        agent = ForensicAgent(config)
        report = agent.run()
        reports = [report]
    else:
        reports = run_ensemble(config)

    eval_results = []
    if args.evaluate:
        from researchpkg.forensic_llm.evaluator import (
            EVAL_ELIGIBLE_FINISH_REASONS,
            compare_runs,
            evaluate_and_save,
        )

        for rpt in reports:
            if rpt.termination_reason not in EVAL_ELIGIBLE_FINISH_REASONS:
                log.warning(
                    "Skipping evaluation for run %s: terminated with reason=%r "
                    "(only %s runs are evaluated).",
                    rpt.run_id[:8],
                    rpt.termination_reason,
                    "/".join(sorted(EVAL_ELIGIBLE_FINISH_REASONS)),
                )
                continue
            # Save evaluation into the same run directory as the report
            er = evaluate_and_save(
                rpt,
                db_cfg,
                output_dir=rpt.run_id
                and _find_run_dir(args.output_dir, rpt.run_id)
                or args.output_dir,
                confidence_threshold=args.confidence_threshold,
            )
            eval_results.append(er)

        if len(eval_results) > 1:
            table = compare_runs(eval_results)
            print("\n" + table)
            cmp_path = os.path.join(args.output_dir, "ensemble_comparison.md")
            with open(cmp_path, "w") as fh:
                fh.write("# Ensemble Run Comparison\n\n" + table)
            log.info("Comparison table saved to %s", cmp_path)
        elif eval_results:
            er = eval_results[0]
            print(
                f"\nEvaluation results (JE-level):"
                f"\n  Accuracy  : {er.entry_accuracy:.3f}"
                f"\n  Precision : {er.entry_precision:.3f}"
                f"\n  Recall    : {er.entry_recall:.3f}"
                f"\n  F1        : {er.entry_f1:.3f}"
                f"\n  ROC-AUC   : {er.roc_auc or 'N/A'}"
                f"\n  PR-AUC    : {er.pr_auc or 'N/A'}"
                f"\n  Flagged   : {er.n_flagged} / {er.n_true_anomalies} true anomalies"
            )
            if er.scheme_eval and er.scheme_eval.n_true_schemes > 0:
                se = er.scheme_eval
                print(
                    f"\nEvaluation results (scheme-level):"
                    f"\n  Detection Rate (SDR) : {se.scheme_detection_rate:.3f}"
                    f"\n  Scheme Precision     : {se.scheme_precision:.3f}"
                    f"\n  Scheme F1            : {se.scheme_f1:.3f}"
                    f"\n  Mean Coverage        : {se.mean_coverage:.3f}"
                    f"\n  Perpetrator ID Rate  : {se.perpetrator_identification_rate:.3f}"
                    f"\n  Detected schemes     : {se.n_detected} / {se.n_true_schemes}"
                )

    # Final summary
    print()
    for rpt in reports:
        run_dir = _find_run_dir(args.output_dir, rpt.run_id)
        print(
            f"Run {rpt.run_id[:8]}:"
            f"  {len(rpt.suspicion_list):>3} suspicion(s)"
            f"  |  {rpt.steps_taken:>3} steps"
            f"  |  {rpt.total_tokens_input:>10,} prompt tokens (budget)"
            f"  |  {rpt.total_tokens_input + rpt.total_tokens_output:>10,} billed"
        )
        if run_dir:
            print(f"  └─ {run_dir}/")
            print(
                f"       narrative.md  suspicion_list.json  audit_trace.json"
                f"  budget_summary.json  run_manifest.json  plots/"
            )

    return 0


def _find_run_dir(base: str, run_id: str) -> str:
    """Return the path to the run subdirectory that contains this run_id."""
    import pathlib

    for d in pathlib.Path(base).iterdir():
        if d.is_dir() and d.name.endswith(run_id[:8]):
            return str(d)
    return base


if __name__ == "__main__":
    sys.exit(main())
