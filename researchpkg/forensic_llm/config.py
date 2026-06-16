"""
Forensic LLM Investigator – configuration.

Supports any OpenAI-compatible endpoint (vLLM, RunPod, OpenAI) as well as
the native Anthropic SDK.  All settings can be overridden via environment
variables for easy deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Settings for the LLM used by the agent."""

    # Base URL for an OpenAI-compatible API server (vLLM / RunPod / OpenAI).
    # For a local vLLM server: "http://localhost:8020/v1"
    # For RunPod: "https://<pod-id>-8000.proxy.runpod.net/v1"
    # For OpenAI: "https://api.openai.com/v1"
    base_url: str = "http://localhost:8020/v1"

    # API key.  vLLM accepts any non-empty string.
    api_key: str = "dummy"

    # Model identifier as reported by GET /v1/models.
    model: str = "Qwen/Qwen3.5-35B-A3B"

    # HuggingFace tokenizer id for token counting (defaults to ``model``).
    tokenizer_model: Optional[str] = None
    tokenizer_trust_remote_code: bool = True

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.8
    # Seed passed to the chat completion API for reproducibility (temperature=0
    # alone is not sufficient with GPU non-determinism and thinking models).
    seed: Optional[int] = None

    # Maximum *new* tokens per chat completion (copied from ``ContextTokenConfig``
    # in ``InvestigatorConfig.apply_token_budgets_to_llm()`` unless
    # ``FORENSIC_LLM_MAX_TOKENS_PER_STEP`` set ``max_tokens_per_step_explicit``).
    max_tokens_per_step: int = 16_384

    # True when max_tokens_per_step came from the environment (skip sync from tokens).
    max_tokens_per_step_explicit: bool = field(default=False, repr=False)

    # Maximum completion tokens for the one-shot planning step (large JSON plan).
    # Kept in lockstep with ``ContextTokenConfig.max_tokens_planning`` (apply_token_budgets_to_llm).
    max_tokens_planning: int = 16_384

    # Attempt native OpenAI function/tool calling first.
    use_native_tools: bool = True

    # Explicitly enable model "thinking"/reasoning mode on supported backends
    # (vLLM chat-template kwarg `enable_thinking=true`).  When True the client
    # sends {"chat_template_kwargs": {"enable_thinking": true}} in extra_body.
    # Recommended for Qwen3 and other models with a dedicated reasoning stream.
    # For Qwen3 thinking mode, Alibaba recommends temperature=0.6 (not 0).
    # Mutually exclusive with disable_thinking; enable_thinking takes precedence.
    enable_thinking: bool = False

    # Disable model "thinking"/reasoning mode when supported by the backend
    # (primarily for OpenAI-compatible servers like vLLM that expose chat-template
    # kwargs such as `enable_thinking=false`).
    #
    # NOTE: This is a best-effort flag. It is ignored for providers/backends
    # that do not support such controls.
    disable_thinking: bool = False

    # HTTP timeout for a single LLM request (seconds). Use 300+ for long runs
    # so final/forced-finish responses have time to complete.
    request_timeout: float = 1200

    # Retries for a single chat completion on transient errors (network, 429,
    # 5xx). The first attempt is not counted; max_retries=5 allows up to six
    # attempts total.
    max_retries: int = 10

    # "openai_compatible"  — use the openai SDK with a custom base_url.
    # "anthropic"          — use the anthropic SDK (set api_key properly).
    provider: str = "openai_compatible"

    # Anthropic-specific model name (ignored when provider != "anthropic").
    anthropic_model: str = "N/A"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        cfg = cls()
        cfg.base_url = os.environ.get("FORENSIC_LLM_BASE_URL", cfg.base_url)
        cfg.api_key = os.environ.get("FORENSIC_LLM_API_KEY", cfg.api_key)
        cfg.model = os.environ.get("FORENSIC_LLM_MODEL", cfg.model)
        cfg.temperature = float(os.environ.get("FORENSIC_LLM_TEMP", cfg.temperature))
        cfg.top_p = float(os.environ.get("FORENSIC_LLM_TOP_P", cfg.top_p))
        cfg.provider = os.environ.get("FORENSIC_LLM_PROVIDER", cfg.provider)
        if "FORENSIC_LLM_ENABLE_THINKING" in os.environ:
            cfg.enable_thinking = os.environ.get(
                "FORENSIC_LLM_ENABLE_THINKING", "0"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_LLM_DISABLE_THINKING" in os.environ:
            cfg.disable_thinking = os.environ.get(
                "FORENSIC_LLM_DISABLE_THINKING", "0"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_LLM_REQUEST_TIMEOUT" in os.environ:
            cfg.request_timeout = float(os.environ.get("FORENSIC_LLM_REQUEST_TIMEOUT"))
        # Hub tokenizer id for local token counting (defaults to FORENSIC_LLM_MODEL).
        # When the API model id is not on HF (e.g. Eden zhipuai/GLM-5.1), set this to
        # the official repo (zai-org/GLM-5.1) or rely on model_tokenizer auto-mapping
        # for edenai.run endpoints.
        if "FORENSIC_TOKENIZER_MODEL" in os.environ:
            cfg.tokenizer_model = os.environ["FORENSIC_TOKENIZER_MODEL"]
        if "FORENSIC_TOKENIZER_TRUST_REMOTE_CODE" in os.environ:
            cfg.tokenizer_trust_remote_code = os.environ.get(
                "FORENSIC_TOKENIZER_TRUST_REMOTE_CODE", "1"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_LLM_MAX_RETRIES" in os.environ:
            cfg.max_retries = max(
                0, int(os.environ.get("FORENSIC_LLM_MAX_RETRIES", "5"))
            )
        if "FORENSIC_LLM_MAX_TOKENS_PER_STEP" in os.environ:
            cfg.max_tokens_per_step = max(
                1, int(os.environ["FORENSIC_LLM_MAX_TOKENS_PER_STEP"])
            )
            cfg.max_tokens_per_step_explicit = True
        if "FORENSIC_LLM_MAX_TOKENS_PLANNING" in os.environ:
            cfg.max_tokens_planning = max(
                1, int(os.environ["FORENSIC_LLM_MAX_TOKENS_PLANNING"])
            )
        return cfg


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@dataclass
class DatabaseConfig:
    """PostgreSQL connection settings (read-only usage enforced at runtime)."""

    host: str = "localhost"
    port: int = 5432
    # By default the investigator connects to the *sanitised* copy of the
    # forensic database, which omits label columns such as `is_fraud`.
    # The full labelled database (e.g. for manual analysis) can live in a
    # separate database name.
    database: str = "datasynth_forensic_public"
    user: str = "postgres"
    password: str = None

    # Per-query statement timeout (ms).  Prevents runaway table scans.
    statement_timeout_ms: int = 30_000

    # Default maximum rows the sql() tool returns.
    default_max_rows: int = 2000

    # Absolute hard cap (agent cannot exceed this even if it asks).
    hard_max_rows: int = 10_000

    @property
    def dsn(self) -> str:
        # When host is localhost and no password is set, omit the host so
        # psycopg2 uses the UNIX domain socket (peer authentication).
        use_socket = (
            self.host in ("localhost", "127.0.0.1", "::1") and not self.password
        )
        if use_socket:
            return f"dbname={self.database} user={self.user}"
        if self.password:
            return (
                f"host={self.host} port={self.port} dbname={self.database} "
                f"user={self.user} password={self.password}"
            )
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user}"
        )

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        cfg = cls()
        cfg.host = os.environ.get("FORENSIC_DB_HOST", cfg.host)
        cfg.port = int(os.environ.get("FORENSIC_DB_PORT", cfg.port))
        cfg.database = os.environ.get("FORENSIC_DB_NAME", cfg.database)
        cfg.user = os.environ.get("FORENSIC_DB_USER", cfg.user)
        cfg.password = os.environ.get("FORENSIC_DB_PASSWORD", cfg.password)
        return cfg


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Controls how many tokens the agent may consume."""

    # Run-wide **input** token budget (prompt + separately reported reasoning).
    # Completion/output tokens are tracked for reporting only (see ``BudgetTracker``).
    max_tokens: int = 100_000_000

    # Hard limit on tool-call iterations per run (default effectively unlimited).
    max_steps: int = 10_000_000_000  # 1e10

    # Log a warning once usage crosses this fraction.
    warn_threshold: float = 0.80

    # Stop issuing new tool calls at this fraction; let the model finalize.
    stop_threshold: float = 0.95


# ---------------------------------------------------------------------------
# Context window, prompt slots, completions, and excerpt truncation (tokens)
# ---------------------------------------------------------------------------


@dataclass
class TextTruncationLimits:
    """
    Token caps for stored excerpts and prompt injection.

    ``0`` on any field means no limit. Override any field via
    ``FORENSIC_TRUNC_<FIELD_NAME_UPPER>_TOKENS`` (see ``from_env``).
    """

    # Legacy name: cap for persisted orientation digest in summary.json (unused when
    # planning reads orientation_report.md directly).
    orientation_summary_store: int = 100_000
    # Max tokens from orientation/orientation_report.md fed into planning (head slice).
    planning_orientation_prompt: int = 100_000
    # Max tokens for the live orientation report slot during screening (not chat history).
    orientation_report_slot_tokens: int = 64_000
    # Max tokens for ephemeral [Current Step] chat (SQL + last assistant/tool turn).
    orientation_current_step_slot_tokens: int = 8_000
    # Max tokens for prior orientation turns (assistant/tool bundles), slot "recent".
    orientation_recent_slot_tokens: int = 16_000
    # Max SQL preview data rows shown in ephemeral tool feedback during orientation.
    orientation_sql_context_max_rows: int = 5
    # Max scratchpad + manifest tokens fed into orientation digest compression (tokenizer).
    orientation_memo_synthesis_input: int = 110_000
    # Max new tokens for orientation digest LLM passes (fits planning excerpt work).
    orientation_memo_synthesis_output: int = 16_000
    global_memory_excerpt: int = 4_000
    shared_context_blob: int = 12_000
    shared_context_orientation: int = 2_000
    shared_context_blackboard: int = 4_000
    shared_context_key_finding: int = 300
    shared_context_peer_finding: int = 200
    hypothesis_text_store: int = 1_500
    hypothesis_rationale_store: int = 2_000
    key_findings_store: int = 8_000
    salient_finding: int = 1_200
    parent_context: int = 4_000
    worker_orientation_snippet: int = 2_000
    worker_plan_context: int = 1_000
    worker_parent_context: int = 3_000
    hypothesis_worker_shared_memory: int = 8_000
    hypothesis_worker_hypothesis_text: int = 800
    hypothesis_worker_rationale: int = 1_200
    orchestrator_followup_json: int = 6_000
    orchestrator_followup_memory: int = 4_000
    auto_scratchpad_tail: int = 8_000
    planning_error_content: int = 600
    planning_error_reasoning: int = 400
    injection_scratchpad: int = 2_000
    injection_hypothesis_preview: int = 400
    hypothesis_track_scratchpad: int = 8_000
    trace_tool_result: int = 4_000
    trace_narrative: int = 8_000
    trace_rationale: int = 200
    memory_excerpt_clip: int = 700
    rationale_short: int = 120
    rationale_medium: int = 240
    evidence_hypothesis_line: int = 200
    worker_notes: int = 500
    track_reasoning_preview: int = 500
    track_sql_query_preview: int = 400
    track_tool_code_preview: int = 300
    track_tool_note_preview: int = 200
    track_json_preview: int = 200

    @classmethod
    def from_env(
        cls, base: Optional["TextTruncationLimits"] = None
    ) -> "TextTruncationLimits":
        lim = base or cls()
        for f in fields(cls):
            env_key = f"FORENSIC_TRUNC_{f.name.upper()}_TOKENS"
            if env_key in os.environ:
                setattr(lim, f.name, int(os.environ[env_key]))
        return lim


@dataclass
class ContextTokenConfig:
    """
    Absolute token limits for **model context** (default 128k window).

    Slots, completions, safety margin, and excerpt caps only — not the run-wide
    token budget (see ``InvestigatorConfig.orientation_budget_fraction`` and
    ``BudgetConfig.max_tokens``). ``InvestigatorConfig`` delegates attribute
    access here for context fields; excerpts live in ``excerpts``.
    """

    # Must match the served model (e.g. vLLM ``max_model_len``).
    model_context_window: int = 131_072

    # Fraction of ``model_context_window`` withheld so prompt + completion stay
    # below the hard API limit (default 2% → 125_440 effective on a 128k model).
    context_window_margin_fraction: float = 0.02

    # Extra fixed slack for tokenizer vs server counting differences.
    context_safety_margin_tokens: int = 2048

    # Per API call — completion / output (``max_tokens`` in chat completions).
    max_tokens_per_step: int = 16_384
    max_tokens_planning: int = 16_384

    # Multi-slot prompt assembly ceilings (see ``ForensicAgent._build_slot_payload``).
    # Plan slot is 8k; past / scratch / recent / input are 2× the former 4k/10k/30.8k/4k layout.
    slot_plan_tokens: int = 8_000
    slot_past_tokens: int = 8_192
    slot_scratchpad_tokens: int = 20_480
    slot_recent_tokens: int = 61_600
    slot_input_tokens: int = 8_192

    # Packing slack inside ``max_input`` before splitting the slot pool (2× prior defaults).
    pack_system_reserve_tokens: int = 8_192
    pack_buffer_tokens: int = 2_048
    slot_pack_weights: Tuple[int, int, int, int, int] = (1, 1, 2, 1, 6)

    excerpts: TextTruncationLimits = field(default_factory=TextTruncationLimits)

    def effective_context_window(self) -> int:
        """Hard window minus the fractional margin (e.g. 128k → 125_440 at 2%)."""
        w = int(self.model_context_window)
        frac = max(0.0, min(0.25, float(self.context_window_margin_fraction)))
        if frac > 0:
            w = max(1024, int(w * (1.0 - frac)))
        return w

    def max_input_tokens(self, completion_reserve: Optional[int] = None) -> int:
        """API-safe maximum prompt tokens for one request."""
        out = (
            int(completion_reserve)
            if completion_reserve is not None
            else int(self.max_tokens_per_step)
        )
        return max(
            1024,
            self.effective_context_window()
            - out
            - int(self.context_safety_margin_tokens),
        )

    @classmethod
    def from_env(
        cls, base: Optional["ContextTokenConfig"] = None
    ) -> "ContextTokenConfig":
        cfg = base or cls()
        _env_map = {
            "FORENSIC_MODEL_CONTEXT_WINDOW": "model_context_window",
            "FORENSIC_CONTEXT_WINDOW_MARGIN_FRACTION": "context_window_margin_fraction",
            "FORENSIC_CONTEXT_SAFETY_MARGIN": "context_safety_margin_tokens",
            "FORENSIC_LLM_MAX_TOKENS_PER_STEP": "max_tokens_per_step",
            "FORENSIC_LLM_MAX_TOKENS_PLANNING": "max_tokens_planning",
            "FORENSIC_SLOT_PLAN_TOKENS": "slot_plan_tokens",
            "FORENSIC_SLOT_PAST_TOKENS": "slot_past_tokens",
            "FORENSIC_SLOT_SCRATCHPAD_TOKENS": "slot_scratchpad_tokens",
            "FORENSIC_SLOT_RECENT_TOKENS": "slot_recent_tokens",
            "FORENSIC_SLOT_INPUT_TOKENS": "slot_input_tokens",
            "FORENSIC_PACK_SYSTEM_RESERVE_TOKENS": "pack_system_reserve_tokens",
            "FORENSIC_PACK_BUFFER_TOKENS": "pack_buffer_tokens",
        }
        for env_key, attr in _env_map.items():
            if env_key in os.environ:
                val = os.environ[env_key]
                if attr == "context_window_margin_fraction":
                    setattr(cfg, attr, float(val))
                else:
                    setattr(cfg, attr, int(val))
        cfg.excerpts = TextTruncationLimits.from_env(cfg.excerpts)
        return cfg


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class InvestigatorConfig:
    """Master configuration for a forensic investigation run."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    tokens: ContextTokenConfig = field(default_factory=ContextTokenConfig)

    # Investigation focus.
    # "full"                           — open-ended investigation (canonical benchmark mode).
    # "fictitious_ap_disbursements"    — focused on fictitious AP / expense-laundering patterns.
    # "revenue_manipulation"           — focused on window-dressing / period-end manipulation.
    # "vendor_collusion"               — focused on kickback + related-party schemes.
    # "shadow_payroll"                 — focused on ghost employee / payroll fraud.
    # "inventory_manipulation"         — focused on fictitious receipts / write-off abuse.
    task: str = "full"

    # Per-scheme hypothesis count guardrails (enforced after planning).
    min_hypotheses_per_scheme: int = 5
    max_hypotheses_per_scheme: int = 10
    # Target total dispatch_queue size (~5 schemes × min_hypotheses); used in planning prompt.
    planning_min_dispatch_items: int = 25

    # Concurrent hypothesis task workers (default 5 for throughput; use 1 for
    # maximally reproducible ordering). Workers run in **threads** (not OS
    # processes): the bottleneck is I/O (LLM API, DB). Override via
    # --max-parallel-workers or FORENSIC_MAX_PARALLEL_WORKERS.
    max_parallel_workers: int = 5

    # After each completed hypothesis task, the orchestrator may enqueue follow-up
    # hypotheses from open questions / findings (not fixed at plan time).
    # Optional mid-run hypothesis enqueue (rule-based + LLM orchestrator_inject).
    # Disabled by default for benchmark-grade runs: the dispatch queue is fixed at
    # planning time (plan → execute → programmatic merge). Set
    # FORENSIC_DYNAMIC_HYPOTHESIS_INJECTION=1 to re-enable exploratory injection.
    dynamic_hypothesis_injection: bool = False
    max_dynamic_hypotheses: int = 40

    # Run-wide token budget: fraction of ``budget.max_tokens`` (global) for orientation.
    # See ``forensic_llm/docs/BUDGET_ORCHESTRATOR.md`` for the full orchestrator model.
    orientation_budget_fraction: float = 0.05  # 5% of orientation budget

    # Expect orientation profiling to use roughly this fraction of the orientation cap
    # before considering ``complete_orientation`` (scratchpad + nudges).
    orientation_budget_encourage_deep_until_fraction: float = 0.70

    # Reject ``complete_orientation`` until at least this fraction of the orientation
    # cap has been consumed (input tokens), unless the global budget forces exit.
    orientation_budget_min_fraction_for_complete: float = 0.30

    # When True, inject brief user-role band nudges (when protocol allows) so the
    # model does not wrap orientation after shallow work.
    orientation_budget_band_nudges: bool = True

    # When the orchestrator splits remaining budget across hypothesis tasks, this
    # fraction is reserved for orchestrator overhead (planning, follow-ups, etc.).
    orchestrator_reserve_budget_fraction: float = 0.05

    # When True, each ``dispatch_queue`` row's ``budget_tokens`` from the planning
    # LLM is a **complexity weight**; the runtime scales weights to fill the worker
    # token pool after planning. When False, legacy per-dispatch equal split of
    # ``remaining // pending`` is used instead.
    use_planner_weighted_task_budgets: bool = True

    # Extra multiplier on payload token estimates (1.0 when using the model
    # tokenizer; increase only if you still see validation overflows).
    context_estimate_inflation: float = 1.0

    # Stop a hypothesis task before the next LLM turn when usage reaches this
    # fraction of the task's allocated budget_tokens (1.0 = use full allocation).
    task_budget_stop_fraction: float = 1.0

    # Option B: re-split remaining hypothesis tokens across pending queue items after
    # each task (deterministic min + weighted pool). When False, budgets are fixed at
    # planning time (``apply_worker_budget_pool_to_dispatch_queue`` once).
    use_runtime_hypothesis_budget_pool: bool = True

    # Floor per pending hypothesis when applying the runtime pool (tokens counted like
    # ``BudgetTracker.used_tokens`` for the task worker).
    hypothesis_task_min_tokens: int = 25_000

    # Inject user pacing hints at this **task** budget fraction (informational).
    task_budget_warn_fraction: float = 0.80

    # Urgent: call ``report_suspicion`` if confirmed before this fraction of the task cap.
    task_budget_report_deadline_fraction: float = 0.90

    # Directory for output artefacts (reports, plots, traces).
    output_dir: str = "./forensic_output"

    # RNG seed for reproducible ensemble runs.
    seed: int = 42

    # Number of independent agent runs (different system-prompt orderings).
    n_agents: int = 1

    # Whether to score the run against anomaly_labels after completion.
    run_evaluation: bool = True

    # Maximum BFS depth for the graph_query tool.
    graph_depth: int = 5

    # Deprecated: minimum SQL floors are no longer enforced (kept for CLI compat).
    sql_min_quota_base: int = 0
    sql_min_quota_per_core: int = 0

    # Optional per-scheme budget_sql_calls hint written into the plan JSON when set.
    # Not used for early termination — phases end on token cap or model stop.
    sql_max_per_core: Optional[int] = None

    # Default per-scheme token budget used when the planner omits
    # "budget_tokens" in a SchemePhase. This is distinct from the run-wide
    # BudgetConfig.max_tokens which caps the whole run (or orchestrator, in
    # parallel mode). The CLI can derive this from the same "per-core" split
    # as sql_max_per_core when set to "auto".
    scheme_phase_budget_tokens_default: Optional[int] = None

    # Minimum fraction of token budget that must be consumed before
    # finish_investigation is accepted (unless budget is truly exhausted).
    # Set to 0.0 to disable the budget-usage termination guardrail.
    min_budget_fraction_for_finish: float = 0.0

    # Explicit list of tool names to expose to the model.
    # When set, takes precedence over enable_grep / enable_graph_tools.
    # Valid names: sql, grep, read_image, scratchpad, write_csv,
    #              code_interpreter, report_suspicion, finish_investigation.
    # When None, the tool set is derived from enable_grep + enable_graph_tools.
    enabled_tools: Optional[List[str]] = None

    # Deprecated: graph_query was removed from tool_defs (use sql export + code_interpreter).
    enable_graph_tools: bool = False

    # When True, the grep tool may be exposed (see enabled_tools / agent wiring).
    # Persisted in run metadata; FORENSIC_ENABLE_GREP can override via from_env().
    enable_grep: bool = False

    # Root path that grep searches are anchored to (when enable_grep is True).
    grep_root: str = "./output"

    # When True, append each completed step to run_dir/audit_trace_stream.ndjson
    # so a viewer can run `view --follow run_dir` for a live streaming trace.
    # Enabled by default so every run has a live-streamable trace unless
    # explicitly disabled by the caller/CLI.
    stream_trace: bool = True

    # When True (default), each emitted step is also printed to stderr in
    # human-readable form.  Set to False for subagent workers so that 10
    # concurrent streams do not produce interleaved terminal output while
    # still writing their own audit_trace_stream.ndjson files.
    stream_trace_terminal: bool = True

    # ---------------------------------------------------------------------------
    # Parallel scheme execution (legacy parent / subagent layout)
    #
    # When enabled with ``task == "full"``, the global token budget is split
    # between an orchestrator cap and per-scheme workers (see
    # ``parallel_orchestrator_token_slots``). The v2 hypothesis loop uses
    # ``max_parallel_workers`` on the dispatch queue instead of this path unless
    # tooling explicitly enables parallel_scheme_execution.
    # ---------------------------------------------------------------------------

    # Legacy flag: affects planning metadata only in v2 (hypothesis_orchestrated).
    # Hypothesis dispatch uses ``max_parallel_workers`` threads, not subprocesses.
    parallel_scheme_execution: bool = False

    # When ``parallel_scheme_execution`` and ``task == "full"``, the CLI
    # ``budget.max_tokens`` value is treated as a **global** prompt-token budget
    # for the whole run.  One unit is ``max_tokens // (n_schemes + weight)``;
    # each subagent gets one unit; the orchestrator gets ``weight`` units
    # (orientation, planning, reflection, cross-scheme, synthesis, etc.).
    #
    # Reduced from 7 to 3: empirically the orchestrator uses ~1.8 M tokens out
    # Budget formula: per_agent = global / (n_schemes + orchestrator_weight).
    # With global=15M, n_schemes=5, weight=2.5:
    #   per_agent = 15M / 7.5 = 2M exactly; orchestrator = 2.5 × 2M = 5M.
    # float is supported so the denominator can be non-integer.
    parallel_orchestrator_token_slots: float = 2.5

    # Scheme count assumed before the investigation plan exists (matches the 8
    # core catalogue used in planning).  Reconciled against ``len(plan.phases)``
    # after planning when they differ.
    parallel_token_budget_num_schemes: int = 5

    # Optional experiment-wide token budget for reporting when ``budget.max_tokens``
    # holds a per-actor cap (parallel subagents).  When None, ``budget.max_tokens``
    # is used for both.
    token_budget_global_max: Optional[int] = None

    # Maximum number of scheme workers running concurrently.  Workers are
    # IO-bound (LLM + DB calls) so all 10 can run at the same time on a
    # single machine without CPU contention.  Reduce if DB connection limits
    # or LLM API rate-limits are a concern.
    parallel_scheme_max_workers: int = 10

    def __post_init__(self) -> None:
        self.apply_token_budgets_to_llm()

    def apply_token_budgets_to_llm(self) -> None:
        """Sync ``LLMConfig`` completion caps from ``tokens`` (absolute, not fractional)."""
        if not self.llm.max_tokens_per_step_explicit:
            self.llm.max_tokens_per_step = int(self.tokens.max_tokens_per_step)
        self.llm.max_tokens_planning = int(self.tokens.max_tokens_planning)

    def reconcile_derived_context_budgets(self) -> None:
        """Backward-compatible alias for ``apply_token_budgets_to_llm()``."""
        self.apply_token_budgets_to_llm()

    def __getattr__(self, name: str) -> Any:
        if name == "text_limits":
            return self.tokens.excerpts
        if name in ContextTokenConfig.__dataclass_fields__:
            return getattr(self.tokens, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dataclass_fields__:
            object.__setattr__(self, name, value)
            return
        if name == "text_limits":
            object.__setattr__(self.tokens, "excerpts", value)
            return
        if name in ContextTokenConfig.__dataclass_fields__:
            setattr(self.tokens, name, value)
            return
        object.__setattr__(self, name, value)

    @classmethod
    def from_env(cls) -> "InvestigatorConfig":
        cfg = cls(llm=LLMConfig.from_env())
        cfg.database = DatabaseConfig.from_env()
        cfg.output_dir = os.environ.get("FORENSIC_OUTPUT_DIR", cfg.output_dir)
        cfg.task = os.environ.get("FORENSIC_TASK", cfg.task)
        cfg.tokens = ContextTokenConfig.from_env(cfg.tokens)
        if cfg.llm.max_tokens_per_step_explicit:
            cfg.tokens.max_tokens_per_step = cfg.llm.max_tokens_per_step
        if "FORENSIC_DYNAMIC_HYPOTHESIS_INJECTION" in os.environ:
            cfg.dynamic_hypothesis_injection = os.environ.get(
                "FORENSIC_DYNAMIC_HYPOTHESIS_INJECTION", "0"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_MAX_PARALLEL_WORKERS" in os.environ:
            cfg.max_parallel_workers = max(
                1, int(os.environ["FORENSIC_MAX_PARALLEL_WORKERS"])
            )
        if "FORENSIC_MIN_HYPOTHESES_PER_SCHEME" in os.environ:
            cfg.min_hypotheses_per_scheme = max(
                1, int(os.environ["FORENSIC_MIN_HYPOTHESES_PER_SCHEME"])
            )
        if "FORENSIC_MAX_HYPOTHESES_PER_SCHEME" in os.environ:
            cfg.max_hypotheses_per_scheme = max(
                1, int(os.environ["FORENSIC_MAX_HYPOTHESES_PER_SCHEME"])
            )
        if "FORENSIC_PLANNING_MIN_DISPATCH_ITEMS" in os.environ:
            cfg.planning_min_dispatch_items = max(
                1, int(os.environ["FORENSIC_PLANNING_MIN_DISPATCH_ITEMS"])
            )
        if "FORENSIC_ORIENTATION_BUDGET_FRACTION" in os.environ:
            cfg.orientation_budget_fraction = float(
                os.environ["FORENSIC_ORIENTATION_BUDGET_FRACTION"]
            )
        if "FORENSIC_ORIENTATION_ENCOURAGE_DEEP_UNTIL_FRACTION" in os.environ:
            cfg.orientation_budget_encourage_deep_until_fraction = float(
                os.environ["FORENSIC_ORIENTATION_ENCOURAGE_DEEP_UNTIL_FRACTION"]
            )
        if "FORENSIC_ORIENTATION_MIN_FRACTION_FOR_COMPLETE" in os.environ:
            cfg.orientation_budget_min_fraction_for_complete = float(
                os.environ["FORENSIC_ORIENTATION_MIN_FRACTION_FOR_COMPLETE"]
            )
        if "FORENSIC_ORIENTATION_BAND_NUDGES" in os.environ:
            cfg.orientation_budget_band_nudges = os.environ.get(
                "FORENSIC_ORIENTATION_BAND_NUDGES", "1"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_ORCHESTRATOR_RESERVE_BUDGET_FRACTION" in os.environ:
            cfg.orchestrator_reserve_budget_fraction = float(
                os.environ["FORENSIC_ORCHESTRATOR_RESERVE_BUDGET_FRACTION"]
            )
        if "FORENSIC_USE_PLANNER_WEIGHTED_TASK_BUDGETS" in os.environ:
            cfg.use_planner_weighted_task_budgets = os.environ.get(
                "FORENSIC_USE_PLANNER_WEIGHTED_TASK_BUDGETS", "1"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_TASK_BUDGET_STOP_FRACTION" in os.environ:
            cfg.task_budget_stop_fraction = float(
                os.environ["FORENSIC_TASK_BUDGET_STOP_FRACTION"]
            )
        if "FORENSIC_USE_RUNTIME_HYPOTHESIS_BUDGET_POOL" in os.environ:
            cfg.use_runtime_hypothesis_budget_pool = os.environ.get(
                "FORENSIC_USE_RUNTIME_HYPOTHESIS_BUDGET_POOL", "1"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_HYPOTHESIS_TASK_MIN_TOKENS" in os.environ:
            cfg.hypothesis_task_min_tokens = max(
                1, int(os.environ["FORENSIC_HYPOTHESIS_TASK_MIN_TOKENS"])
            )
        if "FORENSIC_TASK_BUDGET_WARN_FRACTION" in os.environ:
            cfg.task_budget_warn_fraction = float(
                os.environ["FORENSIC_TASK_BUDGET_WARN_FRACTION"]
            )
        if "FORENSIC_TASK_BUDGET_REPORT_DEADLINE_FRACTION" in os.environ:
            cfg.task_budget_report_deadline_fraction = float(
                os.environ["FORENSIC_TASK_BUDGET_REPORT_DEADLINE_FRACTION"]
            )
        if "FORENSIC_ENABLE_GREP" in os.environ:
            cfg.enable_grep = os.environ.get(
                "FORENSIC_ENABLE_GREP", "0"
            ).strip().lower() in ("1", "true", "yes")
        if "FORENSIC_SQL_MIN_QUOTA_BASE" in os.environ:
            cfg.sql_min_quota_base = int(os.environ.get("FORENSIC_SQL_MIN_QUOTA_BASE"))
        if "FORENSIC_SQL_MIN_QUOTA_PER_CORE" in os.environ:
            cfg.sql_min_quota_per_core = int(
                os.environ.get("FORENSIC_SQL_MIN_QUOTA_PER_CORE")
            )
        if "FORENSIC_MIN_BUDGET_FRACTION_FOR_FINISH" in os.environ:
            cfg.min_budget_fraction_for_finish = float(
                os.environ.get("FORENSIC_MIN_BUDGET_FRACTION_FOR_FINISH")
            )
        if "FORENSIC_PARALLEL_ORCHESTRATOR_WEIGHT" in os.environ:
            cfg.parallel_orchestrator_token_slots = float(
                os.environ.get("FORENSIC_PARALLEL_ORCHESTRATOR_WEIGHT")
            )
        cfg.apply_token_budgets_to_llm()
        return cfg
