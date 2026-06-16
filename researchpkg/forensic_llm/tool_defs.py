"""
OpenAI-compatible tool/function definitions.

These are injected into every LLM call so the model knows exactly what
tools are available, their parameters, and any constraints.
"""
from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Individual tool schemas
# ---------------------------------------------------------------------------

SQL_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sql",
        "description": (
            "Execute a read-only SQL SELECT query against the journal-entry "
            "database (PostgreSQL). "
            "Tables available:\n"
            "  je_header(document_id, company_code, fiscal_year, fiscal_period, "
            "posting_date, document_date, document_type, currency, exchange_rate, "
            "reference, header_text, created_by, source, business_process, ledger)\n"
            "  je_line(document_id, line_number, company_code, gl_account, "
            "debit_amount, credit_amount, local_amount, cost_center, profit_center, "
            "line_text, auxiliary_account_number, auxiliary_account_label, "
            "lettrage, lettrage_date)\n"
            "  employees(employee_id, user_id, display_name, first_name, last_name, "
            "email, company_code, department_id, cost_center, manager_id, status, "
            "hire_date, termination_date, creation_date, location, "
            "payroll_bank_name, payroll_bank_country, payroll_account_number, "
            "payroll_routing_code)\n"
            "  hr_employees(employee_id, user_id, display_name, first_name, last_name, "
            "email, company_code, department_id, cost_center, manager_id, status, "
            "hire_date, termination_date, creation_date, location)\n"
            "  **HR tip:** `hr_employees` excludes ghost employees — LEFT JOIN to find gaps\n"
            "  vendors(vendor_id, name, country, account_number, tax_id, currency, "
            "reconciliation_account, auxiliary_gl_account, is_intercompany, "
            "behavior, payment_terms, bank_account_count, "
            "primary_bank_name, primary_bank_country, primary_account_number, "
            "primary_routing_code)\n"
            "  customers(customer_id, name, country, account_number, tax_id, "
            "currency, reconciliation_account, auxiliary_gl_account, "
            "is_intercompany, credit_rating, bank_account_count, "
            "primary_bank_name, primary_bank_country, primary_account_number, "
            "primary_routing_code)\n\n"
            "**Modes:**\n"
            "- `preview` (default): small markdown table in context (max_rows default 50, "
            "hard cap 50). Use for aggregates, EXISTS checks, and small top-k drills.\n"
            "- `export`: write the full result set to `sql_outputs/<filename>.csv` "
            "(max_rows default 10 000, hard cap 50 000). The model receives a manifest "
            "and a short sample only — load the file in `code_interpreter` for "
            "population-level statistics.\n\n"
            "IMPORTANT:\n"
            "- Only SELECT/WITH statements are allowed.\n"
            "- If you expect more than ~50 rows or need Benford / distributions / "
            "clustering, use `mode='export'` then `code_interpreter` — do not loop "
            "`preview` with arbitrary LIMIT."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Valid SQL SELECT statement to execute.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["preview", "export"],
                    "description": (
                        "'preview' returns a table in context; 'export' writes CSV "
                        "under sql_outputs/ and returns path + sample rows only."
                    ),
                    "default": "preview",
                },
                "max_rows": {
                    "type": "integer",
                    "description": (
                        "Row cap for the query. preview: default 50, max 50 (use export for more). "
                        "export: default 10000, max 50000."
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Export only: basename for the CSV (e.g. 'payroll_period_end.csv'). "
                        "Auto-generated if omitted."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

GREP_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search for a regex pattern in flat output files "
            "(CSV, JSON, JSONL, YAML) produced by the data generator. "
            "Useful for inspecting document descriptions, vendor names, "
            "or config parameters that are not indexed in the database.\n\n"
            "The search is anchored to the output directory; you cannot "
            "traverse outside it.  Returns up to max_matches matching lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Python-compatible regex pattern to search for. "
                        "Case-insensitive by default."
                    ),
                },
                "file_glob": {
                    "type": "string",
                    "description": (
                        "Glob pattern relative to the output root, e.g. "
                        '"*.csv", "master_data/vendors.json", "**/*.jsonl". '
                        'Default: "**/*" (all files).'
                    ),
                    "default": "**/*",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default 50).",
                    "default": 50,
                },
            },
            "required": ["pattern"],
        },
    },
}

READ_IMAGE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_image",
        "description": (
            "Load a locally saved plot image (PNG/JPG) and return it so a vision-capable "
            "model can interpret the chart. "
            "Use this after calling plot() when you need to read axis labels, legend, "
            "outliers, or annotations from the generated PNG.\n\n"
            "Security: only files under the run output directory are allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the image file to read (must be under the forensic output directory)."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}

SCRATCHPAD_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "scratchpad",
        "description": (
            "Active reflection tool: use the scratchpad to reflect after tool results, "
            "not only to store notes. It is re-injected into your context every step.\n\n"
            "**Reflect after SQL/plot:** Before running more queries, append a short "
            "reflection: what the last result showed, what you conclude for the hypothesis "
            "(supported / ruled out / inconclusive), and the next distinct action. "
            "This prevents repeated queries and keeps reasoning explicit.\n\n"
            "Include: current phase/scheme, key numbers, sample document_ids, hypotheses "
            "marked CONFIRMED/RULED_OUT/INCONCLUSIVE/NOT_INVESTIGATED, and "
            "Investigation gaps / Next steps.\n\n"
            "Modes:\n"
            '  "append" (default) – add a reflection or note (never deletes prior notes).\n'
            '  "replace" – update the structured outline block only; **appended notes are '
            "kept** (SQL auto-stubs and prior reflections). Use this to refresh §headings "
            "without losing evidence.\n"
            '  "read" – return current contents without modifying.\n\n'
            "Not available during **orientation** (use `orientation_report` there)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": (
                        "Reflection or note: what you learned from the last tool results, "
                        "conclusion for the hypothesis, next step; or phase/scheme, key numbers, "
                        "sample document_ids, hypothesis status (CONFIRMED/RULED_OUT/etc.), gaps."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["append", "replace", "read"],
                    "description": "Operation mode (default: append).",
                    "default": "append",
                },
            },
            "required": [],
        },
    },
}

WRITE_CSV_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_csv",
        "description": (
            "Deprecated alias for sql(query, mode='export', filename=...). "
            "Prefer sql with mode='export'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Destination filename (basename only, e.g. 'payroll_je.csv'). "
                        "A .csv extension is added automatically if missing. "
                        "Directory components are stripped for security."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Valid SQL SELECT statement whose results are written to the CSV.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to export (default 10 000, hard cap 50 000).",
                    "default": 10000,
                },
            },
            "required": ["filename", "query"],
        },
    },
}

CODE_INTERPRETER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "Execute arbitrary Python code in a secure, isolated E2B sandbox.\n\n"
            "**Use this for ALL computation and ALL charts/visualisations** — there is no "
            "separate plot tool. This is your primary analytical instrument:\n\n"
            "**Charts & visualisations (matplotlib / seaborn / plotly):**\n"
            "  - Call plt.show() OR plt.savefig('/home/user/<name>.png') — both work.\n"
            "  - E2B automatically captures matplotlib figures as rich PNG outputs and "
            "    returns their paths in the tool result.\n"
            "  - After execution, call read_image('<path>') on any returned file path to "
            "    visually inspect the chart in your next turn.\n"
            "  - Useful chart patterns: histogram of amounts (Benford / threshold clustering), "
            "    time-series of posting dates (period-end spikes), heatmap of user × GL account "
            "    (concentration), Lorenz curve (payroll disparity), scatter of debit vs credit "
            "    (circular flow), bar chart of vendor payment frequency.\n\n"
            "**Statistical analysis:**\n"
            "  - Benford's Law: first-digit distribution + chi-squared p-value\n"
            "  - Outlier detection: z-scores, IQR, isolation forest on amount columns\n"
            "  - Time clustering: period-end / weekend / after-hours posting concentration\n"
            "  - Mann-Whitney, KS test, or t-test to compare suspicious vs. normal populations\n\n"
            "**Data wrangling & advanced investigation:**\n"
            "  - Pandas/NumPy on large datasets exported by sql(mode='export')\n"
            "  - NetworkX graph construction for circular payment or related-party cluster detection\n"
            "  - Fuzzy name matching (difflib / rapidfuzz) to find ghost employees or shell vendors\n"
            "  - Monetary-impact rollup: confidence-weighted exposure totals, Pareto curves\n\n"
            "**Sandbox:** Local microsandbox process — Python 3.x, pandas, numpy, scipy, "
            "matplotlib, seaborn, scikit-learn, networkx pre-installed. No network access.\n\n"
            "**File access:** CSV files under the run output directory (including "
            "sql_outputs/) are available to the sandbox. Load with pd.read_csv('<path "
            "returned by sql export>'). Image files saved to /tmp/ are automatically harvested and "
            "downloaded to the output directory after each call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Valid Python source code to execute. "
                        "Use print() to emit results. "
                        "Create charts with matplotlib/seaborn and call plt.show() or "
                        "plt.savefig('/home/user/<name>.png') — both are captured. "
                        "Load prior sql export output with pd.read_csv('<returned_path>')."
                    ),
                },
                "upload_csvs": {
                    "type": "boolean",
                    "description": (
                        "If true (default), CSV files in the run output directory are "
                        "available to the sandbox (legacy flag; local backend always "
                        "uses the run output dir as cwd)."
                    ),
                    "default": True,
                },
            },
            "required": ["code"],
        },
    },
}

REPORT_SUSPICION_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_suspicion",
        "description": (
            "Report suspicious journal entries **as soon as you identify them** — this is "
            "**required** for benchmark evaluation (writes detections.json). Call after SQL "
            "or code_interpreter when you have concrete je_header.document_id UUIDs. Use the "
            "`suspicions` array to batch many JEs per call. Call multiple times if needed. "
            "Hypothesis workers: call before stopping with clear SQL-backed evidence; "
            "do not rely on the summary JSON alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": (
                        "MUST be the UUID from je_header.document_id (the primary key of the journal entry). "
                        "Take it directly from your SQL result column document_id or jh.document_id. "
                        "Do NOT use reference, line_number, auxiliary_account_number, or any other field — "
                        "evaluation matches only on je_header.document_id."
                    ),
                },
                "scheme_type": {
                    "type": "string",
                    "enum": [
                        "fictitious_ap_disbursements",
                        "revenue_manipulation",
                        "vendor_collusion",
                        "shadow_payroll",
                        "inventory_manipulation",
                        "unknown",
                    ],
                    "description": "Fraud scheme type (only these 5 canonical schemes exist in the data).",
                },
                "rationale": {
                    "type": "string",
                    "description": "One-sentence evidence summary (required for benchmark flags).",
                },
                "suspicions": {
                    "type": "array",
                    "description": (
                        "Optional: report multiple JEs in one call. "
                        "Each object: document_id (je_header UUID from SQL), scheme_type, "
                        "optional rationale."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "je_header.document_id UUID only (not reference, not line id).",
                            },
                            "scheme_type": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["document_id", "scheme_type"],
                    },
                },
            },
            "required": [],
        },
    },
}

ORIENTATION_REPORT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "orientation_report",
        "description": (
            "Write the **ONLY memory** during orientation: `orientation/orientation_report.md` "
            "(flushed to disk every call). Planning reads the **raw report** (first ~100k tokens) — "
            "NOT conversation history and **no** separate digest.\n\n"
            "Call **often** after SQL/analysis. Append `##` sections as **investigator field notes**: "
            "2+ paragraphs explaining linkages, baselines, and structural facts (%, counts, €, dates, "
            "named IDs), then supporting tables (vendor census, top-10 GL/aux/users, cross-tabs). "
            "Do **not** write table-only sections. Mandatory before `complete_orientation`: "
            "Vendor/AP, Revenue/COA, O2C×vendor, Planning leads; no TBD; no fraud vocabulary. "
            "No raw 1000-row SQL grids.\n\n"
            "**replace** restructures the full report (rare). **read** previews the on-disk file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["append", "replace", "read"],
                    "description": "append (default): add a section; replace: full rewrite; read: fetch current file.",
                    "default": "append",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Markdown for append/replace: substantive `##` sections — **prose-first** field notes "
                        "with measured facts, then tables where useful. Required: Vendor/AP, Revenue/COA, "
                        "O2C×vendor, Planning leads. No table-only dumps; no TBD; no fraud vocabulary. "
                        "Ignored when mode=read."
                    ),
                },
            },
            "required": [],
        },
    },
}

COMPLETE_ORIENTATION_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "complete_orientation",
        "description": (
            "End orientation when the report reads like complete **field notes**: rich prose on schema, "
            "linkages, process behavior, actors, period-end, and controls; plus vendor census, "
            "revenue/COA reconciliation, O2C×vendor cross-tab, top-10 summaries, and `## Planning leads`. "
            "Not table-only. No TBD; no fraud vocabulary. Planning reads the raw report (first ~100k tokens) "
            "— put the most important narrative early."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "readiness_note": {
                    "type": "string",
                    "description": (
                        "Optional short note on what you covered and why orientation is complete."
                    ),
                },
            },
            "required": [],
        },
    },
}

FINISH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish_investigation",
        "description": (
            "Signal that the investigation is complete and submit the final "
            "artefacts.  Call this when you have gathered sufficient evidence "
            "or the token budget is nearly exhausted.\n\n"
            "You MUST provide:\n"
            "1. suspicion_list – JSON array of suspicion objects.\n"
            "2. narrative      – Markdown report explaining the investigation.\n\n"
            "document_id rule: For each flagged JE, document_id MUST be the UUID from "
            "je_header.document_id (from your SQL results). Never use reference, line_number, "
            "or any other field — evaluation matches only on this UUID. Use document_id=null "
            "only for entity-level findings (e.g. ghost employee) that do not map to a single JE.\n"
            "Each suspicion object: document_id (str|null), entity_id (str|null), "
            "entity_type (str|null), scheme_type (str), confidence (float 0-1), "
            "severity (int 1-5), rationale (str), supporting_evidence (list[str]), "
            "related_document_ids (list[str]), monetary_impact (float|null), gl_accounts (list[str])."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "suspicion_list": {
                    "type": "array",
                    "description": "Array of suspicion objects as described above.",
                    "items": {"type": "object"},
                },
                "narrative": {
                    "type": "string",
                    "description": "Markdown narrative report.",
                },
            },
            "required": ["suspicion_list", "narrative"],
        },
    },
}

SPAWN_WORKER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_worker",
        "description": (
            "Spawn a dedicated autonomous worker that investigates a narrow goal in "
            "parallel. Use this when a lead deserves sustained attention or when you "
            "want to split the investigation into specialised tracks.\n\n"
            "The worker gets its own run directory, scratchpad, tool history, and "
            "budget. Candidate schemes should stay within the closed-world label set "
            "when possible, but the worker goal itself may be a hypothesis, entity "
            "cluster, reconciliation task, or rival explanation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scheme_or_goal": {
                    "type": "string",
                    "description": "Short label for the worker's mission.",
                },
                "brief": {
                    "type": "string",
                    "description": "Concrete instruction for what the worker should investigate.",
                },
                "budget_sql_calls": {
                    "type": "integer",
                    "description": "Operational budget field (runtime-enforced; not a target query count).",
                    "default": 12,
                },
                "budget_tokens": {
                    "type": "integer",
                    "description": "Approximate prompt/reasoning token budget for the worker.",
                    "default": 1000000,
                },
                "candidate_schemes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional closed-world scheme labels this worker is most likely "
                        "to inform."
                    ),
                    "default": [],
                },
            },
            "required": ["scheme_or_goal", "brief"],
        },
    },
}

LIST_WORKERS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_workers",
        "description": (
            "List all known workers with their live status, mailbox depth, progress, "
            "candidate schemes, and current run directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MESSAGE_WORKER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "message_worker",
        "description": (
            "Send a short bounded steering instruction to a running worker. "
            "Use this to redirect, narrow, broaden, or request a specific follow-up "
            "check without splicing raw transcripts into the worker history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "worker_id": {
                    "type": "string",
                    "description": "Worker identifier returned by spawn_worker.",
                },
                "instruction": {
                    "type": "string",
                    "description": "Short steering instruction for the worker mailbox.",
                },
            },
            "required": ["worker_id", "instruction"],
        },
    },
}

BLACKBOARD_WRITE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "blackboard_write",
        "description": (
            "Record a salient entity finding on the shared evidence blackboard "
            "for cross-scheme coordination. Use when an entity may link multiple "
            "fraud schemes. Do not write raw SQL or full narratives."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "entity_type": {
                    "type": "string",
                    "description": "vendor | employee | customer | user | other",
                },
                "scheme": {
                    "type": "string",
                    "description": "Canonical scheme_type for this finding.",
                },
                "confidence": {"type": "number"},
                "rationale": {
                    "type": "string",
                    "description": "One short sentence.",
                },
            },
            "required": ["entity_id", "entity_type", "scheme", "rationale"],
        },
    },
}

COLLECT_WORKER_SUMMARY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "collect_worker_summary",
        "description": (
            "Collect a worker's latest normalized summary. This updates the parent "
            "coverage ledger and gives you a compact evidence snapshot without reading "
            "the raw worker transcript."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "worker_id": {
                    "type": "string",
                    "description": "Worker identifier returned by spawn_worker.",
                }
            },
            "required": ["worker_id"],
        },
    },
}


# ---------------------------------------------------------------------------
# Registry: name → schema (single source of truth)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    t["function"]["name"]: t
    for t in [
        SQL_TOOL,
        GREP_TOOL,
        READ_IMAGE_TOOL,
        SCRATCHPAD_TOOL,
        WRITE_CSV_TOOL,
        CODE_INTERPRETER_TOOL,
        REPORT_SUSPICION_TOOL,
        ORIENTATION_REPORT_TOOL,
        COMPLETE_ORIENTATION_TOOL,
        FINISH_TOOL,
        SPAWN_WORKER_TOOL,
        LIST_WORKERS_TOOL,
        MESSAGE_WORKER_TOOL,
        COLLECT_WORKER_SUMMARY_TOOL,
        BLACKBOARD_WRITE_TOOL,
    ]
}

ORIENTATION_TOOL_NAMES: List[str] = [
    "sql",
    "read_image",
    "orientation_report",
    "code_interpreter",
    "complete_orientation",
]

HYPOTHESIS_WORKER_TOOL_NAMES: List[str] = [
    "sql",
    "read_image",
    "scratchpad",
    "code_interpreter",
    "report_suspicion",
    "blackboard_write",
]

# Names of all registered tools (stable order).
ALL_TOOL_NAMES: List[str] = list(TOOL_REGISTRY)

# Default tool set: SQL + analytical tools, no grep (labels risk), no graph.
DEFAULT_TOOL_NAMES: List[str] = [
    "sql",
    "read_image",
    "scratchpad",
    "code_interpreter",
    "report_suspicion",
    "finish_investigation",
]

AUTONOMOUS_ORCHESTRATION_TOOL_NAMES: List[str] = [
    "spawn_worker",
    "list_workers",
    "message_worker",
    "collect_worker_summary",
]


def tools_from_names(names: List[str]) -> List[Dict[str, Any]]:
    """Return tool schemas for the given tool names (preserving registry order)."""
    unknown = set(names) - TOOL_REGISTRY.keys()
    if unknown:
        raise ValueError(
            f"Unknown tool name(s): {sorted(unknown)}. Valid: {ALL_TOOL_NAMES}"
        )
    return [TOOL_REGISTRY[n] for n in ALL_TOOL_NAMES if n in names]


# ---------------------------------------------------------------------------
# Convenience pre-composed lists (kept for backward compatibility)
# ---------------------------------------------------------------------------

ALL_TOOLS: List[Dict[str, Any]] = list(TOOL_REGISTRY.values())

# Subset without grep when flat files are unavailable
DB_ONLY_TOOLS: List[Dict[str, Any]] = tools_from_names(DEFAULT_TOOL_NAMES)

# Alias kept for call sites that imported the old name.
ALL_TOOLS_NO_GRAPH: List[Dict[str, Any]] = DB_ONLY_TOOLS
DB_ONLY_TOOLS_NO_GRAPH: List[Dict[str, Any]] = DB_ONLY_TOOLS

ALL_TOOLS_WITH_GREP: List[Dict[str, Any]] = tools_from_names(
    [
        "sql",
        "grep",
        "read_image",
        "scratchpad",
        "code_interpreter",
        "report_suspicion",
        "finish_investigation",
    ]
)
