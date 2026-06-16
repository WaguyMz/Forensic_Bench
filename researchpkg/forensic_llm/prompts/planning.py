from researchpkg.forensic_llm.config import (
    TextTruncationLimits,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_text_to_tokens,
)

from .investigation_discipline import (
    COMPREHENSIVE_INVESTIGATION_GUIDANCE,
    HYPOTHESIS_FORMATION_GUIDANCE,
)


def build_planning_prompt(
    orientation_scratchpad: str = "",
    anomaly_cards_context: str = "",
    parallel_mode: bool = False,
    sql_max_per_core: int | None = None,
    n_schemes: int = 8,
    min_hypotheses_per_scheme: int = 5,
    max_hypotheses_per_scheme: int = 10,
    min_dispatch_items: int = 25,
    text_limits: TextTruncationLimits | None = None,
    *,
    global_run_token_budget: int | None = None,
    tokens_remaining_before_planning: int | None = None,
    orientation_budget_fraction: float | None = None,
    orchestrator_reserve_budget_fraction: float | None = None,
) -> str:
    """
      Build the one-shot investigation planning prompt (hypothesis_orchestrated v2).

      Produces ``investigation_plan.json`` with a global ``dispatch_queue`` and per-scheme
    hypothesis briefs.  No reflection / synthesis / replan phases.
    """
    _ = sql_max_per_core
    _ = n_schemes
    _ = parallel_mode  # v2 dispatches one hypothesis per task; parallelism via max_parallel_workers (default 5)

    orientation_block = ""
    if orientation_scratchpad and orientation_scratchpad.strip() not in (
        "",
        "(scratchpad is empty)",
    ):
        lim = text_limits or TextTruncationLimits()
        cap = int(
            lim.planning_orientation_prompt or lim.orientation_summary_store or 100_000
        )
        snippet = orientation_scratchpad.strip()
        if cap > 0:
            snippet = truncate_text_to_tokens(snippet, cap, side=TruncationSide.HEAD)
        orientation_block = (
            "\n\n**Orientation report for planning "
            f"(first ≤{cap:,} tokens of `orientation/orientation_report.md`; "
            "no separate digest — preserve measured baselines, aux IDs, GL accounts, "
            "process %, control-field rates):**\n```markdown\n" + snippet + "\n```\n"
        )

    cards_block = ""
    if anomaly_cards_context and anomaly_cards_context.strip():
        cards_block = (
            "\n\n**Anomaly signals (if any) — use to rank hypotheses, still cover all core schemes:**\n"
            + anomaly_cards_context.strip()
            + "\n"
        )

    orient_f = float(orientation_budget_fraction or 0.10)
    reserve_f = float(orchestrator_reserve_budget_fraction or 0.05)
    global_b = int(global_run_token_budget or 0)
    remaining_b = int(tokens_remaining_before_planning or global_b)
    nominal_worker = int(max(0, global_b) * max(0.0, min(1.0, 1.0 - orient_f)))
    if global_b > 0:
        budget_context = f"""

### Per-hypothesis worker budgets (`budget_tokens`) — **required on every `dispatch_queue` row**

Emit a positive integer **`budget_tokens`** on **each** `dispatch_queue` entry. Optionally mirror the **same** integer on the matching object in `phases[].initial_hypotheses` (same `scheme` + `hypothesis_id`) so your intent survives any internal queue rebuild.

**Meaning:** these values are **relative weights** (expected complexity × forensic importance × depth of SQL / iteration). The runtime **rescales** all rows so the **sum** of executed hypothesis-worker budgets matches the **post-planning** worker token envelope (after a small orchestrator reserve). You are **not** asked to hand-sum to an exact global total — only to **spread** weights so harder angles get larger integers than lightweight checks.

**Run context (authoritative numbers for this invocation):**
- Global run token cap: **{global_b:,}** tokens.
- Nominal orientation allowance (fraction of global): **{orient_f * 100:.0f}%** (~**{nominal_worker:,}** tokens nominally available for non-orientation work if orientation stayed on budget).
- Tokens **remaining right now** (orientation and any prior steps already charged): **{remaining_b:,}**.
- After your planning response, the runtime sets aside ~**{reserve_f * 100:.0f}%** of *remaining* tokens for orchestration overhead; **hypothesis workers share the rest** in proportion to your weights.

**How to allocate like an orchestrator (not equal slices):**
- Push **much larger** `budget_tokens` on population scans, heavy joins, multi-step falsification, and material risks from orientation; use **smaller** (but still positive) weights for quick sanity checks.
- Avoid a **flat** profile (every row ~equal) unless every hypothesis is truly the same depth — flat weights waste capacity because nothing is prioritized for deep follow-through.
- Aim for workers to **use** their envelopes: assign generously where ambiguity remains; keep weights modest only when a short SQL path can rule the angle in or out.

"""
    else:
        budget_context = """

### Per-hypothesis worker budgets (`budget_tokens`) — **required on every `dispatch_queue` row**

Emit a positive integer on each row. Values are **relative weights** (complexity × importance); the runtime rescales them to fit the remaining run budget. Use larger weights for deeper / population-scale tests and smaller ones for quick checks — avoid an entirely flat profile.

"""

    return (
        "You have completed **orientation**. Produce a structured **investigation plan** as JSON."
        + orientation_block
        + cards_block
        + budget_context
        + f"""

## Output contract

Respond with **ONLY** a JSON object (no markdown outside the JSON).

### Top-level fields

- `"orientation_risk_summary"`: bullets — strongest orientation risk signals
- `"execution_notes"`: short global notes (what to front-load, cross-scheme overlaps to watch)
- `"dispatch_queue"`: **required** — global ordered list of hypothesis tasks (see below)
- `"phases"`: per-scheme detail (hypotheses, exit criteria, query families)
- `"total_budget_sql"`: optional metadata only (not enforced)
- `"reflection_after_phases"`: always `[]` (retired — do not populate)

### `dispatch_queue` (authoritative run order — **global, not per-scheme**)

Each entry **must** include:
- `"scheme"`: canonical scheme_type string
- `"hypothesis_id"`: `"P1"`, `"P2"`, …
- `"hypothesis_text"`: concise forensic claim (free form; **not** ``If … then …`` boilerplate)
- `"hypothesis_rationale"`: **one paragraph** — why this angle now, what to test, confirm vs rule-out, one benign rival
- `"budget_tokens"`: positive integer — **complexity / depth weight** for this row (see section *Per-hypothesis worker budgets* above)
- `"dispatch_priority"`: optional integer (recomputed at runtime) — list rows in **P-wave order**

**Do not** block by scheme (all fictitious_ap rows, then all revenue, …). The runtime runs
**every scheme's P1**, then **every scheme's P2**, etc. Within a wave, order follows your
`dispatch_queue` row order (then scheme name). Use `hypothesis_id` `P1`…`P5` per scheme for depth, not global risk reordering.

**Hard requirement (enforced by the runtime):** every core scheme must have **{min_hypotheses_per_scheme}–{max_hypotheses_per_scheme}**
distinct hypotheses in **both** `dispatch_queue` and `phases[].initial_hypotheses` (same wording in both).
Plans with fewer than {min_hypotheses_per_scheme} per scheme are rejected and repaired.

Target **{min_dispatch_items}–{min_dispatch_items + 25}** total `dispatch_queue` rows (≈5 schemes × 5–10).
High-risk schemes should use the upper part of the range; lower-priority schemes still need **at least {min_hypotheses_per_scheme}**
mutually distinct tests. Add cross-cutting angles (period-end, creator concentration, lettrage absence,
master-data gaps, amount tails) under the most relevant scheme — do not collapse everything into one umbrella line.

The runtime executes tasks in **P1 wave → P2 wave → …** order (all schemes per wave), with up to ``max_parallel_workers`` concurrent hypothesis workers (default 5; use 1 for serial).

### Per-scheme `phases[]` entries

- `"scheme"`, `"priority"` (scheme-level rank), `"plan_rationale"`, `"priority_signals"`
- `"benign_rival_explanations"`, `"planned_query_sequence"`, `"grounding_query_templates"` (optional SQL templates to retrieve `document_id`s if needed)
- `"exit_criteria"` — how you will decide CONFIRMED / RULED_OUT / INCONCLUSIVE for this scheme
- `"initial_hypotheses"`: array of **{min_hypotheses_per_scheme}–{max_hypotheses_per_scheme}** objects per scheme, each:
  - `"hypothesis_id"`: `"P1"`, …
  - `"hypothesis_text"`: same wording as in `dispatch_queue`
  - `"hypothesis_rationale"`: same paragraph as in `dispatch_queue`
  - `"budget_tokens"`: optional but **recommended** — same integer as the matching `dispatch_queue` row for this scheme + id

**Coverage:** hypotheses must be **mutually distinct** (different SQL populations or control tests — not rephrasings).
Use sequential `P1`…`P{max_hypotheses_per_scheme}` within each scheme. `exit_criteria` must be **full sentences** (not single letters).
Do **not** wrap the JSON in markdown fences or nest it under `"investigation_plan"`.

Cover all **5 core schemes**:
1. fictitious_ap_disbursements
2. revenue_manipulation
3. vendor_collusion
4. shadow_payroll
5. inventory_manipulation

"""
        + HYPOTHESIS_FORMATION_GUIDANCE
        + COMPREHENSIVE_INVESTIGATION_GUIDANCE
        + """

Keep `hypothesis_text` short; put investigative detail in `hypothesis_rationale`. Order `dispatch_queue` rows P1 across schemes, then P2, etc.; put the strongest angles in P1/P2 slots per scheme.

### Example structure (do not copy literally)

```json
{{
  "orientation_risk_summary": ["<signal>"],
  "execution_notes": ["<note>"],
  "dispatch_queue": [
    {{
      "scheme": "shadow_payroll",
      "hypothesis_id": "P1",
      "budget_tokens": 420000,
      "hypothesis_text": "Payroll lines reference actors missing from the HR master at material volume.",
      "hypothesis_rationale": "Orientation showed H2R auxiliary gaps; test full-population mismatch rate on payroll postings, drill exemplar JEs, and rule out interface mapping noise before CONFIRMED."
    }},
    {{
      "scheme": "revenue_manipulation",
      "hypothesis_id": "P1",
      "budget_tokens": 180000,
      "hypothesis_text": "Revenue postings cluster in the last three days of fiscal periods above baseline.",
      "hypothesis_rationale": "Period-end concentration can indicate cut-off abuse; compare day-of-month rates on revenue accounts, then check for next-period reversals; benign rival is normal billing calendar."
    }}
  ],
  "phases": [
    {{
      "scheme": "shadow_payroll",
      "priority": 1,
      "plan_rationale": "<why now>",
      "priority_signals": ["<signal>"],
      "benign_rival_explanations": ["<rival>"],
      "planned_query_sequence": ["<step>"],
      "grounding_query_templates": ["SELECT document_id FROM je_header WHERE ..."],
      "exit_criteria": ["Each hypothesis has a recorded verdict with supporting evidence"],
      "initial_hypotheses": [
        {{
          "hypothesis_id": "P2",
          "budget_tokens": 420000,
          "hypothesis_text": "Payroll lines reference actors missing from the HR master at material volume.",
          "hypothesis_rationale": "Orientation showed H2R auxiliary gaps; test full-population mismatch rate on payroll postings, drill exemplar JEs, and rule out interface mapping noise before CONFIRMED."
        }}
      ]
    }}
  ],
  "total_budget_sql": 0,
  "reflection_after_phases": []
}}
```"""
    )
