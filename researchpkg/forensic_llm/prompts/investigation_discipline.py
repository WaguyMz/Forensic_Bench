"""Shared investigation discipline blocks for prompts."""

TOOL_NEUTRAL_OPERATING = (
    "Every tool call must serve a specific, falsifiable hypothesis: surface a signal, "
    "discriminate rival explanations, or retrieve JE-level evidence. Choose the tool "
    "that fits the question — do not default to sql preview when population-level analysis, "
    "graphs, or exports are the right next step."
)

TOOL_SELECTION_GUIDANCE = """
### Tool selection (no SQL default)
| Need | Preferred tools |
|------|-----------------|
| Aggregates, filters, small result sets | `sql` with `mode='preview'` (default max_rows 50) |
| Full population, >50 rows, Benford, distributions, clustering | `sql` with `mode='export'` → `code_interpreter` → `read_image` if charted |
| Vendor–employee bank overlap, circular payments, related-party paths | `sql(mode='export')` + `code_interpreter` (NetworkX joins on exported tables) |
| Flagging specific journal entries | `sql` preview/export + filter in `code_interpreter`, then `report_suspicion` when appropriate |

**Anti-patterns:** repeated `sql` preview with arbitrary `LIMIT` when the answer needs the full population; finishing a hypothesis after one null preview without reformulation or export-and-analyze; entity-level conclusions without supporting transaction evidence when JEs exist.
"""

HYPOTHESIS_FORMATION_GUIDANCE = """
### Hypothesis formation (planning)
Form **ranked, testable** investigation angles from orientation evidence — write in **plain forensic
language**, not a rigid template.

**Do not** use boilerplate ``If … then …`` phrasing. Each hypothesis needs two fields:
- ``hypothesis_text``: a concise claim about what might be wrong (one or two sentences).
- ``hypothesis_rationale``: **one paragraph** tying the claim to orientation signals, what SQL or
  analysis would test it, and what outcome would support vs rule out the scheme (including one
  plausible benign rival).

Other rules:
- Assign **P1, P2, …** within each scheme (P1 = strongest angle for that scheme).
- **Cover the forensic surface area** before merging ideas: entities & master data, timing & cut-off,
  amounts & concentration, approvals & segregation, period-end behavior, control-account
  reconciliations, and cross-party links (vendor↔customer↔employee) where data allows.
  One queue entry ≈ one distinct test.
- **Prefer breadth over a single umbrella line** per scheme: high-risk schemes deserve many distinct
  angles; lower-priority schemes still need enough lines for a credible RULED_OUT.
- In ``dispatch_queue``, list rows in **P-wave order** (all schemes' ``P1``, then all ``P2``, …); do not block by scheme.
- Every ``dispatch_queue`` entry must include both ``hypothesis_text`` and ``hypothesis_rationale``.
- Mirror the same ``P`` ids, text, and rationale in ``phases[].initial_hypotheses`` (as objects, not
  legacy string lines).
"""

COMPREHENSIVE_INVESTIGATION_GUIDANCE = """
### Comprehensive investigation standard (prompt discipline — not a step quota)
Treat this engagement like a **full forensic file**, not a quick anomaly scan. Depth comes from
**evidence quality**, not from repeating the same query.

**Orientation & planning**
- Build measured baselines (rates, concentrations, period structure) before accusing fraud.
- Plans should enumerate **distinct** hypotheses and concrete query families per scheme — workers
  inherit your brief; vague plans produce shallow tasks.

**Per-hypothesis execution**
- Typical thorough path: **screen** (does the signal exist?) → **drill** (who/what/when/how much?) →
  **falsify a benign rival** (could this be process design, master-data gap, or IC timing?) →
  **discriminative check** when populations are large → **`report_suspicion`** for exemplar JEs.
- A null or weak screen is not a verdict: reformulate (different join, window, account slice, or
  export-and-analyze) before RULED_OUT.
- When a pattern is material, trace **follow-on questions** (linked accounts, same actor, adjacent
  periods, monetary roll-up) until the story is complete or clearly explained away.
- Use **`sql(mode='export')` + `code_interpreter`** whenever the question is about distributions,
  rates, concentration, or >50 rows — previews are for EXISTS and small top-k, not population proof.

**When to stop on a hypothesis**
- Stop when you can defend CONFIRMED / RULED_OUT / INCONCLUSIVE with cited SQL outcomes — not when
  you have merely run a few queries. Premature closure wastes the budget and misses fraud.
"""

HYPOTHESIS_INVESTIGATION_DEPTH = """
### Hypothesis investigation depth
Use **as many tool calls as the hypothesis requires** (`sql`, `code_interpreter`, `read_image`).
There is no target query count — there is a **standard of proof**.

Before a verdict, you should usually have:
- screened whether the signal exists in the population,
- drilled into the strongest candidates (entities, amounts, periods, control bypass),
- tested at least one **plausible benign rival** explanation with data,
- retrieved `document_id` UUIDs for any JEs you will report or reference.

If a screening step returns zero rows, reformulate once (time window, account range, join path, or
export-and-analyze) before RULED_OUT. If a pattern is confirmed, consider whether adjacent slices
(same user, vendor, period cluster) warrant a short follow-up before closing the hypothesis.
"""

DISCRIMINATIVENESS_BEFORE_FLAG = """
Before `report_suspicion`, verify the signal is **discriminative**.
Use `sql(mode='export')` + `code_interpreter` to compare flagged vs non-flagged rates when the population is large.
"""

REPORT_SUSPICION_DISCIPLINE = """
### JE reporting (`report_suspicion`) — required for evaluation
Benchmark scoring uses **`detections.json`**, which is populated **only** via `report_suspicion`
(not from scratchpad prose alone).

**When you MUST call `report_suspicion`:**
- You retrieved one or more `je_header.document_id` UUIDs that exemplify the fraud pattern for this hypothesis.
- Status is **CONFIRMED** with clear SQL-backed evidence — include exemplar JEs only (not every row in a large export).
- After `sql`/`code_interpreter` identifies a bounded set of suspicious postings — report **before** stopping tool use.

**How to report:**
- Prefer batch: `suspicions: [{document_id, scheme_type, rationale}, ...]` (up to ~50 per call; repeat if needed).
- `document_id` = UUID from column `document_id` / `jh.document_id` only — never reference, line_number, or auxiliary account.
- `scheme_type` = the task scheme (e.g. `shadow_payroll`).

**When NOT to report:** hypothesis **RULED OUT** with no suspicious JEs after reformulation — leave detections empty.

**Workflow:** screen → drill → (optional discriminative check) → **`report_suspicion`** → scratchpad verdict.
"""

# Back-compat alias for imports that still reference the old name.
GROUNDING_AFTER_CONFIRMED = HYPOTHESIS_INVESTIGATION_DEPTH
