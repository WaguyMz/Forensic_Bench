from __future__ import annotations

ORIENTATION_MANDATORY_CHECKLIST = """\
**Mandatory coverage before `complete_orientation`** — write like an investigator's **field notes**:
dense explanatory prose first; tables are supporting evidence, not a substitute for understanding.

**Every major `##` section must include:**
- **2–5 paragraphs** (or equivalent bullet depth) explaining *what you learned*, *how entities link*,
  *what is structurally normal here*, and *what planning should know* — with measured numbers woven in.
- **At least one supporting table** only where it helps (vendor census, top-10 GL/aux/users, cross-tab);
  never a report that is only tables with one-line captions.

**Required topics** (exact `##` headings where noted):

1. **`## Vendor / AP master`** — narrative on AP linkage (`auxiliary_gl_account` ↔ lines), P2P behavior,
   self-balancing patterns; **plus** a table of **every** vendor:
   `vendor_id` | `auxiliary_gl_account` | P2P lines | debit | credit | self-balancing docs.
   **Call out** shell vendors `V-000021`–`V-000023` (same `V-{n:06}` / `401{n:04}` style as `V-000001`–`V-000020`);
   contrast P2P volume, self-balancing doc share, and posters — not ID format alone.

2. **`## Revenue / COA`** — explain COA membership vs `je_line` usage; revenue timing and posters;
   **plus** each revenue GL for the latest material year (credit, line/JE counts). Resolve totals with SQL.

3. **`## O2C × vendor/aux`** — explain whether vendor auxiliaries hit customer/O2C paths; **plus** cross-tab
   or explicit zero-row proof with the filter used.

4. **Structural baselines in prose** (dedicated `##` sections): schema/grain, process mix, fiscal/period-end
   behavior, control-field coverage, user/actor tiers, reference patterns, per-process GL corridors.

5. **Top-N summaries** — top 10 GL, top 10 aux, top 10 users: **table + short interpretation**
   (why these dominate, which processes drive them).

6. **`## Planning leads`** (single final section) — neutral **test intents** only; no fraud/scheme vocabulary.

**Completion hygiene**
- **No** `TBD`, `TODO`, “need to verify” — run follow-up SQL first.
- **Banned** in orientation: fraud, suspicious, anomaly, red flag, scheme, manipulation, window dressing, etc.
- **Bad:** 300 lines of tables and no ledger narrative. **Good:** rich prose sections with selective supporting tables.
"""

ORIENTATION_SCREENING_RULES = """\
Begin a thorough **orientation / database profiling phase** for this ledger.

**CRITICAL SEPARATION OF CONCERNS:**
Your role in this phase is **DATABASE UNDERSTANDING ONLY** — build a comprehensive statistical and structural profile.
**DO NOT** draw fraud conclusions, identify suspicious patterns, or form hypotheses. That happens in the **planning phase**.

**Your mission:**
1. Understand the **schema** — tables, keys, relationships, data types
2. Profile the **structure** — chart of accounts, business processes, hierarchies
3. Quantify **statistical distributions** — amounts, frequencies, time patterns
4. Map **data linkages** — how master data connects to transactions
5. Establish **baselines** — what "normal" looks like in this dataset

**NO fraud conclusions yet.** Planning phase will use your profiles to design targeted tests.

**Depth expectation:** Produce a **long, decision-ready** report — many `##` sections of **investigative
narrative** (how the ledger works, how data links, what baselines look like), not a spreadsheet export.
Planning reads the **raw orientation report** (first ~100k tokens); every critical baseline must be
**explained in prose** in the report, not only in SQL you ran once.

**No scratchpad.** Persist findings via `orientation_report(mode="append")` — **update the report often**
(after each meaningful SQL or analysis batch, not only at the end).

**Context layout every LLM call (fixed prefix):**
1. **System prompt** — role, schema, tools
2. **[Current Orientation Report]** — cumulative findings on disk
3. **Recent steps** — last assistant/tool turns (token-capped; may truncate old SQL)
4. **[Current Step]** — active checkpoint + in-flight SQL/tool output

**Per-step workflow**
1. Screen with `sql` / `code_interpreter` as needed.
2. **Synthesize in your own words** what the data means (linkage, grain, distributions, gaps).
3. **`orientation_report(mode="append")`** — add a `##` section: **prose first**, then tables if useful.

**Report format — investigator field notes (tables + prose):**
- **Lead with explanatory prose** (paragraphs or rich bullets): schema relationships, process behavior,
  who posts what, how auxiliaries tie to master data, period-end shape, control-field emptiness.
- Use **markdown tables** for enumerations (full vendor census, top-10 GL/aux/users, cross-tabs) — synthesized,
  not 1000-row SQL dumps — **after** you explain why the table matters.
- Each `##` section needs **numbers embedded in narrative** (%, counts, €, dates, named IDs) — not table-only sections.
- Drill into **specific entities** when SQL surfaces concentration (named vendor/customer/user/document patterns).

**SQL patterns (orientation)**
- Vendor census + P2P stats: join `vendors` → `je_line` on `auxiliary_account_number = auxiliary_gl_account`
  filtered to `business_process = 'P2P'`; aggregate lines and self-balancing `document_id` counts.
- Revenue by account: filter `chart_of_accounts.account_type = 'Revenue'` (or class 7) for the target fiscal year.
- O2C×vendor: O2C headers/lines with `411%` aux intersecting same document with vendor `401%` aux.

**Entity- and linkage-specific profiling (high value for planning):**
- **P2P / AP:** vendor `auxiliary_gl_account` ↔ `je_line.auxiliary_account_number` join
  patterns; lines with vs without aux; top auxiliaries by volume; self-balancing documents
- **O2C / AR:** customer auxiliaries; any vendor aux appearing in O2C; revenue account usage
- **H2R:** payroll GL concentration; whether employee IDs appear on lines
- **Master data:** NULL rates **per field** (not only “many NULLs”); list **named** vendors,
  customers, or employees when a pattern is concentrated (IDs, aux numbers, bank fields)
- **Controls:** lettrage match %, `reconciliation_account`, cost_center / profit_center fill rates

**Required coverage before `complete_orientation`** (each as a substantive `##` section with prose + numbers)
- Schema & grain — how headers/lines/master tables join; fiscal span; company/currency/doc type constraints
- Chart of Accounts — hierarchy, account types, which GLs dominate **and why** they appear across processes
- Process mix & sources — what each BP is used for in *this* ledger; manual vs automated split
- Monetary corridors — typical debit/credit pairings per BP; background vs process-specific accounts
- Actors — user tiers, service accounts, concentration; tie to departments/employees where possible
- Period structure — month/quarter/year-end volume; weekend/off-hours if relevant
- Master data — vendors/customers/employees: coverage, NULL fields, naming/id conventions
- Controls & references — lettrage, cost/profit center, aux fill rates; reference prefix families (PO-, SO-, …)

**Depth**
- Prefer `sql(mode='export')` + `code_interpreter` for populations and cross-tabs.
- Run **follow-up** SQL when a preview row looks structurally unusual (e.g. one aux with
  debit=credit totals, sparse revenue months, weekend posting in a process).
- Aim to **use most of the orientation token budget** on substantive profiling, not idle turns.
- Focus on **WHAT** and **HOW MUCH**, not **WHY** or **SUSPICIOUS**.
- Do **not** emit `investigation_plan.json`, `report_suspicion`, or `finish_investigation` here.
- Do **not** use fraud vocabulary (see banned list in checklist above) — save interpretive testing for planning.

**Chart of Accounts — ALWAYS profile in detail:**
- Hierarchy levels (Level 1: classes, Level 2: subclasses, Level 3-6: groups/detail)
- Account type distribution (Asset, Liability, Equity, Revenue, Expense)
- Top 20 accounts by turnover (debit and credit separately)
- Account usage frequency (documents per account)
- Debit/credit pairing patterns (which accounts commonly pair together)

**Example of correct framing:**
✅ GOOD (prose): "Revenue is sparse: only 321 lines (0.05%) hit class-7 accounts, totaling €89.6M credit
across 701000 and 758000. Most lines are expense/tiers; planning should treat revenue as a thin corridor
and test concentration in 758000 (23 lines, all R2R, exec0003, 2025-P4–P6)."
✅ GOOD (table + prose): Top-10 GL table, then a paragraph on why 603000/641100/645100 appear in every BP.
❌ BAD: A section that is only a markdown table with no explanation.
❌ BAD: "Revenue accounts are suspiciously low — possible fraud."

**Remember:** You are building the **foundation** for planning. Planning phase will interpret your findings.
"""

ORIENTATION_SCREENING_RULES = (
    ORIENTATION_SCREENING_RULES.rstrip() + "\n\n" + ORIENTATION_MANDATORY_CHECKLIST
)

ORIENTATION_CURRENT_STEP_PREFIX = (
    "[Current Step]\n\n"
    "Active database profiling step. SQL/tool results in this turn are **not** retained next step — "
    "only `orientation_report` appends persist.\n"
)

ORIENTATION_CURRENT_STEP_KICKOFF_SUFFIX = (
    "\n\nStart database profiling, then **`orientation_report(mode=append)`** early and often "
    "(substantive `##` sections: **investigative prose** with measured facts, tables as support). "
    "**Explain the ledger like field notes — NO fraud conclusions.**"
)

ORIENTATION_CURRENT_STEP_EPHEMERAL_SQL = (
    "[Current Step]\n\n"
    "Review SQL/analysis output **below** (ephemeral). Write what you **understand** about the data, then "
    '`orientation_report(mode="append")` with a `##` section: **2+ paragraphs of synthesis** plus '
    "supporting tables if needed (no raw 1000-row grids). **Update the report before your next query** — "
    "chat SQL is not kept on the next step. **Structural facts only — no fraud vocabulary.**\n"
)

ORIENTATION_CURRENT_STEP_FRESH = (
    "[Current Step]\n\n"
    "New step — no SQL in context. Run screening tools, then append to the orientation report. "
    "**Focus: What exists? How much? How distributed?**\n"
)

# Backward-compatible alias for imports.
ORIENTATION_PROMPT = ORIENTATION_SCREENING_RULES


def build_orientation_prompt(*, budget_block: str = "") -> str:
    """Legacy: full screening rules (prefer kickoff under [Current Step] only)."""
    if not budget_block.strip():
        return ORIENTATION_SCREENING_RULES
    return f"{budget_block.strip()}\n\n---\n\n{ORIENTATION_SCREENING_RULES}"


def build_orientation_kickoff_message(*, budget_block: str = "") -> str:
    """First orientation user message: rules live under [Current Step], not a separate history slot."""
    body = ORIENTATION_CURRENT_STEP_PREFIX + ORIENTATION_SCREENING_RULES
    if budget_block.strip():
        body = f"{budget_block.strip()}\n\n---\n\n{body}"
    return body + ORIENTATION_CURRENT_STEP_KICKOFF_SUFFIX
