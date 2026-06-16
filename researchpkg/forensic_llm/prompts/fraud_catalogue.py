FRAUD_CATALOGUE = """
## Known Fraud Scheme Catalogue

**Only these 5 scheme types exist in the data** (no other fraud types are injected).
Each entry describes the **business process**, what “healthy” operations look like in substance,
and the **kinds of breakdown** that can indicate manipulation. It does **not** name expected GL
accounts, auxiliary formats, posting templates, or calendar cutoffs — those must come from **your**
profiling of this dataset (chart of accounts, distributions, joins, and master data).

---

### 1. Fictitious AP Disbursements

**Normal process (substance):** The organisation should only settle payables for goods or services
that were ordered, received or accepted, and invoiced under segregation of duties (e.g. different
actors for recording vs authorising payment). Vendor identity and payment references should be
consistent with normal operating practice.

**When something is wrong (conceptual):** Value may leave the entity without a defensible
procurement-to-pay trail: missing or weak evidence of receipt or approval, payables cleared without
a credible invoice trail, the same actors repeatedly spanning incompatible roles, timing or
vendor populations that diverge sharply from the rest of the ledger, or reference behaviour that
does not match how the company usually books AP and bank movements.

---

### 2. Revenue Manipulation (Window Dressing)

**Normal process (substance):** Revenue aligns with delivery or completion of performance
obligations; expenses and provisions are recognised in the appropriate periods; releases of
reserves or similar items require defensible economic events and governance.

**When something is wrong (conceptual):** Earnings can be inflated by recognising revenue too early,
pushing expenses out of the period, or releasing balances that should still reflect risk or
obligation. Suspect patterns are **relative**: unusual concentration in time, abnormal mixes of
lines or accounts for the same commercial story, or journal narratives and flows that do not
cohere with the rest of the order-to-cash and record-to-report activity in **this** database.

---

### 3. Vendor Collusion (Kickback + Related-Party)

**Normal process (substance):** Procurement and payment follow approved sourcing, pricing, and
approval chains; vendors are genuine third parties at arm’s length; segregation limits a single
actor from controlling selection, invoicing, and payment.

**When something is wrong (conceptual):** Inflated or biased spend, repeated coupling of the same
internal and external parties without rotation, payment behaviour that fragments or obscures totals,
weak master-data linkage, or indications that beneficiary or party data overlaps with insiders.
Signals are contextual: compare to peer vendors, amounts, and approval patterns in the same data.

---

### 4. Shadow Payroll (Ghost Employee)

**Normal process (substance):** Payroll pays real employees under HR and payroll policies, with
accruals and settlements that match the entity’s payroll design (which may include multiple line
types, schedules, and banking arrangements you must infer from the data).

**When something is wrong (conceptual):** Beneficiaries or pay runs that are structurally odd
relative to the population (e.g. accrual shape, timing clusters, banking or institution patterns,
recency of employment lifecycle events vs pay activity, or concentration of authorship) can indicate
diversion or synthetic beneficiaries. **Plausible HR rows do not rule out fraud** — treat master
data as necessary context, not as a whitelist.

---

### 5. Inventory Manipulation (Asset Misappropriation)

**Normal process (substance):** Inventory moves should trace to economic events (receipts,
consumption, sales, authorised adjustments). Write-downs and adjustments require authorised,
documented physical or operational justification.

**When something is wrong (conceptual):** Stock increases without credible upstream support,
write-offs disconnected from prior stock history, suspicious symmetry or pairing of movements,
period-close clustering of adjustments vs norms, or split roles across related postings may indicate
inflation, concealment, or misappropriation. Map **this** entity’s inventory-related accounts and
references from the ledger before concluding.

---

**How to use this catalogue:** Use it to frame **hypotheses** and **questions**, not as a
checklist of built-in tests. All concrete controls, accounts, SQL filters, and thresholds must be
**derived and validated** against the chart of accounts and facts in the database.
""".strip()
