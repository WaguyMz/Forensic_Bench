TASK_MISSIONS: dict[str, str] = {
    "full": (
        "Your mission is a **comprehensive forensic investigation** focused exclusively "
        "on the five fraud schemes in the catalogue. You have no prior knowledge of "
        "which schemes, if any, are present in this ledger. "
        "Apply the full fraud catalogue: (1) fictitious AP disbursements, (2) revenue "
        "manipulation, (3) vendor collusion, (4) shadow payroll, "
        "and (5) inventory manipulation. "
        "Begin with a broad orientation of the ledger, then form and test hypotheses "
        "systematically across these five scheme types only. "
        "**You have a large investigation budget — treat this as a full engagement, not a quick scan. "
        "A superficial investigation that finishes early is a failed investigation. "
        "Use your budget to go deep: multiple hypotheses per scheme, multiple queries per hypothesis, "
        "drill-down on every promising lead before moving on.**"
    ),
    # -----------------------------------------------------------------------
    # Per-scheme missions used by parallel worker agents.
    # Each worker investigates exactly one scheme in isolation.
    # -----------------------------------------------------------------------
    "fictitious_ap_disbursements": (
        "Your mission is a **focused investigation of fictitious AP disbursements** "
        "(salami-slicing via fictitious invoices, expense-laundering through real vendors, "
        "systematic small-amount diversions to shell or front vendors). "
        "You are a **parallel worker** — other agents are simultaneously investigating "
        "the four other fraud schemes. Do NOT investigate other scheme types. "
        "Prioritise your queries on AP-cycle JEs, below-threshold payments, repeated small credits "
        "to the same GL/vendor, suspense account activity, and vendor master anomalies. "
        "Call `report_suspicion` immediately for every confirmed finding. "
        "Do NOT call `finish_investigation`."
    ),
    "revenue_manipulation": (
        "Your mission is a **focused investigation of revenue manipulation** "
        "(quarter-end inflation, premature recognition, next-period reversals). "
        "You are a **parallel worker** — other agents are simultaneously investigating "
        "the four other fraud schemes. Do NOT investigate other scheme types. "
        "Prioritise your queries on revenue GL accounts, period-end postings, "
        "manual journal entries, and reversal patterns. "
        "Call `report_suspicion` immediately for every confirmed finding. "
        "Do NOT call `finish_investigation`."
    ),
    "vendor_collusion": (
        "Your mission is a **focused investigation of vendor collusion** "
        "(inflated invoices with kickbacks, related-party undisclosed relationships, "
        "vendor-employee bank account sharing, insider approval of affiliated-vendor payments). "
        "You are a **parallel worker** — other agents are simultaneously investigating "
        "the four other fraud schemes. Do NOT investigate other scheme types. "
        "Prioritise your queries on vendor payment amounts, vendor–employee relationships, "
        "bank account cross-matches, high-value single-vendor concentration, and approval metadata. "
        "Call `report_suspicion` immediately for every confirmed finding. "
        "Do NOT call `finish_investigation`."
    ),
    "shadow_payroll": (
        "Your mission is a **focused investigation of shadow payroll fraud** "
        "(ghost employees, payroll diversions, and terminated-employee payments). "
        "You are a **parallel worker** — other agents are simultaneously investigating "
        "the four other fraud schemes. Do NOT investigate other scheme types. "
        "Prioritise your queries on payroll-related GL accounts, the employees table, "
        "and je_header/je_line entries linked to payroll postings. "
        "Call `report_suspicion` immediately for every confirmed finding. "
        "Do NOT call `finish_investigation`."
    ),
    "inventory_manipulation": (
        "Your mission is a **focused investigation of inventory manipulation** "
        "(fictitious receipt postings, write-off abuse, shrinkage overstatement). "
        "You are a **parallel worker** — other agents are simultaneously investigating "
        "the four other fraud schemes. Do NOT investigate other scheme types. "
        "Prioritise your queries on inventory GL accounts, goods-receipt JEs, "
        "write-off entries, and cost-of-goods postings. "
        "Call `report_suspicion` immediately for every confirmed finding. "
        "Do NOT call `finish_investigation`."
    ),
}
