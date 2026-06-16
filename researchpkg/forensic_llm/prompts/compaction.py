COMPACTION_PROMPT = """Summarise the key findings, hypotheses tested, evidence discovered, and any dead ends from the investigation steps above.

Your goal is to produce a **rich, highly detailed memory summary** that can be safely reused later, up to the memory summary token budget provided in the system instructions. Do **not** self-limit to a few hundred tokens; instead, use as much detail as is useful while staying within that budget.

The summary MUST explicitly preserve:
- Entity IDs (employee, vendor, customer) that are suspicious
- Document IDs that have been flagged
- GL account numbers that showed anomalies
- Monetary amounts and date ranges
- Which scheme types each finding relates to
- Open hypotheses and next investigation steps

Do NOT include raw SQL queries or full table outputs — only the conclusions drawn from them."""
