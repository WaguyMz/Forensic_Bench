SCHEME_CARDS = {
    "fictitious_ap_disbursements": {
        "process_anchor": "P2P / accounts payable / disbursements",
        "economic_meaning": (
            "Cash leaves the company for goods or services that were not genuinely "
            "received, approved, or owed."
        ),
    },
    "revenue_manipulation": {
        "process_anchor": "O2C and R2R / revenue recognition / period close",
        "economic_meaning": (
            "Revenue, expense timing, or reserves are adjusted to inflate current-period "
            "earnings (early revenue, expense deferral, reserve release)."
        ),
    },
    "vendor_collusion": {
        "process_anchor": "Procurement / vendor onboarding / invoice approval",
        "economic_meaning": (
            "An internal actor and an external vendor coordinate to divert value "
            "through biased, inflated, or related-party purchasing activity."
        ),
    },
    "shadow_payroll": {
        "process_anchor": "H2R / payroll / employee master data",
        "economic_meaning": (
            "Payroll is diverted to a synthetic or colluding beneficiary, often "
            "onboarded in HR with plausible identity "
        ),
    },
    "inventory_manipulation": {
        "process_anchor": "Inventory / stock movements / write-offs / close",
        "economic_meaning": (
            "Inventory is inflated, concealed, or written off in a way that is not "
            "consistent with real stock movement or physical operations."
        ),
    },
}


def build_minimal_scheme_cards(task: str = "full") -> str:
    """
    Render a minimal closed-world scheme catalogue.

    This variant provides only high-level business-process anchors and economic meaning.
    It avoids prescribing investigative angles, control checks, SQL patterns, or expected
    anomaly signatures so the model must derive its own path from the ledger.
    """

    if task != "full" and task in SCHEME_CARDS:
        keys = [task]
    else:
        keys = list(SCHEME_CARDS.keys())

    lines = [
        "## Closed-World Scheme Cards",
        "",
        "The benchmark contains only the scheme families listed below.",
        "These are problem labels and process anchors, not solution templates.",
        "You are NOT given expected SQL tests, control-failure checklists, or",
        "known account patterns. You must derive those from the ledger.",
        "",
    ]

    for idx, key in enumerate(keys, start=1):
        card = SCHEME_CARDS[key]
        lines.append(f"### {idx}. {key}")
        lines.append(f"- Process anchor: {card['process_anchor']}")
        lines.append(f"- Economic meaning: {card['economic_meaning']}")
        lines.append("")

    allowed = ", ".join(
        keys if task != "full" and task in SCHEME_CARDS else SCHEME_CARDS
    )
    lines.extend(
        [
            "Allowed final labels:",
            f"- {allowed}",
            "- unknown",
        ]
    )
    return "\n".join(lines)
