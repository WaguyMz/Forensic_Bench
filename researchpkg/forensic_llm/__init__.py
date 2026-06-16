"""
forensic_llm – Agentic LLM Forensic Investigator
=================================================

Autonomous forensic auditing agent that investigates journal-entry databases
for fraud schemes, control failures, and statistical anomalies.

Quick start
-----------
>>> from forensic_llm import ForensicAgent, InvestigatorConfig
>>> cfg = InvestigatorConfig()        # uses defaults: local vLLM on :8020
>>> report = ForensicAgent(cfg).run()
>>> print(f"{len(report.suspicion_list)} suspicion(s) found")

CLI
---
python -m forensic_llm.run --help
python -m forensic_llm.run --dry-run
python -m forensic_llm.run --task shadow_payroll --evaluate
"""

from .agent import ForensicAgent, run_ensemble
from .config import BudgetConfig, DatabaseConfig, InvestigatorConfig, LLMConfig
from .evaluator import compare_runs, evaluate, evaluate_and_save, evaluate_schemes
from .models import (
    EvaluationResult,
    ForensicReport,
    InvestigationPlan,
    SchemeEvalMetrics,
    SchemeEvalSummary,
    SchemePhase,
    SchemeReport,
    SchemeType,
    SuspicionItem,
)
from .token_budget import BudgetTracker

__all__ = [
    # Agent
    "ForensicAgent",
    "run_ensemble",
    # Config
    "InvestigatorConfig",
    "LLMConfig",
    "DatabaseConfig",
    "BudgetConfig",
    # Models
    "ForensicReport",
    "SuspicionItem",
    "SchemeType",
    "SchemeReport",
    "SchemePhase",
    "InvestigationPlan",
    "EvaluationResult",
    "SchemeEvalSummary",
    "SchemeEvalMetrics",
    # Evaluation
    "evaluate",
    "evaluate_and_save",
    "evaluate_schemes",
    "compare_runs",
    # Budget
    "BudgetTracker",
]

__version__ = "0.1.0"
