"""
Prompts package (hypothesis_orchestrated v2).

Active prompts: orientation, planning, system, scheme/hypothesis workers, compaction.
Retired PER-phase prompts (reflection, synthesis, cross-scheme, replan, phase_summary,
legacy worker orchestration) were removed — see git history.
"""

from .compaction import COMPACTION_PROMPT
from .fraud_catalogue import FRAUD_CATALOGUE
from .hypothesis_worker import (
    build_hypothesis_summary_prompt,
    build_hypothesis_worker_prompt,
    build_worker_brief_prompt,
    build_worker_summary_prompt,
    parse_hypothesis_summary_json,
)
from .minimal_scheme_cards import build_minimal_scheme_cards
from .missions import TASK_MISSIONS
from .orientation import (
    ORIENTATION_PROMPT,
    build_orientation_kickoff_message,
    build_orientation_prompt,
)
from .planning import build_planning_prompt
from .react_suffix import REACT_SUFFIX
from .schema import DB_SCHEMA
from .scheme_phase import build_scheme_phase_prompt
from .system_prompt import build_system_prompt

__all__ = [
    "DB_SCHEMA",
    "FRAUD_CATALOGUE",
    "TASK_MISSIONS",
    "build_minimal_scheme_cards",
    "build_system_prompt",
    "ORIENTATION_PROMPT",
    "build_orientation_prompt",
    "build_orientation_kickoff_message",
    "build_planning_prompt",
    "build_scheme_phase_prompt",
    "build_hypothesis_worker_prompt",
    "build_hypothesis_summary_prompt",
    "parse_hypothesis_summary_json",
    "build_worker_brief_prompt",
    "build_worker_summary_prompt",
    "COMPACTION_PROMPT",
    "REACT_SUFFIX",
]
