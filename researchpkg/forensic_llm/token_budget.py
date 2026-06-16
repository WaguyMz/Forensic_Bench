"""
Token budget tracking for the forensic agent.

Counts tokens using:
1. Actual usage from the API response (``response.usage``) when available.
2. tiktoken (OpenAI models) or the model HuggingFace tokenizer (via
   ``model_tokenizer``) for context sizing — configured at agent startup.

The tracker is shared across all steps in a single run and exposed to the
orchestrator (never to the LLM itself).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from researchpkg.forensic_llm.model_tokenizer import (
    configure_token_counter,
    count_messages_tokens,
    count_tokens,
    get_token_counter,
)

log = logging.getLogger(__name__)

__all__ = [
    "BudgetTracker",
    "configure_token_counter",
    "count_messages_tokens",
    "count_tokens",
    "effective_reasoning_tokens",
    "estimate_tool_context_tokens",
    "get_token_counter",
    "parallel_orchestrator_token_cap",
    "parallel_token_budget_unit",
    "per_agent_token_cap",
]


def effective_reasoning_tokens(
    api_reasoning: int,
    reasoning_text: str = "",
) -> int:
    """
    Reasoning tokens to count against the budget.

    Prefer the API ``reasoning_tokens`` field when the backend reports it.
    When it is zero but a separate reasoning trace is present (common on some
    vLLM thinking setups), estimate from the reasoning text only — never from
    visible completion/content, to avoid double-counting with completion_tokens.
    """
    reported = max(0, int(api_reasoning or 0))
    if reported > 0:
        return reported
    text = (reasoning_text or "").strip()
    if not text:
        return 0
    return count_tokens(text)


def estimate_tool_context_tokens(text: str, tool_name: str = "") -> int:
    """
    Approximate tokens for a tool result shown in the audit trace.

    This is informational only and does NOT affect :meth:`BudgetTracker.used_tokens`.
    ``read_image`` results omit base64 payloads (billed on the next LLM turn as
    prompt_tokens when the image is attached).
    """
    if not text:
        return 0
    if tool_name == "read_image":
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("data_url"):
                size = payload.get("size_bytes", "?")
                slim = {k: v for k, v in payload.items() if k != "data_url"}
                slim["data_url"] = f"<omitted image {size} bytes>"
                text = json.dumps(slim)
        else:
            text = re.sub(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
                "<omitted image data_url>",
                text,
            )
    return count_tokens(text)


def parallel_token_budget_unit(
    global_max_tokens: int,
    num_scheme_phases: int,
    orchestrator_weight: float = 7.0,
) -> int:
    """
    One share of the global prompt-token budget.

    The global budget is divided into ``num_scheme_phases + orchestrator_weight``
    equal units.  Each subagent receives **one** unit; the orchestrator receives
    ``orchestrator_weight`` units (see ``parallel_orchestrator_token_cap``).
    orchestrator_weight may be a float (e.g. 2.5) to achieve exact per-agent
    budgets when global / (n_schemes + weight) must be a round number.
    """
    denom = max(1.0, float(num_scheme_phases) + float(orchestrator_weight))
    return max(1, int(global_max_tokens / denom))


def parallel_orchestrator_token_cap(
    global_max_tokens: int,
    num_scheme_phases: int,
    orchestrator_weight: float = 7.0,
) -> int:
    """Prompt-token cap for the parallel parent (orchestrator)."""
    w = max(0.0, float(orchestrator_weight))
    unit = parallel_token_budget_unit(
        global_max_tokens, num_scheme_phases, orchestrator_weight
    )
    return max(1, int(round(w * unit)))


def per_agent_token_cap(
    global_max_tokens: int,
    num_scheme_phases: int,
    orchestrator_weight: float = 7.0,
) -> int:
    """
    Per-subagent prompt-token cap (one budget unit).

    Alias for :func:`parallel_token_budget_unit` kept for call-site clarity.
    """
    return parallel_token_budget_unit(
        global_max_tokens, num_scheme_phases, orchestrator_weight
    )


class BudgetTracker:
    """
    Tracks token consumption across the lifetime of one investigation run.

    Budget accounting
    -----------------
    The budget is enforced against **prompt_tokens plus separately reported
    reasoning_tokens**. Prompt growth captures context pressure, while reasoning
    tokens matter for thinking-mode runs where hidden deliberation can dominate
    per-step cost. Visible completion tokens and tool-output tokens are still
    tracked separately for informational / reporting purposes and do NOT count
    against the budget unless the backend reports them specifically as
    reasoning_tokens.
    """

    def __init__(
        self,
        max_tokens: int,
        warn_threshold: float = 0.80,
        stop_threshold: float = 0.90,
    ) -> None:
        self.max_tokens = max_tokens
        self.warn_threshold = warn_threshold
        self.stop_threshold = stop_threshold

        self._prompt_tokens: int = 0  # INPUT tokens — budgeted
        self._reasoning_tokens: int = 0  # hidden reasoning — budgeted when exposed
        self._completion_tokens: int = 0  # OUTPUT tokens — informational
        self._tool_tokens: int = 0  # tool-result text tokens — informational
        self._code_interpreter_calls: int = 0
        self._steps: int = 0
        self._warned: bool = False
        self._phases: dict = {}  # phase_name → {prompt_tokens_start, ...}

    def set_max_tokens(self, max_tokens: int) -> None:
        """Adjust the prompt-token ceiling (e.g. after the plan fixes scheme count)."""
        self.max_tokens = max(1, int(max_tokens))
        self._check_thresholds()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        reasoning_text: str = "",
        tool_result_text: str = "",
    ) -> None:
        """
        Call this after every LLM API response.

        Parameters
        ----------
        prompt_tokens     : from response.usage.prompt_tokens (counts toward budget)
        completion_tokens : from response.usage.completion_tokens (informational)
        reasoning_tokens  : API-reported reasoning/thinking tokens (budgeted)
        reasoning_text    : separate reasoning trace; token-estimated for the budget
                            only when ``reasoning_tokens`` is zero
        tool_result_text  : deprecated no-op (tool context is recorded per tool via
                            :meth:`record_tool_tokens` to avoid double counting)
        """
        self._prompt_tokens += prompt_tokens
        self._reasoning_tokens += effective_reasoning_tokens(
            reasoning_tokens, reasoning_text
        )
        self._completion_tokens += completion_tokens
        if tool_result_text:
            log.debug(
                "record_llm_call tool_result_text is ignored; use record_tool_tokens"
            )
        self._steps += 1
        self._check_thresholds()

    def record_tool_tokens(self, text: str, tool_name: str = "") -> int:
        """
        Record informational tool-context tokens and return the count.

        Does not affect :attr:`used_tokens`. Per-tool counts are stored on audit
        trace ``tool_calls[].tokens_output`` (tool context size, not LLM output).
        """
        n = estimate_tool_context_tokens(text, tool_name)
        self._tool_tokens += n
        return n

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def used_tokens(self) -> int:
        """Tokens counted against the budget (prompt + reasoning)."""
        return self._prompt_tokens + self._reasoning_tokens

    @property
    def total_billed_tokens(self) -> int:
        """
        Prompt + completion tokens reported by the API.

        reasoning_tokens are not added again here because some providers expose
        them as a subset of completion_tokens.
        """
        return self._prompt_tokens + self._completion_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def fraction_used(self) -> float:
        if self.max_tokens == 0:
            return 1.0
        return self.used_tokens / self.max_tokens

    @property
    def steps(self) -> int:
        return self._steps

    def should_warn(self) -> bool:
        return not self._warned and self.fraction_used >= self.warn_threshold

    def should_stop(self) -> bool:
        """True when the agent should stop issuing new tool calls."""
        return self.fraction_used >= self.stop_threshold

    def is_exhausted(self) -> bool:
        return self.used_tokens >= self.max_tokens

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_thresholds(self) -> None:
        if self.should_warn():
            self._warned = True
            log.warning(
                "Token budget %.0f%% used (%d / %d tokens). "
                "Stop threshold at %.0f%%.",
                self.fraction_used * 100,
                self.used_tokens,
                self.max_tokens,
                self.stop_threshold * 100,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-phase budget tracking (Plan-Execute-Reflect)
    # ------------------------------------------------------------------

    def start_phase(self, phase_name: str) -> None:
        """Begin tracking a named budget phase."""
        self._phases[phase_name] = {
            "prompt_tokens_start": self._prompt_tokens,
            "reasoning_tokens_start": self._reasoning_tokens,
            "completion_tokens_start": self._completion_tokens,
            "steps_start": self._steps,
            "sql_calls": 0,
            "code_interpreter_calls": 0,
            "write_csv_calls": 0,
            "graph_query_calls": 0,
        }

    def record_phase_sql(self, phase_name: str) -> None:
        """Increment SQL call count for the active phase."""
        if phase_name in self._phases:
            self._phases[phase_name]["sql_calls"] += 1

    def record_phase_code_interpreter(self, phase_name: str) -> None:
        """Increment code_interpreter call count for the active phase."""
        self._code_interpreter_calls += 1
        if phase_name in self._phases:
            self._phases[phase_name]["code_interpreter_calls"] += 1

    def record_phase_write_csv(self, phase_name: str) -> None:
        if phase_name in self._phases:
            self._phases[phase_name]["write_csv_calls"] += 1

    def record_phase_graph_query(self, phase_name: str) -> None:
        if phase_name in self._phases:
            self._phases[phase_name]["graph_query_calls"] += 1

    def phase_tokens_used(self, phase_name: str) -> int:
        """Budget-counted tokens consumed since this phase started."""
        if phase_name not in self._phases:
            return 0
        return (
            self._prompt_tokens
            - self._phases[phase_name]["prompt_tokens_start"]
            + self._reasoning_tokens
            - self._phases[phase_name]["reasoning_tokens_start"]
        )

    def phase_sql_calls(self, phase_name: str) -> int:
        if phase_name not in self._phases:
            return 0
        return self._phases[phase_name]["sql_calls"]

    def phase_code_interpreter_calls(self, phase_name: str) -> int:
        if phase_name not in self._phases:
            return 0
        return self._phases[phase_name]["code_interpreter_calls"]

    def phase_effective_sql_calls(self, phase_name: str) -> int:
        """Investigation-depth proxy: sql + analysis + export + graph tools."""
        if phase_name not in self._phases:
            return 0
        p = self._phases[phase_name]
        return (
            p["sql_calls"]
            + p["code_interpreter_calls"]
            + p.get("write_csv_calls", 0)
            + p.get("graph_query_calls", 0)
        )

    def phase_steps(self, phase_name: str) -> int:
        if phase_name not in self._phases:
            return 0
        return self._steps - self._phases[phase_name]["steps_start"]

    def phase_budget_exceeded(self, phase_name: str, max_tokens: int) -> bool:
        return self.phase_tokens_used(phase_name) >= max_tokens

    def phase_summary(self, phase_name: str) -> dict:
        if phase_name not in self._phases:
            return {}
        p = self._phases[phase_name]
        return {
            "phase": phase_name,
            "prompt_tokens": self._prompt_tokens - p["prompt_tokens_start"],
            "reasoning_tokens": self._reasoning_tokens - p["reasoning_tokens_start"],
            "budget_counted_tokens": (
                self._prompt_tokens
                - p["prompt_tokens_start"]
                + self._reasoning_tokens
                - p["reasoning_tokens_start"]
            ),
            "completion_tokens": self._completion_tokens - p["completion_tokens_start"],
            "steps": self._steps - p["steps_start"],
            "sql_calls": p["sql_calls"],
            "code_interpreter_calls": p["code_interpreter_calls"],
            "effective_sql_calls": p["sql_calls"] + p["code_interpreter_calls"],
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        base = {
            "budget_prompt_tokens": self._prompt_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "budget_counted_tokens": self.used_tokens,
            "budget_max_tokens": self.max_tokens,
            "budget_fraction_used": round(self.fraction_used, 4),
            "budget_remaining_tokens": self.remaining_tokens,
            "budget_exhausted": self.is_exhausted(),
            "completion_tokens": self._completion_tokens,
            "total_billed_tokens": self.total_billed_tokens,
            "tool_context_tokens_approx": self._tool_tokens,
            "code_interpreter_calls": self._code_interpreter_calls,
            "steps": self._steps,
        }
        if self._phases:
            base["phases"] = {name: self.phase_summary(name) for name in self._phases}
        return base

    def __repr__(self) -> str:
        return (
            f"BudgetTracker("
            f"used={self.used_tokens:,}/{self.max_tokens:,} "
            f"[{self.fraction_used:.1%}], steps={self._steps})"
        )
