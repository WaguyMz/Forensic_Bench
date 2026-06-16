"""
Forensic LLM Agent — the core agentic loop.

Architecture
------------
Default strategy: **hypothesis_orchestrated** (core_spec v2).

  1. Orientation — ``orientation_budget_fraction`` of global run budget; ledger screening.
     Cumulative memory is ``orientation/orientation_report.md`` (model appends via
     ``orientation_report``; flushed to disk on each append and at step end).
  2. Planning — one-shot ``investigation_plan.json`` with ``dispatch_queue``.
  3. Hypothesis tasks — one worker per hypothesis (P1…Pn), serial by default.
  4. Shared memory — ``memory/global.json`` + blackboard (no reflection/synthesis).

The runtime is **hypothesis_orchestrated** only (orientation → planning JSON →
hypothesis task workers).

All phases share tool-calling, message-history, and persistence machinery.
A 5-slot context memory keeps the payload within the model input budget.

Scratchpad: updated live to run_dir/scratchpad.md on every scratchpad() call;
step-wise snapshots are written to run_dir/scratchpad_steps/step_NNNN.md after
each step for post-debug (how the agent's reasoning evolved).

Context management: before every LLM call the payload is checked against
``effective_context_window - completion_reserve - context_safety_margin_tokens``
(2% below the hard window by default); if over, context is trimmed. See
CONTEXT_MANAGEMENT.md.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from researchpkg.forensic_llm.config import (
    InvestigatorConfig,
)
from researchpkg.forensic_llm.llm_client import (
    LLMResponse,
    ToolCallRequest,
    build_client,
)
from researchpkg.forensic_llm.models import (
    AgentStep,
    CoverageEntry,
    CoverageLedger,
    EvidenceEntry,
    ForensicReport,
    HypothesisResult,
    HypothesisTaskBrief,
    InvestigationPlan,
    PhaseSummary,
    SchemePhase,
    SchemeReport,
    SchemeType,
    SharedEvidenceBlackboard,
    SuspicionItem,
    ToolCallRecord,
    WorkerBrief,
    WorkerMessage,
    WorkerStatus,
    WorkerSummary,
)
from researchpkg.forensic_llm.orientation_utils import (
    build_orientation_synthesis_bundle,
    orientation_budget_pct,
)
from researchpkg.forensic_llm.plan_utils import (
    fallback_initial_hypotheses_for_scheme,
)
from researchpkg.forensic_llm.prompts import (
    COMPACTION_PROMPT,
    ORIENTATION_PROMPT,
    REACT_SUFFIX,
    build_planning_prompt,
    build_scheme_phase_prompt,
    build_system_prompt,
)
from researchpkg.forensic_llm.prompts.hypothesis_worker import (
    build_worker_brief_prompt,
    build_worker_summary_prompt,
)
from researchpkg.forensic_llm.text_truncation import (
    TruncationSide,
    truncate_message_to_tokens,
    truncate_text_to_tokens,
)
from researchpkg.forensic_llm.token_budget import (
    BudgetTracker,
    configure_token_counter,
    count_messages_tokens,
    count_tokens,
)
from researchpkg.forensic_llm.tool_defs import (
    AUTONOMOUS_ORCHESTRATION_TOOL_NAMES,
    DB_ONLY_TOOLS,
    DB_ONLY_TOOLS_NO_GRAPH,
)
from researchpkg.forensic_llm.tools import (
    dispatch,
    get_scratchpad_text,
    init_tools,
    reset_e2b_sandbox,
    reset_graph,
    reset_scratchpad,
    scratchpad,
    set_scratchpad_run_context,
)

log = logging.getLogger(__name__)

# Thread-local scheme label — set by worker threads so every log line emitted
# within a parallel subagent is automatically prefixed with "[scheme_name]".
_THREAD_SCHEME: threading.local = threading.local()
_CLOSED_WORLD_SCHEMES: Tuple[str, ...] = (
    "fictitious_ap_disbursements",
    "revenue_manipulation",
    "vendor_collusion",
    "shadow_payroll",
    "inventory_manipulation",
)

_JE_DOCUMENT_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _normalise_doc_id(doc_id: Optional[str]) -> str:
    if not doc_id:
        return ""
    return str(doc_id).lower().strip().replace("-", "")


def _looks_like_je_document_uuid(doc_id: Optional[str]) -> bool:
    if not doc_id:
        return False
    s = str(doc_id).strip()
    if s.lower().startswith("scheme-"):
        return False
    return bool(_JE_DOCUMENT_UUID_RE.match(s))


@dataclass
class _WorkerRuntimeState:
    brief: WorkerBrief
    worker_dir: pathlib.Path
    thread: Optional[threading.Thread] = None
    status: str = "pending"
    mailbox: List[WorkerMessage] = field(default_factory=list)
    summary: Optional[WorkerSummary] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    steps_taken: int = 0
    sql_calls_used: int = 0
    code_interpreter_calls_used: int = 0
    flagged_document_ids: List[str] = field(default_factory=list)
    latest_summary: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class _SchemeFilter(logging.Filter):
    """Prepend [scheme_name] to log messages when running in a worker thread."""

    def filter(self, record: logging.LogRecord) -> bool:
        scheme = getattr(_THREAD_SCHEME, "name", None)
        if scheme:
            record.msg = f"[{scheme}] {record.msg}"
        return True


log.addFilter(_SchemeFilter())

# ---------------------------------------------------------------------------
# ReAct parser
# ---------------------------------------------------------------------------

_REACT_ACTION_RE = re.compile(
    r"Action\s*:\s*(\w+)\s*\nAction\s*Input\s*:\s*(\{.*?\}|\[.*?\])",
    re.DOTALL | re.IGNORECASE,
)

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.DOTALL,
)


def _parse_react(text: str) -> List[ToolCallRequest]:
    """
    Extract tool calls from a ReAct-style assistant message.

    Supports three formats:
    1. Action: name\\nAction Input: {...}
    2. ```json\\n{...}\\n``` blocks containing "action"/"action_input" keys
    3. XML-style <tool_call>{"name":...,"arguments":...}</tool_call>
    """
    calls: List[ToolCallRequest] = []

    # Format 1 – explicit Action/Action Input
    for m in _REACT_ACTION_RE.finditer(text):
        name = m.group(1).strip()
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCallRequest(id=str(uuid.uuid4()), name=name, arguments=args))

    if calls:
        return calls

    # Format 2 – JSON code blocks with action/action_input keys
    for m in _JSON_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            name = obj["action"]
            args = obj.get("action_input", obj.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            calls.append(
                ToolCallRequest(id=str(uuid.uuid4()), name=name, arguments=args)
            )

    if calls:
        return calls

    # Format 3 – <tool_call>...</tool_call> XML-ish tags
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool") or ""
        args = obj.get("arguments") or obj.get("parameters") or {}
        if name:
            calls.append(
                ToolCallRequest(id=str(uuid.uuid4()), name=name, arguments=args)
            )

    return calls


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Find and parse the first JSON object in *text* using a proper decoder
    (handles nested braces correctly; avoids the greedy-regex trap).
    Falls back to code-block extraction if raw_decode finds nothing.
    """
    # Try raw_decode from the first '{' found
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: extract from a fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Suspicion list parser / validator
# ---------------------------------------------------------------------------

_SCHEME_NAMES = {s.value for s in SchemeType}
_DOCUMENT_ID_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# Evaluator treats every reported JE as a hard flag (no numeric confidence layer).
_EVAL_FLAG_CONFIDENCE = 1.0


def _detection_record(document_id: str, scheme_id: str) -> Dict[str, str]:
    return {"document_id": document_id, "scheme_id": scheme_id}


def _parse_report_suspicion_args(args: Dict[str, Any]) -> List[SuspicionItem]:
    """
    Parse report_suspicion tool arguments into a list of SuspicionItems.

    Accepts several common key aliases so a misspelled argument never silently
    discards findings — the most common model error is using ``"suspicious"``
    (adjective) instead of ``"suspicions"`` (noun).
    """
    items: List[SuspicionItem] = []
    # Accept any of these keys — first non-empty list wins.
    _LIST_KEYS = (
        "suspicions",
        "suspicious",
        "items",
        "detections",
        "findings",
        "entries",
    )
    suspicions = None
    for key in _LIST_KEYS:
        candidate = args.get(key)
        if candidate and isinstance(candidate, list):
            suspicions = candidate
            break
    if suspicions:
        items = _parse_suspicion_list(suspicions)
    else:
        doc_id = args.get("document_id")
        scheme = args.get("scheme_type", "unknown")
        if scheme not in _SCHEME_NAMES:
            scheme = "unknown"
        if doc_id:
            items = [
                SuspicionItem(
                    document_id=str(doc_id),
                    scheme_type=SchemeType(scheme),
                    confidence=_EVAL_FLAG_CONFIDENCE,
                    rationale=args.get("rationale", ""),
                )
            ]
    return items


def _parse_suspicion_list(raw: Any) -> List[SuspicionItem]:
    """
    Parse and validate the suspicion list, tolerating partial / messy JSON.
    Returns a list of SuspicionItem instances (possibly empty).
    Handles suspicion_list passed as a JSON array string (e.g. "[{...}]").
    """
    if isinstance(raw, str):
        raw = raw.strip()
        # Try parsing as a direct JSON array first (tool sometimes sends string)
        if raw.startswith("["):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                pass
        if isinstance(raw, str):
            # Fallback: extract a JSON array from prose
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    raw = json.loads(m.group(0))
                except json.JSONDecodeError:
                    log.warning("Could not parse suspicion list JSON")
                    return []
            else:
                return []

    if not isinstance(raw, list):
        return []

    items: List[SuspicionItem] = []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        # Normalise scheme_type
        st = obj.get("scheme_type", "unknown")
        if st not in _SCHEME_NAMES:
            st = "unknown"
        try:
            item = SuspicionItem(
                document_id=obj.get("document_id"),
                entity_id=obj.get("entity_id"),
                entity_type=obj.get("entity_type"),
                scheme_type=SchemeType(st),
                confidence=_EVAL_FLAG_CONFIDENCE,
                severity=max(1, min(5, int(obj.get("severity", 3)))),
                rationale=obj.get("rationale", ""),
                supporting_evidence=obj.get("supporting_evidence", []),
                related_document_ids=obj.get("related_document_ids", []),
                monetary_impact=obj.get("monetary_impact"),
                gl_accounts=obj.get("gl_accounts", []),
            )
            items.append(item)
        except Exception as exc:
            log.warning("Skipping malformed suspicion item: %s – %s", obj, exc)
    return items


# ---------------------------------------------------------------------------
# Message builder helpers
# ---------------------------------------------------------------------------


def _system_msg(content: str) -> Dict[str, Any]:
    return {"role": "system", "content": content}


def _user_msg(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def _user_msg_multimodal(content_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a multimodal user message for OpenAI-compatible vision backends.

    content_blocks is typically:
      [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:..."}}]
    """
    return {"role": "user", "content": content_blocks}


def _assistant_msg(
    content: str, tool_calls_raw: Optional[List] = None
) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls_raw:
        msg["tool_calls"] = tool_calls_raw
    return msg


def _tool_result_msg(tool_call_id: str, result: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def _observation_msg(observation: str) -> Dict[str, Any]:
    """Used in ReAct mode: observation is appended as a user message."""
    return _user_msg(f"Observation:\n{observation}")


def _append_tool_feedback(
    messages: List[Dict[str, Any]],
    *,
    tool_call_id: str,
    content: str,
    native_tool_calling: bool,
) -> None:
    """
    Append a tool result back into the conversation history.

    Native tool-calling backends expect every assistant tool request to be
    followed by a matching `role="tool"` response. ReAct mode instead feeds the
    result back as a user-side observation.
    """
    if native_tool_calling:
        messages.append(_tool_result_msg(tool_call_id, content))
    else:
        messages.append(_observation_msg(content))


# ---------------------------------------------------------------------------
# SQL result context compression
# ---------------------------------------------------------------------------

_SQL_RESULT_CONTEXT_MAX_ROWS: int = 10
_ORIENTATION_SQL_SYNTHESIS_REMINDER: str = (
    "\n\n[Orientation: summarize in `orientation_report(mode=append)` (short `##` section). "
    "SQL output is not kept next step — only the on-disk report persists.]"
)
"""Maximum SQL data rows kept verbatim in the message context window.

Full results are preserved in the audit trace (ToolCallRecord.result).
Reduced to 10 rows (from 25) to slow tail growth across 100-150 SQL calls,
maximising the shared prefix length reused by the KV cache between steps.
The row-count footer is always preserved so the model knows the population size.
"""


_SQL_RUNTIME_FOOTER_RE = re.compile(r"\*(\d+ row\(s\) returned) in \d+ ms\*")


def _sanitize_sql_for_context(result: str) -> str:
    """
    Remove runtime-only noise from model-visible SQL results.

    Query latency varies from run to run and should not pollute the next prompt.
    Keep the row-count footer, but strip the volatile "... in X ms" suffix.
    """
    if not result:
        return result
    return _SQL_RUNTIME_FOOTER_RE.sub(r"*\1*", result)


def _truncate_sql_for_context(
    result: str,
    *,
    max_data_rows: Optional[int] = None,
    suffix: str = "",
) -> str:
    """Trim a SQL markdown-table result to _SQL_RESULT_CONTEXT_MAX_ROWS data rows.

    The footer keeps the row count but strips volatile runtime information so
    model-visible history stays stable between equivalent runs. Non-table
    results (errors, short tables, zero rows) are returned unchanged apart from
    runtime sanitisation.
    """
    result = _sanitize_sql_for_context(result)
    if not result or not result.startswith("|"):
        return result
    lines = result.split("\n")
    # Partition into table body and footer (* ... row(s) ... *)
    body = [ln for ln in lines if not ln.startswith("*")]
    footer = [ln for ln in lines if ln.startswith("*")]
    # body[0] = header row, body[1] = separator, body[2:] = data rows
    cap = max_data_rows if max_data_rows is not None else _SQL_RESULT_CONTEXT_MAX_ROWS
    data_rows = body[2:] if len(body) > 2 else []
    if len(data_rows) <= cap:
        out = result
    else:
        truncated = body[:2] + data_rows[:cap]
        omitted = len(data_rows) - cap
        truncated.append(
            f"*… {omitted} more rows omitted from context (full result in audit trace)*"
        )
        out = "\n".join(truncated + footer)
    if suffix and out:
        return out + suffix
    return out


def _tool_result_for_model_history(
    tool_name: str,
    result: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    phase_name: str = "",
    orientation_sql_max_rows: int = 5,
) -> str:
    """Stable, model-visible version of a tool result."""
    if tool_name in ("sql", "write_csv"):
        if (
            tool_name == "write_csv"
            or str((arguments or {}).get("mode", "preview")).lower() == "export"
        ):
            if phase_name == "orientation":
                return (result or "").strip() + _ORIENTATION_SQL_SYNTHESIS_REMINDER
            return result
        if result.startswith("[SQL_EXPORT]"):
            if phase_name == "orientation":
                return (result or "").strip() + _ORIENTATION_SQL_SYNTHESIS_REMINDER
            return result
        if phase_name == "orientation":
            return _truncate_sql_for_context(
                result,
                max_data_rows=max(1, int(orientation_sql_max_rows)),
                suffix=_ORIENTATION_SQL_SYNTHESIS_REMINDER,
            )
        return _truncate_sql_for_context(result)
    if phase_name == "orientation" and tool_name == "code_interpreter":
        tail = truncate_text_to_tokens(
            (result or "").strip(),
            2_000,
            side=TruncationSide.TAIL,
        )
        return tail + _ORIENTATION_SQL_SYNTHESIS_REMINDER
    return result


def _estimate_full_payload_tokens(full_messages: List[Dict[str, Any]]) -> int:
    """
    Token count for a full message list (system + conversation).

    Uses the model HuggingFace tokenizer configured at agent startup.
    """
    return count_messages_tokens(full_messages)


def _msg_text_content(msg: Dict[str, Any]) -> str:
    """
    Safely extract a plain-text string from a message's content field.

    Handles:
    - str  — returned as-is (stripped)
    - list — multimodal content blocks; joins all "text" typed blocks
    - None — returns ""
    """
    content = msg.get("content")
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content).strip()


def _truncate_message_to_tokens(msg: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Return a copy of the message with content truncated to fit in max_tokens."""
    return truncate_message_to_tokens(msg, max_tokens)


def _assistant_tool_call_ids(msg: Dict[str, Any]) -> set[str]:
    return {
        str(tc.get("id", "")).strip()
        for tc in msg.get("tool_calls", [])
        if isinstance(tc, dict) and tc.get("id")
    }


def _message_bundle_end(messages: List[Dict[str, Any]], start_idx: int) -> int:
    """
    Return the exclusive end index of a message bundle.

    Assistant turns that contain `tool_calls` are grouped with their immediately
    following contiguous `role="tool"` results so context trimming never splits
    a tool request from its matching tool responses.
    """
    msg = messages[start_idx]
    if msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return start_idx + 1

    expected_ids = _assistant_tool_call_ids(msg)
    end_idx = start_idx + 1
    while end_idx < len(messages) and messages[end_idx].get("role") == "tool":
        tool_call_id = str(messages[end_idx].get("tool_call_id", "")).strip()
        if expected_ids and tool_call_id and tool_call_id in expected_ids:
            end_idx += 1
            continue
        break
    return end_idx


def _split_message_bundles(
    messages: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    bundles: List[List[Dict[str, Any]]] = []
    idx = 0
    while idx < len(messages):
        end_idx = _message_bundle_end(messages, idx)
        bundles.append(messages[idx:end_idx])
        idx = end_idx
    return bundles


def _current_input_bundle_start(messages: List[Dict[str, Any]]) -> int:
    """
    Return the start index of the final message bundle.

    If the last message is a tool result, include the preceding assistant
    tool-call turn and any contiguous tool results as part of the current input.
    """
    if not messages:
        return 0

    last_idx = len(messages) - 1
    if messages[last_idx].get("role") != "tool":
        return last_idx

    first_tool_idx = last_idx
    while first_tool_idx > 0 and messages[first_tool_idx - 1].get("role") == "tool":
        first_tool_idx -= 1

    assistant_idx = first_tool_idx - 1
    if (
        assistant_idx >= 0
        and messages[assistant_idx].get("role") == "assistant"
        and messages[assistant_idx].get("tool_calls")
    ):
        return assistant_idx
    return last_idx


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------


class ForensicAgent:
    """
    Autonomous forensic investigation agent.

    Usage
    -----
    cfg = InvestigatorConfig()
    agent = ForensicAgent(cfg)
    report = agent.run()
    """

    def __init__(
        self,
        config: InvestigatorConfig,
        _worker_run_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.config = config
        self.config.reconcile_derived_context_budgets()
        self._parent_blackboard: Optional[SharedEvidenceBlackboard] = None
        # Sync the run-level seed into LLMConfig so it reaches the API call,
        # giving reproducibility at temperature=0 despite GPU non-determinism.
        if config.llm.seed is None:
            config.llm.seed = config.seed
        self.client = build_client(config.llm)
        self._token_counter = configure_token_counter(
            model=config.llm.model,
            tokenizer_model=config.llm.tokenizer_model,
            base_url=config.llm.base_url,
            trust_remote_code=config.llm.tokenizer_trust_remote_code,
        )
        log.info(
            "Token counter: %s (%s)",
            self._token_counter.backend,
            self._token_counter.tokenizer_hub_id,
        )
        self._global_token_budget: int = (
            config.token_budget_global_max
            if config.token_budget_global_max is not None
            else config.budget.max_tokens
        )
        # Hypothesis-orchestrated runs use the full CLI/global budget. The legacy
        # parallel-scheme slot formula (per-scheme units) does not apply here.
        self.budget = BudgetTracker(
            max_tokens=self._global_token_budget,
            warn_threshold=config.budget.warn_threshold,
            stop_threshold=config.budget.stop_threshold,
        )
        self._use_native_tools: bool = config.llm.use_native_tools
        self._tool_failure_count: int = 0
        self._run_id: str = str(uuid.uuid4())
        self._started_at: datetime = datetime.utcnow()
        self._sql_call_count: int = 0
        self._code_interpreter_call_count: int = 0
        self._write_csv_call_count: int = 0
        self._graph_query_call_count: int = 0
        self._orchestrator_sql_count: int = 0
        self._orchestrator_code_interpreter_count: int = 0
        self._orchestrator_write_csv_count: int = 0
        self._orchestrator_graph_query_count: int = 0
        self._sql_error_count: int = 0  # SQL calls that returned [SQL ERROR]
        self._sql_zero_row_count: int = 0  # SQL calls that returned 0 rows
        self._tool_call_counts: Dict[str, int] = {}  # per-tool usage counters
        self._step_num: int = 0
        self._stream_event_lock = threading.Lock()
        # 5-slot memory state
        self._past_memory: str = (
            ""  # Slot 2: legacy eviction note (now backed by register)
        )
        self._plan_slot: str = ""  # Slot 1: plan JSON + phase summaries
        self._scratchpad_slot: str = (
            ""  # Slot 4: pinned scratchpad (after tail, lazy update)
        )
        self._scratchpad_dirty: bool = True  # rebuild slot before next LLM call
        self._orientation_report_dirty: bool = True
        self._orientation_prompt_message: Optional[Dict[str, Any]] = None
        self._current_phase_name: str = ""

        # MemGPT-style archival evidence register (Packer et al., 2023).
        # Structured hypothesis outcomes extracted from evicted turns; never
        # lost to context compaction.  Rendered as a compact table in slot 2.
        self._evidence_register: List[EvidenceEntry] = []

        # MetaGPT-style shared blackboard for parallel scheme workers.
        # Workers write entity findings; orchestrator reads for cross-scheme
        # entity linking after all workers complete.
        self._shared_blackboard: SharedEvidenceBlackboard = SharedEvidenceBlackboard()

        # Structured summaries collected at the end of each scheme phase.
        # Used to ground synthesis and metric-driven replanning.
        self._phase_summaries: Dict[str, PhaseSummary] = {}

        # Scratchpad text captured after orientation; injected into planning prompt.
        self._orientation_scratchpad: str = ""
        # Last message payload actually sent to the LLM (for memory.md snapshot).
        self._last_llm_payload: List[Dict[str, Any]] = []
        self._last_tool_signature: str = ""
        self._last_tool_signature_repeats: int = 0
        # Paths of images already attached as multimodal messages; used to
        # deduplicate read_image calls so the same base64 blob is never injected twice.
        self._shown_image_paths: set = set()

        self._worker_run_dir: Optional[pathlib.Path] = _worker_run_dir
        self._run_dir: pathlib.Path = (
            _worker_run_dir if _worker_run_dir is not None else self._make_run_dir()
        )
        self._plots_dir: pathlib.Path = self._run_dir / "plots"
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._scratchpad_steps_dir: pathlib.Path = self._run_dir / "scratchpad_steps"
        # Directory for per-step LLM call logs (full request + response).
        self._calls_steps_dir: pathlib.Path = self._run_dir / "calls_steps"

        if config.enabled_tools is not None:
            from researchpkg.forensic_llm.tool_defs import (
                tools_from_names,
            )

            self._tools = tools_from_names(config.enabled_tools)
        else:
            self._tools = DB_ONLY_TOOLS

        if self._should_expose_worker_tools():
            from researchpkg.forensic_llm.tool_defs import (
                tools_from_names,
            )

            self._tools = self._tools + tools_from_names(
                AUTONOMOUS_ORCHESTRATION_TOOL_NAMES
            )

        self._stream_trace_file = None
        self._stream_trace_lock = threading.Lock()
        self._parent_stream_sink: Optional[Tuple[threading.Lock, pathlib.Path]] = None

        # Live detections: JE-level items accumulated from any finish_investigation
        # attempt (rejected or accepted). Flushed to detections.json so partial
        # results are saved even if the run never reaches synthesis.
        self._live_detections: List[Dict[str, Any]] = []
        # Hypothesis workers: UUIDs seen in SQL/export output (for reporting nudges).
        self._worker_sql_document_ids: Set[str] = set()
        self._hypothesis_pool_lock = threading.Lock()
        self._hypothesis_worker_pool_remaining: Optional[int] = None
        self._hypothesis_task_cap: int = 0
        self._task_budget_pacing_sent: Set[str] = set()
        self._orientation_budget_pacing_sent: Set[str] = set()
        self._worker_registry: Dict[str, _WorkerRuntimeState] = {}
        self._worker_registry_lock = threading.Lock()
        self._coverage_ledger: CoverageLedger = CoverageLedger(
            entries={
                scheme: CoverageEntry(scheme=scheme) for scheme in _CLOSED_WORLD_SCHEMES
            }
        )

    # ------------------------------------------------------------------
    # Quota helpers
    # ------------------------------------------------------------------

    @property
    def _effective_sql_call_total(self) -> int:
        """Investigation-depth proxy for finish guardrails."""
        return (
            self._sql_call_count
            + self._code_interpreter_call_count
            + self._write_csv_call_count
            + self._graph_query_call_count
        )

    @property
    def _effective_orchestrator_sql_total(self) -> int:
        return (
            self._orchestrator_sql_count
            + self._orchestrator_code_interpreter_count
            + self._orchestrator_write_csv_count
            + self._orchestrator_graph_query_count
        )

    def _phase_effective_sql_calls(self, phase_name: str) -> int:
        return self.budget.phase_effective_sql_calls(phase_name)

    def _budget_sql_calls_from_plan(self, raw: Any = None) -> int:
        """Plan metadata only — not used to force early continuation."""
        if self.config.sql_max_per_core is not None:
            return int(self.config.sql_max_per_core)
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
        return int(SchemePhase.model_fields["budget_sql_calls"].default)  # type: ignore[attr-defined]

    def _record_investigation_tool(
        self,
        tool_name: str,
        phase_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track investigation-depth tools for quotas and phase metrics."""
        orchestrator = getattr(_THREAD_SCHEME, "name", None) is None
        args = arguments or {}
        if tool_name == "sql":
            if str(args.get("mode", "preview")).lower() == "export":
                self._write_csv_call_count += 1
                if phase_name:
                    self.budget.record_phase_write_csv(phase_name)
                if orchestrator:
                    self._orchestrator_write_csv_count += 1
            else:
                self._sql_call_count += 1
                if phase_name:
                    self.budget.record_phase_sql(phase_name)
                if orchestrator:
                    self._orchestrator_sql_count += 1
        elif tool_name == "write_csv":
            self._write_csv_call_count += 1
            if phase_name:
                self.budget.record_phase_write_csv(phase_name)
            if orchestrator:
                self._orchestrator_write_csv_count += 1
        elif tool_name == "code_interpreter":
            self._code_interpreter_call_count += 1
            self.budget.record_phase_code_interpreter(phase_name or "")
            if orchestrator:
                self._orchestrator_code_interpreter_count += 1
        elif tool_name == "graph_query":
            self._graph_query_call_count += 1
            if phase_name:
                self.budget.record_phase_graph_query(phase_name)
            if orchestrator:
                self._orchestrator_graph_query_count += 1

    def _is_parent_autonomous_orchestrated(self) -> bool:
        """Reserved hook for parent-managed specialist workers (disabled)."""
        return False

    def _should_expose_worker_tools(self) -> bool:
        return self._is_parent_autonomous_orchestrated()

    def _coverage_status_from_verdict(self, verdict: str) -> str:
        v = (verdict or "").strip().lower()
        if any(token in v for token in ("strong evidence", "confirmed", "present")):
            return "strong_evidence"
        if any(
            token in v
            for token in (
                "no material evidence",
                "ruled out",
                "not present",
                "no evidence",
            )
        ):
            return "no_material_evidence"
        if "insufficient" in v:
            return "insufficient_data"
        if "in progress" in v:
            return "in_progress"
        return "uncovered"

    def _update_coverage_entry(
        self,
        scheme: str,
        *,
        status: Optional[str] = None,
        worker_id: Optional[str] = None,
        flagged_document_ids: Optional[List[str]] = None,
        confidence: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        if scheme not in self._coverage_ledger.entries:
            self._coverage_ledger.entries[scheme] = CoverageEntry(scheme=scheme)
        entry = self._coverage_ledger.entries[scheme]
        if status:
            entry.status = status
        if worker_id and worker_id not in entry.supporting_worker_ids:
            entry.supporting_worker_ids.append(worker_id)
        if flagged_document_ids:
            merged = list(
                dict.fromkeys(entry.flagged_document_ids + flagged_document_ids)
            )
            entry.flagged_document_ids = merged
        if confidence:
            entry.confidence = confidence
        if notes:
            entry.notes = notes
        self._coverage_ledger.updated_at = datetime.utcnow()

    def _update_coverage_from_scratchpad(
        self,
        scratchpad_text: str,
        tentative_items: Optional[List[SuspicionItem]] = None,
    ) -> None:
        text = (scratchpad_text or "").lower()
        flagged_by_scheme: Dict[str, List[str]] = {}
        for item in tentative_items or []:
            scheme = item.scheme_type.value if item.scheme_type else "unknown"
            if scheme in _CLOSED_WORLD_SCHEMES and item.document_id:
                flagged_by_scheme.setdefault(scheme, []).append(item.document_id)

        for scheme in _CLOSED_WORLD_SCHEMES:
            if scheme in flagged_by_scheme:
                self._update_coverage_entry(
                    scheme,
                    status="strong_evidence",
                    flagged_document_ids=flagged_by_scheme[scheme],
                )
                continue

            if any(
                marker in text
                for marker in (
                    f"{scheme}: ruled out",
                    f"{scheme} ruled out",
                    f"{scheme}: no evidence",
                    f"{scheme}: no material evidence",
                    f"{scheme} no material evidence",
                    f"{scheme}: not present",
                )
            ):
                self._update_coverage_entry(scheme, status="no_material_evidence")
            elif any(
                marker in text
                for marker in (
                    f"{scheme}: insufficient data",
                    f"{scheme} insufficient data",
                )
            ):
                self._update_coverage_entry(scheme, status="insufficient_data")

    def _autonomous_finish_validation_errors(
        self, tentative_items: List[SuspicionItem]
    ) -> List[str]:
        errors: List[str] = []
        self._update_coverage_from_scratchpad(get_scratchpad_text(), tentative_items)
        uncovered = [
            scheme
            for scheme, entry in self._coverage_ledger.entries.items()
            if entry.status == "uncovered"
        ]
        if uncovered:
            errors.append(
                "coverage ledger still lacks verdicts for: "
                + ", ".join(sorted(uncovered))
            )
        return errors

    def _hypothesis_first_finish_validation_errors(
        self,
        tentative_items: List[SuspicionItem],
        plan: InvestigationPlan,
    ) -> List[str]:
        """Extra finish gates for plan-execute-reflect (coverage)."""
        errors: List[str] = []

        planned_schemes = {p.scheme.value for p in plan.phases}
        for required in _CLOSED_WORLD_SCHEMES:
            if required not in planned_schemes:
                errors.append(
                    f"investigation plan missing required scheme '{required}'"
                )

        scratchpad_text = get_scratchpad_text().lower()
        flagged_schemes = {
            item.scheme_type.value
            for item in tentative_items
            if item.scheme_type and item.document_id
        }
        for required in _CLOSED_WORLD_SCHEMES:
            if required in planned_schemes and required not in flagged_schemes:
                ruled_out = any(
                    marker in scratchpad_text
                    for marker in (
                        f"{required}: ruled out",
                        f"{required} ruled out",
                        f"{required}: no material evidence",
                        f"{required} no material evidence",
                        f"no {required}",
                        f"{required}: not present",
                    )
                )
                if not ruled_out:
                    errors.append(
                        f"scheme '{required}' has no flagged document_ids and no "
                        "RULED_OUT / NO MATERIAL EVIDENCE verdict in scratchpad"
                    )

        return errors

    def _worker_status_snapshot(self, state: _WorkerRuntimeState) -> WorkerStatus:
        with state.lock:
            return WorkerStatus(
                worker_id=state.brief.worker_id,
                scheme_or_goal=state.brief.scheme_or_goal,
                state=state.status,
                candidate_schemes=list(state.brief.candidate_schemes),
                mailbox_depth=sum(0 if msg.consumed else 1 for msg in state.mailbox),
                steps_taken=state.steps_taken,
                sql_calls_used=state.sql_calls_used,
                code_interpreter_calls_used=state.code_interpreter_calls_used,
                flagged_document_ids=list(state.flagged_document_ids),
                latest_summary=state.latest_summary,
                run_dir=str(state.worker_dir),
                started_at=state.started_at,
                completed_at=state.completed_at,
                error=state.error,
            )

    # ------------------------------------------------------------------
    # Run-directory naming
    # ------------------------------------------------------------------

    def _emit_suspicion_notice(
        self,
        reported: List[SuspicionItem],
        *,
        source: str = "report_suspicion",
        terminal: bool = True,
    ) -> None:
        """Stream an immediate notice when new JE-level suspicions are recorded."""
        if not reported:
            return

        task_id = getattr(_THREAD_SCHEME, "name", None) or ""
        prefix = f"[{task_id}] " if task_id else ""
        entries: List[Dict[str, Any]] = []
        for item in reported:
            if not item.document_id:
                continue
            scheme_val = item.scheme_type.value if item.scheme_type else "unknown"
            entries.append(
                {
                    "document_id": item.document_id,
                    "scheme": scheme_val,
                    "rationale": truncate_text_to_tokens(
                        item.rationale or "",
                        self.config.text_limits.rationale_medium,
                        side=TruncationSide.TAIL,
                    ),
                    "entity_id": item.entity_id or None,
                }
            )
        if not entries:
            return

        total = len(self._live_detections)
        event = {
            "event": "suspicion_reported",
            "source": source,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step_number": self._step_num,
            "task_id": task_id or None,
            "items": entries,
            "total_detections": total,
        }

        line_json = json.dumps(event, default=str) + "\n"
        if self._stream_trace_file is not None:
            try:
                self._stream_trace_file.write(line_json)
                self._stream_trace_file.flush()
            except Exception as exc:
                log.debug("Suspicion stream write failed: %s", exc)

        parent_sink = getattr(self, "_parent_stream_sink", None)
        if parent_sink is not None:
            lock, parent_path = parent_sink
            try:
                with lock:
                    with open(parent_path, "a", encoding="utf-8") as parent_stream:
                        parent_stream.write(line_json)
                        parent_stream.flush()
            except Exception as exc:
                log.debug("Parent suspicion stream write failed: %s", exc)

        if terminal:
            for entry in entries:
                line = (
                    f"{prefix}>>> SUSPICION REPORTED: {entry['scheme']} "
                    f"document_id={entry['document_id']} "
                    f"(total_detections={total})"
                )
                print(line, file=sys.stderr, flush=True)
                log.info(line)

    def _emit_orchestrator_event(
        self,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        echo_terminal: bool = False,
    ) -> None:
        """
        Record orchestration decisions (dispatch, injection, queue state) in the
        live NDJSON stream and ``orchestrator/events.ndjson``.

        Only the parent run (not hypothesis-task workers) writes these lines.
        """
        if self._worker_run_dir is not None:
            return
        record: Dict[str, Any] = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step_number": self._step_num,
            "run_id": self._run_id[:8],
        }
        if payload:
            record.update(payload)
        line = json.dumps(record, default=str) + "\n"
        with self._stream_event_lock:
            if self._stream_trace_file is not None:
                try:
                    self._stream_trace_file.write(line)
                    self._stream_trace_file.flush()
                except Exception as exc:
                    log.debug("Orchestrator stream write failed: %s", exc)
            orch_dir = self._run_dir / "orchestrator"
            try:
                orch_dir.mkdir(parents=True, exist_ok=True)
                orch_file = orch_dir / "events.ndjson"
                with open(orch_file, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception as exc:
                log.debug("Orchestrator events.ndjson write failed: %s", exc)
        if echo_terminal:
            log.info("ORCHESTRATOR %s %s", event, payload or {})

    def _llm_error_counts_from_client(self) -> Tuple[int, int]:
        """Return (total_failures, unrecovered_failures) for this agent's LLM client."""
        return (
            int(getattr(self.client, "llm_errors_total", 0) or 0),
            int(getattr(self.client, "llm_errors_unrecovered", 0) or 0),
        )

    def _aggregate_llm_errors_with_children(self) -> Tuple[int, int]:
        """
        Sum LLM error counters from this agent's client plus child workers.

        Hypothesis workers each own a separate ``build_client()`` instance; only
        aggregating ``self.client`` under-counts when workers hit the endpoint.
        """
        total, unrecovered = self._llm_error_counts_from_client()
        if self._worker_run_dir is not None:
            return total, unrecovered
        for child_root in (self._run_dir / "subagents", self._run_dir / "tasks"):
            for child_bs in self._load_child_budget_summaries(child_root).values():
                total += int(child_bs.get("llm_errors_total", 0) or 0)
                unrecovered += int(child_bs.get("llm_errors_unrecovered", 0) or 0)
        return total, unrecovered

    @staticmethod
    def _load_child_budget_summaries(
        child_root: pathlib.Path,
    ) -> Dict[str, Dict[str, Any]]:
        """Load budget_summary.json from each subdirectory under *child_root*."""
        budgets: Dict[str, Dict[str, Any]] = {}
        if not child_root.is_dir():
            return budgets
        for child_dir in sorted(child_root.iterdir()):
            if not child_dir.is_dir():
                continue
            bs_path = child_dir / "budget_summary.json"
            if not bs_path.is_file():
                continue
            try:
                bs = json.loads(bs_path.read_text(encoding="utf-8"))
                key = (
                    bs.get("task_id")
                    or bs.get("worker_id")
                    or bs.get("scheme")
                    or child_dir.name
                )
                budgets[str(key)] = bs
            except Exception:
                continue
        return budgets

    @staticmethod
    def _rollup_child_budget_into_summary(
        budget_summary: Dict[str, Any],
        child_budgets: Dict[str, Dict[str, Any]],
        prefix: str,
    ) -> None:
        """Attach per-child budgets and ``{prefix}_*`` aggregate token fields."""
        if not child_budgets:
            return
        budget_summary[prefix] = child_budgets
        budget_summary[f"{prefix}_prompt_tokens"] = sum(
            int(v.get("budget_prompt_tokens", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_reasoning_tokens"] = sum(
            int(v.get("reasoning_tokens", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_completion_tokens"] = sum(
            int(v.get("completion_tokens", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_budget_counted_tokens"] = sum(
            int(v.get("budget_counted_tokens", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_total_billed_tokens"] = sum(
            int(v.get("total_billed_tokens", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_steps"] = sum(
            int(v.get("steps", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_sql_calls"] = sum(
            int(
                v.get(
                    "sql_calls_used",
                    v.get("effective_sql_calls_used", v.get("sql_calls", 0)),
                )
                or 0
            )
            for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_code_interpreter_calls"] = sum(
            int(
                v.get(
                    "code_interpreter_calls_used",
                    v.get("code_interpreter_calls", 0),
                )
                or 0
            )
            for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_effective_sql_calls"] = sum(
            int(v.get("effective_sql_calls_used", 0) or 0)
            for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_tool_context_tokens_approx"] = sum(
            int(v.get("tool_context_tokens_approx", 0) or 0)
            for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_llm_errors_total"] = sum(
            int(v.get("llm_errors_total", 0) or 0) for v in child_budgets.values()
        )
        budget_summary[f"{prefix}_llm_errors_unrecovered"] = sum(
            int(v.get("llm_errors_unrecovered", 0) or 0) for v in child_budgets.values()
        )

    @staticmethod
    def _apply_grand_budget_totals(
        budget_summary: Dict[str, Any],
        child_prefixes: List[str],
    ) -> None:
        """Set ``grand_total_*`` = orchestrator + all rolled-up child prefixes."""
        if not child_prefixes:
            return
        orch_prompt = int(budget_summary.get("budget_prompt_tokens", 0) or 0)
        orch_reasoning = int(budget_summary.get("reasoning_tokens", 0) or 0)
        orch_completion = int(budget_summary.get("completion_tokens", 0) or 0)
        orch_billed = int(budget_summary.get("total_billed_tokens", 0) or 0)
        orch_steps = int(budget_summary.get("steps", 0) or 0)
        orch_tool = int(budget_summary.get("tool_context_tokens_approx", 0) or 0)
        orch_sql = int(budget_summary.get("sql_calls_total", 0) or 0)
        orch_ci = int(budget_summary.get("code_interpreter_calls", 0) or 0)
        orch_eff_sql = int(budget_summary.get("effective_sql_calls", 0) or 0)
        orch_llm_err = int(budget_summary.get("orchestrator_llm_errors_total", 0) or 0)
        orch_llm_unrec = int(
            budget_summary.get("orchestrator_llm_errors_unrecovered", 0) or 0
        )

        child_prompt = sum(
            int(budget_summary.get(f"{p}_prompt_tokens", 0) or 0)
            for p in child_prefixes
        )
        child_reasoning = sum(
            int(budget_summary.get(f"{p}_reasoning_tokens", 0) or 0)
            for p in child_prefixes
        )
        child_completion = sum(
            int(budget_summary.get(f"{p}_completion_tokens", 0) or 0)
            for p in child_prefixes
        )
        child_billed = sum(
            int(budget_summary.get(f"{p}_total_billed_tokens", 0) or 0)
            for p in child_prefixes
        )
        child_steps = sum(
            int(budget_summary.get(f"{p}_steps", 0) or 0) for p in child_prefixes
        )
        child_tool = sum(
            int(budget_summary.get(f"{p}_tool_context_tokens_approx", 0) or 0)
            for p in child_prefixes
        )
        child_sql = sum(
            int(budget_summary.get(f"{p}_sql_calls", 0) or 0) for p in child_prefixes
        )
        child_ci = sum(
            int(budget_summary.get(f"{p}_code_interpreter_calls", 0) or 0)
            for p in child_prefixes
        )
        child_eff_sql = sum(
            int(budget_summary.get(f"{p}_effective_sql_calls", 0) or 0)
            for p in child_prefixes
        )
        child_llm_err = sum(
            int(budget_summary.get(f"{p}_llm_errors_total", 0) or 0)
            for p in child_prefixes
        )
        child_llm_unrec = sum(
            int(budget_summary.get(f"{p}_llm_errors_unrecovered", 0) or 0)
            for p in child_prefixes
        )

        budget_summary["orchestrator_prompt_tokens"] = orch_prompt
        budget_summary["orchestrator_reasoning_tokens"] = orch_reasoning
        budget_summary["grand_total_prompt_tokens"] = orch_prompt + child_prompt
        budget_summary["grand_total_reasoning_tokens"] = (
            orch_reasoning + child_reasoning
        )
        budget_summary["grand_total_budget_counted_tokens"] = (
            budget_summary["grand_total_prompt_tokens"]
            + budget_summary["grand_total_reasoning_tokens"]
        )
        budget_summary["grand_total_completion_tokens"] = (
            orch_completion + child_completion
        )
        budget_summary["grand_total_billed_tokens"] = orch_billed + child_billed
        budget_summary["grand_total_steps"] = orch_steps + child_steps
        budget_summary["grand_total_tool_context_tokens_approx"] = (
            orch_tool + child_tool
        )
        budget_summary["grand_total_effective_sql_calls"] = orch_eff_sql + child_eff_sql
        budget_summary["grand_total_sql_calls"] = orch_sql + child_sql
        budget_summary["grand_total_code_interpreter_calls"] = orch_ci + child_ci
        budget_summary["grand_total_llm_errors"] = orch_llm_err + child_llm_err
        budget_summary["grand_total_llm_errors_unrecovered"] = (
            orch_llm_unrec + child_llm_unrec
        )
        budget_summary["llm_errors_total"] = budget_summary["grand_total_llm_errors"]
        budget_summary["llm_errors_unrecovered"] = budget_summary[
            "grand_total_llm_errors_unrecovered"
        ]

        _gmax = int(
            budget_summary.get(
                "configured_global_max_tokens",
                budget_summary.get("configured_max_tokens", 0),
            )
            or 0
        )
        if _gmax > 0:
            budget_summary["grand_budget_fraction_used"] = round(
                budget_summary["grand_total_budget_counted_tokens"] / _gmax,
                4,
            )

    def _harvest_document_ids_from_tool_result(self, result: str) -> None:
        """Track je_header UUIDs seen in hypothesis-worker SQL/export output."""
        if self._worker_run_dir is None or not result:
            return
        for match in _DOCUMENT_ID_UUID_RE.finditer(result):
            self._worker_sql_document_ids.add(match.group(0))

    def _promote_flagged_document_ids_to_detections(
        self,
        flagged_ids: List[str],
        scheme: str,
        status: str = "inconclusive",
    ) -> int:
        """
        Backfill detections.json when the model listed UUIDs in the summary but skipped
        report_suspicion during the loop. IDs are validated against je_header.

        Only ``confirmed`` hypotheses are promoted — inconclusive summaries do not become
        benchmark flags (use ``report_suspicion`` during the task for evidence-backed JEs).
        """
        if not flagged_ids:
            return 0
        status_l = (status or "").strip().lower()
        if status_l != "confirmed":
            return 0
        try:
            scheme_type = SchemeType(scheme)
        except ValueError:
            scheme_type = SchemeType.UNKNOWN
        items = [
            SuspicionItem(
                document_id=str(doc_id).strip(),
                scheme_type=scheme_type,
                confidence=_EVAL_FLAG_CONFIDENCE,
            )
            for doc_id in flagged_ids
            if doc_id
        ]
        valid_items, invalid_ids = self._validate_suspicion_ids(items)
        if invalid_ids:
            log.warning(
                "Summary flagged_document_ids: dropped %d invalid UUID(s) (sample %s)",
                len(invalid_ids),
                invalid_ids[:3],
            )
        if valid_items:
            self._flush_live_detections(valid_items)
        return len(valid_items)

    def _nudge_hypothesis_worker_report_suspicion(
        self,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
        brief: HypothesisTaskBrief,
        phase_name: str,
    ) -> None:
        """Extra steps urging report_suspicion when SQL ran but detections are still empty."""
        if self._live_detections:
            return
        sql_calls = self.budget.phase_sql_calls(phase_name)
        if sql_calls < 2 and not self._worker_sql_document_ids:
            return
        hint = ""
        if self._worker_sql_document_ids:
            sample = ", ".join(sorted(self._worker_sql_document_ids)[:5])
            hint = f" Sample document_ids from your SQL output: {sample}."
        messages.append(
            _user_msg(
                f"**Checkpoint before task close**\n\n"
                f"You have not called `report_suspicion` yet.{hint}\n\n"
                f"- If investigation is **incomplete** (no benign-rival test, or only a single "
                f"null screen), continue with sql/export analysis — do not close early.\n"
                f"- If **CONFIRMED** or you have exemplar suspicious JEs, call `report_suspicion` "
                f"now (`suspicions` batch; scheme_type=`{brief.scheme}`).\n"
                f"- If **RULED OUT** with evidence-backed negative tests, reply in one sentence "
                f"(no tools).\n"
            )
        )
        for _ in range(3):
            if self._live_detections:
                break
            _, tool_calls, _ = self._run_one_step(
                report,
                messages,
                allow_finish=False,
                phase_name=phase_name,
            )
            if not tool_calls:
                break

    def _inject_hypothesis_task_budget_pacing(
        self,
        messages: List[Dict[str, Any]],
        phase_name: str,
        used: int,
        cap: int,
    ) -> None:
        """One-shot user nudges as the task worker approaches its input cap."""
        if not self._worker_run_dir or cap <= 0 or not phase_name:
            return
        fr = used / float(cap)
        warn = float(self.config.task_budget_warn_fraction)
        deadline = float(self.config.task_budget_report_deadline_fraction)
        stop = float(self.config.task_budget_stop_fraction)

        def _emit(msg: str, key: str, threshold: float) -> None:
            if fr + 1e-9 < threshold or key in self._task_budget_pacing_sent:
                return
            self._task_budget_pacing_sent.add(key)
            if messages and messages[-1].get("role") == "tool":
                return
            messages.append(_user_msg(msg))

        _emit(
            f"**Task budget pacing ({fr * 100.0:.1f}% of {cap:,} input tokens used).** "
            f"If SQL already supports a **confirmed** fraud signal for this hypothesis, call "
            f"`report_suspicion` on specific `je_header.document_id` UUIDs before spending more tokens.",
            "task_budget_warn",
            warn,
        )
        _emit(
            f"**Task budget deadline (~{deadline * 100.0:.0f}% of cap).** Finalize now: if CONFIRMED, "
            f"call `report_suspicion` immediately; if RULED OUT, state it briefly — **avoid new large "
            f"exports and code_interpreter** unless essential.",
            "task_budget_deadline",
            deadline,
        )
        _emit(
            f"**Task budget critical (~{stop * 100.0:.0f}% of cap).** Last tool round: either "
            f"`report_suspicion` with SQL-backed UUIDs or a text-only verdict — no broad new investigation.",
            "task_budget_critical",
            stop,
        )

    def _inject_orientation_budget_band_nudge(
        self,
        messages: List[Dict[str, Any]],
        used: int,
        cap: int,
    ) -> None:
        """Optional user-role nudges so orientation is not ended while budget usage is still low."""
        if not self.config.orientation_budget_band_nudges or cap <= 0:
            return
        if messages and messages[-1].get("role") == "tool":
            return
        pct = 100.0 * float(used) / float(cap)
        band = min(9, int(pct // 10.0))
        key = f"orient_band_{band}"
        if key in self._orientation_budget_pacing_sent:
            return
        self._orientation_budget_pacing_sent.add(key)
        enc_until = (
            float(self.config.orientation_budget_encourage_deep_until_fraction) * 100.0
        )
        min_c = float(self.config.orientation_budget_min_fraction_for_complete) * 100.0
        rem = max(0, cap - used)
        if pct + 1e-6 < enc_until:
            messages.append(
                _user_msg(
                    f"**Orientation budget check:** you have used **{pct:.1f}%** of this phase's cap "
                    f"({used:,} / {cap:,} input tokens; **{rem:,} remaining**). You are **encouraged "
                    f"to continue deeply** — more SQL + dense `orientation_report` sections — until "
                    f"roughly **{enc_until:.0f}%** usage. Avoid `complete_orientation` while still this "
                    f"far under the depth target. Write investigative prose (field notes), not table-only sections."
                )
            )
        else:
            messages.append(
                _user_msg(
                    f"**Orientation budget check:** **{pct:.1f}%** used ({used:,} / {cap:,}; "
                    f"{rem:,} remaining). Finish mandatory sections (Vendor/AP, Revenue/COA, O2C×vendor, "
                    f"Planning leads, rich narrative per section); no TBD; `complete_orientation` requires at least "
                    f"~**{min_c:.0f}%** cap usage."
                )
            )

    @staticmethod
    def _is_orientation_current_step_user_msg(msg: Dict[str, Any]) -> bool:
        return msg.get("role") == "user" and "[Current Step]" in (
            _msg_text_content(msg) or ""
        )

    def _orientation_messages_without_step_headers(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep prior assistant/tool bundles; drop ephemeral [Current Step] user headers."""
        retained: List[Dict[str, Any]] = []
        for bundle in _split_message_bundles(messages):
            if bundle and self._is_orientation_current_step_user_msg(bundle[0]):
                continue
            retained.extend(bundle)
        return retained

    def _compact_orientation_message_history(
        self,
        messages: List[Dict[str, Any]],
        *,
        used_sql_or_plot: bool,
        wrote_orientation_report: bool,
    ) -> None:
        """
        Reset the [Current Step] header but keep prior turns for the recent slot.

        Cumulative findings remain on disk in orientation_report.md; recent chat holds
        the last assistant/tool bundles (token-capped at the next LLM call).
        """
        ob = getattr(self, "_orientation_budget_ctx", None)
        cap = int(ob["cap"]) if isinstance(ob, dict) and ob.get("cap") else 0
        used = self.budget.phase_tokens_used("orientation")
        from researchpkg.forensic_llm.prompts.orientation import (
            ORIENTATION_CURRENT_STEP_FRESH,
        )

        extra: List[str] = []
        if cap > 0:
            extra.append(
                f"- **Budget**: **{orientation_budget_pct(used, cap):.1f}%** of orientation cap "
                f"({used:,} / {cap:,} input tokens; {max(0, cap - used):,} remaining)."
            )
        if used_sql_or_plot and not wrote_orientation_report:
            extra.append(
                "- **Reminder**: Last step ran SQL without `orientation_report(mode=append)`. "
                "Append a short `##` section with measured findings before more queries."
            )
        body = ORIENTATION_CURRENT_STEP_FRESH
        if extra:
            body += "\n" + "\n".join(extra)
        retained = self._orientation_messages_without_step_headers(messages)
        messages.clear()
        messages.extend(retained)
        messages.append(_user_msg(body))

    def _trim_orientation_to_ephemeral_workspace(
        self, messages: List[Dict[str, Any]]
    ) -> None:
        """
        Keep prior turns (recent slot) plus the latest assistant+tool bundle.

        Findings must still be written to orientation_report.md to persist across steps.
        """
        last_asst = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_asst = i
                break
        bundle = list(messages[last_asst:]) if last_asst >= 0 else list(messages[-4:])
        prior = messages[:last_asst] if last_asst >= 0 else []
        retained = self._orientation_messages_without_step_headers(prior)
        from researchpkg.forensic_llm.prompts.orientation import (
            ORIENTATION_CURRENT_STEP_EPHEMERAL_SQL,
        )

        extra: List[str] = []
        ob = getattr(self, "_orientation_budget_ctx", None)
        cap = int(ob["cap"]) if isinstance(ob, dict) and ob.get("cap") else 0
        used = self.budget.phase_tokens_used("orientation")
        if cap > 0:
            extra.append(
                f"- **Budget**: **{orientation_budget_pct(used, cap):.1f}%** of orientation cap "
                f"({used:,} / {cap:,} input tokens; {max(0, cap - used):,} remaining)."
            )
        extra.append(
            "- **Reminder**: Append `orientation_report(mode=append)` with a short `##` section "
            "(measured facts from the SQL below) before running more queries."
        )
        body = ORIENTATION_CURRENT_STEP_EPHEMERAL_SQL
        if extra:
            body += "\n" + "\n".join(extra)
        messages.clear()
        messages.extend(retained)
        messages.append(_user_msg(body))
        messages.extend(bundle)

    def _sync_orientation_report_live_after_step(
        self,
        step: AgentStep,
        *,
        wrote_orientation_report: bool,
    ) -> None:
        """
        Flush/reload ``orientation/orientation_report.md`` at step end when the model wrote
        via ``orientation_report`` (or prior appends); keeps the report slot in sync with disk.
        """
        from researchpkg.forensic_llm.artefacts import (
            load_orientation_report_text,
            sync_orientation_report_live,
        )

        stats = sync_orientation_report_live(self._run_dir)
        report_text = load_orientation_report_text(self._run_dir)
        self._scratchpad_slot = report_text.strip()
        self._orientation_report_dirty = False

        log.info(
            "Step %d | orientation_report.md live: %d sections, %d chars%s",
            step.step_number,
            stats["sections"],
            stats["chars"],
            " (appended this step)" if wrote_orientation_report else "",
        )

    def _flush_live_detections(self, items: List[SuspicionItem]) -> None:
        """
        Merge JE-level items into live detections and write detections.json.
        Called on ``report_suspicion`` and on confirmed summary backfill.
        Dedupes by document_id (scheme from the reporting call).
        """
        if not items:
            return

        by_doc: Dict[str, Dict[str, Any]] = {}
        for d in self._live_detections:
            doc_id = d.get("document_id") or ""
            if doc_id:
                by_doc[str(doc_id).strip().lower()] = d

        newly_reported: List[SuspicionItem] = []
        for s in items:
            if not s.document_id:
                continue
            key = str(s.document_id).strip().lower()
            scheme_val = s.scheme_type.value if s.scheme_type else "unknown"
            prev = by_doc.get(key)
            if prev is None:
                newly_reported.append(s)
                by_doc[key] = _detection_record(s.document_id, scheme_val)
        self._live_detections = list(by_doc.values())
        path = self._run_dir / "detections.json"
        payload = {
            "detections": self._live_detections,
            "count": len(self._live_detections),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        log.debug(
            "Live detections flushed: %d JE(s) -> %s", len(self._live_detections), path
        )
        if newly_reported:
            self._emit_suspicion_notice(newly_reported)

    def _validate_suspicion_ids(
        self, items: List[SuspicionItem]
    ) -> Tuple[List[SuspicionItem], List[str]]:
        """
        Query je_header to verify that each document_id in *items* actually
        exists in the database.  Returns (valid_items, invalid_ids).

        This catches the failure mode where the model submits:
          - Aggregated / constructed UUIDs (not from je_header)
          - Hallucinated IDs from memory
          - IDs from intermediate GROUP BY results rather than raw rows

        Fails open: if the DB query itself errors, all items are returned as
        valid so the run is not blocked by a transient connection issue.
        """
        if not items:
            return items, []

        candidate_ids = [s.document_id for s in items if s.document_id]
        if not candidate_ids:
            return items, []

        try:
            from .tools.db import _get_cursor

            with _get_cursor() as cur:
                cur.execute(
                    "SELECT document_id::text FROM je_header "
                    "WHERE document_id::text = ANY(%s)",
                    (candidate_ids,),
                )
                valid_set = {
                    row[0].lower().strip().replace("-", "") for row in cur.fetchall()
                }

            def _norm(doc_id: str) -> str:
                return doc_id.lower().strip().replace("-", "")

            valid_items = [
                s for s in items if s.document_id and _norm(s.document_id) in valid_set
            ]
            invalid_ids = [
                s.document_id
                for s in items
                if s.document_id and _norm(s.document_id) not in valid_set
            ]
            if invalid_ids:
                log.warning(
                    "%d/%d submitted document_ids not found in je_header: %s",
                    len(invalid_ids),
                    len(items),
                    invalid_ids[:10],
                )
            return valid_items, invalid_ids

        except Exception as exc:
            log.warning(
                "Could not validate document_ids against je_header (%s); "
                "accepting all items (fail-open).",
                exc,
            )
            return items, []

    def _make_run_dir(self) -> pathlib.Path:
        """
        Create and return a timestamped, human-readable run directory.

        Naming convention:
          {base_output_dir}/{YYYY-MM-DDTHH-MM-SS}_{task}_{model_short}_{run_id[:8]}/

        Example:
          ./forensic_output/2026-03-05T14-32-11_full_gpt_oss_20b_a3f8b2c1/
        """
        ts = self._started_at.strftime("%Y-%m-%dT%H-%M-%S")
        model_short = re.sub(r"[^\w]", "_", self.config.llm.model)[:24].strip("_")
        dir_name = f"{ts}_{self.config.task}_{model_short}_{self._run_id[:8]}"
        run_dir = pathlib.Path(self.config.output_dir) / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _worker_enabled_tool_names(self) -> List[str]:
        tool_names: List[str] = []
        blocked = set(AUTONOMOUS_ORCHESTRATION_TOOL_NAMES)
        for tool in self._tools:
            name = tool.get("function", {}).get("name")
            if name and name not in blocked:
                tool_names.append(name)
        return tool_names

    def _normalise_candidate_schemes(self, values: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(values, list):
            return out
        for value in values:
            scheme = str(value).strip().lower()
            if scheme in _CLOSED_WORLD_SCHEMES and scheme not in out:
                out.append(scheme)
        return out

    def _merge_detections_from_worker_dir(self, worker_dir: pathlib.Path) -> int:
        det_path = worker_dir / "detections.json"
        if not det_path.exists():
            return 0
        try:
            payload = json.loads(det_path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for det in self._live_detections:
            doc_id = str(det.get("document_id", "")).strip().lower()
            scheme_id = str(det.get("scheme_id", "unknown")).strip().lower()
            if doc_id:
                by_key[(doc_id, scheme_id)] = det

        merged = 0
        newly_reported: List[SuspicionItem] = []
        for det in payload.get("detections", []):
            doc_id = str(det.get("document_id", "")).strip()
            if not doc_id:
                continue
            scheme_id = str(det.get("scheme_id", "unknown")).strip().lower()
            key = (doc_id.lower(), scheme_id)
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = _detection_record(doc_id, scheme_id)
                merged += 1
                try:
                    scheme_type = SchemeType(scheme_id)
                except ValueError:
                    scheme_type = SchemeType.UNKNOWN
                newly_reported.append(
                    SuspicionItem(
                        document_id=doc_id,
                        scheme_type=scheme_type,
                        confidence=_EVAL_FLAG_CONFIDENCE,
                    )
                )

        self._live_detections = list(by_key.values())
        if newly_reported:
            self._emit_suspicion_notice(
                newly_reported,
                source=f"worker_merge:{worker_dir.name}",
                terminal=False,
            )
        return merged

    def _load_worker_summary(self, worker_dir: pathlib.Path) -> Optional[WorkerSummary]:
        summary_path = worker_dir / "worker_summary.json"
        if not summary_path.exists():
            return None
        try:
            return WorkerSummary.model_validate_json(
                summary_path.read_text(encoding="utf-8")
            )
        except Exception:
            return None

    def _apply_worker_summary_to_coverage(self, summary: WorkerSummary) -> None:
        if summary.flagged_document_ids:
            for scheme in summary.candidate_schemes:
                self._update_coverage_entry(
                    scheme,
                    worker_id=summary.worker_id,
                    status="strong_evidence",
                    flagged_document_ids=summary.flagged_document_ids,
                    confidence=summary.confidence,
                    notes=truncate_text_to_tokens(
                        summary.key_findings or "",
                        self.config.text_limits.worker_notes,
                        side=TruncationSide.TAIL,
                    ),
                )

        for scheme, verdict in summary.recommended_scheme_verdicts.items():
            if scheme not in _CLOSED_WORLD_SCHEMES:
                continue
            self._update_coverage_entry(
                scheme,
                worker_id=summary.worker_id,
                status=self._coverage_status_from_verdict(verdict),
                flagged_document_ids=summary.flagged_document_ids,
                confidence=summary.confidence,
                notes=verdict,
            )

    def _spawn_worker_tool(self, args: Dict[str, Any]) -> str:
        return (
            "[TOOL ERROR] spawn_worker is not available in this build; the v2 "
            "hypothesis loop uses the planner dispatch queue and per-hypothesis workers."
        )

    def _list_workers_tool(self) -> str:
        workers = []
        with self._worker_registry_lock:
            for worker_id in sorted(self._worker_registry):
                workers.append(
                    self._worker_status_snapshot(
                        self._worker_registry[worker_id]
                    ).model_dump(mode="json")
                )
        payload = {
            "workers": workers,
            "coverage_ledger": self._coverage_ledger.model_dump(mode="json"),
        }
        return json.dumps(payload, indent=2, default=str)

    def _message_worker_tool(self, args: Dict[str, Any]) -> str:
        worker_id = str(args.get("worker_id", "")).strip()
        instruction = str(args.get("instruction", "")).strip()
        if not worker_id or not instruction:
            return "[TOOL ERROR] message_worker requires worker_id and instruction."
        state = self._worker_registry.get(worker_id)
        if state is None:
            return f"[TOOL ERROR] Unknown worker_id: {worker_id}"
        with state.lock:
            if state.status not in {"running", "pending"}:
                return (
                    f"[TOOL ERROR] Worker {worker_id} is not running "
                    f"(current status: {state.status})."
                )
            state.mailbox.append(WorkerMessage(instruction=instruction))
            mailbox_depth = sum(0 if msg.consumed else 1 for msg in state.mailbox)
        return json.dumps(
            {
                "worker_id": worker_id,
                "status": state.status,
                "mailbox_depth": mailbox_depth,
                "queued_instruction": instruction,
            },
            indent=2,
        )

    def _collect_worker_summary_tool(self, args: Dict[str, Any]) -> str:
        worker_id = str(args.get("worker_id", "")).strip()
        if not worker_id:
            return "[TOOL ERROR] collect_worker_summary requires worker_id."
        state = self._worker_registry.get(worker_id)
        if state is None:
            return f"[TOOL ERROR] Unknown worker_id: {worker_id}"

        summary: Optional[WorkerSummary]
        with state.lock:
            summary = state.summary
        if summary is None:
            summary = self._load_worker_summary(state.worker_dir)
            if summary is not None:
                with state.lock:
                    state.summary = summary
                    state.latest_summary = summary.key_findings
                    state.flagged_document_ids = list(summary.flagged_document_ids)

        merged_detections = self._merge_detections_from_worker_dir(state.worker_dir)
        if summary is not None:
            self._apply_worker_summary_to_coverage(summary)

        payload = {
            "worker": self._worker_status_snapshot(state).model_dump(mode="json"),
            "summary": summary.model_dump(mode="json") if summary is not None else None,
            "merged_detections": merged_detections,
            "coverage_ledger": self._coverage_ledger.model_dump(mode="json"),
        }
        return json.dumps(payload, indent=2, default=str)

    def _collect_all_worker_summaries(self) -> None:
        with self._worker_registry_lock:
            worker_ids = list(self._worker_registry.keys())
        for worker_id in worker_ids:
            self._collect_worker_summary_tool({"worker_id": worker_id})

    def _normalize_autonomous_worker_outputs(self, report: ForensicReport) -> None:
        if not self._is_parent_autonomous_orchestrated():
            return

        self._collect_all_worker_summaries()

        confidence_order = {"low": 0, "medium": 1, "high": 2}
        scheme_meta: Dict[str, Dict[str, Any]] = {
            scheme: {
                "flagged_entities": [],
                "open_questions": [],
                "key_findings": [],
                "tested": 0,
                "amount": 0.0,
                "confidence": "",
            }
            for scheme in _CLOSED_WORLD_SCHEMES
        }

        with self._worker_registry_lock:
            items = list(self._worker_registry.items())

        report.worker_summaries = {}
        for worker_id, state in items:
            self._merge_detections_from_worker_dir(state.worker_dir)
            summary = state.summary or self._load_worker_summary(state.worker_dir)
            if summary is None:
                continue
            report.worker_summaries[worker_id] = summary
            self._apply_worker_summary_to_coverage(summary)
            relevant_schemes = summary.candidate_schemes or [
                scheme
                for scheme in summary.recommended_scheme_verdicts
                if scheme in _CLOSED_WORLD_SCHEMES
            ]
            for scheme in relevant_schemes:
                meta = scheme_meta.setdefault(
                    scheme,
                    {
                        "flagged_entities": [],
                        "open_questions": [],
                        "key_findings": [],
                        "tested": 0,
                        "amount": 0.0,
                        "confidence": "",
                    },
                )
                meta["flagged_entities"] = list(
                    dict.fromkeys(meta["flagged_entities"] + summary.flagged_entities)
                )
                meta["open_questions"] = list(
                    dict.fromkeys(meta["open_questions"] + summary.open_questions)
                )
                if summary.key_findings:
                    meta["key_findings"].append(summary.key_findings.strip())
                meta["tested"] += max(
                    1,
                    len(summary.evidence_checks_run)
                    or len(summary.recommended_scheme_verdicts),
                )
                meta["amount"] += float(summary.total_flagged_amount or 0.0)
                if confidence_order.get(summary.confidence, -1) > confidence_order.get(
                    meta["confidence"], -1
                ):
                    meta["confidence"] = summary.confidence

        for scheme in _CLOSED_WORLD_SCHEMES:
            entry = self._coverage_ledger.entries.get(
                scheme, CoverageEntry(scheme=scheme)
            )
            meta = scheme_meta.get(
                scheme,
                {
                    "flagged_entities": [],
                    "open_questions": [],
                    "key_findings": [],
                    "tested": 0,
                    "amount": 0.0,
                    "confidence": "",
                },
            )
            status = entry.status
            hypotheses_confirmed = 1 if status == "strong_evidence" else 0
            hypotheses_ruled_out = 1 if status == "no_material_evidence" else 0
            hypotheses_open = (
                1 if status in {"insufficient_data", "in_progress", "uncovered"} else 0
            )
            findings = " ".join(
                dict.fromkeys([entry.notes] + meta["key_findings"])
            ).strip()
            self._phase_summaries[scheme] = PhaseSummary(
                scheme=scheme,
                hypotheses_tested=max(meta["tested"], len(entry.supporting_worker_ids)),
                hypotheses_confirmed=hypotheses_confirmed,
                hypotheses_ruled_out=hypotheses_ruled_out,
                hypotheses_open=hypotheses_open,
                flagged_entities=list(meta["flagged_entities"]),
                flagged_document_ids=list(entry.flagged_document_ids),
                total_flagged_amount=float(meta["amount"]),
                confidence=meta["confidence"] or entry.confidence or "low",
                sql_calls_used=0,
                open_questions=list(meta["open_questions"]),
                key_findings=findings,
            )

        report.coverage_ledger = self._coverage_ledger

    def _handle_worker_runtime_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[str]:
        if tool_name == "spawn_worker":
            return self._spawn_worker_tool(args)
        if tool_name == "list_workers":
            return self._list_workers_tool()
        if tool_name == "message_worker":
            return self._message_worker_tool(args)
        if tool_name == "collect_worker_summary":
            return self._collect_worker_summary_tool(args)
        return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> ForensicReport:
        """
        Execute the full investigation and return a ForensicReport.
        """
        reset_scratchpad()
        reset_graph()
        reset_e2b_sandbox()
        self._tool_call_counts = {}
        self._scratchpad_steps_dir.mkdir(parents=True, exist_ok=True)
        self._calls_steps_dir.mkdir(parents=True, exist_ok=True)
        init_tools(
            self.config.database,
            output_dir=str(self._plots_dir),
            grep_root=self.config.grep_root,
            graph_depth=self.config.graph_depth,
            scratchpad_live_path=str(self._run_dir / "scratchpad.md"),
        )

        report = ForensicReport(
            run_id=self._run_id,
            model=self.config.llm.model,
            task=self.config.task,
            strategy="hypothesis_orchestrated",
            started_at=datetime.utcnow(),
        )

        log.info(
            "Starting forensic investigation | run_id=%s | model=%s | task=%s "
            "| input_budget=%d M tokens",
            self._run_id[:8],
            self.config.llm.model,
            self.config.task,
            self._global_token_budget // 1_000_000,
        )

        if self.config.stream_trace:
            stream_path = self._run_dir / "audit_trace_stream.ndjson"
            self._stream_trace_file = open(stream_path, "a", encoding="utf-8")
            log.info(
                "Streaming trace to %s (view with: python -m forensic_llm.trace_viewer %s --follow)",
                stream_path,
                self._run_dir,
            )

        try:
            from researchpkg.forensic_llm.hypothesis_loop import (
                run_hypothesis_orchestrated_loop,
            )

            run_hypothesis_orchestrated_loop(self, report)

            # Build scheme reports from the flat suspicion_list
            if report.suspicion_list:
                report.scheme_reports = SchemeReport.from_suspicion_list(
                    report.suspicion_list
                )
            if not report.termination_reason:
                report.termination_reason = (
                    "budget_exhausted" if report.budget_exhausted else "completed"
                )
        except KeyboardInterrupt:
            log.warning("Investigation interrupted by user")
            report.termination_reason = "interrupted"
        except Exception as exc:
            log.exception("Unexpected error in agent loop: %s", exc)
            report.termination_reason = "error"
            report.error_message = str(exc)
            report.error_traceback = traceback.format_exc()
        finally:
            if self._stream_trace_file is not None:
                try:
                    self._stream_trace_file.close()
                except Exception:
                    pass
                self._stream_trace_file = None
            report.completed_at = datetime.utcnow()
            # Start with orchestrator-only values; overwritten below for parallel runs.
            report.total_tokens_input = self.budget._prompt_tokens
            report.total_tokens_output = self.budget._completion_tokens
            report.total_tokens_reasoning = self.budget._reasoning_tokens
            report.total_tool_tokens = self.budget._tool_tokens
            report.steps_taken = self.budget.steps
            (
                report.llm_errors_total,
                report.llm_errors_unrecovered,
            ) = self._aggregate_llm_errors_with_children()
            report.scratchpad = get_scratchpad_text()

            # Aggregate child-worker costs (scheme subagents + hypothesis tasks).
            if self._worker_run_dir is None:
                sub_prompt = sub_completion = sub_reasoning = sub_tool = sub_steps = 0
                for child_root in (
                    self._run_dir / "subagents",
                    self._run_dir / "tasks",
                ):
                    for child_bs in self._load_child_budget_summaries(
                        child_root
                    ).values():
                        sub_prompt += int(child_bs.get("budget_prompt_tokens", 0) or 0)
                        sub_completion += int(child_bs.get("completion_tokens", 0) or 0)
                        sub_reasoning += int(child_bs.get("reasoning_tokens", 0) or 0)
                        sub_tool += int(
                            child_bs.get("tool_context_tokens_approx", 0) or 0
                        )
                        sub_steps += int(child_bs.get("steps", 0) or 0)
                if sub_prompt > 0:
                    report.total_tokens_input += sub_prompt
                    report.total_tokens_output += sub_completion
                    report.total_tokens_reasoning += sub_reasoning
                    report.total_tool_tokens += sub_tool
                    report.steps_taken += sub_steps

            self._save_report(report)

        log.info(
            "Investigation complete | run_dir=%s | steps=%d | suspects=%d"
            " | prompt_tokens=%s | reasoning_tokens=%s | budget_counted=%s"
            " | total_billed=%s",
            self._run_dir,
            report.steps_taken,
            len(report.suspicion_list),
            f"{report.total_tokens_input:,}",
            f"{report.total_tokens_reasoning:,}",
            f"{report.budget_counted_tokens:,}",
            f"{report.total_tokens_input + report.total_tokens_output:,}",
        )
        return report

    # ------------------------------------------------------------------
    # Main agentic loop
    # ------------------------------------------------------------------

    def _run_hypothesis_task_worker(
        self,
        brief: HypothesisTaskBrief,
        parent_run_dir: Optional[pathlib.Path] = None,
    ) -> HypothesisResult:
        """
        Investigate a single hypothesis (core_spec v2 task worker).

        Writes canonical artefact to ``parent_run_dir/schemes/<scheme>/p<n>.json``.
        """
        from researchpkg.forensic_llm.artefacts import (
            save_hypothesis_result,
        )
        from researchpkg.forensic_llm.prompts.hypothesis_worker import (
            build_hypothesis_summary_prompt,
            build_hypothesis_worker_prompt,
            parse_hypothesis_summary_json,
        )
        from researchpkg.forensic_llm.prompts.scheme_phase import (
            build_scheme_phase_prompt,
        )

        artefact_root = parent_run_dir or self._run_dir
        _THREAD_SCHEME.name = brief.task_id

        reset_scratchpad()
        reset_graph()
        reset_e2b_sandbox()
        self._tool_call_counts = {}
        self._scratchpad_steps_dir.mkdir(parents=True, exist_ok=True)
        self._calls_steps_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        init_tools(
            self.config.database,
            output_dir=str(self._plots_dir),
            grep_root=self.config.grep_root,
            graph_depth=self.config.graph_depth,
            scratchpad_live_path=str(self._run_dir / "scratchpad.md"),
        )

        if self.config.stream_trace:
            stream_path = self._run_dir / "audit_trace_stream.ndjson"
            self._stream_trace_file = open(stream_path, "a", encoding="utf-8")

        report = ForensicReport(
            run_id=self._run_id,
            model=self.config.llm.model,
            task=self.config.task,
            strategy="hypothesis_task_worker",
            started_at=datetime.utcnow(),
        )

        hyp_line = f"[{brief.hypothesis_id}] {brief.hypothesis_text}"
        plan_ctx = brief.plan_rationale
        if brief.priority_signals:
            plan_ctx += "\nSignals: " + "; ".join(brief.priority_signals[:4])

        scheme_discipline = build_scheme_phase_prompt(
            brief.scheme,
            [hyp_line],
            hypothesis_task_mode=True,
            plan_context=plan_ctx,
        )
        messages: List[Dict[str, Any]] = [
            _user_msg(
                build_hypothesis_worker_prompt(
                    brief, text_limits=self.config.text_limits
                )
                + "\n\n---\n\n"
                + scheme_discipline
            ),
        ]

        phase_name = brief.task_id
        self.budget.start_phase(phase_name)
        self._task_budget_pacing_sent = set()
        self._hypothesis_task_cap = brief.budget_tokens
        finish_reason = "completed"
        stop_frac = self.config.task_budget_stop_fraction

        while not self.budget.should_stop():
            used = self.budget.phase_tokens_used(phase_name)
            if used >= brief.budget_tokens:
                finish_reason = "task_budget_exhausted"
                break
            if used >= int(brief.budget_tokens * stop_frac):
                finish_reason = "task_budget_stop_frac"
                break

            _, tool_calls, finished = self._run_one_step(
                report,
                messages,
                allow_finish=False,
                phase_name=phase_name,
            )
            if finished:
                break
            if not tool_calls:
                break

        used_final = self.budget.phase_tokens_used(phase_name)
        cap = brief.budget_tokens
        if used_final >= cap:
            finish_reason = "task_budget_exhausted"
        elif finish_reason == "completed" and self.budget.should_stop():
            finish_reason = "task_budget_stop_frac"
        elif finish_reason == "completed" and used_final >= int(cap * stop_frac):
            finish_reason = "task_budget_stop_frac"

        self._nudge_hypothesis_worker_report_suspicion(
            report, messages, brief, phase_name
        )

        summary_text = self._run_hypothesis_summary_llm(brief, messages)
        raw = parse_hypothesis_summary_json(summary_text or "", brief)
        raw["tokens_used"] = self.budget.phase_tokens_used(phase_name)
        raw["steps"] = self.budget.phase_steps(phase_name)
        raw["sql_calls"] = self.budget.phase_sql_calls(phase_name)
        raw["effective_sql_calls"] = self._phase_effective_sql_calls(phase_name)
        summ_fr = raw.get("finish_reason")
        if finish_reason != "completed":
            raw["finish_reason"] = finish_reason
        elif summ_fr:
            raw["finish_reason"] = summ_fr
        else:
            raw["finish_reason"] = "completed"
        raw["task_id"] = brief.task_id
        raw["benign_rivals_considered"] = raw.get("benign_rivals_considered") or list(
            brief.benign_rivals
        )

        status_l = str(raw.get("status", "inconclusive")).lower()
        summary_flagged = list(raw.get("flagged_document_ids") or [])
        before_n = len(self._live_detections)
        if summary_flagged:
            self._promote_flagged_document_ids_to_detections(
                summary_flagged, brief.scheme, status_l
            )
        if len(self._live_detections) > before_n:
            log.info(
                "Hypothesis %s: promoted %d flagged_document_id(s) to detections",
                brief.task_id,
                len(self._live_detections) - before_n,
            )
        raw["flagged_document_ids"] = [
            d.get("document_id") for d in self._live_detections if d.get("document_id")
        ]

        result = HypothesisResult.model_validate(raw)
        save_hypothesis_result(artefact_root, result)

        task_budget = self.budget.summary()
        task_budget["task_id"] = brief.task_id
        task_budget["scheme"] = brief.scheme
        task_budget["hypothesis_id"] = brief.hypothesis_id
        task_budget["sql_calls_used"] = self.budget.phase_sql_calls(phase_name)
        task_budget[
            "code_interpreter_calls_used"
        ] = self.budget.phase_code_interpreter_calls(phase_name)
        task_budget["effective_sql_calls_used"] = self._phase_effective_sql_calls(
            phase_name
        )
        task_budget["configured_global_max_tokens"] = self._global_token_budget
        task_budget["run_prompt_token_cap"] = self.budget.max_tokens
        (
            task_budget["llm_errors_total"],
            task_budget["llm_errors_unrecovered"],
        ) = self._llm_error_counts_from_client()
        (self._run_dir / "budget_summary.json").write_text(
            json.dumps(task_budget, indent=2, default=str),
            encoding="utf-8",
        )

        if self._stream_trace_file is not None:
            try:
                self._stream_trace_file.close()
            except Exception:
                pass
            self._stream_trace_file = None

        _THREAD_SCHEME.name = None
        return result

    def _run_hypothesis_summary_llm(
        self,
        brief: HypothesisTaskBrief,
        messages: List[Dict[str, Any]],
    ) -> str:
        from researchpkg.forensic_llm.prompts.hypothesis_worker import (
            build_hypothesis_summary_prompt,
        )

        system = build_system_prompt(
            task=self.config.task,
            scratchpad_text=get_scratchpad_text(),
            is_worker=True,
        )
        recent = messages[-6:] if len(messages) > 6 else list(messages)
        payload = (
            [_system_msg(system)]
            + recent
            + [
                _user_msg(
                    build_hypothesis_summary_prompt(
                        brief, text_limits=self.config.text_limits
                    )
                )
            ]
        )
        max_input = self._get_max_input_tokens()
        if self._inflated_token_estimate(payload) > max_input:
            payload = self._hard_trim_to_fit(payload, max_input)
        try:
            resp = self.client.chat(messages=payload, tools=None)
            self.budget.record_llm_call(
                resp.prompt_tokens,
                resp.completion_tokens,
                reasoning_tokens=resp.reasoning_tokens,
            )
            return resp.content or ""
        except Exception as exc:
            log.warning("Hypothesis summary LLM failed: %s", exc)
            return ""

    def _run_one_step(
        self,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
        allow_finish: bool = True,
        phase_name: str = "",
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AgentStep, List[ToolCallRequest], bool]:
        """
        Execute a single agent step: LLM call → tool execution → history update.
        Returns (step, tool_calls, finished).
        """
        self._step_num += 1
        self._current_phase_name = phase_name or ""
        ctx_extra: Dict[str, Any] = {}
        if phase_name == "orientation":
            ob = getattr(self, "_orientation_budget_ctx", None)
            if isinstance(ob, dict) and ob.get("cap"):
                orient_used = self.budget.phase_tokens_used("orientation")
                orient_cap = int(ob["cap"])
                ctx_extra.update(
                    orientation_tokens_used=orient_used,
                    orientation_tokens_cap=orient_cap,
                    orientation_planning_input_cap=int(
                        ob.get("planning_input_cap")
                        or self.config.text_limits.planning_orientation_prompt
                        or self.config.text_limits.orientation_summary_store
                    ),
                    orientation_encourage_deep_until_fraction=(
                        self.config.orientation_budget_encourage_deep_until_fraction
                    ),
                    orientation_min_fraction_for_complete=(
                        self.config.orientation_budget_min_fraction_for_complete
                    ),
                )
                self._inject_orientation_budget_band_nudge(
                    messages, orient_used, orient_cap
                )
        elif (
            self._worker_run_dir is not None
            and phase_name
            and phase_name not in ("orientation", "planning")
            and self._hypothesis_task_cap > 0
        ):
            cap = self._hypothesis_task_cap
            used = self.budget.phase_tokens_used(phase_name)
            ctx_extra["task_tokens_used"] = used
            ctx_extra["task_tokens_cap"] = cap
            ctx_extra[
                "task_budget_warn_fraction"
            ] = self.config.task_budget_warn_fraction
            ctx_extra[
                "task_budget_report_deadline_fraction"
            ] = self.config.task_budget_report_deadline_fraction
            ctx_extra[
                "task_budget_stop_fraction"
            ] = self.config.task_budget_stop_fraction
            self._inject_hypothesis_task_budget_pacing(messages, phase_name, used, cap)
        set_scratchpad_run_context(
            self._step_num,
            phase_name or "",
            self._effective_sql_call_total,
            **ctx_extra,
        )
        if phase_name != "orientation":
            self._persist_scratchpad_step()

        system_content = build_system_prompt(
            task=self.config.task,
            scratchpad_text=get_scratchpad_text(),
            is_worker=self._worker_run_dir is not None,
            orientation_phase=(phase_name == "orientation"),
        )
        # Native tool calling path.

        if self.budget.should_stop():
            report.budget_exhausted = True
            # Only inject a user-role instruction when the last message is not a
            # tool result; inserting user after tool violates the OpenAI message
            # protocol on strict backends.
            if not messages or messages[-1].get("role") != "tool":
                if phase_name == "orientation":
                    messages.append(
                        _user_msg(
                            "**GLOBAL BUDGET NEARLY EXHAUSTED.** "
                            "Call `complete_orientation` now only after "
                            "`orientation/orientation_report.md` is dense and complete."
                        )
                    )
                elif self._worker_run_dir is not None and phase_name not in (
                    "orientation",
                    "planning",
                    "",
                ):
                    messages.append(
                        _user_msg(
                            "**GLOBAL INPUT BUDGET STOP THRESHOLD REACHED.** "
                            "If you have SQL-backed suspicious `je_header.document_id` values, "
                            "call `report_suspicion` immediately. Otherwise give a one-sentence verdict "
                            "(no new SQL, exports, or code_interpreter)."
                        )
                    )
                else:
                    messages.append(
                        _user_msg(
                            "**BUDGET NEARLY EXHAUSTED.** "
                            "You MUST now call `finish_investigation` immediately."
                        )
                    )

        step = AgentStep(step_number=self._step_num, timestamp=datetime.utcnow())

        max_completion = (
            self.config.llm.max_tokens_planning
            if response_schema is not None
            else self.config.llm.max_tokens_per_step
        )
        try:
            resp = self._call_llm(
                messages,
                system_content=system_content,
                force_finish=report.budget_exhausted,
                response_schema=response_schema,
                max_completion_tokens=max_completion,
                phase_name=phase_name,
            )
        except Exception as exc:
            log.error(
                "LLM call failed at step %d (unrecovered): %s", self._step_num, exc
            )
            report.termination_reason = "llm_unrecovered"
            return step, [], False

        step.llm_tokens_input = resp.prompt_tokens
        step.llm_tokens_output = resp.completion_tokens
        step.llm_tokens_reasoning = resp.reasoning_tokens
        # For thinking models: resp.reasoning = chain-of-thought, resp.content = output.
        # For standard models: resp.reasoning is empty, resp.content is the full output.
        step.reasoning = resp.reasoning.strip() or None if resp.reasoning else None
        step.content = resp.content.strip() or None if resp.content else None
        # Legacy field used by _parse_plan and audit display: prefer content when present,
        # fall back to reasoning (covers non-thinking models where content holds everything).
        if not step.reasoning:
            step.reasoning = step.content

        # plan_execute_reflect relies exclusively on native tool calling.
        # Native tool calling path; no ReAct text-parsing fallback in this build.
        tool_calls = resp.tool_calls

        if resp.has_tool_calls or tool_calls:
            tc_list = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in (resp.tool_calls or tool_calls)
            ]
            messages.append(_assistant_msg(resp.content or "", tc_list))
        else:
            messages.append(_assistant_msg(resp.content or ""))

        tool_result_texts: List[str] = []
        finished = False
        used_sql_or_plot = False
        wrote_scratchpad = False
        wrote_orientation_report = False

        # Track repeated identical tool-call signatures across steps to break loops.
        # Signature is computed from tool name + canonicalised JSON args.
        tool_signature_parts: List[str] = []

        for tc in tool_calls:
            tc_rec = ToolCallRecord(
                tool=tc.name,
                args=tc.arguments,
                timestamp=datetime.utcnow(),
            )

            try:
                canon_args = json.dumps(
                    tc.arguments, sort_keys=True, ensure_ascii=False
                )
            except Exception:
                canon_args = str(tc.arguments)
            tool_signature_parts.append(f"{tc.name}:{canon_args}")

            # Per-tool usage counters (plan_execute_reflect loop)
            self._tool_call_counts[tc.name] = self._tool_call_counts.get(tc.name, 0) + 1
            if tc.name in ("sql", "code_interpreter", "write_csv"):
                self._record_investigation_tool(
                    tc.name, phase_name, arguments=tc.arguments
                )
                used_sql_or_plot = True

            if tc.name == "report_suspicion":
                report_items = _parse_report_suspicion_args(tc.arguments)
                if report_items:
                    self._flush_live_detections(report_items)
                if report_items:
                    base_msg = f"Recorded {len(report_items)} suspicion(s); detections.json updated."
                elif self._worker_run_dir and self._worker_sql_document_ids:
                    sample = ", ".join(sorted(self._worker_sql_document_ids)[:3])
                    base_msg = (
                        "No valid suspicion parsed. Use suspicions array with "
                        "objects {{document_id, scheme_type, rationale}} and "
                        f"je_header UUIDs (e.g. from your SQL: {sample})."
                    )
                else:
                    base_msg = "No valid suspicion (need document_id and scheme_type, or suspicions array)."
                tc_rec.result = base_msg
                # Write blackboard entity findings for parallel cross-scheme linking
                _scheme_label = (
                    phase_name or getattr(_THREAD_SCHEME, "name", None) or ""
                )
                for item in report_items:
                    if item.entity_id and _scheme_label:
                        self._shared_blackboard.write_entity(
                            scheme=_scheme_label.replace("scheme_", ""),
                            entity_id=item.entity_id,
                            entity_type=item.entity_type or "unknown",
                            confidence=_EVAL_FLAG_CONFIDENCE,
                            rationale=truncate_text_to_tokens(
                                item.rationale or "",
                                self.config.text_limits.rationale_medium,
                                side=TruncationSide.TAIL,
                            ),
                        )
                step.tool_calls.append(tc_rec)
                tool_result_texts.append(tc_rec.result)
                if resp.has_tool_calls:
                    messages.append(_tool_result_msg(tc.id, tc_rec.result))
                else:
                    messages.append(_observation_msg(tc_rec.result))
                continue

            if tc.name == "blackboard_write":
                board = self._parent_blackboard or self._shared_blackboard
                args = tc.arguments or {}
                entity_id = str(args.get("entity_id", "")).strip()
                if entity_id:
                    board.write_entity(
                        scheme=str(args.get("scheme", phase_name or "unknown")),
                        entity_id=entity_id,
                        entity_type=str(args.get("entity_type", "unknown")),
                        confidence=float(args.get("confidence", 0.5)),
                        rationale=truncate_text_to_tokens(
                            str(args.get("rationale", "")),
                            self.config.text_limits.rationale_medium,
                            side=TruncationSide.TAIL,
                        ),
                    )
                tc_rec.result = "Recorded entity finding on shared blackboard."
                step.tool_calls.append(tc_rec)
                tool_result_texts.append(tc_rec.result)
                _append_tool_feedback(
                    messages,
                    tool_call_id=tc.id,
                    content=tc_rec.result,
                    native_tool_calling=resp.has_tool_calls,
                )
                continue

            if tc.name == "scratchpad":
                if phase_name == "orientation":
                    tc_rec.result = (
                        "scratchpad is not available during orientation. After each SQL/analysis "
                        "batch, call `orientation_report` (mode=append) with a short `##` section — "
                        "not raw SQL dumps."
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue

            if tc.name == "orientation_report":
                if phase_name != "orientation":
                    tc_rec.result = (
                        "orientation_report is only available during orientation."
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                from researchpkg.forensic_llm.artefacts import (
                    append_orientation_report_text,
                    load_orientation_report_text,
                    replace_orientation_report_text,
                    validate_orientation_section,
                )

                args = tc.arguments or {}
                mode = str(args.get("mode", "append")).lower()
                text_raw = str(args.get("text", ""))
                if mode == "read":
                    body = load_orientation_report_text(self._run_dir)
                    n_sections = len(
                        [ln for ln in body.splitlines() if ln.strip().startswith("##")]
                    )
                    tc_rec.result = (
                        f"[orientation/orientation_report.md] {len(body):,} chars, "
                        f"{n_sections} section(s). Full text is in the "
                        "[Current Orientation Report] slot — not repeated here."
                    )
                    step.tool_calls.append(tc_rec)
                    tool_result_texts.append(tc_rec.result)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                if not text_raw.strip():
                    tc_rec.result = "orientation_report: provide non-empty `text` for append/replace."
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                if mode == "replace":
                    nbytes = replace_orientation_report_text(self._run_dir, text_raw)
                    validation = validate_orientation_section(text_raw)
                    tc_rec.result = (
                        f"Replaced orientation/orientation_report.md "
                        f"({len(text_raw):,} chars). File ~{nbytes:,} bytes.\n\n"
                        f"**Section validation:** {'✓ Valid' if validation['valid'] else '⚠ Issues detected'}\n"
                        f"{validation['feedback']}"
                    )
                else:
                    nbytes = append_orientation_report_text(self._run_dir, text_raw)
                    validation = validate_orientation_section(text_raw)
                    feedback_suffix = ""
                    if validation.get("issues"):
                        feedback_suffix = (
                            f"\n\n**Note:**\n{validation['feedback']}\n\n"
                            f"Update the report often; next step clears [Current Step] SQL."
                        )
                    tc_rec.result = (
                        f"Appended to orientation/orientation_report.md "
                        f"({len(text_raw):,} chars). File ~{nbytes:,} bytes.{feedback_suffix}"
                    )
                self._orientation_report_dirty = True
                wrote_orientation_report = True
                step.tool_calls.append(tc_rec)
                tool_result_texts.append(tc_rec.result)
                _append_tool_feedback(
                    messages,
                    tool_call_id=tc.id,
                    content=tc_rec.result,
                    native_tool_calling=resp.has_tool_calls,
                )
                continue

            worker_runtime_result = self._handle_worker_runtime_tool(
                tc.name, tc.arguments
            )
            if worker_runtime_result is not None:
                tc_rec.result = worker_runtime_result
                step.tool_calls.append(tc_rec)
                tool_result_texts.append(tc_rec.result)
                _append_tool_feedback(
                    messages,
                    tool_call_id=tc.id,
                    content=tc_rec.result,
                    native_tool_calling=resp.has_tool_calls,
                )
                continue

            if tc.name == "complete_orientation":
                if phase_name != "orientation":
                    tc_rec.result = (
                        "complete_orientation is only available during orientation."
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                from researchpkg.forensic_llm.artefacts import (
                    is_orientation_report_substantive,
                    load_orientation_report_text,
                    orientation_completion_issues,
                )

                note = str((tc.arguments or {}).get("readiness_note", "")).strip()
                report_body = load_orientation_report_text(self._run_dir)
                ob_ctx = getattr(self, "_orientation_budget_ctx", None)
                cap_o = int(ob_ctx.get("cap") or 0) if isinstance(ob_ctx, dict) else 0
                used_o = self.budget.phase_tokens_used("orientation")
                min_fr = float(self.config.orientation_budget_min_fraction_for_complete)
                if (
                    cap_o > 0
                    and min_fr > 0.0
                    and (used_o / float(cap_o)) + 1e-9 < min_fr
                ):
                    pct_o = 100.0 * float(used_o) / float(cap_o)
                    tc_rec.result = (
                        f"Cannot complete orientation yet: only **{pct_o:.1f}%** of the orientation "
                        f"cap is used ({used_o:,} / {cap_o:,} input tokens; target at least ~"
                        f"{min_fr * 100.0:.0f}% ≈ **{int(min_fr * cap_o):,}** tokens). Continue "
                        f"profiling — append measured `orientation_report` sections. "
                        f"(Checkpoint messages show live orientation %.)"
                        + (f" Note: {note}" if note else "")
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                if not is_orientation_report_substantive(report_body):
                    tc_rec.result = (
                        "Cannot complete orientation yet: `orientation/orientation_report.md` "
                        "is still too thin. Append measured sections (schema/grain, process mix %, "
                        "fiscal range, CoA corridors, actor baselines, period-end behaviour, "
                        "master-data gaps, planning leads) via `orientation_report` before "
                        "calling `complete_orientation` again."
                        + (f" Note: {note}" if note else "")
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                completion_gaps = orientation_completion_issues(report_body)
                if completion_gaps:
                    bullets = "\n".join(f"- {g}" for g in completion_gaps)
                    tc_rec.result = (
                        "Cannot complete orientation yet — planning checklist gaps:\n"
                        f"{bullets}\n\n"
                        "Append via `orientation_report` (prefer markdown tables). "
                        "Resolve all items before calling `complete_orientation` again."
                        + (f" Note: {note}" if note else "")
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue
                tc_rec.result = (
                    "Orientation phase complete. Planning will use "
                    "`orientation/orientation_report.md` (raw report, first ~100k tokens)."
                    + (f" Note: {note}" if note else "")
                )
                step.tool_calls.append(tc_rec)
                _append_tool_feedback(
                    messages,
                    tool_call_id=tc.id,
                    content=tc_rec.result,
                    native_tool_calling=resp.has_tool_calls,
                )
                finished = True
                continue

            if tc.name == "finish_investigation":
                suspicion_raw = tc.arguments.get("suspicion_list", [])

                if not allow_finish:
                    # Still accumulate JE-level detections from this attempt so
                    # partial results are saved even when synthesis is never reached.
                    tentative = _parse_suspicion_list(suspicion_raw)
                    if tentative:
                        self._flush_live_detections(tentative)
                    tc_rec.result = (
                        "finish_investigation not allowed in this phase. "
                        "Continue investigating."
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=(
                            "finish_investigation is not available during this phase. "
                            "Continue with your investigation."
                        ),
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue

                narrative = tc.arguments.get("narrative", "")

                if not report.budget_exhausted:
                    tentative_items = _parse_suspicion_list(suspicion_raw)

                    # Budget-fraction floor check (plan_execute_reflect loop).
                    min_frac = self.config.min_budget_fraction_for_finish
                    if min_frac > 0 and self.budget.fraction_used < min_frac:
                        if tentative_items:
                            self._flush_live_detections(tentative_items)
                        tc_rec.result = (
                            f"finish_investigation ignored: only "
                            f"{self.budget.fraction_used * 100:.1f}% of the token "
                            f"budget has been used (minimum required: "
                            f"{min_frac * 100:.0f}%). Continue exploring."
                        )
                        step.tool_calls.append(tc_rec)
                        _append_tool_feedback(
                            messages,
                            tool_call_id=tc.id,
                            content=(
                                f"Investigation ended too early "
                                f"({self.budget.fraction_used * 100:.1f}% budget used). "
                                "You must explore more of the database before finishing. "
                                "Continue investigation (sql, write_csv, "
                                "code_interpreter) for schemes not yet "
                                "covered, then call `finish_investigation` again."
                            ),
                            native_tool_calling=resp.has_tool_calls,
                        )
                        continue

                # Validate document IDs against je_header before accepting.
                tentative_final = _parse_suspicion_list(suspicion_raw)
                valid_items, invalid_ids = self._validate_suspicion_ids(tentative_final)

                if invalid_ids and not report.budget_exhausted:
                    self._flush_live_detections(valid_items)
                    sample = invalid_ids[:5]
                    tc_rec.result = (
                        f"finish_investigation rejected: {len(invalid_ids)} of "
                        f"{len(tentative_final)} submitted document_ids were NOT "
                        f"found in je_header. Invalid IDs (sample): {sample}. "
                        "You must report document_id values queried DIRECTLY from "
                        "je_header (e.g. SELECT document_id FROM je_header WHERE ...). "
                        "Do NOT submit IDs from memory, aggregated queries, or "
                        "constructed values. Retrieve actual document_ids from je_header "
                        "(sql or write_csv), then call `finish_investigation` again."
                    )
                    step.tool_calls.append(tc_rec)
                    _append_tool_feedback(
                        messages,
                        tool_call_id=tc.id,
                        content=tc_rec.result,
                        native_tool_calling=resp.has_tool_calls,
                    )
                    continue

                report.suspicion_list = valid_items
                report.narrative = narrative
                self._flush_live_detections(report.suspicion_list)
                finished = True
                dropped_msg = (
                    f" ({len(invalid_ids)} invalid IDs dropped)" if invalid_ids else ""
                )
                tc_rec.result = (
                    f"Investigation finalised. "
                    f"{len(report.suspicion_list)} suspicion(s) recorded{dropped_msg}."
                )
                step.tool_calls.append(tc_rec)
                break

            t_tool = time.perf_counter()
            result = dispatch(tc.name, tc.arguments)
            tc_rec.elapsed_ms = (time.perf_counter() - t_tool) * 1000
            tc_rec.result = result
            if tc.name in ("sql", "write_csv"):
                if "[SQL ERROR]" in result or "[SQL_EXPORT ERROR]" in result:
                    self._sql_error_count += 1
                elif (
                    tc.name == "sql"
                    and str((tc.arguments or {}).get("mode", "preview")).lower()
                    == "preview"
                    and "(no rows returned)" in result
                ):
                    self._sql_zero_row_count += 1
            tc_rec.tokens_output = self.budget.record_tool_tokens(result, tc.name)
            step.tool_calls.append(tc_rec)
            model_result = _tool_result_for_model_history(
                tc.name,
                result,
                tc.arguments,
                phase_name=phase_name,
                orientation_sql_max_rows=int(
                    self.config.text_limits.orientation_sql_context_max_rows
                ),
            )
            tool_result_texts.append(model_result)
            if self._worker_run_dir and tc.name in (
                "sql",
                "code_interpreter",
                "write_csv",
            ):
                self._harvest_document_ids_from_tool_result(result)
            # Update step file live when the agent writes to the scratchpad.
            if tc.name == "scratchpad":
                wrote_scratchpad = True
                self._persist_scratchpad_step()
                self._scratchpad_dirty = True  # rebuild slot before next LLM call
            ctx_result = model_result
            if resp.has_tool_calls:
                messages.append(_tool_result_msg(tc.id, ctx_result))
            else:
                messages.append(_observation_msg(ctx_result))

        # ------------------------------------------------------------------
        # Context-stability guardrail:
        # Hypothesis phases: if SQL/code_interpreter ran but scratchpad was not updated,
        # append a short auto scratchpad stub (tool excerpts OK there — not in orientation_report).
        # Orientation: never append SQL excerpts to orientation_report.md; synthesize via
        # orientation_report. Ephemeral SQL feedback exists only within the current step;
        # _compact_orientation_message_history clears chat before the next step.
        # ------------------------------------------------------------------
        needs_reflection = used_sql_or_plot and (
            (phase_name == "orientation" and not wrote_orientation_report)
            or (phase_name != "orientation" and not wrote_scratchpad)
        )
        if needs_reflection:
            try:
                if phase_name == "orientation":
                    log.debug(
                        "Orientation step %d: SQL/analysis without orientation_report append — "
                        "findings must be synthesized via orientation_report (chat has tool output).",
                        self._step_num,
                    )
                else:
                    tail = "\n\n".join(
                        (
                            tool_result_texts[-2:]
                            if len(tool_result_texts) >= 2
                            else tool_result_texts[-1:]
                        )
                    )
                    tail = truncate_text_to_tokens(
                        (tail or "").strip(),
                        self.config.text_limits.auto_scratchpad_tail,
                        side=TruncationSide.TAIL,
                    )
                    auto_note = (
                        f"## Auto-scratchpad (step {self._step_num}, phase={phase_name or '(none)'})\n"
                        f"- Ran SQL/plot but no scratchpad reflection was written.\n"
                        f"- Last tool result(s):\n{tail or '(empty tool output)'}\n\n"
                        f"Next: interpret the result, update hypotheses status, and choose a distinct next query."
                    )
                    scratchpad(note=auto_note, mode="append")
                    self._persist_scratchpad_step()
            except Exception as exc:
                log.debug("Auto reflection stub failed: %s", exc)

        if not tool_calls:
            self._tool_failure_count += 1
        else:
            self._tool_failure_count = 0

        # ------------------------------------------------------------------
        # Loop breaker: same tool call repeated too many times
        # ------------------------------------------------------------------
        tool_signature = "|".join(tool_signature_parts).strip()
        if tool_signature and tool_signature == self._last_tool_signature:
            self._last_tool_signature_repeats += 1
        else:
            self._last_tool_signature = tool_signature
            self._last_tool_signature_repeats = 1 if tool_signature else 0

        if not finished and tool_signature and self._last_tool_signature_repeats >= 5:
            # Inject a forced reflection step to break repetition.
            # Only append a user-role message when the last message is not a tool
            # result (inserting user after tool violates strict OpenAI backends).
            if not messages or messages[-1].get("role") != "tool":
                if self._current_phase_name == "orientation":
                    breaker = (
                        "**LOOP BREAKER (orientation)**\n\n"
                        "You repeated the same tool call 5 times. STOP.\n\n"
                        "1) Interpret the last result.\n"
                        "2) `orientation_report(mode=append)` with a new `##` section (measured facts).\n"
                        "3) Run a **different** query or call `complete_orientation` if the report is complete.\n"
                    )
                else:
                    breaker = (
                        "**LOOP BREAKER (forced reflection)**\n\n"
                        "You have repeated the exact same tool call 5 times in a row. "
                        "STOP repeating it.\n\n"
                        "1) Interpret the last SQL/plot result: what it returned, what you conclude, and the next distinct step.\n"
                        '2) Write a structured reflection to the scratchpad (`scratchpad` with `mode: "append"` or `"replace"`).\n'
                        "3) Then proceed with a different query or tool (or call `finish_investigation` if truly complete).\n"
                    )
                messages.append(_user_msg(breaker))
            # Reset so we don't spam the same breaker message every step.
            self._last_tool_signature_repeats = 0

        from researchpkg.forensic_llm.token_budget import (
            effective_reasoning_tokens,
        )

        step.llm_tokens_reasoning = effective_reasoning_tokens(
            resp.reasoning_tokens, resp.reasoning or ""
        )
        self.budget.record_llm_call(
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            reasoning_tokens=resp.reasoning_tokens,
            reasoning_text=resp.reasoning or "",
        )
        step.llm_tokens_input = resp.prompt_tokens
        step.llm_tokens_output = resp.completion_tokens
        report.steps.append(step)
        self._emit_step(step)
        if phase_name != "orientation":
            self._persist_scratchpad_step()
        self._persist_memory_snapshot(messages)

        if phase_name == "orientation":
            self._sync_orientation_report_live_after_step(
                step,
                wrote_orientation_report=wrote_orientation_report,
            )
            if wrote_orientation_report:
                self._compact_orientation_message_history(
                    messages,
                    used_sql_or_plot=used_sql_or_plot,
                    wrote_orientation_report=True,
                )
            elif used_sql_or_plot:
                self._trim_orientation_to_ephemeral_workspace(messages)
            else:
                self._compact_orientation_message_history(
                    messages,
                    used_sql_or_plot=False,
                    wrote_orientation_report=False,
                )

        step_budget_tokens = resp.prompt_tokens + step.llm_tokens_reasoning
        phase_used = (
            self.budget.phase_tokens_used(phase_name)
            if phase_name
            else self.budget.used_tokens
        )
        phase_suffix = f" | phase={phase_name}" if phase_name else ""
        if phase_name == "orientation":
            ob = getattr(self, "_orientation_budget_ctx", None)
            orient_cap = int(ob["cap"]) if isinstance(ob, dict) and ob.get("cap") else 0
            if orient_cap > 0:
                orient_pct = 100.0 * phase_used / orient_cap
                log.info(
                    "Step %d | tools=%d | step=%d | phase=%d/%d (%.1f%% orient)%s",
                    self._step_num,
                    len(tool_calls),
                    step_budget_tokens,
                    phase_used,
                    orient_cap,
                    orient_pct,
                    phase_suffix,
                )
            else:
                log.info(
                    "Step %d | tools=%d | step=%d | phase=%d%s",
                    self._step_num,
                    len(tool_calls),
                    step_budget_tokens,
                    phase_used,
                    phase_suffix,
                )
        elif phase_name and self._hypothesis_task_cap > 0:
            task_cap = self._hypothesis_task_cap
            task_pct = 100.0 * phase_used / task_cap
            log.info(
                "Step %d | tools=%d | step=%d | phase=%d/%d (%.1f%% task)%s",
                self._step_num,
                len(tool_calls),
                step_budget_tokens,
                phase_used,
                task_cap,
                task_pct,
                phase_suffix,
            )
        elif phase_name:
            log.info(
                "Step %d | tools=%d | step=%d | phase=%d | global=%.1f%%%s",
                self._step_num,
                len(tool_calls),
                step_budget_tokens,
                phase_used,
                self.budget.fraction_used * 100,
                phase_suffix,
            )
        else:
            log.info(
                "Step %d | tools=%d | step=%d | global=%.1f%%",
                self._step_num,
                len(tool_calls),
                step_budget_tokens,
                self.budget.fraction_used * 100,
            )

        return step, tool_calls, finished

    def _persist_scratchpad_step(self) -> None:
        """Write current scratchpad to scratchpad_steps/step_NNNN.md for post-debug."""
        try:
            path = self._scratchpad_steps_dir / f"step_{self._step_num:04d}.md"
            path.write_text(get_scratchpad_text(), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not persist scratchpad step %d: %s", self._step_num, exc)

    def _persist_memory_snapshot(
        self,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Persist the current 4-slot memory state to memory.md."""
        try:
            slot_plan = self.config.slot_plan_tokens
            slot_past = self.config.slot_past_tokens
            slot_recent = self.config.slot_recent_tokens
            slot_input = self.config.slot_input_tokens

            plan_tok = count_tokens(self._plan_slot) if self._plan_slot else 0
            past_tok = count_tokens(self._past_memory) if self._past_memory else 0

            # Recent window from the live messages list (excluding current/last msg).
            older = messages[:-1] if len(messages) > 1 else []
            recent: List[Dict[str, Any]] = []
            recent_tok = 0
            for msg in reversed(older):
                mt = _estimate_full_payload_tokens([msg])
                if recent_tok + mt > slot_recent:
                    break
                recent.insert(0, msg)
                recent_tok += mt

            recent_blocks: List[str] = []
            for m in recent:
                role = m.get("role", "")
                content = _msg_text_content(m)
                if content:
                    recent_blocks.append(f"### [{role}]\n\n{content}\n")

            current_tok = (
                _estimate_full_payload_tokens([messages[-1]]) if messages else 0
            )

            memory_text = (
                "# 4-Slot Memory — payload sent to the LLM\n\n"
                f"| Slot | Budget | Used |\n"
                f"|---|---|---|\n"
                f"| 1 – Plan state | {slot_plan:,} tok | {plan_tok:,} tok |\n"
                f"| 2 – Past memory | {slot_past:,} tok | {past_tok:,} tok |\n"
                f"| 3 – Recent msgs | {slot_recent:,} tok | {recent_tok:,} tok |\n"
                f"| 4 – Current input | {slot_input:,} tok | {current_tok:,} tok |\n\n"
                "---\n\n"
                f"## Slot 1 — Plan State ({plan_tok:,} tokens)\n\n"
                f"{self._plan_slot or '(empty)'}\n\n"
                "---\n\n"
                f"## Slot 2 — Past Memory ({past_tok:,} tokens)\n\n"
                f"{self._past_memory or '(empty)'}\n\n"
                "---\n\n"
                f"## Slot 3 — Recent Messages ({recent_tok:,} tokens)\n\n"
                + "\n".join(recent_blocks)
            )

            path = self._run_dir / "memory.md"
            path.write_text(memory_text, encoding="utf-8")
        except Exception as exc:
            log.warning("Could not persist memory snapshot: %s", exc)

    def _persist_investigation_plan(
        self,
        plan: Optional[InvestigationPlan],
        *,
        tag: str,
    ) -> None:
        """Persist the current InvestigationPlan early (and on re-plans)."""
        if plan is None or not getattr(plan, "phases", None):
            return
        try:
            latest = self._run_dir / "investigation_plan.json"
            payload = json.dumps(
                plan.model_dump(exclude_none=True),
                indent=2,
                default=str,
            )
            latest.write_text(payload, encoding="utf-8")

            steps_dir = self._run_dir / "investigation_plan_steps"
            steps_dir.mkdir(parents=True, exist_ok=True)
            safe_tag = re.sub(r"[^\w.-]+", "_", tag).strip("_") or "plan"
            snap = steps_dir / f"step_{self._step_num:04d}_{safe_tag}.json"
            snap.write_text(payload, encoding="utf-8")
        except Exception as exc:
            log.warning("Could not persist investigation plan (%s): %s", tag, exc)

    def _orientation_digest_tokens(self, text: str) -> int:
        from researchpkg.forensic_llm.model_tokenizer import (
            count_tokens,
        )

        if not text:
            return 0
        try:
            return int(count_tokens(text))
        except RuntimeError:
            return max(1, len(text) // 4)

    def _compress_orientation_digest_llm(self, text: str, max_out_tokens: int) -> str:
        """Single LLM pass to shorten orientation markdown while preserving numbers."""
        if not text.strip() or max_out_tokens <= 0:
            return ""
        out_cap = max(512, int(max_out_tokens))
        system = build_system_prompt(
            task=self.config.task,
            scratchpad_text="",
            is_worker=self._worker_run_dir is not None,
        )
        prompt = (
            "Condense forensic **orientation notes** (markdown) for downstream planning.\n\n"
            f"**Hard output ceiling:** about **{out_cap:,} tokens** on this server's tokenizer "
            "(stay clearly below it).\n\n"
            "**Rules:**\n"
            "- Do **not** invent facts, entities, or numbers.\n"
            "- Preserve **all** quantitative baselines: counts, %, dates, GL accounts, medians, "
            "top-N shares, and stated anomalies.\n"
            "- Remove redundancy and low-signal prose.\n"
            "- Output **markdown only** (no preamble).\n\n"
            "Source:\n```markdown\n"
            f"{text.strip()}\n```"
        )
        payload = [_system_msg(system), _user_msg(prompt)]
        max_input = self._get_max_input_tokens(completion_reserve=out_cap + 4096)
        if _estimate_full_payload_tokens(payload) > max_input:
            payload = self._hard_trim_to_fit(payload, max_input)
        try:
            resp = self.client.chat(
                messages=payload,
                tools=None,
                max_tokens=out_cap,
            )
            self.budget.record_llm_call(
                resp.prompt_tokens,
                resp.completion_tokens,
                reasoning_tokens=resp.reasoning_tokens,
                reasoning_text=resp.reasoning or "",
            )
            memo = (resp.content or "").strip()
            if memo:
                return truncate_text_to_tokens(memo, out_cap, side=TruncationSide.TAIL)
        except Exception as exc:
            log.warning("Orientation digest compression failed: %s", exc)
        return ""

    def fit_orientation_report_for_planning(
        self,
        markdown: str,
        cap_tokens: int,
        *,
        digest_source: str = "orientation_report",
        max_iterations: int = 5,
    ) -> Tuple[str, str]:
        """
        Ensure orientation markdown fits the planning context cap.

        Returns
        -------
        (planning_text, provenance_tag)
        """
        cur = (markdown or "").strip()
        if not cur:
            return "", f"{digest_source}_empty"
        if cap_tokens <= 0:
            return cur, f"{digest_source}_uncapped"
        if self._orientation_digest_tokens(cur) <= cap_tokens:
            return cur, f"{digest_source}_direct"
        if self.budget.should_stop():
            clipped = truncate_text_to_tokens(cur, cap_tokens, side=TruncationSide.TAIL)
            return clipped, f"{digest_source}_truncated_budget"

        iterations = 0
        while (
            iterations < max_iterations
            and self._orientation_digest_tokens(cur) > cap_tokens
        ):
            iterations += 1
            over = self._orientation_digest_tokens(cur) - cap_tokens
            target_out = min(
                int(cap_tokens * 0.92),
                max(1024, self._orientation_digest_tokens(cur) - max(400, over // 2)),
            )
            compressed = self._compress_orientation_digest_llm(cur, target_out)
            if not compressed.strip():
                break
            if (
                self._orientation_digest_tokens(compressed)
                >= self._orientation_digest_tokens(cur) * 0.995
            ):
                break
            cur = compressed.strip()

        if self._orientation_digest_tokens(cur) > cap_tokens:
            cur = truncate_text_to_tokens(cur, cap_tokens, side=TruncationSide.TAIL)
            if iterations > 0:
                return cur, f"{digest_source}_compressed_then_truncated"
            return cur, f"{digest_source}_truncated"
        if iterations > 0:
            return cur, f"{digest_source}_compressed"
        return cur, f"{digest_source}_direct"

    def synthesize_orientation_summary_for_planning(
        self,
        orientation_scratchpad: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build an evidence bundle from the orientation scratchpad (+ exports / snapshots),
        then fit it to the planning orientation token cap.

        Notes
        -----
        The canonical v2 path writes `orientation/orientation_report.md` during orientation;
        this method remains for compatibility and scratchpad-only fallbacks.
        """
        _ = messages  # reserved for API compatibility with older callers
        scratchpad = (orientation_scratchpad or "").strip()
        if not scratchpad:
            return ""
        lim = self.config.text_limits
        cap = int(
            lim.planning_orientation_prompt or lim.orientation_summary_store or 16_000
        )
        bundle = build_orientation_synthesis_bundle(
            scratchpad,
            run_dir=self._run_dir,
            limits=lim,
            max_tokens=int(lim.orientation_memo_synthesis_input or 110_000),
        )
        if not bundle.strip():
            return ""
        text, _src = self.fit_orientation_report_for_planning(
            bundle,
            cap,
            digest_source="scratchpad_bundle",
        )
        return text

    def expand_orientation_summary_for_planning(
        self,
        orientation_scratchpad: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Backward-compatible alias for :meth:`synthesize_orientation_summary_for_planning`."""
        return self.synthesize_orientation_summary_for_planning(
            orientation_scratchpad, messages
        )

    def _ensure_orientation_planning_memo(
        self,
        messages: List[Dict[str, Any]],
        orientation_scratchpad: str,
    ) -> str:
        """Return orientation digest for planning (scratchpad bundle → fit to cap)."""
        return self.synthesize_orientation_summary_for_planning(
            orientation_scratchpad, messages
        )

    def _plan_from_obj(self, obj: Dict[str, Any]) -> InvestigationPlan:
        """Build an InvestigationPlan from parsed JSON."""
        phases: List[SchemePhase] = []
        _token_cap = (
            int(self.config.scheme_phase_budget_tokens_default)
            if self.config.scheme_phase_budget_tokens_default is not None
            else None
        )
        for p in obj.get("phases", []):
            scheme_str = p.get("scheme", "unknown")
            try:
                scheme = SchemeType(scheme_str)
            except ValueError:
                continue
            _parsed_budget_tokens = int(
                p.get(
                    "budget_tokens",
                    SchemePhase.model_fields["budget_tokens"].default,  # type: ignore[attr-defined]
                )
            )
            phases.append(
                SchemePhase(
                    scheme=scheme,
                    priority=len(phases) + 1,
                    budget_sql_calls=self._budget_sql_calls_from_plan(
                        p.get("budget_sql_calls")
                    ),
                    budget_tokens=(
                        min(_parsed_budget_tokens, _token_cap)
                        if _token_cap is not None
                        else _parsed_budget_tokens
                    ),
                    initial_hypotheses=p.get("initial_hypotheses", []),
                    plan_rationale=str(p.get("plan_rationale", "")),
                    priority_signals=[
                        str(item) for item in p.get("priority_signals", [])[:6]
                    ],
                    benign_rival_explanations=[
                        str(item) for item in p.get("benign_rival_explanations", [])[:4]
                    ],
                    planned_query_sequence=[
                        str(item) for item in p.get("planned_query_sequence", [])[:6]
                    ],
                    grounding_query_templates=[
                        str(item) for item in p.get("grounding_query_templates", [])[:3]
                    ],
                    exit_criteria=[
                        str(item)
                        for item in p.get("exit_criteria", [])[:4]
                        if len(str(item).strip()) > 2
                    ]
                    or [
                        "Each hypothesis receives CONFIRMED, RULED_OUT, or INCONCLUSIVE with cited evidence."
                    ],
                )
            )

        reflection_after_phases = obj.get("reflection_after_phases", [])
        if getattr(self.config, "parallel_scheme_execution", False):
            reflection_after_phases = []

        dispatch_raw = obj.get("dispatch_queue", [])
        dispatch_queue: List[Any] = []
        if isinstance(dispatch_raw, list):
            from researchpkg.forensic_llm.models import (
                DispatchQueueItem,
            )

            for entry in dispatch_raw:
                if isinstance(entry, dict):
                    dispatch_queue.append(DispatchQueueItem.model_validate(entry))

        plan = InvestigationPlan(
            phases=phases,
            dispatch_queue=dispatch_queue,
            total_budget_sql=int(
                obj.get("total_budget_sql", sum(p.budget_sql_calls for p in phases))
            ),
            orientation_risk_summary=[
                str(item) for item in obj.get("orientation_risk_summary", [])[:10]
            ],
            execution_notes=[str(item) for item in obj.get("execution_notes", [])[:6]],
            reflection_after_phases=reflection_after_phases,
        )
        if not plan.dispatch_queue and phases:
            from researchpkg.forensic_llm.plan_utils import (
                build_dispatch_queue,
            )

            plan.dispatch_queue = build_dispatch_queue(
                plan, self.config.max_hypotheses_per_scheme
            )
        return plan

    def _repair_or_fallback_plan(
        self,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
        partial_plan: InvestigationPlan,
    ) -> InvestigationPlan:
        """Repair a planner response that omitted phases, or synthesize a safe fallback."""
        if not self.budget.should_stop():
            system = build_system_prompt(
                task=self.config.task,
                scratchpad_text=get_scratchpad_text(),
                is_worker=self._worker_run_dir is not None,
            )
            recent = messages[-6:] if len(messages) > 6 else list(messages)
            reflection_hint = (
                "Return `reflection_after_phases`: [] exactly."
                if getattr(self.config, "parallel_scheme_execution", False)
                else "Return a short `reflection_after_phases` list like [2, 4]."
            )
            repair_prompt = (
                "Your previous planning response omitted the required `phases` array.\n\n"
                "Return ONLY a compact JSON object containing:\n"
                "- `phases`: exactly 5 entries, one for each canonical scheme\n"
                "- `reflection_after_phases`\n\n"
                "Required schemes exactly once each:\n"
                "- fictitious_ap_disbursements\n"
                "- revenue_manipulation\n"
                "- vendor_collusion\n"
                "- shadow_payroll\n"
                "- inventory_manipulation\n\n"
                "For each phase include only these fields:\n"
                "- `scheme`\n"
                "- `budget_sql_calls`\n"
                "- `plan_rationale`\n"
                "- `priority_signals`\n"
                "- `benign_rival_explanations`\n"
                "- `planned_query_sequence`\n"
                "- `exit_criteria`\n"
                "- `initial_hypotheses`: array of objects with `hypothesis_id`, `hypothesis_text`, "
                "`hypothesis_rationale` (one paragraph each; no ``If … then …`` boilerplate)\n\n"
                "Keep it compact but **cover distinct forensic angles** per scheme (timing, actors, "
                "amounts, master-data, process controls, period-end). Each scheme needs **"
                f"{self.config.min_hypotheses_per_scheme}–{self.config.max_hypotheses_per_scheme}** "
                "distinct `initial_hypotheses` objects (P1…Pn). " + reflection_hint
            )
            payload = [_system_msg(system)] + recent + [_user_msg(repair_prompt)]
            max_input = self._get_max_input_tokens()
            if _estimate_full_payload_tokens(payload) > max_input:
                payload = self._hard_trim_to_fit(payload, max_input)
            try:
                resp = self.client.chat(messages=payload, tools=None)
                self.budget.record_llm_call(
                    resp.prompt_tokens,
                    resp.completion_tokens,
                    reasoning_tokens=resp.reasoning_tokens,
                )
                obj = _extract_first_json_object(resp.content or "")
                if obj:
                    repaired = self._plan_from_obj(obj)
                    if repaired.phases:
                        repaired.orientation_risk_summary = (
                            repaired.orientation_risk_summary
                            or partial_plan.orientation_risk_summary
                        )
                        repaired.execution_notes = (
                            repaired.execution_notes or partial_plan.execution_notes
                        )
                        if not repaired.total_budget_sql:
                            repaired.total_budget_sql = partial_plan.total_budget_sql
                        log.info(
                            "Recovered missing planning phases with a compact repair step"
                        )
                        return repaired
            except Exception as exc:
                log.warning("Plan repair step failed: %s", exc)

        ordered_schemes = [
            SchemeType.FICTITIOUS_AP_DISBURSEMENTS,
            SchemeType.SHADOW_PAYROLL,
            SchemeType.VENDOR_COLLUSION,
            SchemeType.REVENUE_MANIPULATION,
            SchemeType.INVENTORY_MANIPULATION,
        ]
        _cap = max(1, int(getattr(self.config, "max_hypotheses_per_scheme", 15)))
        fallback_phases = [
            SchemePhase(
                scheme=scheme,
                priority=idx + 1,
                budget_sql_calls=self._budget_sql_calls_from_plan(),
                initial_hypotheses=fallback_initial_hypotheses_for_scheme(
                    scheme.value, _cap
                ),
                plan_rationale=(
                    "Fallback stable phase because the planner omitted `phases`; "
                    "default exhaustive hypothesis checklist — refine using orientation memo and SQL."
                ),
                priority_signals=partial_plan.orientation_risk_summary[:3],
                benign_rival_explanations=[
                    "service or system-account behavior",
                    "master-data design mismatch",
                ],
                planned_query_sequence=[
                    "discover relevant GL accounts and process baselines",
                    "screen the strongest scheme-specific anomaly pattern",
                    "drill down top candidates and falsify a benign rival",
                    "retrieve exact journal-entry UUIDs for any confirmed lead",
                ],
                grounding_query_templates=[
                    "SELECT document_id FROM journal_entries WHERE <confirmed_pattern_filters> LIMIT 500",
                ],
                exit_criteria=[
                    "record a scheme verdict grounded in SQL results",
                    "leave no high-priority hypothesis unresolved",
                ],
            )
            for idx, scheme in enumerate(ordered_schemes)
        ]
        log.warning(
            "Using synthesized fallback investigation plan after planner omitted phases"
        )
        from researchpkg.forensic_llm.plan_utils import (
            enforce_plan_hypothesis_guardrails,
        )

        fallback_plan = InvestigationPlan(
            phases=fallback_phases,
            total_budget_sql=sum(p.budget_sql_calls for p in fallback_phases),
            orientation_risk_summary=partial_plan.orientation_risk_summary,
            execution_notes=partial_plan.execution_notes,
            reflection_after_phases=(
                []
                if getattr(self.config, "parallel_scheme_execution", False)
                else [2, 4]
            ),
        )
        fallback_plan, _ = enforce_plan_hypothesis_guardrails(
            fallback_plan,
            min_per_scheme=self.config.min_hypotheses_per_scheme,
            max_per_scheme=self.config.max_hypotheses_per_scheme,
        )
        return fallback_plan

    def _parse_plan(self, step: AgentStep) -> InvestigationPlan:
        """Extract an InvestigationPlan from the LLM's response text.

        For thinking models, the JSON is in step.content (visible output), not in
        step.reasoning (chain-of-thought). Try content first, fall back to reasoning.
        """
        text = step.content or step.reasoning or ""
        obj = _extract_first_json_object(text)
        if not obj:
            return InvestigationPlan()
        nested = obj.get("investigation_plan")
        if isinstance(nested, dict):
            obj = {
                **nested,
                **{k: v for k, v in obj.items() if k != "investigation_plan"},
            }
        return self._plan_from_obj(obj)

    def _run_reflection(
        self, report: ForensicReport, messages: List[Dict[str, Any]]
    ) -> None:
        """Retired in hypothesis_orchestrated v2."""
        log.debug("_run_reflection skipped (retired)")

    def _run_replan(
        self,
        phase_idx: int,
        plan: InvestigationPlan,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
        reflection_summary: str,
        phase_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[SchemePhase]]:
        """
        ADaPT re-planning: ask the LLM to revise remaining phases given reflection.
        Returns a list of SchemePhase for remaining work, or None if no revision.
        """
        self.budget.start_phase(f"replan_{self._step_num}")
        completed = [p.scheme.value for p in plan.phases[: phase_idx + 1]]
        remaining = [
            {
                "scheme": p.scheme.value,
                "priority": p.priority,
                "budget_sql_calls": p.budget_sql_calls,
                "initial_hypotheses": getattr(p, "initial_hypotheses", []),
                "plan_rationale": getattr(p, "plan_rationale", ""),
                "priority_signals": getattr(p, "priority_signals", []),
                "benign_rival_explanations": getattr(
                    p, "benign_rival_explanations", []
                ),
                "planned_query_sequence": getattr(p, "planned_query_sequence", []),
                "grounding_query_templates": getattr(
                    p, "grounding_query_templates", []
                ),
                "exit_criteria": getattr(p, "exit_criteria", []),
            }
            for p in plan.phases[phase_idx + 1 :]
        ]
        if not remaining:
            return None
        log.debug("_run_replan skipped (retired)")
        return None

    def _parse_revised_plan(self, step: AgentStep) -> Optional[List[SchemePhase]]:
        """
        Parse revised phases from the LLM response after re-plan prompt.
        Returns list of SchemePhase if valid JSON found, None if NO_REVISION or parse failure.
        """
        text = (step.reasoning or "").strip()
        if "NO_REVISION" in text.upper() and len(text) < 50:
            return None
        obj = _extract_first_json_object(text)
        if not obj:
            return None
        phases_raw = obj.get("phases", [])
        if not isinstance(phases_raw, list):
            return None
        phases: List[SchemePhase] = []
        _token_cap = (
            int(self.config.scheme_phase_budget_tokens_default)
            if self.config.scheme_phase_budget_tokens_default is not None
            else None
        )
        for p in phases_raw:
            if not isinstance(p, dict):
                continue
            scheme_str = p.get("scheme", "unknown")
            try:
                scheme = SchemeType(scheme_str)
            except ValueError:
                continue
            _parsed_budget_tokens = int(
                p.get(
                    "budget_tokens",
                    SchemePhase.model_fields["budget_tokens"].default,  # type: ignore[attr-defined]
                )
            )
            phases.append(
                SchemePhase(
                    scheme=scheme,
                    priority=len(phases) + 1,
                    budget_sql_calls=self._budget_sql_calls_from_plan(
                        p.get("budget_sql_calls")
                    ),
                    budget_tokens=(
                        min(_parsed_budget_tokens, _token_cap)
                        if _token_cap is not None
                        else _parsed_budget_tokens
                    ),
                    initial_hypotheses=p.get("initial_hypotheses", []),
                    plan_rationale=str(p.get("plan_rationale", "")),
                    priority_signals=[
                        str(item) for item in p.get("priority_signals", [])[:6]
                    ],
                    benign_rival_explanations=[
                        str(item) for item in p.get("benign_rival_explanations", [])[:4]
                    ],
                    planned_query_sequence=[
                        str(item) for item in p.get("planned_query_sequence", [])[:6]
                    ],
                    grounding_query_templates=[
                        str(item) for item in p.get("grounding_query_templates", [])[:3]
                    ],
                    exit_criteria=[
                        str(item) for item in p.get("exit_criteria", [])[:4]
                    ],
                )
            )
        return phases if phases else None

    def _run_synthesis(
        self, report: ForensicReport, messages: List[Dict[str, Any]]
    ) -> None:
        """Retired in hypothesis_orchestrated v2."""
        log.debug("_run_synthesis skipped (retired)")

    # ------------------------------------------------------------------
    # Phase summary collection and cross-scheme analysis
    # ------------------------------------------------------------------

    def _run_phase_summary_step(
        self,
        scheme: str,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Retired in hypothesis_orchestrated v2 (per-hypothesis pN.json instead)."""
        log.debug("_run_phase_summary_step skipped for %s (retired)", scheme)
        return

        if self.budget.should_stop():  # pragma: no cover
            return

        system = build_system_prompt(
            task=self.config.task,
            scratchpad_text=get_scratchpad_text(),
            is_worker=self._worker_run_dir is not None,
        )
        recent = messages[-6:] if len(messages) > 6 else list(messages)
        payload = [_system_msg(system)] + recent + [_user_msg("")]

        # Enforce context limit on this lightweight call.
        max_input = self._get_max_input_tokens()
        if _estimate_full_payload_tokens(payload) > max_input:
            payload = self._hard_trim_to_fit(payload, max_input)

        try:
            resp = self.client.chat(messages=payload, tools=None)
            self.budget.record_llm_call(
                resp.prompt_tokens,
                resp.completion_tokens,
                reasoning_tokens=resp.reasoning_tokens,
            )
            ps = self._parse_phase_summary(resp.content or "", scheme)
            if ps:
                phase_name = f"scheme_{scheme}"
                actual_effective_sql = self._phase_effective_sql_calls(phase_name)
                if actual_effective_sql > 0:
                    ps.sql_calls_used = actual_effective_sql
                self._phase_summaries[scheme] = ps
                log.info(
                    "Phase summary [%s]: hypotheses=%d/%d confirmed, "
                    "flagged_jes=%d, entities=%d, confidence=%s",
                    scheme,
                    ps.hypotheses_confirmed,
                    ps.hypotheses_tested,
                    len(ps.flagged_document_ids),
                    len(ps.flagged_entities),
                    ps.confidence,
                )
            else:
                log.warning("Could not parse phase summary for %s", scheme)
        except Exception as exc:
            log.warning("Phase summary LLM call failed for %s: %s", scheme, exc)

    def _run_worker_summary_step(
        self,
        brief: WorkerBrief,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
    ) -> Optional[WorkerSummary]:
        """Ask the LLM for a structured worker summary without using tools."""
        if self.budget.should_stop():
            return None

        system = build_system_prompt(
            task=self.config.task,
            scratchpad_text=get_scratchpad_text(),
            is_worker=True,
        )
        recent = messages[-6:] if len(messages) > 6 else list(messages)
        payload = (
            [_system_msg(system)]
            + recent
            + [_user_msg(build_worker_summary_prompt(brief))]
        )
        max_input = self._get_max_input_tokens()
        if _estimate_full_payload_tokens(payload) > max_input:
            payload = self._hard_trim_to_fit(payload, max_input)

        try:
            resp = self.client.chat(messages=payload, tools=None)
            self.budget.record_llm_call(
                resp.prompt_tokens,
                resp.completion_tokens,
                reasoning_tokens=resp.reasoning_tokens,
            )
            obj = _extract_first_json_object(resp.content or "")
            if not obj:
                return None
            summary = WorkerSummary(
                worker_id=brief.worker_id,
                scheme_or_goal=str(obj.get("scheme_or_goal", brief.scheme_or_goal)),
                status=str(obj.get("status", "completed")),
                candidate_schemes=self._normalise_candidate_schemes(
                    obj.get("candidate_schemes", brief.candidate_schemes)
                ),
                flagged_document_ids=list(obj.get("flagged_document_ids", [])),
                flagged_entities=list(obj.get("flagged_entities", [])),
                total_flagged_amount=float(obj.get("total_flagged_amount", 0.0)),
                key_findings=str(obj.get("key_findings", "")),
                open_questions=list(obj.get("open_questions", [])),
                evidence_checks_run=list(obj.get("evidence_checks_run", [])),
                recommended_scheme_verdicts={
                    str(k): str(v)
                    for k, v in dict(obj.get("recommended_scheme_verdicts", {})).items()
                },
                confidence=str(obj.get("confidence", "medium")),
                sql_calls_used=int(
                    obj.get(
                        "sql_calls_used",
                        self.budget.phase_sql_calls(f"worker_{brief.worker_id}"),
                    )
                ),
                code_interpreter_calls_used=int(
                    obj.get(
                        "code_interpreter_calls_used",
                        self.budget.phase_code_interpreter_calls(
                            f"worker_{brief.worker_id}"
                        ),
                    )
                ),
            )
            return summary
        except Exception as exc:
            log.warning(
                "Worker summary LLM call failed for %s: %s", brief.worker_id, exc
            )
            return None

    def _parse_phase_summary(self, text: str, scheme: str) -> Optional[PhaseSummary]:
        """Parse a PhaseSummary from LLM response text."""
        obj = _extract_first_json_object(text)
        if not obj:
            return None
        try:
            return PhaseSummary(
                scheme=scheme,
                hypotheses_tested=int(obj.get("hypotheses_tested", 0)),
                hypotheses_confirmed=int(obj.get("hypotheses_confirmed", 0)),
                hypotheses_ruled_out=int(obj.get("hypotheses_ruled_out", 0)),
                hypotheses_open=int(obj.get("hypotheses_open", 0)),
                flagged_entities=list(obj.get("flagged_entities", [])),
                flagged_document_ids=list(obj.get("flagged_document_ids", [])),
                total_flagged_amount=float(obj.get("total_flagged_amount", 0.0)),
                confidence=str(obj.get("confidence", "low")),
                sql_calls_used=int(obj.get("sql_calls_used", 0)),
                open_questions=list(obj.get("open_questions", [])),
                key_findings=str(obj.get("key_findings", "")),
            )
        except Exception as exc:
            log.warning("Could not construct PhaseSummary for %s: %s", scheme, exc)
            return None

    def _build_phase_summaries_context(self) -> str:
        """Return a formatted Markdown block of all collected phase summaries."""
        if not self._phase_summaries:
            return ""
        lines = ["## Per-Scheme Investigation Summaries\n"]
        for scheme, ps in self._phase_summaries.items():
            lines.append(f"### {scheme}")
            lines.append(
                f"- Hypotheses: tested={ps.hypotheses_tested}, "
                f"confirmed={ps.hypotheses_confirmed}, "
                f"ruled_out={ps.hypotheses_ruled_out}, "
                f"open={ps.hypotheses_open}"
            )
            lines.append(
                f"- Confidence: {ps.confidence} | Investigation tool calls: {ps.sql_calls_used}"
            )
            if ps.flagged_entities:
                lines.append(
                    f"- Flagged entities: {', '.join(ps.flagged_entities[:10])}"
                )
            n_docs = len(ps.flagged_document_ids)
            if n_docs:
                sample = ", ".join(ps.flagged_document_ids[:5])
                suffix = " ..." if n_docs > 5 else ""
                lines.append(f"- Flagged JEs ({n_docs}): {sample}{suffix}")
            if ps.total_flagged_amount:
                lines.append(f"- Total flagged amount: {ps.total_flagged_amount:,.2f}")
            if ps.key_findings:
                lines.append(f"- Key findings: {ps.key_findings}")
            if ps.open_questions:
                lines.append(f"- Open questions: {'; '.join(ps.open_questions[:3])}")
            lines.append("")
        return "\n".join(lines)

    def _run_cross_scheme_analysis(
        self,
        report: ForensicReport,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Retired in hypothesis_orchestrated v2 (blackboard + global.json instead)."""
        log.debug("_run_cross_scheme_analysis skipped (retired)")
        return

        if self.budget.should_stop():  # pragma: no cover
            return

        self.budget.start_phase("cross_scheme_analysis")
        for _ in range(0):
            if self.budget.should_stop():
                break
            step, tool_calls, finished = self._run_one_step(
                report, messages, allow_finish=False, phase_name="cross_scheme_analysis"
            )
            if not tool_calls:
                break

        log.info(
            "Cross-scheme analysis complete at step %d | total effective SQL so far: %d",
            self._step_num,
            self._effective_sql_call_total,
        )

    # ------------------------------------------------------------------
    # Context management — multi-slot payload (see CONTEXT_MANAGEMENT.md)
    # ------------------------------------------------------------------

    def _get_max_input_tokens(self, completion_reserve: Optional[int] = None) -> int:
        """API-safe maximum input token count."""
        reserve = (
            int(completion_reserve)
            if completion_reserve is not None
            else int(self.config.llm.max_tokens_per_step)
        )
        return self.config.tokens.max_input_tokens(completion_reserve=reserve)

    def _inflated_token_estimate(self, messages: List[Dict[str, Any]]) -> int:
        """Payload token count (HuggingFace model tokenizer)."""
        factor = getattr(self.config, "context_estimate_inflation", 1.0)
        return int(_estimate_full_payload_tokens(messages) * factor)

    def _effective_slot_caps(self, phase_name: str = "") -> Dict[str, int]:
        """
        Per-slot caps that fit inside ``_get_max_input_tokens()``.

        System framing + slack use absolute ``pack_*_tokens`` from
        ``ContextTokenConfig``; the remainder ``pool`` is split across slots using
        ``slot_pack_weights``, each capped by the configured slot ceilings.
        """
        max_input = self._get_max_input_tokens()
        t = self.config.tokens
        system_reserve = min(
            int(t.pack_system_reserve_tokens),
            max(256, max_input // 4),
        )
        buffer = min(int(t.pack_buffer_tokens), max(256, max_input // 16))
        pool = max(4_096, max_input - system_reserve - buffer)

        weights: Tuple[int, int, int, int, int] = t.slot_pack_weights
        if len(weights) != 5 or sum(weights) <= 0:
            weights = (2, 1, 2, 2, 8)
        w_plan, w_past, w_scratch, w_inp, w_recent = weights
        tw = int(w_plan + w_past + w_scratch + w_inp + w_recent)

        plan = min(t.slot_plan_tokens, max(512, pool * w_plan // tw))
        past = min(t.slot_past_tokens, max(256, pool * w_past // tw))
        scratch = min(t.slot_scratchpad_tokens, max(512, pool * w_scratch // tw))
        inp = min(t.slot_input_tokens, max(512, pool * w_inp // tw))

        fixed = plan + past + scratch + inp
        recent = max(2_048, min(t.slot_recent_tokens, pool - fixed))

        overflow = fixed + recent - pool
        if overflow > 0:
            plan = max(512, plan - overflow // 4)
            scratch = max(512, scratch - overflow // 4)
            inp = max(512, inp - overflow // 4)
            past = max(256, past - overflow // 4)
            fixed = plan + past + scratch + inp
            recent = max(2_048, min(t.slot_recent_tokens, pool - fixed))

        caps = {
            "plan": plan,
            "past": past,
            "scratch": scratch,
            "recent": recent,
            "input": inp,
        }
        phase = (phase_name or self._current_phase_name or "").strip()
        if phase == "orientation":
            lim = self.config.text_limits
            report_slot = int(
                lim.orientation_report_slot_tokens
                or lim.planning_orientation_prompt
                or lim.orientation_summary_store
                or 16_000
            )
            step_slot = int(lim.orientation_current_step_slot_tokens or 8_000)
            recent_slot = int(lim.orientation_recent_slot_tokens or 8_000)
            caps["scratch"] = max(caps["scratch"], report_slot)
            caps["input"] = max(caps["input"], step_slot)
            caps["recent"] = max(caps["recent"], recent_slot)
            caps["plan"] = 0
            caps["past"] = 0
        elif phase == "planning":
            caps["scratch"] = 0
            caps["plan"] = 0
            caps["past"] = 0
            caps["recent"] = 0
        return caps

    def _build_orientation_slot_payload(
        self,
        system_content: str,
        messages: List[Dict[str, Any]],
        *,
        slot_report_tokens: int,
        slot_recent_tokens: int,
        slot_input_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Orientation layout: System → [Current Orientation Report] → recent chat → [Current Step].

        Report = cumulative on disk. Recent = prior assistant/tool bundles (token-capped).
        Current Step = latest checkpoint + optional in-flight SQL/tool turn.
        """
        result: List[Dict[str, Any]] = [_system_msg(system_content)]

        report_text = (self._scratchpad_slot or "").strip()
        if report_text:
            report_msg = _user_msg(f"[Current Orientation Report]\n\n{report_text}")
            if _estimate_full_payload_tokens([report_msg]) > slot_report_tokens:
                report_msg = _truncate_message_to_tokens(report_msg, slot_report_tokens)
            result.append(report_msg)
        else:
            result.append(
                _user_msg(
                    "[Current Orientation Report]\n\n"
                    "(empty — append measured sections via `orientation_report`.)"
                )
            )

        current_start = _current_input_bundle_start(messages) if messages else 0
        if (
            current_start > 0
            and messages[current_start].get("role") == "assistant"
            and self._is_orientation_current_step_user_msg(messages[current_start - 1])
        ):
            current_start -= 1

        current_bundle = list(messages[current_start:]) if messages else []
        older = messages[:current_start]

        recent: List[Dict[str, Any]] = []
        if slot_recent_tokens > 0 and older:
            recent_tokens = 0
            recent_bundles: List[List[Dict[str, Any]]] = []
            for bundle in reversed(_split_message_bundles(older)):
                if bundle and self._is_orientation_current_step_user_msg(bundle[0]):
                    continue
                bt = _estimate_full_payload_tokens(bundle)
                if recent_tokens + bt > slot_recent_tokens:
                    break
                recent_bundles.insert(0, bundle)
                recent_tokens += bt
            for bundle in recent_bundles:
                recent.extend(bundle)

            evict_count = len(older) - len(recent)
            if evict_count > 0:
                del messages[:evict_count]

        if recent:
            result.append(
                _user_msg(
                    "[Recent orientation steps]\n\n"
                    "(Prior assistant/tool turns below; SQL may be truncated in older steps.)"
                )
            )
            result.extend(recent)

        if current_bundle:
            cur_tok = _estimate_full_payload_tokens(current_bundle)
            if cur_tok > slot_input_tokens and len(current_bundle) == 1:
                current_bundle = [
                    _truncate_message_to_tokens(current_bundle[0], slot_input_tokens)
                ]
            result.extend(current_bundle)
        else:
            from researchpkg.forensic_llm.prompts.orientation import (
                ORIENTATION_CURRENT_STEP_FRESH,
            )

            result.append(_user_msg(ORIENTATION_CURRENT_STEP_FRESH))

        return result

    def _build_slot_payload(
        self,
        system_content: str,
        messages: List[Dict[str, Any]],
        *,
        phase_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Assemble the 5-slot payload sent to the LLM on every call.

        Prefix-cache layout — ordered from least to most frequently mutated:
          0  System prefix   — system_content (STATIC: fraud catalogue + schema + rules)
          1  Plan state      — self._plan_slot (updated 5× at phase boundaries only)
          2  Past memory     — self._past_memory (structured drop summary, no LLM call)
          3  Recent messages — verbatim tail of `messages` ≤ slot_recent_tokens (append-only)
          4  Scratchpad      — self._scratchpad_slot (lazy: rebuilt only when dirty)
          5  Current input   — last message in `messages`, ≤ slot_input_tokens (always new)

        Scratchpad is placed AFTER the tail so that a scratchpad write only
        invalidates the scratchpad + current-input suffix. The append-only tail
        (recent slot) stays a stable prefix and is not rebuilt when scratchpad flips.
        """
        phase = phase_name or self._current_phase_name
        caps = self._effective_slot_caps(phase)
        slot_plan = caps["plan"]
        slot_past = caps["past"]
        slot_scratchpad = caps["scratch"]
        slot_recent = caps["recent"]
        slot_input = caps["input"]

        # Lazily rebuild working-notes slot (orientation report vs scratchpad).
        if phase == "orientation":
            if self._orientation_report_dirty:
                from researchpkg.forensic_llm.artefacts import (
                    load_orientation_report_text,
                )

                report_text = load_orientation_report_text(self._run_dir)
                self._scratchpad_slot = report_text.strip()
                self._orientation_report_dirty = False
        elif self._scratchpad_dirty:
            sp_text = get_scratchpad_text()
            if sp_text and sp_text != "(scratchpad is empty)":
                sp_msg = _user_msg(f"[Current Scratchpad]\n\n{sp_text}")
                if _estimate_full_payload_tokens([sp_msg]) > slot_scratchpad:
                    sp_msg = _truncate_message_to_tokens(sp_msg, slot_scratchpad)
                self._scratchpad_slot = _msg_text_content(sp_msg)
            else:
                self._scratchpad_slot = ""
            self._scratchpad_dirty = False

        if phase == "orientation":
            return self._build_orientation_slot_payload(
                system_content,
                messages,
                slot_report_tokens=slot_scratchpad,
                slot_recent_tokens=slot_recent,
                slot_input_tokens=slot_input,
            )

        result: List[Dict[str, Any]] = [_system_msg(system_content)]

        if not messages:
            return result

        # --- Slot 1: Plan state ------------------------------------------------
        if self._plan_slot:
            plan_msg = _user_msg(f"[Investigation Plan State]\n\n{self._plan_slot}")
            if _estimate_full_payload_tokens([plan_msg]) > slot_plan:
                plan_msg = _truncate_message_to_tokens(plan_msg, slot_plan)
            result.append(plan_msg)

        # --- Slot 2: Past memory -----------------------------------------------
        if self._past_memory:
            past_msg = _user_msg(f"[Past Investigation Memory]\n\n{self._past_memory}")
            if _estimate_full_payload_tokens([past_msg]) > slot_past:
                past_msg = _truncate_message_to_tokens(past_msg, slot_past)
            result.append(past_msg)

        # --- Slot 5: Current input (last bundle, truncated when safe) ----------
        current_start = _current_input_bundle_start(messages)
        current_bundle = list(messages[current_start:])
        cur_tok = _estimate_full_payload_tokens(current_bundle)
        if cur_tok > slot_input and len(current_bundle) == 1:
            current_bundle = [
                _truncate_message_to_tokens(current_bundle[0], slot_input)
            ]

        # --- Slot 3: Recent messages (tail, excluding current bundle) ----------
        older = messages[:current_start]
        recent: List[Dict[str, Any]] = []
        if phase != "orientation" and slot_recent > 0:
            recent_tokens = 0
            recent_bundles: List[List[Dict[str, Any]]] = []
            for bundle in reversed(_split_message_bundles(older)):
                bt = _estimate_full_payload_tokens(bundle)
                if recent_tokens + bt > slot_recent:
                    break
                recent_bundles.insert(0, bundle)
                recent_tokens += bt
            for bundle in recent_bundles:
                recent.extend(bundle)

            evict_count = len(older) - len(recent)
            if evict_count > 0:
                to_evict = older[:evict_count]
                self._evict_to_past_memory(to_evict)
                del messages[:evict_count]

        result.extend(recent)

        # --- Slot 4: Working notes (scratchpad; orientation uses dedicated builder) ---
        if self._scratchpad_slot:
            if phase != "planning":
                result.append(
                    _user_msg(f"[Current Scratchpad]\n\n{self._scratchpad_slot}")
                )

        result.extend(current_bundle)
        return result

    def _evict_to_past_memory(self, old_messages: List[Dict[str, Any]]) -> None:
        """
        MemGPT-style archival eviction (Packer et al., 2023).

        Instead of storing a lossy free-text summary, this method:
        1. Extracts *structured* evidence entries from evicted turns:
           - report_suspicion calls → EvidenceEntry(status=CONFIRMED, doc_ids…)
           - scratchpad hypothesis lines → EvidenceEntry(status from H-prefix)
        2. Appends them to self._evidence_register (persistent archival store).
        3. Renders the register as a compact table for self._past_memory (slot 2),
           so every subsequent LLM call sees a fresh lossless evidence summary
           rather than an ever-growing free-text blob.

        The scratchpad (slot 4) captures full SQL reasoning; the register captures
        structured conclusions.  Together they form the agent's "archival memory".
        """
        if not old_messages:
            return

        slot_past = getattr(self.config, "slot_past_tokens", 8_000)
        evicted_steps = sum(1 for m in old_messages if m.get("role") == "assistant")

        # ── 1. Extract structured entries from evicted messages ────────────────
        scheme_hint = getattr(_THREAD_SCHEME, "name", None) or "unknown"

        # 1a. Rescue report_suspicion calls
        for msg in old_messages:
            content = _msg_text_content(msg)
            if not content:
                continue
            # Find report_suspicion JSON payloads embedded in assistant or tool messages
            for json_match in re.finditer(
                r'"scheme_type"\s*:\s*"([^"]+)"[^}]*"document_id"\s*:\s*"([0-9a-f\-]{36})"',
                content,
                re.IGNORECASE | re.DOTALL,
            ):
                scheme_found = json_match.group(1)
                doc_id = json_match.group(2)
                # Check if this doc_id already has an entry
                existing_ids = {
                    did for e in self._evidence_register for did in e.document_ids
                }
                if doc_id not in existing_ids:
                    # Extract confidence if present
                    conf_m = re.search(
                        r'"confidence"\s*:\s*([0-9.]+)',
                        content[
                            max(0, json_match.start() - 200) : json_match.end() + 200
                        ],
                    )
                    rationale_m = re.search(
                        r'"rationale"\s*:\s*"([^"]{0,120})"',
                        content[
                            max(0, json_match.start() - 200) : json_match.end() + 200
                        ],
                    )
                    self._evidence_register.append(
                        EvidenceEntry(
                            scheme=scheme_found,
                            hypothesis="(from report_suspicion call)",
                            status="CONFIRMED",
                            key_finding=rationale_m.group(1) if rationale_m else "",
                            document_ids=[doc_id],
                            step_added=self._step_num,
                        )
                    )

        # 1b. Parse hypothesis status lines from scratchpad-style text in messages
        hypothesis_re = re.compile(
            r"^\s*[-*]?\s*\[?(H\d+)\]?\s+(.+?)\s+[-–—]\s+(CONFIRMED|RULED[_ ]OUT|INCONCLUSIVE|OPEN)",
            re.IGNORECASE | re.MULTILINE,
        )
        for msg in old_messages:
            content = _msg_text_content(msg)
            if not content:
                continue
            for hm in hypothesis_re.finditer(content):
                h_label = hm.group(1)
                h_text = truncate_text_to_tokens(
                    hm.group(2).strip(),
                    self.config.text_limits.evidence_hypothesis_line,
                    side=TruncationSide.HEAD,
                )
                h_status = hm.group(3).upper().replace(" ", "_")
                # De-duplicate by (scheme, hypothesis label)
                already = any(
                    e.scheme == scheme_hint and h_label in e.hypothesis
                    for e in self._evidence_register
                )
                if not already:
                    self._evidence_register.append(
                        EvidenceEntry(
                            scheme=scheme_hint,
                            hypothesis=f"[{h_label}] {h_text}",
                            status=h_status,
                            key_finding="",
                            step_added=self._step_num,
                        )
                    )

        # ── 2. Render register as compact table for past_memory slot ───────────
        _render_limit = 60  # max rows to render (most recent first)
        recent_entries = self._evidence_register[-_render_limit:]
        if recent_entries:
            header = (
                "| id | scheme | status | finding (truncated) | doc_ids |\n"
                "|---|---|---|---|---|"
            )
            rows = [e.to_table_row() for e in recent_entries]
            table = header + "\n" + "\n".join(rows)
            total = len(self._evidence_register)
            summary = (
                f"[MemGPT Evidence Register — {total} entries total, "
                f"showing latest {len(recent_entries)}]\n\n{table}"
            )
        else:
            summary = (
                f"[Evicted {evicted_steps} assistant turn(s) — "
                "no structured evidence entries extracted yet.]"
            )

        # Truncate to fit token budget
        if count_tokens(summary) > slot_past:
            target = len(summary)
            while target > 200 and count_tokens(summary[:target]) > slot_past:
                target = target * 3 // 4
            summary = (
                summary[:target].rstrip() + "\n[... truncated to fit memory budget ...]"
            )

        self._past_memory = summary

        # Persist register to disk so workers and parent can share it
        try:
            register_path = self._run_dir / "evidence_register.json"
            register_path.write_text(
                json.dumps(
                    [e.model_dump() for e in self._evidence_register],
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

        log.info(
            "Evicted %d old message(s) → evidence_register now %d entries; "
            "past_memory ~%d tokens",
            len(old_messages),
            len(self._evidence_register),
            count_tokens(self._past_memory),
        )

    def _update_plan_slot(self, plan: Optional["InvestigationPlan"] = None) -> None:
        """
        Rebuild the plan-state slot from the current InvestigationPlan and all
        collected PhaseSummaries.  Called at phase boundaries so the slot always
        reflects the latest validated findings.
        """
        slot_plan = self.config.slot_plan_tokens
        parts: List[str] = []

        if plan is not None:
            try:
                plan_json = json.dumps(plan.model_dump(exclude_none=True), indent=2)
                parts.append(
                    f"## Current Investigation Plan\n```json\n{plan_json}\n```"
                )
            except Exception:
                pass

        phase_ctx = self._build_phase_summaries_context()
        if phase_ctx:
            parts.append(phase_ctx)

        content = "\n\n".join(parts)
        if count_tokens(content) > slot_plan:
            target = max(200, len(content))
            while target > 200 and count_tokens(content[:target]) > slot_plan:
                target = target * 3 // 4
            content = content[:target].rstrip() + "\n[... plan slot truncated ...]"
        self._plan_slot = content

    def _hard_trim_to_fit(
        self,
        payload: List[Dict[str, Any]],
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Safety-net last-resort: keep system (index 0) + keep as many trailing
        messages as fit.  Should rarely be triggered after 4-slot assembly.
        """
        if len(payload) <= 1:
            return payload
        system_only = payload[:1]
        sys_tok = _estimate_full_payload_tokens(system_only)
        budget = max_tokens - sys_tok
        tail: List[Dict[str, Any]] = []
        tail_tok = 0
        tail_bundles: List[List[Dict[str, Any]]] = []
        for bundle in reversed(_split_message_bundles(payload[1:])):
            bt = _estimate_full_payload_tokens(bundle)
            if tail_tok + bt > budget:
                break
            tail_bundles.insert(0, bundle)
            tail_tok += bt
        for bundle in tail_bundles:
            tail.extend(bundle)
        return system_only + tail if tail else system_only

    # ------------------------------------------------------------------
    # LLM call wrapper
    # ------------------------------------------------------------------

    def _tools_for_phase(
        self, phase_name: str, *, force_finish: bool
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Tool schemas and tool_choice for the current phase."""
        from researchpkg.forensic_llm.tool_defs import (
            COMPLETE_ORIENTATION_TOOL,
            FINISH_TOOL,
            ORIENTATION_TOOL_NAMES,
            tools_from_names,
        )

        if force_finish:
            if phase_name == "orientation":
                return [COMPLETE_ORIENTATION_TOOL], "required"
            return [FINISH_TOOL], "required"
        if phase_name == "orientation":
            return tools_from_names(ORIENTATION_TOOL_NAMES), "auto"
        return self._tools, "auto"

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        system_content: str,
        force_finish: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        max_completion_tokens: Optional[int] = None,
        phase_name: str = "",
    ) -> LLMResponse:
        """
        Call the LLM via the 4-slot context budget.

        Parameters
        ----------
        messages     : conversation history (mutable; old messages are evicted
                       in-place to the past-memory slot when recent window is full).
        system_content: system prompt string (assembled fresh each call).
        force_finish : if True, only the finish_investigation tool is offered.
        """
        payload = self._build_slot_payload(
            system_content=system_content,
            messages=messages,
            phase_name=phase_name,
        )

        completion_reserve = (
            int(max_completion_tokens)
            if max_completion_tokens is not None
            else int(self.config.llm.max_tokens_per_step)
        )
        # Hard-cap safety net: if assembled payload still exceeds limit, drop from the
        # front of the non-system, non-slot messages (should be very rare after slotting).
        max_input = self._get_max_input_tokens(completion_reserve=completion_reserve)
        est = self._inflated_token_estimate(payload)
        if est > max_input:
            log.warning(
                "Slot payload still over limit (~%d > %d prompt tokens; "
                "window=%d effective=%d completion_reserve=%d); hard-trimming",
                est,
                max_input,
                self.config.model_context_window,
                self.config.tokens.effective_context_window(),
                completion_reserve,
            )
            payload = self._hard_trim_to_fit(payload, max_input)
            while (
                len(payload) > 1 and self._inflated_token_estimate(payload) > max_input
            ):
                payload = self._hard_trim_to_fit(payload, max(1024, max_input // 2))

        # Persist the exact payload that will be sent (for memory.md).
        self._last_llm_payload = list(payload)

        if self._use_native_tools:
            tools, tool_choice = self._tools_for_phase(
                phase_name, force_finish=force_finish
            )
        else:
            tools = None
            tool_choice = "required" if force_finish else "auto"

        try:
            resp = self.client.chat(
                messages=payload,
                tools=tools,
                tool_choice=tool_choice,
                response_schema=response_schema,
                max_tokens=completion_reserve,
            )
            # Persist a per-step LLM call log for debugging/audit.
            try:
                import json

                step_tag = f"step_{self._step_num:04d}"
                path = self._calls_steps_dir / f"{step_tag}.json"
                log_payload: Dict[str, Any] = {
                    "step_number": self._step_num,
                    "force_finish": force_finish,
                    "tool_choice": tool_choice,
                    "use_native_tools": self._use_native_tools,
                    "tools": tools,
                    "request": payload,
                    "response": {
                        "content": resp.content,
                        "tool_calls": [tc.__dict__ for tc in resp.tool_calls],
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "finish_reason": resp.finish_reason,
                        "reasoning": resp.reasoning,
                    },
                }
                path.write_text(
                    json.dumps(log_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning(
                    "Could not persist LLM call log for step %d: %s",
                    self._step_num,
                    exc,
                )

            return resp
        except Exception as exc:
            # Surface the error to the caller; _call_with_retries in the LLM client
            # has already exhausted all retries for transient errors.  Silently
            # downgrading the session to ReAct mode here would corrupt the message
            # history (role="tool" results without matching tool_call_ids) and make
            # failures invisible.  Let the caller decide how to terminate.
            raise

    # ------------------------------------------------------------------
    # Forced final extraction
    # ------------------------------------------------------------------

    def _forced_finish(
        self, report: ForensicReport, messages: List[Dict[str, Any]]
    ) -> None:
        """
        When the loop exits without a finish call, ask the model to produce
        its output one final time (no tools allowed).
        """
        system_content = build_system_prompt(
            task=self.config.task,
            scratchpad_text=get_scratchpad_text(),
            is_worker=self._worker_run_dir is not None,
        )
        phase_ctx = self._build_phase_summaries_context()
        prompt = (
            (phase_ctx + "\n\n---\n\n" if phase_ctx else "")
            + "The investigation loop has ended. "
            "Based on everything you have investigated so far "
            "(use the per-scheme summaries above as your primary reference), "
            "produce a JSON object with two keys:\n"
            '1. `"suspicion_list"` – array of suspicion objects\n'
            '2. `"narrative"` – markdown report\n\n'
            "Respond with ONLY the JSON object, no other text."
        )
        messages_final = [_system_msg(system_content)] + messages + [_user_msg(prompt)]
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.client.chat(messages=messages_final, tools=None)
                self.budget.record_llm_call(
                    resp.prompt_tokens,
                    resp.completion_tokens,
                    reasoning_tokens=resp.reasoning_tokens,
                )
                content_raw = resp.content
                if isinstance(content_raw, list):
                    content = " ".join(
                        block.get("text", "")
                        for block in content_raw
                        if isinstance(block, dict) and block.get("type") == "text"
                    ).strip()
                else:
                    content = str(content_raw or "").strip()

                # Try JSON extraction
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    obj = json.loads(m.group(0))
                    report.suspicion_list = _parse_suspicion_list(
                        obj.get("suspicion_list", [])
                    )
                    report.narrative = obj.get("narrative", content)
                else:
                    report.narrative = content
                # Backfill from live detections if we have none
                if not report.suspicion_list and self._live_detections:
                    report.suspicion_list = [
                        SuspicionItem(
                            document_id=it["document_id"],
                            scheme_type=(
                                SchemeType(it["scheme_id"])
                                if it.get("scheme_id") in _SCHEME_NAMES
                                else SchemeType.UNKNOWN
                            ),
                            confidence=float(it.get("confidence", 0.5)),
                        )
                        for it in self._live_detections
                        if it.get("document_id")
                    ]
                self._flush_live_detections(report.suspicion_list)
                return
            except Exception as exc:
                last_exc = exc
                err_lower = str(exc).lower()
                if "timeout" in err_lower or "timed out" in err_lower:
                    log.warning(
                        "Forced finish attempt %d failed (timeout), retrying...",
                        attempt + 1,
                    )
                else:
                    log.error("Forced finish extraction failed: %s", exc)
                    report.narrative = (
                        "(Investigation completed without a final report.)"
                    )
                    return
        log.error("Forced finish extraction failed after 3 attempts: %s", last_exc)
        report.narrative = "(Investigation completed without a final report.)"

    # ------------------------------------------------------------------
    # Streaming trace (optional)
    # ------------------------------------------------------------------

    # Regex to detect and strip inline base64 data URLs from trace output.
    _B64_DATA_URL_RE = re.compile(r"data:[a-zA-Z0-9+/.-]+;base64,[A-Za-z0-9+/=]{20,}")

    @staticmethod
    def _scrub_base64(obj: Any) -> Any:
        """
        Recursively replace inline base64 data URLs in a serialised structure
        with a short placeholder.  This keeps audit_trace_stream.ndjson
        human-readable (and small) without losing the file-path reference.
        """
        if isinstance(obj, str):
            return ForensicAgent._B64_DATA_URL_RE.sub(
                "[base64-image-data omitted — see plots/]", obj
            )
        if isinstance(obj, dict):
            return {k: ForensicAgent._scrub_base64(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ForensicAgent._scrub_base64(v) for v in obj]
        return obj

    def _emit_step(self, step: AgentStep) -> None:
        """Append one step as a JSON line to the stream trace file (if open)
        and print a human-readable version to the terminal."""
        if self._stream_trace_file is None:
            return
        try:
            raw = self._scrub_base64(step.model_dump())
            raw["llm_tokens_budgeted"] = step.llm_tokens_budgeted
            self._stream_trace_file.write(json.dumps(raw, default=str) + "\n")
            self._stream_trace_file.flush()
        except Exception as exc:
            log.debug("Stream trace write failed: %s", exc)

        # Stream the same step to the terminal in human-readable form.
        # Suppressed for subagent workers (stream_trace_terminal=False) to avoid
        # interleaved output from 10 concurrent streams.
        if getattr(self.config, "stream_trace_terminal", True):
            try:
                from researchpkg.forensic_llm.trace_viewer import (
                    format_step,
                )

                terminal_truncate = 1200  # keep tool results readable but bounded
                text = format_step(raw, result_truncate=terminal_truncate)
                print(text, file=sys.stderr, flush=True)
            except Exception as exc:
                log.debug("Stream to terminal failed: %s", exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _merge_live_detections_into_report(self, report: ForensicReport) -> None:
        """
        Programmatic reconciliation of JE-level flags across tasks.

        Workers stream detections into ``_live_detections`` via ``report_suspicion``;
        merge again here so ``ForensicReport.suspicion_list`` matches the union before
        ``investigation_report.json`` assembly.         Key ``(document_id, scheme_id)`` — first seen wins per key.
        """
        if not self._live_detections:
            return
        merged_by_key: Dict[Tuple[str, str], SuspicionItem] = {}

        def _norm(v: Any) -> str:
            return str(v).strip().lower()

        for s in report.suspicion_list or []:
            if not s.document_id:
                continue
            scheme_id = s.scheme_type.value if s.scheme_type else "unknown"
            key = (_norm(s.document_id), _norm(scheme_id))
            if key not in merged_by_key:
                merged_by_key[key] = s

        for d in self._live_detections:
            doc_id = d.get("document_id")
            if not doc_id:
                continue
            scheme_id_raw = d.get("scheme_id", "unknown")
            scheme_id = _norm(scheme_id_raw)
            scheme_type = (
                SchemeType(scheme_id)
                if scheme_id in _SCHEME_NAMES
                else SchemeType.UNKNOWN
            )
            key = (_norm(doc_id), scheme_id)
            if key not in merged_by_key:
                merged_by_key[key] = SuspicionItem(
                    document_id=doc_id,
                    scheme_type=scheme_type,
                    confidence=_EVAL_FLAG_CONFIDENCE,
                )

        before_n = len(report.suspicion_list or [])
        after_n = len(merged_by_key)
        report.suspicion_list = list(merged_by_key.values())
        if before_n != after_n:
            log.info(
                "Merged live detections into report suspicion_list: %d -> %d",
                before_n,
                after_n,
            )

    def _save_report(self, report: ForensicReport) -> None:
        """
        Write all investigation artefacts into the run directory.

        Directory layout
        ----------------
        {run_dir}/
          narrative.md              Full forensic narrative report (markdown)
          suspicion_list.json       Ranked suspicion items (machine-readable)
          audit_trace.json          Complete step-by-step tool call trace
          budget_summary.json       Token accounting and budget statistics
          scratchpad.md             Running investigation notes
          run_manifest.json         Run metadata: config, timing, counts
          plots/                    Matplotlib charts produced by the plot tool
            *.png
        """
        out = self._run_dir

        self._merge_live_detections_into_report(report)

        # ---- 1. Narrative markdown ----------------------------------------
        narrative_path = out / "narrative.md"
        narrative_path.write_text(
            report.narrative or "(No narrative produced.)",
            encoding="utf-8",
        )

        # ---- 2. Suspicion list --------------------------------------------
        ranked = sorted(
            report.suspicion_list,
            key=lambda s: (
                s.scheme_type.value if s.scheme_type else "",
                s.document_id or "",
            ),
        )
        susp_path = out / "suspicion_list.json"
        susp_path.write_text(
            json.dumps([s.model_dump() for s in ranked], indent=2, default=str),
            encoding="utf-8",
        )

        # ---- 2b. Detections list (document_id, scheme_id) for evaluation ----
        # JSON object with detections array so evaluation scripts can join against ground truth.
        detections_list = [
            _detection_record(
                s.document_id,
                s.scheme_type.value if s.scheme_type else "unknown",
            )
            for s in report.suspicion_list
            if s.document_id
        ]
        detections_payload = {
            "detections": detections_list,
            "count": len(detections_list),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        (out / "detections.json").write_text(
            json.dumps(detections_payload, indent=2, default=str),
            encoding="utf-8",
        )

        if report.worker_summaries:
            (out / "worker_summaries.json").write_text(
                json.dumps(
                    {
                        worker_id: summary.model_dump(mode="json")
                        for worker_id, summary in report.worker_summaries.items()
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        if report.coverage_ledger is not None:
            (out / "coverage_ledger.json").write_text(
                json.dumps(
                    report.coverage_ledger.model_dump(mode="json"),
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        if self._phase_summaries:
            (out / "worker_phase_summaries.json").write_text(
                json.dumps(
                    {
                        scheme: summary.model_dump(mode="json")
                        for scheme, summary in self._phase_summaries.items()
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        # ---- 3. Full audit trace ------------------------------------------
        trace_path = out / "audit_trace.json"
        trace_path.write_text(
            json.dumps(
                self._scrub_base64(json.loads(report.model_dump_json())),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # ---- 4. Budget summary -------------------------------------------
        budget_summary = self.budget.summary()
        budget_summary["run_id"] = report.run_id
        budget_summary["model"] = report.model
        budget_summary["task"] = report.task
        budget_summary["configured_max_tokens"] = self._global_token_budget
        budget_summary["configured_global_max_tokens"] = self._global_token_budget
        budget_summary["run_prompt_token_cap"] = self.budget.max_tokens
        if (
            self.config.parallel_scheme_execution
            and self.config.task == "full"
            and self._worker_run_dir is None
        ):
            budget_summary[
                "parallel_token_budget_slots"
            ] = self.config.parallel_orchestrator_token_slots
            budget_summary[
                "parallel_orchestrator_weight"
            ] = self.config.parallel_orchestrator_token_slots
        budget_summary["sql_calls_total"] = self._sql_call_count
        budget_summary["code_interpreter_calls"] = self._code_interpreter_call_count
        budget_summary["effective_sql_calls"] = self._effective_sql_call_total
        budget_summary["orchestrator_sql_calls"] = self._orchestrator_sql_count
        budget_summary[
            "orchestrator_code_interpreter_calls"
        ] = self._orchestrator_code_interpreter_count
        budget_summary[
            "orchestrator_effective_sql_calls"
        ] = self._effective_orchestrator_sql_total
        budget_summary["sql_errors_total"] = self._sql_error_count
        (
            budget_summary["orchestrator_llm_errors_total"],
            budget_summary["orchestrator_llm_errors_unrecovered"],
        ) = self._llm_error_counts_from_client()
        budget_summary["llm_errors_total"] = budget_summary[
            "orchestrator_llm_errors_total"
        ]
        budget_summary["llm_errors_unrecovered"] = budget_summary[
            "orchestrator_llm_errors_unrecovered"
        ]
        budget_summary["sql_error_rate"] = round(
            self._sql_error_count / max(1, self._sql_call_count), 4
        )
        budget_summary["sql_zero_row_calls"] = self._sql_zero_row_count
        budget_summary["sql_zero_row_rate"] = round(
            self._sql_zero_row_count / max(1, self._sql_call_count), 4
        )
        budget_summary["tool_call_counts"] = dict(
            sorted(self._tool_call_counts.items(), key=lambda x: x[1], reverse=True)
        )

        # Per-child budget breakdowns (scheme subagents + hypothesis task workers).
        child_prefixes: List[str] = []
        subagent_budgets = self._load_child_budget_summaries(out / "subagents")
        if subagent_budgets:
            self._rollup_child_budget_into_summary(
                budget_summary, subagent_budgets, "subagents"
            )
            child_prefixes.append("subagents")
        task_budgets = self._load_child_budget_summaries(out / "tasks")
        if task_budgets:
            self._rollup_child_budget_into_summary(
                budget_summary, task_budgets, "hypothesis_tasks"
            )
            child_prefixes.append("hypothesis_tasks")
        if child_prefixes:
            self._apply_grand_budget_totals(budget_summary, child_prefixes)
            if "subagents" in child_prefixes:
                budget_summary["orchestrator_sql_calls"] = max(
                    0,
                    int(budget_summary.get("sql_calls_total", 0) or 0)
                    - int(budget_summary.get("subagents_sql_calls", 0) or 0)
                    - int(budget_summary.get("hypothesis_tasks_sql_calls", 0) or 0),
                )

        rs_path = out / "run_stats.json"
        if rs_path.is_file():
            try:
                rs = json.loads(rs_path.read_text(encoding="utf-8"))
                budget_summary["orientation_tokens"] = int(
                    rs.get("orientation_tokens", 0) or 0
                )
                budget_summary["hypothesis_task_tokens"] = int(
                    rs.get("worker_tokens", 0) or 0
                )
                budget_summary["orchestrator_tokens_excl_tasks"] = int(
                    rs.get("orchestrator_tokens", 0) or 0
                )
            except Exception:
                pass

        (out / "budget_summary.json").write_text(
            json.dumps(budget_summary, indent=2),
            encoding="utf-8",
        )

        # ---- 5. Scratchpad --------------------------------------------------
        sp_text = get_scratchpad_text()
        (out / "scratchpad.md").write_text(sp_text, encoding="utf-8")

        # ---- 6. Scheme reports ------------------------------------------------
        if report.scheme_reports:
            schemes_path = out / "scheme_reports.json"
            schemes_path.write_text(
                json.dumps(
                    [s.model_dump() for s in report.scheme_reports],
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        # ---- 7. Investigation plan (Plan-Execute-Reflect only) ---------------
        if report.investigation_plan and report.investigation_plan.phases:
            plan_path = out / "investigation_plan.json"
            plan_path.write_text(
                json.dumps(
                    report.investigation_plan.model_dump(),
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        # ---- 8. Run manifest -----------------------------------------------
        manifest = {
            "run_id": report.run_id,
            "run_dir": str(out),
            "model": report.model,
            "provider": self.config.llm.provider,
            "strategy": report.strategy,
            "task": report.task,
            "config": {
                "llm": {
                    "provider": self.config.llm.provider,
                    "base_url": self.config.llm.base_url,
                    "model": self.config.llm.model,
                    "anthropic_model": self.config.llm.anthropic_model,
                    "temperature": self.config.llm.temperature,
                    "top_p": self.config.llm.top_p,
                    "max_tokens_per_step": self.config.llm.max_tokens_per_step,
                    "use_native_tools": self.config.llm.use_native_tools,
                    "disable_thinking": self.config.llm.disable_thinking,
                    "request_timeout": self.config.llm.request_timeout,
                    "max_retries": self.config.llm.max_retries,
                },
                "budget": {
                    "max_tokens": self.config.budget.max_tokens,
                    "max_steps": self.config.budget.max_steps,
                    "warn_threshold": self.config.budget.warn_threshold,
                    "stop_threshold": self.config.budget.stop_threshold,
                    "global_max_tokens": self._global_token_budget,
                    "run_prompt_token_cap": self.budget.max_tokens,
                },
                "investigation": {
                    "task": self.config.task,
                    "seed": self.config.seed,
                    "n_agents": self.config.n_agents,
                    "output_dir": self.config.output_dir,
                    "enable_grep": self.config.enable_grep,
                    "grep_root": self.config.grep_root,
                    "enable_graph_tools": self.config.enable_graph_tools,
                    "graph_depth": self.config.graph_depth,
                    "orientation_budget_fraction": self.config.orientation_budget_fraction,
                    "sql_min_quota_base": self.config.sql_min_quota_base,
                    "sql_min_quota_per_core": self.config.sql_min_quota_per_core,
                    "min_budget_fraction_for_finish": self.config.min_budget_fraction_for_finish,
                    "stream_trace": self.config.stream_trace,
                },
                # Do not record DB password/credentials in manifests.
                "database": {
                    "host": self.config.database.host,
                    "port": self.config.database.port,
                    "database": self.config.database.database,
                    "user": self.config.database.user,
                    "statement_timeout_ms": self.config.database.statement_timeout_ms,
                    "default_max_rows": self.config.database.default_max_rows,
                    "hard_max_rows": self.config.database.hard_max_rows,
                },
            },
            "started_at": report.started_at.isoformat() if report.started_at else None,
            "completed_at": report.completed_at.isoformat()
            if report.completed_at
            else None,
            "duration_seconds": (
                (report.completed_at - report.started_at).total_seconds()
                if report.completed_at and report.started_at
                else None
            ),
            "termination_reason": (
                report.termination_reason
                or ("budget_exhausted" if report.budget_exhausted else "completed")
            ),
            "error_message": report.error_message,
            "error_traceback": report.error_traceback,
            "steps_taken": report.steps_taken,
            "n_suspicions": len(report.suspicion_list),
            "n_high_severity": sum(1 for s in report.suspicion_list if s.severity >= 4),
            "sql_calls_total": self._sql_call_count,
            "code_interpreter_calls": self._code_interpreter_call_count,
            "effective_sql_calls": self._effective_sql_call_total,
            "sql_errors_total": self._sql_error_count,
            "llm_errors_total": report.llm_errors_total,
            "llm_errors_unrecovered": report.llm_errors_unrecovered,
            "sql_error_rate": round(
                self._sql_error_count / max(1, self._sql_call_count), 4
            ),
            "sql_zero_row_calls": self._sql_zero_row_count,
            "sql_zero_row_rate": round(
                self._sql_zero_row_count / max(1, self._sql_call_count), 4
            ),
            "tool_call_counts": dict(
                sorted(self._tool_call_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "budget_exhausted": report.budget_exhausted,
            "token_budget": self._global_token_budget,
            "prompt_tokens": report.total_tokens_input,
            "completion_tokens": report.total_tokens_output,
            "reasoning_tokens": report.total_tokens_reasoning,
            "budget_counted_tokens": report.budget_counted_tokens,
            "total_billed_tokens": report.total_tokens_input
            + report.total_tokens_output,
            "budget_fraction_used": round(
                report.budget_counted_tokens / max(1, self._global_token_budget), 4
            ),
            "scheme_types_found": sorted(
                {s.scheme_type.value for s in report.suspicion_list}
            ),
            "artefacts": {
                "narrative": "narrative.md",
                "suspicion_list": "suspicion_list.json",
                "detections": "detections.json",
                "audit_trace": "audit_trace.json",
                "budget_summary": "budget_summary.json",
                "scratchpad": "scratchpad.md",
                "scratchpad_steps": "scratchpad_steps/",
                "plots_dir": "plots/",
                **(
                    {"worker_summaries": "worker_summaries.json"}
                    if (self._run_dir / "worker_summaries.json").exists()
                    else {}
                ),
                **(
                    {"coverage_ledger": "coverage_ledger.json"}
                    if (self._run_dir / "coverage_ledger.json").exists()
                    else {}
                ),
                **(
                    {"worker_phase_summaries": "worker_phase_summaries.json"}
                    if (self._run_dir / "worker_phase_summaries.json").exists()
                    else {}
                ),
            },
        }
        (out / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )

        log.info(
            "Run artefacts saved to %s\n"
            "  narrative.md          – forensic narrative\n"
            "  suspicion_list.json   – %d suspicion(s), ranked\n"
            "  detections.json       – JSON object: {detections, count, updated_at} for evaluation\n"
            "  audit_trace.json      – full tool-call trace\n"
            "  budget_summary.json   – token accounting\n"
            "  scratchpad.md         – investigation notes (live during run)\n"
            "  scratchpad_steps/     – step-wise scratchpad (step_NNNN.md) for post-debug\n"
            "  run_manifest.json     – run metadata\n"
            "  plots/                – %d chart(s)",
            out,
            len(ranked),
            len(list(self._plots_dir.glob("*.png"))),
        )


# ---------------------------------------------------------------------------
# Parallel scheme worker entry point (thread-level function)
# ---------------------------------------------------------------------------


def run_ensemble(config: InvestigatorConfig) -> List[ForensicReport]:
    """
    Run N independent forensic agents (different seeds) over the same dataset.

    All runs write into the same base output_dir; each agent automatically
    creates its own timestamped subdirectory so runs never collide.
    """
    import copy

    reports: List[ForensicReport] = []
    for i in range(config.n_agents):
        cfg_i = copy.deepcopy(config)
        cfg_i.seed = config.seed + i
        # output_dir stays the same – run_dir naming includes a UUID suffix
        agent = ForensicAgent(cfg_i)
        log.info(
            "Ensemble agent %d/%d starting (seed=%d)",
            i + 1,
            config.n_agents,
            cfg_i.seed,
        )
        reports.append(agent.run())
    return reports
