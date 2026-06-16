"""
Unified LLM client for the forensic investigator.

Supports:
- Any OpenAI-compatible endpoint: vLLM, RunPod, LM Studio, OpenAI, Azure
- Anthropic Claude via the native SDK (provider="anthropic")

The client exposes a single ``chat()`` method that returns a normalised
``LLMResponse`` regardless of the backend.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

log = logging.getLogger(__name__)

_T = TypeVar("_T")


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """
    True for transient LLM/HTTP failures worth retrying.

    Does not retry obvious client errors (4xx except 408/429) or
    control-flow exceptions.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    code = getattr(exc, "status_code", None)
    if code is not None:
        if code in (400, 401, 403, 404, 422):
            return False
        if code in (408, 429) or (isinstance(code, int) and code >= 500):
            return True
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError, OSError)):
        return True
    try:
        import httpx
    except ImportError:
        pass
    else:
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
            ),
        ):
            return True
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:
        pass
    else:
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
    try:
        import anthropic
    except ImportError:
        pass
    else:
        for name in (
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "OverloadedError",
        ):
            cls = getattr(anthropic, name, None)
            if cls is not None and isinstance(exc, cls):
                return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _is_retryable_llm_error(cause)
    return False


def _is_connection_like_error(exc: BaseException) -> bool:
    """True when the failure looks like a transport / connection problem."""
    if isinstance(exc, (ConnectionError, BrokenPipeError, TimeoutError, OSError)):
        return True
    try:
        from openai import APIConnectionError, APITimeoutError
    except ImportError:
        pass
    else:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    try:
        import httpx
    except ImportError:
        pass
    else:
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
            ),
        ):
            return True
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "connection error",
            "connection refused",
            "connection reset",
            "connect error",
            "network",
            "timed out",
            "timeout",
            "broken pipe",
            "remote protocol",
        )
    ):
        return True
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _is_connection_like_error(cause)
    return False


def _log_llm_exhausted_failure(
    what: str,
    exc: BaseException,
    *,
    attempts_total: int,
    attempt_num: int,
    retryable: bool,
) -> None:
    """
    Emit a single high-signal ERROR when an LLM call will not be retried again.

    Makes connection-loss storms easy to grep in bench ``parallel.log`` files.
    """
    if not isinstance(exc, Exception):
        return
    if _is_connection_like_error(exc):
        headline = "LLM CONNECTION ERROR"
    elif retryable:
        headline = "LLM TRANSIENT ERROR"
    else:
        headline = "LLM ERROR"
    log.error(
        "%s after %d/%d attempt(s) [%s]: %s",
        headline,
        attempt_num,
        attempts_total,
        what,
        exc,
    )


def _call_with_retries(
    fn: Callable[[], _T],
    max_retries: int,
    *,
    what: str,
    max_delay_s: float = 120.0,
    on_failure: Optional[
        Callable[[BaseException, int, int, bool, bool], None]
    ] = None,  # (exc, attempt_num, attempts_total, will_retry, retryable)
) -> _T:
    """
    Run ``fn()`` with up to ``max_retries`` additional attempts after failures.

    Parameters
    ----------
    max_retries :
        Number of retries after the first failed attempt (0 = no retry).
    """
    last_exc: Optional[BaseException] = None
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(attempts):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            retryable = _is_retryable_llm_error(exc)
            will_retry = retryable and attempt < attempts - 1
            if on_failure is not None:
                try:
                    on_failure(exc, attempt + 1, attempts, will_retry, retryable)
                except Exception:
                    # Never let accounting break the main call path.
                    pass

            if attempt >= attempts - 1 or not retryable:
                _log_llm_exhausted_failure(
                    what,
                    exc,
                    attempts_total=attempts,
                    attempt_num=attempt + 1,
                    retryable=retryable,
                )
                raise
            delay = min(max_delay_s, 1.0 * (2**attempt))
            log.warning(
                "%s failed (attempt %s/%s, transient): %s; retrying in %.1fs",
                what,
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _message_content_to_text(content: Any) -> str:
    """Best-effort conversion of chat content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                elif btype == "image_url":
                    parts.append(
                        "[image attachment omitted during message-order normalization]"
                    )
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _check_openai_messages(
    messages: List[Dict[str, Any]], *, assistant_filler: str = ""
) -> None:
    """
    Normalize the message history in place before sending it to an
    OpenAI-compatible backend.

    Repairs applied:
    - Drop orphan `role="tool"` messages that no longer have a matching
      preceding assistant `tool_calls` turn.
    - Preserve only the subset of assistant `tool_calls` that still have
      matching tool results in the trimmed payload; strip incomplete tool-call
      metadata from assistant turns instead of sending invalid wire format.
    - If a `role="user"` still ends up immediately after a `role="tool"`,
      insert a minimal assistant turn between them for strict backends.
    - Ensure at least one `role="user"` turn exists.
    """
    repaired: List[Dict[str, Any]] = []
    staged_assistant: Optional[Dict[str, Any]] = None
    staged_tool_ids: set[str] = set()
    staged_tool_results: List[Dict[str, Any]] = []

    def _flush_staged_assistant() -> None:
        nonlocal staged_assistant, staged_tool_ids, staged_tool_results
        if staged_assistant is None:
            return

        assistant_msg = dict(staged_assistant)
        if staged_tool_results:
            matched_ids = {
                str(msg.get("tool_call_id", "")).strip()
                for msg in staged_tool_results
                if msg.get("tool_call_id")
            }
            assistant_msg["tool_calls"] = [
                tc
                for tc in staged_assistant.get("tool_calls", [])
                if str(tc.get("id", "")).strip() in matched_ids
            ]
            repaired.append(assistant_msg)
            repaired.extend(staged_tool_results)
        else:
            assistant_msg.pop("tool_calls", None)
            # Some strict OpenAI-compatible backends reject assistant messages with
            # empty content when no tool_calls are present. When we strip tool_calls
            # (because their tool results were evicted), ensure content is non-empty.
            if not _message_content_to_text(assistant_msg.get("content", "")).strip():
                assistant_msg["content"] = assistant_filler
            repaired.append(assistant_msg)

        staged_assistant = None
        staged_tool_ids = set()
        staged_tool_results = []

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            _flush_staged_assistant()
            staged_assistant = msg
            staged_tool_ids = {
                str(tc.get("id", "")).strip()
                for tc in msg.get("tool_calls", [])
                if tc.get("id")
            }
            staged_tool_results = []
            continue

        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id", "")).strip()
            if (
                staged_assistant is None
                or not tool_call_id
                or tool_call_id not in staged_tool_ids
            ):
                continue
            staged_tool_results.append(msg)
            continue

        _flush_staged_assistant()
        if role == "user" and repaired and repaired[-1].get("role") == "tool":
            # Some strict OpenAI-compatible backends reject empty assistant messages.
            # Allow callers (e.g., Mistral) to provide a non-empty filler string.
            repaired.append({"role": "assistant", "content": assistant_filler})
        repaired.append(msg)

    _flush_staged_assistant()

    if not any(m.get("role") == "user" for m in repaired):
        insert_at = 1 if repaired and repaired[0].get("role") == "system" else 0
        repaired.insert(insert_at, {"role": "user", "content": ""})

    messages[:] = repaired


# ---------------------------------------------------------------------------
# Normalised response object
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRequest:
    """A single tool call requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]  # already parsed from JSON


def _stable_tool_call_id(index: int, *, style: str = "default") -> str:
    """
    Deterministic tool-call ID for model-visible chat history.

    Backend-generated IDs often contain random suffixes, which pollute the next
    prompt and break byte-for-byte determinism even when the model output is
    otherwise identical. We keep the tool-call ordering but replace provider IDs
    with a stable local identifier that is sufficient for matching assistant
    tool_calls to subsequent role="tool" messages.
    """
    if style == "mistral9":
        # Some OpenAI-compatible Mistral backends validate tool_call_id strictly:
        # - alphanumeric only (no underscores)
        # - fixed length of 9
        # Use: "call" + 5 digits => 9 chars, ASCII alnum only.
        return f"call{index:05d}"
    return f"tool_call_{index:04d}"


@dataclass
class LLMResponse:
    content: str  # Plain text content (may be empty)
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Some backends expose reasoning tokens as a separate usage detail. These may
    # already be a subset of completion_tokens, so callers should avoid blindly
    # adding them to billed-token totals.
    reasoning_tokens: int = 0
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"
    # Optional: reasoning trace for thinking models (vLLM reasoning outputs, etc.).
    reasoning: Optional[str] = None
    raw: Any = None  # Original SDK response object

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# OpenAI-compatible client (vLLM, RunPod, OpenAI)
# ---------------------------------------------------------------------------


class OpenAICompatibleClient:
    """
    Wraps the ``openai`` Python SDK and points it at any base_url.

    vLLM exposes the full OpenAI chat completions API including function
    calling / tool use for models whose chat templates support it.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens_per_step: int = 4096,
        request_timeout: float = 600.0,
        disable_thinking: bool = False,
        enable_thinking: bool = False,
        max_retries: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        from openai import OpenAI  # lazy import

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens_per_step = max_tokens_per_step
        self.disable_thinking = bool(disable_thinking)
        self.enable_thinking = bool(enable_thinking)
        self.seed = seed
        self.max_retries = max(0, int(max_retries))
        self._tool_call_id_style: str = (
            "mistral9" if "mistral" in str(model or "").lower() else "default"
        )
        self.llm_errors_total: int = 0
        self.llm_errors_unrecovered: int = 0

        # SDK-level retries disabled; ``chat()`` applies configurable backoff.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=0,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Parameters
        ----------
        messages         : OpenAI message list (system, user, assistant, tool).
        tools            : OpenAI tool definitions list, or None to disable.
        tool_choice      : "auto" | "required" | "none".
        response_schema  : Pydantic model_json_schema() dict.  When provided,
                           passes response_format={"type":"json_schema",...} to
                           enforce structured output.  Tools are disabled
                           automatically (incompatible with structured output).

        Returns
        -------
        LLMResponse with normalised content and tool_calls.
        """
        # Validate message history before sending.  Logs errors for any
        # role-ordering violations so they surface immediately without aborting.
        _check_openai_messages(
            messages,
            assistant_filler=(" " if self._tool_call_id_style == "mistral9" else ""),
        )

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(max_tokens or self.max_tokens_per_step),
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if response_schema is not None:
            # Structured output: enforce exact JSON schema via vLLM / OpenAI API.
            # Tools are disabled — mixing tool_calls with json_schema response_format
            # is not supported and would cause a 400 on most backends.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "investigation_plan",
                    "schema": response_schema,
                },
            }
        elif tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Best-effort: control "thinking"/reasoning mode for Qwen3-style chat
        # templates when using OpenAI-compatible self-hosted backends (vLLM).
        # We must NOT send unknown parameters to OpenAI's hosted API.
        base = (self.base_url or "").lower()
        is_openai_hosted = "api.openai.com" in base or "openai.com/v1" in base
        if not is_openai_hosted:
            if self.enable_thinking:
                # enable_thinking takes precedence over disable_thinking.
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            elif self.disable_thinking:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

        def _count_failure(
            _exc: BaseException,
            _attempt_num: int,
            _attempts_total: int,
            will_retry: bool,
            _retryable: bool,
        ) -> None:
            self.llm_errors_total += 1
            if not will_retry:
                self.llm_errors_unrecovered += 1

        resp = _call_with_retries(
            lambda: self._client.chat.completions.create(**kwargs),
            self.max_retries,
            what="OpenAI-compatible chat completion",
            on_failure=_count_failure,
        )

        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason or "stop"

        # Parse tool calls
        tool_calls: List[ToolCallRequest] = []
        if msg.tool_calls:
            for idx, tc in enumerate(msg.tool_calls, start=1):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    log.warning(
                        "Could not parse tool call arguments for %s: %r",
                        tc.function.name,
                        tc.function.arguments,
                    )
                    args = {}
                tool_calls.append(
                    ToolCallRequest(
                        id=_stable_tool_call_id(idx, style=self._tool_call_id_style),
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = resp.usage
        reasoning = getattr(msg, "reasoning", None)
        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            reasoning_tokens=_extract_reasoning_tokens(usage),
            finish_reason=finish,
            reasoning=reasoning,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------


class AnthropicClient:
    """
    Wraps the ``anthropic`` Python SDK.

    Translates OpenAI-style message lists and tool definitions to the
    Anthropic format on the fly so the agent code stays format-agnostic.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
        temperature: float = 0.1,
        max_tokens_per_step: int = 4096,
        max_retries: int = 5,
    ) -> None:
        import anthropic  # lazy import

        self.model = model
        self.temperature = temperature
        self.max_tokens_per_step = max_tokens_per_step
        self.max_retries = max(0, int(max_retries))
        self.llm_errors_total: int = 0
        self.llm_errors_unrecovered: int = 0
        self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Format converters
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool defs to Anthropic tool defs."""
        out = []
        for t in openai_tools:
            fn = t.get("function", {})
            out.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return out

    @staticmethod
    def _convert_messages(
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Split system message out; convert tool result messages.

        Returns (system_text, anthropic_messages).
        """
        system_parts: List[str] = []
        ant_msgs: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role", "")
            if role == "system":
                system_parts.append(m.get("content", ""))
                continue

            if role == "tool":
                # Anthropic uses role="user" with tool_result content blocks
                ant_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id", ""),
                                "content": str(m.get("content", "")),
                            }
                        ],
                    }
                )
                continue

            if role == "assistant" and m.get("tool_calls"):
                # Convert OpenAI assistant tool-call message
                content_blocks: List[Dict[str, Any]] = []
                if m.get("content"):
                    content_blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args,
                        }
                    )
                ant_msgs.append({"role": "assistant", "content": content_blocks})
                continue

            ant_msgs.append({"role": role, "content": m.get("content", "")})

        return "\n\n".join(system_parts), ant_msgs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # Anthropic does not support response_format / json_schema natively on
        # all models.  Enforce structured output via two complementary techniques:
        #   1. Append the required JSON schema to the last user message so the
        #      model knows exactly what structure to produce.
        #   2. Add an assistant prefill turn starting with `{` to force the
        #      response to begin as a JSON object.
        if response_schema is not None:
            import json as _json

            schema_str = _json.dumps(response_schema, indent=2)
            schema_note = (
                "\n\nYour response MUST be a JSON object that strictly conforms "
                "to the following JSON Schema (no extra text, no markdown fences):\n"
                f"```json\n{schema_str}\n```"
            )
            # Inject schema note into the last user message.
            msgs = list(messages)
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    content = msgs[i].get("content", "")
                    if isinstance(content, str):
                        msgs[i] = {**msgs[i], "content": content + schema_note}
                    break
            # Append assistant prefill `{` — Anthropic continues from this token.
            messages = msgs + [{"role": "assistant", "content": "{"}]

        system, ant_msgs = self._convert_messages(messages)
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=int(max_tokens or self.max_tokens_per_step),
            temperature=self.temperature,
            messages=ant_msgs,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}
            else:
                kwargs["tool_choice"] = {"type": "auto"}

        def _count_failure(
            _exc: BaseException,
            _attempt_num: int,
            _attempts_total: int,
            will_retry: bool,
            _retryable: bool,
        ) -> None:
            self.llm_errors_total += 1
            if not will_retry:
                self.llm_errors_unrecovered += 1

        resp = _call_with_retries(
            lambda: self._client.messages.create(**kwargs),
            self.max_retries,
            what="Anthropic messages.create",
            on_failure=_count_failure,
        )

        # Extract text and tool_use blocks
        text_parts: List[str] = []
        tool_calls: List[ToolCallRequest] = []
        next_tool_idx = 1
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=_stable_tool_call_id(next_tool_idx),
                        name=block.name,
                        arguments=block.input or {},
                    )
                )
                next_tool_idx += 1

        # When using assistant prefill for structured output, Anthropic returns only
        # the continuation after `{` — restore the leading brace.
        if response_schema is not None and text_parts:
            text_parts[0] = "{" + text_parts[0]

        finish = "tool_calls" if tool_calls else "stop"
        if resp.stop_reason == "max_tokens":
            finish = "length"

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            reasoning_tokens=_extract_reasoning_tokens(getattr(resp, "usage", None)),
            finish_reason=finish,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_client(config) -> OpenAICompatibleClient | AnthropicClient:
    """
    Construct the correct LLM client from an LLMConfig object.

    Parameters
    ----------
    config : LLMConfig
    """
    if config.provider == "anthropic":
        return AnthropicClient(
            api_key=config.api_key,
            model=config.anthropic_model,
            temperature=config.temperature,
            max_tokens_per_step=config.max_tokens_per_step,
            max_retries=config.max_retries,
        )

    return OpenAICompatibleClient(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens_per_step=config.max_tokens_per_step,
        request_timeout=config.request_timeout,
        enable_thinking=config.enable_thinking,
        disable_thinking=config.disable_thinking,
        max_retries=config.max_retries,
        seed=config.seed,
    )


def _usage_get(obj: Any, *path: str) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _extract_reasoning_tokens(usage: Any) -> int:
    """
    Best-effort extraction of reasoning-token counts from provider-specific usage
    payloads.

    We store the count separately for budgeting/reporting, but do not assume it
    should be added to billed-token totals because some APIs report it as a
    subset of completion/output tokens.
    """
    candidates = (
        _usage_get(usage, "completion_tokens_details", "reasoning_tokens"),
        _usage_get(usage, "output_tokens_details", "reasoning_tokens"),
        _usage_get(usage, "completion_tokens_details", "thinking_tokens"),
        _usage_get(usage, "output_tokens_details", "thinking_tokens"),
        _usage_get(usage, "reasoning_tokens"),
        _usage_get(usage, "thinking_tokens"),
    )
    for value in candidates:
        try:
            n = int(value or 0)
        except Exception:
            continue
        if n > 0:
            return n
    return 0
