"""
Token counting: tiktoken for OpenAI models, HuggingFace tokenizer otherwise.

Uses each library's standard APIs.  HF models whose chat template does not
accept OpenAI-style tool messages are counted after flattening tool turns
into plain role/content text (same tokenizer, still model-accurate).
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

log = logging.getLogger(__name__)

# Gateway/API model ids that differ from the HuggingFace Hub tokenizer repo.
_HF_TOKENIZER_ALIASES: Dict[str, str] = {
    "zhipuai/glm-5.1": "zai-org/GLM-5.1",
    "glm-5.1": "zai-org/GLM-5.1",
}


def resolve_tokenizer_hub_id(
    model: str,
    *,
    tokenizer_model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Hub repo id used for local token counting.

    API providers (Eden AI, etc.) often expose model ids that are not valid
    ``from_pretrained`` paths — map those to the official weights repo when known.
    """
    if tokenizer_model and str(tokenizer_model).strip():
        return str(tokenizer_model).strip()

    key = (model or "").strip().lower()
    if key in _HF_TOKENIZER_ALIASES:
        return _HF_TOKENIZER_ALIASES[key]

    base = (base_url or "").lower()
    if "edenai.run" in base and ("glm-5.1" in key or key.endswith("/glm-5.1")):
        return "zai-org/GLM-5.1"

    return model


_CACHE: Dict[str, "ModelTokenCounter"] = {}
_CACHE_LOCK = threading.Lock()
_ACTIVE: Optional["ModelTokenCounter"] = None


def _is_openai_model(model: str) -> bool:
    name = (model or "").lower()
    return any(
        name.startswith(p)
        for p in ("gpt-", "o1", "o2", "o3", "o4", "chatgpt", "text-davinci")
    )


def _tiktoken_encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        name = model.lower()
        if "gpt-4o" in name or "gpt-4" in name and "turbo" not in name:
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")


def _tiktoken_count_messages(encoding, messages: List[Dict[str, Any]]) -> int:
    """OpenAI cookbook recipe (tiktoken)."""
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key == "content":
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value, disallowed_special=()))
                elif isinstance(value, list):
                    for block in value:
                        if isinstance(block, dict) and block.get("type") == "text":
                            num_tokens += len(
                                encoding.encode(
                                    str(block.get("text", "")),
                                    disallowed_special=(),
                                )
                            )
            elif key == "tool_calls" and isinstance(value, list):
                for tc in value:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    num_tokens += len(
                        encoding.encode(
                            fn.get("name", "") + fn.get("arguments", ""),
                            disallowed_special=(),
                        )
                    )
            elif isinstance(value, str):
                num_tokens += len(encoding.encode(value, disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize None content and strip keys chat templates may reject."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def _flatten_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Represent tool calls/results as plain text for HF chat templates (e.g. Qwen3).

    vLLM still receives native tool messages; this is only for token counting.
    """
    flat: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _text_content(msg.get("content"))

        if msg.get("tool_calls"):
            parts = [content] if content else []
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                parts.append(
                    f"[tool_call {fn.get('name', '')}] {fn.get('arguments', '')}"
                )
            content = "\n".join(parts)
            role = "assistant"
        elif role == "tool":
            name = msg.get("name", "tool")
            content = f"[tool_result {name}]\n{content}"
            role = "user"
        elif role not in ("system", "user", "assistant"):
            role = "user"

        flat.append({"role": role, "content": content})
    return flat


def _hf_apply_chat_template_count(tok: Any, messages: List[Dict[str, Any]]) -> int:
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(ids, list):
        return len(ids)
    raise TypeError(f"unexpected apply_chat_template return type: {type(ids)}")


@dataclass(frozen=True)
class TokenCounterConfig:
    model: str
    tokenizer_model: Optional[str] = None
    base_url: Optional[str] = None
    trust_remote_code: bool = True


def _load_hf_autotokenizer(
    tokenizer_id: str,
    *,
    trust_remote_code: bool,
) -> Tuple[Any, str]:
    """
    Load ``AutoTokenizer``, retrying kwargs Hub configs sometimes break on older
    ``transformers`` (e.g. ``tokenizer_class: TokenizersBackend``).
    """
    from transformers import AutoTokenizer

    last_exc: Optional[BaseException] = None
    for extra in ({}, {"use_fast": False}):
        kwargs = {"trust_remote_code": trust_remote_code, **extra}
        try:
            tok = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
            slow = extra.get("use_fast") is False
            desc = f"{tokenizer_id} ({'slow' if slow else 'fast'})"
            return tok, desc
        except Exception as exc:
            last_exc = exc
            log.debug(
                "AutoTokenizer.from_pretrained(%r) kwargs=%s failed: %s",
                tokenizer_id,
                extra,
                exc,
            )
    assert last_exc is not None
    raise last_exc


def _tokenizer_fallback_candidates(primary_id: str) -> List[str]:
    """Ordered Hub ids to try when ``primary_id`` fails (token counting only)."""
    out: List[str] = []
    fb = os.environ.get("FORENSIC_TOKENIZER_FALLBACK", "").strip()
    if fb:
        out.append(fb)
    low = primary_id.lower()
    if any(x in low for x in ("mistral", "ministral", "mixtral")):
        m = "mistralai/Mistral-7B-Instruct-v0.3"
        if m not in out:
            out.append(m)
    return out


class ModelTokenCounter:
    """tiktoken (OpenAI) or HuggingFace AutoTokenizer (vLLM / HF models)."""

    def __init__(self, config: TokenCounterConfig) -> None:
        self.config = config
        self.model = config.model
        self._tokenizer_id = resolve_tokenizer_hub_id(
            config.model,
            tokenizer_model=config.tokenizer_model,
            base_url=config.base_url,
        )
        self._hf_tokenizer_desc = self._tokenizer_id
        self._tiktoken_encoding = None
        self._hf_tokenizer = None
        if _is_openai_model(self.model):
            self._tiktoken_encoding = _tiktoken_encoding_for_model(self.model)
            log.info("Token counter: tiktoken (%s)", self.model)
        else:
            chain = [self._tokenizer_id]
            chain.extend(
                tid
                for tid in _tokenizer_fallback_candidates(self._tokenizer_id)
                if tid not in chain
            )
            last_exc: Optional[BaseException] = None
            for tid in chain:
                try:
                    (
                        self._hf_tokenizer,
                        self._hf_tokenizer_desc,
                    ) = _load_hf_autotokenizer(
                        tid,
                        trust_remote_code=config.trust_remote_code,
                    )
                    if tid != self._tokenizer_id:
                        log.warning(
                            "Token counter: using fallback tokenizer %r for model %r "
                            "(primary id %r failed on this transformers build — counts remain approximate). "
                            "Upgrade transformers/tokenizers, set FORENSIC_TOKENIZER_MODEL, or FORENSIC_TOKENIZER_FALLBACK.",
                            tid,
                            self.model,
                            self._tokenizer_id,
                        )
                    break
                except Exception as exc:
                    last_exc = exc
                    log.debug("Tokenizer candidate %r failed: %s", tid, exc)
            else:
                raise RuntimeError(
                    "Could not load a HuggingFace tokenizer for token counting. "
                    "Typical fixes: pip install -U transformers tokenizers; "
                    "set FORENSIC_TOKENIZER_MODEL to a compatible Hub tokenizer id; "
                    "or set FORENSIC_TOKENIZER_FALLBACK (e.g. mistralai/Mistral-7B-Instruct-v0.3 for Mistral endpoints)."
                ) from last_exc
            log.info(
                "Token counter: HuggingFace (model=%s, tokenizer=%s)",
                self.model,
                self._hf_tokenizer_desc,
            )

    @property
    def backend(self) -> str:
        return "tiktoken" if self._tiktoken_encoding is not None else "huggingface"

    @property
    def tokenizer_hub_id(self) -> str:
        return self._tokenizer_id

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tiktoken_encoding is not None:
            return len(self._tiktoken_encoding.encode(text, disallowed_special=()))
        return len(
            self._hf_tokenizer.encode(  # type: ignore[union-attr]
                text, add_special_tokens=False, truncation=False
            )
        )

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        if not messages:
            return 0
        if self._tiktoken_encoding is not None:
            return _tiktoken_count_messages(
                self._tiktoken_encoding, _sanitize_messages(messages)
            )

        tok = self._hf_tokenizer
        sanitized = _sanitize_messages(messages)

        for candidate in (sanitized, _flatten_tool_messages(sanitized)):
            if not candidate:
                continue
            if not hasattr(tok, "apply_chat_template"):
                break
            try:
                return _hf_apply_chat_template_count(tok, candidate)
            except Exception as exc:
                log.debug(
                    "apply_chat_template failed for %s (%d msgs): %s",
                    self._tokenizer_id,
                    len(candidate),
                    exc,
                )

        # Last resort: encode flattened transcript with the model tokenizer.
        flat = _flatten_tool_messages(sanitized)
        blob = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in flat
        )
        return self.count(blob)


def configure_token_counter(
    *,
    model: str,
    tokenizer_model: Optional[str] = None,
    base_url: Optional[str] = None,
    trust_remote_code: bool = True,
) -> ModelTokenCounter:
    """Install the process-wide token counter (cached per model)."""
    global _ACTIVE
    cfg = TokenCounterConfig(
        model=model,
        tokenizer_model=tokenizer_model,
        base_url=base_url,
        trust_remote_code=trust_remote_code,
    )
    hub_id = resolve_tokenizer_hub_id(
        model,
        tokenizer_model=tokenizer_model,
        base_url=base_url,
    )
    key = f"{model}|{hub_id}|{base_url or ''}"
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = ModelTokenCounter(cfg)
        _ACTIVE = _CACHE[key]
    return _ACTIVE


def get_token_counter() -> Optional[ModelTokenCounter]:
    return _ACTIVE


def count_tokens(text: str) -> int:
    if _ACTIVE is None:
        raise RuntimeError("Token counter not configured")
    return _ACTIVE.count(text)


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    if _ACTIVE is None:
        raise RuntimeError("Token counter not configured")
    return _ACTIVE.count_messages(messages)
