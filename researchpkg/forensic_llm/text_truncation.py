"""
Token-based text truncation for forensic LLM context and persistence.

All content size limits are expressed in tokens (via the run's configured
tokenizer). Character slicing must not be used to cap model-visible or
stored excerpts — use :func:`truncate_text_to_tokens` instead.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from researchpkg.forensic_llm.model_tokenizer import (
    count_tokens,
)

TRUNCATION_MARKER = "\n[... truncated for token limit ...]"


class TruncationSide(str, Enum):
    """Which portion of the text to keep when over the token budget."""

    HEAD = "head"
    TAIL = "tail"


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    side: TruncationSide = TruncationSide.HEAD,
    marker: str = TRUNCATION_MARKER,
) -> str:
    """
    Truncate ``text`` to at most ``max_tokens`` (including the marker when applied).

    Parameters
    ----------
    max_tokens:
        Maximum tokens allowed. ``0`` means no limit; negative values are treated as 0.
    """
    if not text:
        return text
    if max_tokens <= 0:
        return text

    if count_tokens(text) <= max_tokens:
        return text

    marker = marker or TRUNCATION_MARKER
    marker_tokens = count_tokens(marker) if marker else 0
    content_budget = max(1, max_tokens - marker_tokens)

    if side == TruncationSide.TAIL:
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if count_tokens(text[mid:]) <= content_budget:
                hi = mid
            else:
                lo = mid + 1
        return f"{marker}{text[lo:]}"

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= content_budget:
            lo = mid
        else:
            hi = mid - 1
    return f"{text[:lo]}{marker}"


def truncate_message_to_tokens(msg: dict, max_tokens: int) -> dict:
    """Return a copy of an OpenAI-style message with string content truncated."""
    out = dict(msg)
    content = msg.get("content", "") or ""
    if not isinstance(content, str) or not content:
        return out
    out["content"] = truncate_text_to_tokens(content, max_tokens, side=TruncationSide.HEAD)
    return out
