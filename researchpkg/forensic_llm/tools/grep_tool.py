from __future__ import annotations

import os
import re
from typing import Optional

from .context import get_grep_root


def grep(pattern: str, path: Optional[str] = None) -> str:
    """
    Very small grep helper used by the agent for searching flat files produced
    during a run. Anchored to grep_root for safety.
    """
    root = get_grep_root()
    rel = (path or "").strip()
    target = os.path.abspath(os.path.join(root, rel)) if rel else root
    if not target.startswith(os.path.abspath(root)):
        return "[GREP ERROR] path escapes grep root"

    pat = pattern or ""
    if not pat:
        return "[GREP ERROR] empty pattern"

    # Walk and match lines (small, safe implementation).
    out_lines = []
    try:
        rx = re.compile(pat)
    except re.error as exc:
        return f"[GREP ERROR] invalid regex: {exc}"

    for dirpath, _, filenames in os.walk(target):
        for fn in filenames:
            fpath = os.path.join(dirpath, fn)
            # Skip very large binaries by heuristic.
            try:
                if os.path.getsize(fpath) > 5_000_000:
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if rx.search(line):
                            relp = os.path.relpath(fpath, root)
                            out_lines.append(f"{relp}:{i}:{line.rstrip()}")
            except Exception:
                continue

    return "\n".join(out_lines) if out_lines else "[GREP] no matches"
