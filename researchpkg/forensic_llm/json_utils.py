"""JSON extraction helpers shared across agent modules."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find and parse the first JSON object in *text*."""
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None
