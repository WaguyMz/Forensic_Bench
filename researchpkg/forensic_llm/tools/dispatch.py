from __future__ import annotations

import logging
from typing import Any, Dict

from .code_interpreter import code_interpreter
from .graph import graph_query
from .grep_tool import grep
from .image import read_image
from .scratchpad_tool import scratchpad
from .sql_tool import sql
from .write_csv_tool import write_csv

log = logging.getLogger(__name__)


TOOL_DISPATCH: Dict[str, Any] = {
    "sql": sql,
    "grep": grep,
    "read_image": read_image,
    "graph_query": graph_query,
    "scratchpad": scratchpad,
    "write_csv": write_csv,
    "code_interpreter": code_interpreter,
    # "finish_investigation" is handled specially by the agent loop
}


def dispatch(tool_name: str, args: Dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments.

    Returns the tool's string output or an error message.
    """
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return f"[TOOL ERROR] Unknown tool: '{tool_name}'"
    try:
        return fn(**args)
    except TypeError as exc:
        return f"[TOOL ERROR] Bad arguments for '{tool_name}': {exc}"
    except Exception as exc:
        log.exception("Tool '%s' raised an exception", tool_name)
        return f"[TOOL ERROR] '{tool_name}' failed: {exc}"
