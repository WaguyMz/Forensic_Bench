"""
Tooling surface for the forensic LLM agent.

This package contains one tool per module and re-exports the stable public API
used by the agent loop (`init_tools`, `dispatch`, etc.).

The legacy module `forensic_llm/tools.py` remains as a compatibility facade.
"""

# code_interpreter backend (localsandbox)
from .code_interpreter import code_interpreter, reset_e2b_sandbox
from .context import init_tools
from .db import _get_cursor
from .dispatch import TOOL_DISPATCH, dispatch
from .graph import graph_query, reset_graph
from .grep_tool import grep
from .image import read_image
from .scratchpad_tool import (
    get_scratchpad_text,
    reset_scratchpad,
    scratchpad,
    set_scratchpad_run_context,
)
from .sql_tool import sql
from .write_csv_tool import write_csv

__all__ = [
    "TOOL_DISPATCH",
    "_get_cursor",
    "code_interpreter",
    "dispatch",
    "get_scratchpad_text",
    "graph_query",
    "grep",
    "init_tools",
    "read_image",
    "reset_e2b_sandbox",
    "reset_graph",
    "reset_scratchpad",
    "scratchpad",
    "set_scratchpad_run_context",
    "sql",
    "write_csv",
]
