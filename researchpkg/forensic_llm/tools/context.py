from __future__ import annotations

import logging
import os
import pathlib
import threading
from typing import Any, Optional

from .db import set_db_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local tool context
#
# Each thread (e.g. a parallel scheme worker) can have its own output dir,
# scratchpad path, grep root, and graph depth by calling init_tools() from
# that thread.  The global defaults below are used as fall-backs when no
# thread-local value has been set (e.g. in the parent / single-agent runs).
# ---------------------------------------------------------------------------

_LOCAL = threading.local()

_DEFAULT_PLOT_OUTPUT_DIR: str = "./forensic_output"
_DEFAULT_GREP_ROOT: str = "./output"
_DEFAULT_GRAPH_DEPTH: int = 3
_DEFAULT_SCRATCHPAD_LIVE_PATH: Optional[str] = None


def get_plot_output_dir() -> str:
    return getattr(_LOCAL, "plot_output_dir", _DEFAULT_PLOT_OUTPUT_DIR)


def get_sql_outputs_dir() -> str:
    """Directory for ``sql(mode='export')`` CSV files (under the run output dir)."""
    out = pathlib.Path(get_plot_output_dir()) / "sql_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def get_grep_root() -> str:
    return getattr(_LOCAL, "grep_root", _DEFAULT_GREP_ROOT)


def get_default_graph_depth() -> int:
    return getattr(_LOCAL, "graph_depth", _DEFAULT_GRAPH_DEPTH)


def get_scratchpad_live_path() -> Optional[str]:
    return getattr(_LOCAL, "scratchpad_live_path", _DEFAULT_SCRATCHPAD_LIVE_PATH)


def init_tools(
    db_config: Any,
    output_dir: str = "./forensic_output",
    grep_root: str = "./output",
    graph_depth: int = 3,
    scratchpad_live_path: Optional[str] = None,
    warmup_sandbox: bool = True,
) -> None:
    """
    Initialise tool state for the **calling thread**.

    Sets both the module-level global defaults (for backwards compatibility
    with single-threaded callers) and the thread-local values so that
    parallel scheme workers are fully isolated from each other.
    Must be called once per thread before any tool is invoked.
    """
    # pylint: disable=global-statement
    global _DEFAULT_PLOT_OUTPUT_DIR, _DEFAULT_GREP_ROOT
    global _DEFAULT_GRAPH_DEPTH, _DEFAULT_SCRATCHPAD_LIVE_PATH

    set_db_config(db_config)

    abs_grep_root = os.path.abspath(grep_root)

    # Update module-level defaults (used by threads that haven't called init_tools).
    _DEFAULT_PLOT_OUTPUT_DIR = output_dir
    _DEFAULT_GREP_ROOT = abs_grep_root
    _DEFAULT_GRAPH_DEPTH = graph_depth
    _DEFAULT_SCRATCHPAD_LIVE_PATH = scratchpad_live_path

    # Set thread-local values for the calling thread.
    _LOCAL.plot_output_dir = output_dir
    _LOCAL.grep_root = abs_grep_root
    _LOCAL.graph_depth = graph_depth
    _LOCAL.scratchpad_live_path = scratchpad_live_path

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if warmup_sandbox:
        # Local subprocess backend: no persistent sandbox to warm up.
        pass
