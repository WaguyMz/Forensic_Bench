from __future__ import annotations

import csv as _csv
import hashlib
import pathlib
import re
import time
from typing import Any, List, Optional, Tuple

from .context import get_sql_outputs_dir
from .db import _get_cursor
from .sql_common import apply_row_limit, filter_hidden_columns, reject_non_readonly

_PREVIEW_DEFAULT_ROWS = 50
_PREVIEW_MAX_ROWS = 50
_EXPORT_DEFAULT_ROWS = 10_000
_EXPORT_MAX_ROWS = 50_000
_EXPORT_SAMPLE_ROWS = 5


def sql(
    query: str,
    mode: str = "preview",
    max_rows: Optional[int] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Execute a read-only SQL query.

    - ``mode='preview'`` (default): return a markdown table for the model (at most 50 rows;
      use ``export`` for larger result sets).
    - ``mode='export'``: write full results to ``sql_outputs/*.csv``; return manifest only.
    """
    mode_key = (mode or "preview").strip().lower()
    if mode_key == "preview":
        cap = _PREVIEW_DEFAULT_ROWS if max_rows is None else int(max_rows)
        return _sql_preview(query, cap)
    if mode_key == "export":
        cap = _EXPORT_DEFAULT_ROWS if max_rows is None else int(max_rows)
        return _sql_export(query, cap, filename)
    return f"[SQL ERROR] invalid mode {mode!r}; use 'preview' or 'export'"


def _sql_preview(query: str, max_rows: int) -> str:
    try:
        q = reject_non_readonly(query)
    except ValueError as exc:
        return f"[SQL ERROR] {exc}"

    cap = max(1, min(int(max_rows), _PREVIEW_MAX_ROWS))
    q_stripped = apply_row_limit(q, cap)

    try:
        with _get_cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(q_stripped)
            rows = cur.fetchall()
            elapsed = time.perf_counter() - t0
            columns = [desc[0] for desc in cur.description] if cur.description else []
    except Exception as exc:
        return f"[SQL ERROR] {exc}"

    columns, rows = filter_hidden_columns(columns, rows)
    return _format_markdown_table(columns, rows, elapsed)


def _sql_export(query: str, max_rows: int, filename: Optional[str]) -> str:
    try:
        q = reject_non_readonly(query)
    except ValueError as exc:
        return f"[SQL_EXPORT ERROR] {exc}"

    cap = max(1, min(int(max_rows), _EXPORT_MAX_ROWS))
    q_stripped = apply_row_limit(q, cap)

    try:
        with _get_cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(q_stripped)
            rows: List[Tuple[Any, ...]] = cur.fetchall()
            elapsed = time.perf_counter() - t0
            columns = [desc[0] for desc in cur.description] if cur.description else []
    except Exception as exc:
        return f"[SQL_EXPORT ERROR] {exc}"

    columns, rows = filter_hidden_columns(columns, rows)

    safe_name = _safe_export_filename(filename, q)
    out_dir = pathlib.Path(get_sql_outputs_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / safe_name

    try:
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
    except Exception as exc:
        return f"[SQL_EXPORT ERROR] Failed to write file: {exc}"

    sample = _format_markdown_table(columns, rows[:_EXPORT_SAMPLE_ROWS], elapsed)
    if len(rows) > _EXPORT_SAMPLE_ROWS:
        sample += (
            f"\n\n*… {len(rows) - _EXPORT_SAMPLE_ROWS} more rows in file "
            f"(not shown in context)*"
        )

    return (
        f"[SQL_EXPORT] mode=export rows={len(rows)} cols={len(columns)} "
        f"path={fpath!s}\n"
        f"Columns: {', '.join(columns)}\n\n"
        f"Sample (first {min(len(rows), _EXPORT_SAMPLE_ROWS)} row(s)):\n{sample}\n\n"
        f"Load in code_interpreter: pd.read_csv({str(fpath)!r})"
    )


def _safe_export_filename(filename: Optional[str], query: str) -> str:
    if filename and str(filename).strip():
        safe = pathlib.Path(str(filename).strip()).name
    else:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:10]
        safe = f"export_{digest}.csv"
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    return safe


def _format_markdown_table(
    columns: List[str],
    rows: List[Tuple[Any, ...]],
    elapsed: float,
) -> str:
    if not columns:
        return f"*0 column(s) returned in {elapsed * 1000:.0f} ms*"

    def _cell(x: Any) -> str:
        if x is None:
            return "None"
        return str(x)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    data_rows = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    footer = f"\n\n*{len(rows)} row(s) returned in {elapsed * 1000:.0f} ms*"
    if not data_rows:
        return "\n".join([header, sep]) + footer
    return "\n".join([header, sep] + data_rows) + footer
