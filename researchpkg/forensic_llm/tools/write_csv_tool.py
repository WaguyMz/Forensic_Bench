"""Backward-compatible alias for ``sql(mode='export')``."""

from __future__ import annotations

from .sql_tool import sql


def write_csv(filename: str, query: str, max_rows: int = 10000) -> str:
    return sql(
        query=query,
        mode="export",
        max_rows=max_rows,
        filename=filename,
    )
