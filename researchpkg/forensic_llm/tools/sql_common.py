"""Shared read-only SQL helpers for preview and export modes."""

from __future__ import annotations

import re
from typing import Any, List, Tuple

_HIDDEN_COLUMNS = {"is_fraud", "fraud_scheme", "label", "is_injected"}

_FORBIDDEN_SQL = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE|"
    r"CREATE|ALTER|DROP|TRUNCATE|REINDEX|VACUUM|ANALYZE|"
    r"GRANT|REVOKE|"
    r"CALL|DO|EXECUTE|"
    r"COPY|"
    r"SET|RESET|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|"
    r"LOCK|"
    r"LISTEN|NOTIFY|UNLISTEN"
    r")\b",
    re.IGNORECASE,
)


def strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"(?m)--.*?$", "", sql)
    sql = re.sub(r"(?s)/\*.*?\*/", "", sql)
    return sql


def reject_non_readonly(query: str) -> str:
    q0 = (query or "").strip()
    if not q0:
        raise ValueError("empty query")

    q = strip_sql_comments(q0).strip()
    if not q:
        raise ValueError("empty query")

    q_no_trailing = q.rstrip().rstrip(";").strip()
    if ";" in q_no_trailing:
        raise ValueError("multiple SQL statements per call are not allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", q_no_trailing):
        raise ValueError("only SELECT/WITH queries are allowed")
    if _FORBIDDEN_SQL.search(q_no_trailing):
        raise ValueError("query contains forbidden (non-readonly) operation(s)")
    return q_no_trailing


def apply_row_limit(query: str, cap: int) -> str:
    cap = max(1, int(cap))
    if not re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
        return f"{query} LIMIT {cap}"
    return re.sub(
        r"\bLIMIT\s+(\d+)",
        lambda m: f"LIMIT {min(int(m.group(1)), cap)}",
        query,
        flags=re.IGNORECASE,
    )


def filter_hidden_columns(
    columns: List[str], rows: List[Tuple[Any, ...]]
) -> tuple[List[str], List[Tuple[Any, ...]]]:
    keep = [i for i, c in enumerate(columns) if c not in _HIDDEN_COLUMNS]
    if len(keep) == len(columns):
        return columns, rows
    columns = [columns[i] for i in keep]
    rows = [tuple(row[i] for i in keep) for row in rows]
    return columns, rows
