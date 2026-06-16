from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

_DB_CONFIG: Any = None  # set by init_tools()
_DB_CONN: Any = None  # psycopg2 connection (reconnected as needed)


def set_db_config(db_config: Any) -> None:
    global _DB_CONFIG
    _DB_CONFIG = db_config


def get_db_config() -> Any:
    return _DB_CONFIG


@contextmanager
def _get_cursor() -> Generator:
    """
    Yield a psycopg2 cursor that runs in a read-only transaction with a
    statement timeout. Reconnects transparently if the connection is closed.
    """
    import psycopg2

    global _DB_CONN

    if _DB_CONFIG is None:
        raise RuntimeError("tools not initialised – call init_tools() first")

    # Reconnect if necessary
    if _DB_CONN is None or getattr(_DB_CONN, "closed", 1):
        _DB_CONN = psycopg2.connect(_DB_CONFIG.dsn)
        _DB_CONN.autocommit = False

    cur = _DB_CONN.cursor()
    try:
        # Enforce read-only transaction and statement timeout
        cur.execute("BEGIN READ ONLY;")
        cur.execute(f"SET LOCAL statement_timeout = {_DB_CONFIG.statement_timeout_ms};")
        yield cur
        cur.execute("ROLLBACK;")
    except Exception:
        try:
            cur.execute("ROLLBACK;")
        except Exception:
            pass
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass


def reconnect_db() -> None:
    global _DB_CONN
    try:
        if _DB_CONN is not None:
            _DB_CONN.close()
    except Exception:
        pass
    _DB_CONN = None
