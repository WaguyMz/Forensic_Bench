from __future__ import annotations

import json
import os
import pathlib
import selectors
import subprocess
import sys
import threading
import time
from typing import List, Optional

from .context import get_plot_output_dir

# Thread-local storage: each parallel worker thread gets its own subprocess so
# concurrent code_interpreter calls never share stdin/stdout and reset_e2b_sandbox
# in one thread cannot kill another thread's live process.
_THREAD_LOCAL = threading.local()


def _get_thread_worker_proc() -> Optional[subprocess.Popen[str]]:
    return getattr(_THREAD_LOCAL, "worker", None)


def _set_thread_worker_proc(proc: Optional[subprocess.Popen[str]]) -> None:
    _THREAD_LOCAL.worker = proc


def reset_e2b_sandbox() -> None:
    """Stop the calling thread's persistent code_interpreter worker (if any)."""
    proc = _get_thread_worker_proc()
    _set_thread_worker_proc(None)
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def code_interpreter(code: str, upload_csvs: bool = True) -> str:
    """
    Execute Python code in a persistent local subprocess (per experiment run).

    Best-effort file I/O guard: only allow file reads/writes under the run output
    directory. If a call times out, the worker is killed and restarted on the
    next call.
    """
    _ = upload_csvs  # local backend reads output dir directly

    out_dir = pathlib.Path(get_plot_output_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    before = {p.name for p in out_dir.glob("*")}
    t0 = time.time()

    proc = _get_worker(out_dir)

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    # Make it harder to accidentally use proxies/credentials.
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("NO_PROXY", None)

    timeout_s = float(os.environ.get("FORENSIC_CODE_INTERPRETER_TIMEOUT_S", "300"))

    request = {"code": code, "allowed_root": str(out_dir)}
    try:
        stdout, stderr, returncode = _worker_request(proc, request, timeout_s=timeout_s)
    except TimeoutError:
        reset_e2b_sandbox()
        after = {p.name for p in out_dir.glob("*")}
        new_files = sorted(after - before)
        elapsed = time.time() - t0
        out_parts: List[str] = [
            "=== stderr ===\nExecution timed out; worker killed. Re-run will start a fresh worker."
        ]
        if new_files:
            out_parts.append(
                "=== saved files ===\n" + "\n".join(str(out_dir / n) for n in new_files)
            )
        header = f"[CODE_INTERPRETER] exit_code=124 elapsed_s={elapsed:.2f}"
        return header + "\n" + "\n\n".join(out_parts)

    after = {p.name for p in out_dir.glob("*")}
    new_files = sorted(after - before)
    elapsed = time.time() - t0

    out_parts: List[str] = []
    if stdout and stdout.strip():
        out_parts.append(f"=== stdout ===\n{stdout.strip()}")
    if stderr and stderr.strip():
        out_parts.append(f"=== stderr ===\n{stderr.strip()}")
    if new_files:
        out_parts.append(
            "=== saved files ===\n" + "\n".join(str(out_dir / n) for n in new_files)
        )

    header = f"[CODE_INTERPRETER] exit_code={returncode} elapsed_s={elapsed:.2f}"
    body = "\n\n".join(out_parts) if out_parts else "(no output)"
    return header + "\n" + body


def _get_worker(out_dir: pathlib.Path) -> subprocess.Popen[str]:
    proc = _get_thread_worker_proc()
    if proc is not None and proc.poll() is None:
        return proc

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("NO_PROXY", None)

    py = os.environ.get("FORENSIC_CODE_INTERPRETER_PYTHON", sys.executable)
    worker_code = _worker_program()
    proc = subprocess.Popen(
        [py, "-u", "-c", worker_code],
        cwd=str(out_dir),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _set_thread_worker_proc(proc)
    return proc


def _worker_request(
    proc: subprocess.Popen[str], request: dict, timeout_s: float
) -> tuple[str, str, int]:
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("Worker process missing stdin/stdout pipes.")

    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    deadline = time.time() + timeout_s
    sel = selectors.DefaultSelector()
    try:
        sel.register(proc.stdout, selectors.EVENT_READ)
        line: str = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            remaining = max(0.0, deadline - time.time())
            events = sel.select(timeout=min(0.25, remaining))
            if not events:
                continue
            # stdout has data available; safe to readline without blocking forever
            line = proc.stdout.readline()
            if line:
                break
        if not line:
            raise TimeoutError("No response from worker before timeout.")
    finally:
        try:
            sel.close()
        except Exception:
            pass

    msg = json.loads(line)
    return (
        msg.get("stdout", "") or "",
        msg.get("stderr", "") or "",
        int(msg.get("exit_code", 0)),
    )


def _worker_program() -> str:
    return r"""
import builtins
import contextlib
import io
import json
import os
import pathlib
import sys
import traceback

STATE = {}

_WRITE_MODES = frozenset("wax")

def _install_guard(allowed_root: pathlib.Path) -> None:
    allowed_root = allowed_root.resolve()
    _orig_open = builtins.open

    def _is_write(mode: str) -> bool:
        return bool(_WRITE_MODES & set(mode))

    def _guard_path(p):
        pp = pathlib.Path(p).expanduser()
        if not pp.is_absolute():
            pp = (pathlib.Path.cwd() / pp)
        pp = pp.resolve()
        try:
            pp.relative_to(allowed_root)
        except Exception as e:
            raise PermissionError(f"Write access denied outside {allowed_root}: {pp}") from e
        return pp

    def open(file, mode="r", *args, **kwargs):
        # Only restrict writes; reads from library/system paths are allowed.
        if _is_write(mode):
            _guard_path(file)
        return _orig_open(file, mode, *args, **kwargs)

    builtins.open = open

    _orig_path_open = pathlib.Path.open
    def _path_open(self, mode="r", *args, **kwargs):
        if _is_write(mode):
            _guard_path(self)
        return _orig_path_open(self, mode, *args, **kwargs)
    pathlib.Path.open = _path_open

def _run(code: str, allowed_root: str) -> dict:
    out = io.StringIO()
    err = io.StringIO()
    exit_code = 0
    try:
        root = pathlib.Path(allowed_root).resolve()
        os.chdir(str(root))
        _install_guard(root)
        STATE["OUTPUT_DIR"] = str(root)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(code, STATE, STATE)
    except Exception:
        exit_code = 1
        err.write(traceback.format_exc())
    return {"stdout": out.getvalue(), "stderr": err.getvalue(), "exit_code": exit_code}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    resp = _run(req.get("code", ""), req.get("allowed_root", os.getcwd()))
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"""


def _build_wrapper(user_code: str, allowed_root: pathlib.Path) -> str:
    root = str(allowed_root)
    # Write-only guard: restrict builtins.open + pathlib.Path.open *writes* to allowed_root.
    # Reads from library/system paths are left unrestricted so imports work normally.
    return f"""\
import builtins
import os
import pathlib
import sys

ALLOWED_ROOT = pathlib.Path({root!r}).resolve()
_WRITE_MODES = frozenset("wax")
_orig_open = builtins.open

def _guard_path(p: str | os.PathLike) -> pathlib.Path:
    pp = pathlib.Path(p).expanduser()
    if not pp.is_absolute():
        pp = (pathlib.Path.cwd() / pp)
    pp = pp.resolve()
    try:
        pp.relative_to(ALLOWED_ROOT)
    except Exception as e:
        raise PermissionError(f"Write access denied outside {{ALLOWED_ROOT}}: {{pp}}") from e
    return pp

def open(file, mode="r", *args, **kwargs):
    if _WRITE_MODES & set(mode):
        _guard_path(file)
    return _orig_open(file, mode, *args, **kwargs)

builtins.open = open

_orig_path_open = pathlib.Path.open
def _path_open(self, mode="r", *args, **kwargs):
    if _WRITE_MODES & set(mode):
        _guard_path(self)
    return _orig_path_open(self, mode, *args, **kwargs)
pathlib.Path.open = _path_open

# expose helper for user code
OUTPUT_DIR = str(ALLOWED_ROOT)

{user_code}
"""
