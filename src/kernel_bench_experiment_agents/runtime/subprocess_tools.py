"""Wrap subprocess execution and error serialization for measured run and profile commands.

The rest of the harness uses these helpers to keep stdout/stderr capture and failure reporting uniform.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kernel_bench_experiment_agents.runtime.project import now_iso


PROCESS_TERMINATION_GRACE_SECONDS = 10.0


@dataclass
class SubprocessStart:
    pid: int
    pgid: int | None
    started_at: str
    timeout_seconds: float | None


@dataclass
class SubprocessResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    pid: int
    pgid: int | None
    started_at: str
    finished_at: str
    duration_seconds: float
    timeout_seconds: float | None
    timed_out: bool = False


class SubprocessTimeoutError(RuntimeError):
    def __init__(self, result: SubprocessResult):
        self.result = result
        command = " ".join(result.args)
        super().__init__(
            f"Subprocess timed out after {result.timeout_seconds:.1f} seconds "
            f"(pid={result.pid}, pgid={result.pgid}): {command}"
        )


def run_subprocess_capture(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: float | None = None,
    on_start: Callable[[SubprocessStart], None] | None = None,
) -> SubprocessResult:
    started_monotonic = time.monotonic()
    started_at = now_iso()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    start = _subprocess_start(process, started_at=started_at, timeout_seconds=timeout_seconds)
    _notify_start(process, start, on_start)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return SubprocessResult(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            pid=start.pid,
            pgid=start.pgid,
            started_at=started_at,
            finished_at=now_iso(),
            duration_seconds=time.monotonic() - started_monotonic,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, start.pgid)
        stdout, stderr = process.communicate()
        result = SubprocessResult(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            pid=start.pid,
            pgid=start.pgid,
            started_at=started_at,
            finished_at=now_iso(),
            duration_seconds=time.monotonic() - started_monotonic,
            timeout_seconds=timeout_seconds,
            timed_out=True,
        )
        raise SubprocessTimeoutError(result) from None


def run_subprocess_streaming(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: float | None = None,
    on_start: Callable[[SubprocessStart], None] | None = None,
) -> SubprocessResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    started_at = now_iso()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        start = _subprocess_start(process, started_at=started_at, timeout_seconds=timeout_seconds)
        _notify_start(process, start, on_start)
        try:
            returncode = process.wait(timeout=timeout_seconds)
            finished_at = now_iso()
            duration_seconds = time.monotonic() - started_monotonic
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, start.pgid)
            returncode = process.wait()
            result = SubprocessResult(
                args=command,
                returncode=returncode,
                stdout=_read_text_if_present(stdout_path),
                stderr=_read_text_if_present(stderr_path),
                pid=start.pid,
                pgid=start.pgid,
                started_at=started_at,
                finished_at=now_iso(),
                duration_seconds=time.monotonic() - started_monotonic,
                timeout_seconds=timeout_seconds,
                timed_out=True,
            )
            raise SubprocessTimeoutError(result) from None
    return SubprocessResult(
        args=command,
        returncode=returncode,
        stdout=_read_text_if_present(stdout_path),
        stderr=_read_text_if_present(stderr_path),
        pid=start.pid,
        pgid=start.pgid,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        timeout_seconds=timeout_seconds,
    )


def excerpt(text: str, *, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return "[truncated]...\n" + text[-limit:]


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object at {path}, got {type(payload).__name__}.")
    return payload


def serialize_exception(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }
    if isinstance(exc, SubprocessTimeoutError):
        payload["subprocess"] = subprocess_result_metadata(exc.result)
        payload["stdout_excerpt"] = excerpt(exc.result.stdout)
        payload["stderr_excerpt"] = excerpt(exc.result.stderr)
    return payload


def subprocess_start_metadata(start: SubprocessStart) -> dict[str, Any]:
    return {
        "pid": start.pid,
        "pgid": start.pgid,
        "started_at": start.started_at,
        "timeout_seconds": start.timeout_seconds,
    }


def subprocess_result_metadata(result: SubprocessResult) -> dict[str, Any]:
    return {
        "pid": result.pid,
        "pgid": result.pgid,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": result.duration_seconds,
        "timeout_seconds": result.timeout_seconds,
        "timed_out": result.timed_out,
        "returncode": result.returncode,
    }


def timeout_seconds_from_env(env_name: str, default: float) -> float | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default
    if raw_value.strip().lower() in {"0", "none", "disabled", "false"}:
        return None
    return float(raw_value)


def _subprocess_start(
    process: subprocess.Popen[str],
    *,
    started_at: str,
    timeout_seconds: float | None,
) -> SubprocessStart:
    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pgid = None
    return SubprocessStart(
        pid=process.pid,
        pgid=pgid,
        started_at=started_at,
        timeout_seconds=timeout_seconds,
    )


def _terminate_process_group(process: subprocess.Popen[str], pgid: int | None) -> None:
    if process.poll() is not None:
        return
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()


def _notify_start(
    process: subprocess.Popen[str],
    start: SubprocessStart,
    on_start: Callable[[SubprocessStart], None] | None,
) -> None:
    if on_start is None:
        return
    try:
        on_start(start)
    except Exception:
        _terminate_process_group(process, start.pgid)
        process.wait()
        raise


def _read_text_if_present(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
