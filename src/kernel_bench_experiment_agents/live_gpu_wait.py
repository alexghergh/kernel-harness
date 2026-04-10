"""Track in-flight GPU-lease wait time so live budget accounting can exclude it immediately.

Run and profile wrappers create short-lived marker files here while they are blocked on a GPU slot.
Once a lease is acquired, the marker is converted into a fixed wait-duration record and kept alive
until the command persists its normal archive payload, so the budget view never "gives back" the
queued time in the middle of a long run or profile.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .project import ensure_dir, now_iso, state_dir, validate_run_name, write_json


def _problem_runtime_dir(run_name: str, level: int, problem_id: int) -> Path:
    run_name = validate_run_name(run_name)
    return ensure_dir(
        state_dir() / "problem_runtime" / run_name / f"level_{level}" / f"problem_{problem_id}"
    )


def live_gpu_wait_dir(run_name: str, level: int, problem_id: int) -> Path:
    return ensure_dir(_problem_runtime_dir(run_name, level, problem_id) / "live_gpu_wait")


def create_live_gpu_wait_marker(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    operation: str,
    requested_gpu: int | None,
    num_gpu_slots: int,
) -> Path:
    """Create one transient marker for an in-flight GPU lease wait."""
    marker_dir = live_gpu_wait_dir(run_name, level, problem_id)
    marker_path = marker_dir / f"{operation}_{os.getpid()}_{uuid.uuid4().hex}.json"
    payload = {
        "created_at": now_iso(),
        "started_at": now_iso(),
        "started_epoch_seconds": time.time(),
        "pid": os.getpid(),
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "operation": operation,
        "requested_gpu": requested_gpu,
        "num_gpu_slots": num_gpu_slots,
    }
    write_json(marker_path, payload)
    return marker_path


def settle_live_gpu_wait_marker(marker_path: Path | None, *, wait_seconds: float) -> None:
    """Freeze the queued wait once a GPU lease is acquired but before payload persistence."""
    if marker_path is None:
        return
    payload = _load_json_object(marker_path) or {}
    payload["settled_at"] = now_iso()
    payload["settled_wait_seconds"] = max(0.0, float(wait_seconds))
    write_json(marker_path, payload)


def clear_live_gpu_wait_marker(marker_path: Path | None) -> None:
    if marker_path is None:
        return
    try:
        marker_path.unlink()
    except FileNotFoundError:
        return


def active_live_gpu_wait_seconds(run_name: str, level: int, problem_id: int) -> float:
    """Return the currently active queued GPU-wait time for one problem.

    In the normal design there is only one live measurement wrapper per problem at a time. We still
    sum marker contributions here so fixed settled waits continue to count during long in-flight
    runs/profiles until their archive payloads are written.
    """
    marker_dir = live_gpu_wait_dir(run_name, level, problem_id)
    total_seconds = 0.0
    for marker_path in marker_dir.glob("*.json"):
        payload = _load_json_object(marker_path)
        if payload is None:
            continue
        pid = payload.get("pid")
        if isinstance(pid, int) and pid > 0 and not _pid_is_alive(pid):
            clear_live_gpu_wait_marker(marker_path)
            continue

        settled_wait_seconds = payload.get("settled_wait_seconds")
        if isinstance(settled_wait_seconds, (int, float)):
            total_seconds += max(0.0, float(settled_wait_seconds))
            continue

        started_epoch = _started_epoch_from_payload(payload)
        if started_epoch is None:
            continue
        total_seconds += max(0.0, time.time() - started_epoch)

    return total_seconds


def _load_started_epoch(marker_path: Path) -> float | None:
    payload = _load_json_object(marker_path)
    if payload is None:
        return None
    return _started_epoch_from_payload(payload)


def _started_epoch_from_payload(payload: dict[str, Any]) -> float | None:
    started_epoch = payload.get("started_epoch_seconds")
    if isinstance(started_epoch, (int, float)):
        return float(started_epoch)

    started_at = payload.get("started_at")
    if not isinstance(started_at, str) or not started_at.strip():
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started.astimezone(timezone.utc).timestamp()


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

# Backward-compatible aliases while the wrapper commands settle on the marker naming.
begin_live_gpu_wait = create_live_gpu_wait_marker
clear_live_gpu_wait = clear_live_gpu_wait_marker
