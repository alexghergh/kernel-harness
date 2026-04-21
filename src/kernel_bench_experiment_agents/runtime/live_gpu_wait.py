"""Track in-flight GPU-lease wait time so live budget accounting can exclude it immediately.

Run and profile wrappers create short-lived marker files under `state/locks/live_gpu_wait/` while
waiting for a GPU slot. Once a lease is acquired, the marker is converted into a fixed wait-duration
record and kept alive until the command persists its normal archive payload, so the budget view never
"gives back" the queued time in the middle of a long run or profile.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.runtime.project import ensure_dir, locks_dir, now_iso, validate_run_name, write_json


def live_gpu_wait_dir() -> Path:
    return ensure_dir(locks_dir() / "live_gpu_wait")



def _problem_prefix(run_name: str, level: int, problem_id: int) -> str:
    run_name = validate_run_name(run_name)
    return f"{run_name}_level_{level}_problem_{problem_id}"



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
    prefix = _problem_prefix(run_name, level, problem_id)
    marker_path = (
        live_gpu_wait_dir() / f"{prefix}_{operation}_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
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
    total_seconds = 0.0
    prefix = _problem_prefix(run_name, level, problem_id)
    for marker_path in live_gpu_wait_dir().glob(f"{prefix}_*.json"):
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

        started_epoch_seconds = payload.get("started_epoch_seconds")
        if isinstance(started_epoch_seconds, (int, float)):
            total_seconds += max(0.0, time.time() - float(started_epoch_seconds))
            continue

        started_at = payload.get("started_at")
        parsed_seconds = _epoch_seconds_from_any(started_at)
        if parsed_seconds is not None:
            total_seconds += max(0.0, time.time() - parsed_seconds)

    return total_seconds



def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None



def _epoch_seconds_from_any(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None

    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        from datetime import datetime

        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            from datetime import timezone

            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None



def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

