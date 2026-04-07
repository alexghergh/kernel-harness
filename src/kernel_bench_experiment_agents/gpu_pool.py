from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .project import artifact_lock_path, gpu_lock_dir, now_iso


@dataclass
class GPULease:
    slot_id: int
    lock_path: str
    wait_seconds: float
    device_selector: str
    isolated_visible_devices: str
    logical_gpu_id: int
    selector_source: str


@dataclass
class ArtifactLease:
    lock_path: str
    wait_seconds: float


@contextmanager
def lease_gpu_slot(
    *,
    num_slots: int,
    requested_slot: int | None,
    lease_name: str,
    poll_interval_seconds: float = 2.0,
    max_wait_seconds: float | None = None,
) -> Iterator[GPULease]:
    if num_slots <= 0:
        raise RuntimeError("num_slots must be >= 1")

    selectors, selector_source = resolve_gpu_device_selectors(num_slots=num_slots)
    if requested_slot is not None and (requested_slot < 0 or requested_slot >= len(selectors)):
        raise RuntimeError(
            f"requested GPU slot {requested_slot} is outside the configured range 0..{len(selectors) - 1}"
        )

    lock_root = gpu_lock_dir()
    slot_ids = [requested_slot] if requested_slot is not None else list(range(len(selectors)))
    started = time.monotonic()
    effective_max_wait_seconds = _gpu_lease_timeout_seconds(max_wait_seconds)

    while True:
        for slot_id in slot_ids:
            device_selector = selectors[slot_id]
            lock_path = lock_root / f"gpu_{slot_id}.lock"
            handle = _try_lock(
                lock_path,
                payload={
                    "pid": os.getpid(),
                    "slot_id": slot_id,
                    "lease_name": lease_name,
                    "device_selector": device_selector,
                    "selector_source": selector_source,
                    "acquired_at": now_iso(),
                },
            )
            if handle is None:
                continue

            wait_seconds = time.monotonic() - started
            lease = GPULease(
                slot_id=slot_id,
                lock_path=str(lock_path),
                wait_seconds=wait_seconds,
                device_selector=device_selector,
                isolated_visible_devices=device_selector,
                logical_gpu_id=0,
                selector_source=selector_source,
            )
            try:
                yield lease
            finally:
                _unlock(handle)
            return

        if (
            effective_max_wait_seconds is not None
            and time.monotonic() - started >= effective_max_wait_seconds
        ):
            raise RuntimeError(
                "Timed out waiting for a GPU slot lease after "
                f"{effective_max_wait_seconds:.1f} seconds. "
                f"Current lock holders: {_gpu_lock_snapshot(slot_ids)}"
            )

        time.sleep(poll_interval_seconds)


@contextmanager
def lease_problem_artifacts(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    lease_name: str,
    poll_interval_seconds: float = 0.25,
    max_wait_seconds: float | None = None,
) -> Iterator[ArtifactLease]:
    lock_path = artifact_lock_path(run_name, level, problem_id)
    started = time.monotonic()
    effective_max_wait_seconds = _artifact_lease_timeout_seconds(max_wait_seconds)

    while True:
        handle = _try_lock(
            lock_path,
            payload={
                "pid": os.getpid(),
                "run_name": run_name,
                "level": level,
                "problem_id": problem_id,
                "lease_name": lease_name,
                "acquired_at": now_iso(),
            },
        )
        if handle is not None:
            wait_seconds = time.monotonic() - started
            lease = ArtifactLease(
                lock_path=str(lock_path),
                wait_seconds=wait_seconds,
            )
            try:
                yield lease
            finally:
                _unlock(handle)
            return

        if (
            effective_max_wait_seconds is not None
            and time.monotonic() - started >= effective_max_wait_seconds
        ):
            raise RuntimeError(
                "Timed out waiting for an artifact lease after "
                f"{effective_max_wait_seconds:.1f} seconds. "
                f"Current lock holder: {_read_lock_payload(lock_path)}"
            )

        time.sleep(poll_interval_seconds)


def resolve_gpu_device_selectors(*, num_slots: int) -> tuple[list[str], str]:
    explicit = os.environ.get("KBE_VISIBLE_GPU_DEVICES", "").strip()
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if explicit:
        selectors = _parse_gpu_selector_list(explicit, env_name="KBE_VISIBLE_GPU_DEVICES")
        source = "KBE_VISIBLE_GPU_DEVICES"
    elif inherited:
        selectors = _parse_gpu_selector_list(inherited, env_name="CUDA_VISIBLE_DEVICES")
        source = "CUDA_VISIBLE_DEVICES"
    else:
        selectors = [str(slot_id) for slot_id in range(num_slots)]
        source = "default_range"

    if len(selectors) < num_slots:
        raise RuntimeError(
            f"Configured num_slots={num_slots}, but {source} exposes only {len(selectors)} visible GPU selector(s): {selectors}"
        )
    return selectors[:num_slots], source


def isolated_gpu_environment(*, device_selector: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_selector
    env["KBE_VISIBLE_GPU_DEVICES"] = device_selector
    env["KBE_LEASED_GPU_SELECTOR"] = device_selector
    return env


def _parse_gpu_selector_list(raw_value: str, *, env_name: str) -> list[str]:
    if raw_value == "-1":
        raise RuntimeError(f"{env_name} disables CUDA visibility (-1); no GPU slots are available.")
    selectors = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not selectors:
        raise RuntimeError(f"{env_name} did not contain any usable GPU selectors: {raw_value!r}")
    return selectors


def _try_lock(lock_path: Path, *, payload: dict[str, object]):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None

    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, sort_keys=True))
    handle.flush()
    return handle


def _gpu_lease_timeout_seconds(explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit

    raw_value = os.environ.get("KBE_GPU_LEASE_MAX_WAIT_SECONDS", "").strip()
    if not raw_value:
        return 1800.0

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "KBE_GPU_LEASE_MAX_WAIT_SECONDS must be a positive number or 0 to disable the timeout."
        ) from exc
    if parsed <= 0:
        return None
    return parsed


def _artifact_lease_timeout_seconds(explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit

    raw_value = os.environ.get("KBE_ARTIFACT_LEASE_MAX_WAIT_SECONDS", "").strip()
    if not raw_value:
        return 1800.0

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "KBE_ARTIFACT_LEASE_MAX_WAIT_SECONDS must be a positive number or 0 to disable the timeout."
        ) from exc
    if parsed <= 0:
        return None
    return parsed


def _read_lock_payload(lock_path: Path) -> dict[str, object] | None:
    if not lock_path.exists():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(payload, dict):
        return payload
    return {"raw": raw}


def _gpu_lock_snapshot(slot_ids: list[int]) -> list[dict[str, object]]:
    lock_root = gpu_lock_dir()
    snapshot: list[dict[str, object]] = []
    for slot_id in slot_ids:
        payload = _read_lock_payload(lock_root / f"gpu_{slot_id}.lock")
        if payload is None:
            snapshot.append({"slot_id": slot_id, "status": "unlocked"})
            continue
        payload = dict(payload)
        payload.setdefault("slot_id", slot_id)
        snapshot.append(payload)
    return snapshot


def _unlock(handle) -> None:
    handle.seek(0)
    handle.truncate()
    handle.flush()
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
