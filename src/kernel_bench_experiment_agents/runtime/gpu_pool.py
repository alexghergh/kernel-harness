"""Manage GPU-slot and per-problem artifact leases for run and profile commands.

The launcher and measurement commands depend on these helpers to serialize archive writes and bind work to one visible GPU.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from kernel_bench_experiment_agents.runtime.project import artifact_lock_path, gpu_lock_dir, now_iso

GPU_LEASE_MAX_WAIT_SECONDS = 1800.0
ARTIFACT_LEASE_MAX_WAIT_SECONDS = 1800.0


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
            lock_path = _gpu_lock_path(lock_root, device_selector)
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
                f"Current lock holders: {_gpu_lock_snapshot(selectors, slot_ids)}"
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
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if inherited:
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
    cuda_home = discover_cuda_home()
    if cuda_home:
        env.setdefault("CUDA_HOME", cuda_home)
        env.setdefault("CUDA_PATH", cuda_home)
        _prepend_env_path(env, "PATH", str(Path(cuda_home) / "bin"))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(cuda_home) / "lib64"))
    return env


@lru_cache(maxsize=1)
def discover_cuda_home() -> str | None:
    explicit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit:
        resolved = _validated_cuda_home(Path(explicit))
        if resolved is not None:
            return resolved

    try:
        from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home
    except Exception:
        torch_cuda_home = None
    if torch_cuda_home:
        resolved = _validated_cuda_home(Path(torch_cuda_home))
        if resolved is not None:
            return resolved

    nvcc = shutil.which("nvcc")
    if nvcc:
        resolved = _validated_cuda_home(Path(nvcc).resolve().parent.parent)
        if resolved is not None:
            return resolved

    search_roots = (
        Path("/mnt/nfs/packages/x86_64/cuda"),
        Path("/usr/local"),
        Path("/opt"),
    )
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("cuda*"), key=_cuda_home_sort_key, reverse=True):
            resolved = _validated_cuda_home(candidate)
            if resolved is not None:
                return resolved
    return None


def _validated_cuda_home(path: Path) -> str | None:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    if (resolved / "bin" / "nvcc").exists():
        return str(resolved)
    if (resolved / "include" / "cuda.h").exists():
        return str(resolved)
    return None


def _cuda_home_sort_key(path: Path) -> tuple[int, tuple[int, ...], str]:
    name = path.name
    if name == "cuda":
        return (2, (), name)

    version_text = name
    if version_text.startswith("cuda-"):
        version_text = version_text[5:]
    elif version_text.startswith("cuda"):
        version_text = version_text[4:]

    parts: list[int] = []
    for token in re.split(r"[^0-9]+", version_text):
        if token:
            parts.append(int(token))
    if parts:
        return (1, tuple(parts), name)
    return (0, (), name)


def _prepend_env_path(env: dict[str, str], name: str, entry: str) -> None:
    if not Path(entry).exists():
        return
    current = env.get(name, "")
    parts = [value for value in current.split(os.pathsep) if value]
    if entry in parts:
        return
    env[name] = os.pathsep.join([entry, *parts]) if parts else entry


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
    return GPU_LEASE_MAX_WAIT_SECONDS


def _artifact_lease_timeout_seconds(explicit: float | None) -> float | None:
    if explicit is not None:
        return explicit
    return ARTIFACT_LEASE_MAX_WAIT_SECONDS


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


def _gpu_lock_path(lock_root: Path, device_selector: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", device_selector)
    if not slug:
        raise RuntimeError(f"GPU device selector {device_selector!r} is not usable as a lock key")
    return lock_root / f"gpu_selector_{slug}.lock"


def _gpu_lock_snapshot(selectors: list[str], slot_ids: list[int]) -> list[dict[str, object]]:
    lock_root = gpu_lock_dir()
    snapshot: list[dict[str, object]] = []
    for slot_id in slot_ids:
        device_selector = selectors[slot_id]
        payload = _read_lock_payload(_gpu_lock_path(lock_root, device_selector))
        if payload is None:
            snapshot.append(
                {
                    "slot_id": slot_id,
                    "device_selector": device_selector,
                    "status": "unlocked",
                }
            )
            continue
        payload = dict(payload)
        payload.setdefault("slot_id", slot_id)
        payload.setdefault("device_selector", device_selector)
        snapshot.append(payload)
    return snapshot


def _unlock(handle) -> None:
    handle.seek(0)
    handle.truncate()
    handle.flush()
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
