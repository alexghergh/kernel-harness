from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .project import (
    archive_attempts_dir,
    archive_contract_dir,
    archive_problem_dir,
    archive_profiles_dir,
    artifact_agent_dir,
    write_json,
)


def archive_problem_contract_dir(run_name: str, level: int, problem_id: int) -> Path:
    return archive_contract_dir(run_name, level, problem_id)


def archive_problem_attempts_dir(run_name: str, level: int, problem_id: int) -> Path:
    return archive_attempts_dir(run_name, level, problem_id)


def archive_problem_profiles_dir(run_name: str, level: int, problem_id: int) -> Path:
    return archive_profiles_dir(run_name, level, problem_id)


def history_path(run_name: str, level: int, problem_id: int) -> Path:
    return archive_problem_attempts_dir(run_name, level, problem_id) / "history.jsonl"


def sample_manifest_path(run_name: str, level: int, problem_id: int, sample_id: int) -> Path:
    return archive_problem_attempts_dir(run_name, level, problem_id) / f"sample_{sample_id}.json"


def goal_status_archive_path(run_name: str, level: int, problem_id: int) -> Path:
    return artifact_agent_dir(run_name, level, problem_id) / "goal_status.json"


def profile_index_path(run_name: str, level: int, problem_id: int) -> Path:
    return archive_problem_profiles_dir(run_name, level, problem_id) / "index.jsonl"


def trace_events_path(run_name: str, level: int, problem_id: int) -> Path:
    return artifact_agent_dir(run_name, level, problem_id) / "events.jsonl"


def archive_manifest_path(run_name: str, level: int, problem_id: int) -> Path:
    return archive_problem_dir(run_name, level, problem_id) / "archive_manifest.json"


def read_jsonl_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def history_entries(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_entries(path)


def profile_entries(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_entries(path)


def next_archive_profile_index(run_name: str, level: int, problem_id: int) -> int:
    profiles_dir = archive_problem_profiles_dir(run_name, level, problem_id)
    max_index = 0
    for child in profiles_dir.glob("profile_*.json"):
        match = re.fullmatch(r"profile_(\d+)\.json", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def build_archive_problem_manifest(run_name: str, level: int, problem_id: int) -> dict[str, Any]:
    problem_root = archive_problem_dir(run_name, level, problem_id)
    workspace_stub = f"state/workspaces/{run_name}/level_{level}/problem_{problem_id}"
    return {
        "schema_version": 1,
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "copy_this_directory": str(problem_root),
        "canonical_subdirs": {
            "contract": {
                "path": "contract/",
                "purpose": "exact rendered solver contract and frozen problem inputs",
            },
            "agent": {
                "path": "agent/",
                "purpose": "raw agent events, normalized trace IR, completion, audit, usage, and final message",
            },
            "attempts": {
                "path": "attempts/",
                "purpose": "measured attempt history plus candidate kernel snapshots and per-sample stdout/stderr",
            },
            "profiles": {
                "path": "profiles/",
                "purpose": "Nsight Compute metadata plus text/CSV exports",
            },
        },
        "canonical_files": [
            {"path": "archive_manifest.json", "purpose": "human/operator map of what in this problem archive is canonical"},
            {"path": "contract/problem.json", "purpose": "problem metadata and budget start time"},
            {"path": "contract/baseline.json", "purpose": "baseline runtimes for this problem"},
            {"path": "contract/hardware.json", "purpose": "frozen hardware facts"},
            {"path": "contract/workspace_contract.json", "purpose": "machine-readable solver contract"},
            {"path": "contract/AGENTS.md", "purpose": "rendered solver instructions"},
            {"path": "contract/SPEC.md", "purpose": "rendered problem spec"},
            {"path": "contract/HARDWARE.md", "purpose": "rendered hardware guidance"},
            {"path": "contract/INITIAL_PROMPT.md", "purpose": "exact initial solver prompt"},
            {"path": "contract/helper_agents/", "purpose": "rendered helper-agent specs for the chosen tool runtime"},
            {"path": "agent/events.jsonl", "purpose": "raw streamed agent trace"},
            {"path": "agent/trace_ir.json", "purpose": "normalized trace IR"},
            {"path": "agent/completion.json", "purpose": "final terminal state plus measured outcome"},
            {"path": "agent/final_message.txt", "purpose": "last model message when available"},
            {"path": "agent/goal_status.json", "purpose": "latest archived goal-status snapshot"},
            {"path": "attempts/history.jsonl", "purpose": "append-only measured attempt history"},
            {"path": "attempts/sample_<id>.json", "purpose": "per-attempt measured result metadata"},
            {"path": "attempts/sample_<id>.stdout.txt", "purpose": "evaluation subprocess stdout for an attempt"},
            {"path": "attempts/sample_<id>.stderr.txt", "purpose": "evaluation subprocess stderr for an attempt"},
            {"path": "attempts/kernels/", "purpose": "snapshots of candidate code submitted for measurement"},
            {"path": "profiles/index.jsonl", "purpose": "append-only profile history"},
            {"path": "profiles/profile_<id>.json", "purpose": "per-profile metadata"},
            {"path": "profiles/profile_<id>.summary.txt", "purpose": "first-pass text summary for the solver"},
            {"path": "profiles/profile_<id>.details.txt", "purpose": "full text export from ncu --page details"},
            {"path": "profiles/profile_<id>.raw.csv", "purpose": "raw CSV export from ncu --page raw --csv"},
            {"path": "profiles/profile_<id>.stdout.txt", "purpose": "profiling command stdout"},
            {"path": "profiles/profile_<id>.stderr.txt", "purpose": "profiling command stderr"},
        ],
        "workspace_mirrors_not_required_for_copy_out": [
            f"{workspace_stub}/samples/",
            f"{workspace_stub}/profiles/",
            f"{workspace_stub}/GOAL_STATUS.md",
            f"{workspace_stub}/goal_status.json",
            f"{workspace_stub}/completion.json",
            f"{workspace_stub}/.codex/agents/",
            f"{workspace_stub}/.claude/agents/",
        ],
        "optional_debug_files": [
            {
                "path_glob": "profiles/*.ncu-rep",
                "when_present": "only when KBE_KEEP_NCU_REP is enabled",
            }
        ],
    }


def write_archive_problem_manifest(run_name: str, level: int, problem_id: int) -> Path:
    path = archive_manifest_path(run_name, level, problem_id)
    write_json(path, build_archive_problem_manifest(run_name, level, problem_id))
    return path
