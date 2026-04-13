"""Implement status, best-result, and completion commands for one problem workspace.

These commands are the bridge between solver-visible wrappers and the durable completion artifacts written into the archive.
"""

from __future__ import annotations

import argparse

from .archive_layout import archive_problem_contract_dir, sample_manifest_entries
from .common import emit_json, normalize_tool_name
from .completion_policy import annotate_completion_outcomes, infer_measured_outcome
from .goal_status import write_goal_status_files
from .gpu_pool import lease_problem_artifacts
from .policy_model import SOLVER_TERMINAL_STATES
from .project import archive_agent_dir, now_iso, write_json, write_text
from .run_metrics import best_correct_payload
from .workspace_paths import (
    validate_workspace_assignment,
    workspace_candidate_path,
    workspace_path,
)


def command_best_result(args: argparse.Namespace) -> None:
    entries = sample_manifest_entries(args.run_name, args.level, args.problem_id)
    if not entries:
        raise SystemExit("No measured attempt manifests were found in archive/attempts.")

    best_payload = best_correct_payload(entries)
    if best_payload is None:
        raise SystemExit("No correct runtime-bearing results were found in the attempt manifests.")
    emit_json(best_payload)


# Goal status is regenerated under the per-problem artifact lock so the solver sees
# the latest measured state before it decides whether to continue or complete.
def command_goal_status(args: argparse.Namespace) -> None:
    workspace = workspace_path(args.workspace)
    validate_workspace_assignment(
        workspace,
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
    )
    with lease_problem_artifacts(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        lease_name=f"goal_status:{args.run_name}:level_{args.level}:problem_{args.problem_id}",
    ):
        snapshot = write_goal_status_files(
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            workspace=workspace,
        )
    emit_json(snapshot)


def _write_completion_payload(
    *,
    args: argparse.Namespace,
    terminal_state: str,
    summary: str,
) -> dict[str, object]:
    workspace = workspace_path(args.workspace)
    metadata = validate_workspace_assignment(
        workspace,
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
    )
    tool = normalize_tool_name(metadata.get("tool"))
    agent_dir = archive_agent_dir(args.run_name, args.level, args.problem_id)
    completion_path = agent_dir / "completion.json"
    if completion_path.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"Completion already exists at {completion_path}. Use --allow-overwrite to replace it."
        )

    with lease_problem_artifacts(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        lease_name=f"complete:{args.run_name}:level_{args.level}:problem_{args.problem_id}",
    ):
        snapshot = write_goal_status_files(
            run_name=args.run_name,
            level=args.level,
            problem_id=args.problem_id,
            workspace=workspace,
        )
        measured_outcome = infer_measured_outcome(snapshot)
        payload = {
            "completed_at": now_iso(),
            "run_name": args.run_name,
            "level": args.level,
            "problem_id": args.problem_id,
            "tool": tool,
            "solver_state": terminal_state if terminal_state in SOLVER_TERMINAL_STATES else None,
            "terminal_state": terminal_state,
            "measured_outcome": measured_outcome,
            "success": measured_outcome == "beats_both",
            "summary": summary,
            "goal_status": snapshot,
        }
        payload = annotate_completion_outcomes(payload)
        write_json(completion_path, payload)
        write_json(workspace / "completion.json", payload)
        candidate_path = workspace_candidate_path(workspace)
        if candidate_path.exists():
            write_text(
                archive_problem_contract_dir(args.run_name, args.level, args.problem_id)
                / "candidate_final.py",
                candidate_path.read_text(encoding="utf-8"),
            )
    return payload


# Solver completion is intentionally summary-only: the agent may report that it is
# done, but launcher-only states such as budget exhaustion are recorded elsewhere.
def command_complete_problem(args: argparse.Namespace) -> None:
    emit_json(_write_completion_payload(args=args, terminal_state="done", summary=args.summary))


# Launcher completion keeps the explicit terminal state surface needed for budget
# exhaustion and other non-solver endings.
def command_record_launcher_completion(args: argparse.Namespace) -> None:
    emit_json(
        _write_completion_payload(
            args=args,
            terminal_state=args.state,
            summary=args.summary,
        )
    )
