from __future__ import annotations

import argparse

from .common import emit_json, normalize_tool_name
from .completion_policy import annotate_completion_outcomes, infer_measured_outcome
from .project import artifact_agent_dir, now_iso, write_json, write_text
from .archive_layout import archive_problem_contract_dir, history_path
from .goal_status import write_goal_status_files
from .gpu_pool import lease_problem_artifacts
from .run_metrics import best_correct_payload
from .workspace_contract import SOLVER_TERMINAL_STATES
from .workspace_paths import (
    validate_workspace_assignment,
    workspace_candidate_path,
    workspace_path,
)


def command_best_result(args: argparse.Namespace) -> None:
    history_path_value = history_path(args.run_name, args.level, args.problem_id)
    if not history_path_value.exists():
        raise SystemExit(f"No history found at {history_path_value}")

    best_payload = best_correct_payload(history_path_value)
    if best_payload is None:
        raise SystemExit("No correct runtime-bearing results were found in history.jsonl")
    emit_json(best_payload)


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


def command_complete_problem(args: argparse.Namespace) -> None:
    workspace = workspace_path(args.workspace)
    metadata = validate_workspace_assignment(
        workspace,
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
    )
    tool = normalize_tool_name(metadata.get("tool"))
    agent_dir = artifact_agent_dir(args.run_name, args.level, args.problem_id)
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
            "solver_state": args.state if args.state in SOLVER_TERMINAL_STATES else None,
            "terminal_state": args.state,
            "measured_outcome": measured_outcome,
            "success": measured_outcome == "beats_both",
            "summary": args.summary,
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
    emit_json(payload)
