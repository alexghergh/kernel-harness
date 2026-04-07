from __future__ import annotations

import argparse

from .common import emit_json, normalize_tool_name
from .completion_policy import annotate_completion_outcomes, infer_measured_outcome, substantial_budget_remaining
from .project import artifact_agent_dir, now_iso, write_json, write_text
from .archive_layout import archive_problem_contract_dir, history_path
from .goal_status import write_goal_status_files
from .run_metrics import best_correct_payload
from .workspace_paths import load_workspace_metadata, workspace_candidate_path, workspace_path


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
    snapshot = write_goal_status_files(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        workspace=workspace,
    )
    emit_json(snapshot)


def command_complete_problem(args: argparse.Namespace) -> None:
    workspace = workspace_path(args.workspace)
    metadata = load_workspace_metadata(workspace)
    tool = normalize_tool_name(metadata.get("tool"))
    agent_dir = artifact_agent_dir(args.run_name, args.level, args.problem_id)
    completion_path = agent_dir / "completion.json"
    if completion_path.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"Completion already exists at {completion_path}. Use --allow-overwrite to replace it."
        )

    snapshot = write_goal_status_files(
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        workspace=workspace,
    )
    if args.state == "stalled" and substantial_budget_remaining(snapshot):
        if int(snapshot.get("num_profile_runs") or 0) < 1:
            raise SystemExit(
                "Cannot record stalled while substantial budget remains and no profiler "
                "run has been recorded. Run ./bin/profile_ncu.sh on a strong candidate "
                "and try a new branch first."
            )

    measured_outcome = infer_measured_outcome(snapshot)
    payload = {
        "completed_at": now_iso(),
        "run_name": args.run_name,
        "level": args.level,
        "problem_id": args.problem_id,
        "tool": tool,
        "solver_state": args.state,
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
