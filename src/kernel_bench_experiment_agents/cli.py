from __future__ import annotations

import argparse

from .common import TOOL_CHOICES
from .execution_commands import command_profile_ncu, command_run_candidate
from .status_commands import (
    command_best_result,
    command_complete_problem,
    command_goal_status,
)
from .summary_commands import command_summarize_run
from .trace_commands import (
    command_materialize_agent_trace,
    command_sync_helper_agent_specs,
)
from .workspace_builder import (
    command_prepare_problem_workspace,
    command_problem_info,
)
from .workspace_contract import LAUNCHER_TERMINAL_STATES, SOLVER_TERMINAL_STATES

TERMINAL_STATE_CHOICES = tuple(SOLVER_TERMINAL_STATES + LAUNCHER_TERMINAL_STATES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-problem-workspace")
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--level", type=int, required=True)
    prepare.add_argument("--problem-id", type=int, required=True)
    prepare.add_argument("--dataset-src", default="local")
    prepare.add_argument("--kernelbench-root", default=None)
    prepare.add_argument("--kernelbench-python", required=True)
    prepare.add_argument("--workspace-root", default=None)
    prepare.add_argument("--gpu-name", default="")
    prepare.add_argument("--num-gpus", type=int, default=1)
    prepare.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    prepare.add_argument("--model", default="gpt-5-codex")
    prepare.add_argument("--time-budget-minutes", type=int, default=720)
    prepare.add_argument("--eager-baseline-file", required=True)
    prepare.add_argument("--compile-baseline-file", required=True)

    problem_info = subparsers.add_parser("problem-info")
    problem_info.add_argument("--level", type=int, required=True)
    problem_info.add_argument("--problem-id", type=int, required=True)
    problem_info.add_argument("--dataset-src", default="local")
    problem_info.add_argument("--kernelbench-root", default=None)

    run = subparsers.add_parser("run-candidate")
    run.add_argument("--candidate", required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--level", type=int, required=True)
    run.add_argument("--problem-id", type=int, required=True)
    run.add_argument("--dataset-src", default="local")
    run.add_argument("--kernelbench-root", default=None)
    run.add_argument("--gpu-id", type=int, default=None)
    run.add_argument("--num-gpu-slots", type=int, default=1)
    run.add_argument("--timing-method", default=None)
    run.add_argument("--backend", default="cuda")
    run.add_argument("--precision", default="fp32")
    run.add_argument("--num-correct-trials", type=int, default=5)
    run.add_argument("--num-perf-trials", type=int, default=100)
    run.add_argument("--prompt-path", default=None)
    run.add_argument("--workspace", default=None)

    profile = subparsers.add_parser("profile-ncu")
    profile.add_argument("--candidate", required=True)
    profile.add_argument("--run-name", required=True)
    profile.add_argument("--level", type=int, required=True)
    profile.add_argument("--problem-id", type=int, required=True)
    profile.add_argument("--dataset-src", default="local")
    profile.add_argument("--kernelbench-root", default=None)
    profile.add_argument("--gpu-id", type=int, default=None)
    profile.add_argument("--num-gpu-slots", type=int, default=1)
    profile.add_argument("--sample-id", type=int, default=None)
    profile.add_argument("--ncu-set", default="full")
    profile.add_argument("--workspace", default=None)

    best = subparsers.add_parser("best-result")
    best.add_argument("--run-name", required=True)
    best.add_argument("--level", type=int, required=True)
    best.add_argument("--problem-id", type=int, required=True)

    goal = subparsers.add_parser("goal-status")
    goal.add_argument("--run-name", required=True)
    goal.add_argument("--level", type=int, required=True)
    goal.add_argument("--problem-id", type=int, required=True)
    goal.add_argument("--workspace", required=True)

    complete = subparsers.add_parser("complete-problem")
    complete.add_argument("--run-name", required=True)
    complete.add_argument("--level", type=int, required=True)
    complete.add_argument("--problem-id", type=int, required=True)
    complete.add_argument("--workspace", required=True)
    complete.add_argument("--state", required=True, choices=TERMINAL_STATE_CHOICES)
    complete.add_argument("--summary", default="")
    complete.add_argument("--allow-overwrite", action="store_true")

    sync = subparsers.add_parser("sync-helper-agent-specs")
    sync.add_argument("--workspace", required=True)
    sync.add_argument("--archive-contract-dir", default=None)

    trace = subparsers.add_parser("materialize-agent-trace")
    trace.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    trace.add_argument("--events-path", required=True)
    trace.add_argument("--output-path", required=True)
    trace.add_argument("--completion-path", default=None)
    trace.add_argument("--final-message-path", default=None)
    trace.add_argument("--workspace", default=None)

    legacy_trace = subparsers.add_parser("materialize-codex-trace")
    legacy_trace.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    legacy_trace.add_argument("--events-path", required=True)
    legacy_trace.add_argument("--output-path", required=True)
    legacy_trace.add_argument("--completion-path", default=None)
    legacy_trace.add_argument("--final-message-path", default=None)
    legacy_trace.add_argument("--workspace", default=None)

    summary = subparsers.add_parser("summarize-run")
    summary.add_argument("--run-name", required=True)
    summary.add_argument("--level", type=int, action="append", default=[])
    summary.add_argument("--problem-id", type=int, action="append", default=[])
    summary.add_argument("--dataset-src", default="local")
    summary.add_argument("--kernelbench-root", default=None)
    summary.add_argument("--eager-baseline-file", default=None)
    summary.add_argument("--compile-baseline-file", default=None)
    summary.add_argument("--pass-k", default="1,5,10")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "prepare-problem-workspace": command_prepare_problem_workspace,
        "problem-info": command_problem_info,
        "run-candidate": command_run_candidate,
        "profile-ncu": command_profile_ncu,
        "best-result": command_best_result,
        "goal-status": command_goal_status,
        "complete-problem": command_complete_problem,
        "sync-helper-agent-specs": command_sync_helper_agent_specs,
        "materialize-agent-trace": command_materialize_agent_trace,
        "materialize-codex-trace": command_materialize_agent_trace,
        "summarize-run": command_summarize_run,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
