"""Expose the stable command-line entrypoints that scripts, wrappers, and the MCP server call.

The launcher uses internal commands here while the solver sees only the narrower MCP problem surface.
"""

from __future__ import annotations

import argparse

from kernel_bench_experiment_agents.kernelbench.commands.run_candidate import command_run_candidate
from kernel_bench_experiment_agents.runtime.common import TOOL_CHOICES
from kernel_bench_experiment_agents.kernelbench.commands.profile import command_profile_ncu
from kernel_bench_experiment_agents.surface.policy import LAUNCHER_TERMINAL_STATES
from kernel_bench_experiment_agents.kernelbench.commands.status import (
    command_best_result,
    command_complete_problem,
    command_goal_status,
    command_record_launcher_completion,
)
from kernel_bench_experiment_agents.summary.commands import command_summarize_run
from kernel_bench_experiment_agents.trace.commands import command_materialize_agent_trace
from kernel_bench_experiment_agents.workspace.prepare import command_prepare_problem_workspace


# The solver-facing completion command is summary-only; launcher-only states are
# routed through a separate internal command so the workspace agent cannot choose
# budget or harness failure outcomes for itself.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbharness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-problem-workspace")
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--level", type=int, required=True)
    prepare.add_argument("--problem-id", type=int, required=True)
    prepare.add_argument("--dataset-src", default="local")
    prepare.add_argument("--kernelbench-root", default=None)
    prepare.add_argument("--hardware-name", default="")
    prepare.add_argument("--timings-dir", default=None)
    prepare.add_argument("--num-gpus", type=int, default=1)
    prepare.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    prepare.add_argument("--model", default="gpt-5.4")
    prepare.add_argument("--time-budget-minutes", type=int, default=720)
    prepare.add_argument("--precision", default="bf16")

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
    run.add_argument("--precision", default="bf16")
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
    profile.add_argument("--precision", default="bf16")
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
    complete.add_argument("--summary", required=True)
    complete.add_argument("--allow-overwrite", action="store_true")

    launcher_complete = subparsers.add_parser("record-launcher-completion")
    launcher_complete.add_argument("--run-name", required=True)
    launcher_complete.add_argument("--level", type=int, required=True)
    launcher_complete.add_argument("--problem-id", type=int, required=True)
    launcher_complete.add_argument("--workspace", required=True)
    launcher_complete.add_argument("--state", required=True, choices=LAUNCHER_TERMINAL_STATES)
    launcher_complete.add_argument("--summary", required=True)
    launcher_complete.add_argument("--allow-overwrite", action="store_true")

    trace = subparsers.add_parser("materialize-agent-trace")
    trace.add_argument("--tool", choices=TOOL_CHOICES, default="codex")
    trace.add_argument("--events-path", required=True)
    trace.add_argument("--mcp-events-path", default=None)
    trace.add_argument("--output-path", required=True)
    trace.add_argument("--completion-path", default=None)
    trace.add_argument("--final-message-path", default=None)
    trace.add_argument("--workspace", default=None)

    summary = subparsers.add_parser("summarize-run")
    summary.add_argument("--run-name", required=True)
    summary.add_argument("--level", type=int, action="append", default=[])
    summary.add_argument("--problem-id", type=int, action="append", default=[])
    summary.add_argument("--pass-k", default="1,5,10")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "prepare-problem-workspace": command_prepare_problem_workspace,
        "run-candidate": command_run_candidate,
        "profile-ncu": command_profile_ncu,
        "best-result": command_best_result,
        "goal-status": command_goal_status,
        "complete-problem": command_complete_problem,
        "record-launcher-completion": command_record_launcher_completion,
        "materialize-agent-trace": command_materialize_agent_trace,
        "summarize-run": command_summarize_run,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
