"""Canonical typed policy for solver-visible harness behavior and runtime config.

This module is the single source of truth for the small set of policy objects that
must stay aligned across runtime config rendering, workspace docs, trace audit, and
helper-agent rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_contract import CANDIDATE_FILENAME


@dataclass(frozen=True)
class WrapperCommandSpec:
    """Describes one solver-visible wrapper command in the per-problem workspace."""

    name: str
    path: str
    purpose: str
    uses_gpu: bool = False


@dataclass(frozen=True)
class HelperAgentSpec:
    """Captures the shared intent for rendered Codex and Claude helper agents."""

    name: str
    description: str
    shell_commands: tuple[str, ...]
    read_paths: tuple[str, ...]
    summary_focus: str


ALLOWED_WEB_DOMAINS: tuple[str, ...] = ("docs.nvidia.com",)
SOLVER_TERMINAL_STATES: tuple[str, ...] = ("done",)
LAUNCHER_TERMINAL_STATES: tuple[str, ...] = (
    "budget_exhausted",
    "failed_to_generate",
)

WORKSPACE_COMMAND_SPECS: tuple[WrapperCommandSpec, ...] = (
    WrapperCommandSpec(
        name="hardware_info",
        path="./bin/hardware_info.sh",
        purpose="print frozen hardware facts for this workspace",
    ),
    WrapperCommandSpec(
        name="run_candidate",
        path="./bin/run_candidate.sh",
        purpose="evaluate correctness and runtime for the current candidate",
        uses_gpu=True,
    ),
    WrapperCommandSpec(
        name="profile_ncu",
        path="./bin/profile_ncu.sh",
        purpose="profile the current candidate with Nsight Compute",
        uses_gpu=True,
    ),
    WrapperCommandSpec(
        name="goal_status",
        path="./bin/goal_status.sh",
        purpose="refresh and print live goal status",
    ),
    WrapperCommandSpec(
        name="best_result",
        path="./bin/best_result.sh",
        purpose="print the best measured correct result so far",
    ),
    WrapperCommandSpec(
        name="complete_problem",
        path="./bin/complete_problem.sh",
        purpose="record a terminal completion summary",
    ),
)

WORKSPACE_STANDING_ORDERS: tuple[str, ...] = (
    "Work independently. There is no human approval, acceptance, or confirmation step during the run.",
    "Do not ask whether to proceed. Pick the next reasonable action yourself.",
    "Do not end with a plain assistant message. The only valid exit is ./bin/complete_problem.sh.",
    "After every measured run or profile, re-read GOAL_STATUS.md and keep iterating if it still says UNRESOLVED.",
    "If one branch fails, start another one. Failed attempts are normal, not a stop signal.",
)

WORKSPACE_STUCK_PROTOCOL: tuple[str, ...] = (
    "Re-read SPEC.md, HARDWARE.md, and GOAL_STATUS.md.",
    "Run ./bin/profile_ncu.sh if you do not already have profiling for the current idea.",
    "Read profiles/latest.summary.txt first, then profiles/latest.details.txt if needed.",
    "Use hosted web search only for docs.nvidia.com when you need CUDA or hardware guidance.",
    "Make a new implementation plan and continue without asking the user for permission.",
)

WORKSPACE_READ_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "SPEC.md",
    "HARDWARE.md",
    "GOAL_STATUS.md",
    "goal_status.json",
    "hardware.json",
    "workspace_contract.json",
    "problem.json",
    "problem_reference.py",
    CANDIDATE_FILENAME,
    "samples/",
    "profiles/",
)

WORKSPACE_EDIT_PATHS: tuple[str, ...] = (CANDIDATE_FILENAME,)

HELPER_SPECS: tuple[HelperAgentSpec, ...] = (
    HelperAgentSpec(
        name="runner",
        description=(
            "Execution-focused helper for one assigned optimization problem. "
            "Use proactively to run ./bin/run_candidate.sh and summarize results without polluting the main context."
        ),
        shell_commands=("./bin/run_candidate.sh", "./bin/goal_status.sh"),
        read_paths=(
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "GOAL_STATUS.md",
            "goal_status.json",
            "problem_reference.py",
            CANDIDATE_FILENAME,
            "samples/",
            "samples/best_result.json",
        ),
        summary_focus=(
            "Return a compact summary covering correctness failures, compiler failures, runtime measurements, the current best sample, and the most likely next implementation branch."
        ),
    ),
    HelperAgentSpec(
        name="profiler",
        description=(
            "Profiling helper for one assigned optimization problem. "
            "Use proactively to run ./bin/profile_ncu.sh and summarize bottlenecks and likely next steps."
        ),
        shell_commands=("./bin/profile_ncu.sh", "./bin/goal_status.sh"),
        read_paths=(
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "GOAL_STATUS.md",
            "problem_reference.py",
            CANDIDATE_FILENAME,
            "profiles/latest.summary.txt",
            "profiles/latest.details.txt",
        ),
        summary_focus=(
            "Return short, actionable summaries focused on bottlenecks, dominant kernels, occupancy, memory behavior, and the most promising next optimization directions."
        ),
    ),
)


WORKSPACE_WRAPPER_TRACE_KEYS: dict[str, str] = {
    spec.path: f"{spec.name}_calls" for spec in WORKSPACE_COMMAND_SPECS
}
GPU_WRAPPER_PATHS: tuple[str, ...] = tuple(
    spec.path for spec in WORKSPACE_COMMAND_SPECS if spec.uses_gpu
)
