"""Canonical typed policy for solver-visible harness behavior and runtime config.

This module is the single source of truth for the policy objects that must stay aligned across
runtime config rendering, workspace docs, trace audit, and helper-agent rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernel_bench_experiment_agents.kernelbench.candidate.contract import CANDIDATE_FILENAME

COMMAND_MCP_SERVER_NAME = "kernelbench_commands"


@dataclass(frozen=True)
class WrapperCommandSpec:
    """Describes one backend wrapper command that the harness still records in traces."""

    name: str
    path: str
    purpose: str
    uses_gpu: bool = False


@dataclass(frozen=True)
class SolverCommandSpec:
    """Describes one solver-visible workspace command."""

    name: str
    path: str
    purpose: str
    uses_gpu: bool = False


@dataclass(frozen=True)
class HelperAgentSpec:
    """Captures the shared intent for rendered Codex and Claude helper agents."""

    name: str
    description: str
    commands: tuple[str, ...]
    read_paths: tuple[str, ...]
    summary_focus: str


ALLOWED_WEB_DOMAINS: tuple[str, ...] = ("docs.nvidia.com",)
SOLVER_TERMINAL_STATES: tuple[str, ...] = ("done",)
LAUNCHER_TERMINAL_STATES: tuple[str, ...] = (
    "budget_exhausted",
    "failed_to_generate",
)

# These wrapper paths are the backend command surface used by the harness itself. The broker records
# transport-neutral activity events against these paths so counting/audit can stay stable.
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
        name="research_nvidia_docs",
        path="./bin/research_nvidia_docs.sh",
        purpose="search or fetch official NVIDIA docs through the broker",
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

SOLVER_COMMAND_SPECS: tuple[SolverCommandSpec, ...] = tuple(
    SolverCommandSpec(
        name=spec.name,
        path=spec.path,
        purpose=spec.purpose,
        uses_gpu=spec.uses_gpu,
    )
    for spec in WORKSPACE_COMMAND_SPECS
    if spec.name != "hardware_info"
)

WORKSPACE_STANDING_ORDERS: tuple[str, ...] = (
    "Work independently. There is no human approval, acceptance, or confirmation step during the run.",
    "Do not ask whether to proceed. Pick the next reasonable action yourself.",
    "Do not end with a plain assistant message. The only valid exit is the direct `complete_problem` command tool.",
    "Never start a second harness command tool while another one is still running.",
    "Use direct command tools like `run_candidate`, `profile_ncu`, `research_nvidia_docs`, `goal_status`, `best_result`, and `complete_problem` for all harness actions.",
    "Do not use shell commands or Python snippets for harness actions.",
    "After every measured run or profile, re-read GOAL_STATUS.md or run the direct `goal_status` tool; keep iterating if it still says UNRESOLVED.",
    "After every measured run, inspect `samples/latest.json` first and then `samples/latest.stdout.txt` / `samples/latest.stderr.txt` when the latest attempt failed to compile, validate, or run correctly.",
    "If one branch fails, start another one. Failed attempts are normal, not a stop signal.",
    "Once any candidate compiles and runs but is slower than either baseline, run `profile_ncu` before more than one additional optimization edit.",
    "If a correct candidate is slower than either baseline and no profile exists for that candidate family, profile it before deciding the bottleneck from intuition.",
    "Do not conclude that a slow correct candidate is fundamentally limited until you have profiled it and read `profiles/latest.summary.txt`.",
    "When CUDA, PTX, WMMA/MMA, tensor-core, memory-hierarchy, occupancy, Nsight Compute, or compiler details are uncertain, call `research_nvidia_docs` instead of guessing.",
    "`run_candidate` and `profile_ncu` may take a while. Wait for them to finish instead of assuming they hung.",
)

WORKSPACE_STUCK_PROTOCOL: tuple[str, ...] = (
    "Re-read SPEC.md, HARDWARE.md, and GOAL_STATUS.md.",
    "If the current candidate compiles and runs, run the direct `profile_ncu` tool before the next implementation branch unless you already profiled this candidate family.",
    "If two consecutive measured attempts are correct but slower than either baseline, run `profile_ncu` before editing the candidate again.",
    "Read profiles/latest.summary.txt first, then profiles/latest.details.txt if needed.",
    "Before the next implementation branch, call `research_nvidia_docs` for CUDA/NVIDIA-specific uncertainty; query the relevant CUDA, PTX, WMMA/MMA, tensor-core, occupancy, memory hierarchy, or Nsight Compute topic.",
    "If two consecutive attempts fail for a CUDA API, compile, profiling-metric, or hardware-tuning reason, call `research_nvidia_docs` before editing the candidate again.",
    "Make a new implementation plan and continue without asking the user for permission.",
)

FIXED_WORKSPACE_RESOURCE_PATHS: tuple[str, ...] = (
    ".",
    "AGENTS.md",
    "INITIAL_PROMPT.md",
    "SPEC.md",
    "HARDWARE.md",
    "GOAL_STATUS.md",
    "goal_status.json",
    "hardware.json",
    "workspace_contract.json",
    "problem.json",
    "problem_reference.py",
    CANDIDATE_FILENAME,
    "bin/",
)

WORKSPACE_BROWSE_DIRS: tuple[str, ...] = (
    "samples/",
    "profiles/",
)

WORKSPACE_READ_PATHS: tuple[str, ...] = (
    *FIXED_WORKSPACE_RESOURCE_PATHS,
    *WORKSPACE_BROWSE_DIRS,
)

WORKSPACE_EDIT_PATHS: tuple[str, ...] = (CANDIDATE_FILENAME,)

HELPER_SPECS: tuple[HelperAgentSpec, ...] = (
    HelperAgentSpec(
        name="runner",
        description=(
            "Execution-focused helper for one assigned optimization problem. "
            "Use proactively to run measured evaluations and summarize results without polluting the main context."
        ),
        commands=("./bin/run_candidate.sh", "./bin/goal_status.sh", "./bin/best_result.sh"),
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
            "Use proactively to run Nsight Compute profiling and summarize bottlenecks and likely next steps."
        ),
        commands=("./bin/profile_ncu.sh", "./bin/research_nvidia_docs.sh", "./bin/goal_status.sh"),
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


def command_mcp_tool_names() -> tuple[str, ...]:
    return (
        f"mcp__{COMMAND_MCP_SERVER_NAME}__run_candidate",
        f"mcp__{COMMAND_MCP_SERVER_NAME}__profile_ncu",
        f"mcp__{COMMAND_MCP_SERVER_NAME}__goal_status",
        f"mcp__{COMMAND_MCP_SERVER_NAME}__research_nvidia_docs",
        f"mcp__{COMMAND_MCP_SERVER_NAME}__best_result",
        f"mcp__{COMMAND_MCP_SERVER_NAME}__complete_problem",
    )


def solver_command_names() -> tuple[str, ...]:
    return tuple(spec.path for spec in SOLVER_COMMAND_SPECS)


def _resolve_workspace_surface(
    workspace: Path,
    relative_paths: tuple[str, ...],
) -> tuple[set[Path], tuple[Path, ...]]:
    workspace = workspace.resolve()
    exact_paths: set[Path] = set()
    rooted_paths: list[Path] = []
    for relative_path in relative_paths:
        resolved = (workspace / relative_path).resolve()
        if relative_path.endswith("/"):
            rooted_paths.append(resolved)
        else:
            exact_paths.add(resolved)
    return exact_paths, tuple(rooted_paths)


def workspace_read_surface(workspace: Path) -> tuple[set[Path], tuple[Path, ...]]:
    return _resolve_workspace_surface(workspace, WORKSPACE_READ_PATHS)


def workspace_edit_surface(workspace: Path) -> set[Path]:
    exact_paths, rooted_paths = _resolve_workspace_surface(workspace, WORKSPACE_EDIT_PATHS)
    if rooted_paths:
        raise RuntimeError("workspace edit surface must contain only exact file paths")
    return exact_paths
