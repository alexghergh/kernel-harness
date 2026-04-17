"""Canonical typed policy for solver-visible harness behavior and runtime config.

This module is the single source of truth for the policy objects that must stay aligned across
runtime config rendering, workspace docs, trace audit, and helper-agent rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .candidate_contract import CANDIDATE_FILENAME

MCP_SERVER_NAME = "kernelbench"


@dataclass(frozen=True)
class WrapperCommandSpec:
    """Describes one backend wrapper command that the harness still records in traces."""

    name: str
    path: str
    purpose: str
    uses_gpu: bool = False


@dataclass(frozen=True)
class McpToolSpec:
    """Describes one solver-visible MCP tool."""

    name: str
    purpose: str
    uses_gpu: bool = False
    read_only: bool = False
    destructive: bool = False


@dataclass(frozen=True)
class HelperAgentSpec:
    """Captures the shared intent for rendered Codex and Claude helper agents."""

    name: str
    description: str
    mcp_tools: tuple[str, ...]
    read_paths: tuple[str, ...]
    summary_focus: str


ALLOWED_WEB_DOMAINS: tuple[str, ...] = ("docs.nvidia.com",)
SOLVER_TERMINAL_STATES: tuple[str, ...] = ("done",)
LAUNCHER_TERMINAL_STATES: tuple[str, ...] = (
    "budget_exhausted",
    "failed_to_generate",
)

# These wrapper paths remain the backend command surface used by the harness itself. Synthetic MCP
# trace events refer to them so the existing counting/audit logic can stay stable.
WORKSPACE_COMMAND_SPECS: tuple[WrapperCommandSpec, ...] = (
    WrapperCommandSpec(
        name="hardware_info",
        path="./bin/hardware_info.sh",
        purpose="print frozen hardware facts for this workspace",
    ),
    WrapperCommandSpec(
        name="run_candidate",
        path="./bin/run_candidate.sh",
        purpose="run_candidate() -> JSON result for the current candidate (status, sample_id, correctness/runtime info, and archive-relative outputs); takes no arguments",
        uses_gpu=True,
    ),
    WrapperCommandSpec(
        name="profile_ncu",
        path="./bin/profile_ncu.sh",
        purpose="profile_ncu() -> JSON result plus new profiler artifacts for the current candidate; takes no arguments",
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
        purpose="complete_problem(summary) -> record the final solver summary and end the run; the only valid solver exit path",
    ),
)

MCP_TOOL_SPECS: tuple[McpToolSpec, ...] = (
    McpToolSpec(
        name="workspace_overview",
        purpose="workspace_overview() -> JSON overview of the assigned problem, the fixed read-only resources, the allowed history directories, and the available MCP action tools",
        read_only=True,
    ),
    McpToolSpec(
        name="list_workspace_dir",
        purpose="list_workspace_dir(path='samples'|'profiles') -> JSON directory listing for one allowed history directory only",
        read_only=True,
    ),
    McpToolSpec(
        name="read_workspace_file",
        purpose="read_workspace_file(path) -> file text for one fixed resource or one listed text artifact under `samples/` or `profiles/`",
        read_only=True,
    ),
    McpToolSpec(
        name="write_candidate",
        purpose=f"write_candidate(content) -> overwrite {CANDIDATE_FILENAME}; the only writable workspace file",
        destructive=True,
    ),
    McpToolSpec(
        name="run_candidate",
        purpose="run_candidate() -> JSON result for the current candidate (status, sample_id, correctness/runtime info, and archive-relative outputs); takes no arguments",
        uses_gpu=True,
    ),
    McpToolSpec(
        name="profile_ncu",
        purpose="profile_ncu() -> JSON result plus new profiler artifacts for the current candidate; takes no arguments",
        uses_gpu=True,
    ),
    McpToolSpec(
        name="goal_status",
        purpose="goal_status() -> live JSON status snapshot (remaining budget, attempt counts, baseline progress, best sample); takes no arguments",
        read_only=True,
    ),
    McpToolSpec(
        name="best_result",
        purpose="best_result() -> JSON for the best measured correct attempt so far, including sample_id and archive-relative artifact paths; takes no arguments",
        read_only=True,
    ),
    McpToolSpec(
        name="complete_problem",
        purpose="complete_problem(summary) -> record the final solver summary and end the run; the only valid solver exit path",
        destructive=True,
    ),
)

WORKSPACE_STANDING_ORDERS: tuple[str, ...] = (
    "Act as the planner-manager for this problem. Keep the main context focused on strategy, debugging, and choosing the next branch.",
    "Work independently. There is no human approval, acceptance, or confirmation step during the run.",
    "Do not ask whether to proceed. Pick the next reasonable action yourself.",
    "Do not end with a plain assistant message. The only valid exit is the `complete_problem` MCP tool.",
    "Never start a second harness MCP call while another one is still running.",
    "WHEN you want a measured evaluation, spawn the `runner` helper if available; use direct `run_candidate` yourself only when helper spawning is unavailable.",
    "WHEN you want Nsight Compute output or profile interpretation, spawn the `profiler` helper if available; use direct `profile_ncu` yourself only when helper spawning is unavailable.",
    "After every measured run or profile, re-read GOAL_STATUS.md or call `goal_status`; keep iterating if it still says UNRESOLVED.",
    "If one branch fails, start another one. Failed attempts are normal, not a stop signal.",
    "`run_candidate` and `profile_ncu` may take a while. Wait for them to finish instead of assuming they hung.",
)

WORKSPACE_STUCK_PROTOCOL: tuple[str, ...] = (
    "Re-read SPEC.md, HARDWARE.md, and GOAL_STATUS.md.",
    "WHEN you do not already have profiling for the current idea, call `profile_ncu`.",
    "Read `profiles/latest.summary.txt` first, then `profiles/latest.details.txt` if needed.",
    "WHEN the next idea depends on hardware-specific behavior, use hosted web search on docs.nvidia.com for topics like tensor cores, WMMA, async copy/pipelining, occupancy, bank conflicts, and memory hierarchy limits.",
    "WHEN choosing the next branch, inspect `samples/` and `profiles/` so you do not retry the same failed idea.",
    "Make a new implementation plan and continue without asking the user for permission.",
)

FIXED_WORKSPACE_RESOURCE_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "INITIAL_PROMPT.md",
    "SPEC.md",
    "HARDWARE.md",
    "GOAL_STATUS.md",
    "problem_reference.py",
    CANDIDATE_FILENAME,
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
            "The main solver should delegate measured evaluations to this helper by default so the main context stays focused on planning."
        ),
        mcp_tools=("read_workspace_file", "run_candidate", "goal_status", "best_result"),
        read_paths=(
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "GOAL_STATUS.md",
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
            "The main solver should delegate Nsight Compute work to this helper by default so the main context stays focused on planning."
        ),
        mcp_tools=("read_workspace_file", "profile_ncu", "goal_status"),
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


def mcp_tool_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in MCP_TOOL_SPECS)


def claude_mcp_tool_names() -> tuple[str, ...]:
    return tuple(f"mcp__{MCP_SERVER_NAME}__{name}" for name in mcp_tool_names())


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
