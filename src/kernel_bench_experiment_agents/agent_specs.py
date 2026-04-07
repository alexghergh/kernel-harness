from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from .project import experiment_root, write_text


@dataclass(frozen=True)
class HelperAgentSpec:
    name: str
    description: str
    shell_commands: tuple[str, ...]
    read_paths: tuple[str, ...]
    summary_focus: str


HELPER_SPECS = (
    HelperAgentSpec(
        name="runner",
        description=(
            "Execution-focused helper for a single assigned KernelBench problem. "
            "Use proactively to run ./bin/run_candidate.sh and summarize results without polluting the main context."
        ),
        shell_commands=("./bin/run_candidate.sh", "./bin/goal_status.sh"),
        read_paths=(
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "GOAL_STATUS.md",
            "goal_status.json",
            "samples/",
        ),
        summary_focus=(
            "Return a compact summary covering correctness failures, compiler failures, runtime measurements, and the current best sample."
        ),
    ),
    HelperAgentSpec(
        name="profiler",
        description=(
            "Profiling helper for a single assigned KernelBench problem. Use proactively to run ./bin/profile_ncu.sh and summarize bottlenecks and likely next steps."
        ),
        shell_commands=("./bin/profile_ncu.sh",),
        read_paths=(
            "AGENTS.md",
            "SPEC.md",
            "HARDWARE.md",
            "profiles/latest.summary.txt",
            "profiles/latest.details.txt",
            "profiles/latest.raw.csv",
        ),
        summary_focus=(
            "Return short, actionable summaries focused on bottlenecks, dominant kernels, occupancy, memory behavior, and likely next optimization directions."
        ),
    ),
)


def _codex_agent_toml(spec: HelperAgentSpec) -> str:
    shell_list = " or ".join(f"`{command}`" for command in spec.shell_commands)
    read_list = ", ".join(f"`{path}`" for path in spec.read_paths)
    return dedent(
        f'''
        name = "{spec.name}"
        description = "{spec.description}"
        sandbox_mode = "workspace-write"
        developer_instructions = """
        You are a narrow helper for a single assigned KernelBench problem.

        Read `AGENTS.md` first, then `SPEC.md` and `HARDWARE.md`.
        Use shell only for {shell_list}.
        Use normal file reads only for {read_list}.
        Do not inspect unrelated problems or wander outside the current workspace.
        Do not use ad hoc shell commands to inspect directories or parse files.
        Do not edit any files.
        {spec.summary_focus}
        """
        '''
    ).strip() + "\n"


def _claude_agent_md(spec: HelperAgentSpec) -> str:
    shell_list = " or ".join(f"`{command}`" for command in spec.shell_commands)
    read_list = ", ".join(f"`{path}`" for path in spec.read_paths)
    return dedent(
        f"""
        ---
        name: {spec.name}
        description: {spec.description}
        tools:
          - Read
          - Bash
        ---

        You are a narrow helper for a single assigned KernelBench problem.

        Read `AGENTS.md` first, then `SPEC.md` and `HARDWARE.md`.
        Use `Bash` only for {shell_list}.
        Use `Read` only for {read_list}.
        Do not inspect unrelated problems or wander outside the current workspace.
        Do not use shell commands or Python snippets to inspect profiler outputs or parse files.
        Do not edit any files.
        {spec.summary_focus}
        """
    ).strip() + "\n"


def sync_helper_agent_specs(root: Path | None = None) -> list[Path]:
    base = root or experiment_root()
    written: list[Path] = []
    for spec in HELPER_SPECS:
        codex_path = base / ".codex" / "agents" / f"{spec.name}.toml"
        claude_path = base / ".claude" / "agents" / f"{spec.name}.md"
        write_text(codex_path, _codex_agent_toml(spec))
        write_text(claude_path, _claude_agent_md(spec))
        written.extend([codex_path, claude_path])
    return written
