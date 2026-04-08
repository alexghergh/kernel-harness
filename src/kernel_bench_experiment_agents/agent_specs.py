from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from .project import write_text


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
            "candidate_model_new.py",
            "samples/",
            "samples/best_result.json",
        ),
        summary_focus=(
            "Return a compact summary covering correctness failures, compiler failures, runtime measurements, the current best sample, and the most likely next experiment."
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
            "candidate_model_new.py",
            "profiles/latest.summary.txt",
            "profiles/latest.details.txt",
        ),
        summary_focus=(
            "Return short, actionable summaries focused on bottlenecks, dominant kernels, occupancy, memory behavior, and the most promising next optimization directions."
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
        You are a narrow helper for one assigned optimization problem.

        Read `AGENTS.md` first, then `SPEC.md`, `HARDWARE.md`, and any directly relevant local files.
        Use shell only for {shell_list}.
        Use normal file reads only for {read_list}.
        Do not inspect unrelated files or wander outside the current workspace.
        Do not use ad hoc shell commands to inspect directories or parse files.
        Do not edit any files.
        Work independently: do not ask the user or the main agent for permission to proceed once assigned.
        When finished, return only a concise actionable summary; do not ask follow-up questions.
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

        You are a narrow helper for one assigned optimization problem.

        Read `AGENTS.md` first, then `SPEC.md`, `HARDWARE.md`, and any directly relevant local files.
        Use `Bash` only for {shell_list}.
        Use `Read` only for {read_list}.
        Do not inspect unrelated files or wander outside the current workspace.
        Do not use shell commands or Python snippets to inspect profiler outputs or parse files.
        Do not edit any files.
        Work independently: do not ask the user or the main agent for permission to proceed once assigned.
        When finished, return only a concise actionable summary; do not ask follow-up questions.
        {spec.summary_focus}
        """
    ).strip() + "\n"



def _write_rendered_specs(base: Path, codex_rel: Path, claude_rel: Path) -> list[Path]:
    written: list[Path] = []
    for spec in HELPER_SPECS:
        codex_path = base / codex_rel / f"{spec.name}.toml"
        claude_path = base / claude_rel / f"{spec.name}.md"
        write_text(codex_path, _codex_agent_toml(spec))
        write_text(claude_path, _claude_agent_md(spec))
        written.extend([codex_path, claude_path])
    return written



def write_workspace_helper_agent_specs(
    *,
    workspace: Path,
    archive_contract_dir: Path | None = None,
) -> list[Path]:
    written = _write_rendered_specs(
        workspace,
        Path(".codex") / "agents",
        Path(".claude") / "agents",
    )
    if archive_contract_dir is not None:
        written.extend(
            _write_rendered_specs(
                archive_contract_dir / "helper_agents",
                Path("codex"),
                Path("claude"),
            )
        )
    return written



def describe_helper_spec_paths() -> dict[str, list[str]]:
    return {
        "workspace_codex": [f".codex/agents/{spec.name}.toml" for spec in HELPER_SPECS],
        "workspace_claude": [f".claude/agents/{spec.name}.md" for spec in HELPER_SPECS],
    }
