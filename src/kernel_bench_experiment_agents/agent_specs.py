"""Render helper-agent specs for Codex and Claude from the shared harness policy.

These files sit beside the generated workspace docs so helper agents inherit the same narrow surface as the main solver.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from .policy_model import HELPER_SPECS, HelperAgentSpec
from .project import write_text


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
        If one of the allowed wrappers is slow, wait for it to finish instead of trying to inspect processes or the GPU.
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
        If one of the allowed wrappers is slow, wait for it to finish instead of trying to inspect processes or the GPU.
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
