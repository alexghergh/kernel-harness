"""Render helper-agent specs from the shared harness policy.

The live tool homes under `state/config/` load these helper-agent definitions, while the archive keeps
frozen copies under `contract/helper_agents/` for inspection.
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
        Never start a second wrapper call while another wrapper is still running.
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
        Never start a second wrapper call while another wrapper is still running.
        Do not edit any files.
        Work independently: do not ask the user or the main agent for permission to proceed once assigned.
        When finished, return only a concise actionable summary; do not ask follow-up questions.
        {spec.summary_focus}
        """
    ).strip() + "\n"



def _write_codex_specs(base: Path) -> list[Path]:
    written: list[Path] = []
    for spec in HELPER_SPECS:
        path = base / f"{spec.name}.toml"
        write_text(path, _codex_agent_toml(spec))
        written.append(path)
    return written



def _write_claude_specs(base: Path) -> list[Path]:
    written: list[Path] = []
    for spec in HELPER_SPECS:
        path = base / f"{spec.name}.md"
        write_text(path, _claude_agent_md(spec))
        written.append(path)
    return written



def write_shared_helper_agent_specs(*, codex_home: Path, claude_config_dir: Path) -> list[Path]:
    written: list[Path] = []
    written.extend(_write_codex_specs(codex_home / "agents"))
    written.extend(_write_claude_specs(claude_config_dir / "agents"))
    return written



def write_archive_helper_agent_specs(*, archive_contract_dir: Path) -> list[Path]:
    written: list[Path] = []
    written.extend(_write_codex_specs(archive_contract_dir / "helper_agents" / "codex"))
    written.extend(_write_claude_specs(archive_contract_dir / "helper_agents" / "claude"))
    return written



def describe_helper_spec_paths() -> dict[str, list[str]]:
    return {
        "shared_codex": [f"state/config/codex/agents/{spec.name}.toml" for spec in HELPER_SPECS],
        "shared_claude": [f"state/config/claude/agents/{spec.name}.md" for spec in HELPER_SPECS],
        "archive_codex": [f"contract/helper_agents/codex/{spec.name}.toml" for spec in HELPER_SPECS],
        "archive_claude": [f"contract/helper_agents/claude/{spec.name}.md" for spec in HELPER_SPECS],
    }
