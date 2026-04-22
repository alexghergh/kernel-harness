"""Render helper-agent specs from the shared harness policy.

The live tool homes under `state/config/` load these helper-agent definitions, while the archive keeps
frozen copies under `contract/helper_agents/` for inspection.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from kernel_bench_experiment_agents.agent_contract.policy import MCP_SERVER_NAME, HELPER_SPECS, HelperAgentSpec
from kernel_bench_experiment_agents.runtime.project import write_text


def _codex_agent_toml(spec: HelperAgentSpec) -> str:
    tool_list = ", ".join(f"`{name}`" for name in spec.mcp_tools)
    read_list = ", ".join(f"`{path}`" for path in spec.read_paths)
    return dedent(
        f'''
        name = "{spec.name}"
        description = "{spec.description}"
        sandbox_mode = "read-only"
        developer_instructions = """
        You are a narrow delegated helper for one assigned optimization problem.

        The main solver should treat you as an execution-focused delegate, not as another planner.
        Use only the `{MCP_SERVER_NAME}` MCP tools: {tool_list}.
        Read local problem files only through `read_workspace_file`, and only for {read_list}.
        Do not inspect unrelated files, local config, or hidden harness state.
        Do not use ad hoc shell commands, Python snippets, or local file tools.
        Hosted WebSearch/WebFetch, if available at all, are restricted to docs.nvidia.com only.
        If one of the allowed MCP tools is slow, wait for it to finish instead of trying to inspect processes or the GPU.
        Never start a second harness MCP call while another one is still running.
        Do not edit any files unless the main agent explicitly delegated candidate writing to you.
        Work independently: do not ask the user or the main agent for permission to proceed once assigned.
        When finished, return only a concise actionable summary; do not ask follow-up questions.
        {spec.summary_focus}
        """
        '''
    ).strip() + "\n"



def _claude_agent_md(spec: HelperAgentSpec) -> str:
    tool_list = ", ".join(f"`{name}`" for name in spec.mcp_tools)
    read_list = ", ".join(f"`{path}`" for path in spec.read_paths)
    yaml_tools = "\n".join(
        f"  - mcp__{MCP_SERVER_NAME}__{tool_name}" for tool_name in spec.mcp_tools
    )
    return (
        "---\n"
        f"name: {spec.name}\n"
        f"description: {spec.description}\n"
        "tools:\n"
        f"{yaml_tools}\n"
        "---\n\n"
        "You are a narrow delegated helper for one assigned optimization problem.\n\n"
        "The main solver should treat you as an execution-focused delegate, not as another planner.\n"
        f"Use only the `{MCP_SERVER_NAME}` MCP tools: {tool_list}.\n"
        f"Read local problem files only through `read_workspace_file`, and only for {read_list}.\n"
        "Do not inspect unrelated files, local config, or hidden harness state.\n"
        "Do not use shell commands or Python snippets to inspect profiler outputs or parse files.\n"
        "Hosted WebSearch/WebFetch, if available at all, are restricted to docs.nvidia.com only.\n"
        "If one of the allowed MCP tools is slow, wait for it to finish instead of trying to inspect processes or the GPU.\n"
        "Never start a second harness MCP call while another one is still running.\n"
        "Do not edit any files unless the main agent explicitly delegated candidate writing to you.\n"
        "Work independently: do not ask the user or the main agent for permission to proceed once assigned.\n"
        "When finished, return only a concise actionable summary; do not ask follow-up questions.\n"
        f"{spec.summary_focus}\n"
    )



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
