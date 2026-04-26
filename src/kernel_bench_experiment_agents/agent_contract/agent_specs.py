"""Render helper-agent specs from the shared harness policy.

The live tool homes under `state/config/` load these helper-agent definitions, while the archive keeps
frozen copies under `contract/helper_agents/` for inspection.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from kernel_bench_experiment_agents.agent_contract.policy import MCP_SERVER_NAME, HELPER_SPECS, HelperAgentSpec
from kernel_bench_experiment_agents.agent_contract.prompts import (
    render_claude_helper_body,
    render_codex_helper_instructions,
)
from kernel_bench_experiment_agents.runtime.project import write_text


def _codex_agent_toml(spec: HelperAgentSpec) -> str:
    return dedent(
        f'''
        name = "{spec.name}"
        description = "{spec.description}"
        sandbox_mode = "workspace-write"
        developer_instructions = """
        {render_codex_helper_instructions(spec=spec).rstrip()}
        """
        '''
    ).strip() + "\n"


def _claude_agent_md(spec: HelperAgentSpec) -> str:
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
        f"{render_claude_helper_body(spec=spec)}"
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
