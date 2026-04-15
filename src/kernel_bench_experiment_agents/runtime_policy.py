"""Render shared Codex and Claude runtime config from the harness policy.

The launcher keeps one shared tool-private config home per tool under `state/config/`. Those dirs hold
runtime config, auth, helper agents, and any tool-managed local state, while the workspace remains free
of tool auth/config files.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_specs import write_shared_helper_agent_specs
from .policy_model import (
    ALLOWED_WEB_DOMAINS,
    MCP_SERVER_NAME,
    claude_mcp_tool_names,
)
from .project import ensure_dir, write_text


def render_codex_config() -> str:
    """Render the shared Codex config that lives under CODEX_HOME."""
    allowed_domains = ", ".join(f'"{domain}"' for domain in ALLOWED_WEB_DOMAINS)
    env_vars = ", ".join(
        f'"{name}"'
        for name in (
            "KBH_WORKSPACE",
            "KBH_RUN_NAME",
            "KBH_LEVEL",
            "KBH_PROBLEM_ID",
            "KBH_DATASET_SRC",
            "KBH_KERNELBENCH_ROOT",
            "KBH_NUM_GPU_SLOTS",
            "KBH_PRECISION",
            "KBH_CLIENT_TOOL",
            "KBH_MCP_EVENTS_PATH",
        )
    )
    return (
        '# Generated from src/kernel_bench_experiment_agents/runtime_policy.py\n'
        'personality = "pragmatic"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        'project_root_markers = []\n'
        'allow_login_shell = false\n'
        'web_search = "live"\n'
        'project_doc_max_bytes = 65536\n'
        'model_auto_compact_token_limit = 240000\n\n'
        '[features]\n'
        'unified_exec = false\n\n'
        '[agents]\n'
        'max_threads = 6\n'
        'max_depth = 1\n\n'
        f'[mcp_servers.{MCP_SERVER_NAME}]\n'
        'command = "python"\n'
        'args = ["-m", "kernel_bench_experiment_agents.mcp_server"]\n'
        f'env_vars = [{env_vars}]\n\n'
        '[tools]\n'
        f'web_search = {{ context_size = "low", allowed_domains = [{allowed_domains}] }}\n'
    )



def claude_settings_payload() -> dict[str, object]:
    """Build the Claude settings payload that lives under CLAUDE_CONFIG_DIR/settings.json."""
    allow_tools = [
        *(f"WebFetch(domain:{domain})" for domain in ALLOWED_WEB_DOMAINS),
        *claude_mcp_tool_names(),
    ]
    return {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "disableBypassPermissionsMode": "disable",
        "permissions": {
            "allow": allow_tools,
            "deny": [
                "Read",
                "Write",
                "Edit",
                "MultiEdit",
                "Bash",
                "Glob",
                "Grep",
                "LS",
            ],
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": False,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "allowRead": ["./"],
                "allowWrite": ["./"],
                "denyRead": [
                    "~/",
                    "/etc",
                    "/proc",
                    "/sys",
                    "/var",
                    "/usr",
                    "/bin",
                    "/sbin",
                    "/opt",
                    "/root",
                    "/tmp",
                ],
                "denyWrite": [
                    "~/",
                    "/etc",
                    "/proc",
                    "/sys",
                    "/var",
                    "/usr",
                    "/bin",
                    "/sbin",
                    "/opt",
                    "/root",
                    "/tmp",
                ],
            },
            "network": {
                "allowedDomains": list(ALLOWED_WEB_DOMAINS),
            },
        },
    }



def claude_user_config_payload() -> dict[str, object]:
    """Build the shared Claude user config that registers the harness MCP server."""
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "kernel_bench_experiment_agents.mcp_server"],
            }
        }
    }



def render_claude_settings() -> str:
    return json.dumps(claude_settings_payload(), indent=2, sort_keys=True) + "\n"



def render_claude_user_config() -> str:
    return json.dumps(claude_user_config_payload(), indent=2, sort_keys=True) + "\n"



def write_shared_tool_state(config_root: Path) -> list[Path]:
    config_root = ensure_dir(config_root.expanduser().resolve())
    codex_dir = ensure_dir(config_root / "codex")
    claude_dir = ensure_dir(config_root / "claude")
    codex_path = codex_dir / "config.toml"
    claude_settings_path = claude_dir / "settings.json"
    claude_user_config_path = claude_dir / ".claude.json"
    write_text(codex_path, render_codex_config())
    write_text(claude_settings_path, render_claude_settings())
    write_text(claude_user_config_path, render_claude_user_config())
    written = [codex_path, claude_settings_path, claude_user_config_path]
    written.extend(
        write_shared_helper_agent_specs(codex_home=codex_dir, claude_config_dir=claude_dir)
    )
    return written
