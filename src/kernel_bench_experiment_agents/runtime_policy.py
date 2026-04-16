"""Render shared Codex and Claude runtime config from the harness policy.

The launcher keeps one shared tool-private config home per tool under `state/config/`. Those dirs hold
runtime config, copied auth, helper agents, and any tool-managed local state, while the workspace
remains free of tool auth/config files.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Mapping

from .agent_specs import write_shared_helper_agent_specs
from .policy_model import ALLOWED_WEB_DOMAINS, MCP_SERVER_NAME, claude_mcp_tool_names
from .project import ensure_dir, write_text


MCP_SERVER_ENV_VARS: tuple[str, ...] = (
    "DATA_ROOT",
    "KBH_WORKSPACE",
    "KBH_CLIENT_TOOL",
    "KBH_MCP_EVENTS_PATH",
)


def _python_command() -> str:
    """Return the exact Python executable that launched the harness.

    The MCP server must start under the same environment that has the harness installed. Hard-coding
    `python` is fragile when Codex or Claude launch from outside the activated environment.
    """
    return str(Path(sys.executable).expanduser().resolve())


def _copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists() or not source.is_file():
        return None
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return target


def sync_repo_auth_into_shared_tool_state(config_root: Path, *, repo_root: Path | None = None) -> list[Path]:
    """Copy repo-root auth caches into the generated shared tool homes.

    The user authenticates once under repo-root `.codex/` and `.claude/`; the harness recreates
    `state/config/` as needed and re-seeds just the auth files from those source dirs.
    """
    repo_root = (repo_root or Path.cwd()).expanduser().resolve()
    config_root = ensure_dir(config_root.expanduser().resolve())
    written: list[Path] = []

    copied = _copy_if_exists(repo_root / ".codex" / "auth.json", config_root / "codex" / "auth.json")
    if copied is not None:
        written.append(copied)

    copied = _copy_if_exists(
        repo_root / ".claude" / ".credentials.json",
        config_root / "claude" / ".credentials.json",
    )
    if copied is not None:
        written.append(copied)

    return written



def render_codex_config(*, mcp_env: Mapping[str, str] | None = None) -> str:
    """Render a Codex config.toml.

    Shared state stores the stable defaults, auth cache, and helper agents. The actual launch uses a
    per-problem CODEX_HOME so the MCP server can receive an explicit static `[mcp_servers.*.env]`
    table before Codex starts required MCP servers.
    """
    allowed_domains = ", ".join(f'"{domain}"' for domain in ALLOWED_WEB_DOMAINS)
    python_command = json.dumps(_python_command())
    payload = (
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
        'unified_exec = false\n'
        'shell_tool = false\n\n'
        '[agents]\n'
        'max_threads = 6\n'
        'max_depth = 1\n\n'
        f'[mcp_servers.{MCP_SERVER_NAME}]\n'
        f'command = {python_command}\n'
        'args = ["-m", "kernel_bench_experiment_agents.mcp"]\n'
        'required = true\n'
        'startup_timeout_sec = 20\n\n'
    )
    if mcp_env:
        payload += f'[mcp_servers.{MCP_SERVER_NAME}.env]\n'
        for name in MCP_SERVER_ENV_VARS:
            value = mcp_env.get(name)
            if value is None:
                continue
            payload += f'{name} = {json.dumps(str(value))}\n'
        payload += '\n'
    payload += (
        '[tools]\n'
        f'web_search = {{ context_size = "low", allowed_domains = [{allowed_domains}] }}\n'
    )
    return payload



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
        "sandbox": {"enabled": False},
    }



def claude_user_config_payload() -> dict[str, object]:
    """Build the shared Claude user config that registers the harness MCP server."""
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "stdio",
                "command": _python_command(),
                "args": ["-m", "kernel_bench_experiment_agents.mcp"],
                "env": {name: f"${{{name}:-}}" for name in MCP_SERVER_ENV_VARS},
            }
        }
    }



def render_claude_settings() -> str:
    return json.dumps(claude_settings_payload(), indent=2, sort_keys=True) + "\n"



def render_claude_user_config() -> str:
    return json.dumps(claude_user_config_payload(), indent=2, sort_keys=True) + "\n"


def prepare_codex_session_home(
    *,
    shared_codex_home: Path,
    target_home: Path,
    mcp_env: Mapping[str, str],
) -> list[Path]:
    """Create a per-launch CODEX_HOME with explicit MCP env and shared auth/agents.

    Codex reliably starts the required MCP server when the env table lives in config.toml before
    startup. We therefore keep auth and helper agents under the shared Codex home, but materialize
    a tiny per-problem CODEX_HOME that points at the same auth/agents and owns only local mutable
    Codex state (history, logs, caches).
    """
    shared_codex_home = shared_codex_home.expanduser().resolve()
    target_home = ensure_dir(target_home.expanduser().resolve())

    written: list[Path] = []
    config_path = target_home / 'config.toml'
    write_text(config_path, render_codex_config(mcp_env=mcp_env))
    written.append(config_path)

    for name in ('auth.json',):
        source = shared_codex_home / name
        target = target_home / name
        if target.exists() or target.is_symlink():
            target.unlink()
        if source.exists():
            target.symlink_to(source)
            written.append(target)

    source_agents = shared_codex_home / 'agents'
    target_agents = target_home / 'agents'
    if target_agents.exists() or target_agents.is_symlink():
        if target_agents.is_symlink() or target_agents.is_file():
            target_agents.unlink()
        else:
            shutil.rmtree(target_agents)
    if source_agents.exists():
        target_agents.symlink_to(source_agents, target_is_directory=True)
        written.append(target_agents)

    return written



def write_shared_tool_state(config_root: Path, *, repo_root: Path | None = None) -> list[Path]:
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
    written.extend(sync_repo_auth_into_shared_tool_state(config_root, repo_root=repo_root))
    written.extend(
        write_shared_helper_agent_specs(codex_home=codex_dir, claude_config_dir=claude_dir)
    )
    return written
