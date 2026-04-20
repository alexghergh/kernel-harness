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

from kernel_bench_experiment_agents.surface.agent_specs import write_shared_helper_agent_specs
from kernel_bench_experiment_agents.surface.policy import ALLOWED_WEB_DOMAINS, MCP_SERVER_NAME, claude_mcp_tool_names
from kernel_bench_experiment_agents.runtime.project import ensure_dir, write_text


MCP_SERVER_ENV_VARS: tuple[str, ...] = (
    "DATA_ROOT",
    "KBH_WORKSPACE",
    "KBH_CLIENT_TOOL",
    "KBH_MCP_EVENTS_PATH",
)


def _python_command() -> str:
    """Return the exact Python executable path that launched the harness.

    Do not resolve symlinks here. Virtualenv/pyenv interpreters are often shim or symlink paths into
    a base interpreter; resolving them can bypass the environment that actually has the harness and
    MCP SDK installed.
    """
    return str(Path(sys.executable).expanduser())


def _mirror_optional_file(source: Path, target: Path) -> Path | None:
    if source.exists() and source.is_file():
        ensure_dir(target.parent)
        shutil.copy2(source, target)
        return target
    if target.exists() or target.is_symlink():
        target.unlink()
    return None


def sync_repo_auth_into_shared_tool_state(config_root: Path, *, repo_root: Path | None = None) -> list[Path]:
    """Copy repo-root auth caches into the generated shared tool homes.

    The user authenticates once under repo-root `.codex/` and `.claude/`; the harness recreates
    `state/config/` as needed and re-seeds just the auth files from those source dirs.
    """
    repo_root = (repo_root or Path.cwd()).expanduser().resolve()
    config_root = ensure_dir(config_root.expanduser().resolve())
    written: list[Path] = []

    copied = _mirror_optional_file(
        repo_root / ".codex" / "auth.json",
        config_root / "codex" / "auth.json",
    )
    if copied is not None:
        written.append(copied)

    copied = _mirror_optional_file(
        repo_root / ".claude" / ".credentials.json",
        config_root / "claude" / ".credentials.json",
    )
    if copied is not None:
        written.append(copied)

    return written



def render_codex_config() -> str:
    """Render the shared Codex config.toml.

    The Codex home is shared across problems. The MCP server registration stays static here, while
    the launcher exports the small per-problem context (`DATA_ROOT`, `KBH_WORKSPACE`,
    `KBH_CLIENT_TOOL`, `KBH_MCP_EVENTS_PATH`) and Codex forwards only those names into the stdio
    server via `env_vars`.
    """
    allowed_domains = ", ".join(f'"{domain}"' for domain in ALLOWED_WEB_DOMAINS)
    env_vars = ", ".join(f'"{name}"' for name in MCP_SERVER_ENV_VARS)
    python_command = json.dumps(_python_command())
    payload = (
        '# Generated from src/kernel_bench_experiment_agents/runtime/policy.py\n'
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
        f'env_vars = [{env_vars}]\n'
        'required = true\n'
        'startup_timeout_sec = 20\n\n'
        '[tools]\n'
        f'web_search = {{ context_size = "low", allowed_domains = [{allowed_domains}] }}\n'
    )
    return payload


def claude_settings_payload() -> dict[str, object]:
    """Build the Claude settings payload that lives under CLAUDE_CONFIG_DIR/settings.json."""
    allow_tools = [
        "WebSearch",
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
