"""Render shared Codex and Claude runtime config from the harness policy.

The launcher keeps one shared tool-private config home per tool under `state/config/`. Those dirs hold
runtime config, auth, helper agents, and any tool-managed local state, while the workspace remains free
of tool auth/config files.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_specs import write_shared_helper_agent_specs
from .policy_model import ALLOWED_WEB_DOMAINS
from .project import ensure_dir, write_text

# Claude and Codex expose different native config surfaces. This module renders the
# tool-specific files directly into the shared tool-private config dirs under state/config/.


def render_codex_config() -> str:
    """Render the Codex config that lives under the shared CODEX_HOME dir."""
    allowed_domains = ", ".join(f'"{domain}"' for domain in ALLOWED_WEB_DOMAINS)
    return (
        '# Generated from src/kernel_bench_experiment_agents/runtime_policy.py\n'
        'personality = "pragmatic"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "workspace-write"\n'
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
        '[sandbox_workspace_write]\n'
        'network_access = false\n\n'
        '[tools]\n'
        f'web_search = {{ context_size = "low", allowed_domains = [{allowed_domains}] }}\n'
    )



def claude_settings_payload() -> dict[str, object]:
    """Build the Claude settings payload that lives under the shared CLAUDE_CONFIG_DIR."""
    return {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "disableBypassPermissionsMode": "disable",
        "permissions": {
            "allow": [
                *(f"WebFetch(domain:{domain})" for domain in ALLOWED_WEB_DOMAINS),
            ],
            "deny": [
                "Read(./.env)",
                "Read(./.env.*)",
                "Read(./secrets/**)",
                "Bash(curl:*)",
                "Bash(wget:*)",
                "Bash(ps:*)",
                "Bash(pgrep:*)",
                "Bash(top:*)",
                "Bash(htop:*)",
                "Bash(nvidia-smi:*)",
                "Bash(strace:*)",
            ],
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "allowRead": ["."],
                "allowWrite": ["."],
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



def render_claude_settings() -> str:
    return json.dumps(claude_settings_payload(), indent=2, sort_keys=True) + "\n"



def write_shared_tool_state(config_root: Path) -> list[Path]:
    config_root = ensure_dir(config_root.expanduser().resolve())
    codex_dir = ensure_dir(config_root / "codex")
    claude_dir = ensure_dir(config_root / "claude")
    codex_path = codex_dir / "config.toml"
    claude_path = claude_dir / "settings.json"
    write_text(codex_path, render_codex_config())
    write_text(claude_path, render_claude_settings())
    written = [codex_path, claude_path]
    written.extend(
        write_shared_helper_agent_specs(codex_home=codex_dir, claude_config_dir=claude_dir)
    )
    return written
