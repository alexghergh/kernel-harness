from __future__ import annotations

import json
from pathlib import Path

from .project import ensure_dir, write_text

ALLOWED_DOC_DOMAINS = ("docs.nvidia.com",)

# Claude and Codex expose different settings surfaces. This module defines the
# intended runtime policy once and renders the closest native configuration for each.


def render_codex_config() -> str:
    allowed_domains = ", ".join(f'"{domain}"' for domain in ALLOWED_DOC_DOMAINS)
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
        '[agents]\n'
        'max_threads = 6\n'
        'max_depth = 1\n\n'
        '[sandbox_workspace_write]\n'
        'network_access = false\n\n'
        '[tools]\n'
        f'web_search = {{ context_size = "low", allowed_domains = [{allowed_domains}] }}\n'
    )


def claude_settings_payload() -> dict[str, object]:
    return {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "disableBypassPermissionsMode": "disable",
        "permissions": {
            "allow": [
                *(f"WebFetch(domain:{domain})" for domain in ALLOWED_DOC_DOMAINS),
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
                "allowedDomains": list(ALLOWED_DOC_DOMAINS),
            },
        },
    }


def render_claude_settings() -> str:
    return json.dumps(claude_settings_payload(), indent=2, sort_keys=True) + "\n"


def write_repo_runtime_configs(repo_root: Path) -> list[Path]:
    repo_root = repo_root.expanduser().resolve()
    codex_dir = ensure_dir(repo_root / ".codex")
    claude_dir = ensure_dir(repo_root / ".claude")
    codex_path = codex_dir / "config.toml"
    claude_path = claude_dir / "settings.json"
    write_text(codex_path, render_codex_config())
    write_text(claude_path, render_claude_settings())
    return [codex_path, claude_path]
