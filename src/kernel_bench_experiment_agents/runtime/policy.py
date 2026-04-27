"""Render Codex and Claude runtime config from the harness policy.

The launcher materializes tool-private config homes under disposable `state/`
paths. Those dirs hold runtime config, copied auth, helper agents, and
tool-managed local state, while the solver workspace remains free of tool
auth/config files.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from kernel_bench_experiment_agents.agent_contract.agent_specs import write_shared_helper_agent_specs
from kernel_bench_experiment_agents.agent_contract.policy import (
    ALLOWED_WEB_DOMAINS,
    COMMAND_MCP_SERVER_NAME,
    command_mcp_tool_names,
)
from kernel_bench_experiment_agents.runtime.project import ensure_dir, make_executable, write_text


COMMAND_MCP_SERVER_ENV_VARS: tuple[str, ...] = (
    "KBH_COMMAND_SOCKET",
)

CLAUDE_BUILTIN_TOOLS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "LS",
    "Glob",
    "Grep",
    "Task",
    "WebSearch",
    "WebFetch",
)

CLAUDE_ALLOWED_TOOL_PATTERNS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "LS",
    "Glob",
    "Grep",
    "Task",
    "WebSearch",
    *(f"WebFetch(domain:{domain})" for domain in ALLOWED_WEB_DOMAINS),
    *command_mcp_tool_names(),
)


def _python_command() -> str:
    """Return the interpreter that generated this tool config."""
    return str(Path(sys.executable).expanduser())


def _mirror_optional_file(source: Path, target: Path) -> Path | None:
    if source.exists() and source.is_file():
        ensure_dir(target.parent)
        shutil.copy2(source, target)
        return target
    if target.exists() or target.is_symlink():
        target.unlink()
    return None


def sync_repo_auth_into_tool_state(config_root: Path, *, repo_root: Path | None = None) -> list[Path]:
    """Copy repo-root auth caches into a generated per-problem tool home.

    The user authenticates once under repo-root `.codex/` and `.claude/`; each launch gets
    its own disposable `state/tool_state/...` copy seeded only from those source dirs.
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
    """Render the per-problem Codex config.toml."""
    allowed_domains = ", ".join(f'"{domain}"' for domain in ALLOWED_WEB_DOMAINS)
    env_vars = ", ".join(f'"{name}"' for name in COMMAND_MCP_SERVER_ENV_VARS)
    python_command = json.dumps(_python_command())
    payload = (
        '# Generated from src/kernel_bench_experiment_agents/runtime/policy.py\n'
        'personality = "pragmatic"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "workspace-write"\n'
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
        f'[mcp_servers.{COMMAND_MCP_SERVER_NAME}]\n'
        f'command = {python_command}\n'
        'args = ["-m", "kernel_bench_experiment_agents.command_mcp"]\n'
        f'env_vars = [{env_vars}]\n'
        'required = true\n'
        'startup_timeout_sec = 20\n'
        'tool_timeout_sec = 1200\n\n'
        '[tools]\n'
        f'web_search = {{ context_size = "low", allowed_domains = [{allowed_domains}] }}\n'
    )
    return payload


def claude_websearch_hook_path(claude_config_dir: Path) -> Path:
    return claude_config_dir / "hooks" / "restrict_nvidia_docs_websearch.py"


def claude_file_policy_hook_path(claude_config_dir: Path) -> Path:
    return claude_config_dir / "hooks" / "restrict_workspace_file_tools.py"



def render_claude_websearch_hook() -> str:
    allowed_domains = ", ".join(json.dumps(domain) for domain in ALLOWED_WEB_DOMAINS)
    reason = json.dumps(
        f"Restrict WebSearch to {', '.join(ALLOWED_WEB_DOMAINS)} for this harness."
    )
    return (
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n\n"
        "import json\n"
        "import sys\n\n"
        f"ALLOWED_DOMAINS = [{allowed_domains}]\n\n"
        "def main() -> int:\n"
        "    try:\n"
        "        event = json.load(sys.stdin)\n"
        "    except Exception:\n"
        "        return 0\n\n"
        "    tool_input = event.get('tool_input')\n"
        "    if not isinstance(tool_input, dict):\n"
        "        return 0\n\n"
        "    updated_input = dict(tool_input)\n"
        "    updated_input['allowed_domains'] = ALLOWED_DOMAINS\n"
        "    payload = {\n"
        "        'hookSpecificOutput': {\n"
        "            'hookEventName': 'PreToolUse',\n"
        "            'permissionDecision': 'allow',\n"
        f"            'permissionDecisionReason': {reason},\n"
        "            'updatedInput': updated_input,\n"
        "        }\n"
        "    }\n"
        "    json.dump(payload, sys.stdout)\n"
        "    sys.stdout.write('\\n')\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def render_claude_file_policy_hook() -> str:
    return (
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "READ_TOOLS = {'Read', 'LS', 'Glob', 'Grep'}\n"
        "WRITE_TOOLS = {'Write', 'Edit', 'MultiEdit'}\n"
        "PATH_KEYS = ('file_path', 'path')\n\n"
        "def _decision(decision: str, reason: str) -> dict[str, object]:\n"
        "    return {\n"
        "        'hookSpecificOutput': {\n"
        "            'hookEventName': 'PreToolUse',\n"
        "            'permissionDecision': decision,\n"
        "            'permissionDecisionReason': reason,\n"
        "        }\n"
        "    }\n\n"
        "def _resolve_tool_path(raw_path: object, workspace: Path) -> Path | None:\n"
        "    if raw_path in (None, ''):\n"
        "        return workspace\n"
        "    if not isinstance(raw_path, str):\n"
        "        return None\n"
        "    path = Path(raw_path).expanduser()\n"
        "    if not path.is_absolute():\n"
        "        path = workspace / path\n"
        "    return path.resolve(strict=False)\n\n"
        "def _is_relative_to(path: Path, root: Path) -> bool:\n"
        "    try:\n"
        "        path.relative_to(root)\n"
        "        return True\n"
        "    except ValueError:\n"
        "        return False\n\n"
        "def main() -> int:\n"
        "    try:\n"
        "        event = json.load(sys.stdin)\n"
        "    except Exception:\n"
        "        return 0\n"
        "    tool_name = str(event.get('tool_name') or event.get('toolName') or '').strip()\n"
        "    tool_input = event.get('tool_input')\n"
        "    if not isinstance(tool_input, dict):\n"
        "        tool_input = {}\n"
        "    workspace_raw = os.environ.get('KBH_WORKSPACE', '').strip()\n"
        "    candidate_raw = os.environ.get('KBH_CANDIDATE_PATH', '').strip()\n"
        "    if not workspace_raw or not candidate_raw:\n"
        "        json.dump(_decision('deny', 'KernelBench workspace policy env is missing.'), sys.stdout)\n"
        "        sys.stdout.write('\\n')\n"
        "        return 2\n"
        "    workspace = Path(workspace_raw).resolve(strict=False)\n"
        "    candidate = Path(candidate_raw).resolve(strict=False)\n"
        "    if tool_name in WRITE_TOOLS:\n"
        "        raw_path = next((tool_input.get(key) for key in PATH_KEYS if key in tool_input), None)\n"
        "        target = _resolve_tool_path(raw_path, workspace)\n"
        "        if target != candidate:\n"
        "            reason = 'Only candidate_model_new.py is writable in this KernelBench workspace.'\n"
        "            json.dump(_decision('deny', reason), sys.stdout)\n"
        "            sys.stdout.write('\\n')\n"
        "            return 2\n"
        "        json.dump(_decision('allow', 'Write is within the KernelBench candidate file.'), sys.stdout)\n"
        "        sys.stdout.write('\\n')\n"
        "        return 0\n"
        "    if tool_name in READ_TOOLS:\n"
        "        if tool_name == 'Glob':\n"
        "            pattern = tool_input.get('pattern')\n"
        "            if isinstance(pattern, str):\n"
        "                if '..' in Path(pattern).parts:\n"
        "                    reason = 'Glob patterns may not escape the assigned KernelBench workspace.'\n"
        "                    json.dump(_decision('deny', reason), sys.stdout)\n"
        "                    sys.stdout.write('\\n')\n"
        "                    return 2\n"
        "                if pattern.startswith('/') and not (pattern == str(workspace) or pattern.startswith(str(workspace) + '/')):\n"
        "                    reason = 'Absolute Glob patterns are limited to the assigned KernelBench workspace.'\n"
        "                    json.dump(_decision('deny', reason), sys.stdout)\n"
        "                    sys.stdout.write('\\n')\n"
        "                    return 2\n"
        "        raw_path = next((tool_input.get(key) for key in PATH_KEYS if key in tool_input), None)\n"
        "        target = _resolve_tool_path(raw_path, workspace)\n"
        "        if target is None or not (target == workspace or _is_relative_to(target, workspace)):\n"
        "            reason = 'File reads are limited to the assigned KernelBench workspace.'\n"
        "            json.dump(_decision('deny', reason), sys.stdout)\n"
        "            sys.stdout.write('\\n')\n"
        "            return 2\n"
        "        json.dump(_decision('allow', 'Read is within the KernelBench workspace.'), sys.stdout)\n"
        "        sys.stdout.write('\\n')\n"
        "        return 0\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )



def _claude_websearch_hook_command(hook_path: Path) -> str:
    return f"python3 {json.dumps(str(hook_path))}"


def _claude_file_policy_hook_command(hook_path: Path) -> str:
    return f"python3 {json.dumps(str(hook_path))}"



def claude_settings_payload(
    *,
    websearch_hook_path: Path,
    file_policy_hook_path: Path,
) -> dict[str, object]:
    """Build the Claude settings payload that lives under CLAUDE_CONFIG_DIR/settings.json."""
    allow_tools = list(CLAUDE_ALLOWED_TOOL_PATTERNS)
    file_policy_hook = {
        "type": "command",
        "command": _claude_file_policy_hook_command(file_policy_hook_path),
    }
    return {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "permissions": {
            "allow": allow_tools,
            "deny": ["Bash(*)"],
            "disableBypassPermissionsMode": "disable",
        },
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "WebSearch",
                    "hooks": [
                        {
                            "type": "command",
                            "command": _claude_websearch_hook_command(websearch_hook_path),
                        }
                    ],
                },
                *(
                    {
                        "matcher": tool_name,
                        "hooks": [file_policy_hook],
                    }
                    for tool_name in (
                        "Read",
                        "Write",
                        "Edit",
                        "MultiEdit",
                        "LS",
                        "Glob",
                        "Grep",
                    )
                ),
            ]
        },
        "sandbox": {"enabled": False},
    }



def claude_user_config_payload() -> dict[str, object]:
    """Build the per-problem Claude user config.

    Per-run command MCP registration is generated by the launcher because it
    needs the private command-broker socket path.
    """
    return {}



def render_claude_settings(
    *,
    websearch_hook_path: Path,
    file_policy_hook_path: Path,
) -> str:
    return json.dumps(
        claude_settings_payload(
            websearch_hook_path=websearch_hook_path,
            file_policy_hook_path=file_policy_hook_path,
        ),
        indent=2,
        sort_keys=True,
    ) + "\n"



def render_claude_user_config() -> str:
    return json.dumps(claude_user_config_payload(), indent=2, sort_keys=True) + "\n"


def write_tool_state(config_root: Path, *, repo_root: Path | None = None) -> list[Path]:
    config_root = ensure_dir(config_root.expanduser().resolve())
    codex_dir = ensure_dir(config_root / "codex")
    claude_dir = ensure_dir(config_root / "claude")
    codex_path = codex_dir / "config.toml"
    claude_settings_path = claude_dir / "settings.json"
    claude_user_config_path = claude_dir / ".claude.json"
    claude_websearch_hook = claude_websearch_hook_path(claude_dir)
    claude_file_policy_hook = claude_file_policy_hook_path(claude_dir)
    write_text(codex_path, render_codex_config())
    write_text(claude_websearch_hook, render_claude_websearch_hook())
    make_executable(claude_websearch_hook)
    write_text(claude_file_policy_hook, render_claude_file_policy_hook())
    make_executable(claude_file_policy_hook)
    write_text(
        claude_settings_path,
        render_claude_settings(
            websearch_hook_path=claude_websearch_hook,
            file_policy_hook_path=claude_file_policy_hook,
        ),
    )
    write_text(claude_user_config_path, render_claude_user_config())
    written = [
        codex_path,
        claude_websearch_hook,
        claude_file_policy_hook,
        claude_settings_path,
        claude_user_config_path,
    ]
    written.extend(sync_repo_auth_into_tool_state(config_root, repo_root=repo_root))
    written.extend(
        write_shared_helper_agent_specs(codex_home=codex_dir, claude_config_dir=claude_dir)
    )
    return written
