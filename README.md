# KernelBench harness

This repository runs autonomous coding agents on KernelBench optimization problems, one problem at a time, and records the durable results under `archive/`.

At a high level:

- the harness prepares a fresh per-problem workspace
- the model does **not** use direct local file or shell tools for problem work
- local problem interaction goes through a shared MCP server that exposes the harness tool surface
- hosted web search stays tool-native and domain-restricted
- the durable record lives under `archive/`
- disposable live state lives under `state/`

For the detailed system contract, archive layout, workspace layout, MCP/config split, and runtime boundary notes, read `ARCHITECTURE.md`.

## Install KernelBench and this harness into the same environment

Create and activate the Python environment you want to use for both repos. The important part is that **KernelBench and this harness are installed into the same active environment**.

Example:

```bash
pyenv create <env-name>
# activate that environment in your shell

cd /path/to/KernelBench
uv pip install -e .

cd /path/to/kernel-bench-experiment-agents
uv pip install -e .
```

This harness assumes:

- the official KernelBench checkout already exists
- the KernelBench timing files already exist for your hardware
- `KERNELBENCH_TIMINGS_DIR` is optional; set it only when your timing results live outside the default KernelBench timing tree

## Authenticate the agent tools

Run these commands from the harness repo root.

The harness generates `state/config/` itself on launch. Authenticate once into repo-root tool dirs, and the harness will copy just the auth files into `state/config/` each time it recreates shared tool state.

### Codex

Preferred path: sign in once into repo-root `./.codex/`, using file-backed credentials so the harness can copy `auth.json` into `state/config/codex/` on launch.

```bash
mkdir -p .codex
CODEX_HOME="./.codex" codex -c cli_auth_credentials_store=file login --device-auth
CODEX_HOME="./.codex" codex login status
```

Alternative: export an API key instead.

```bash
export OPENAI_API_KEY=...
```

### Claude Code

Preferred path: sign in once into repo-root `./.claude/`. The harness copies `./.claude/.credentials.json` into `state/config/claude/` on launch.

```bash
mkdir -p .claude
CLAUDE_CONFIG_DIR="./.claude" claude login
```

Alternatives: export API credentials or an OAuth token.

```bash
export ANTHROPIC_API_KEY=...
# or
export ANTHROPIC_AUTH_TOKEN=...
# or
export CLAUDE_CODE_OAUTH_TOKEN=...
```

## Standalone MCP smoke test with Codex

If you want to debug Codex+MCP outside the full harness first, use the tiny smoke server:

```bash
mkdir -p /tmp/kernelbench-mcp-smoke
CODEX_HOME=./.codex-smoke codex mcp add kernelbench-smoke \
  --env MCP_SMOKE_ROOT=/tmp/kernelbench-mcp-smoke \
  -- "$(which python)" -m kernel_bench_experiment_agents.mcp.smoke

CODEX_HOME=./.codex-smoke codex exec --skip-git-repo-check --json \
  --model gpt-5.4 \
  "Use the kernelbench-smoke MCP tools only. Write hello.txt containing exactly hello from MCP. Then read it back and report the contents."
```

That path does not touch any KernelBench workspace or harness state. It only verifies that Codex can initialize the MCP server and call simple read/write tools rooted under `MCP_SMOKE_ROOT`.

If you want to launch the real harness MCP server by hand, prepare a problem workspace first, then export:

- `DATA_ROOT`
- `KBH_WORKSPACE`
- `KBH_CLIENT_TOOL`
- `KBH_MCP_EVENTS_PATH`

and run:

```bash
python -m kernel_bench_experiment_agents.mcp
```

## Most common runs

Run these scripts from the harness repo root.

### Run one problem

```bash
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v3 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=gpt-5.4 \
TIME_BUDGET_MINUTES=180 \
PRECISION=bf16 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_problem.sh
```

### Run one problem with Claude

```bash
TOOL=claude \
RUN_NAME=kernelbench-claude-h100-v3 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=opus-4.6 \
TIME_BUDGET_MINUTES=180 \
PRECISION=bf16 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_problem.sh
```

### Run a contiguous range

```bash
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v3 \
LEVEL=1 \
START_PROBLEM_ID=1 \
END_PROBLEM_ID=10 \
MODEL=gpt-5.4 \
TIME_BUDGET_MINUTES=180 \
PRECISION=bf16 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_range.sh
```

### Run an explicit problem list

```bash
TOOL=claude \
RUN_NAME=kernelbench-claude-h100-v3 \
LEVEL=1 \
PROBLEM_IDS=1,4,9 \
MODEL=opus-4.6 \
TIME_BUDGET_MINUTES=180 \
PRECISION=bf16 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_range.sh
```

### Submit the Slurm wrapper

Submit from the harness repo root. The script itself carries the default `#SBATCH` / `#YBATCH` header block for the common H100 path, so the usual launch is still:

```bash
ybatch --export=TOOL=codex,RUN_NAME=kernelbench-codex-h100-v3,LEVEL=1,START_PROBLEM_ID=1,END_PROBLEM_ID=10,MODEL=gpt-5.4,TIME_BUDGET_MINUTES=180,PRECISION=bf16,KERNELBENCH_ROOT=/path/to/KernelBench,HARDWARE_NAME=H100 ./scripts/run_agent_problem.slurm.sh
```

Override those scheduler defaults in the script header or on the submit command when your cluster needs something different. Use `sbatch` instead of `ybatch` on clusters that expose plain Slurm submission.

### Summarize one archived run

```bash
kbharness summarize-run --run-name kernelbench-codex-h100-v3
```

This scans only `archive/<run_name>/` and writes `archive/<run_name>/run_summary.json`.

## User knobs you will actually use

These are the main variables worth changing:

- `DATA_ROOT=/path/for/archive-and-state` if you want artifacts somewhere other than `./`
- `TOOL=codex|claude`
- `MODEL=...`
- `RUN_NAME=...`
- `LEVEL=...`
- `PROBLEM_ID=...`
- `START_PROBLEM_ID=...` / `END_PROBLEM_ID=...`
- `PROBLEM_IDS=1,4,9`
- `TIME_BUDGET_MINUTES=...`
- `PRECISION=bf16`
- `KERNELBENCH_ROOT=/path/to/KernelBench`
- `HARDWARE_NAME=H100`
- `KERNELBENCH_TIMINGS_DIR=/path/to/results/timing/<hardware>` when you need a non-default timings location
- inherited `CUDA_VISIBLE_DEVICES` when you want to pin visible GPUs from the scheduler or shell

## Where to look after a run

The only durable copy-out root is:

```text
archive/<run_name>/
```

Live workspaces, locks, shared tool config, per-problem scratch directories, and build products live under `state/` and are disposable once no run is active.

## CLI surface

Installing this repo exposes the harness CLI:

```bash
kbharness --help
```

The launcher scripts are the normal entrypoints. The CLI exists mainly so those scripts, workspace wrappers, and the MCP server can call the harness internals in a stable way.

## Manual MCP smoke test

If you want to debug Codex↔MCP wiring outside the harness, use the tiny standalone dev server:

```bash
export MCP_DEV_ROOT=/tmp/kernelbench-mcp-smoke
mkdir -p "$MCP_DEV_ROOT"
mkdir -p /tmp/kernelbench-codex-home

cat > /tmp/kernelbench-codex-home/config.toml <<'EOF'
approval_policy = "never"
sandbox_mode = "read-only"
project_root_markers = []

[features]
shell_tool = false

[mcp_servers.kernelbench_dev]
command = "$(python -c 'import sys; print(sys.executable)')"
args = ["-m", "kernel_bench_experiment_agents.mcp.dev_server"]
env = { MCP_DEV_ROOT = "/tmp/kernelbench-mcp-smoke" }
required = true
startup_timeout_sec = 20
EOF

CODEX_HOME=/tmp/kernelbench-codex-home \
  codex -a never --sandbox read-only --skip-git-repo-check --cd /tmp/kernelbench-mcp-smoke \
  "Use only the kernelbench_dev MCP tools. Write hello.txt with the exact text hello from codex, then read it back."
```

Codex supports stdio MCP servers in `config.toml`, including explicit `env` for the server, and the MCP Inspector is also a good first-step debugger for local stdio servers. See the official Codex MCP docs and MCP Inspector docs for the reference behavior.

## Need more detail?

Read `ARCHITECTURE.md` for:

- archive contents and file meanings
- workspace contents and solver boundaries
- shared Codex / Claude config layout under `state/config/`
- the MCP-only local tool surface
- how profiling, attempts, traces, and summaries are recorded
