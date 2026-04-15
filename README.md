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
- if your timing results live outside the default KernelBench timing tree, you will set `KERNELBENCH_TIMINGS_DIR` when launching runs

## Authenticate the agent tools

Run these commands from the harness repo root.

### Codex

Preferred path: sign in once using the shared harness Codex home.

```bash
DATA_ROOT=. CODEX_HOME="./state/config/codex" codex login --device-auth
DATA_ROOT=. CODEX_HOME="./state/config/codex" codex login status
```

Alternative: export an API key instead.

```bash
export OPENAI_API_KEY=...
```

### Claude Code

Preferred path: sign in once using the shared harness Claude config dir.

```bash
DATA_ROOT=. CLAUDE_CONFIG_DIR="./state/config/claude" claude login
```

Alternatives: export API credentials or an OAuth token.

```bash
export ANTHROPIC_API_KEY=...
# or
export ANTHROPIC_AUTH_TOKEN=...
# or
export CLAUDE_CODE_OAUTH_TOKEN=...
```

## Most common runs

Run these scripts from the harness repo root.

### Run one problem

```bash
TOOL=codex \
RUN_NAME=kernelbench-codex-h100-v3 \
LEVEL=1 \
PROBLEM_ID=1 \
MODEL=gpt-5-codex \
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
MODEL=opus \
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
MODEL=gpt-5-codex \
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
MODEL=opus \
TIME_BUDGET_MINUTES=180 \
PRECISION=bf16 \
KERNELBENCH_ROOT=/path/to/KernelBench \
HARDWARE_NAME=H100 \
./scripts/run_agent_range.sh
```

### Submit the Slurm wrapper

Submit from the harness repo root.

```bash
ybatch \
  --export=TOOL=codex,RUN_NAME=kernelbench-codex-h100-v3,LEVEL=1,START_PROBLEM_ID=1,END_PROBLEM_ID=10,MODEL=gpt-5-codex,TIME_BUDGET_MINUTES=180,PRECISION=bf16,KERNELBENCH_ROOT=/path/to/KernelBench,HARDWARE_NAME=H100 \
  ./scripts/run_agent_problem.slurm.sh
```

Use `sbatch` instead of `ybatch` on clusters that expose plain Slurm submission.

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

## Need more detail?

Read `ARCHITECTURE.md` for:

- archive contents and file meanings
- workspace contents and solver boundaries
- shared Codex / Claude config layout under `state/config/`
- the MCP-only local tool surface
- how profiling, attempts, traces, and summaries are recorded
