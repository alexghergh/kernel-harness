# KernelBench agent harness

Run Codex or Claude on one KernelBench problem at a time through a narrow MCP tool surface, archive every attempt/profile/trace, and compare the final result against eager and `torch.compile` baselines.

At a high level:

- the harness prepares a fresh per-problem workspace
- the model does **not** use direct local file or shell tools for problem work
- local problem interaction goes through a shared MCP server that exposes the harness tool surface
- hosted web search stays tool-native and domain-restricted
- the durable record lives under `archive/`
- disposable live state lives under `state/`

For the detailed system contract, archive layout, workspace layout, MCP/config split, and runtime boundary notes, read `ARCHITECTURE.md`.

## Actual solver surface

For **both** Codex and Claude, the real problem environment is exposed only through the `kernelbench` MCP server. Hosted web access stays separate and tool-native.

- fixed read-only MCP resources: `AGENTS.md`, `INITIAL_PROMPT.md`, `SPEC.md`, `HARDWARE.md`, `GOAL_STATUS.md`, `problem_reference.py`, `candidate_model_new.py`
- bounded read tools: `list_workspace_dir` for `samples/` and `profiles/`, plus `read_workspace_file` for those history files and the fixed resources above
- write/action tools: `write_candidate`, `run_candidate`, `profile_ncu`, `goal_status`, `best_result`, `complete_problem`
- `goal_status` returns the live JSON status snapshot (remaining budget, attempt counts, best sample, baseline progress)
- `best_result` returns the current best measured correct attempt, including `sample_id` and archive-relative artifact paths
- native web stays separate from MCP and is limited to `docs.nvidia.com`

The client-specific enforcement differs slightly:

- **Codex** runs from an empty scratch cwd, with parent project-doc discovery disabled and the default shell tool disabled. There is no separate Codex deny-list for local file browsing, so the harness keeps the real workspace out of Codex’s direct local scope and exposes it only through MCP.
- **Claude** also runs from an empty scratch cwd, and its built-in local file/shell tools are explicitly denied (`Read`, `Write`, `Edit`, `MultiEdit`, `Bash`, `Glob`, `Grep`, `LS`). That means Claude reaches the problem environment only through MCP as well.


## Install KernelBench and this harness into the same environment

Create and activate the Python environment you want to use for both repos. The important part is that **KernelBench and this harness are installed into the same active environment**.

Example:

```bash
pyenv create <env-name>
pyenv activate <env-name>

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

The harness generates `state/config/` itself on launch. Authenticate once into repo-root tool dirs, and the harness will mirror only those repo-root auth files into `state/config/` each time it recreates shared tool state. It intentionally does not read `~/.codex` or `~/.claude`.

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

Preferred path: sign in once into repo-root `./.claude/`. The harness copies only `./.claude/.credentials.json` into `state/config/claude/` on launch. If a fresh `claude login` works in your normal shell but the harness does not, refresh the repo-root `./.claude/.credentials.json` file; the harness intentionally ignores `~/.claude`.

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

## Harness MCP smoke test

Use the real harness MCP server, not a separate dev server. This prepares one real problem workspace, exports the exact `DATA_ROOT` / `KBH_*` context the harness uses, and then talks to `python -m kernel_bench_experiment_agents.mcp` through the official Python MCP client.

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
./scripts/test_harness_mcp.sh
```

That is the supported smoke path for the actual harness server. The real launcher uses the same shared `state/config/codex/config.toml` and forwards the per-problem MCP context (`DATA_ROOT`, `KBH_WORKSPACE`, `KBH_CLIENT_TOOL`, `KBH_MCP_EVENTS_PATH`) into the stdio MCP server through Codex `env_vars`.

The shared helper agents `runner` and `profiler` are also loaded from `state/config/` when the client runtime supports them.

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
MODEL=claude-opus-4-7 \
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
MODEL=claude-opus-4-7 \
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

## Need more detail?

Read `ARCHITECTURE.md` for:

- archive contents and file meanings
- workspace contents and solver boundaries
- shared Codex / Claude config layout under `state/config/`
- the MCP-only local tool surface
- how profiling, attempts, traces, and summaries are recorded
