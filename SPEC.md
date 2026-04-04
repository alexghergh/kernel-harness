# KernelBench Experiment Spec

## objective

Evaluate current frontier coding agents on the full KernelBench benchmark while keeping the harness simple, reproducible, and compatible with the official KernelBench artifact layout.

The immediate target is a tool-neutral harness that can launch either Codex or Claude through the same `run_agent_*` surface.

## scope

In scope:

- KernelBench levels 1, 2, and 3
- one agent session per problem
- one active solver per `(run_name, level, problem_id)`
- local timing and profiling tools callable by the selected agent
- workspace-local `SPEC.md` files that pin each solver to one problem and two explicit runtime targets
- workspace-local `HARDWARE.md` files derived from the configured `GPU_NAME`
- optional live web search restricted to NVIDIA documentation domains
- official-style run artifacts for evaluated kernels
- maintainer-level project memory for harness decisions only

Out of scope for this first implementation:

- LaTeX report generation
- a Python-hosted agent loop that replaces agent CLI planning
- broad web access

## environment assumptions

- execution happens on a remote GPU node
- the node exposes one or more NVIDIA GPUs locally
- the number of GPUs is configurable
- the exact GPU name can be passed in when needed
- the official KernelBench repository is available locally via `KERNELBENCH_ROOT`
- KernelBench should be set up in its own environment first
- the runtime interpreter for that environment is passed in as `KERNELBENCH_PYTHON`
- this repository is run directly from source
- the official KernelBench checkout may be installed via `uv sync --extra gpu` or `pip install -e ".[gpu]"`

## comparison baselines

Every problem is compared against two separate baselines:

- eager PyTorch
- `torch.compile`

Some problems may favor one baseline over the other. Both must be recorded separately.

## success criteria

Per problem:

- generate a correct kernel
- measure the kernel runtime in milliseconds
- compare it against both eager PyTorch and `torch.compile`
- preserve the best observed candidate and its measurements

Aggregate:

- retain official-style per-sample artifacts for analysis
- make it easy to compute overall KernelBench metrics later

## agent constraints

- the main agent should stay scoped to exactly one assigned problem
- the main agent is expected to behave as a planner and coordinator
- when useful, the main agent should spawn subagents for:
  - running kernel timing and summarizing output
  - running `ncu` profiling and summarizing output
- local timing and profiling must happen through local commands only
- live web search, if enabled, is restricted to NVIDIA docs domains only
- problem-solving agents do not write to shared repo-level memory files
- the solver agent should operate from a problem-local workspace and not treat maintainer docs as part of its normal context
- the solver must terminate only through an explicit completion record; missing completion is a failed run

## artifact policy

Evaluated candidates use the official KernelBench naming convention:

- `runs/<run_name>/level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py`

Each evaluated attempt consumes a new `sample_id`.

Scratch drafts that have not been evaluated do not consume an official `sample_id`.

## timing and profiling

Timing:

- use KernelBench-compatible evaluation logic
- serialize GPU-consuming operations through a shared lease count equal to the configured GPU capacity
- store per-attempt runtime outputs in project-local manifests, including failed attempts
- store per-problem agent trace artifacts and an explicit terminal completion record
- compare against both eager and `torch.compile` baselines during summary/analysis using official baseline JSON inputs
- allow the same named run to accumulate results over multiple disjoint problem subsets and multiple days

Profiling:

- use `ncu` locally on the assigned GPU
- store raw profiler output and related manifests under project-local artifacts

## agent CLI policy

- use `codex exec` first for reproducible runs
- support `codex exec` for Codex and `claude --print --output-format stream-json` for Claude through the same launcher surface
- use project-local `.codex/config.toml` for Codex and project-local `.claude/settings.json` for Claude
- allow live web search only for `docs.nvidia.com`
- on the current shared cluster, hard shell/network isolation is not reliably enforceable through the agent CLIs, so path restrictions are currently enforced mainly through the generated workspace contract and post-run trace auditing
- launch the selected agent from the problem workspace rather than from the repository root
- generate a workspace-local `HARDWARE.md` from the configured `GPU_NAME` and fail fast on unknown GPU families

## deliverables

- repo-local docs and memory
- tool-specific config and custom agents
- generated workspace-local problem and hardware docs
- a local helper package with local commands
- generic shell launchers for single-problem and batched runs
- run-summary commands for compile, correctness, pass@k, and baseline-comparison statistics
