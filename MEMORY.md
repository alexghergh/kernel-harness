# Memory

Maintainer-only memory for the harness and experiment design.

This file is not part of the per-problem optimizer workflow and should not be written by problem-solving agents.

## 2026-03-22

### current state

- the harness is now tool-neutral at the launcher surface:
  - `scripts/run_agent_problem.sh`
  - `scripts/run_agent_range.sh`
  - `scripts/run_agent_problem.slurm.sh`
- `TOOL=codex` and `TOOL=claude` are both supported through that shared surface
- raw trace artifacts are stored under `artifacts/<run_name>/.../agent/`
- repo-local `.codex/` and `.claude/` are both part of the checked-in tool configuration
- per-problem runtime state now lives under `.runtime/agent_home/...` when the selected tool needs isolated mutable state

### project direction

- focus first on a KernelBench-only experiment
- use current frontier coding agents and gauge results on the full KernelBench benchmark
- preserve compatibility with the official KernelBench run layout and analysis scripts where possible

### initial harness decision

- initial direction changed from a Python-hosted API planner to a Codex-first harness
- Codex CLI can authenticate with an OpenAI API key, so OpenAI API credits are sufficient
- use `codex exec` first for reproducible runs
- do not build a Python replacement for Codex planning in v2

### initial codex behavior

- Codex is the planner and optimizer
- Codex should stay scoped to one assigned problem at a time
- local commands should provide timing and profiling
- subagents are encouraged for:
  - running the kernel and summarizing output
  - running `ncu` and summarizing profiler output
- keep the main agent context clean by delegating noisy output to those helpers

### web policy

- web access is allowed only for NVIDIA documentation
- intended domains:
  - `docs.nvidia.com`
- shell/network access should stay disabled unless explicitly needed later

### baseline policy

- compare against two baselines separately:
  - eager PyTorch
  - `torch.compile`
- some problems may favor one baseline over the other, so both must be retained

### runtime assumptions

- no local GPU is available on the development machine
- actual execution happens on a remote GPU node
- the number of GPUs is configurable
- the GPU name can be passed in if autodetection is not enough
- the official KernelBench repository is available separately via `KERNELBENCH_ROOT`
- KernelBench should be set up in its own Python environment first
- the runtime should use an explicit KernelBench interpreter path rather than relying on this repo being installed

### artifact policy

- official evaluated kernel filenames should match KernelBench:
  - `runs/<run_name>/level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py`
- each evaluated attempt increments `sample_id`
- scratch drafts do not consume an official `sample_id`
- one active solver per `(run_name, level, problem_id)` is the supported model

### docs required

- `SPEC.md`
- `ARCHITECTURE.md`
- `AGENTS.md`
- `MEMORY.md`
- a small `README.md`

### config and agents

- use project-local Codex config in `.codex/config.toml`
- use project-local custom agents under `.codex/agents/`
- later add project-local Claude config in `.claude/settings.json`
- later add project-local Claude agents under `.claude/agents/`

### implemented scaffold

- created the first docs set:
  - `README.md`
  - `SPEC.md`
  - `ARCHITECTURE.md`
  - `AGENTS.md`
  - `MEMORY.md`
- created project-local Codex config with:
  - live web search enabled
  - NVIDIA docs domain restrictions
  - workspace-write sandbox with shell network disabled
- created two custom Codex agents:
  - `runner`
  - `profiler`
- created a local helper package with commands for:
  - problem workspace preparation
  - problem inspection
  - candidate evaluation
  - `ncu` profiling
  - best-result lookup
- created the initial single-problem launcher
- created a KernelBench environment setup helper:
  - `scripts/setup_kernelbench_env.sh`
- changed the intended solver launch model:
  - workspaces live outside the repo by default
  - solver agents get a workspace-local `AGENTS.md`
  - root contributor docs are not part of the solver’s normal context
- added shared GPU leasing:
  - timing and `ncu` profiling now serialize against the configured GPU slot count
- added separate per-problem artifact locking:
  - `sample_id` reservation and manifest/history writes are serialized independently of GPU leasing
- added a per-problem solver session lock in the launcher:
  - duplicate solver launches for the same run/level/problem now fail fast
- added run accumulation and summarization support:
  - a run can be built up over disjoint problem subsets and multiple days
  - summary command computes compile/correct rates, pass@k estimates, and baseline-beating rates
- changed attempt persistence:
  - per-attempt manifests are created before evaluation starts
  - terminal manifests and `history.jsonl` rows are now written even for failed evaluations
- added a stricter workspace contract:
  - each problem workspace now gets a local `SPEC.md`
  - each problem workspace now gets a local `HARDWARE.md`
  - resolved eager and compile baseline values are written into `baseline.json`
  - `goal_status.json` and `GOAL_STATUS.md` are generated from measured artifacts
- added explicit terminal completion handling:
  - solver runs must finish through `complete-problem`
  - missing completion is converted into `failed_to_generate`
- added raw agent trace capture:
  - raw CLI output is stored per problem under `artifacts/<run_name>/.../agent/events.jsonl`
  - a normalized `conversation.json` is derived for easier inspection
- changed runtime isolation:
  - repo-local `.codex/` remains the canonical login/config source for Codex
  - repo-local `.claude/` remains the canonical checked-in settings source for Claude
  - each problem launch now gets tool-specific isolated runtime state under `.runtime/agent_home/...` when needed
- changed budget handling:
  - remaining budget now excludes recorded GPU lock wait time from timing and profiling
  - the launcher stops unresolved runs at the corrected budget limit and records `budget_exhausted`
- later refactored the launcher surface:
  - `run_agent_problem.sh`, `run_agent_range.sh`, and `run_agent_problem.slurm.sh` became the canonical entrypoints
  - thin per-tool wrapper scripts were later removed once the generic surface was stable
  - the harness gained Claude support without changing the shared workspace/evaluation contract

### current limitations

- the helper package is intentionally thin and untested against a live KernelBench install in this environment
- KernelBench Python API handling includes compatibility assumptions because the official repo cannot be executed locally here
- eager and `torch.compile` baselines are resolved into workspace-local files at launch time, but per-attempt manifests still store only candidate results
- per-problem runs are assumed to finish end-to-end; no resume-specific workspace memory is planned

### later items

- support LaTeX/report generation later
- broader HPC migration benchmark remains separate from this first KernelBench experiment

### useful official KernelBench reminders

- run one problem with `scripts/generate_and_eval_single_sample.py`
- run all problems with `scripts/generate_samples.py` then `scripts/eval_from_generations.py`
- analyze benchmark results with `scripts/benchmark_eval_analysis.py`
- generate hardware-specific baseline timing with `scripts/generate_baseline_time.py`
