# Architecture

This document describes the current system contract for the KernelBench harness.

## Scope

The harness runs one autonomous coding agent on one KernelBench optimization problem at a time, records the run, and stores the durable result under `archive/`.

The harness is responsible for:

- preparing a fresh workspace for one problem
- rendering the solver-facing contract into that workspace
- launching Codex or Claude with a narrow local surface
- exposing direct workspace reads/edits while keeping privileged measured actions behind a launcher-owned command broker
- recording attempts, traces, status, completion, and profiler outputs
- aggregating archived results through `summarize-run`

The harness is responsible for pointing `third_party/KernelBench/` at the configured KernelBench fork. Local harness-specific KernelBench fixes should live in that fork rather than as vendored patches in this repository.

The clean user-facing repo entrypoint is `./kb`. It provides a small setup, run, range, and submit surface while the lower-level shell scripts and Python CLI stay available underneath it. `./kb setup` uses `uv` to provision a Python 3.10 environment under `./.venv` by default, and builds the vendored `third_party/landrun` checkout into `third_party/bin/landrun`.

The committed submission support stays intentionally narrow: `./kb submit` wraps a generic Slurm submission flow and uses `ybatch` only when that command is already present on PATH. On clusters with a site-local `ybatch` wrapper, users must still pass the site-local resource name explicitly. The harness does not probe cluster topology or choose hardware automatically.

## Documentation audiences

There are two documentation surfaces.

Root docs are for maintainers and users:

- `README.md`
- `ARCHITECTURE.md`
- `AGENTS.md`
- `MEMORY.md`

Generated workspace docs are for the solver agent:

- `AGENTS.md`
- `SPEC.md`
- `HARDWARE.md`
- `GOAL_STATUS.md`

The same filename does not imply the same audience.

## Solver policy surfaces

The solver is nudged through several small surfaces rather than one giant prompt:

- generated workspace `AGENTS.md` — durable top-level contract
- generated `INITIAL_PROMPT.md` — opening run-specific instructions
- generated `GOAL_STATUS.md` — live status file re-read after measured actions
- direct command tools — the measured harness action and brokered NVIDIA-docs research surface
- helper-agent specs for `runner` and `profiler` — narrow delegated roles when the client runtime supports helper spawning

The intended behavior is:

- the main agent acts as the planner-manager
- `runner` handles measured evaluation by default
- `profiler` handles Nsight Compute work by default
- direct `run_candidate` / `profile_ncu` from the main agent are valid paths when helper spawning is unavailable
- `research_nvidia_docs` is the canonical audited docs-research path when the next optimization branch depends on hardware-specific behavior

## High-level flow

For one `(run_name, level, problem_id)` tuple, the harness does this:

1. resolves the problem and baseline information from the KernelBench checkout
2. prepares a fresh self-contained workspace
3. renders solver-facing docs and compatibility wrapper scripts into that workspace
4. prepares the shared tool-private config under `state/config/`
5. launches Codex or Claude in the prepared workspace under Landrun
6. gives the model direct workspace reads, a single writable candidate file, and direct-command access to the launcher-owned broker for privileged actions
7. records attempts, traces, completion state, and optional profiles
8. writes the durable record under `archive/<run_name>/level_<level>/problem_<problem_id>/`

The solver does not decide measured outcomes. The harness decides, from recorded attempts, whether the best correct solution beat eager PyTorch, `torch.compile`, both, or neither.

## Durable vs disposable state

### Durable: `archive/`

`archive/` is the only directory you should copy out for post-run analysis.

Per problem, the durable record is:

```text
archive/<run_name>/level_<level>/problem_<problem_id>/
  archive_manifest.json
  contract/
  agent/
  attempts/
  profiles/
```

### Disposable: `state/`

`state/` contains live mutable runtime data only:

- `state/workspaces/`
- `state/build/`
- `state/locks/`
- `state/config/codex/`
- `state/config/claude/`
- `state/cwd/codex/`
- `state/cwd/claude/`

It is safe to delete `state/` only when no run is active.

`DATA_ROOT` controls where `archive/` and `state/` are written. It does **not** decide where repository source files live.

## Shared tool-private config

The workspace is solver-visible. Tool-private config and auth live outside it.

The launcher and wrappers do not require an installed `kbharness` console script. The repo ships a local `scripts/kbharness` wrapper that runs `python -m kernel_bench_experiment_agents.runtime.cli` against the repo source tree, and `./kb setup` provisions the harness into a `uv`-managed virtual environment.

Landrun is treated as a required runtime/security dependency rather than an ambient system tool. The harness vendors the source as `third_party/landrun`, builds `third_party/bin/landrun` during `./kb setup`, verifies that the built binary identifies as Landrun, and has launchers use that repo-local binary by default. `LANDRUN=/path/to/landrun` is an explicit override for testing another binary.

After `./kb setup`, the chosen interpreter is recorded in repo-root `./.kb-python`, and the shell launchers reuse that exact Python instead of blindly preferring `./.venv`.

At the moment, the effective Python floor and ceiling come from vendored KernelBench, whose current `pyproject.toml` pins `requires-python = "==3.10.*"`.

- Codex uses `CODEX_HOME=state/config/codex/`
- Claude uses `CLAUDE_CONFIG_DIR=state/config/claude/`

Those shared tool dirs are where the harness writes:

- generated Codex `config.toml`
- generated Claude `settings.json`
- generated per-run Claude command-MCP registration for the launcher-owned broker
- Claude's own shell sandbox stays disabled on this cluster-oriented setup, but the Bash tool itself is not exposed to the solver; Landrun and the command broker are the active local guardrails
- generated helper-agent definitions for both tools
- tool-managed local state such as auth/session/history files

The harness mirrors shared auth only from repo-root tool dirs:

- `./.codex/auth.json` -> `state/config/codex/auth.json`
- `./.claude/.credentials.json` -> `state/config/claude/.credentials.json`

That keeps `state/config/` disposable while leaving repo-root login state under user control. The harness intentionally does not read `~/.codex` or `~/.claude`.

This split is deliberate:

- the workspace should contain only problem files the solver is meant to read or edit
- tool auth/config should not sit inside the solver-visible workspace
- traces are **not** recovered from shared tool history files; each problem captures its own streamed `agent/events.jsonl` directly from the launcher and its own `agent/activity_ir_events.jsonl` from the command broker
- `agent/events.jsonl` is the exact client stream you watch live; `agent/trace_ir.json` is the normalized merged view used for counts, audit, and summaries

## Codex vs Claude local-surface split

The two tool runtimes expose their native controls differently.

### Codex

Codex keeps its shared user/runtime config under `CODEX_HOME`. The harness generates:

- `state/config/codex/config.toml`
- `state/config/codex/agents/*.toml`

Codex launches from the prepared problem workspace under Landrun. The shared `state/config/codex/config.toml` disables the shell tool and registers only the tiny command MCP server for privileged harness actions.

### Claude

Claude keeps its shared user/runtime config under `CLAUDE_CONFIG_DIR`. The harness generates:

- `state/config/claude/settings.json`
- `state/config/claude/.claude.json`
- `state/config/claude/agents/*.md`

Claude also launches from the prepared problem workspace under Landrun. The shared Claude config carries web/search and helper-agent policy, while the launcher writes a per-run `command-mcp.json` that exposes only the command broker tools (`run_candidate`, `profile_ncu`, `goal_status`, `best_result`, `complete_problem`) for the current problem socket.

The practical result is the same for both tools **with respect to the actual problem environment**:

- no tool auth/config files in the workspace
- the real workspace is directly visible but intentionally small and self-contained
- Landrun mounts the workspace read-only except for `candidate_model_new.py`
- hosted web access stays native to each client and is restricted separately from local actions
- shared web-search policy and helper-agent definitions
- `goal_status` is the live structured status query (remaining budget, attempt counts, best sample, baseline progress)
- `best_result` is the narrow structured query for the current best measured correct attempt

The enforcement mechanism differs:

- **Codex** gets native file/edit tools in the prepared workspace, with the shell tool disabled and privileged actions exposed only through the command MCP server.
- **Claude** gets native file tools in the prepared workspace, with Bash not exposed and privileged actions exposed only through the command MCP server.

Implementation note: the legacy file-access MCP server under `src/kernel_bench_experiment_agents/mcp/` remains for compatibility and tests, but the active launcher path does not use it for workspace reads/writes. The active command MCP server is intentionally tiny and forwards only to the Unix-socket broker.

## Archive contents

### `archive_manifest.json`

Machine-readable description of the problem archive. It explains what is canonical, what is mirrored into the live workspace, and what should be copied out.

### `contract/`

The exact problem contract shown to the solver, plus frozen structured metadata.

Important files:

- `AGENTS.md` — solver instructions and boundaries
- `SPEC.md` — problem goal and success criteria
- `HARDWARE.md` — human-readable hardware facts and guidance
- `INITIAL_PROMPT.md` — the exact initial launch prompt
- `problem_reference.py` — local copy of the reference PyTorch problem code
- `candidate_model_new.py` — initial solver scaffold shown at workspace creation
- `candidate_final.py` — final captured candidate when completion exists
- `problem.json` — machine-readable problem metadata, embedded eager/compile baseline runtimes, budget, and run identifiers
- `hardware.json` — machine-readable hardware facts
- `workspace_contract.json` — machine-readable solver contract
- `provenance.json` — archive-only outward provenance for the source KernelBench checkout and timing files
- `helper_agents/` — archived copies of the generated Codex and Claude helper-agent definitions

### `agent/`

The agent-side run record.

Important files:

- `events.jsonl` — raw streamed output from the agent CLI for this problem only
- `activity_ir_events.jsonl` — synthetic wrapper/direct-command events recorded by the command broker for this problem only
- `mcp_ir_events.jsonl` — legacy synthetic MCP tool events when present
- `trace_ir.json` — normalized trace representation used by the harness
- `completion.json` — terminal state plus measured outcome, trace-derived metadata, and `kernelbench_hacked_kernel_attempt_warnings`
- `goal_status.json` — archived final goal-status snapshot for where the solver ended
- `final_message.txt` — the last assistant-visible model message, when one is available

### `attempts/`

Measured evaluation results for submitted candidates.

Important files:

- `sample_<id>.json` — one measured attempt record with correctness/runtime/build outcome and archive-relative file references
- `sample_<id>.stdout.txt` — evaluation subprocess stdout for that attempt
- `sample_<id>.stderr.txt` — evaluation subprocess stderr for that attempt
- `kernels/level_<level>_problem_<problem_id>_sample_<id>_kernel.py` — archived candidate source measured for that attempt

### `profiles/`

Nsight Compute profiling artifacts.

Important files:

- `profile_<id>.json` — one profile manifest with attempt linkage and archive-relative file references
- `profile_<id>.summary.txt` — short solver-facing summary of the selected metrics
- `profile_<id>.details.txt` — fuller textual profiler output
- `profile_<id>.stdout.txt` — profiler command stdout
- `profile_<id>.stderr.txt` — profiler command stderr

The harness keeps text-first profiler outputs. It does not archive `.ncu-rep` files.

## Workspace surface

Each solver workspace is fresh, self-contained, and intentionally small.

Solver-visible problem files and history mirrors:

- direct read surface:
  - `AGENTS.md`
  - `INITIAL_PROMPT.md`
  - `SPEC.md`
  - `HARDWARE.md`
  - `GOAL_STATUS.md`
  - `problem_reference.py`
  - `candidate_model_new.py`
  - `workspace_contract.json`
  - `problem.json`
  - `hardware.json`
- history directories:
  - `samples/`
  - `profiles/`
- action wrappers:
  - `bin/run_candidate.sh`
  - `bin/profile_ncu.sh`
  - `bin/goal_status.sh`
  - `bin/best_result.sh`
  - `bin/complete_problem.sh`

Workspace-local `samples/` and `profiles/` are mirrors for solver convenience. They are not the durable source of truth.

Notably absent from the workspace:

- Codex `auth.json`
- Codex `config.toml`
- Claude `settings.json`
- Claude `.claude.json`
- Claude credential files
- tool-private helper-agent config

Those live under `state/config/` instead.

## Local Tool Surface

The solver is supposed to work directly in the prepared workspace, not through ad hoc repository or system workflows.

The solver-visible direct workspace surface is:

- read the generated workspace docs, problem reference, candidate, `samples/`, and `profiles/`
- edit only `candidate_model_new.py`
- run privileged actions only through direct command tools

Semantics:

- direct file reads are limited by workspace policy and Landrun to the self-contained problem workspace
- Landrun mounts the workspace read-only and grants write access only to `candidate_model_new.py`
- `candidate_model_new.py` is the only supported local edit path; validation failures return the exact rejected construct and the run does not count toward progress
- `run_candidate` is the only supported measured-evaluation path
- `profile_ncu` is the only supported profiling path
- `research_nvidia_docs` is the only broker-owned docs-research path and is restricted to `docs.nvidia.com`
- `goal_status` is **not** just a cached file read: it refreshes `GOAL_STATUS.md` under the artifact lock and returns the live structured snapshot, including remaining budget, baseline status, current best-run summary, and the latest discarded-attempt reason when present
- `best_result` returns the best measured correct attempt manifest so far, including at least the `sample_id`, the measured result payload, and archive-relative artifact paths such as the archived kernel snapshot
- `complete_problem` is the only valid solver termination path

In other words, the solver surface is intentionally split into:

- **native web**: tool-specific hosted search/fetch, restricted to `docs.nvidia.com`
- **brokered docs research**: normalized `research_nvidia_docs` calls through the command broker, also restricted to `docs.nvidia.com`
- **direct workspace files**: the small fixed problem contract, candidate, and bounded history mirrors
- **brokered actions**: measured run, profiling, NVIDIA-docs research, live status refresh, best-result query, and completion

## Completion model

Solver-written completion is intentionally narrow:

- the only solver-visible exit path is `complete_problem(summary=...)`
- the solver does not choose between `done`, `budget_exhausted`, or other launcher-owned outcomes
- the harness records solver completion as `done` and infers the measured outcome from archived attempts

Launcher-owned terminal states are:

- `budget_exhausted`
- `failed_to_generate`

The harness computes the measured outcome from archived attempts:

- `beats_both`
- `beats_eager_only`
- `beats_compile_only`
- `beats_neither`

Suspicious or otherwise non-counting attempts still remain in `attempts/` for audit, but they do not count toward progress, best-result selection, or summary beat-rate fields.

## Canonical JSON vs rendered Markdown

Where both JSON and Markdown exist, the JSON form is the canonical structured source and the Markdown form is a rendered solver-facing view.

Examples:

- `workspace_contract.json` renders workspace `AGENTS.md` and `INITIAL_PROMPT.md`
- `hardware.json` renders `HARDWARE.md`
- `goal_status.json` renders `GOAL_STATUS.md`

This is intentional, not independent duplicated documentation.
