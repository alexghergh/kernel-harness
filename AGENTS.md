# KernelBench Experiment Agents

## purpose

This repository exists to run autonomous KernelBench optimization experiments with Codex on a remote GPU node.

## always read first

Before making changes or launching work, read:

- `SPEC.md`
- `ARCHITECTURE.md`

Problem-solving Codex sessions are expected to run from generated external workspaces with their own local `AGENTS.md`. This repository-root file is for maintainers of the harness.

## core rules

- stay scoped to exactly one assigned KernelBench problem
- allow at most one active solver per `(run_name, level, problem_id)`
- do not inspect unrelated problems unless explicitly asked
- do not explore unrelated repositories or random files
- preserve official KernelBench run naming for evaluated candidates
- treat eager PyTorch and `torch.compile` as separate baselines
- prefer local tool commands over ad hoc shell logic when both exist
- do not write to the root `MEMORY.md`; that file is for maintainers of the harness
- keep the solver contract solution-only: `problem_reference.py` is read-only, `candidate_model_new.py` is the evaluated file
- keep the candidate scaffold fixed and allow edits only inside its marked editable blocks
- treat vendor-library wrappers and ATen compute helpers as forbidden for this experiment: no cuBLAS, cuBLASLt, CUTLASS, or ATen native/BLAS helper shortcuts

## tool policy

Use local commands for:

- timing and correctness evaluation
- `ncu` profiling
- workspace preparation

Do not invent alternate benchmarking flows when the local tool already covers the need.
Do not bypass the workspace-local wrappers for solver runs.
Do not relax candidate validation to admit pure-PyTorch shortcuts; the target is custom CUDA/C++ kernels.
Do not relax trace auditing to allow shell work outside the problem workspace.

## web policy

Web access is documentation-only.

Allowed live web domains:

- `docs.nvidia.com`

Do not browse arbitrary sites for kernel solutions.
Do not inspect generated PTX, cubins, Triton output, Inductor output, or compiler-emitted kernels as solution hints.

## codex behavior

- the main agent should behave as a planner and optimizer
- when timing or profiling output would pollute context, spawn the dedicated subagent and ask it to summarize
- keep summaries concise and actionable
- do not let subagents expand the task scope

## file discipline

- write scratch work inside the assigned problem workspace
- only evaluated kernels consume an official `sample_id`
- per-problem artifact locks serialize `sample_id` allocation and manifest commits
- keep project-local manifests under this repository, not inside KernelBench internals

## implementation guidance

- prefer the smallest correct change that advances the experiment
- avoid heavy frameworks
- keep helper commands explicit and composable
- when the official KernelBench interface is unclear, prefer a thin wrapper and document the assumption
