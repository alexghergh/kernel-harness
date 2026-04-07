# Maintainer Instructions

This `AGENTS.md` is for people and coding agents working **on this repository**.
It is **not** the solver contract used inside per-problem workspaces. Those workspaces render their own `AGENTS.md` and `SPEC.md` for the solver agent.

## Read order

1. Read this file.
2. Read `MEMORY.md`.
3. Read any relevant ADRs under `docs/adr/`.
4. Read the code that implements the area you are changing.

## Memory policy

`MEMORY.md` is the short rolling handoff for the next maintainer or agent session.

- Update `MEMORY.md` every turn.
- Keep it short.
- Preserve only current direction, locked decisions, open questions, and next steps.
- Compress older detail instead of letting it grow indefinitely.
- Do not turn it into a changelog.

## Repo rules

- Treat root docs and workspace docs as different audiences, even when filenames match.
- Keep the solver contract explicit and narrow.
- Keep generic harness logic separate from tool-specific adapters.
- Prefer one canonical source over duplicated vendor-specific files.
- Do not hand-edit generated workspace helper-agent specs. Edit `src/kernel_bench_experiment_agents/agent_specs.py`; `prepare-problem-workspace` regenerates `.codex/agents/*` and `.claude/agents/*` inside each workspace.
- Keep `archive/` as the single durable copy-out root.
- Keep `state/` disposable.
- Do not re-introduce root `SPEC.md`.

## Validation before finishing

At minimum, run:

```bash
python -m py_compile src/kernel_bench_experiment_agents/*.py
bash -n scripts/run_agent_problem.sh
bash -n scripts/clear_run.sh
```

If you change helper-agent specs, regenerate a workspace and inspect the generated `.codex/agents/*` and `.claude/agents/*` inside that workspace.

## Current architecture direction

- solver-facing workspaces are self-contained under `state/workspaces/...`
- durable run outputs live under `archive/<run_name>/...`
- launcher and harness own measured outcomes
- solver terminal states are narrow: `done`, `stalled`, `harness_failure`
- Codex and Claude remain equal first-class adapters
