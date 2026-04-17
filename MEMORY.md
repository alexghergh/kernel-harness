# Memory

Short rolling maintainer handoff for the KernelBench harness.

## Current stable baseline

- MCP-backed problem access is the intended default for both Codex and Claude.
- The solver-visible surface is intentionally narrow:
  - fixed read-only resources: `AGENTS.md`, `INITIAL_PROMPT.md`, `SPEC.md`, `HARDWARE.md`, `GOAL_STATUS.md`, `problem_reference.py`, `candidate_model_new.py`
  - bounded history reads through `list_workspace_dir(samples|profiles)` and `read_workspace_file(...)`
  - actions through `write_candidate`, `run_candidate`, `profile_ncu`, `goal_status`, `best_result`, `complete_problem`
- Shared tool config lives under `state/config/` and should stay disposable.
- Repo-root auth is the only credential source the harness should mirror:
  - `./.codex/auth.json`
  - `./.claude/.credentials.json`
- Codex should use one shared `state/config/codex/config.toml` with MCP `env_vars`; do not reintroduce per-problem `CODEX_HOME` unless a concrete client bug forces it.

## Fixed in the current pass

- strengthened the planner/manager wording so the main agent is told WHEN to use `runner`, WHEN to use `profiler`, WHEN to inspect old `samples/` / `profiles/`, and WHEN to use hosted NVIDIA docs
- suspicious or validation-blocked `run_candidate` results now carry explicit `counts_toward_progress=false` plus a plain-English `progress_blocked_reason`
- suspicious/blocked runs no longer refresh GOAL_STATUS automatically and no longer count as progress when goal status or best-result helpers are recomputed
- the Slurm wrapper no longer redefines the full variable set that `run_agent_range.sh` already owns
- README/ARCHITECTURE now explain where the solver policy lives and that `agent/events.jsonl` is the exact live client trace while `trace_ir.json` is the normalized merged view

## Still on the radar

- verify that real live runs now actually spawn `runner` / `profiler` instead of keeping everything in the main context
- review whether `best_result` is still worth keeping as a separate MCP tool once the graph/history pass lands
- trace IR ordering across raw client events and the synthetic MCP sidecar is still approximate rather than perfectly chronological
- destructive workspace preparation ordering and process-group kill semantics are still known cleanup targets

## Review reminders

- after live runs, inspect `archive/.../agent/events.jsonl`, `mcp_ir_events.jsonl`, and `trace_ir.json` together
- manually inspect final kernels for forbidden escapes (vendor libraries, ATen compute helpers, Triton, dynamic loader monkeypatches, etc.)
- if Codex or Claude behavior drifts, prefer narrowing the advertised surface and sharpening the docs before adding more enforcement layers

## Current behavior review notes

- helper agents are correctly loaded for both clients and have narrow MCP scopes, but the old wording was too weak; this pass makes helper use a default behavior rather than an optional suggestion
- suspicious KernelBench warnings and static validation failures should now feed back to the model as “not counted toward progress” instead of silently letting the model think the run advanced the goal
- the exact raw model trace is `agent/events.jsonl`; use that first when you want to understand what the model actually did, and use `trace_ir.json` mainly for counts, audit, and compact summaries
- next likely maintainer step after another live run: rebase onto `main`, then move `v3` once the current head is verified in-cluster
