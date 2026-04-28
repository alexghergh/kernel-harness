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
- Keep the `v4` tag moved to the current reviewed base before handing back an archive.

## Current pass notes

- Codex MCP failures showed `user cancelled MCP tool call` on `write_candidate`, after the model had already emitted the MCP call.
- Codex MCP approval config belongs under `[mcp_servers.kernelbench]`, not `[apps.kernelbench]`; keep server-level and per-tool `approval_mode = "approve"` so headless `codex exec` can call both read-only and destructive harness tools.
- Keep Codex in `workspace-write` because it runs from an empty scratch cwd and accesses the actual problem workspace only through MCP.
- The previous small cleanup removed the unused MCP smoke script, stale helper functions, and solver-facing wrapper-command advertising.

## Active TODOs

- rerun traces after this Codex MCP approval fix
- then ask for the full review pass
- sandbox integration replacing most MCP
- later optional PTX exposure
- later second workload surface like `hpc-code/`

## Still on the radar

- verify that real live runs now actually spawn `runner` / `profiler` instead of keeping everything in the main context
- review whether `best_result` is still worth keeping as a separate MCP tool once the graph/history pass lands
- trace IR ordering across raw client events and the synthetic MCP sidecar is still approximate rather than perfectly chronological
- destructive workspace preparation ordering and process-group kill semantics are still known cleanup targets

## Review reminders

- after live runs, inspect `archive/.../agent/events.jsonl`, `mcp_ir_events.jsonl`, and `trace_ir.json` together
- manually inspect final kernels for forbidden escapes (vendor libraries, ATen compute helpers, Triton, dynamic loader monkeypatches, etc.)
- if Codex or Claude behavior drifts, prefer narrowing the advertised surface and sharpening the docs before adding more enforcement layers
