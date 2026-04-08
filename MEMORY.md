# Memory

Short rolling maintainer handoff for the KernelBench harness.

## Current direction

- keep the project small, explicit, and operator-friendly
- keep `README.md` high-level and operational
- keep `ARCHITECTURE.md` as the detailed system contract
- keep the live solver workspace self-contained and free of outward path leakage
- keep one canonical policy source for tool/runtime restrictions, rendered into Codex and Claude configs
- keep `archive/` as the only durable copy-out root; keep `state/` disposable
- prefer simple archive artifacts over append-only ledgers when one manifest per attempt/profile is enough
- keep `summarize-run` for now as the archive-only aggregate view

## Locked decisions

- no root `SPEC.md`
- no ADR directory for this project; stable decisions belong in `ARCHITECTURE.md`
- the harness, not the solver, owns measured outcomes
- solver terminal states stay narrow: `done`, `harness_failure`
- root maintainer docs and workspace solver docs are different audiences even when filenames match
- the active KernelBench environment is the source of truth; this repo should install into that environment rather than managing a separate Python path

## Important reminders

- after a real run, remind the user to manually inspect the produced kernels for invalid shortcuts such as PyTorch calls, Triton, CUTLASS, or similar benchmark-invalid escapes
- do not try to solve that purely with harness logic; later either tighten the sandbox or add narrowly targeted audits
- sandbox validation, billing/timeout trace handling, and real cluster trace review still need a follow-up pass after live runs
- ask for live Codex and Claude traces after the next cluster runs so trace parsing and failure handling can be hardened against real payloads

## Next steps

- validate sandbox behavior on real Codex and Claude runs after the current cleanup lands
- review live normal and failure traces from both tools to harden trace parsing and timeout/billing classification
- sanity-check end-to-end NCU output on the cluster and revisit region-of-interest profiling later
- keep reminding the user to manually inspect produced kernels after real runs
