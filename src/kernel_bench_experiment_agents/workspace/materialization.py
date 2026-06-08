"""Write the generated workspace files and archived contract bundle for one problem.

Workspace preparation calls into this module after metadata resolution so the on-disk workspace and archive stay aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.agent_contract.hardware import render_hardware_markdown
from kernel_bench_experiment_agents.problem_source import (
    BaselinePayload,
    ProblemSource,
    ReferencePayload,
)
from kernel_bench_experiment_agents.runtime.project import now_iso, write_json, write_text
from kernel_bench_experiment_agents.agent_contract.contract import (
    build_workspace_contract,
    render_initial_prompt,
    render_workspace_agents_md,
    render_workspace_spec_md,
)


class HardwarePayloadView:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.display_name = payload["display_name"]
        self.architecture = payload["architecture"]
        self.compute_capability = payload["compute_capability"]
        self.registers_per_sm = payload["registers_per_sm"]
        self.max_registers_per_thread = payload["max_registers_per_thread"]
        self.max_warps_per_sm = payload["max_warps_per_sm"]
        self.max_blocks_per_sm = payload["max_blocks_per_sm"]
        self.shared_memory_per_sm_kb = payload["shared_memory_per_sm_kb"]
        self.max_shared_memory_per_block_kb = payload["max_shared_memory_per_block_kb"]
        self.shared_memory_carveout_kb = tuple(payload["shared_memory_carveout_kb"])
        self.guidance = tuple(payload["guidance"])
        self.doc_urls = tuple(payload["doc_urls"])


def build_problem_metadata(
    *,
    run_name: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    tool: str,
    problem_name: str,
    problem_source: ProblemSource,
    problem_dir: str | None,
    hardware: Any,
    hardware_name: str,
    num_gpus: int,
    model: str,
    time_budget_minutes: int,
    precision: str,
) -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "tool": tool,
        "dataset_src": dataset_src,
        "problem_source": problem_source.name,
        "problem_dir": problem_dir,
        "problem_name": problem_name,
        "hardware_name": hardware_name,
        "gpu_name": hardware.display_name,
        "gpu_architecture": hardware.architecture,
        "gpu_compute_capability": hardware.compute_capability,
        "num_gpus": num_gpus,
        "model": model,
        "time_budget_minutes": time_budget_minutes,
        "precision": precision,
    }


def build_archive_provenance(
    *,
    problem_source_name: str,
    reference_payload: ReferencePayload,
    kernelbench_root_path: str | None = None,
    timings_dir: str | None = None,
    eager_baseline_file: str | None = None,
    compile_baseline_file: str | None = None,
    problem_dir: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "problem_source": problem_source_name,
        "problem_source_metadata": dict(reference_payload.problem_metadata),
        "baseline_label": reference_payload.baseline.label,
        "baseline_runtime_ms": reference_payload.baseline.runtime_ms,
        "baseline_extras": dict(reference_payload.baseline.extras),
    }
    if problem_dir:
        payload["problem_dir"] = problem_dir
    if kernelbench_root_path:
        payload["kernelbench_root"] = kernelbench_root_path
    if timings_dir:
        payload["timings_dir"] = timings_dir
    if eager_baseline_file:
        payload["eager_baseline_file"] = str(eager_baseline_file)
    if compile_baseline_file:
        payload["compile_baseline_file"] = str(compile_baseline_file)
    return payload


def build_hardware_payload(hardware: Any) -> dict[str, Any]:
    return {
        "display_name": hardware.display_name,
        "architecture": hardware.architecture,
        "compute_capability": hardware.compute_capability,
        "registers_per_sm": hardware.registers_per_sm,
        "max_registers_per_thread": hardware.max_registers_per_thread,
        "max_warps_per_sm": hardware.max_warps_per_sm,
        "max_blocks_per_sm": hardware.max_blocks_per_sm,
        "shared_memory_per_sm_kb": hardware.shared_memory_per_sm_kb,
        "max_shared_memory_per_block_kb": hardware.max_shared_memory_per_block_kb,
        "shared_memory_carveout_kb": list(hardware.shared_memory_carveout_kb),
        "guidance": list(hardware.guidance),
        "doc_urls": list(hardware.doc_urls),
    }


def write_contract_bundle(
    *,
    target_dir: Path,
    metadata: dict[str, Any],
    baseline: BaselinePayload,
    hardware_payload: dict[str, Any],
    problem_source: ProblemSource,
    reference_source: str,
) -> dict[str, Any]:
    """Write the generated workspace files and their archived contract mirror."""
    contract = build_workspace_contract(metadata=metadata, problem_source=problem_source)
    problem_payload = dict(metadata)
    problem_payload["baseline_runtime_ms"] = baseline.runtime_ms
    problem_payload["baseline_label"] = baseline.label
    problem_payload["baseline_extras"] = dict(baseline.extras)
    write_json(target_dir / "problem.json", problem_payload)
    write_json(target_dir / "hardware.json", hardware_payload)
    write_json(target_dir / "workspace_contract.json", contract)
    write_text(target_dir / problem_source.reference_filename, reference_source)
    write_text(target_dir / problem_source.candidate_filename, problem_source.candidate_template())
    write_text(
        target_dir / "HARDWARE.md",
        render_hardware_markdown(HardwarePayloadView(hardware_payload)),
    )
    write_text(
        target_dir / "SPEC.md",
        render_workspace_spec_md(
            problem_name=metadata.get("problem_name"),
            metadata=metadata,
            baseline=baseline,
            hardware_markdown_name="HARDWARE.md",
            problem_source=problem_source,
        ),
    )
    write_text(target_dir / "AGENTS.md", render_workspace_agents_md(contract=contract))
    write_text(
        target_dir / "INITIAL_PROMPT.md",
        render_initial_prompt(contract=contract, baseline=baseline, problem_source=problem_source),
    )
    return contract
