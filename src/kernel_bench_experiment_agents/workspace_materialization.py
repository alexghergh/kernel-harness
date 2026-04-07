from __future__ import annotations

from pathlib import Path
from typing import Any

from .candidate_contract import CANDIDATE_FILENAME, candidate_template
from .hardware_catalog import render_hardware_markdown
from .project import now_iso, write_json, write_text
from .workspace_contract import (
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
    problem: Any,
    hardware: Any,
    num_gpus: int,
    model: str,
    time_budget_minutes: int,
) -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "run_name": run_name,
        "level": level,
        "problem_id": problem_id,
        "tool": tool,
        "dataset_src": dataset_src,
        "problem_name": problem.name,
        "gpu_name": hardware.display_name,
        "gpu_architecture": hardware.architecture,
        "gpu_compute_capability": hardware.compute_capability,
        "num_gpus": num_gpus,
        "model": model,
        "time_budget_minutes": time_budget_minutes,
    }


def build_archive_provenance(
    *,
    kernelbench_root_path: str,
    kernelbench_python: str,
    problem: Any,
    eager_baseline_file: str,
    compile_baseline_file: str,
) -> dict[str, Any]:
    return {
        "kernelbench_root": kernelbench_root_path,
        "kernelbench_python": kernelbench_python,
        "problem_source_path": getattr(problem, "path", None),
        "eager_baseline_file": eager_baseline_file,
        "compile_baseline_file": compile_baseline_file,
    }


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
    baseline: dict[str, Any],
    hardware_payload: dict[str, Any],
    problem_code: str,
) -> dict[str, Any]:
    contract = build_workspace_contract(metadata=metadata)
    write_json(target_dir / "problem.json", metadata)
    write_json(target_dir / "baseline.json", baseline)
    write_json(target_dir / "hardware.json", hardware_payload)
    write_json(target_dir / "workspace_contract.json", contract)
    write_text(target_dir / "problem_reference.py", problem_code)
    write_text(target_dir / CANDIDATE_FILENAME, candidate_template())
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
        ),
    )
    write_text(target_dir / "AGENTS.md", render_workspace_agents_md(contract=contract))
    write_text(
        target_dir / "INITIAL_PROMPT.md",
        render_initial_prompt(contract=contract, baseline=baseline),
    )
    return contract
