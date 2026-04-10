"""Store the frozen hardware facts and markdown rendering used in each workspace.

Workspace preparation resolves one named hardware spec here and propagates it into docs, metadata, and prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

CUDA_PROGRAMMING_GUIDE_URL = (
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
    "#features-and-technical-specifications"
)


@dataclass(frozen=True)
class HardwareSpec:
    display_name: str
    architecture: str
    compute_capability: str
    registers_per_sm: str
    max_registers_per_thread: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    shared_memory_per_sm_kb: int
    max_shared_memory_per_block_kb: int
    shared_memory_carveout_kb: tuple[int, ...]
    guidance: tuple[str, ...]
    doc_urls: tuple[str, ...]
    aliases: tuple[str, ...]


def _normalize_gpu_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_CATALOG = (
    HardwareSpec(
        display_name="H100",
        architecture="Hopper",
        compute_capability="9.0",
        registers_per_sm="64K 32-bit registers",
        max_registers_per_thread=255,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        shared_memory_per_sm_kb=228,
        max_shared_memory_per_block_kb=227,
        shared_memory_carveout_kb=(0, 8, 16, 32, 64, 100, 132, 164, 196, 228),
        guidance=(
            "Hopper makes larger shared-memory tiles viable, but register pressure can still collapse occupancy quickly.",
            "Use this hardware budget when choosing tile sizes, stage counts, warp layouts, and accumulator footprints.",
            "Tensor-core and TF32 paths are legitimate if they are implemented inside your custom CUDA kernel and still pass correctness.",
        ),
        doc_urls=(
            "https://docs.nvidia.com/cuda/hopper-tuning-guide/",
            CUDA_PROGRAMMING_GUIDE_URL,
        ),
        aliases=(
            "h100",
            "h100nvl",
            "h100pcie",
            "h100sxm",
            "gh200",
            "h200",
            "nvidiah100",
            "nvidiah100nvl",
            "nvidiah100pcie",
            "nvidiagh200",
            "nvidiah200",
        ),
    ),
    HardwareSpec(
        display_name="A100",
        architecture="Ampere",
        compute_capability="8.0",
        registers_per_sm="64K 32-bit registers",
        max_registers_per_thread=255,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        shared_memory_per_sm_kb=164,
        max_shared_memory_per_block_kb=163,
        shared_memory_carveout_kb=(0, 8, 16, 32, 64, 100, 132, 164),
        guidance=(
            "Ampere exposes TF32 tensor-core math and async global-to-shared copies; both are relevant search directions inside custom CUDA code.",
            "A100 has less shared memory per block than Hopper, so search should balance tile size against occupancy earlier.",
            "Register pressure and shared-memory staging usually trade off directly on Ampere matmul kernels.",
        ),
        doc_urls=(
            "https://docs.nvidia.com/cuda/ampere-tuning-guide/contents.html",
            CUDA_PROGRAMMING_GUIDE_URL,
        ),
        aliases=(
            "a100",
            "a10080gb",
            "a10040gb",
            "a100pcie",
            "a100sxm",
            "nvidiaa100",
            "nvidiaa10080gb",
            "nvidiaa10040gb",
        ),
    ),
    HardwareSpec(
        display_name="L40S",
        architecture="Ada",
        compute_capability="8.9",
        registers_per_sm="64K 32-bit registers",
        max_registers_per_thread=255,
        max_warps_per_sm=48,
        max_blocks_per_sm=24,
        shared_memory_per_sm_kb=100,
        max_shared_memory_per_block_kb=99,
        shared_memory_carveout_kb=(0, 8, 16, 32, 64, 100),
        guidance=(
            "Ada has tighter occupancy and shared-memory ceilings than Hopper or A100, so smaller tiles and shallower staging are often necessary.",
            "Search block sizes, warp layouts, vector widths, and stage counts more aggressively because the feasible region is smaller.",
            "Profile periodically; Ada kernels can become register-limited quickly even when shared memory looks modest.",
        ),
        doc_urls=(
            "https://docs.nvidia.com/cuda/archive/13.1.0/ada-tuning-guide/index.html",
            CUDA_PROGRAMMING_GUIDE_URL,
        ),
        aliases=(
            "l40s",
            "nvidial40s",
            "rtx6000ada",
            "rtx6000adageneration",
            "nvidiartx6000ada",
            "nvidiartx6000adageneration",
        ),
    ),
    HardwareSpec(
        display_name="B200",
        architecture="Blackwell",
        compute_capability="10.0",
        registers_per_sm="64K 32-bit registers",
        max_registers_per_thread=255,
        max_warps_per_sm=64,
        max_blocks_per_sm=32,
        shared_memory_per_sm_kb=228,
        max_shared_memory_per_block_kb=227,
        shared_memory_carveout_kb=(0, 8, 16, 32, 64, 100, 132, 164, 196, 228),
        guidance=(
            "Blackwell keeps Hopper-like shared-memory limits, so Hopper-style tiling ideas are relevant but must still be re-profiled.",
            "Treat the search as hardware-specific: profile and tune stage counts, tile shapes, and memory movement choices rather than assuming Hopper numbers transfer directly.",
            "Use the Blackwell tuning guide and the CUDA programming guide for any uncertain limit instead of guessing.",
        ),
        doc_urls=(
            "https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html",
            CUDA_PROGRAMMING_GUIDE_URL,
        ),
        aliases=("b200", "gb200", "nvidiab200", "nvidiagb200"),
    ),
)


def resolve_hardware_spec(gpu_name: str) -> HardwareSpec:
    normalized = _normalize_gpu_name(gpu_name)
    if not normalized:
        raise ValueError(
            "HARDWARE_NAME is required. Set it to a supported alias such as H100, A100, "
            "L40S, or B200."
        )
    for spec in _CATALOG:
        for alias in spec.aliases:
            if normalized == alias or normalized.startswith(alias):
                return spec
    supported = ", ".join(spec.display_name for spec in _CATALOG)
    raise ValueError(
        f"Unsupported HARDWARE_NAME {gpu_name!r}. Supported GPU families: {supported}."
    )


def render_hardware_markdown(spec: HardwareSpec) -> str:
    carveout_values = ", ".join(str(value) for value in spec.shared_memory_carveout_kb)
    guidance_lines = "\n".join(f"- {line}" for line in spec.guidance)
    doc_lines = "\n".join(f"- {url}" for url in spec.doc_urls)
    return (
        "# Hardware Notes\n\n"
        "This file is part of the solver working set. Re-read it when choosing tile sizes, "
        "block sizes, shared-memory staging, vector widths, register usage, and tensor-core modes.\n\n"
        "Assigned GPU:\n\n"
        f"- GPU family: `{spec.display_name}`\n"
        f"- architecture: `{spec.architecture}`\n"
        f"- compute capability: `{spec.compute_capability}`\n"
        f"- registers per SM: `{spec.registers_per_sm}`\n"
        f"- max registers per thread: `{spec.max_registers_per_thread}`\n"
        f"- max warps per SM: `{spec.max_warps_per_sm}`\n"
        f"- max thread blocks per SM: `{spec.max_blocks_per_sm}`\n"
        f"- shared memory per SM: `{spec.shared_memory_per_sm_kb}` KB\n"
        f"- max shared memory per thread block: `{spec.max_shared_memory_per_block_kb}` KB\n"
        f"- supported shared-memory carveouts per SM: `{carveout_values}` KB\n\n"
        "How to use this:\n\n"
        f"{guidance_lines}\n"
        "- large micro-searches are allowed here; dozens or hundreds of timing runs are normal when tuning tile sizes or stage counts\n"
        "- use `./bin/profile_ncu.sh` periodically during that search so the next branch is informed by measured bottlenecks\n"
        "- if you are unsure about a hardware limit, consult the official NVIDIA docs below rather than guessing\n\n"
        "Official NVIDIA references:\n\n"
        f"{doc_lines}\n"
    )
