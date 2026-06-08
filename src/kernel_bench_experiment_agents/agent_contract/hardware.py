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
            "## General",
            "Identify whether your kernel is compute-bound or memory-bound BEFORE picking techniques. Estimate arithmetic intensity (FLOPs per byte transferred from HBM): H100's roofline knee is ~13 FLOP/byte (989 TFLOPS FP16-TC / ~3.35 TB/s HBM3). Kernels well above 13 FLOP/byte (large GEMM, convolution on big inputs) are compute-bound — see the Compute-bound section. Kernels well below 13 FLOP/byte (elementwise, reductions, normalization, pooling, attention scores) are memory-bound — see the Memory-bound section. Reach for `profile_ncu` if the answer isn't obvious from the problem definition.",
            "Hopper makes larger shared-memory tiles viable, but register pressure can still collapse occupancy quickly.",
            "Use this hardware budget when choosing tile sizes, stage counts, warp layouts, and accumulator footprints.",
            "Tensor-core and TF32 paths are legitimate if they are implemented inside your custom CUDA kernel and still pass correctness.",
            "## Compute-bound kernels (matmul, convolution)",
            "Goal: drive sustained tensor-core throughput close to peak. Under `profile_ncu` the metric to track is `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` (target ≥ 90%). FLOP rate (`sm__inst_executed_pipe_tensor`) without correlating to peak utilization is misleading.",
            "Beyond Ampere-era `cp.async` / `ldmatrix` / `mma.sync.m16n8k16`, Hopper exposes several intrinsics that often dominate top-tier matmul performance. Consider them before settling on an Ampere-style design.",
            "TMA (Tensor Memory Accelerator): host-side `cuTensorMapEncodeTiled` plus device-side `cp.async.bulk.tensor.{2d,3d}.shared::cluster.global` with `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group`. Async bulk global→shared copies with hardware-managed addressing; faster and lower overhead than `cp.async`. PTX syntax reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-access-instructions",
            "wgmma (warp-group MMA): `wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16` family. One warp-group (4 warps) issues one mma covering a 64xN tile; higher tensor-core throughput than `mma.sync.aligned.m16n8k16`. Requires arch `sm_90a` (not `sm_90`). PTX syntax, 64-bit shared-memory descriptor layout, and `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group` sequencing: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions",
            "Thread-block clusters + DSMEM: launch with `__cluster_dims__(x,y,z)`. Blocks within a cluster can read/write each other's shared memory through the distributed-shared address space, enabling larger effective tiles without proportionally more SMEM per block. CUDA C++ Programming Guide reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters",
            "`mbarrier` PTX (`mbarrier.init.shared::cta`, `mbarrier.arrive`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`) is the canonical way to overlap TMA loads with `wgmma` compute in a producer/consumer pipeline. PTX reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier",
            "Stream-K / persistent-block patterns launch one block per SM and loop over an iteration space of output tiles. Avoids wave quantization and improves tail efficiency on shapes that don't tile cleanly.",
            "Performance ladder for H100 matmul. Each rung is roughly where typical kernels plateau before the next technique unlocks more peak. Rungs are largely independent — once the current kernel is correct, profile and pick the rung with the largest expected jump for this shape:",
            "  (1) **Ampere-style baseline** (`cp.async` + `mma.sync.m16n8k16` + `ldmatrix`, software-pipelined). Tops out around 35–45% of H100 dense FP16/FP32-accum peak on large GEMMs — Hopper executes the Ampere tensor-core path but doesn't exploit warp-group MMA.",
            "  (2) **`wgmma.mma_async` + `cp.async` pipeline**. Switching from `mma.sync` to warp-group MMA typically reaches ~55–60% of peak. Requires `-arch=sm_90a`.",
            "  (3) **TMA** in place of `cp.async` for global→shared loads (host-side `cuTensorMapEncodeTiled` + device-side `cp.async.bulk.tensor`). Frees registers and removes the per-thread address-calculation overhead that `cp.async` carries.",
            "  (4) **`mbarrier`-coordinated warp specialization** — dedicate a producer warp-group to TMA loads and consumer warp-groups to `wgmma`, so memory and compute fully overlap. This is the lever that takes a `wgmma`+`cp.async` kernel from ~55–60% to ~70%+ on most shapes.",
            "  (5) **Thread-block clusters + DSMEM** (`__cluster_dims__`, `cluster.cta_id`, distributed-shared addressing). Blocks in a cluster share SMEM, enabling larger effective tiles without proportionally more per-block SMEM; valuable on shapes where the per-block tile is SMEM-bound rather than compute-bound.",
            "  (6) **Stream-K / persistent-block** schedules — launch one block per SM and have it loop over output tiles. Reclaims the tail efficiency lost to wave quantization on shapes that don't tile cleanly into a multiple of SM count.",
            "Default build flag for the above is `-arch=sm_90a` (the `a` variant). Without `a`, `wgmma` and several TMA forms are unavailable. Verify on every run via the `build_command` field of `run_candidate` rather than assuming.",
            "Concrete sample code for TMA setup + wgmma kernels (host-side `cuTensorMapEncodeTiled`, descriptor construction, warp-specialized mainloop): https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ and https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/",
            "Convolution-specific: prefer implicit-GEMM (reinterpret the conv as a virtual GEMM over im2col-shifted tiles, with no materialized im2col buffer) over explicit im2col when shapes allow — same tensor-core path as matmul. Use channels-last (NHWC) so the channel dim becomes the inner GEMM dimension. Winograd F(2,3) / F(4,3) is only worthwhile for 3x3 stride-1 convolutions where the FLOP reduction outweighs the bandwidth cost of transformed tiles.",
            "## Memory-bound kernels (elementwise, normalization, reductions, pooling)",
            "Goal: drive sustained HBM throughput close to peak. Under `profile_ncu` the metric to track is `dram__bytes.sum.per_second` against `dram__bytes.sum.peak_sustained` (target ≥ 80%). FLOP rate and tensor-core utilization are NOT useful targets here — the bottleneck is memory.",
            "Vectorize loads/stores: use `float4` / `__half2` / `ulonglong2` casts so each thread issues an `LDG.128` / `STG.128` (16 bytes per memory transaction). A single thread should move 16 bytes per memory instruction; otherwise the SM stalls on memory-issue rate, not bandwidth.",
            "Ensure coalesced access: the 32 threads of a warp must address 32 contiguous 4-byte words (or 32 contiguous 16-byte words after vectorization). Verify via `smsp__inst_executed_op_global_ld.sum.per_inst` ≈ 1 transaction per warp instruction. Strided / gather access patterns lose 4–8× throughput; reshape the data layout if possible.",
            "Warp-shuffle reductions are the canonical primitive for any cross-thread reduction below block scope: `__shfl_xor_sync(0xffffffff, val, mask)` for tree reductions, `__shfl_down_sync` for sweep reductions. Avoid shared-memory reductions when warp shuffles suffice — they're faster and use no SMEM.",
            "Block-scope reductions: warp-reduce first (32 → 1 per warp), write one value per warp to SMEM, then warp-reduce again on the first warp. `cooperative_groups::reduce` from `<cooperative_groups/reduce.h>` exposes this pattern cleanly. Avoid `__syncthreads()` inside the inner loop.",
            "Grid-stride / persistent-grid loops on small problems: launch `gridDim = #SMs` blocks, each block loops over `tid + n * gridDim.x * blockDim.x` elements. Eliminates launch-overhead amortization on tiny tensors and keeps the SMs warm across iterations.",
            "Fuse epilogues into the producer kernel: a separate `add_bias` or `cast_to_half` pass after a reduction kernel doubles the HBM traffic. Inline these into the same kernel even if they look unrelated.",
            "Do NOT reach for `wgmma`, TMA, or thread-block clusters on a pure memory-bound kernel — they target compute-throughput and SMEM-staging problems that you do not have. The bottleneck is HBM, not the tensor cores.",
            "Normalization (BatchNorm, LayerNorm, GroupNorm, RMSNorm): parallelize over the normalized-channel axis in `blockIdx`, reduce over the within-channel axis with the warp+block reduction pattern above, then broadcast `mean` / `var` back through the normalize step in the same kernel. Two-pass (`mean → var → normalize`) is only needed when numerical stability dictates Welford's algorithm.",
            "Pooling: keep the reduction inside one block when the window fits in shared memory. For large windows, use a two-pass kernel rather than spilling intermediates to global memory.",
            "## Specialized patterns",
            "Softmax: use the online (numerically stable) formulation — track running max and running sum-of-exponentials in a single pass. Reference: Milakov & Gimelshein, 'Online Normalizer Calculation for Softmax' (arXiv:1805.02867). Combines naturally with attention.",
            "Attention (scaled dot-product): implement the flash-attention pattern — tile Q over rows, tile K/V over columns, fuse the online softmax inside the K-tile loop, never materialize the full N×N attention matrix in HBM. Reference: Dao et al., 'FlashAttention' (arXiv:2205.14135). This converts attention from memory-bound to compute-bound on long sequences.",
            "Scans (cumsum, cumprod, masked scans): warp-level scan via `__shfl_up_sync` (5 shuffles for a warp of 32), then block-level scan over per-warp sums via shared memory, then grid-level via persistent kernels or a second propagation pass. Single-launch implementation: Merrill & Garland, 'Single-pass Parallel Prefix Scan with Decoupled Look-back'.",
            "Loss functions (MSE, CrossEntropy, Huber, KLDiv, TripletMargin): implement as fused reduction kernels. CrossEntropy in particular needs softmax + log + gather + sum in one pass — splitting into separate kernels doubles HBM traffic.",
            "## Workflow",
            "Large micro-searches are allowed here; dozens or hundreds of timing runs are normal when tuning tile sizes or stage counts.",
            "Use `profile_ncu` periodically during that search so the next branch is informed by measured bottlenecks.",
            "If you are unsure about a hardware limit, consult the official NVIDIA docs below rather than guessing.",
            "WebFetch caveat: the full `parallel-thread-execution/index.html` is multi-MB and gets truncated when fetched whole. Always fetch the anchor URL for the section you need (the URLs above include the correct anchors). The developer-blog posts and arXiv abstracts referenced above are normal-length HTML pages with concrete code or pseudocode and render cleanly.",
        ),
        doc_urls=(
            "https://docs.nvidia.com/cuda/hopper-tuning-guide/",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-access-instructions",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier",
            "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters",
            "https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
            "https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/",
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
            "Large micro-searches are allowed here; dozens or hundreds of timing runs are normal when tuning tile sizes or stage counts.",
            "Use `profile_ncu` periodically during that search so the next branch is informed by measured bottlenecks.",
            "If you are unsure about a hardware limit, consult the official NVIDIA docs below rather than guessing.",
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
            "Large micro-searches are allowed here; dozens or hundreds of timing runs are normal when tuning tile sizes or stage counts.",
            "Use `profile_ncu` periodically during that search so the next branch is informed by measured bottlenecks.",
            "If you are unsure about a hardware limit, consult the official NVIDIA docs below rather than guessing.",
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
            "Large micro-searches are allowed here; dozens or hundreds of timing runs are normal when tuning tile sizes or stage counts.",
            "Use `profile_ncu` periodically during that search so the next branch is informed by measured bottlenecks.",
            "If you are unsure about a hardware limit, consult the official NVIDIA docs below rather than guessing.",
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


def _render_guidance(guidance: tuple[str, ...]) -> str:
    """Render the guidance tuple, treating ``## `` prefixes as H3 subheadings.

    Any guidance string starting with ``## `` is rendered as an H3 heading
    (preceded by a blank line); all other strings render as ``- `` bullets.
    Specs that do not use section markers render as a flat bullet list.
    """
    parts: list[str] = []
    for line in guidance:
        if line.startswith("## "):
            parts.append("")
            parts.append(f"### {line[3:]}")
            parts.append("")
        else:
            parts.append(f"- {line}")
    return "\n".join(parts).strip("\n")


def render_hardware_markdown(spec: HardwareSpec) -> str:
    carveout_values = ", ".join(str(value) for value in spec.shared_memory_carveout_kb)
    guidance_block = _render_guidance(spec.guidance)
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
        "## How to use this\n\n"
        f"{guidance_block}\n\n"
        "## Official NVIDIA references\n\n"
        f"{doc_lines}\n"
    )
