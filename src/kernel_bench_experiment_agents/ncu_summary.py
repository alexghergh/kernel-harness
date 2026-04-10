"""Build a compact solver-facing Nsight Compute summary from the raw CSV export.

The harness profiles with the full NCU set, but this reducer keeps the text promptable by
highlighting the most actionable counters plus a CudaForge-style reference subset.
"""

from __future__ import annotations

import csv
import io

from .common import as_float

# Nsight Compute references for the metric groups below:
# - Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
# - CLI Reference / metric naming: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
#
# The summary intentionally stays much smaller than the raw export, but it now also
# exposes the exact CudaForge-style 24-metric subset when those counters are present
# in the raw CSV. This keeps the agent focused without discarding common judge signals.

CUDAFORGE_METRIC_GROUPS = (
    (
        "CudaForge reference: execution and occupancy",
        (
            ("SM cycles active", "sm__cycles_active.avg"),
            ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
            ("block limit by blocks", "launch__occupancy_limit_blocks"),
            ("block limit by registers", "launch__occupancy_limit_registers"),
            ("block limit by shared memory", "launch__occupancy_limit_shared_mem"),
            ("registers per thread", "launch__registers_per_thread"),
            ("instructions executed", "sm__inst_executed.sum"),
            (
                "FP32 pipe utilization",
                "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
            ),
            (
                "tensor pipe utilization",
                "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
            ),
            (
                "branch target uniformity",
                "smsp__sass_average_branch_targets_threads_uniform.pct",
            ),
        ),
    ),
    (
        "CudaForge reference: memory system",
        (
            ("DRAM bytes read", "dram__bytes_read.sum"),
            ("DRAM bytes written", "dram__bytes_write.sum"),
            ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            ("DRAM bytes per second", "dram__bytes.sum.per_second"),
            ("GPU DRAM throughput", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
            ("L1/TEX hit rate", "l1tex__t_sector_hit_rate.pct"),
            ("L1/TEX throughput", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
            ("L2 hit rate", "lts__t_sector_hit_rate.pct"),
            ("L2 throughput", "lts__throughput.avg.pct_of_peak_sustained_active"),
        ),
    ),
    (
        "CudaForge reference: dominant stalls",
        (
            (
                "stall: memory dependency",
                "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
            ),
            (
                "stall: short scoreboard",
                "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            ),
            (
                "stall: long scoreboard",
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            ),
            (
                "stall: barrier",
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            ),
            (
                "stall: branch resolving",
                "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
            ),
        ),
    ),
)


# The first section stays short and architecture-agnostic so the agent can skim it
# before diving into the fuller CudaForge-aligned subset below.
KEY_METRIC_GROUPS = (
    (
        "Key performance metrics",
        (
            ("duration", "gpu__time_duration.sum"),
            ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            (
                "compute+memory throughput",
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            ),
            ("registers per thread", "launch__registers_per_thread"),
            ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ),
    ),
    (
        "Memory and shared-memory indicators",
        (
            ("L1/TEX throughput", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
            ("L2 throughput", "lts__throughput.avg.pct_of_peak_sustained_active"),
            ("DRAM throughput", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            ("shared-memory conflict n-way", "derived__memory_l1_conflicts_shared_nway"),
            (
                "shared-memory excessive wavefronts",
                "derived__memory_l1_wavefronts_shared_excessive",
            ),
        ),
    ),
    (
        "Occupancy limiters",
        (
            ("block limit by blocks", "launch__occupancy_limit_blocks"),
            ("block limit by registers", "launch__occupancy_limit_registers"),
            ("block limit by shared memory", "launch__occupancy_limit_shared_mem"),
            ("block limit by warps", "launch__occupancy_limit_warps"),
        ),
    ),
)


def summarize_ncu_raw_csv(raw_csv_text: str) -> str:
    """Select the richest kernel row and render a solver-oriented text summary."""
    rows = list(csv.DictReader(io.StringIO(raw_csv_text)))
    if not rows:
        return (
            "NCU summary could not be generated because the raw CSV had no data rows.\n"
            "Read profiles/latest.details.txt for the full text report.\n"
        )

    def score(row: dict[str, str]) -> int:
        return sum(
            1 for value in row.values() if isinstance(value, str) and any(ch.isdigit() for ch in value)
        )

    row = max(rows, key=score)

    def first_value(*keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    lines = [
        "# NCU Summary",
        "",
        "Prefer this file first. Read `profiles/latest.details.txt` only when you need the full report.",
        "",
    ]

    for title, metrics in KEY_METRIC_GROUPS:
        lines.append(f"## {title}")
        wrote_any = False
        for label, key in metrics:
            value = first_value(key)
            if value is None:
                continue
            lines.append(f"- {label}: {value}")
            wrote_any = True
        if not wrote_any:
            lines.append("- no values found in the exported raw CSV")
        lines.append("")

    for title, metrics in CUDAFORGE_METRIC_GROUPS:
        lines.append(f"## {title}")
        wrote_any = False
        for label, key in metrics:
            value = first_value(key)
            if value is None:
                continue
            lines.append(f"- {label}: {value}")
            wrote_any = True
        if not wrote_any:
            lines.append("- none of these counters were present in the exported raw CSV")
        lines.append("")

    stall_entries: list[tuple[str, float, str]] = []
    for key, value in row.items():
        if "stalled_" not in key:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        numeric = as_float(value)
        if numeric is None or numeric <= 0:
            continue
        if "smsp__average_warps_issue_stalled_" in key:
            stall_name = key.split("stalled_", 1)[1].split("_per_", 1)[0]
        else:
            stall_name = key.rsplit("__", 1)[-1]
        stall_entries.append((stall_name, numeric, value.strip()))

    lines.append("## Top warp stalls")
    if stall_entries:
        for stall_name, _, raw_value in sorted(stall_entries, key=lambda item: -item[1])[:8]:
            lines.append(f"- {stall_name}: {raw_value}")
    else:
        lines.append("- no positive warp-stall metrics were found in the exported raw CSV")
    lines.append("")
    lines.append("## Next step")
    lines.append(
        "- Re-read `HARDWARE.md`, then use this summary plus `profiles/latest.details.txt` to pick the next branch."
    )
    lines.append("")
    return "\n".join(lines)
