"""Expose the top-level archived-run summarization command.

This command ties together scan and report helpers to produce one run-level JSON summary from durable artifacts.
"""

from __future__ import annotations

import argparse

from kernel_bench_experiment_agents.runtime.common import emit_json
from kernel_bench_experiment_agents.runtime.project import archive_dir, write_json
from kernel_bench_experiment_agents.summary.summary_math import parse_pass_k_list
from kernel_bench_experiment_agents.summary.summary_report import build_run_summary_payload
from kernel_bench_experiment_agents.summary.summary_scan import collect_problem_rows


def command_summarize_run(args: argparse.Namespace) -> None:
    pass_k_values = parse_pass_k_list(args.pass_k)
    run_root = archive_dir() / args.run_name
    if not run_root.exists():
        raise SystemExit(f"No run artifacts found at {run_root}")

    selected_levels = set(args.level)
    selected_problem_ids = set(args.problem_id)
    problem_rows = collect_problem_rows(
        run_root=run_root,
        selected_levels=selected_levels,
        selected_problem_ids=selected_problem_ids,
    )
    payload = build_run_summary_payload(
        run_name=args.run_name,
        selected_levels=selected_levels,
        selected_problem_ids=selected_problem_ids,
        pass_k_values=pass_k_values,
        problem_rows=problem_rows,
    )
    write_json(run_root / "run_summary.json", payload)
    emit_json(payload)
