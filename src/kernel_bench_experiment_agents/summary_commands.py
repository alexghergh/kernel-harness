from __future__ import annotations

import argparse

from .common import emit_json
from .project import artifacts_dir, write_json
from .run_metrics import load_baseline_file
from .summary_math import parse_pass_k_list
from .summary_report import build_run_summary_payload
from .summary_scan import SummaryScanConfig, collect_problem_rows


def command_summarize_run(args: argparse.Namespace) -> None:
    pass_k_values = parse_pass_k_list(args.pass_k)
    run_root = artifacts_dir() / args.run_name
    if not run_root.exists():
        raise SystemExit(f"No run artifacts found at {run_root}")

    selected_levels = set(args.level)
    selected_problem_ids = set(args.problem_id)
    config = SummaryScanConfig(
        dataset_src=args.dataset_src,
        kernelbench_root=args.kernelbench_root,
        eager_baseline=load_baseline_file(args.eager_baseline_file),
        compile_baseline=load_baseline_file(args.compile_baseline_file),
    )
    problem_rows = collect_problem_rows(
        run_root=run_root,
        selected_levels=selected_levels,
        selected_problem_ids=selected_problem_ids,
        config=config,
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
