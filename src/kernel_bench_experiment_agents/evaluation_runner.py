"""Launch the actual KernelBench evaluation for one frozen candidate snapshot.

The higher-level run command shells out to this module so the measured subprocess stays small and isolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kernelbench import evaluate_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated candidate evaluation.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--kernelbench-root", default=None)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--dataset-src", default="local")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--timing-method", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_src = Path(args.candidate).read_text(encoding="utf-8")
    payload = evaluate_candidate(
        candidate_src=candidate_src,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        run_name=args.run_name,
        sample_id=args.sample_id,
        gpu_id=args.gpu_id,
        timing_method=args.timing_method,
        backend=args.backend,
        precision=args.precision,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=args.verbose,
        explicit_kernelbench_root=args.kernelbench_root,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
