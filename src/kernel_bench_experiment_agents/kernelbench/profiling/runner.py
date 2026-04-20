"""Launch one warmup-style model execution under Nsight Compute for a frozen candidate snapshot.

The profiler command shells out to this module so NCU sees only the candidate execution path, not the full harness orchestration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.kernelbench.problems import load_problem
from kernel_bench_experiment_agents.runtime.project import build_problem_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a candidate kernel for ncu profiling.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--kernelbench-root", default=None)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--dataset-src", default="local")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sample-label", default="scratch")
    parser.add_argument("--precision", default="bf16")
    return parser.parse_args()


def _resolve_precision_dtype(eval_module: Any, precision: str | None) -> Any:
    if not precision:
        return None
    helper = getattr(eval_module, "get_torch_dtype_from_string", None)
    if callable(helper):
        return helper(precision)
    return None


def _move_tree_to_device(value: Any, *, device: Any, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        target_dtype = value.dtype
        if dtype is not None and value.is_floating_point():
            target_dtype = dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, list):
        return [_move_tree_to_device(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(item, device=device, dtype=dtype) for item in value)
    if isinstance(value, dict):
        return {key: _move_tree_to_device(item, device=device, dtype=dtype) for key, item in value.items()}
    return value


def main() -> None:
    """Load one candidate module and execute it under the requested profiling precision."""
    args = parse_args()

    problem = load_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        explicit_kernelbench_root=args.kernelbench_root,
    )
    candidate_src = Path(args.candidate).read_text(encoding="utf-8")

    import torch
    from kernelbench import eval as kb_eval

    context: dict[str, object] = {}
    _, get_init_inputs, get_inputs = kb_eval.load_original_model_and_inputs(
        problem.code,
        context,
    )

    device = torch.device(f"cuda:{args.gpu_id}")
    precision_dtype = _resolve_precision_dtype(kb_eval, args.precision)
    init_inputs = _move_tree_to_device(get_init_inputs(), device=device, dtype=precision_dtype)
    inputs = _move_tree_to_device(get_inputs(), device=device, dtype=precision_dtype)

    model_type = kb_eval.load_custom_model(
        candidate_src,
        context,
        str(build_problem_dir(args.run_name, args.level, args.problem_id, args.sample_label)),
    )
    model = model_type(*init_inputs).cuda(device=device)
    torch.cuda.synchronize(device=device)

    for _ in range(5):
        _ = model(*inputs)
        torch.cuda.synchronize(device=device)


if __name__ == "__main__":
    main()
