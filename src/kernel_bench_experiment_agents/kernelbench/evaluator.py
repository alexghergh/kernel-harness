"""Evaluate KernelBench candidates through the official evaluation helper."""

from __future__ import annotations

import inspect
import json
from typing import Any

from kernel_bench_experiment_agents.kernelbench.problems import import_kernelbench_modules, load_problem
from kernel_bench_experiment_agents.runtime.project import build_problem_dir


def _maybe_precision(eval_module: Any, precision: str | None) -> Any:
    if not precision:
        return None
    helper = getattr(eval_module, "get_torch_dtype_from_string", None)
    if callable(helper):
        return helper(precision)
    return precision


def _serializable(payload: Any) -> Any:
    try:
        json.dumps(payload)
        return payload
    except TypeError:
        if isinstance(payload, dict):
            return {str(k): _serializable(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [_serializable(v) for v in payload]
        return str(payload)


def evaluate_candidate(
    *,
    candidate_src: str,
    level: int,
    problem_id: int,
    dataset_src: str,
    run_name: str,
    sample_id: int,
    gpu_id: int = 0,
    timing_method: str | None = None,
    backend: str | None = None,
    precision: str | None = None,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    verbose: bool = False,
    explicit_kernelbench_root: str | None = None,
) -> dict[str, Any]:
    _, eval_module = import_kernelbench_modules(explicit_kernelbench_root)
    problem = load_problem(
        level=level,
        problem_id=problem_id,
        dataset_src=dataset_src,
        explicit_kernelbench_root=explicit_kernelbench_root,
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for evaluation. Install KernelBench GPU dependencies first."
        ) from exc

    build_dir = build_problem_dir(run_name, level, problem_id, f"sample_{sample_id}")
    device = torch.device(f"cuda:{gpu_id}")
    evaluator = getattr(eval_module, "eval_kernel_against_ref", None)
    if not callable(evaluator):
        raise RuntimeError("kernelbench.eval.eval_kernel_against_ref was not found.")

    signature = inspect.signature(evaluator)
    call_kwargs: dict[str, Any] = {
        "original_model_src": problem.code,
        "custom_model_src": candidate_src,
        "measure_performance": True,
        "verbose": verbose,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "build_dir": str(build_dir),
        "device": device,
    }
    optional_kwargs = {
        "timing_method": timing_method,
        "backend": backend,
        "precision": _maybe_precision(eval_module, precision),
    }
    for key, value in optional_kwargs.items():
        if value is not None and key in signature.parameters:
            call_kwargs[key] = value

    result = evaluator(**call_kwargs)
    payload = {
        "compiled": getattr(result, "compiled", None),
        "correctness": getattr(result, "correctness", None),
        "runtime": getattr(result, "runtime", None),
        "runtime_stats": _serializable(getattr(result, "runtime_stats", None)),
        "ref_runtime": getattr(result, "ref_runtime", None),
        "ref_runtime_stats": _serializable(getattr(result, "ref_runtime_stats", None)),
        "metadata": _serializable(getattr(result, "metadata", None)),
        "raw_repr": repr(result),
    }
    return payload
