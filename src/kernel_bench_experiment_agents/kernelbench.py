"""Wrap the official KernelBench loaders and evaluation helpers behind harness-friendly functions.

The measured runner modules import this so the rest of the harness does not need to know KernelBench internals directly.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project import build_problem_dir, kernelbench_root


@dataclass
class ProblemRecord:
    level: int
    problem_id: int
    dataset_src: str
    name: str | None
    path: str | None
    code: str


def _ensure_kernelbench_importable(explicit_root: str | None = None) -> Path:
    root = kernelbench_root(explicit_root)
    src_dir = root / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _import_kernelbench_modules(explicit_root: str | None = None) -> tuple[Any, Any]:
    _ensure_kernelbench_importable(explicit_root)
    dataset_module = importlib.import_module("kernelbench.dataset")
    eval_module = importlib.import_module("kernelbench.eval")
    return dataset_module, eval_module


def _construct_dataset(dataset_module: Any, level: int, dataset_src: str) -> Any:
    if not hasattr(dataset_module, "construct_kernelbench_dataset"):
        raise RuntimeError(
            "kernelbench.dataset.construct_kernelbench_dataset was not found."
        )

    builder = dataset_module.construct_kernelbench_dataset
    attempts = [
        {"level": level, "source": dataset_src},
        {"level": level, "dataset_src": dataset_src},
        {"level": level, "src": dataset_src},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return builder(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to construct the KernelBench dataset. "
        f"Tried kwargs {attempts}."
    ) from last_error


def _problem_from_dataset(dataset: Any, problem_id: int) -> Any:
    if hasattr(dataset, "get_problem_by_id"):
        try:
            return dataset.get_problem_by_id(problem_id)
        except Exception:
            pass

    if hasattr(dataset, "__iter__"):
        for item in dataset:
            if getattr(item, "problem_id", None) == problem_id:
                return item

    if hasattr(dataset, "__getitem__"):
        index_candidates = [problem_id, problem_id - 1]
        for index in index_candidates:
            try:
                item = dataset[index]
            except Exception:
                continue
            if getattr(item, "problem_id", None) == problem_id:
                return item

    raise RuntimeError(f"Failed to locate problem_id={problem_id} in the dataset.")


def load_problem(
    *,
    level: int,
    problem_id: int,
    dataset_src: str,
    explicit_kernelbench_root: str | None = None,
) -> ProblemRecord:
    dataset_module, _ = _import_kernelbench_modules(explicit_kernelbench_root)
    dataset = _construct_dataset(dataset_module, level=level, dataset_src=dataset_src)
    problem = _problem_from_dataset(dataset, problem_id)

    code = getattr(problem, "code", None)
    if not isinstance(code, str) or not code:
        problem_path = getattr(problem, "path", None)
        if problem_path:
            code = Path(problem_path).read_text(encoding="utf-8")
        else:
            raise RuntimeError("Problem code could not be resolved from the dataset.")

    return ProblemRecord(
        level=level,
        problem_id=problem_id,
        dataset_src=dataset_src,
        name=getattr(problem, "name", None),
        path=str(getattr(problem, "path", "")) or None,
        code=code,
    )


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
    _, eval_module = _import_kernelbench_modules(explicit_kernelbench_root)
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
