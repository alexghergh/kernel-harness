"""Load KernelBench problems through the official dataset helpers."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.runtime.project import kernelbench_root


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


def import_kernelbench_modules(explicit_root: str | None = None) -> tuple[Any, Any]:
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
    dataset_module, _ = import_kernelbench_modules(explicit_kernelbench_root)
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
