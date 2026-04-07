from __future__ import annotations

import argparse

from .common import emit_json
from .kernelbench import load_problem


def command_problem_info(args: argparse.Namespace) -> None:
    problem = load_problem(
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        explicit_kernelbench_root=args.kernelbench_root,
    )
    emit_json(
        {
            "level": problem.level,
            "problem_id": problem.problem_id,
            "dataset_src": problem.dataset_src,
            "name": problem.name,
            "path": problem.path,
            "code": problem.code,
        }
    )
