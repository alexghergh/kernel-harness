from __future__ import annotations

import argparse
from pathlib import Path

from .kernelbench import load_problem
from .project import build_problem_dir


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
    return parser.parse_args()


def main() -> None:
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
    init_inputs = get_init_inputs()
    inputs = get_inputs()
    init_inputs = [
        tensor.cuda(device=device) if isinstance(tensor, torch.Tensor) else tensor
        for tensor in init_inputs
    ]
    inputs = [
        tensor.cuda(device=device) if isinstance(tensor, torch.Tensor) else tensor
        for tensor in inputs
    ]

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
