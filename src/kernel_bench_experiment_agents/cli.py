"""Backward-compatible CLI entrypoint wrapper."""

from kernel_bench_experiment_agents.runtime.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
