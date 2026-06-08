"""Validate a CUDA-source candidate against the locked scaffold.

The validator collapses both the candidate and the locked template's editable spans to a
sentinel and demands the non-editable spans match byte-for-byte. It additionally rejects
forbidden vendor library tokens (cuBLAS, CUTLASS) inside the editable spans only — the
locked driver may, and does, call into cuBLAS for the reference comparison.
"""

from __future__ import annotations

from kernel_bench_experiment_agents.cuda_source.candidate_scaffold import (
    candidate_template,
    extract_editable_blocks,
    normalize_candidate_template,
)
from kernel_bench_experiment_agents.problem_source import CandidateValidationError


# Forbidden tokens inside the editable spans only. The locked driver section may include
# cublas headers and call cublasGemmEx; the agent's edits may not.
EDITABLE_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "cublas",
    "CUBLAS",
    "cuBLAS",
    "cutlass",
    "CUTLASS",
    "<cublas",
    "<cutlass",
)


def validate_candidate_source(candidate_src: str) -> None:
    """Validate one candidate.cu file. Raise ``CandidateValidationError`` on rejection."""
    try:
        normalized_candidate = normalize_candidate_template(candidate_src)
        normalized_template = normalize_candidate_template(candidate_template())
    except ValueError as exc:
        raise CandidateValidationError(str(exc)) from exc

    if normalized_candidate != normalized_template:
        raise CandidateValidationError(
            "candidate.cu must keep the locked driver scaffold unchanged and only edit the"
            " marked blocks. Restore the file to the template, then change only the contents"
            " inside the EDITABLE INCLUDES / KERNEL / LAUNCH markers."
        )

    try:
        blocks = extract_editable_blocks(candidate_src)
    except ValueError as exc:
        raise CandidateValidationError(str(exc)) from exc

    for marker, contents in blocks.items():
        for token in EDITABLE_FORBIDDEN_TOKENS:
            if token in contents:
                raise CandidateValidationError(
                    f"Token {token!r} is forbidden inside the editable block {marker!r}."
                    " Vendor libraries (cuBLAS, CUTLASS) may not be used in candidate code;"
                    " only the locked driver may call the reference."
                )
