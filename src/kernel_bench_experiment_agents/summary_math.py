"""Collect small mathematical helpers used while computing archived-run summaries.

Keeping these utilities separate keeps the reporting layer focused on structure instead of scalar math details.
"""

from __future__ import annotations

import math


def parse_pass_k_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("pass@k values must be positive integers")
        values.append(value)
    return sorted(set(values))


def pass_at_k_estimate(n: int, c: int, k: int) -> float | None:
    if n <= 0 or k <= 0 or n < k:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    numerator = math.comb(n - c, k)
    denominator = math.comb(n, k)
    return 1.0 - (numerator / denominator)
