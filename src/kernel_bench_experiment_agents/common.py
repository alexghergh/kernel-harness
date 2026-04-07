from __future__ import annotations

from typing import Any

TOOL_CHOICES = ("codex", "claude")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_tool_name(raw: Any) -> str:
    tool = str(raw or "codex").strip().lower()
    if tool not in TOOL_CHOICES:
        raise ValueError(
            f"Unsupported tool {tool!r}. Expected one of: {', '.join(TOOL_CHOICES)}."
        )
    return tool
