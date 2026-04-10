"""Collect small shared helpers used across commands, summaries, and trace handling.

Keeping these primitives here avoids repeating JSON emission, numeric coercion, and tool-name normalization logic.
"""

from __future__ import annotations

import json
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


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
