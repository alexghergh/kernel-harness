"""Record and load synthetic MCP tool events for one solver problem.

The raw Codex/Claude client streams remain the source of truth for model turns and token usage,
but local harness access now goes through an MCP server. This module keeps a small per-problem
JSONL sidecar so trace counts and audit can still reason about solver-visible reads, writes, and
wrapper-equivalent tool calls without depending on client-specific MCP trace formats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kernel_bench_experiment_agents.runtime.project import append_jsonl



def append_mcp_event(events_path: Path, payload: dict[str, Any]) -> None:
    append_jsonl(events_path, payload)



def load_mcp_ir_events(
    events_path: Path,
    *,
    warn: bool = False,
    starting_line: int = 1_000_000,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    malformed_lines: list[int] = []
    if not events_path.exists():
        return events

    next_line = starting_line
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line_number)
            continue
        if not isinstance(payload, dict):
            malformed_lines.append(line_number)
            continue
        event = dict(payload)
        event.setdefault("line", next_line)
        event.setdefault("source_event_type", "mcp_tool_call")
        event.setdefault("metadata", {})
        events.append(event)
        next_line += 1

    if warn and malformed_lines:
        preview = ", ".join(str(line_number) for line_number in malformed_lines[:8])
        suffix = "" if len(malformed_lines) <= 8 else ", ..."
        print(
            f"warning: ignored {len(malformed_lines)} malformed MCP JSON line(s) in {events_path} at line(s) {preview}{suffix}",
            file=sys.stderr,
        )
    return events
