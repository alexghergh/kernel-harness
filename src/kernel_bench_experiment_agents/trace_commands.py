from __future__ import annotations

import argparse
from pathlib import Path

from .agent_specs import write_workspace_helper_agent_specs
from .common import emit_json, normalize_tool_name
from .completion_policy import (
    annotate_completion_outcomes,
    apply_completion_policy,
    apply_trace_audit_to_completion,
)
from .project import now_iso, write_json, write_text
from .trace_analysis import audit_trace, trace_cost_usd, trace_counts, trace_usage_summary, web_searches_from_ir
from .trace_ir import final_message_from_raw_events, load_trace_event_entries, materialize_trace_ir
from .workspace_paths import read_json_file


def command_sync_helper_agent_specs(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).expanduser().resolve()
    archive_contract_dir = (
        Path(args.archive_contract_dir).expanduser().resolve()
        if args.archive_contract_dir
        else None
    )
    written = [
        str(path)
        for path in write_workspace_helper_agent_specs(
            workspace=workspace,
            archive_contract_dir=archive_contract_dir,
        )
    ]
    emit_json({"written": written, "workspace": str(workspace)})


def write_final_message(
    *,
    output_path: Path,
    tool: str,
    raw_events: list[dict],
) -> None:
    if (
        normalize_tool_name(tool) == "codex"
        and output_path.exists()
        and output_path.read_text(encoding="utf-8").strip()
    ):
        return
    final_text = final_message_from_raw_events(raw_events, tool=tool)
    if final_text:
        write_text(output_path, final_text)


def command_materialize_agent_trace(args: argparse.Namespace) -> None:
    tool = normalize_tool_name(args.tool)
    events_path = Path(args.events_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    raw_events, raw_event_entries = load_trace_event_entries(events_path)
    ir_events = materialize_trace_ir(raw_event_entries, tool=tool)

    token_usage = trace_usage_summary(raw_events, tool=tool)
    cost_usd = trace_cost_usd(raw_events, tool=tool)
    trace_counts_payload = trace_counts(ir_events, raw_events=raw_events, tool=tool)
    web_searches = web_searches_from_ir(ir_events)
    audit = None
    if args.workspace:
        audit = audit_trace(
            ir_events=ir_events,
            workspace=Path(args.workspace).expanduser().resolve(),
            raw_events=raw_events,
            tool=tool,
        )
    payload = {
        "tool": tool,
        "source_events_path": str(events_path),
        "generated_at": now_iso(),
        "trace_ir_version": 1,
        "num_raw_events": len(raw_events),
        "num_ir_events": len(ir_events),
        "token_usage": token_usage,
        "cost_usd": cost_usd,
        "trace_counts": trace_counts_payload,
        "web_searches": web_searches,
        "audit": audit,
        "ir_events": ir_events,
    }
    write_json(output_path, payload)
    if args.final_message_path:
        write_final_message(
            output_path=Path(args.final_message_path).expanduser().resolve(),
            tool=tool,
            raw_events=raw_events,
        )
    if args.completion_path:
        completion_path = Path(args.completion_path).expanduser().resolve()
        if completion_path.exists():
            completion_payload = read_json_file(completion_path)
            completion_payload["tool"] = tool
            completion_payload["token_usage"] = token_usage
            completion_payload["cost_usd"] = cost_usd
            completion_payload["trace_counts"] = trace_counts_payload
            completion_payload["web_searches"] = web_searches
            if audit is not None:
                completion_payload = apply_trace_audit_to_completion(
                    completion_payload,
                    audit,
                )
            completion_payload = apply_completion_policy(completion_payload)
            completion_payload = annotate_completion_outcomes(completion_payload)
            write_json(completion_path, completion_payload)
            if args.workspace:
                write_json(
                    Path(args.workspace).expanduser().resolve() / "completion.json",
                    completion_payload,
                )
    emit_json(
        {
            "output_path": str(output_path),
            "num_raw_events": len(raw_events),
            "num_ir_events": len(ir_events),
            "source_events_path": str(events_path),
            "token_usage": token_usage,
            "cost_usd": cost_usd,
            "audit": audit,
        }
    )
