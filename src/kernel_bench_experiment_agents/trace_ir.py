from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .common import normalize_tool_name


def load_trace_event_entries(
    events_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    raw_events: list[dict[str, Any]] = []
    raw_event_entries: list[tuple[int, dict[str, Any]]] = []
    if not events_path.exists():
        return raw_events, raw_event_entries

    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_events.append(payload)
        raw_event_entries.append((line_number, payload))
    return raw_events, raw_event_entries


def _collect_text_fragments(payload: Any, fragments: list[str], limit: int = 6) -> None:
    if len(fragments) >= limit:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if len(fragments) >= limit:
                return
            if key in {"text", "message", "summary", "content", "delta"} and isinstance(
                value, str
            ):
                stripped = value.strip()
                if stripped:
                    fragments.append(stripped)
                    continue
            _collect_text_fragments(value, fragments, limit=limit)
    elif isinstance(payload, list):
        for value in payload:
            if len(fragments) >= limit:
                return
            _collect_text_fragments(value, fragments, limit=limit)


def _collect_urls(payload: Any, *, urls: set[str]) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_urls(value, urls=urls)
        return
    if isinstance(payload, list):
        for value in payload:
            _collect_urls(value, urls=urls)
        return
    if isinstance(payload, str):
        for match in re.findall(r"https?://[^\s\"'<>]+", payload):
            urls.add(match)


def _find_first_value(payload: Any, keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and value not in (None, "", [], {}):
                return value
            found = _find_first_value(value, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_first_value(value, keys)
            if found not in (None, "", [], {}):
                return found
    return None


def _sample_refs(payload: Any) -> list[str]:
    serialized = json.dumps(payload, sort_keys=True)
    return sorted(set(re.findall(r"sample_(\d+)", serialized)))


# Claude helpers

def claude_content_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("type") != "assistant":
        return []
    message = payload.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]



def claude_tool_use_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for block in claude_content_blocks(payload)
        if block.get("type") == "tool_use"
    ]



def claude_tool_name(block: dict[str, Any]) -> str:
    return str(block.get("name") or "").strip()



def claude_tool_input(block: dict[str, Any]) -> dict[str, Any]:
    value = block.get("input")
    return value if isinstance(value, dict) else {}



def claude_tool_command(block: dict[str, Any]) -> str | None:
    tool_input = claude_tool_input(block)
    for key in ("command", "cmd", "shell_command"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None



def claude_tool_path(block: dict[str, Any]) -> str | None:
    tool_input = claude_tool_input(block)
    for key in ("file_path", "path", "target_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None



def _text_excerpt(payload: Any) -> str | None:
    fragments: list[str] = []
    _collect_text_fragments(payload, fragments)
    excerpt = " ".join(fragment.replace("\n", " ") for fragment in fragments).strip()
    if len(excerpt) > 400:
        excerpt = excerpt[:397] + "..."
    return excerpt or None



def _base_ir_event(
    *,
    tool: str,
    line_number: int,
    payload: dict[str, Any],
    kind: str,
    source_event_type: str,
    source_subtype: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    command: str | None = None,
    path: str | None = None,
    text: str | None = None,
    block_index: int | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
    item_id: str | None = None,
    message_id: str | None = None,
    parent_tool_use_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "line": line_number,
        "tool": tool,
        "kind": kind,
        "source_event_type": source_event_type,
        "source_subtype": source_subtype,
        "role": role,
        "tool_name": tool_name,
        "command": command[:400] if isinstance(command, str) else None,
        "path": path,
        "text": text,
        "block_index": block_index,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "item_id": item_id,
        "message_id": message_id,
        "parent_tool_use_id": parent_tool_use_id,
        "sample_refs": _sample_refs(payload),
        "metadata": metadata or {},
    }
    return event



def _claude_domains(payload: Any) -> list[str]:
    urls: set[str] = set()
    _collect_urls(payload, urls=urls)
    return sorted(
        {
            parsed.hostname
            for parsed in (urlparse(url) for url in urls)
            if parsed.hostname
        }
    )



def _claude_ir_events(
    line_number: int,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    source_event_type = str(payload.get("type") or "unknown")
    source_subtype = (
        str(payload.get("subtype")) if payload.get("subtype") is not None else None
    )
    role = source_event_type if source_event_type in {"assistant", "user"} else None
    message = payload.get("message")
    message_id = message.get("id") if isinstance(message, dict) else None
    parent_tool_use_id = None
    if isinstance(message, dict):
        parent_tool_use_id = message.get("parent_tool_use_id")
    if parent_tool_use_id is None:
        parent_tool_use_id = payload.get("parent_tool_use_id")

    if source_event_type == "assistant":
        blocks = claude_content_blocks(payload)
        for block_index, block in enumerate(blocks):
            block_type = str(block.get("type") or "unknown")
            if block_type == "text":
                text = block.get("text")
                events.append(
                    _base_ir_event(
                        tool="claude",
                        line_number=line_number,
                        payload=payload,
                        kind="assistant_text",
                        source_event_type=source_event_type,
                        source_subtype=source_subtype,
                        role="assistant",
                        text=str(text).strip() if isinstance(text, str) else None,
                        block_index=block_index,
                        message_id=message_id if isinstance(message_id, str) else None,
                        parent_tool_use_id=(
                            str(parent_tool_use_id)
                            if isinstance(parent_tool_use_id, str)
                            else None
                        ),
                    )
                )
                continue

            if block_type == "tool_use":
                tool_name = claude_tool_name(block)
                tool_name_lower = tool_name.strip().lower()
                kind = "tool_use"
                command = claude_tool_command(block)
                path = claude_tool_path(block)
                metadata: dict[str, Any] = {}
                if tool_name_lower == "bash":
                    kind = "command_execution"
                elif tool_name_lower in {"edit", "multiedit", "write"}:
                    kind = "file_change"
                elif tool_name_lower in {"read", "view", "open"}:
                    kind = "file_read"
                elif tool_name_lower in {"websearch", "web_search"}:
                    kind = "web_search"
                    tool_input = claude_tool_input(block)
                    query = tool_input.get("query")
                    raw_queries = tool_input.get("queries")
                    metadata["query"] = str(query) if query else None
                    metadata["queries"] = (
                        [str(value) for value in raw_queries if value]
                        if isinstance(raw_queries, list)
                        else ([str(query)] if query else [])
                    )
                    metadata["domains"] = _claude_domains(block)
                elif tool_name_lower in {"task", "subagent", "agent"}:
                    kind = "subagent_spawn"
                    tool_input = claude_tool_input(block)
                    metadata["description"] = tool_input.get("description")
                    metadata["subagent_type"] = tool_input.get("subagent_type")
                elif tool_name_lower == "wait":
                    kind = "wait"

                events.append(
                    _base_ir_event(
                        tool="claude",
                        line_number=line_number,
                        payload=payload,
                        kind=kind,
                        source_event_type=source_event_type,
                        source_subtype=source_subtype,
                        role="assistant",
                        tool_name=tool_name or None,
                        command=command,
                        path=path,
                        text=_text_excerpt(block),
                        block_index=block_index,
                        message_id=message_id if isinstance(message_id, str) else None,
                        parent_tool_use_id=(
                            str(parent_tool_use_id)
                            if isinstance(parent_tool_use_id, str)
                            else None
                        ),
                        metadata=metadata,
                    )
                )
                continue

            events.append(
                _base_ir_event(
                    tool="claude",
                    line_number=line_number,
                    payload=payload,
                    kind="assistant_block",
                    source_event_type=source_event_type,
                    source_subtype=source_subtype,
                    role="assistant",
                    text=_text_excerpt(block),
                    block_index=block_index,
                    message_id=message_id if isinstance(message_id, str) else None,
                    parent_tool_use_id=(
                        str(parent_tool_use_id)
                        if isinstance(parent_tool_use_id, str)
                        else None
                    ),
                    metadata={"block_type": block_type},
                )
            )

    if events:
        return events

    return [
        _base_ir_event(
            tool="claude",
            line_number=line_number,
            payload=payload,
            kind="raw_event",
            source_event_type=source_event_type,
            source_subtype=source_subtype,
            role=role,
            text=_text_excerpt(payload),
            message_id=message_id if isinstance(message_id, str) else None,
            parent_tool_use_id=(
                str(parent_tool_use_id) if isinstance(parent_tool_use_id, str) else None
            ),
        )
    ]



def _codex_ir_events(
    line_number: int,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    source_event_type = str(payload.get("type") or "unknown")
    role = _find_first_value(payload, {"role", "sender", "author"})
    source_subtype = None
    thread_id = payload.get("thread_id")
    turn_id = payload.get("turn_id")
    item_id = None
    tool_name = None
    command = None
    path = None
    metadata: dict[str, Any] = {}
    kind = "raw_event"

    if source_event_type == "item.completed":
        item = payload.get("item")
        if isinstance(item, dict):
            item_id = item.get("id")
            item_type = str(item.get("type") or "unknown")
            source_subtype = item_type
            text = _text_excerpt(item)
            if item_type == "command_execution":
                kind = "command_execution"
                command = item.get("command") if isinstance(item.get("command"), str) else None
            elif item_type == "file_change":
                kind = "file_change"
                paths = item.get("paths")
                if isinstance(paths, list) and paths:
                    first_path = paths[0]
                    path = str(first_path) if first_path is not None else None
                elif isinstance(item.get("path"), str):
                    path = str(item.get("path"))
            elif item_type == "web_search":
                kind = "web_search"
                query = item.get("query")
                action = item.get("action")
                queries: list[str] = []
                if isinstance(action, dict):
                    raw_queries = action.get("queries")
                    if isinstance(raw_queries, list):
                        queries = [str(value) for value in raw_queries if value]
                    if not query:
                        query = action.get("query")
                metadata["query"] = str(query) if query else None
                metadata["queries"] = queries or ([str(query)] if query else [])
                metadata["domains"] = _claude_domains(item)
            elif item_type == "collab_tool_call":
                tool_name = str(item.get("tool") or "").strip() or None
                lowered = (tool_name or "").lower()
                if lowered == "spawn_agent":
                    kind = "subagent_spawn"
                    receiver_ids = item.get("receiver_thread_ids")
                    if isinstance(receiver_ids, list):
                        metadata["receiver_thread_ids"] = [
                            str(value) for value in receiver_ids if isinstance(value, str)
                        ]
                elif lowered == "wait":
                    kind = "wait"
                elif lowered == "web_search":
                    kind = "web_search"
                else:
                    kind = "tool_call"
            else:
                kind = item_type
            return [
                _base_ir_event(
                    tool="codex",
                    line_number=line_number,
                    payload=payload,
                    kind=kind,
                    source_event_type=source_event_type,
                    source_subtype=source_subtype,
                    role=str(role) if role is not None else None,
                    tool_name=tool_name,
                    command=command,
                    path=path,
                    text=text,
                    thread_id=str(thread_id) if isinstance(thread_id, str) else None,
                    turn_id=str(turn_id) if isinstance(turn_id, str) else None,
                    item_id=str(item_id) if isinstance(item_id, str) else None,
                    metadata=metadata,
                )
            ]

    text = _text_excerpt(payload)
    tool_name = _find_first_value(
        payload,
        {"tool_name", "recipient_name", "function_name", "command_name"},
    )
    command = _find_first_value(payload, {"command", "cmd", "shell_command"})
    return [
        _base_ir_event(
            tool="codex",
            line_number=line_number,
            payload=payload,
            kind=kind,
            source_event_type=source_event_type,
            role=str(role) if role is not None else None,
            tool_name=str(tool_name) if isinstance(tool_name, str) else None,
            command=str(command) if isinstance(command, str) else None,
            text=text,
            thread_id=str(thread_id) if isinstance(thread_id, str) else None,
            turn_id=str(turn_id) if isinstance(turn_id, str) else None,
        )
    ]



def materialize_trace_ir(
    raw_event_entries: list[tuple[int, dict[str, Any]]],
    *,
    tool: str = "codex",
) -> list[dict[str, Any]]:
    tool = normalize_tool_name(tool)
    ir_events: list[dict[str, Any]] = []
    for line_number, payload in raw_event_entries:
        if tool == "claude":
            ir_events.extend(_claude_ir_events(line_number, payload))
        else:
            ir_events.extend(_codex_ir_events(line_number, payload))
    return ir_events



def final_message_from_raw_events(
    raw_events: list[dict[str, Any]],
    *,
    tool: str,
) -> str | None:
    tool = normalize_tool_name(tool)
    final_text = None
    if tool == "claude":
        for payload in reversed(raw_events):
            if payload.get("type") != "assistant":
                continue
            fragments = [
                str(block.get("text")).strip()
                for block in claude_content_blocks(payload)
                if block.get("type") == "text" and isinstance(block.get("text"), str)
            ]
            final_text = "\n\n".join(fragment for fragment in fragments if fragment)
            if final_text:
                break
    else:
        for payload in reversed(raw_events):
            fragments: list[str] = []
            _collect_text_fragments(payload, fragments)
            final_text = "\n\n".join(fragment for fragment in fragments if fragment)
            if final_text:
                break

    if final_text:
        return final_text.strip() + "\n"
    return None
