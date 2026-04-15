"""Run the KernelBench stdio MCP server."""

from __future__ import annotations

import sys

from . import DEFAULT_PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION
from .context import load_context
from .handlers import TOOL_HANDLERS, text_result
from .protocol import read_message, write_error, write_response
from .registry import build_tool_descriptors
from .resources import (
    list_workspace_resource_templates,
    read_workspace_resource,
    workspace_resource_descriptors,
)

TOOL_DESCRIPTORS = build_tool_descriptors(TOOL_HANDLERS)
TOOL_DESCRIPTOR_MAP = {descriptor.name: descriptor for descriptor in TOOL_DESCRIPTORS}



def main() -> None:
    ctx = load_context()
    while True:
        request = read_message()
        if request is None:
            return
        method = request.get("method")
        request_id = request.get("id")

        if method == "notifications/initialized":
            continue
        if method == "initialize":
            params = request.get("params") or {}
            protocol_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
            write_response(
                request_id,
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            continue
        if method == "ping":
            write_response(request_id, {})
            continue
        if method == "resources/list":
            write_response(request_id, {"resources": workspace_resource_descriptors(ctx)})
            continue
        if method == "resources/templates/list":
            write_response(request_id, list_workspace_resource_templates())
            continue
        if method == "resources/read":
            params = request.get("params") or {}
            try:
                result = read_workspace_resource(ctx, str(params.get("uri") or ""))
            except Exception as exc:  # pragma: no cover - best effort server error surface
                write_error(request_id, -32602, f"resource read failed: {exc}")
            else:
                write_response(request_id, result)
            continue
        if method == "tools/list":
            write_response(request_id, {"tools": [descriptor.to_payload() for descriptor in TOOL_DESCRIPTORS]})
            continue
        if method == "tools/call":
            params = request.get("params") or {}
            tool_name = str(params.get("name") or "")
            descriptor = TOOL_DESCRIPTOR_MAP.get(tool_name)
            if descriptor is None:
                write_error(request_id, -32602, f"unknown tool: {tool_name}")
                continue
            try:
                result = descriptor.handler(ctx, params.get("arguments") or {})
            except SystemExit as exc:
                write_response(request_id, text_result(str(exc), is_error=True))
            except Exception as exc:  # pragma: no cover - best effort server error surface
                write_response(
                    request_id,
                    text_result(f"{type(exc).__name__}: {exc}", is_error=True),
                )
            else:
                write_response(request_id, result)
            continue
        if request_id is None:
            continue
        write_error(request_id, -32601, f"unsupported method: {method}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - startup failures should be visible in stderr logs
        print(
            f"{SERVER_NAME} MCP server failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
