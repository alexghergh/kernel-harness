"""Call the launcher-owned workspace command broker from local wrapper scripts."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any


def send_request(*, socket_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(socket_path))
            client.sendall((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
            client.shutdown(socket.SHUT_WR)
            response = b""
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                response += chunk
    except OSError as exc:
        reason = exc.strerror or exc.__class__.__name__
        raise SystemExit(f"command broker request failed: {reason}") from exc
    if not response.strip():
        raise SystemExit("command broker returned an empty response")
    try:
        decoded = json.loads(response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit("command broker returned an invalid JSON response") from exc
    if not isinstance(decoded, dict):
        raise SystemExit("command broker returned a non-object response")
    return decoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbh-command-client")
    parser.add_argument("--socket", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run-candidate")
    subparsers.add_parser("profile-ncu")
    subparsers.add_parser("goal-status")
    research = subparsers.add_parser("research-nvidia-docs")
    research.add_argument("--query", default="")
    research.add_argument("--url", default="")
    research.add_argument("--max-results", type=int, default=8)
    research.add_argument("--max-chars", type=int, default=12000)
    subparsers.add_parser("best-result")
    complete = subparsers.add_parser("complete-problem")
    complete.add_argument("--summary", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    request: dict[str, Any] = {"command": args.command.replace("-", "_")}
    if args.command == "complete-problem":
        request["summary"] = args.summary
    if args.command == "research-nvidia-docs":
        request["query"] = args.query
        request["url"] = args.url
        request["max_results"] = args.max_results
        request["max_chars"] = args.max_chars

    response = send_request(
        socket_path=Path(args.socket).expanduser().resolve(),
        payload=request,
    )
    stdout = response.get("stdout")
    if isinstance(stdout, str) and stdout:
        sys.stdout.write(stdout)
    if response.get("ok") is True:
        return 0
    error = response.get("error")
    if isinstance(error, str) and error.strip():
        print(error, file=sys.stderr)
    else:
        print("command broker request failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
