"""Host a launcher-owned Unix-socket command broker for one problem workspace."""

from __future__ import annotations

import argparse
import io
import json
import signal
import socketserver
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kernel_bench_experiment_agents.activity_trace import append_activity_event
from kernel_bench_experiment_agents.kernelbench.commands.nvidia_docs import command_research_nvidia_docs
from kernel_bench_experiment_agents.kernelbench.commands.profile import command_profile_ncu
from kernel_bench_experiment_agents.kernelbench.commands.run_candidate import command_run_candidate
from kernel_bench_experiment_agents.runtime.project import archive_problem_dir
from kernel_bench_experiment_agents.runtime.solver_sanitize import (
    sanitize_solver_text,
    sanitize_solver_value,
)
from kernel_bench_experiment_agents.kernelbench.commands.status import (
    command_best_result,
    command_complete_problem,
    command_goal_status,
)
from kernel_bench_experiment_agents.workspace.paths import (
    validate_workspace_assignment,
    workspace_candidate_path,
)


@dataclass(frozen=True)
class BrokerContext:
    workspace: Path
    run_name: str
    level: int
    problem_id: int
    dataset_src: str
    kernelbench_root: str | None
    num_gpu_slots: int
    precision: str
    client_tool: str
    activity_events_path: Path


class InvocationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stdout: str,
        payload: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.payload = payload


def _problem_archive_root(ctx: BrokerContext) -> Path:
    return archive_problem_dir(ctx.run_name, ctx.level, ctx.problem_id)


def _sanitize_text_for_solver(ctx: BrokerContext, text: str | None) -> str | None:
    if text is None:
        return None
    return sanitize_solver_text(
        text,
        workspace=ctx.workspace,
        problem_archive_root=_problem_archive_root(ctx),
    )


def _sanitize_value_for_solver(ctx: BrokerContext, value: Any) -> Any:
    return sanitize_solver_value(
        value,
        workspace=ctx.workspace,
        problem_archive_root=_problem_archive_root(ctx),
    )


def _sanitize_response_for_solver(
    ctx: BrokerContext,
    response: dict[str, Any],
) -> dict[str, Any]:
    return _sanitize_value_for_solver(ctx, response)


def _append_activity(
    ctx: BrokerContext,
    *,
    kind: str,
    command: str,
    command_name: str,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    event_metadata = _sanitize_value_for_solver(ctx, dict(metadata or {}))
    event_metadata.setdefault("command_name", command_name)
    append_activity_event(
        ctx.activity_events_path,
        {
            "tool": ctx.client_tool,
            "kind": kind,
            "command": command,
            "text": _sanitize_text_for_solver(ctx, text),
            "metadata": event_metadata,
        },
    )


def _invoke(handler: Callable[[argparse.Namespace], None], namespace: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            handler(namespace)
        stdout = buffer.getvalue()
    except (Exception, SystemExit) as exc:
        stdout = buffer.getvalue()
        payload: dict[str, Any] = {}
        if stdout.strip():
            try:
                decoded = json.loads(stdout)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                payload = decoded
        raise InvocationError(str(exc), stdout=stdout, payload=payload) from exc
    payload = json.loads(stdout) if stdout.strip() else {}
    return stdout, payload


def _handle_run_candidate(ctx: BrokerContext) -> tuple[str, dict[str, Any]]:
    try:
        stdout, payload = _invoke(
            command_run_candidate,
            argparse.Namespace(
                candidate=str(workspace_candidate_path(ctx.workspace)),
                run_name=ctx.run_name,
                level=ctx.level,
                problem_id=ctx.problem_id,
                dataset_src=ctx.dataset_src,
                kernelbench_root=ctx.kernelbench_root,
                gpu_id=None,
                num_gpu_slots=ctx.num_gpu_slots,
                timing_method=None,
                backend="cuda",
                precision=ctx.precision,
                num_correct_trials=5,
                num_perf_trials=100,
                prompt_path=None,
                workspace=str(ctx.workspace),
            ),
        )
    except InvocationError as exc:
        _append_activity(
            ctx,
            kind="command_execution",
            command="./bin/run_candidate.sh",
            command_name="run_candidate",
            metadata={
                "status": exc.payload.get("status") or "failed",
                "sample_id": exc.payload.get("sample_id"),
                "error": str(exc),
            },
        )
        raise
    _append_activity(
        ctx,
        kind="command_execution",
        command="./bin/run_candidate.sh",
        command_name="run_candidate",
        metadata={"status": payload.get("status"), "sample_id": payload.get("sample_id")},
    )
    return stdout, payload


def _handle_profile_ncu(ctx: BrokerContext) -> tuple[str, dict[str, Any]]:
    try:
        stdout, payload = _invoke(
            command_profile_ncu,
            argparse.Namespace(
                candidate=str(workspace_candidate_path(ctx.workspace)),
                run_name=ctx.run_name,
                level=ctx.level,
                problem_id=ctx.problem_id,
                dataset_src=ctx.dataset_src,
                kernelbench_root=ctx.kernelbench_root,
                gpu_id=None,
                num_gpu_slots=ctx.num_gpu_slots,
                sample_id=None,
                ncu_set="full",
                precision=ctx.precision,
                workspace=str(ctx.workspace),
            ),
        )
    except InvocationError as exc:
        _append_activity(
            ctx,
            kind="command_execution",
            command="./bin/profile_ncu.sh",
            command_name="profile_ncu",
            metadata={
                "status": exc.payload.get("status") or "failed",
                "profile_id": exc.payload.get("profile_id"),
                "error": str(exc),
            },
        )
        raise
    _append_activity(
        ctx,
        kind="command_execution",
        command="./bin/profile_ncu.sh",
        command_name="profile_ncu",
        metadata={"status": payload.get("status"), "profile_id": payload.get("profile_id")},
    )
    return stdout, payload


def _handle_goal_status(ctx: BrokerContext) -> tuple[str, dict[str, Any]]:
    try:
        stdout, payload = _invoke(
            command_goal_status,
            argparse.Namespace(
                run_name=ctx.run_name,
                level=ctx.level,
                problem_id=ctx.problem_id,
                workspace=str(ctx.workspace),
            ),
        )
    except InvocationError as exc:
        _append_activity(
            ctx,
            kind="command_execution",
            command="./bin/goal_status.sh",
            command_name="goal_status",
            metadata={
                "status_mode": exc.payload.get("status_mode"),
                "error": str(exc),
            },
        )
        raise
    _append_activity(
        ctx,
        kind="command_execution",
        command="./bin/goal_status.sh",
        command_name="goal_status",
        metadata={"status_mode": payload.get("status_mode")},
    )
    return stdout, payload


def _handle_research_nvidia_docs(
    ctx: BrokerContext,
    *,
    query: str,
    url: str,
    max_results: int,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    try:
        stdout, payload = _invoke(
            command_research_nvidia_docs,
            argparse.Namespace(
                query=query,
                url=url,
                max_results=max_results,
                max_chars=max_chars,
            ),
        )
    except InvocationError as exc:
        _append_activity(
            ctx,
            kind="command_execution",
            command="./bin/research_nvidia_docs.sh",
            command_name="research_nvidia_docs",
            text=query or url,
            metadata={
                "status": exc.payload.get("status") or "failed",
                "query": query or None,
                "url": url or None,
                "error": str(exc),
            },
        )
        raise
    _append_activity(
        ctx,
        kind="command_execution",
        command="./bin/research_nvidia_docs.sh",
        command_name="research_nvidia_docs",
        text=query or url,
        metadata={
            "status": payload.get("status"),
            "query": payload.get("query"),
            "url": payload.get("url"),
            "result_count": len(payload.get("results") or []),
            "document_url": (
                payload.get("document", {}).get("url")
                if isinstance(payload.get("document"), dict)
                else None
            ),
            "domains": ["docs.nvidia.com"],
        },
    )
    return stdout, payload


def _handle_best_result(ctx: BrokerContext) -> tuple[str, dict[str, Any]]:
    try:
        stdout, payload = _invoke(
            command_best_result,
            argparse.Namespace(
                run_name=ctx.run_name,
                level=ctx.level,
                problem_id=ctx.problem_id,
            ),
        )
    except InvocationError as exc:
        _append_activity(
            ctx,
            kind="command_execution",
            command="./bin/best_result.sh",
            command_name="best_result",
            metadata={
                "sample_id": exc.payload.get("sample_id"),
                "error": str(exc),
            },
        )
        raise
    _append_activity(
        ctx,
        kind="command_execution",
        command="./bin/best_result.sh",
        command_name="best_result",
        metadata={"sample_id": payload.get("sample_id")},
    )
    return stdout, payload


def _handle_complete_problem(ctx: BrokerContext, *, summary: str) -> tuple[str, dict[str, Any]]:
    try:
        stdout, payload = _invoke(
            command_complete_problem,
            argparse.Namespace(
                run_name=ctx.run_name,
                level=ctx.level,
                problem_id=ctx.problem_id,
                workspace=str(ctx.workspace),
                summary=summary,
                allow_overwrite=False,
            ),
        )
    except InvocationError as exc:
        _append_activity(
            ctx,
            kind="command_execution",
            command="./bin/complete_problem.sh",
            command_name="complete_problem",
            text=summary,
            metadata={"summary": summary, "error": str(exc)},
        )
        raise
    _append_activity(
        ctx,
        kind="command_execution",
        command="./bin/complete_problem.sh",
        command_name="complete_problem",
        text=summary,
        metadata={"summary": summary},
    )
    return stdout, payload


def _dispatch_request(ctx: BrokerContext, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    command = str(request.get("command") or "").strip()
    if command == "run_candidate":
        return _handle_run_candidate(ctx)
    if command == "profile_ncu":
        return _handle_profile_ncu(ctx)
    if command == "goal_status":
        return _handle_goal_status(ctx)
    if command == "research_nvidia_docs":
        query = str(request.get("query") or "").strip()
        url = str(request.get("url") or "").strip()
        max_results = int(request.get("max_results") or 8)
        max_chars = int(request.get("max_chars") or 12000)
        return _handle_research_nvidia_docs(
            ctx,
            query=query,
            url=url,
            max_results=max_results,
            max_chars=max_chars,
        )
    if command == "best_result":
        return _handle_best_result(ctx)
    if command == "complete_problem":
        summary = str(request.get("summary") or "").strip()
        if not summary:
            raise RuntimeError("summary is required for complete_problem")
        return _handle_complete_problem(ctx, summary=summary)
    raise RuntimeError(f"unsupported broker command: {command}")


class BrokerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: str, *, context: BrokerContext) -> None:
        self.context = context
        self.command_lock = threading.Lock()
        self._active_requests = 0
        self._active_request_condition = threading.Condition()
        super().__init__(socket_path, BrokerHandler)

    def note_request_start(self) -> None:
        with self._active_request_condition:
            self._active_requests += 1

    def note_request_end(self) -> None:
        with self._active_request_condition:
            self._active_requests -= 1
            self._active_request_condition.notify_all()

    def wait_for_idle(self, *, timeout: float | None = None) -> None:
        with self._active_request_condition:
            self._active_request_condition.wait_for(
                lambda: self._active_requests == 0,
                timeout=timeout,
            )


class BrokerHandler(socketserver.StreamRequestHandler):
    server: BrokerServer

    def handle(self) -> None:
        self.server.note_request_start()
        try:
            raw_request = self.rfile.readline()
            if not raw_request:
                return
            try:
                request = json.loads(raw_request.decode("utf-8"))
                if not isinstance(request, dict):
                    raise RuntimeError("request payload must be a JSON object")
                with self.server.command_lock:
                    stdout, payload = _dispatch_request(self.server.context, request)
                response = {
                    "ok": True,
                    "stdout": stdout,
                    "payload": payload,
                }
            except InvocationError as exc:
                response = {
                    "ok": False,
                    "error": str(exc),
                    "stdout": exc.stdout,
                    "payload": exc.payload,
                }
            except Exception as exc:  # noqa: BLE001
                response = {
                    "ok": False,
                    "error": str(exc),
                }
            response = _sanitize_response_for_solver(self.server.context, response)
            self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            self.server.note_request_end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the KernelBench workspace command broker.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--dataset-src", default="local")
    parser.add_argument("--kernelbench-root", default=None)
    parser.add_argument("--num-gpu-slots", type=int, default=1)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--activity-events-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    socket_path = Path(args.socket).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    validate_workspace_assignment(
        workspace,
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
    )
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    context = BrokerContext(
        workspace=workspace,
        run_name=args.run_name,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        kernelbench_root=args.kernelbench_root,
        num_gpu_slots=args.num_gpu_slots,
        precision=args.precision,
        client_tool=args.tool,
        activity_events_path=Path(args.activity_events_path).expanduser().resolve(),
    )
    server = BrokerServer(str(socket_path), context=context)
    shutdown_requested = threading.Event()

    def _request_shutdown(*_: object) -> None:
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.wait_for_idle(timeout=30.0)
        server.server_close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
