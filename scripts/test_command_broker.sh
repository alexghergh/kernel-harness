#!/usr/bin/env bash
# Prepare one problem workspace and smoke-test the active command broker path.
#
# Run from the harness repo root with the same launcher env you would use for run_agent_problem.sh.
#
# Required environment:
#   KERNELBENCH_ROOT=/path/to/KernelBench
#   HARDWARE_NAME=<timings-subdir name>
#
# Common overrides:
#   DATA_ROOT=/path/for/archive-and-state   (defaults to ./ from the launch directory)
#   TOOL=codex|claude
#   RUN_NAME=kernelbench-codex-h100-v4
#   LEVEL=1
#   PROBLEM_ID=1
#   MODEL=gpt-5.4|claude-opus-4-7
#   TIME_BUDGET_MINUTES=180
#   PRECISION=bf16
#   KERNELBENCH_TIMINGS_DIR=/path/to/KernelBench/results/timing/<hardware>  # optional override
set -euo pipefail

if [[ ! -f "./pyproject.toml" || ! -d "./src/kernel_bench_experiment_agents" ]]; then
  echo "Run scripts/test_command_broker.sh from the harness repo root." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python is not on PATH. Export PYTHON or run ./kb setup first." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-.}"
mkdir -p "${DATA_ROOT}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
export DATA_ROOT

TOOL="${TOOL:-codex}"
RUN_NAME="${RUN_NAME:-kernelbench-${TOOL}-h100-v4}"
LEVEL="${LEVEL:-1}"
PROBLEM_ID="${PROBLEM_ID:-1}"
DATASET_SRC="${DATASET_SRC:-local}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-180}"
PRECISION="${PRECISION:-bf16}"
HARDWARE_NAME="${HARDWARE_NAME:-}"
KERNELBENCH_TIMINGS_DIR="${KERNELBENCH_TIMINGS_DIR:-}"
MODEL="${MODEL:-gpt-5.4}"
if [[ "${TOOL}" == "claude" && "${MODEL}" == "gpt-5.4" ]]; then
  MODEL="claude-opus-4-7"
fi

if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the KernelBench checkout." >&2
  exit 1
fi
if [[ -z "${HARDWARE_NAME}" ]]; then
  echo "HARDWARE_NAME must name the KernelBench timings subdirectory to use." >&2
  exit 1
fi

PREP_OUTPUT="$({
  "${PYTHON_BIN}" -m kernel_bench_experiment_agents.runtime.cli prepare-problem-workspace \
    --run-name "${RUN_NAME}" \
    --level "${LEVEL}" \
    --problem-id "${PROBLEM_ID}" \
    --dataset-src "${DATASET_SRC}" \
    --kernelbench-root "${KERNELBENCH_ROOT}" \
    --timings-dir "${KERNELBENCH_TIMINGS_DIR}" \
    --hardware-name "${HARDWARE_NAME}" \
    --num-gpus 1 \
    --tool "${TOOL}" \
    --model "${MODEL}" \
    --time-budget-minutes "${TIME_BUDGET_MINUTES}" \
    --precision "${PRECISION}"
})"

WORKSPACE="$({
  PREP_OUTPUT="${PREP_OUTPUT}" "${PYTHON_BIN}" - <<'PY'
import json
import os

print(json.loads(os.environ["PREP_OUTPUT"])["workspace"])
PY
})"

PROBLEM_ARCHIVE="${DATA_ROOT}/archive/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}"
AGENT_DIR="${PROBLEM_ARCHIVE}/agent"
BROKER_SOCKET_ROOT="${DATA_ROOT}/state/s"
mkdir -p "${BROKER_SOCKET_ROOT}"
BROKER_STATE_DIR="$(mktemp -d -p "${BROKER_SOCKET_ROOT}" "b.smoke.XXXXXX")"
SOCKET_PATH="${BROKER_STATE_DIR}/c"
BROKER_STDOUT="${AGENT_DIR}/command_broker.smoke.stdout.txt"
BROKER_STDERR="${AGENT_DIR}/command_broker.smoke.stderr.txt"
ACTIVITY_EVENTS_PATH="${AGENT_DIR}/activity_ir_events.jsonl"
mkdir -p "${AGENT_DIR}" "${BROKER_STATE_DIR}"
rm -f "${SOCKET_PATH}"

BROKER_PID=""
cleanup() {
  if [[ -n "${BROKER_PID}" ]] && kill -0 "${BROKER_PID}" 2>/dev/null; then
    kill "${BROKER_PID}" 2>/dev/null || true
    wait "${BROKER_PID}" 2>/dev/null || true
  fi
  rm -rf "${BROKER_STATE_DIR}"
}
trap cleanup EXIT

"${PYTHON_BIN}" -m kernel_bench_experiment_agents.command_broker \
  --socket "${SOCKET_PATH}" \
  --workspace "${WORKSPACE}" \
  --run-name "${RUN_NAME}" \
  --level "${LEVEL}" \
  --problem-id "${PROBLEM_ID}" \
  --dataset-src "${DATASET_SRC}" \
  --kernelbench-root "${KERNELBENCH_ROOT}" \
  --num-gpu-slots 1 \
  --precision "${PRECISION}" \
  --tool "${TOOL}" \
  --activity-events-path "${ACTIVITY_EVENTS_PATH}" \
  >"${BROKER_STDOUT}" \
  2>"${BROKER_STDERR}" &
BROKER_PID=$!

attempts=200
while (( attempts > 0 )); do
  if [[ -S "${SOCKET_PATH}" ]]; then
    break
  fi
  if ! kill -0 "${BROKER_PID}" 2>/dev/null; then
    echo "command broker exited before creating its socket" >&2
    [[ -s "${BROKER_STDERR}" ]] && cat "${BROKER_STDERR}" >&2
    exit 1
  fi
  attempts=$((attempts - 1))
  sleep 0.1
done

if [[ ! -S "${SOCKET_PATH}" ]]; then
  echo "timed out waiting for command broker socket" >&2
  [[ -s "${BROKER_STDERR}" ]] && cat "${BROKER_STDERR}" >&2
  exit 1
fi

PYTHON="${PYTHON_BIN}" KBH_COMMAND_SOCKET="${SOCKET_PATH}" "${WORKSPACE}/bin/goal_status.sh"
if ! PYTHON="${PYTHON_BIN}" KBH_COMMAND_SOCKET="${SOCKET_PATH}" "${WORKSPACE}/bin/best_result.sh"; then
  echo "best_result returned no attempts, as expected for a fresh broker smoke." >&2
fi

KBH_COMMAND_SOCKET="${SOCKET_PATH}" "${PYTHON_BIN}" - <<'PY'
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


EXPECTED_TOOLS = {
    "run_candidate",
    "profile_ncu",
    "goal_status",
    "research_nvidia_docs",
    "best_result",
    "complete_problem",
}


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kernel_bench_experiment_agents.command_mcp"],
        env={"KBH_COMMAND_SOCKET": os.environ["KBH_COMMAND_SOCKET"]},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            missing = sorted(EXPECTED_TOOLS - tool_names)
            if missing:
                raise RuntimeError(f"command MCP missing tool(s): {', '.join(missing)}")
            goal = await session.call_tool("goal_status", arguments={})
            payload = {
                "command_mcp_tools": sorted(tool_names),
                "goal_status_preview": goal.content[0].text[:200] if goal.content else "",
                "goal_status_error": bool(goal.isError),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))


asyncio.run(main())
PY
