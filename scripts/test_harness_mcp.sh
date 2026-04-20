#!/usr/bin/env bash
# Prepare one problem workspace and smoke-test the real harness MCP server without Codex/Claude.
#
# Run from the harness repo root with the same launcher env you would use for run_agent_problem.sh.
#
# Required environment:
#   TOOL=codex|claude
#   KERNELBENCH_ROOT=/path/to/KernelBench
#   HARDWARE_NAME=<timings-subdir name>
#
# Common overrides:
#   DATA_ROOT=/path/for/archive-and-state   (defaults to ./ from the launch directory)
#   RUN_NAME=kernelbench-codex-h100-v4
#   LEVEL=1
#   PROBLEM_ID=1
#   MODEL=gpt-5.4|claude-opus-4-7
#   TIME_BUDGET_MINUTES=180
#   PRECISION=bf16
#   KERNELBENCH_TIMINGS_DIR=/path/to/KernelBench/results/timing/<hardware>  # optional override
set -euo pipefail

if [[ ! -f "./pyproject.toml" || ! -d "./src/kernel_bench_experiment_agents" ]]; then
  echo "Run scripts/test_harness_mcp.sh from the harness repo root." >&2
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
if [[ "${TOOL}" == "claude" ]]; then
  MODEL="${MODEL:-claude-opus-4-7}"
fi

if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the official KernelBench checkout." >&2
  exit 1
fi
if [[ -z "${HARDWARE_NAME}" ]]; then
  echo "HARDWARE_NAME must name the KernelBench timings subdirectory to use." >&2
  exit 1
fi

PREP_OUTPUT="$({
  kbharness prepare-problem-workspace \
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
  PREP_OUTPUT="${PREP_OUTPUT}" python - <<'PY'
import json
import os
print(json.loads(os.environ["PREP_OUTPUT"])["workspace"])
PY
})"

AGENT_DIR="${DATA_ROOT}/archive/${RUN_NAME}/level_${LEVEL}/problem_${PROBLEM_ID}/agent"
mkdir -p "${AGENT_DIR}"

export KBH_WORKSPACE="${WORKSPACE}"
export KBH_CLIENT_TOOL="${TOOL}"
export KBH_MCP_EVENTS_PATH="${AGENT_DIR}/mcp_ir_events.jsonl"

python - <<'PY'
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "kernel_bench_experiment_agents.mcp"],
    env={
        "DATA_ROOT": os.environ["DATA_ROOT"],
        "KBH_WORKSPACE": os.environ["KBH_WORKSPACE"],
        "KBH_CLIENT_TOOL": os.environ["KBH_CLIENT_TOOL"],
        "KBH_MCP_EVENTS_PATH": os.environ["KBH_MCP_EVENTS_PATH"],
    },
)

async def main() -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            overview = await session.call_tool("workspace_overview", arguments={})
            agents = await session.call_tool("read_workspace_file", arguments={"path": "AGENTS.md"})
            payload = {
                "tools": [tool.name for tool in tools.tools],
                "resources": [str(resource.uri) for resource in resources.resources],
                "overview": overview.content[0].text if overview.content else "",
                "agents_preview": agents.content[0].text[:200] if agents.content else "",
            }
            print(json.dumps(payload, indent=2, sort_keys=True))

asyncio.run(main())
PY
