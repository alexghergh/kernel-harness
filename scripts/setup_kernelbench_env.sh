#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-uv}"

if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the official KernelBench checkout." >&2
  exit 1
fi

cd "${KERNELBENCH_ROOT}"

case "${MODE}" in
  uv)
    uv sync --extra gpu
    "${KERNELBENCH_ROOT}/.venv/bin/python" -m pip install -e "${HARNESS_ROOT}"
    echo
    echo "KernelBench environment created with uv."
    echo "Harness installed into the same environment."
    echo "Runtime interpreter:"
    echo "  ${KERNELBENCH_ROOT}/.venv/bin/python"
    ;;
  pip)
    PYTHON_BIN="${PYTHON_BIN:-python3.10}"
    "${PYTHON_BIN}" -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -e ".[gpu]"
    python -m pip install -e "${HARNESS_ROOT}"
    echo
    echo "KernelBench environment created with pip editable install."
    echo "Harness installed into the same environment."
    echo "Runtime interpreter:"
    echo "  ${KERNELBENCH_ROOT}/.venv/bin/python"
    ;;
  *)
    echo "Unsupported mode: ${MODE}. Use 'uv' or 'pip'." >&2
    exit 1
    ;;
esac
