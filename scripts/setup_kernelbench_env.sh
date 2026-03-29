#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-uv}"

if [[ -z "${KERNELBENCH_ROOT:-}" ]]; then
  echo "KERNELBENCH_ROOT must point to the official KernelBench checkout." >&2
  exit 1
fi

cd "${KERNELBENCH_ROOT}"

case "${MODE}" in
  uv)
    uv sync --extra gpu
    echo
    echo "KernelBench environment created with uv."
    echo "Runtime interpreter:"
    echo "  ${KERNELBENCH_ROOT}/.venv/bin/python"
    ;;
  pip)
    PYTHON_BIN="${PYTHON_BIN:-python3.10}"
    "${PYTHON_BIN}" -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -e ".[gpu]"
    echo
    echo "KernelBench environment created with pip editable install."
    echo "Runtime interpreter:"
    echo "  ${KERNELBENCH_ROOT}/.venv/bin/python"
    ;;
  *)
    echo "Unsupported mode: ${MODE}. Use 'uv' or 'pip'." >&2
    exit 1
    ;;
esac
