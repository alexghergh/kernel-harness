#!/usr/bin/env bash

resolve_repo_python() {
  local repo_root="$1"
  local bootstrap_hint="${2:-./kb setup}"
  local configured_python_file="${repo_root}/.kb-python"
  local python_bin="${PYTHON:-}"

  if [[ -z "${python_bin}" && -f "${configured_python_file}" ]]; then
    python_bin="$(<"${configured_python_file}")"
  fi

  if [[ -z "${python_bin}" ]]; then
    python_bin="python"
  fi

  if [[ "${python_bin}" == */* ]]; then
    if [[ ! -x "${python_bin}" ]]; then
      echo "Configured Python is not executable: ${python_bin}" >&2
      echo "Run ${bootstrap_hint} again, or set PYTHON=/path/to/python." >&2
      exit 1
    fi
  elif ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python interpreter is not on PATH: ${python_bin}" >&2
    echo "Run ${bootstrap_hint} again, or set PYTHON=/path/to/python." >&2
    exit 1
  fi

  printf '%s\n' "${python_bin}"
}

default_run_name() {
  local tool="$1"
  printf 'kernelbench-%s-%s-%s\n' "${tool}" "$(date -u '+%Y%m%dT%H%M%SZ')" "$$"
}

resolve_repo_landrun() {
  local repo_root="$1"
  local bootstrap_hint="${2:-./kb setup}"
  local landrun_bin="${LANDRUN:-}"

  if [[ -z "${landrun_bin}" ]]; then
    landrun_bin="${repo_root}/third_party/bin/landrun"
  fi

  if [[ ! -x "${landrun_bin}" ]]; then
    echo "Landrun binary is not executable: ${landrun_bin}" >&2
    echo "Run ${bootstrap_hint} to build the vendored third_party/landrun binary, or set LANDRUN=/path/to/landrun." >&2
    exit 1
  fi

  printf '%s\n' "${landrun_bin}"
}
