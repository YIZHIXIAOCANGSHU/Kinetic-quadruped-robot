#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"

if [ ! -d "$ROOT_DIR/.venv" ]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN not found. Install Python 3.10 or set PYTHON_BIN." >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "Creating virtual environment at $ROOT_DIR/.venv"
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Error: virtual environment activation script not found at $VENV_ACTIVATE" >&2
  return 1 2>/dev/null || exit 1
fi

if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

source "$VENV_ACTIVATE"

export PYTHONNOUSERSITE=1
export MPLBACKEND=TkAgg
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install --upgrade pip setuptools wheel
if [ -f "$REQUIREMENTS_FILE" ]; then
  python -m pip install -r "$REQUIREMENTS_FILE"
fi
python -m pip install -e "$ROOT_DIR" --no-deps

echo "Activated venv: $VIRTUAL_ENV"
echo "Python: $(command -v python)"
