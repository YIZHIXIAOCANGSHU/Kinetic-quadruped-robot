#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

source "$ROOT_DIR/.venv/bin/activate"

export PYTHONNOUSERSITE=1
export MPLBACKEND=TkAgg

echo "Activated venv: $VIRTUAL_ENV"
echo "Python: $(command -v python)"
