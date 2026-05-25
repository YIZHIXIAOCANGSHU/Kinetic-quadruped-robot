#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/activate_go2_venv.sh"
export GO2_ENABLE_LIDAR="${GO2_ENABLE_LIDAR:-0}"
export GO2_DEPTH_OBSTACLE="${GO2_DEPTH_OBSTACLE:-1}"

exec python3 "$ROOT_DIR/simulate_go2.py"
