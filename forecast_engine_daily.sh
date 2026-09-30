#!/bin/bash
# Daily forecast engine runner
set -euo pipefail

BASE_DIR="/home/cmswh/Desktop/python/forecast_engine"
PYTHON="${BASE_DIR}/.venv/bin/python"
LOG_DIR="${BASE_DIR}/logs"
LOG_FILE="${LOG_DIR}/forecast_engine_daily.log"
LOCK_FILE="/tmp/forecast_engine_daily.lock"

mkdir -p "${LOG_DIR}"
cd "${BASE_DIR}"

# Prevent overlapping Hermes-triggered runs from racing on the same forecast period.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "=== Forecast Engine Daily Run skipped: another run is still active at $(date) ==="
    exit 75
fi

# Keep a persistent execution trail for Hermes-scheduled runs.
exec >>"${LOG_FILE}" 2>&1

echo "=== Starting Forecast Engine Daily Run at $(date) ==="
echo "Working directory: ${BASE_DIR}"
echo "Python: ${PYTHON}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: virtualenv Python not found or not executable: ${PYTHON}"
    exit 1
fi

"${PYTHON}" main.py

echo "=== Forecast Engine Daily Run completed successfully at $(date) ==="
