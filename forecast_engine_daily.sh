#!/bin/bash
# Daily forecast engine runner
set -euo pipefail

cd /home/cmswh/Desktop/python/forecast_engine

# Activate virtual environment
source .venv/bin/activate

# Run the forecast engine
echo "=== Starting Forecast Engine Daily Run at $(date) ==="
python main.py

echo "=== Forecast Engine Daily Run completed at $(date) ==="