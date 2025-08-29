#!/bin/bash

set -euo pipefail

# Navigate to repo root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="output.log"

echo "🚀 Starting E2E-DRO (logging to $LOG_FILE)"
echo "Timestamp: $(date)" >> "$LOG_FILE"

# Start unbuffered run in background, capture PID
nohup python -u main.py > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > .run.pid

echo "✅ Started with PID $PID"
echo "📄 Log: $LOG_FILE"
echo "🧵 Tail follows (Ctrl+C to stop tail; job continues running)"

# Follow new lines only
tail -n 0 -f "$LOG_FILE"


