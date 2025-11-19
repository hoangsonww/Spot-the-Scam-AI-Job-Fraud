#!/usr/bin/env bash
set -euo pipefail
PID="${1:-}" 
LOG_FILE="${2:-training.log}"
if [[ -z "$PID" ]]; then
  PID=$(pgrep -f "spot_scam\.pipeline\.train" | head -n 1 || true)
fi
if [[ -z "$PID" ]]; then
  echo "No running spot_scam.pipeline.train process found." >&2
  exit 1
fi
if [[ ! -f "$LOG_FILE" ]]; then
  LOG_FILE=""
fi
while kill -0 "$PID" >/dev/null 2>&1; do
  clear
  date
  ps -p "$PID" -o pid,etimes,pcpu,pmem,vsz,rss,command
  if [[ -n "$LOG_FILE" ]]; then
    echo
    echo "Last 20 log lines (from $LOG_FILE):"
    tail -n 20 "$LOG_FILE"
  else
    echo
    echo "No log file provided/found. Run with: scripts/monitor_training.sh <pid> <logfile>"
  fi
  sleep 5
done
