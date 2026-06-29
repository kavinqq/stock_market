#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/market_report.log"
TAG="# stock_market_daily_report"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/data"

CRON_LINE="30 18 * * 1-5 cd \"$ROOT\" && \"$PYTHON_BIN\" \"$ROOT/daily_market_report.py\" >> \"$LOG_FILE\" 2>&1 $TAG"

(
  crontab -l 2>/dev/null | grep -v "$TAG" || true
  echo "$CRON_LINE"
) | crontab -

echo "Installed daily market report cron:"
echo "$CRON_LINE"
echo "Log file: $LOG_FILE"
