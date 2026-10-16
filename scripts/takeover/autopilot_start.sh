#!/usr/bin/env bash
# Start the takeover autopilot unless it is already running; returns at once (safe over ssh).
# Usage: bash scripts/takeover/autopilot_start.sh
set -euo pipefail
echo "LEGACY_AUTOPILOT_RETIRED: use docs/current/DUAL_BRANCH_BASELINE_V1_ZH.md and a new finite Task Packet" >&2
exit 64
# Historical body retained for receipt lookup; no longer an execution entry.
cd /path/to/chaoyang
W=_run/caches/takeover_autopilot
mkdir -p "$W"
if [ -f "$W/autopilot.pid" ] && grep -q autopilot.py "/proc/$(cat "$W/autopilot.pid")/cmdline" 2>/dev/null; then
  echo "already running pid $(cat "$W/autopilot.pid")"
  exit 0
fi
setsid nohup /usr/local/bin/python -B scripts/takeover/autopilot.py >> "$W/autopilot.out" 2>&1 < /dev/null &
sleep 3
echo "started pid $(cat "$W/autopilot.pid")"
