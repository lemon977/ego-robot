#!/usr/bin/env bash
# Retired: the takeover plan is sealed. Refuse before touching any project file.
echo "LEGACY_TAKEOVER_HELPER_RETIRED: watchdog; register a new finite Task Packet" >&2
exit 64
# Historical implementation below is evidence, not a launch instruction.
set -euo pipefail
cd /path/to/chaoyang
W=_run/caches/takeover_autopilot
mkdir -p "$W"
if [ -f "$W/watchdog.pid" ] && grep -q autopilot_watchdog.sh "/proc/$(cat "$W/watchdog.pid")/cmdline" 2>/dev/null; then
  if [ "$(cat "$W/watchdog.pid")" != "$$" ]; then
    echo "another watchdog holds pid $(cat "$W/watchdog.pid")"
    exit 0
  fi
fi
echo $$ > "$W/watchdog.pid"
while true; do
  date -Iseconds > "$W/HEARTBEAT"
  if ! { [ -f "$W/autopilot.pid" ] && grep -q autopilot.py "/proc/$(cat "$W/autopilot.pid")/cmdline" 2>/dev/null; }; then
    echo "$(date -Iseconds) autopilot missing; starting"
    bash scripts/takeover/autopilot_start.sh || echo "$(date -Iseconds) start failed"
  fi
  sleep 120
done
