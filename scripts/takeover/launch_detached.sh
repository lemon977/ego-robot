#!/usr/bin/env bash
# Launch one takeover task spec detached from the SSH session (no tmux on this host).
# Usage: bash scripts/takeover/launch_detached.sh docs/plans/EXACT78_TAKEOVER_20261004/specs/<task_id>.json
set -euo pipefail
ROOT=/path/to/chaoyang
SPEC=$(realpath "$1")
cd "$ROOT"
TASK_ID=$(basename "$SPEC" .json)
LOGDIR="$ROOT/_run/caches/takeover_launch"
mkdir -p "$LOGDIR/tmp"
LOG="$LOGDIR/$TASK_ID.log"
[ -e "$LOG" ] && { echo "launch log exists: $LOG (task ids are single-use)"; exit 1; }
export PYTHONPATH="$ROOT/src" TMPDIR="$LOGDIR/tmp"
setsid nohup /usr/local/bin/python -B -m chaoyang.governance.execute_exact78_takeover_v1 \
  --execute "$SPEC" > "$LOG" 2>&1 < /dev/null &
echo "launched task=$TASK_ID pid=$! log=$LOG"
