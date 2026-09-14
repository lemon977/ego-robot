#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$TOOL_DIR/config.json}"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffmpeg/ffprobe not found on PATH." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -x "$TOOL_DIR/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$TOOL_DIR/.venv"
fi
"$TOOL_DIR/.venv/bin/python" -m pip install --disable-pip-version-check -r "$TOOL_DIR/requirements-lock.txt"
exec "$TOOL_DIR/.venv/bin/python" "$TOOL_DIR/annotator.py" run --config "$CONFIG_PATH"
