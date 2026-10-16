#!/usr/bin/env bash
set -euo pipefail
ICT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ICT_PROJECT_ROOT}"
export PYTHONPATH="${ICT_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/humanego_ict/preflight_pick_place.sh <prepared-paired-manifest.json>' >&2
  echo 'The two NAS source directories are not prepared training manifests.' >&2
  exit 2
fi
exec "${ICT_PYTHON:-python}" -B -m chaoyang.humanego_ict.cli preflight --manifest "$1"
