"""Read-only, phase-exact CPU admission adapter for the fresh 0915 VST row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.cli_v5.exact78_four_session_manifest_candidate_v1 import _entry, plan_one
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def check_one(session: Path, expected_input_sha256: str) -> dict:
    row, _ = _entry(session.resolve(strict=True), expected_input_sha256)
    if row["session_id"] != "play_cards_0915_004":
        raise ColdStartInputError("CLI_V5_CPU_FRESH_004_ONLY")
    decision = plan_one(session, expected_input_sha256)
    if decision.get("node") != "CPU_ADMISSION":
        raise ColdStartInputError("CLI_V5_CPU_PHASE_DRIFT")
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v5.exact78_newraw_cpu_submit_manifest_v1")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--check-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = check_one(args.session, args.expected_input_sha256)
    except (ColdStartInputError, RuntimeError) as error:
        print(json.dumps({"status": "REJECTED", "reason": str(error),
                          "registration_started": False, "model_calls": 0}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
