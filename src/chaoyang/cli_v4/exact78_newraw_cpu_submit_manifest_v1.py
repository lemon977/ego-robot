"""Phase-exact CPU child for the manifest-fenced CLI-v4 candidate.

The old producer is reused unchanged. This adapter cannot register without
the separate v4 code-acceptance receipt, even when called directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chaoyang.cli_v3 import exact78_newraw_cpu_submit_phase_exact_v2 as phase_exact
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def plan_one(session: Path, expected_input_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from chaoyang.cli_v4.exact78_raw_to_robot_manifest_candidate_v1 import _entry

    source = phase_exact.prior.legacy_submit._raw_session_path(session)
    _entry(source, expected_input_sha256)
    return phase_exact.plan_one(source, expected_input_sha256)


def submit_one(session: Path, expected_input_sha256: str, *, check_only: bool = False) -> dict[str, Any]:
    decision, packet = plan_one(session, expected_input_sha256)
    if check_only:
        return decision
    from chaoyang.cli_v4.exact78_raw_to_robot_manifest_candidate_v1 import accepted_code_for_execution

    accepted_code_for_execution(Path(packet["inputs"]["source_root"]), expected_input_sha256)
    if decision["status"] != "READY":
        return decision
    if budget := phase_exact.prior._budget("CPU_ADMISSION"):
        return {**decision, **budget}
    phase_exact.producer.register(packet)
    return phase_exact.producer.run_registered_child(packet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v4.exact78_newraw_cpu_submit_manifest_v1")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = submit_one(args.session, args.expected_input_sha256,
                            check_only=args.check_only)
    except (ColdStartInputError, RuntimeError) as error:
        print(json.dumps({"status": "REJECTED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
