"""Direct CPU child with its own v6 code receipt gate; no v4 receipt reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.cli_v6 import exact78_cards004_submit_candidate_v1 as driver
from chaoyang.cli_v3 import exact78_newraw_cpu_submit_phase_exact_v2 as phase_exact
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def submit_one(session: Path, expected_sha: str, *, check_only: bool = False) -> dict:
    source = phase_exact.prior.legacy_submit._raw_session_path(session)
    if source != driver.SESSION or expected_sha != driver.SOURCE_SHA:
        raise ColdStartInputError("CLI_V6_CPU_CARDS004_ONLY")
    decision = driver.plan_one(source, expected_sha)
    if check_only or decision["status"] != "READY":
        return decision
    if decision["node"] != "CPU_ADMISSION":
        raise ColdStartInputError("CLI_V6_CPU_PHASE_ONLY")
    _, cutoff = driver.accepted_code_for_execution(source, expected_sha)
    if budget := driver._budget("CPU_ADMISSION", cutoff):
        return {**decision, **budget}
    verified, packet = phase_exact.plan_one(source, expected_sha)
    if (verified.get("status") != "READY"
            or verified.get("task_id") != decision.get("task_id")):
        raise ColdStartInputError("CLI_V6_CPU_PACKET_DECISION_DRIFT")
    phase_exact.producer.register(packet)
    return phase_exact.producer.run_registered_child(packet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v6.exact78_cards004_cpu_child_v1")
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
