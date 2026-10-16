"""Phase-exact CPU admission successor; no new producer or task queue.

Only an ``EXACT78_NEWRAW_CPU_ADMISSION`` row may attach or conflict with the
new CPU admission attempt. Code-acceptance and downstream rows are separate
phases and never masquerade as prior CPU work.
"""

from __future__ import annotations

import argparse
import hmac
import json
from pathlib import Path
import re
from typing import Any

from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as prior
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def plan_one(session: Path, expected_input_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = prior.legacy_submit._raw_session_path(session)
    if source in prior.legacy_submit._frozen_t0_selected_paths():
        raise ColdStartInputError("T0_FROZEN_SESSION_NOT_NEWRAW_CANDIDATE")
    prepared = producer.preflight(source)
    packet = producer.packet_value(prepared)
    actual = packet["inputs"]["submission_input_sha256"]
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_input_sha256)
            or not hmac.compare_digest(actual, expected_input_sha256)):
        raise ColdStartInputError("PREFROZEN_SUBMISSION_INPUT_SHA256_DRIFT")
    decision = prior._node(packet, prior.PHASES[0], prior.SCHEMAS[0])
    if decision["status"] == "READY":
        producer._assert_cli_admission_ready(packet)
    return decision, packet


def submit_one(session: Path, expected_input_sha256: str, *, check_only: bool = False) -> dict[str, Any]:
    decision, packet = plan_one(session, expected_input_sha256)
    if check_only or decision["status"] != "READY":
        return decision
    # Direct invocation of this adapter must not bypass the v3 code receipt.
    from chaoyang.cli_v3.exact78_raw_to_robot_candidate_v2 import accepted_code_for_execution

    accepted_code_for_execution(Path(packet["inputs"]["source_root"]), expected_input_sha256)
    if budget := prior._budget("CPU_ADMISSION"):
        return {**decision, **budget}
    producer.register(packet)
    return producer.run_registered_child(packet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v3.exact78_newraw_cpu_submit_phase_exact_v2")
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
