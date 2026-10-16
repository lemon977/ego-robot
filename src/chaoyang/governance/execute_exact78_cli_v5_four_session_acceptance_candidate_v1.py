"""Read-only v5 code-acceptance preparation; execution admission fails closed.

The v5 CLI currently exposes only ``--check-only``.  This module can assemble
an immutable acceptance input preview, but cannot publish a PASSED acceptance
receipt or register an algorithm task until a real, separately reviewed submit
path exists.  No function in this module writes files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chaoyang.cli_v5 import exact78_four_session_manifest_candidate_v1 as cli
from chaoyang.cli_v5 import exact78_newraw_cpu_submit_manifest_v1 as cpu
from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    canonical_bytes, load_json, validate_artifact_ref,
)
from chaoyang.governance.execute_exact78_cli_v4_manifest_acceptance_candidate_v1 import (
    _published_operations,
)


TASK_ID = "exact78_newraw_cli_v5_four_session_acceptance_v1"
RESULT_SCHEMA = "exact78-newraw-cli-v5-four-session-acceptance-result-v1"
RECEIPT_SCHEMA = "exact78-newraw-cli-v5-four-session-acceptance-receipt-v1"
CLI_FILES = tuple(f"src/chaoyang/cli_v5/{name}" for name in (
    "__init__.py", "__main__.py", "exact78_four_session_manifest_candidate_v1.py",
    "exact78_newraw_cpu_submit_manifest_v1.py"))
TEST_FILE = "tests/unit/test_exact78_cli_v5_four_session_candidate_v1.py"
ACCEPTANCE_TEST_FILE = "tests/unit/test_exact78_cli_v5_four_session_acceptance_candidate_v1.py"
SCRIPT = Path(__file__).resolve()
EXPECTED_IDS = (
    "get_potato_chips_0901_001", "play_cards_0902_002",
    "play_cards_0915_003", "play_cards_0915_004",
)
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})


def _no_real_submit_path() -> None:
    """Require an executable path before any acceptance could be certified."""
    raise RuntimeError("V5_REAL_SUBMIT_PATH_ABSENT")


def _source_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, manifest_ref = cli.frozen_manifest()
    rows = manifest["sessions"]
    if tuple(row["session_id"] for row in rows) != EXPECTED_IDS:
        raise RuntimeError("V5_FOUR_ROW_SET_DRIFT")
    old, _ = cli.v4.frozen_manifest()
    if rows[:2] != old["sessions"]:
        raise RuntimeError("V5_TRACKER_ROW_DRIFT")
    fresh = rows[3]
    if set(fresh["source_files"]) != cli.RAW_ROLES:
        raise RuntimeError("V5_004_ROLE_SET_DRIFT")
    for role, ref in fresh["source_files"].items():
        expected = Path(fresh["source_root"]) / role
        if (ref.get("path") != str(expected) or expected.is_symlink()
                or validate_artifact_ref(ref)):
            raise RuntimeError("V5_004_SOURCE_REF_DRIFT:" + role)
    return rows, manifest_ref


def _sealed_003(row: dict[str, Any]) -> dict[str, Any]:
    decision = cli._sealed_003(row)
    if (decision.get("sealed_attachment") is not True
            or decision.get("status") != "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS"
            or decision.get("visual_qualified") is not False
            or decision.get("training_eligible") is not False):
        raise RuntimeError("V5_003_ATTACHMENT_DRIFT")
    return decision


def _fresh_004(row: dict[str, Any]) -> dict[str, Any]:
    decision = cpu.check_one(Path(row["source_root"]), row["submission_input_sha256"])
    if (decision.get("status") != "READY" or decision.get("node") != "CPU_ADMISSION"
            or decision.get("registration_started") is not False
            or decision.get("model_calls") != 0
            or decision.get("code_acceptance") != "NOT_SEALED_CODE_PREP_ONLY"):
        raise RuntimeError("V5_004_CPU_CHECK_ONLY_NOT_READY")
    return decision


def inspect() -> dict[str, Any]:
    """Inspect identity and planning, with no registration or artifact writes."""
    rows, manifest_ref = _source_rows()
    sealed = _sealed_003(rows[2])
    state = load_json(TASK_STATE_PATH)
    active = [item.get("task_id") for item in state.get("tasks", [])
              if item.get("status") in ACTIVE]
    active_writer = bool(active or state.get("next_task"))
    fresh: dict[str, Any]
    if active_writer:
        fresh = {"status": "ACTIVE_DIFFERENT_WRITER_CONFLICT", "node": "CPU_ADMISSION",
                 "registration_started": False, "model_calls": 0}
    else:
        fresh = _fresh_004(rows[3])
    code = {name: artifact_ref(REPO_ROOT / name) for name in CLI_FILES}
    inherited = {name: artifact_ref(REPO_ROOT / name) for name in (
        *cli.v4.CLI_CODE, *cli.v4.v3.CLI_CODE, *cli.v4.prior.CLI_CODE)}
    tests = {name: artifact_ref(REPO_ROOT / name) for name in
             (TEST_FILE, ACCEPTANCE_TEST_FILE)}
    runtime = artifact_ref(SCRIPT)
    routes = _published_operations()
    inputs = {"manifest": manifest_ref, "source_rows": [{
        "session_id": row["session_id"], "source_root": row["source_root"],
        "submission_input_sha256": row["submission_input_sha256"]} for row in rows],
        "source_files_004": rows[3]["source_files"],
        "sealed_robot_terminal_003": rows[2]["sealed_robot_terminal"],
        "sealed_003": sealed, "fresh_004": fresh, "cli_code": code,
        "inherited_code": inherited, "test_code": tests,
        "runtime_code": runtime, "published_operations": routes,
        "algorithm_contract": artifact_ref(ALGORITHM_CONTRACT_PATH)}
    signature = hashlib.sha256(canonical_bytes({"schema": RESULT_SCHEMA,
        "task_id": TASK_ID, "inputs": inputs,
        "budget": {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
                   "attempts_max": 1, "wall_seconds_max": 600}})).hexdigest()
    return {"schema_version": RESULT_SCHEMA, "task_id": TASK_ID,
            "status": "BLOCKED_REAL_SUBMIT_PATH_ABSENT" if not active_writer else
                      "BLOCKED_ACTIVE_WRITER_AND_REAL_SUBMIT_PATH_ABSENT",
            "claim_limit": "Planning integrity only; no execution, visual, control or training admission",
            "run_signature_preview": signature, "inputs": inputs,
            "registration_started": False, "model_calls": 0,
            "batch_execution_authorized": False,
            "code_acceptance_receipt_published": False,
            "active_task_ids": active}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    if args.register_and_run:
        _no_real_submit_path()
    print(json.dumps(inspect(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
