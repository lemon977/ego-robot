"""Finite CPU-only code acceptance for the versioned Exact78 submit candidate.

This is a governance proof, not an algorithm runner.  ``--check-only`` does
not register or write.  The optional registered run only seals code, source,
route and read-only CLI preflight evidence; it never starts HaWoR or Robot.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT,
    TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes, load_json,
    now_iso, process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "exact78_newraw_cli_v2_acceptance_play_cards_0902_002_v1"
PHASE = "EXACT78_NEWRAW_CLI_V2_CODE_ACCEPTANCE"
SESSION_ID = "play_cards_0902_002"
SESSION = (Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902") /
           "playing_cards" / SESSION_ID)
EXPECTED_INPUT_SHA256 = "9e9de7fe6cc21209af6d9c96c1f6df356855d26c86b1b9227930a5a854223f63"
EXPECTED_RAW_SHA256 = "2c42380141cceafb4a6f4105b01fb0c52ad6e123e7c4175c85d846b7b869e132"
EXPECTED_MANIFEST_SHA256 = "d8de7ab150cff75cab6302a7b92c30943da1c7b487026c434c69e1c9c1710b20"
CLI_PATHS = (
    "src/chaoyang/cli_v2/__init__.py",
    "src/chaoyang/cli_v2/__main__.py",
    "src/chaoyang/cli_v2/exact78_raw_to_robot_candidate.py",
)
TEST_PATH = "tests/unit/test_exact78_newraw_submit_dag_candidate_v1.py"
ACCEPTANCE_TEST_PATH = "tests/unit/test_exact78_cli_v2_acceptance_candidate_v1.py"
OPERATIONS = {
    "exact78_newraw_cpu_admission_candidate_v1": "run_exact78_newraw_cpu_admission_candidate_v1",
    "exact78_newraw_hawor_gpu_candidate_v4": "run_exact78_newraw_hawor_gpu_candidate_v4",
    "exact78_newraw_robot_cpu_candidate_v1": "run_exact78_newraw_robot_cpu_candidate_v1",
}
SCRIPT = Path(__file__).resolve()
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
RESULT_SCHEMA = "exact78-newraw-cli-v2-acceptance-result-v1"
RECEIPT_SCHEMA = "exact78-newraw-cli-v2-acceptance-receipt-v1"


def _idle() -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Check the receipt-bound ledger and the exact fresh write targets."""
    receipt, state, index = load_json(RECEIPT_PATH), load_json(TASK_STATE_PATH), load_json(INDEX)
    revision = int(receipt["governance_revision"])
    for key, path in (("task_state", TASK_STATE_PATH),
                      ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if (not isinstance(ref, dict) or ref.get("path") != str(path)
                or validate_artifact_ref(ref)):
            raise RuntimeError(f"GOVERNANCE_RECEIPT_{key.upper()}_DRIFT")
    if (state.get("next_task") or any(row.get("status") in ACTIVE for row in state["tasks"])
            or any(row.get("execution_allowed") is True for row in index["task_packets"])):
        raise RuntimeError("ACTIVE_TASK_OR_WRITER_CONFLICT")
    if (any(row.get("task_id") == TASK_ID for row in state["tasks"])
            or any(row.get("task_id") == TASK_ID for row in index["task_packets"])
            or any(path.exists() for path in (ATTEMPT, PACKET, RECEIPT))):
        raise RuntimeError("ACCEPTANCE_TASK_ALREADY_USED")
    return revision, state, index


def _published_operations() -> dict[str, dict[str, Any]]:
    contract = load_json(ALGORITHM_CONTRACT_PATH)
    routes = contract.get("special_status_contracts")
    if not isinstance(routes, dict):
        raise RuntimeError("ALGORITHM_CONTRACT_ROUTES_ABSENT")
    accepted: dict[str, dict[str, Any]] = {}
    for key, operation in OPERATIONS.items():
        route = routes.get(key)
        path = REPO_ROOT / "src/chaoyang/ops" / f"{operation}.py"
        if (not isinstance(route, dict) or route.get("candidate_operation") != operation
                or route.get("task_packet_required") is not True
                or route.get("current_runnable_operation") is not None
                or not isinstance(route.get("code_closure"), list)):
            raise RuntimeError(f"PUBLISHED_ROUTE_ABSENT_OR_DRIFT:{key}")
        expected = artifact_ref(path)
        if route["code_closure"].count(expected) != 1:
            raise RuntimeError(f"PUBLISHED_OPERATION_SHA_DRIFT:{key}")
        accepted[key] = {"candidate_operation": operation, "operation_ref": expected}
    return accepted


def _source_evidence() -> dict[str, Any]:
    # The old producer's raw guard is read-only here; no historical action is consumed.
    from chaoyang.governance.execute_exact78_first_submit_cpu_candidate_v1 import preflight

    prepared = preflight(SESSION)
    descriptor = prepared["descriptor"]
    if (descriptor.get("session_id") != SESSION_ID
            or descriptor.get("submission_input_sha256") != EXPECTED_INPUT_SHA256
            or descriptor.get("decoded_frames") != 189
            or descriptor.get("source_kind") != "TRACKER_CLIP_RGB"):
        raise RuntimeError("PREFROZEN_SOURCE_IDENTITY_OR_SHA_DRIFT")
    raw = artifact_ref(SESSION / f"CameraRecord_{SESSION_ID}.mp4")
    manifest = artifact_ref(SESSION / "clip_manifest.json")
    if (raw["sha256"] != EXPECTED_RAW_SHA256
            or manifest["sha256"] != EXPECTED_MANIFEST_SHA256):
        raise RuntimeError("PREFROZEN_RAW_ROLE_SHA_DRIFT")
    return {"source_submission_input_sha256": EXPECTED_INPUT_SHA256,
            "source_session": SESSION_ID, "source_root": str(SESSION),
            "source_files": descriptor["source_files"],
            "raw_video": raw, "raw_manifest": manifest}


def _probe_cli() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli_v2", "submit",
               "exact78-raw-to-robot", "--session", str(SESSION),
               "--expected-input-sha256", EXPECTED_INPUT_SHA256, "--check-only"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
                       XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment,
                               capture_output=True, check=False, timeout=120)
    if completed.returncode:
        raise RuntimeError("CLI_V2_CHECK_ONLY_FAILED:" +
                           completed.stderr.decode("utf-8", "replace")[-400:])
    try:
        result = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("CLI_V2_CHECK_ONLY_NOT_JSON") from exc
    if (not isinstance(result, dict) or result.get("status") != "READY"
            or result.get("node") != "CPU_ADMISSION"
            or result.get("source_session") != SESSION_ID
            or result.get("submission_input_sha256") != EXPECTED_INPUT_SHA256):
        raise RuntimeError("CLI_V2_CHECK_ONLY_NOT_FRESH_CPU_READY")
    return {"status": "READY", "node": "CPU_ADMISSION",
            "source_session": SESSION_ID,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest()}


def _probe_tests() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "unittest", "discover",
               "-s", "tests/unit", "-p", Path(TEST_PATH).name]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
                       XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment,
                               capture_output=True, check=False, timeout=120)
    summary = completed.stdout + completed.stderr
    match = re.search(rb"Ran (\d+) tests?", summary)
    if completed.returncode or match is None or int(match.group(1)) < 11 or b"OK" not in summary:
        raise RuntimeError("CLI_V2_NEGATIVE_TESTS_NOT_PASS:" +
                           summary.decode("utf-8", "replace")[-400:])
    # unittest's elapsed seconds vary across identical executions; exclude only
    # that duration so the Task Packet signature remains reproducible.
    normalized = re.sub(rb"(Ran \d+ tests? in )\d+(?:\.\d+)?s", rb"\g<1><elapsed>s", summary)
    return {"status": "PASSED", "test_count": int(match.group(1)),
            "output_sha256": hashlib.sha256(normalized).hexdigest()}


def inspect() -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    revision, state, index = _idle()
    cli_code = {name: artifact_ref(REPO_ROOT / name) for name in CLI_PATHS}
    test_code = artifact_ref(REPO_ROOT / TEST_PATH)
    inputs = {**_source_evidence(), "cli_code": cli_code,
              "test_code": test_code,
              "acceptance_test_code": artifact_ref(REPO_ROOT / ACCEPTANCE_TEST_PATH),
              "published_operations": _published_operations(),
              "cli_check_only": _probe_cli(), "negative_tests": _probe_tests()}
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("GOVERNANCE_REVISION_CHANGED_DURING_PROBE")
    for ref in (*cli_code.values(), test_code, inputs["acceptance_test_code"], *(
            item["operation_ref"] for item in inputs["published_operations"].values())):
        if validate_artifact_ref(ref):
            raise RuntimeError("CODE_SHA_CHANGED_DURING_PROBE")
    return revision, state, index, inputs


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    code = [artifact_ref(SCRIPT), artifact_ref(REPO_ROOT / "src/chaoyang/governance/common.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py")]
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m "
               "chaoyang.governance.execute_exact78_cli_v2_acceptance_candidate_v1 "
               "--register-and-run")
    config = {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "render_calls": 0, "training_steps": 0, "session_attempts_max": 1,
              "wall_seconds_max": 600, "calibration_or_absent": "NOT_CONSUMED",
              "weights": "NOT_CONSUMED"}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": code,
        "config": config, "command": command, "schema": RESULT_SCHEMA})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Seal one versioned CLI code and Cards002 raw admission readiness without algorithm calls",
        "phase": PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_V1",
        "execution_mode": "EXACT78_GOVERNANCE_ONLY", "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(minutes=10)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     "src/chaoyang/cli_v2/__init__.py",
                     "src/chaoyang/cli_v2/__main__.py",
                     "src/chaoyang/cli_v2/exact78_raw_to_robot_candidate.py",
                     TEST_PATH, ACCEPTANCE_TEST_PATH,
                     "docs/governance/ALGORITHM_CONTRACT.json"],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)),
                      str(INDEX.relative_to(REPO_ROOT)), "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "PREFROZEN_SOURCE_SHA",
                          "CLI_V2_CHECK_ONLY_READY", "THREE_PUBLISHED_STAGE_ROUTES",
                          "CPU_ONLY", "0916_READ_ONLY", "NO_MODEL_OR_RENDER"],
        "required_outputs": ["RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": 600, "session_attempts_max": 1},
        "stop_conditions": ["CODE_OR_SOURCE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "CHECK_ONLY_NOT_READY", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "CLI code admission only; no model, Robot execution, Clean, visual, product, general or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": code,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("ACCEPTANCE_PACKET_INVALID:" + ";".join(errors))
    return packet


def result_value(packet: dict[str, Any], packet_ref: dict[str, Any],
                 *, status: str = "PASSED", error: str | None = None) -> dict[str, Any]:
    inputs = packet["inputs"]
    return {"schema_version": RESULT_SCHEMA, "task_id": TASK_ID,
            "status": status, "source_session": SESSION_ID,
            "source_submission_input_sha256": inputs["source_submission_input_sha256"],
            "run_signature": packet["run_signature"], "task_packet": packet_ref,
            "cli_code": inputs["cli_code"], "test_code": inputs["test_code"],
            "published_operations": inputs["published_operations"],
            "cli_check_only": inputs["cli_check_only"],
            "negative_tests": inputs["negative_tests"], "error": error,
            "model_calls": 0, "render_calls": 0, "visual_qualified": False,
            "training_eligible": False}


def receipt_value(packet: dict[str, Any], packet_ref: dict[str, Any],
                  result_ref: dict[str, Any], *, status: str = "PASSED") -> dict[str, Any]:
    return {"schema_version": RECEIPT_SCHEMA, "task_id": TASK_ID,
            "status": status, "source_session": SESSION_ID,
            "run_signature": packet["run_signature"],
            "task_packet": packet_ref, "result": result_ref}


def _index_base(index: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in index.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def register(packet: dict[str, Any], revision: int) -> None:
    """Governance-only CAS.  Do not call while another task owns the writer."""
    _idle()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("ACCEPTANCE_REVISION_CHANGED")
    for ref in [*packet["runtime_code"], *packet["inputs"]["cli_code"].values(),
                packet["inputs"]["test_code"], packet["inputs"]["acceptance_test_code"], *(
                    item["operation_ref"] for item in
                    packet["inputs"]["published_operations"].values()),
                *packet["inputs"]["source_files"].values()]:
        if validate_artifact_ref(ref):
            raise RuntimeError("ACCEPTANCE_INPUT_SHA_DRIFT")
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid() or
            writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]):
        raise RuntimeError("ACCEPTANCE_WRITER_IDENTITY_DRIFT")
    packet_ref = _virtual_ref(PACKET, packet)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK_ID, "phase": PHASE, "attempt": 1,
        "status": "RUNNING", "session": SESSION_ID, "updated_at": stamp,
        "heartbeat_at": stamp, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": packet_ref, "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK_ID, "session": SESSION_ID,
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS_NO_GPU",
        "stop_condition": "one immutable code acceptance result or ten-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("ACCEPTANCE_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK_ID,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_V1_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET) != packet_ref:
        raise RuntimeError("ACCEPTANCE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (process_identity(os.getpid())["start_ticks"] != writer["proc_start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or load_json(PACKET) != packet):
        raise RuntimeError("ACCEPTANCE_WRITER_FENCE_DRIFT")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK_ID]
    live = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(live) != 1 or live[0].get("task_id") != TASK_ID
            or live[0].get("packet_sha256") != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("ACCEPTANCE_DOUBLE_WRITER_OR_PACKET_DRIFT")


def finish(packet: dict[str, Any]) -> dict[str, Any]:
    _fence(packet)
    error = None
    try:
        if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
            error = "ACCEPTANCE_DEADLINE"
        else:
            refs = [*packet["runtime_code"], *packet["inputs"]["cli_code"].values(),
                    packet["inputs"]["test_code"], packet["inputs"]["acceptance_test_code"], *(
                        item["operation_ref"] for item in
                        packet["inputs"]["published_operations"].values()),
                    *packet["inputs"]["source_files"].values()]
            if any(validate_artifact_ref(ref) for ref in refs):
                error = "ACCEPTANCE_CODE_OR_SOURCE_SHA_DRIFT"
            elif _published_operations() != packet["inputs"]["published_operations"]:
                error = "ACCEPTANCE_PUBLISHED_OPERATION_DRIFT"
    except Exception as exc:
        error = f"ACCEPTANCE_VERIFICATION_ERROR:{type(exc).__name__}:{exc}"
    status = "PASSED" if error is None else "FAILED_RUNTIME_FINAL"
    packet_ref = artifact_ref(PACKET)
    atomic_json(ATTEMPT / "RESULT.json", result_value(packet, packet_ref,
                                                       status=status, error=error))
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(RECEIPT, receipt_value(packet, packet_ref, result_ref, status=status))
    receipt_ref = artifact_ref(RECEIPT)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    row = next(row for row in state["tasks"] if row.get("task_id") == TASK_ID)
    row.update(status=status, result=result_ref, terminal_receipt=receipt_ref,
               updated_at=now_iso(), heartbeat_at=now_iso(), quality="MACHINE_CODE_ACCEPTANCE_ONLY"
               if status == "PASSED" else "FAILED")
    state["next_task"] = None
    for item in index["task_packets"]:
        if item.get("task_id") == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_V1_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_TERMINAL",
                   expected_revision=int(load_json(RECEIPT_PATH)["governance_revision"]),
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return load_json(ATTEMPT / "RESULT.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    revision, _state, _index, inputs = inspect()
    packet = packet_value(inputs)
    if args.check_only or args.dry_run:
        print(json.dumps({"status": "READY_NOT_REGISTERED", "task_id": TASK_ID,
                          "governance_revision": revision,
                          "run_signature_preview": packet["run_signature"],
                          "source_submission_input_sha256": EXPECTED_INPUT_SHA256,
                          "cli_code": inputs["cli_code"],
                          "test_code": inputs["test_code"],
                          "published_operations": inputs["published_operations"],
                          "cli_check_only": inputs["cli_check_only"],
                          "negative_tests": inputs["negative_tests"],
                          "model_calls": 0, "render_calls": 0,
                          "registration_started": False},
                         ensure_ascii=False, sort_keys=True))
        return 0
    register(packet, revision)
    result = finish(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
