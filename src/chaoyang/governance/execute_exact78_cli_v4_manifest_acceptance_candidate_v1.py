"""Finite CPU-only code acceptance for the Exact78 CLI-v4 manifest candidate.

``--check-only`` is read-only. The registered mode seals code/source/route
facts only: it never dispatches CPU admission, HaWoR, Robot or video rendering.
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

from chaoyang.cli_v4 import exact78_raw_to_robot_manifest_candidate_v1 as candidate
from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT,
    TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes, load_json,
    now_iso, process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = candidate.TASK_ID
PHASE = candidate.PHASE
RESULT_SCHEMA = "exact78-newraw-cli-v4-manifest-acceptance-result-v1"
RECEIPT_SCHEMA = "exact78-newraw-cli-v4-manifest-acceptance-receipt-v1"
ENTRY = "chaoyang.cli_v4"
SCRIPT = Path(__file__).resolve()
TEST_PATH = candidate.TEST_CODE
ACCEPTANCE_TEST_PATH = "tests/unit/test_exact78_cli_v4_manifest_acceptance_candidate_v1.py"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = candidate.PACKET
RECEIPT = candidate.RECEIPT
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
OPERATIONS = {
    "exact78_newraw_cpu_admission_candidate_v1": "run_exact78_newraw_cpu_admission_candidate_v1",
    "exact78_newraw_hawor_gpu_candidate_v4": "run_exact78_newraw_hawor_gpu_candidate_v4",
    "exact78_newraw_robot_cpu_candidate_v1": "run_exact78_newraw_robot_cpu_candidate_v1",
}


def _idle() -> tuple[int, dict[str, Any], dict[str, Any]]:
    receipt, state, index = load_json(RECEIPT_PATH), load_json(TASK_STATE_PATH), load_json(INDEX)
    revision = int(receipt["governance_revision"])
    for key, path in (("task_state", TASK_STATE_PATH),
                      ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if (not isinstance(ref, dict) or ref.get("path") != str(path)
                or validate_artifact_ref(ref)):
            raise RuntimeError("V4_GOVERNANCE_RECEIPT_REF_DRIFT:" + key)
    if (state.get("next_task") or any(row.get("status") in ACTIVE for row in state["tasks"])
            or any(row.get("execution_allowed") is True for row in index["task_packets"])):
        raise RuntimeError("V4_ACTIVE_TASK_OR_WRITER_CONFLICT")
    if (any(row.get("task_id") == TASK_ID for row in state["tasks"])
            or any(row.get("task_id") == TASK_ID for row in index["task_packets"])
            or any(path.exists() for path in (ATTEMPT.parents[1], ATTEMPT,
                                               PACKET.parent, PACKET, RECEIPT))):
        raise RuntimeError("V4_ACCEPTANCE_TASK_ALREADY_USED")
    return revision, state, index


def _published_operations() -> dict[str, dict[str, Any]]:
    routes = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts")
    if not isinstance(routes, dict):
        raise RuntimeError("V4_ALGORITHM_CONTRACT_ROUTES_ABSENT")
    accepted: dict[str, dict[str, Any]] = {}
    for key, operation in OPERATIONS.items():
        route = routes.get(key)
        path = REPO_ROOT / "src/chaoyang/ops" / f"{operation}.py"
        if (not isinstance(route, dict) or route.get("candidate_operation") != operation
                or route.get("task_packet_required") is not True
                or route.get("current_runnable_operation") is not None
                or not isinstance(route.get("code_closure"), list)):
            raise RuntimeError("V4_PUBLISHED_ROUTE_ABSENT_OR_DRIFT:" + key)
        expected = artifact_ref(path)
        if route["code_closure"].count(expected) != 1:
            raise RuntimeError("V4_PUBLISHED_OPERATION_SHA_DRIFT:" + key)
        accepted[key] = {"candidate_operation": operation, "operation_ref": expected}
    return accepted


def _source_evidence() -> dict[str, Any]:
    from chaoyang.governance.execute_exact78_first_submit_cpu_candidate_v1 import preflight

    value, _ = candidate.frozen_manifest()
    evidence: dict[str, Any] = {}
    for row in value["sessions"]:
        source = Path(row["source_root"])
        descriptor = preflight(source)["descriptor"]
        for key, expected in (("session_id", row["session_id"]),
                              ("submission_input_sha256", row["submission_input_sha256"]),
                              ("source_kind", row["source_kind"]),
                              ("physical_eye", row["physical_eye"]),
                              ("time_mode", row["time_mode"]),
                              ("decoded_frames", row["frame_count_decoded"]),
                              ("fps_nominal", row["fps_nominal"])):
            if descriptor.get(key) != expected:
                raise RuntimeError("V4_PREFROZEN_SOURCE_DRIFT:" + row["session_id"] + ":" + key)
        evidence[row["session_id"]] = {
            "source_root": row["source_root"],
            "submission_input_sha256": row["submission_input_sha256"],
            "source_files": descriptor["source_files"],
            "decoded_frames": descriptor["decoded_frames"],
            "fps_nominal": descriptor["fps_nominal"],
            "time_mode": descriptor["time_mode"],
            "physical_eye": descriptor["physical_eye"],
        }
    if len(evidence) != 2:
        raise RuntimeError("V4_SOURCE_SET_DRIFT")
    return evidence


def _probe_cli(manifest_ref: dict[str, Any]) -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", ENTRY, "submit",
               "exact78-raw-to-robot", "--manifest", str(candidate.MANIFEST),
               "--check-only"]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
               XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env,
                               capture_output=True, check=False, timeout=180)
    if completed.returncode:
        raise RuntimeError("V4_CLI_CHECK_ONLY_FAILED:" +
                           completed.stderr.decode("utf-8", "replace")[-400:])
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("V4_CLI_CHECK_ONLY_NOT_JSON") from exc
    rows = value.get("sessions") if isinstance(value, dict) else None
    if (value.get("status") != "READ_ONLY_MATRIX" or value.get("manifest") != manifest_ref
            or value.get("registration_started") is not False
            or value.get("model_calls") != 0
            or value.get("batch_execution_authorized") is not False
            or not isinstance(rows, list) or len(rows) != 2):
        raise RuntimeError("V4_CLI_CHECK_ONLY_MATRIX_DRIFT")
    by_session = {row.get("source_session"): row for row in rows if isinstance(row, dict)}
    if (set(by_session) != {"get_potato_chips_0901_001", "play_cards_0902_002"}
            or by_session["get_potato_chips_0901_001"].get("status") != "READY"
            or by_session["get_potato_chips_0901_001"].get("node") != "CPU_ADMISSION"
            or by_session["play_cards_0902_002"].get("node") != "ROBOT_V1"
            or by_session["play_cards_0902_002"].get("status") !=
               "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS"):
        raise RuntimeError("V4_CLI_CHECK_ONLY_SESSION_STATE_DRIFT")
    frozen, _ = candidate.frozen_manifest()
    for row in frozen["sessions"]:
        if by_session[row["session_id"]].get("submission_input_sha256") != row["submission_input_sha256"]:
            raise RuntimeError("V4_CLI_CHECK_ONLY_SUBMISSION_SHA_DRIFT")
    return {"status": "READ_ONLY_MATRIX", "source_sessions": sorted(by_session),
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "registration_started": False, "model_calls": 0}


def _probe_tests() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "unittest", "discover",
               "-s", "tests/unit", "-p", Path(TEST_PATH).name]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
               XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env,
                               capture_output=True, check=False, timeout=120)
    summary = completed.stdout + completed.stderr
    match = re.search(rb"Ran (\d+) tests?", summary)
    if completed.returncode or match is None or int(match.group(1)) < 10 or b"OK" not in summary:
        raise RuntimeError("V4_CLI_NEGATIVE_TESTS_NOT_PASS:" +
                           summary.decode("utf-8", "replace")[-400:])
    normalized = re.sub(rb"(Ran \d+ tests? in )\d+(?:\.\d+)?s",
                        rb"\g<1><elapsed>s", summary)
    return {"status": "PASSED", "test_count": int(match.group(1)),
            "output_sha256": hashlib.sha256(normalized).hexdigest()}


def _refs(inputs: dict[str, Any], runtime: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_files = [ref for item in inputs["source_evidence"].values()
                    for ref in item["source_files"].values()]
    manifest, _ = candidate.frozen_manifest()
    cards = next(item for item in manifest["sessions"]
                 if item["session_id"] == "play_cards_0902_002")
    sealed_robot = cards["sealed_robot_terminal"]
    return [*runtime, inputs["manifest"], *inputs["cli_code"].values(),
            *inputs["inherited_cli_v3_code"].values(),
            *inputs["inherited_cli_v2_code"].values(), inputs["test_code"],
            inputs["acceptance_test_code"],
            *(item["operation_ref"] for item in inputs["published_operations"].values()),
            *source_files, *(sealed_robot[key] for key in
                             ("task_packet", "result", "receipt"))]


def inspect() -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    revision, state, index = _idle()
    _, manifest_ref = candidate.frozen_manifest()
    inputs = {
        "accepted_entry": ENTRY,
        "manifest": manifest_ref,
        "source_evidence": _source_evidence(),
        "cli_code": {name: artifact_ref(REPO_ROOT / name) for name in candidate.CLI_CODE},
        "inherited_cli_v3_code": {name: artifact_ref(REPO_ROOT / name)
                                  for name in candidate.v3.CLI_CODE},
        "inherited_cli_v2_code": {name: artifact_ref(REPO_ROOT / name)
                                  for name in candidate.prior.CLI_CODE},
        "test_code": artifact_ref(REPO_ROOT / TEST_PATH),
        "acceptance_test_code": artifact_ref(REPO_ROOT / ACCEPTANCE_TEST_PATH),
        "published_operations": _published_operations(),
        "cli_check_only": _probe_cli(manifest_ref),
        "negative_tests": _probe_tests(),
    }
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("V4_GOVERNANCE_REVISION_CHANGED_DURING_PROBE")
    if any(validate_artifact_ref(ref) for ref in _refs(inputs, [])):
        raise RuntimeError("V4_ACCEPTANCE_SOURCE_OR_CODE_SHA_DRIFT")
    return revision, state, index, inputs


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    runtime = [artifact_ref(SCRIPT),
               artifact_ref(REPO_ROOT / "src/chaoyang/governance/common.py"),
               artifact_ref(REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py")]
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m "
               "chaoyang.governance.execute_exact78_cli_v4_manifest_acceptance_candidate_v1 "
               "--register-and-run")
    config = {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "render_calls": 0, "training_steps": 0, "session_attempts_max": 1,
              "wall_seconds_max": 600, "calibration_or_absent": "NOT_CONSUMED",
              "weights": "NOT_CONSUMED", "batch_algorithm_dispatch": False}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": command, "schema": RESULT_SCHEMA})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Seal CLI-v4 two-session manifest code and read-only admission proof",
        "phase": PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_CLI_V4_MANIFEST_ACCEPTANCE_V1",
        "execution_mode": "EXACT78_GOVERNANCE_ONLY", "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(minutes=10)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     str(candidate.MANIFEST.relative_to(REPO_ROOT)),
                     "src/chaoyang/cli_v4/__init__.py",
                     "src/chaoyang/cli_v4/exact78_raw_to_robot_manifest_candidate_v1.py",
                     "src/chaoyang/cli_v4/exact78_newraw_cpu_submit_manifest_v1.py",
                     TEST_PATH, "docs/governance/ALGORITHM_CONTRACT.json"],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)),
                      str(INDEX.relative_to(REPO_ROOT)), "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "TWO_PREFROZEN_SOURCE_SHA",
                          "CLI_V4_CHECK_ONLY_MATRIX", "THREE_PUBLISHED_STAGE_ROUTES",
                          "CPU_ONLY", "0916_READ_ONLY", "NO_MODEL_OR_RENDER"],
        "required_outputs": ["RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": 600, "session_attempts_max": 1},
        "stop_conditions": ["CODE_OR_SOURCE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "CHECK_ONLY_NOT_READY", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "CLI-v4 code admission only; no model, Robot execution, 0915 VST admission, visual, product, control, general or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("V4_ACCEPTANCE_PACKET_INVALID:" + ";".join(errors))
    return packet


def result_value(packet: dict[str, Any], packet_ref: dict[str, Any],
                 *, status: str = "PASSED", error: str | None = None) -> dict[str, Any]:
    inputs = packet["inputs"]
    return {"schema_version": RESULT_SCHEMA, "task_id": TASK_ID, "status": status,
            "accepted_entry": ENTRY, "run_signature": packet["run_signature"],
            "task_packet": packet_ref, "manifest": inputs["manifest"],
            "source_evidence": inputs["source_evidence"],
            "cli_code": inputs["cli_code"],
            "inherited_cli_v3_code": inputs["inherited_cli_v3_code"],
            "inherited_cli_v2_code": inputs["inherited_cli_v2_code"],
            "test_code": inputs["test_code"],
            "acceptance_test_code": inputs["acceptance_test_code"],
            "published_operations": inputs["published_operations"],
            "cli_check_only": inputs["cli_check_only"],
            "negative_tests": inputs["negative_tests"], "error": error,
            "model_calls": 0, "render_calls": 0, "visual_qualified": False,
            "training_eligible": False}


def receipt_value(packet: dict[str, Any], packet_ref: dict[str, Any],
                  result_ref: dict[str, Any], *, status: str = "PASSED") -> dict[str, Any]:
    return {"schema_version": RECEIPT_SCHEMA, "task_id": TASK_ID,
            "status": status, "run_signature": packet["run_signature"],
            "task_packet": packet_ref, "result": result_ref}


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _index_base(index: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in index.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def register(packet: dict[str, Any], revision: int) -> None:
    _idle()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("V4_ACCEPTANCE_REVISION_CHANGED")
    if any(validate_artifact_ref(ref) for ref in _refs(packet["inputs"], packet["runtime_code"])):
        raise RuntimeError("V4_ACCEPTANCE_INPUT_SHA_DRIFT")
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]):
        raise RuntimeError("V4_ACCEPTANCE_WRITER_IDENTITY_DRIFT")
    packet_ref = _virtual_ref(PACKET, packet)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": PHASE, "attempt": 1,
        "status": "RUNNING", "session": "exact78_two_session_manifest_v1",
        "updated_at": stamp, "heartbeat_at": stamp, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": packet_ref, "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {
        "task_id": TASK_ID, "session": "exact78_two_session_manifest_v1",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_TWO_THREADS_NO_GPU",
        "stop_condition": "one immutable manifest code acceptance or ten-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("V4_ACCEPTANCE_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_NEWRAW_CLI_V4_MANIFEST_ACCEPTANCE_V1_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET) != packet_ref:
        raise RuntimeError("V4_ACCEPTANCE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_CLI_V4_MANIFEST_ACCEPTANCE_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or load_json(PACKET) != packet):
        raise RuntimeError("V4_ACCEPTANCE_WRITER_FENCE_DRIFT")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK_ID]
    live = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(live) != 1 or live[0].get("task_id") != TASK_ID
            or live[0].get("packet_sha256") != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("V4_ACCEPTANCE_DOUBLE_WRITER_OR_PACKET_DRIFT")


def finish(packet: dict[str, Any]) -> dict[str, Any]:
    _fence(packet)
    error = None
    try:
        if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
            error = "V4_ACCEPTANCE_DEADLINE"
        elif any(validate_artifact_ref(ref) for ref in
                 _refs(packet["inputs"], packet["runtime_code"])):
            error = "V4_ACCEPTANCE_CODE_OR_SOURCE_SHA_DRIFT"
        elif candidate.frozen_manifest()[1] != packet["inputs"]["manifest"]:
            error = "V4_ACCEPTANCE_MANIFEST_DRIFT"
        elif candidate._sealed_cards002_attachment(next(
                item for item in candidate.frozen_manifest()[0]["sessions"]
                if item["session_id"] == "play_cards_0902_002"))["sealed_attachment"] is not True:
            error = "V4_ACCEPTANCE_CARDS002_SEALED_CHAIN_DRIFT"
        elif _source_evidence() != packet["inputs"]["source_evidence"]:
            error = "V4_ACCEPTANCE_SOURCE_DRIFT"
        elif _published_operations() != packet["inputs"]["published_operations"]:
            error = "V4_ACCEPTANCE_OPERATION_ROUTE_DRIFT"
    except Exception as exc:
        error = f"V4_ACCEPTANCE_VERIFICATION_ERROR:{type(exc).__name__}:{exc}"
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
               updated_at=now_iso(), heartbeat_at=now_iso(),
               quality="MACHINE_CODE_ACCEPTANCE_ONLY" if status == "PASSED" else "FAILED")
    state["next_task"] = None
    for item in index["task_packets"]:
        if item.get("task_id") == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_NEWRAW_CLI_V4_MANIFEST_ACCEPTANCE_V1_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_CLI_V4_MANIFEST_ACCEPTANCE_TERMINAL",
                   expected_revision=int(load_json(RECEIPT_PATH)["governance_revision"]),
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return load_json(ATTEMPT / "RESULT.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    revision, _state, _index, inputs = inspect()
    packet = packet_value(inputs)
    if args.check_only:
        print(json.dumps({"status": "READY_NOT_REGISTERED", "task_id": TASK_ID,
                          "governance_revision": revision,
                          "run_signature_preview": packet["run_signature"],
                          "accepted_entry": ENTRY, "manifest": inputs["manifest"],
                          "source_evidence": inputs["source_evidence"],
                          "cli_code": inputs["cli_code"],
                          "inherited_cli_v3_code": inputs["inherited_cli_v3_code"],
                          "inherited_cli_v2_code": inputs["inherited_cli_v2_code"],
                          "test_code": inputs["test_code"],
                          "acceptance_test_code": inputs["acceptance_test_code"],
                          "published_operations": inputs["published_operations"],
                          "cli_check_only": inputs["cli_check_only"],
                          "negative_tests": inputs["negative_tests"],
                          "model_calls": 0, "render_calls": 0,
                          "registration_started": False}, ensure_ascii=False, sort_keys=True))
        return 0
    register(packet, revision)
    result = finish(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
