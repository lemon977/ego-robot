"""CPU-only Cards004 CLI-v6 code acceptance candidate.

The check-only mode writes nothing. Registration and terminal publication use
one governed writer and never dispatch CPU admission, HaWoR, Robot or media.
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

from chaoyang.cli_v6 import exact78_cards004_submit_candidate_v1 as driver
from chaoyang.cli_v5 import exact78_four_session_manifest_candidate_v1 as v5
from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT,
    TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes, load_json,
    now_iso, process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID, PHASE = driver.ACCEPTANCE_TASK_ID, driver.ACCEPTANCE_PHASE
RESULT_SCHEMA = "exact78-newraw-cli-v6-cards004-acceptance-result-v1"
RECEIPT_SCHEMA = "exact78-newraw-cli-v6-cards004-acceptance-receipt-v1"
SCRIPT = Path(__file__).resolve()
TEST_FILE = "tests/unit/test_exact78_cli_v6_cards004_acceptance_candidate_v1.py"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PACKET, RECEIPT = driver.ACCEPTANCE_PACKET, driver.ACCEPTANCE_RECEIPT
ATTEMPT = driver.ACCEPTANCE_RESULT.parent
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
OPERATIONS = {
    "exact78_newraw_cpu_admission_candidate_v1": "run_exact78_newraw_cpu_admission_candidate_v1",
    "exact78_newraw_hawor_gpu_candidate_v4": "run_exact78_newraw_hawor_gpu_candidate_v4",
    "exact78_newraw_robot_cpu_candidate_v1": "run_exact78_newraw_robot_cpu_candidate_v1",
}


def _governance_snapshot() -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    receipt, state, index = load_json(RECEIPT_PATH), load_json(TASK_STATE_PATH), load_json(INDEX)
    for key, path in (("task_state", TASK_STATE_PATH),
                      ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if (not isinstance(ref, dict) or ref.get("path") != str(path)
                or validate_artifact_ref(ref)):
            raise RuntimeError("V6_ACCEPTANCE_GOVERNANCE_REF_DRIFT:" + key)
    if (any(row.get("task_id") == TASK_ID for row in state.get("tasks", []))
            or any(row.get("task_id") == TASK_ID for row in index.get("task_packets", []))
            or any(path.exists() for path in (ATTEMPT.parents[1], ATTEMPT,
                                               PACKET.parent, PACKET, RECEIPT))):
        raise RuntimeError("V6_ACCEPTANCE_TASK_ALREADY_USED")
    active = bool(state.get("next_task")
                  or any(row.get("status") in ACTIVE for row in state.get("tasks", []))
                  or any(row.get("execution_allowed") is True
                         for row in index.get("task_packets", [])))
    return int(receipt["governance_revision"]), state, index, active


def _published_operations() -> dict[str, dict[str, Any]]:
    contracts = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts")
    if not isinstance(contracts, dict):
        raise RuntimeError("V6_ACCEPTANCE_ROUTES_ABSENT")
    output: dict[str, dict[str, Any]] = {}
    for key, operation in OPERATIONS.items():
        route = contracts.get(key)
        path = REPO_ROOT / "src/chaoyang/ops" / f"{operation}.py"
        ref = artifact_ref(path)
        if (not isinstance(route, dict)
                or route.get("candidate_operation") != operation
                or route.get("task_packet_required") is not True
                or route.get("current_runnable_operation") is not None
                or (route.get("code_closure") or []).count(ref) != 1):
            raise RuntimeError("V6_ACCEPTANCE_OPERATION_ROUTE_DRIFT:" + key)
        output[key] = {"candidate_operation": operation, "operation_ref": ref}
    return output


def _source_evidence() -> dict[str, Any]:
    from chaoyang.governance.execute_exact78_first_submit_cpu_candidate_v1 import preflight

    row, _ = v5._entry(driver.SESSION, driver.SOURCE_SHA)
    descriptor = preflight(driver.SESSION)["descriptor"]
    if (set(descriptor.get("source_files") or {}) != v5.RAW_ROLES
            or descriptor.get("source_files") != row["source_files"]
            or descriptor.get("session_id") != row["session_id"]
            or descriptor.get("session_root") != row["source_root"]
            or descriptor.get("submission_input_sha256") != row["submission_input_sha256"]
            or descriptor.get("source_kind") != "0915_RAW_VST"
            or descriptor.get("decoded_frames") != 152
            or descriptor.get("fps_nominal") != 30
            or descriptor.get("physical_eye") != "left_sourceIndex_1"
            or descriptor.get("crop_xywh") != [2048, 0, 2048, 1536]
            or descriptor.get("time_mode") != "TIMEBASE_UNVERIFIED"
            or descriptor.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
            or descriptor.get("head_pose_source") not in
               (None, "ABSENT_UNTIL_FIELD_PROJECTION_AND_CLOCK_PROOF")
            or descriptor.get("independent_source_group") != "UNVERIFIED"
            or descriptor.get("world_pose_authority") is not False
            or descriptor.get("metric_authority") is not False):
        raise RuntimeError("V6_ACCEPTANCE_SOURCE_OR_FORBIDDEN_ROLE_DRIFT")
    frames = descriptor.get("frame_rows")
    if (not isinstance(frames, list) or len(frames) != 152
            or any(frame.get("decoded_display_index") != index
                   or frame.get("pts") is not None
                   or frame.get("best_effort_timestamp") is not None
                   or frame.get("sbs_physical_eye") != "left_sourceIndex_1"
                   or frame.get("crop_xywh") != [2048, 0, 2048, 1536]
                   or frame.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
                   for index, frame in enumerate(frames))
            or any(frames[index][key] >= frames[index + 1][key]
                   for key in ("qpc_raw_tick", "capture_timestamp_raw")
                   for index in range(151))):
        raise RuntimeError("V6_ACCEPTANCE_FRAME_MAP_DRIFT")
    return {"source_session": row["session_id"], "source_root": row["source_root"],
            "submission_input_sha256": row["submission_input_sha256"],
            "source_files": row["source_files"], "decoded_frames": 152,
            "physical_eye": "left_sourceIndex_1", "crop_xywh": [2048, 0, 2048, 1536],
            "time_mode": "TIMEBASE_UNVERIFIED",
            "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
            "head_pose_source": "ABSENT_UNTIL_FIELD_PROJECTION_AND_CLOCK_PROOF",
            "control_ground_truth": False, "independent_source_group": "UNVERIFIED"}


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
               XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    return env


def _probe_cli() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli_v6", "submit",
               "exact78-raw-to-robot", "--session", str(driver.SESSION),
               "--expected-input-sha256", driver.SOURCE_SHA, "--check-only"]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=_environment(),
                               capture_output=True, check=False, timeout=180)
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("V6_ACCEPTANCE_CLI_CHECK_NOT_JSON") from exc
    if (not isinstance(value, dict) or completed.returncode != 0
            or value.get("status") != "READY"
            or value.get("node") != "CPU_ADMISSION"
            or value.get("source_session") != "play_cards_0915_004"
            or value.get("submission_input_sha256") != driver.SOURCE_SHA
            or value.get("time_mode") != "TIMEBASE_UNVERIFIED"
            or value.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
            or value.get("registration_started") is not False
            or value.get("model_calls") != 0):
        reason = value.get("reason") if isinstance(value, dict) else None
        if reason in {"ACTIVE_DIFFERENT_WRITER_CONFLICT", "ACTIVE_TASK_WITHOUT_MATCHING_PACKET"}:
            raise RuntimeError("V6_ACCEPTANCE_WAIT_ACTIVE_WRITER:" + reason)
        raise RuntimeError("V6_ACCEPTANCE_CLI_CHECK_NOT_READY:" + str(reason))
    return {"status": "READY", "node": "CPU_ADMISSION",
            "source_session": "play_cards_0915_004",
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "registration_started": False, "model_calls": 0}


def _probe_tests() -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m", "unittest", "-q",
               "tests.unit.test_exact78_cli_v5_four_session_candidate_v1",
               "tests.unit.test_exact78_cli_v6_cards004_candidate_v1"]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=_environment(),
                               capture_output=True, check=False, timeout=180)
    summary = completed.stdout + completed.stderr
    match = re.search(rb"Ran (\d+) tests?", summary)
    if completed.returncode or match is None or int(match.group(1)) != 18 or b"OK" not in summary:
        raise RuntimeError("V6_ACCEPTANCE_NEGATIVE_TESTS_NOT_PASS:" +
                           summary.decode("utf-8", "replace")[-400:])
    normalized = re.sub(rb"(Ran \d+ tests? in )\d+(?:\.\d+)?s",
                        rb"\g<1><elapsed>s", summary)
    return {"status": "PASSED", "test_count": 18,
            "output_sha256": hashlib.sha256(normalized).hexdigest()}


def _refs(inputs: dict[str, Any], runtime: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*runtime, inputs["manifest"], *inputs["source_files"].values(),
            *inputs["code"].values(), inputs["test_code"],
            inputs["acceptance_test_code"],
            *(ref for group in driver.INHERITED_CODE_GROUPS
              for ref in inputs[group].values()),
            *(item["operation_ref"] for item in inputs["published_operations"].values())]


def inspect() -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    revision, state, index, active = _governance_snapshot()
    row, manifest_ref = v5._entry(driver.SESSION, driver.SOURCE_SHA)
    source = _source_evidence()
    code = {name: artifact_ref(REPO_ROOT / name) for name in driver.CODE_FILES}
    inherited = {group: {name: artifact_ref(REPO_ROOT / name) for name in names}
                 for group, names in driver.INHERITED_CODE_GROUPS.items()}
    operations = _published_operations()
    tests = _probe_tests()
    check = ({"status": "WAIT_ACTIVE_WRITER", "node": "CPU_ADMISSION",
              "source_session": row["session_id"],
              "registration_started": False, "model_calls": 0}
             if active else _probe_cli())
    inputs = {
        "source_session": row["session_id"], "source_root": row["source_root"],
        "submission_input_sha256": row["submission_input_sha256"],
        "source_files": source["source_files"],
        "time_mode": "TIMEBASE_UNVERIFIED",
        "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
        "manifest": manifest_ref, "code": code,
        "test_code": artifact_ref(REPO_ROOT / driver.TEST_FILE),
        "acceptance_test_code": artifact_ref(REPO_ROOT / TEST_FILE),
        **inherited, "published_operations": operations,
        "cli_check_only": check, "negative_tests": tests,
    }
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("V6_ACCEPTANCE_REVISION_CHANGED_DURING_PROBE")
    if any(validate_artifact_ref(ref) for ref in _refs(inputs, [])):
        raise RuntimeError("V6_ACCEPTANCE_CODE_OR_SOURCE_SHA_DRIFT")
    return revision, state, index, inputs, active


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    inputs = copy.deepcopy(inputs)
    inputs["hard_cutoff_at"] = (now + driver.MAX_WINDOW).isoformat()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    runtime = [artifact_ref(SCRIPT),
               artifact_ref(REPO_ROOT / "src/chaoyang/governance/common.py"),
               artifact_ref(REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py")]
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m "
               "chaoyang.governance.execute_exact78_cli_v6_cards004_acceptance_candidate_v1 "
               "--register-and-run")
    config = {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "render_calls": 0, "training_steps": 0, "session_attempts_max": 1,
              "wall_seconds_max": 600, "calibration_or_absent": "NOT_CONSUMED",
              "weights": "NOT_CONSUMED", "batch_algorithm_dispatch": False}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": command, "schema": RESULT_SCHEMA})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Seal Cards004 CLI-v6 code and raw-only diagnostic admission proof",
        "phase": PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_CLI_V6_CARDS004_ACCEPTANCE_V1",
        "execution_mode": "EXACT78_GOVERNANCE_ONLY", "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(minutes=10)).isoformat(),
        "read_set": ["AGENTS.md", str(v5.MANIFEST.relative_to(REPO_ROOT)),
                     "src/chaoyang/cli_v6/exact78_cards004_submit_candidate_v1.py",
                     "src/chaoyang/cli_v6/exact78_cards004_cpu_child_v1.py",
                     driver.TEST_FILE, TEST_FILE,
                     "docs/governance/ALGORITHM_CONTRACT.json",
                     "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)),
                      str(INDEX.relative_to(REPO_ROOT)), "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "CARDS004_FIVE_RAW_ROLE_SHA",
                          "CLI_V6_CHECK_ONLY_READY", "THREE_PUBLISHED_STAGE_ROUTES",
                          "CPU_ONLY", "0916_READ_ONLY", "NO_MODEL_OR_RENDER"],
        "required_outputs": ["RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": 600, "session_attempts_max": 1},
        "stop_conditions": ["CODE_OR_SOURCE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "CHECK_ONLY_NOT_READY", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Cards004 v6 code acceptance only; VST time/index unverified; no model, Robot execution, visual, product, control or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("V6_ACCEPTANCE_PACKET_INVALID:" + ";".join(errors))
    return packet


def result_value(packet: dict[str, Any], packet_ref: dict[str, Any],
                 *, status: str = "PASSED", error: str | None = None) -> dict[str, Any]:
    inputs = packet["inputs"]
    return {"schema_version": RESULT_SCHEMA, "task_id": TASK_ID, "status": status,
            "run_signature": packet["run_signature"], "task_packet": packet_ref,
            "manifest": inputs["manifest"], "source_session": inputs["source_session"],
            "source_files": inputs["source_files"],
            "time_mode": inputs["time_mode"],
            "capture_to_decode_alignment": inputs["capture_to_decode_alignment"],
            "code": inputs["code"], "test_code": inputs["test_code"],
            "acceptance_test_code": inputs["acceptance_test_code"],
            **{group: inputs[group] for group in driver.INHERITED_CODE_GROUPS},
            "published_operations": inputs["published_operations"],
            "cli_check_only": inputs["cli_check_only"],
            "negative_tests": inputs["negative_tests"],
            "hard_cutoff_at": inputs["hard_cutoff_at"], "error": error,
            "model_calls": 0, "render_calls": 0,
            "visual_qualified": False, "product_qualified": False,
            "training_eligible": False, "control_ground_truth": False}


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
    current, _state, _index, active = _governance_snapshot()
    if active:
        raise RuntimeError("V6_ACCEPTANCE_ACTIVE_TASK_OR_WRITER_CONFLICT")
    if current != revision or int(load_json(RECEIPT_PATH)["governance_revision"]) != revision:
        raise RuntimeError("V6_ACCEPTANCE_REVISION_CHANGED")
    if packet["inputs"]["cli_check_only"].get("status") != "READY":
        raise RuntimeError("V6_ACCEPTANCE_CHECK_ONLY_NOT_READY")
    if any(validate_artifact_ref(ref) for ref in _refs(packet["inputs"], packet["runtime_code"])):
        raise RuntimeError("V6_ACCEPTANCE_INPUT_SHA_DRIFT")
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]):
        raise RuntimeError("V6_ACCEPTANCE_WRITER_IDENTITY_DRIFT")
    packet_ref = _virtual_ref(PACKET, packet)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": PHASE, "attempt": 1,
        "status": "RUNNING", "session": "play_cards_0915_004",
        "updated_at": stamp, "heartbeat_at": stamp, "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": packet_ref, "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK_ID, "session": "play_cards_0915_004",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "CPU_TWO_THREADS_NO_GPU",
                          "stop_condition": "one immutable code acceptance or ten-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("V6_ACCEPTANCE_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_NEWRAW_CLI_V6_CARDS004_ACCEPTANCE_V1_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(ATTEMPT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET) != packet_ref:
        raise RuntimeError("V6_ACCEPTANCE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_CLI_V6_CARDS004_ACCEPTANCE_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or load_json(PACKET) != packet):
        raise RuntimeError("V6_ACCEPTANCE_WRITER_FENCE_DRIFT")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state["tasks"] if row.get("task_id") == TASK_ID]
    live = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(live) != 1 or live[0].get("task_id") != TASK_ID
            or live[0].get("packet_sha256") != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("V6_ACCEPTANCE_DOUBLE_WRITER_OR_PACKET_DRIFT")


def finish(packet: dict[str, Any]) -> dict[str, Any]:
    _fence(packet)
    error = None
    try:
        if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
            error = "V6_ACCEPTANCE_DEADLINE"
        elif any(validate_artifact_ref(ref) for ref in
                 _refs(packet["inputs"], packet["runtime_code"])):
            error = "V6_ACCEPTANCE_CODE_OR_SOURCE_SHA_DRIFT"
        elif v5.frozen_manifest()[1] != packet["inputs"]["manifest"]:
            error = "V6_ACCEPTANCE_MANIFEST_DRIFT"
        elif _source_evidence()["source_files"] != packet["inputs"]["source_files"]:
            error = "V6_ACCEPTANCE_SOURCE_DRIFT"
        elif _published_operations() != packet["inputs"]["published_operations"]:
            error = "V6_ACCEPTANCE_OPERATION_ROUTE_DRIFT"
        elif packet["inputs"]["cli_check_only"].get("status") != "READY":
            error = "V6_ACCEPTANCE_CLI_NOT_READY"
    except Exception as exc:
        error = f"V6_ACCEPTANCE_VERIFICATION_ERROR:{type(exc).__name__}:{exc}"
    status = "PASSED" if error is None else "FAILED_RUNTIME_FINAL"
    packet_ref = artifact_ref(PACKET)
    atomic_json(ATTEMPT / "RESULT.json", result_value(packet, packet_ref,
                                                       status=status, error=error))
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(RECEIPT, receipt_value(packet, packet_ref, result_ref, status=status))
    receipt_ref = artifact_ref(RECEIPT)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_base(load_json(INDEX))
    row = next(row for row in state["tasks"] if row["task_id"] == TASK_ID)
    row.update(status=status, result=result_ref, terminal_receipt=receipt_ref,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               quality="MACHINE_CODE_ACCEPTANCE_ONLY" if status == "PASSED" else "FAILED")
    state["next_task"] = None
    for item in index["task_packets"]:
        if item.get("task_id") == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_NEWRAW_CLI_V6_CARDS004_ACCEPTANCE_V1_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_CLI_V6_CARDS004_ACCEPTANCE_TERMINAL",
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
    revision, _state, _index, inputs, active = inspect()
    if args.check_only:
        if active:
            print(json.dumps({"status": "WAIT_ACTIVE_WRITER", "task_id": TASK_ID,
                              "governance_revision": revision,
                              "source_session": inputs["source_session"],
                              "source_files": inputs["source_files"],
                              "negative_tests": inputs["negative_tests"],
                              "registration_started": False, "model_calls": 0},
                             ensure_ascii=False, sort_keys=True))
            return 0
        packet = packet_value(inputs)
        print(json.dumps({"status": "READY_NOT_REGISTERED", "task_id": TASK_ID,
                          "governance_revision": revision,
                          "run_signature_preview": packet["run_signature"],
                          "hard_cutoff_at_preview": packet["inputs"]["hard_cutoff_at"],
                          "source_session": inputs["source_session"],
                          "source_files": inputs["source_files"],
                          "cli_check_only": inputs["cli_check_only"],
                          "negative_tests": inputs["negative_tests"],
                          "registration_started": False, "model_calls": 0},
                         ensure_ascii=False, sort_keys=True))
        return 0
    if active:
        raise RuntimeError("V6_ACCEPTANCE_ACTIVE_TASK_OR_WRITER_CONFLICT")
    packet = packet_value(inputs)
    register(packet, revision)
    result = finish(packet)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
