"""Two-session, manifest-fenced Exact78 one-submit candidate.

This is not a production route. Execution is impossible until an independent
CLI-v4 code-acceptance Task Packet, RESULT and receipt have sealed successfully.
Every algorithm call remains a separate governed ``chaoyang run`` operation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any

from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as prior
from chaoyang.cli_v3 import exact78_raw_to_robot_candidate_v2 as v3
from chaoyang.governance.common import (
    REPO_ROOT, TASK_STATE_PATH, artifact_ref, load_json, process_identity,
    validate_artifact_ref,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


MANIFEST = (REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930"
            / "46_TWO_SESSION_NATIVE25_CLI_V4_MANIFEST_CANDIDATE_V1.json")
MANIFEST_SHA = "e28f55cba0261b7ff6a891229f294d188863c9f8dc05dfae27ec076ac1dbf6e6"
TASK_ID = "exact78_newraw_cli_v4_manifest_acceptance_two_session_v1"
PHASE = "EXACT78_NEWRAW_CLI_V4_MANIFEST_CODE_ACCEPTANCE"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RESULT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001/RESULT.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
CLI_CODE = tuple(f"src/chaoyang/cli_v4/{name}" for name in (
    "__init__.py", "__main__.py", "exact78_raw_to_robot_manifest_candidate_v1.py",
    "exact78_newraw_cpu_submit_manifest_v1.py"))
TEST_CODE = "tests/unit/test_exact78_cli_v4_manifest_candidate_v1.py"
OPERATIONS = prior.ACCEPTANCE_OPERATIONS


def _ref(path: Path, value: Any, reason: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or value.get("path") != str(path)
            or validate_artifact_ref(value)):
        raise ColdStartInputError(reason)
    return value


def frozen_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the project-local manifest, not raw data or prior results."""
    if not MANIFEST.is_file():
        raise ColdStartInputError("CLI_V4_MANIFEST_MISSING")
    ref = artifact_ref(MANIFEST)
    if not hmac.compare_digest(ref["sha256"], MANIFEST_SHA):
        raise ColdStartInputError("CLI_V4_MANIFEST_SHA_DRIFT")
    value = load_json(MANIFEST)
    rows = value.get("sessions")
    domains = value.get("source_domains")
    if (value.get("schema_version") != "exact78-newraw-cli-v4-two-session-manifest-candidate-v1"
            or not isinstance(rows, list) or len(rows) != 2
            or not isinstance(domains, dict)
            or set(domains) != {"0902_tracker_clip", "0901_tracker_native25", "0915_raw_vst"}
            or domains["0901_tracker_native25"].get("source_kind") != "TRACKER_CLIP_RGB"
            or domains["0901_tracker_native25"].get("execution_status") != "ONE_WHITELISTED_SESSION_ONLY"
            or domains["0902_tracker_clip"].get("source_kind") != "TRACKER_CLIP_RGB"
            or domains["0902_tracker_clip"].get("execution_status") != "ONE_WHITELISTED_SESSION_ONLY"
            or domains["0915_raw_vst"].get("source_kind") != "0915_RAW_VST"
            or domains["0915_raw_vst"].get("physical_eye") != "left_sourceIndex_1"
            or domains["0915_raw_vst"].get("time_mode") != "TIMEBASE_UNVERIFIED"
            or domains["0915_raw_vst"].get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
            or domains["0915_raw_vst"].get("glove_motion_input") != "FORBIDDEN"
            or domains["0915_raw_vst"].get("execution_status") != "SCHEMA_RESERVED_NO_SESSION_ADMITTED"):
        raise ColdStartInputError("CLI_V4_MANIFEST_SCHEMA")
    expected = {"get_potato_chips_0901_001": (799, 25),
                "play_cards_0902_002": (189, 30)}
    if {row.get("session_id") for row in rows if isinstance(row, dict)} != set(expected):
        raise ColdStartInputError("CLI_V4_MANIFEST_SESSION_SET_DRIFT")
    roots: set[str] = set()
    for row in rows:
        name = row["session_id"]
        root = Path(str(row.get("source_root", "")))
        digest = row.get("submission_input_sha256")
        if (not root.is_absolute() or root.name != name
                or str(root) in roots or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or row.get("source_kind") != "TRACKER_CLIP_RGB"
                or row.get("physical_eye") != "tracker_session_left_RGB"
                or row.get("time_mode") != "TRACKER_CLIP_METADATA_ONLY"
                or (row.get("frame_count_decoded"), row.get("fps_nominal")) != expected[name]):
            raise ColdStartInputError("CLI_V4_MANIFEST_IDENTITY_OR_TIME_DRIFT")
        roots.add(str(root))
    return value, ref


def _entry(source: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, ref = frozen_manifest()
    rows = [row for row in value["sessions"] if row["source_root"] == str(source)]
    if (len(rows) != 1 or not hmac.compare_digest(rows[0]["submission_input_sha256"],
                                                    expected_sha)):
        raise ColdStartInputError("CLI_V4_SESSION_NOT_IN_FROZEN_MANIFEST")
    return rows[0], ref


def _sealed_stage(task_id: str, phase: str, result_schema: str,
                  receipt_schema: str, source: Path, expected_sha: str,
                  *, expected_refs: dict[str, Any] | None = None,
                  ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Attach by immutable ledger refs; never regenerate a historical packet."""
    state, index = load_json(TASK_STATE_PATH), load_json(prior.INDEX)
    rows = [item for item in state.get("tasks", []) if item.get("task_id") == task_id]
    entries = [item for item in index.get("task_packets", [])
               if item.get("task_id") == task_id]
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    result_path = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001/RESULT.json"
    receipt_path = REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json"
    if (len(rows) != 1 or len(entries) != 1 or rows[0].get("phase") != phase
            or rows[0].get("session") != source.name
            or rows[0].get("status") != "PASSED"
            or entries[0].get("execution_allowed") is not False
            or entries[0].get("packet_path") != str(packet_path.relative_to(REPO_ROOT))
            or entries[0].get("packet_sha256") != rows[0].get("task_packet", {}).get("sha256")):
        raise ColdStartInputError("CLI_V4_SEALED_STAGE_LEDGER_DRIFT:" + phase)
    row = rows[0]
    refs = {"task_packet": _ref(packet_path, row.get("task_packet"),
                                 "CLI_V4_STAGE_PACKET_SHA_DRIFT:" + phase),
            "result": _ref(result_path, row.get("result"),
                           "CLI_V4_STAGE_RESULT_SHA_DRIFT:" + phase),
            "receipt": _ref(receipt_path, row.get("terminal_receipt"),
                            "CLI_V4_STAGE_RECEIPT_SHA_DRIFT:" + phase)}
    if expected_refs is not None and refs != expected_refs:
        raise ColdStartInputError("CLI_V4_FROZEN_ROBOT_TRIPLE_DRIFT")
    packet, result, receipt = load_json(packet_path), load_json(result_path), load_json(receipt_path)
    inputs = packet.get("inputs") or {}
    if (packet.get("task_id") != task_id or packet.get("phase") != phase
            or packet.get("run_signature") != row.get("run_signature")
            or inputs.get("source_session") != source.name
            or inputs.get("source_root") != str(source)
            or inputs.get("submission_input_sha256") != expected_sha
            or result.get("schema_version") != result_schema
            or result.get("task_id") != task_id or result.get("status") != "PASSED"
            or result.get("source_session") != source.name
            or receipt.get("schema_version") != receipt_schema
            or receipt.get("task_id") != task_id or receipt.get("status") != "PASSED"
            or receipt.get("result") != refs["result"]):
        raise ColdStartInputError("CLI_V4_SEALED_STAGE_IDENTITY_DRIFT:" + phase)
    return packet, result, refs


def _sealed_cards002_attachment(row: dict[str, Any]) -> dict[str, Any]:
    """Read-only terminal Cards002 path; no old code/signature recomputation."""
    source, expected_sha = Path(row["source_root"]), row["submission_input_sha256"]
    triple = row.get("sealed_robot_terminal")
    if not isinstance(triple, dict) or set(triple) != {"task_packet", "result", "receipt"}:
        raise ColdStartInputError("CLI_V4_CARDS002_TERMINAL_PIN_ABSENT")
    cpu_id = "exact78_newraw_play_cards_0902_002_50b73360d82cd4a7_v1"
    hawor_id = "exact78_newraw_hawor_play_cards_0902_002_50b73360d82cd4a7_v4"
    robot_id = "exact78_newraw_robot_play_cards_0902_002_50b73360d82cd4a7_v1"
    cpu_packet, cpu_result, cpu_refs = _sealed_stage(
        cpu_id, prior.PHASES[0], prior.SCHEMAS[0], prior.RECEIPT_SCHEMAS[0],
        source, expected_sha)
    hawor_packet, hawor_result, hawor_refs = _sealed_stage(
        hawor_id, prior.PHASES[1], prior.SCHEMAS[1], prior.RECEIPT_SCHEMAS[1],
        source, expected_sha)
    robot_packet, robot_result, robot_refs = _sealed_stage(
        robot_id, prior.PHASES[2], prior.SCHEMAS[2], prior.RECEIPT_SCHEMAS[2],
        source, expected_sha, expected_refs=triple)
    cpu_inputs, hawor_inputs, robot_inputs = (
        cpu_packet["inputs"], hawor_packet["inputs"], robot_packet["inputs"])
    admission, stage, render = cpu_result.get("admission") or {}, \
        hawor_result.get("stage") or {}, robot_result.get("stage") or {}
    if (cpu_result.get("source_root") != str(source)
            or cpu_result.get("submission_input_sha256") != expected_sha
            or cpu_result.get("source_sha_stable_at_terminal") is not True
            or hawor_result.get("producer_result") != cpu_refs["result"]
            or hawor_result.get("source_frames_expected") != row["frame_count_decoded"]
            or hawor_result.get("stage_status") != "SEALED_CAMERA_PREDICTIONS"
            or hawor_result.get("source_sha_stable_at_terminal") is not True
            or hawor_inputs.get("upstream_admission_result") != cpu_refs["result"]
            or robot_inputs.get("upstream_admission_result") != cpu_refs["result"]
            or robot_inputs.get("hawor_outer_result") != hawor_refs["result"]
            or robot_result.get("cpu_admission_result") != cpu_refs["result"]
            or robot_result.get("hawor_outer_result") != hawor_refs["result"]
            or robot_result.get("source_frames_expected") != row["frame_count_decoded"]
            or robot_result.get("render_complete") is not True
            or robot_result.get("post_input_sha_errors") != []
            or render.get("decoded_frames") != row["frame_count_decoded"]
            or robot_result.get("visual_qualified") is not False
            or robot_result.get("training_eligible") is not False
            or robot_result.get("general_operation") is not None):
        raise ColdStartInputError("CLI_V4_CARDS002_SEALED_CHAIN_DRIFT")
    for key, upstream_key in (("descriptor", "upstream_admission_descriptor"),
                              ("frame_time_map", "upstream_admission_frame_time_map"),
                              ("seal", "upstream_admission_seal")):
        if (admission.get(key) != hawor_inputs.get(upstream_key)
                or admission.get(key) != robot_inputs.get(upstream_key)):
            raise ColdStartInputError("CLI_V4_CARDS002_ADMISSION_CHAIN_DRIFT:" + key)
    for key, downstream_key in (("result", "hawor_inner_result"),
                                ("seal", "hawor_stage_seal"),
                                ("native_open_audit", "hawor_native_open_audit")):
        if stage.get(key) != robot_inputs.get(downstream_key):
            raise ColdStartInputError("CLI_V4_CARDS002_HAWOR_CHAIN_DRIFT:" + key)
    source_files = cpu_inputs.get("source_files")
    if (not isinstance(source_files, dict) or set(source_files) != {
            "video", "source_stereo_video", "camera_params", "slam", "clip_manifest"}
            or source_files != cpu_result.get("source_files")
            or source_files != hawor_inputs.get("source_files")
            or source_files != robot_inputs.get("source_files")
            or cpu_inputs.get("raw_video") != hawor_inputs.get("raw_video")
            or cpu_inputs.get("raw_video") != robot_inputs.get("raw_video")):
        raise ColdStartInputError("CLI_V4_CARDS002_RAW_SOURCE_CHAIN_DRIFT")
    for ref in (*source_files.values(), *admission.values(),
                *(stage.get(key) for key in ("result", "seal", "native_open_audit")),
                *(render.get(key) for key in ("result", "stage_result", "frame_states"))):
        if not isinstance(ref, dict) or validate_artifact_ref(ref):
            raise ColdStartInputError("CLI_V4_CARDS002_CHAIN_REF_SHA_DRIFT")
    videos = render.get("videos")
    if (not isinstance(videos, dict)
            or set(videos) != {"ego_robot", "external_full_robot", "raw_motion_source"}
            or any(not isinstance(ref, dict) or validate_artifact_ref(ref)
                   for ref in videos.values())):
        raise ColdStartInputError("CLI_V4_CARDS002_VIDEO_REF_SHA_DRIFT")
    return {"schema_version": "exact78-newraw-submit-dag-candidate-v4",
            "source_session": source.name, "submission_input_sha256": expected_sha,
            "status": "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS",
            "node": "ROBOT_V1", "task_id": robot_id, "result": robot_refs["result"],
            "receipt": robot_refs["receipt"], "sealed_attachment": True,
            "scene_clean": "NOT_WIRED", "general_operation": None,
            "visual_qualified": False, "product_qualified": False,
            "training_eligible": False}


def accepted_code_for_execution(source: Path, expected_sha: str) -> dict[str, Any]:
    """Full receipt gate; no receipt exists while this remains code-prep only."""
    _, manifest_ref = _entry(source, expected_sha)
    if not all(path.is_file() for path in (PACKET, RESULT, RECEIPT)):
        raise ColdStartInputError("CLI_V4_CODE_ACCEPTANCE_NOT_SEALED")
    state, index = load_json(TASK_STATE_PATH), load_json(prior.INDEX)
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID]
    entries = [item for item in index.get("task_packets", []) if item.get("task_id") == TASK_ID]
    if (len(rows) != 1 or rows[0].get("phase") != PHASE
            or rows[0].get("status") != "PASSED" or len(entries) != 1
            or entries[0].get("execution_allowed") is not False
            or entries[0].get("packet_path") != str(PACKET.relative_to(REPO_ROOT))
            or entries[0].get("packet_sha256") != rows[0].get("task_packet", {}).get("sha256")):
        raise ColdStartInputError("CLI_V4_CODE_ACCEPTANCE_NOT_PASSED")
    row = rows[0]
    _ref(PACKET, row.get("task_packet"), "CLI_V4_PACKET_SHA_DRIFT")
    _ref(RESULT, row.get("result"), "CLI_V4_RESULT_SHA_DRIFT")
    _ref(RECEIPT, row.get("terminal_receipt"), "CLI_V4_RECEIPT_SHA_DRIFT")
    packet, result, receipt = load_json(PACKET), load_json(RESULT), load_json(RECEIPT)
    signature = row.get("run_signature")
    inputs = packet.get("inputs") or {}
    if (packet.get("task_id") != TASK_ID or packet.get("phase") != PHASE
            or packet.get("run_signature") != signature
            or packet.get("execution_mode") != "EXACT78_GOVERNANCE_ONLY"
            or inputs.get("accepted_entry") != "chaoyang.cli_v4"
            or inputs.get("manifest") != manifest_ref
            or result.get("schema_version") != "exact78-newraw-cli-v4-manifest-acceptance-result-v1"
            or result.get("task_id") != TASK_ID or result.get("status") != "PASSED"
            or result.get("run_signature") != signature
            or result.get("accepted_entry") != "chaoyang.cli_v4"
            or result.get("manifest") != manifest_ref
            or result.get("task_packet") != row["task_packet"]
            or receipt.get("schema_version") != "exact78-newraw-cli-v4-manifest-acceptance-receipt-v1"
            or receipt.get("task_id") != TASK_ID or receipt.get("status") != "PASSED"
            or receipt.get("run_signature") != signature
            or receipt.get("task_packet") != row["task_packet"]
            or receipt.get("result") != row["result"]):
        raise ColdStartInputError("CLI_V4_ACCEPTANCE_IDENTITY_DRIFT")
    for key, names in (("cli_code", CLI_CODE),
                       ("inherited_cli_v3_code", v3.CLI_CODE),
                       ("inherited_cli_v2_code", prior.CLI_CODE)):
        refs = result.get(key)
        if (not isinstance(refs, dict) or set(refs) != set(names)
                or inputs.get(key) != refs):
            raise ColdStartInputError("CLI_V4_CODE_SET_DRIFT:" + key)
        for name in names:
            _ref(REPO_ROOT / name, refs[name], "CLI_V4_CODE_SHA_DRIFT:" + name)
    test_ref = _ref(REPO_ROOT / TEST_CODE, result.get("test_code"),
                    "CLI_V4_TEST_SHA_DRIFT")
    if inputs.get("test_code") != test_ref:
        raise ColdStartInputError("CLI_V4_TEST_INPUT_DRIFT")
    acceptance_test = "tests/unit/test_exact78_cli_v4_manifest_acceptance_candidate_v1.py"
    acceptance_test_ref = _ref(REPO_ROOT / acceptance_test,
                               result.get("acceptance_test_code"),
                               "CLI_V4_ACCEPTANCE_TEST_SHA_DRIFT")
    if inputs.get("acceptance_test_code") != acceptance_test_ref:
        raise ColdStartInputError("CLI_V4_ACCEPTANCE_TEST_INPUT_DRIFT")
    source_evidence = result.get("source_evidence")
    if (not isinstance(source_evidence, dict)
            or set(source_evidence) != {"get_potato_chips_0901_001", "play_cards_0902_002"}
            or inputs.get("source_evidence") != source_evidence):
        raise ColdStartInputError("CLI_V4_SOURCE_EVIDENCE_SET_DRIFT")
    frozen, _ = frozen_manifest()
    for item in frozen["sessions"]:
        evidence = source_evidence[item["session_id"]]
        if (not isinstance(evidence, dict)
                or evidence.get("source_root") != item["source_root"]
                or evidence.get("submission_input_sha256") != item["submission_input_sha256"]
                or evidence.get("decoded_frames") != item["frame_count_decoded"]
                or evidence.get("fps_nominal") != item["fps_nominal"]
                or evidence.get("time_mode") != item["time_mode"]
                or evidence.get("physical_eye") != item["physical_eye"]
                or not isinstance(evidence.get("source_files"), dict)
                or set(evidence["source_files"]) != {"video", "source_stereo_video",
                                                     "camera_params", "slam", "clip_manifest"}):
            raise ColdStartInputError("CLI_V4_SOURCE_EVIDENCE_DRIFT:" + item["session_id"])
        for ref in evidence["source_files"].values():
            if not isinstance(ref, dict) or validate_artifact_ref(ref):
                raise ColdStartInputError("CLI_V4_SOURCE_FILE_SHA_DRIFT")
    check = result.get("cli_check_only")
    tests = result.get("negative_tests")
    if (inputs.get("cli_check_only") != check or not isinstance(check, dict)
            or check.get("status") != "READ_ONLY_MATRIX"
            or check.get("registration_started") is not False
            or check.get("model_calls") != 0
            or not re.fullmatch(r"[0-9a-f]{64}", check.get("stdout_sha256") or "")
            or inputs.get("negative_tests") != tests or not isinstance(tests, dict)
            or tests.get("status") != "PASSED" or tests.get("test_count", 0) < 10):
        raise ColdStartInputError("CLI_V4_PREFLIGHT_OR_TEST_EVIDENCE_DRIFT")
    published = result.get("published_operations")
    if (not isinstance(published, dict) or set(published) != set(OPERATIONS)
            or inputs.get("published_operations") != published):
        raise ColdStartInputError("CLI_V4_OPERATION_SET_DRIFT")
    contracts = load_json(prior.ALGORITHM_CONTRACT).get("special_status_contracts") or {}
    for key in OPERATIONS:
        contract, sealed = contracts.get(key) or {}, published[key]
        operation = contract.get("candidate_operation")
        if (not isinstance(operation, str)
                or not re.fullmatch(r"run_exact78_newraw_[a-z0-9_]+", operation)
                or contract.get("task_packet_required") is not True
                or contract.get("current_runnable_operation") is not None
                or sealed.get("candidate_operation") != operation
                or (contract.get("code_closure") or []).count(sealed.get("operation_ref")) != 1):
            raise ColdStartInputError("CLI_V4_OPERATION_ROUTE_DRIFT:" + key)
        _ref(REPO_ROOT / "src/chaoyang/ops" / f"{operation}.py",
             sealed.get("operation_ref"), "CLI_V4_OPERATION_SHA_DRIFT:" + key)
    return row["result"]


def plan_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    source = prior.legacy_submit._raw_session_path(session)
    entry, manifest_ref = _entry(source, expected_input_sha256)
    decision = (_sealed_cards002_attachment(entry)
                if entry["session_id"] == "play_cards_0902_002" else
                prior.plan_one(source, expected_input_sha256))
    return {**decision, "candidate_cli": "chaoyang.cli_v4",
            "manifest": manifest_ref, "code_acceptance": "NOT_CHECKED_CHECK_ONLY"}


def plan_manifest() -> dict[str, Any]:
    value, ref = frozen_manifest()
    decisions = [plan_one(Path(row["source_root"]), row["submission_input_sha256"])
                 for row in value["sessions"]]
    return {"schema_version": "exact78-newraw-cli-v4-manifest-plan-v1",
            "status": "READ_ONLY_MATRIX", "manifest": ref,
            "registration_started": False, "model_calls": 0,
            "sessions": decisions, "batch_execution_authorized": False}


def _launch_cpu(source: Path, expected_sha: str, task_id: str) -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m",
               "chaoyang.cli_v4.exact78_newraw_cpu_submit_manifest_v1",
               "--session", str(source), "--expected-input-sha256", expected_sha,
               "--register-and-run"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1")
    previous = {kind: signal.getsignal(kind) for kind in (signal.SIGINT, signal.SIGTERM)}
    interrupted: list[int] = []
    def defer_signal(signum: int, _frame: Any) -> None:
        interrupted.append(signum)
    for kind in previous:
        signal.signal(kind, defer_signal)
    child: subprocess.Popen[Any] | None = None
    try:
        if interrupted:
            raise ColdStartInputError("CLI_V4_INTERRUPTED_BEFORE_CHILD")
        print(json.dumps({"event": "CLI_V4_CPU_CHILD_PREPARED", "task_id": task_id,
                          "at": datetime.now(timezone.utc).isoformat()}, sort_keys=True),
              file=sys.stderr, flush=True)
        child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        identity = process_identity(child.pid)
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        return {"exit_code": code, "interrupted": bool(interrupted),
                "child_pid": child.pid, "child_start_ticks": identity.get("start_ticks")}
    finally:
        if child is not None and child.poll() is None:
            child.wait()
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def submit_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    source = prior.legacy_submit._raw_session_path(session)
    accepted = accepted_code_for_execution(source, expected_input_sha256)
    launched: list[str] = []
    for _ in prior.PHASES:
        decision = plan_one(source, expected_input_sha256)
        if decision["status"] != "READY":
            return {**decision, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted, "nodes_launched_here": launched}
        node = decision["node"]
        if budget := prior._budget(node):
            return {**decision, **budget, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted, "nodes_launched_here": launched}
        upstream = None
        if node == "HAWOR_V4":
            prepared = prior.cpu.preflight(source)
            parent = prior.cpu.packet_value(prepared)
            upstream = prior.cpu._paths(parent["task_id"])[0] / "RESULT.json"
        elif node == "ROBOT_V1":
            prepared = prior.cpu.preflight(source)
            parent = prior.cpu.packet_value(prepared)
            hawor_id = prior.hawor.packet_value(prior.cpu._paths(parent["task_id"])[0] /
                                                  "RESULT.json")["task_id"]
            upstream = prior.hawor._paths(hawor_id)[0] / "RESULT.json"
        launch = (_launch_cpu(source, expected_input_sha256, decision["task_id"])
                  if node == "CPU_ADMISSION" else
                  prior._launch(node, source, upstream, decision["task_id"]))
        launched.append(node)
        after = plan_one(source, expected_input_sha256)
        if launch["exit_code"] != 0 and after.get("status") not in {
                "STOP_TERMINAL", "ATTACHED_ACTIVE"}:
            raise ColdStartInputError(f"{node}_EXECUTOR_EXIT_{launch['exit_code']}_WITHOUT_TERMINAL")
        if launch["interrupted"]:
            return {**after, "status": "INTERRUPTED_AFTER_FENCED_CHILD_TERMINAL",
                    "last_node_status": after["status"], "nodes_launched_here": launched,
                    "code_acceptance": "PASSED", "code_acceptance_result": accepted,
                    "child_pid": launch["child_pid"],
                    "child_start_ticks": launch["child_start_ticks"]}
        if after.get("node") == node:
            if after.get("status") == "READY":
                raise ColdStartInputError(f"{node}_UNSEALED_AFTER_EXECUTOR")
            return {**after, "code_acceptance": "PASSED",
                    "nodes_launched_here": launched}
    return {**plan_one(source, expected_input_sha256), "code_acceptance": "PASSED",
            "code_acceptance_result": accepted, "nodes_launched_here": launched}


__all__ = ["frozen_manifest", "accepted_code_for_execution", "plan_one",
           "plan_manifest", "submit_one"]
