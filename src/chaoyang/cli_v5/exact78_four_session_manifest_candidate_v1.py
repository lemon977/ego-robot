"""Fixed four-session VST admission design; read-only until a new finite executor is accepted."""

from __future__ import annotations

import hmac
from pathlib import Path
import re
from typing import Any

from chaoyang.cli_v4 import exact78_raw_to_robot_manifest_candidate_v1 as v4
from chaoyang.cli_v3 import exact78_newraw_cpu_submit_phase_exact_v2 as phase_exact
from chaoyang.governance.common import (
    REPO_ROOT, TASK_STATE_PATH, artifact_ref, load_json, validate_artifact_ref,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


MANIFEST = (REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930"
            / "48_FOUR_SESSION_CLI_V5_MANIFEST_CANDIDATE_V1.json")
MANIFEST_SHA = "01ef3da3b7174585752cf497e6f45e392942126b99b4d381f5a12898e87c9387"
RAW_ROLES = frozenset({"raw/vst.h264", "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl",
                       "raw/camera_params.json", "raw/camera_params.meta.json"})
VST_ROOT = "/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915"
EXPECTED = {
    "play_cards_0915_003": ("003", 145, "291ed851e8e98b196783801a4421826552713fbc7c7d8bc984c8e2f4d925974f"),
    "play_cards_0915_004": ("004", 152, "fbd99999a28849a3f47ccedf43912f955afa720c2ae94ccaa93c25c47a07ae41"),
}
STAGE_IDS_003 = (
    "exact78_newraw_play_cards_0915_003_12e2d890c490fbae_v1",
    "exact78_newraw_hawor_play_cards_0915_003_12e2d890c490fbae_v4",
    "exact78_newraw_robot_play_cards_0915_003_12e2d890c490fbae_v1",
)


def frozen_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MANIFEST.is_file():
        raise ColdStartInputError("CLI_V5_MANIFEST_MISSING")
    ref = artifact_ref(MANIFEST)
    if not hmac.compare_digest(ref["sha256"], MANIFEST_SHA):
        raise ColdStartInputError("CLI_V5_MANIFEST_SHA_DRIFT")
    value = load_json(MANIFEST)
    rows = value.get("sessions")
    if (value.get("schema_version") != "exact78-newraw-cli-v5-four-session-manifest-candidate-v1"
            or not isinstance(rows, list) or len(rows) != 4
            or not isinstance(value.get("source_domains"), dict)):
        raise ColdStartInputError("CLI_V5_MANIFEST_SCHEMA")
    old, _ = v4.frozen_manifest()
    if rows[:2] != old["sessions"]:
        raise ColdStartInputError("CLI_V5_TRACKER_ROWS_DRIFT")
    if [row.get("session_id") for row in rows[2:] if isinstance(row, dict)] != list(EXPECTED):
        raise ColdStartInputError("CLI_V5_VST_SESSION_SET_DRIFT")
    domain = value["source_domains"].get("0915_raw_vst") or {}
    if (domain.get("source_kind") != "0915_RAW_VST"
            or domain.get("physical_eye") != "left_sourceIndex_1"
            or domain.get("crop") != "SBS_RIGHT_HALF_IS_PHYSICAL_LEFT_BY_CAMERA_METADATA"
            or domain.get("time_mode") != "TIMEBASE_UNVERIFIED"
            or domain.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
            or domain.get("glove_motion_input") != "FORBIDDEN"
            or domain.get("execution_status") != "TWO_FROZEN_PLANNING_ROWS_NOT_EXECUTION_ADMITTED"):
        raise ColdStartInputError("CLI_V5_VST_DOMAIN_DRIFT")
    for row in rows[2:]:
        suffix, frames, digest = EXPECTED[row["session_id"]]
        if (row.get("source_root") != f"{VST_ROOT}/{suffix}"
                or row.get("submission_input_sha256") != digest
                or row.get("source_kind") != "0915_RAW_VST"
                or row.get("physical_eye") != "left_sourceIndex_1"
                or row.get("crop_xywh") != [2048, 0, 2048, 1536]
                or row.get("time_mode") != "TIMEBASE_UNVERIFIED"
                or row.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
                or row.get("frame_count_decoded") != frames
                or row.get("fps_nominal") != 30
                or row.get("independent_source_group") != "UNVERIFIED"
                or row.get("head_pose_source") != "ABSENT_UNTIL_FIELD_PROJECTION_AND_CLOCK_PROOF"
                or row.get("control_ground_truth") is not False):
            raise ColdStartInputError("CLI_V5_VST_IDENTITY_OR_TIME_DRIFT")
        if row["session_id"].endswith("004"):
            refs = row.get("source_files")
            if not isinstance(refs, dict) or set(refs) != RAW_ROLES:
                raise ColdStartInputError("CLI_V5_RAW_ROLE_SET_DRIFT")
            for role, source_ref in refs.items():
                if (not isinstance(source_ref, dict)
                        or source_ref.get("path") != f"{VST_ROOT}/004/{role}"
                        or validate_artifact_ref(source_ref)):
                    raise ColdStartInputError("CLI_V5_RAW_ROLE_SHA_DRIFT:" + role)
            if "sealed_robot_terminal" in row:
                raise ColdStartInputError("CLI_V5_FRESH_SESSION_HAS_TERMINAL")
        elif set(row.get("sealed_robot_terminal") or {}) != {"task_packet", "result", "receipt"}:
            raise ColdStartInputError("CLI_V5_003_TERMINAL_ABSENT")
    return value, ref


def _entry(session: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value, ref = frozen_manifest()
    rows = [row for row in value["sessions"] if row["source_root"] == str(session)]
    if (len(rows) != 1 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not hmac.compare_digest(rows[0]["submission_input_sha256"], expected_sha)):
        raise ColdStartInputError("CLI_V5_SESSION_NOT_IN_FROZEN_MANIFEST")
    return rows[0], ref


def _sealed_raw_stage(task_id: str, phase: str, result_schema: str,
                      receipt_schema: str, row: dict[str, Any],
                      expected_refs: dict[str, Any] | None = None,
                      ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Attach a 0915 root named 003 without confusing it with the session ID."""
    state, index = load_json(TASK_STATE_PATH), load_json(v4.prior.INDEX)
    matches = [item for item in state.get("tasks", []) if item.get("task_id") == task_id]
    entries = [item for item in index.get("task_packets", []) if item.get("task_id") == task_id]
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    result_path = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001/RESULT.json"
    receipt_path = REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json"
    if (len(matches) != 1 or len(entries) != 1
            or matches[0].get("phase") != phase
            or matches[0].get("session") != row["session_id"]
            or matches[0].get("status") != "PASSED"
            or entries[0].get("execution_allowed") is not False
            or entries[0].get("packet_path") != str(packet_path.relative_to(REPO_ROOT))
            or entries[0].get("packet_sha256") != matches[0].get("task_packet", {}).get("sha256")):
        raise ColdStartInputError("CLI_V5_003_STAGE_LEDGER_DRIFT:" + phase)
    item = matches[0]
    refs = {"task_packet": v4._ref(packet_path, item.get("task_packet"), "CLI_V5_003_PACKET_DRIFT"),
            "result": v4._ref(result_path, item.get("result"), "CLI_V5_003_RESULT_DRIFT"),
            "receipt": v4._ref(receipt_path, item.get("terminal_receipt"), "CLI_V5_003_RECEIPT_DRIFT")}
    if expected_refs is not None and refs != expected_refs:
        raise ColdStartInputError("CLI_V5_003_FROZEN_ROBOT_TRIPLE_DRIFT")
    packet, result, receipt = load_json(packet_path), load_json(result_path), load_json(receipt_path)
    inputs = packet.get("inputs") or {}
    if (packet.get("task_id") != task_id or packet.get("phase") != phase
            or packet.get("run_signature") != item.get("run_signature")
            or inputs.get("source_session") != row["session_id"]
            or inputs.get("source_root") != row["source_root"]
            or inputs.get("submission_input_sha256") != row["submission_input_sha256"]
            or result.get("schema_version") != result_schema
            or result.get("task_id") != task_id or result.get("status") != "PASSED"
            or result.get("source_session") != row["session_id"]
            or receipt.get("schema_version") != receipt_schema
            or receipt.get("task_id") != task_id or receipt.get("status") != "PASSED"
            or receipt.get("result") != refs["result"]):
        raise ColdStartInputError("CLI_V5_003_STAGE_IDENTITY_DRIFT:" + phase)
    return packet, result, refs


def _sealed_003(row: dict[str, Any]) -> dict[str, Any]:
    cpu_packet, cpu_result, cpu_refs = _sealed_raw_stage(
        STAGE_IDS_003[0], v4.prior.PHASES[0], v4.prior.SCHEMAS[0],
        v4.prior.RECEIPT_SCHEMAS[0], row)
    hawor_packet, hawor_result, hawor_refs = _sealed_raw_stage(
        STAGE_IDS_003[1], v4.prior.PHASES[1], v4.prior.SCHEMAS[1],
        v4.prior.RECEIPT_SCHEMAS[1], row)
    robot_packet, robot_result, robot_refs = _sealed_raw_stage(
        STAGE_IDS_003[2], v4.prior.PHASES[2], v4.prior.SCHEMAS[2],
        v4.prior.RECEIPT_SCHEMAS[2], row,
        expected_refs=row["sealed_robot_terminal"])
    admission = cpu_result.get("admission") or {}
    stage = hawor_result.get("stage") or {}
    render = robot_result.get("stage") or {}
    if (cpu_result.get("source_files") != cpu_packet["inputs"].get("source_files")
            or set(cpu_result.get("source_files") or {}) != RAW_ROLES
            or cpu_result.get("source_files") != hawor_packet["inputs"].get("source_files")
            or cpu_result.get("source_files") != robot_packet["inputs"].get("source_files")
            or cpu_packet["inputs"].get("raw_video") != hawor_packet["inputs"].get("raw_video")
            or cpu_packet["inputs"].get("raw_video") != robot_packet["inputs"].get("raw_video")
            or cpu_result.get("source_sha_stable_at_terminal") is not True
            or hawor_result.get("producer_result") != cpu_refs["result"]
            or hawor_result.get("source_sha_stable_at_terminal") is not True
            or hawor_result.get("stage_status") != "SEALED_CAMERA_PREDICTIONS"
            or hawor_packet["inputs"].get("upstream_admission_result") != cpu_refs["result"]
            or robot_packet["inputs"].get("upstream_admission_result") != cpu_refs["result"]
            or robot_packet["inputs"].get("hawor_outer_result") != hawor_refs["result"]
            or robot_result.get("cpu_admission_result") != cpu_refs["result"]
            or robot_result.get("hawor_outer_result") != hawor_refs["result"]
            or hawor_result.get("source_frames_expected") != 145
            or robot_result.get("source_frames_expected") != 145
            or render.get("decoded_frames") != 145
            or robot_result.get("render_complete") is not True
            or robot_result.get("post_input_sha_errors") != []
            or robot_result.get("general_operation") is not None
            or any(item.get("visual_qualified") is not False or item.get("training_eligible") is not False
                   for item in (cpu_result, hawor_result, robot_result))):
        raise ColdStartInputError("CLI_V5_003_SEALED_CHAIN_DRIFT")
    for key, downstream in (("descriptor", "upstream_admission_descriptor"),
                            ("frame_time_map", "upstream_admission_frame_time_map"),
                            ("seal", "upstream_admission_seal")):
        if (admission.get(key) != hawor_packet["inputs"].get(downstream)
                or admission.get(key) != robot_packet["inputs"].get(downstream)):
            raise ColdStartInputError("CLI_V5_003_ADMISSION_REF_DRIFT:" + key)
    for key, downstream in (("result", "hawor_inner_result"),
                            ("seal", "hawor_stage_seal"),
                            ("native_open_audit", "hawor_native_open_audit")):
        if stage.get(key) != robot_packet["inputs"].get(downstream):
            raise ColdStartInputError("CLI_V5_003_HAWOR_REF_DRIFT:" + key)
    for ref in (*cpu_result["source_files"].values(), *admission.values(),
                *(stage.get(key) for key in ("result", "seal", "native_open_audit")),
                *(render.get(key) for key in ("result", "stage_result", "frame_states"))):
        if not isinstance(ref, dict) or validate_artifact_ref(ref):
            raise ColdStartInputError("CLI_V5_003_REF_SHA_DRIFT")
    videos = render.get("videos")
    if (not isinstance(videos, dict)
            or set(videos) != {"ego_robot", "external_full_robot", "raw_motion_source"}
            or any(not isinstance(ref, dict) or validate_artifact_ref(ref)
                   for ref in videos.values())):
        raise ColdStartInputError("CLI_V5_003_VIDEO_SHA_DRIFT")
    return {"status": "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS",
            "node": "ROBOT_V1", "task_id": STAGE_IDS_003[2], "result": robot_refs["result"],
            "receipt": robot_refs["receipt"], "sealed_attachment": True,
            "visual_qualified": False, "product_qualified": False, "training_eligible": False}


def plan_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    source = v4.prior.legacy_submit._raw_session_path(session)
    row, manifest_ref = _entry(source, expected_input_sha256)
    name = row["session_id"]
    if name == "play_cards_0915_003":
        decision = _sealed_003(row)
    elif name == "play_cards_0915_004":
        decision, packet = phase_exact.plan_one(source, expected_input_sha256)
        prepared = phase_exact.producer.preflight(source)
        descriptor = prepared["descriptor"]
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
            raise ColdStartInputError("CLI_V5_004_FRAME_TIME_MAP_DRIFT")
        if (packet["inputs"].get("source_files") != row["source_files"]
                or descriptor.get("source_files") != row["source_files"]
                or descriptor.get("decoded_frames") != 152
                or descriptor.get("physical_eye") != "left_sourceIndex_1"
                or descriptor.get("crop_xywh") != [2048, 0, 2048, 1536]
                or descriptor.get("time_mode") != "TIMEBASE_UNVERIFIED"
                or descriptor.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"):
            raise ColdStartInputError("CLI_V5_004_CPU_DESCRIPTOR_DRIFT")
    else:
        decision = v4.plan_one(source, expected_input_sha256)
    return {**decision, "candidate_cli": "chaoyang.cli_v5", "manifest": manifest_ref,
            "source_session": name, "registration_started": False, "model_calls": 0,
            "code_acceptance": "NOT_SEALED_CODE_PREP_ONLY"}


def plan_manifest() -> dict[str, Any]:
    value, ref = frozen_manifest()
    decisions = [plan_one(Path(row["source_root"]), row["submission_input_sha256"])
                 for row in value["sessions"]]
    return {"schema_version": "exact78-newraw-cli-v5-four-session-plan-v1",
            "status": "READ_ONLY_MATRIX", "manifest": ref,
            "registration_started": False, "model_calls": 0,
            "batch_execution_authorized": False, "sessions": decisions}


__all__ = ["MANIFEST", "frozen_manifest", "plan_one", "plan_manifest"]
