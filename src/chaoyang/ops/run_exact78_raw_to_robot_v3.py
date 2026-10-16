"""Frozen T0 six-session Exact78 fresh RGB -> HaWoR -> Robot candidate.

Four development and two preselected cross-group holdout captures share the
same code and rule. Holdouts may not tune the rule after execution. No prior
HaWoR arrays, hardware hand measurements, special ROI/masks, or failed Robot
partial are inputs. This is not a production route.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_raw_to_robot_v3"
TARGETS = {
    "get_potato_chips_0901_013": {
        "kind": "TRACKER_CLIP_RGB", "frames": 870, "role": "development",
        "time_mode": "TRACKER_CLIP_METADATA_ONLY", "physical_eye": "tracker_session_left_RGB",
    },
    "get_potato_chips_0902_103": {
        "kind": "TRACKER_CLIP_RGB", "frames": 284, "role": "development",
        "time_mode": "TRACKER_CLIP_METADATA_ONLY", "physical_eye": "tracker_session_left_RGB",
    },
    "play_cards_0901_001": {
        "kind": "TRACKER_CLIP_RGB", "frames": 645, "role": "holdout",
        "time_mode": "TRACKER_CLIP_METADATA_ONLY", "physical_eye": "tracker_session_left_RGB",
    },
    "play_cards_0902_042": {
        "kind": "TRACKER_CLIP_RGB", "frames": 171, "role": "development",
        "time_mode": "TRACKER_CLIP_METADATA_ONLY", "physical_eye": "tracker_session_left_RGB",
    },
    "play_cards_0915_001": {
        "kind": "0915_RAW_VST", "frames": 150, "role": "development",
        "time_mode": "TIMEBASE_UNVERIFIED", "physical_eye": "left_sourceIndex_1",
    },
    "play_cards_0915_002": {
        "kind": "0915_RAW_VST", "frames": 131, "role": "holdout",
        "time_mode": "TIMEBASE_UNVERIFIED", "physical_eye": "left_sourceIndex_1",
    },
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _validate_t0_target(source: dict[str, Any], *, freeze: Path,
                        frame_map: Path) -> dict[str, Any]:
    """Admit only the six preselected T0 captures, preserving role/clock limits."""
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref

    session_id = source.get("session_id")
    expected = TARGETS.get(session_id)
    if expected is None:
        raise ValueError("T0_FROZEN_TARGET_NOT_AUTHORIZED")
    selected = _read_object(freeze).get("selected", {}).get(session_id)
    mapped = _read_object(frame_map).get("sessions", {}).get(session_id)
    if not isinstance(selected, dict) or not isinstance(mapped, dict):
        raise ValueError("T0_TARGET_MAP_ABSENT")
    if (selected.get("role") != expected["role"]
            or selected.get("source_path") != source.get("session_root")
            or source.get("kind") != expected["kind"]
            or source.get("physical_eye") != expected["physical_eye"]
            or source.get("time_mode") != expected["time_mode"]
            or mapped.get("decoded_frames") != expected["frames"]
            or mapped.get("time_mode") != expected["time_mode"]):
        raise ValueError("T0_TARGET_IDENTITY_OR_TIME_MODE_DRIFT")
    role = "video" if expected["kind"] == "TRACKER_CLIP_RGB" else "raw/vst.h264"
    if mapped.get("source_video") != source["files"][role]:
        raise ValueError("T0_RAW_VIDEO_SHA_DRIFT")
    rows = mapped.get("frame_rows")
    if (not isinstance(rows, list) or len(rows) != expected["frames"]
            or any(row.get("decoded_display_index") != index
                   or row.get("capture_to_decode_alignment")
                      != mapped.get("capture_to_decode_alignment")
                   for index, row in enumerate(rows))):
        raise ValueError("T0_FRAME_INDEX_MAP_DRIFT")
    if expected["kind"] == "0915_RAW_VST":
        if (mapped.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
                or any(mapped.get(field) != "UNVERIFIED" for field in
                       ("qpc_unit", "qpc_frequency", "qpc_sampling_location",
                        "head_pose_clock_domain"))
                or any(row.get("sbs_physical_eye") != "left_sourceIndex_1"
                       or row.get("crop_xywh") != source.get("crop_xywh")
                       or not isinstance(row.get("qpc_raw_tick"), int)
                       for row in rows)):
            raise ValueError("0915_TIMEBASE_OR_PHYSICAL_EYE_CLAIM_DRIFT")
    if file_ref(Path(source["files"][role]["path"])) != source["files"][role]:
        raise ValueError("RAW_VIDEO_SHA_DRIFT")
    return {"session_id": session_id, "source_kind": source["kind"],
            "source_role": expected["role"], "frame_count": expected["frames"],
            "time_mode": expected["time_mode"], "physical_eye": expected["physical_eye"],
            "capture_to_decode_alignment": mapped["capture_to_decode_alignment"],
            "raw_video": source["files"][role]}


def _hawor_environment(output: Path, model_env: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_EXE",
                 "CONDA_SHLVL", "CONDA_PROMPT_MODIFIER", "_CE_CONDA", "_CE_M",
                 "LD_PRELOAD"):
        environment.pop(name, None)
    cache = output / "hawor_cache"
    cache.mkdir()
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[3] / "src"),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        CONDA_PREFIX=str(model_env), PATH=f"{model_env / 'bin'}:/usr/bin:/bin",
        LD_LIBRARY_PATH=str(model_env / "lib"),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
        YOLO_CONFIG_DIR=str(cache / "ultralytics"),
        TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
    )
    return environment


def _validate_hawor_stage_claims(provenance: dict[str, Any], audit: dict[str, Any],
                                 t0_rows: list[dict[str, Any]],
                                 target: dict[str, Any], kind: str) -> None:
    """Do not let a model stage silently upgrade raw eye, time or world claims."""
    model_time = ("CLIP_FRAME_INDEX_ONLY" if kind == "tracker"
                  else "TIMEBASE_UNVERIFIED")
    rows = provenance.get("frame_rows")
    if (provenance.get("schema_version") != "exact78-hawor-frame-provenance-v3"
            or provenance.get("session_id") != target["session_id"]
            or provenance.get("physical_eye") != target["physical_eye"]
            or provenance.get("time_mode") != model_time
            or not isinstance(rows, list) or len(rows) != target["frame_count"]
            or len(t0_rows) != len(rows)):
        raise ValueError("HAWOR_SOURCE_FRAME_TIME_CLAIM_DRIFT")
    if (audit.get("schema_version") != "exact78-hawor-input-audit-v3"
            or audit.get("session_id") != target["session_id"]
            or audit.get("source_kind") != kind
            or audit.get("raw_source_ref") != target["raw_video"]
            or audit.get("world_pose_emitted") is not False):
        raise ValueError("HAWOR_INPUT_OR_WORLD_CLAIM_DRIFT")
    for index, (row, original) in enumerate(zip(rows, t0_rows)):
        if (row.get("decoded_display_index") != index
                or row.get("model_input_index") != index
                or row.get("output_index") != index
                or row.get("sbs_physical_eye") != target["physical_eye"]
                or row.get("crop_xywh") != original.get("crop_xywh")
                or row.get("time_mode") != model_time
                or any(row.get(field) != original.get(field) for field in
                       ("qpc_raw_tick", "capture_timestamp_raw", "pts",
                        "best_effort_timestamp", "capture_to_decode_alignment"))):
            raise ValueError(f"HAWOR_TO_T0_FRAME_PROVENANCE_DRIFT:{index}")


def run_candidate(*, session: Path, output: Path) -> dict[str, Any]:
    """Run one fresh model stage and consume it only after a SHA-verified seal."""
    # This import only supplies already-frozen generic governance/source helpers.
    # It never invokes v2.run_candidate or reads its Cards042 result/partial.
    from chaoyang.ops import run_exact78_raw_to_robot_v2 as prior
    from chaoyang.cli.exact78_submit import build_request
    from chaoyang.governance.exact78_cold_start_guard_v1 import (
        file_ref, stage_signature, validate_output_path,
    )

    packet, attempt = prior._active_attempt()
    output = validate_output_path(output, attempt_root=attempt)
    if output.exists():
        raise ValueError("FRESH_OUTPUT_REQUIRED")
    prior._assert_t0_sha()
    request = build_request(session=session)
    if len(request["sources"]) != 1:
        raise ValueError("SINGLE_T0_SESSION_REQUIRED")
    source = request["sources"][0]
    target = _validate_t0_target(source, freeze=prior.T0_FREEZE,
                                 frame_map=prior.T0_MAP)
    if packet.get("inputs", {}).get("submission_input_sha256") != request["submission_input_sha256"]:
        raise ValueError("SUBMISSION_INPUT_SHA_DRIFT")
    if packet.get("inputs", {}).get("source_files") != source["files"]:
        raise ValueError("PACKET_SOURCE_FILE_SET_DRIFT")
    if packet.get("inputs", {}).get("raw_video") != target["raw_video"]:
        raise ValueError("PACKET_RAW_VIDEO_SHA_DRIFT")
    if not prior.HAWOR_PYTHON.is_file() or not prior.STRACE.is_file():
        raise ValueError("PINNED_HAWOR_RUNTIME_OR_NATIVE_TRACER_ABSENT")

    output.mkdir(parents=True)
    prior._write_new(output / "INPUT_REQUEST.json", request)
    prior._write_new(output / "T0_TARGET_AUDIT.json", target)
    stage_status = {name: "NOT_EVALUATED" for name in
                    ("hawor", "robot", "clean", "depth", "object6d", "composite")}
    stages: dict[str, Any] = {}
    try:
        hawor_output = output / "hawor"
        kind = prior._source_kind(request)
        command = [str(prior.HAWOR_PYTHON), "-B", "-m", "chaoyang.ops.run_exact78_hawor_raw_v3",
                   "--session-root", str(session), "--session-id", source["session_id"],
                   "--source-kind", kind, "--input-freeze", str(prior.T0_FREEZE),
                   "--frame-time-map", str(prior.T0_MAP),
                   "--output-root", str(hawor_output)]
        trace = output / "hawor_open_trace.log"
        traced = [str(prior.STRACE), "-f", "-s", "4096", "-e",
                  "trace=open,openat,openat2", "-o", str(trace), *command]
        with (output / "hawor_worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(traced, cwd=prior.PROJECT_ROOT,
                                     env=_hawor_environment(output, prior.HAWOR_ENV),
                                     stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"HAWOR_STAGE_EXIT_{process.returncode}")
        native = prior._audit_native_source_opens(
            trace, source, output / "NATIVE_OPEN_AUDIT.json")
        if native["status"] != "PASS_NATIVE_OPEN_SCOPE":
            raise RuntimeError("HAWOR_NATIVE_OPEN_AUDIT_FAILED")
        hawor_result_path = hawor_output / "RESULT.json"
        hawor_result = _read_object(hawor_result_path)
        if (hawor_result.get("schema_version") != "exact78-hawor-pure-rgb-result-v3"
                or hawor_result.get("status") != "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS"
                or hawor_result.get("session_id") != source["session_id"]
                or hawor_result.get("source_kind") != kind):
            raise RuntimeError("HAWOR_STAGE_IDENTITY_DRIFT")
        for key in ("npz", "frame_provenance", "input_audit"):
            ref = hawor_result.get(key)
            if not isinstance(ref, dict) or file_ref(Path(ref["path"])) != ref:
                raise RuntimeError(f"HAWOR_STAGE_SHA_DRIFT:{key}")
        _validate_hawor_stage_claims(
            _read_object(Path(hawor_result["frame_provenance"]["path"])),
            _read_object(Path(hawor_result["input_audit"]["path"])),
            _read_object(prior.T0_MAP)["sessions"][source["session_id"]]["frame_rows"],
            target, kind)
        stages["hawor"] = file_ref(hawor_result_path)
        stages["hawor_native_open_audit"] = file_ref(output / "NATIVE_OPEN_AUDIT.json")
        stage_status["hawor"] = "SEALED_CAMERA_PREDICTIONS"
        seal = {
            "schema_version": "exact78-raw-hawor-stage-seal-v1",
            "session_id": source["session_id"], "result": stages["hawor"],
            "native_open_audit": stages["hawor_native_open_audit"],
            "signature": stage_signature(
                inputs=[*source["files"].values(), file_ref(prior.T0_FREEZE),
                        file_ref(prior.T0_MAP)],
                code=[ref for ref in packet["runtime_code"] if ref["path"].endswith(
                    "run_exact78_hawor_raw_v3.py")],
                config={"source_kind": kind, "frame_mapping": "T0_EXACT",
                        "infiller": False, "output_coordinate_frame": "CAMERA"},
                weights=packet["inputs"]["assets"][:4],
                calibration=source["files"].get("camera_params", source["files"].get(
                    "raw/camera_params.json")), schema_version="exact78-hawor-stage-v3"),
            "downstream_use": "SEALED_STAGE_ONLY",
        }
        prior._write_new(output / "HAWOR_STAGE_SEAL.json", seal)
        stages["hawor_seal"] = file_ref(output / "HAWOR_STAGE_SEAL.json")

        from chaoyang.pipeline.exact78_fresh_robot_stage_v2 import run_robot_stage_ffmpeg44

        robot_output = output / "robot"
        robot_result = run_robot_stage_ffmpeg44(
            hawor_npz=Path(hawor_result["npz"]["path"]),
            frame_provenance=Path(hawor_result["frame_provenance"]["path"]),
            input_audit=Path(hawor_result["input_audit"]["path"]),
            raw_video=Path(target["raw_video"]["path"]), source_kind=kind,
            session_id=source["session_id"], output_root=robot_output)
        if (robot_result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
                or robot_result.get("frame_count") != target["frame_count"]
                or not (robot_output / "RESULT.json").is_file()):
            raise RuntimeError("ROBOT_STAGE_RESULT_UNVERIFIED")
        stages["robot"] = file_ref(robot_output / "RESULT.json")
        stage_status["robot"] = "SEALED_CAMERA_DIAGNOSTIC"
        prior._write_new(output / "ROBOT_STAGE_SEAL.json", {
            "schema_version": "exact78-raw-robot-stage-seal-v2",
            "session_id": source["session_id"], "result": stages["robot"],
            "input_hawor_seal": stages["hawor_seal"],
            "signature": stage_signature(
                inputs=[stages["hawor_seal"], stages["hawor"]],
                code=[ref for ref in packet["runtime_code"] if ref["path"].endswith((
                    "exact78_fresh_robot_stage_v1.py",
                    "exact78_fresh_robot_stage_v2.py", "exact78_robot_raster_v1.py"))],
                config={"base_rule": "same_frozen_grid_for_all_sources_v1",
                        "coordinate_frame": "CAMERA_DIAGNOSTIC_FIXED",
                        "encoder": "FFMPEG44_VSYNC_0"},
                weights="NO_ROBOT_MODEL_WEIGHTS", calibration="WORLD_POSE_UNVERIFIED",
                schema_version="exact78-robot-stage-v2"),
        })
        stages["robot_seal"] = file_ref(output / "ROBOT_STAGE_SEAL.json")
        status, error = "CAMERA_DIAGNOSTIC_STAGES_COMPLETE", None
    except Exception as exc:
        status = "FAILED_RUNTIME_FINAL"
        error = {"type": type(exc).__name__, "message": str(exc)}

    result = {
        "schema_version": "exact78-raw-to-robot-candidate-result-v3",
        "status": status, "session_id": source["session_id"],
        "submission_input_sha256": request["submission_input_sha256"],
        "input_request": file_ref(output / "INPUT_REQUEST.json"),
        "t0_target_audit": file_ref(output / "T0_TARGET_AUDIT.json"),
        "stages": stages, "stage_status": stage_status, "error": error,
        "execution_complete": status == "CAMERA_DIAGNOSTIC_STAGES_COMPLETE",
        "robot_video_complete": stage_status["robot"] == "SEALED_CAMERA_DIAGNOSTIC",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "general_runnable_operation": None,
        "claim_limit": "One of six T0 preselected captures only (four development, two cross-group holdout) under one frozen RGB rule. Camera diagnostic; 0915 timebase and all metric/world, occlusion, Clean, product, and training qualifications remain separate.",
    }
    prior._write_new(output / "RESULT.json", result)
    return result


def main() -> int:
    authorize_exact78_operation(OPERATION, sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_candidate(session=args.session, output=args.output_root)
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref

    print(json.dumps({"status": result["status"],
                      "result": file_ref(args.output_root / "RESULT.json")},
                     ensure_ascii=False))
    return 0 if result["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
