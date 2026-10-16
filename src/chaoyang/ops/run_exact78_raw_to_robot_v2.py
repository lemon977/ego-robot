"""Fenced, single-session Raw RGB -> HaWoR -> Robot diagnostic candidate.

This is a successor to the T1-A input preflight, not a production route.  It
never reads historical session motion or hardware hand measurements.  Clean,
metric scene geometry, and compositing require independent later packets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from chaoyang.cli.exact78_submit import build_request
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    ColdStartInputError, PROJECT_ROOT, file_ref, stage_signature, validate_output_path,
)
from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


T0_ROOT = PROJECT_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
T0_FREEZE = T0_ROOT / "INPUT_FREEZE.json"
T0_MAP = T0_ROOT / "FRAME_TIME_MAP.json"
T0_FREEZE_SHA = "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88"
T0_MAP_SHA = "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a"
HAWOR_ENV = PROJECT_ROOT / "_run/current/environments/hawor-py310-v1"
HAWOR_PYTHON = HAWOR_ENV / "bin/python"
STRACE = Path("/usr/bin/strace")


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _active_attempt() -> tuple[dict[str, Any], Path]:
    index = json.loads((PROJECT_ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active) != 1:
        raise ColdStartInputError("CANDIDATE_PACKET_REQUIRED")
    packet_path = PROJECT_ROOT / active[0]["packet_path"]
    if file_ref(packet_path)["sha256"] != active[0]["packet_sha256"]:
        raise ColdStartInputError("ACTIVE_PACKET_SHA_DRIFT")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("execution_mode") != "EXACT78_CANDIDATE":
        raise ColdStartInputError("CANDIDATE_PACKET_REQUIRED")
    attempts = [(Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw).resolve()
                for raw in packet.get("write_set", []) if Path(raw).name.startswith("attempt_")]
    if (len(attempts) != 1
            or attempts[0].parent != PROJECT_ROOT / "_run/current" / packet["task_id"] / "attempts"):
        raise ColdStartInputError("CANDIDATE_ATTEMPT_WRITE_SET")
    return packet, attempts[0]


def _source_kind(request: dict[str, Any]) -> str:
    sources = request["sources"]
    if len(sources) != 1:
        raise ColdStartInputError("T1B_SINGLE_SESSION_ONLY")
    kind = sources[0]["kind"]
    if kind == "TRACKER_CLIP_RGB":
        return "tracker"
    if kind == "0915_RAW_VST":
        return "vst0915"
    raise ColdStartInputError("UNKNOWN_SOURCE_KIND")


def _assert_t0_sha() -> None:
    if file_ref(T0_FREEZE)["sha256"] != T0_FREEZE_SHA:
        raise ColdStartInputError("T0_INPUT_FREEZE_SHA_DRIFT")
    if file_ref(T0_MAP)["sha256"] != T0_MAP_SHA:
        raise ColdStartInputError("T0_FRAME_TIME_MAP_SHA_DRIFT")


def _audit_native_source_opens(trace: Path, source: dict[str, Any], output: Path) -> dict[str, Any]:
    """Check OS-observed HaWoR opens, including OpenCV/native-code reads."""
    permitted = {str(Path(ref["path"]).resolve()) for ref in source["files"].values()}
    data_root = "/path/to/private_data/egodata/"
    old_result_prefix = str(PROJECT_ROOT / "_run/current/exact78_")
    opened_source: set[str] = set()
    forbidden: set[str] = set()
    calls = 0
    quoted = re.compile(r'"((?:[^"\\]|\\.)*)"')
    with trace.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "open(" not in line and "openat(" not in line and "openat2(" not in line:
                continue
            if " = -1 " in line:
                continue
            calls += 1
            match = quoted.search(line)
            if not match:
                continue
            raw = match.group(1)
            write_open = any(flag in line for flag in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC"))
            if raw.startswith(data_root):
                resolved = str(Path(raw).resolve(strict=False))
                if write_open:
                    forbidden.add("EXTERNAL_WRITE_OPEN:" + resolved)
                elif resolved in permitted:
                    opened_source.add(resolved)
                else:
                    forbidden.add(resolved)
            if write_open:
                continue
            if ("preprocess/all_data" in raw or "trackingData" in raw
                    or "pico.jsonl" in raw or "manus" in raw.lower()
                    or (raw.startswith(old_result_prefix)
                        and not raw.startswith(str(T0_ROOT))
                        and not raw.startswith(str(output.parent.resolve())))):
                forbidden.add(raw)
    source_video = source["files"]["raw/vst.h264" if source["kind"] == "0915_RAW_VST" else "video"]["path"]
    status = ("PASS_NATIVE_OPEN_SCOPE" if source_video in opened_source and not forbidden
              else "FAILED_OR_INCOMPLETE_NATIVE_OPEN_SCOPE")
    value = {"schema_version": "exact78-native-open-audit-v1", "status": status,
             "trace": file_ref(trace), "successful_open_calls_seen": calls,
             "permitted_source_refs": source["files"],
             "opened_source_paths": sorted(opened_source), "forbidden_paths": sorted(forbidden),
             "source_video_open_observed": source_video in opened_source,
             "scope_limit": "OS open/openat/openat2 tracing of HaWoR child only; does not prove model accuracy or time synchronization."}
    _write_new(output, value)
    return value


def run_candidate(*, session: Path, output: Path) -> dict[str, Any]:
    """Execute one frozen session, sealing each valid stage before consumption."""
    packet, attempt = _active_attempt()
    output = validate_output_path(output, attempt_root=attempt)
    if output.exists():
        raise ColdStartInputError("FRESH_OUTPUT_REQUIRED")
    _assert_t0_sha()
    request = build_request(session=session)
    if packet.get("inputs", {}).get("submission_input_sha256") != request["submission_input_sha256"]:
        raise ColdStartInputError("SUBMISSION_INPUT_SHA_DRIFT")
    source = request["sources"][0]
    if source["session_root"] != str(session.resolve(strict=True)):
        raise ColdStartInputError("SESSION_ROOT_IDENTITY_DRIFT")

    output.mkdir(parents=True)
    _write_new(output / "INPUT_REQUEST.json", request)
    stage_status = {name: "NOT_EVALUATED" for name in
                    ("hawor", "robot", "clean", "depth", "object6d", "composite")}
    stages: dict[str, Any] = {}
    try:
        # HaWoR's pinned environment contains smplx while the rendering
        # environment contains numba. Both stages are children of this one
        # fenced CLI operation, with an explicit immutable stage boundary.
        if not HAWOR_PYTHON.is_file():
            raise RuntimeError("PINNED_HAWOR_PYTHON_ABSENT")
        hawor_output = output / "hawor"
        command = [str(HAWOR_PYTHON), "-B", "-m", "chaoyang.ops.run_exact78_hawor_raw_v3",
            "--session-root", str(session), "--session-id", source["session_id"],
            "--source-kind", _source_kind(request),
            "--input-freeze", str(T0_FREEZE), "--frame-time-map", str(T0_MAP),
            "--output-root", str(hawor_output),
        ]
        environment = os.environ.copy()
        for name in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_EXE",
                     "CONDA_SHLVL", "CONDA_PROMPT_MODIFIER", "_CE_CONDA", "_CE_M", "LD_PRELOAD"):
            environment.pop(name, None)
        environment.update(PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONNOUSERSITE="1",
                           PYTHONDONTWRITEBYTECODE="1", CONDA_PREFIX=str(HAWOR_ENV),
                           PATH=f"{HAWOR_ENV / 'bin'}:/usr/bin:/bin",
                           LD_LIBRARY_PATH=str(HAWOR_ENV / "lib"),
                           XDG_CACHE_HOME=str(output / "hawor_cache"),
                           HF_HOME=str(output / "hawor_cache/huggingface"),
                           MPLCONFIGDIR=str(output / "hawor_cache/matplotlib"),
                           NUMBA_CACHE_DIR=str(output / "hawor_cache/numba"),
                           TORCH_EXTENSIONS_DIR=str(output / "hawor_cache/torch-extensions"),
                           YOLO_CONFIG_DIR=str(output / "hawor_cache/ultralytics"),
                           TMPDIR=str(output / "hawor_cache"), TMP=str(output / "hawor_cache"),
                           TEMP=str(output / "hawor_cache"))
        (output / "hawor_cache").mkdir()
        trace = output / "hawor_open_trace.log"
        if not STRACE.is_file():
            raise RuntimeError("NATIVE_OPEN_TRACER_ABSENT")
        traced_command = [str(STRACE), "-f", "-s", "4096", "-e",
                          "trace=open,openat,openat2", "-o", str(trace), *command]
        with (output / "hawor_worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(traced_command, cwd=PROJECT_ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"HAWOR_STAGE_EXIT_{process.returncode}")
        native_audit = _audit_native_source_opens(trace, source, output / "NATIVE_OPEN_AUDIT.json")
        if native_audit["status"] != "PASS_NATIVE_OPEN_SCOPE":
            raise RuntimeError("HAWOR_NATIVE_OPEN_AUDIT_FAILED")
        hawor_result = hawor_output / "RESULT.json"
        if not hawor_result.is_file():
            raise RuntimeError("HAWOR_STAGE_RESULT_ABSENT")
        hawor_value = json.loads(hawor_result.read_text(encoding="utf-8"))
        for key in ("npz", "frame_provenance", "input_audit"):
            ref = hawor_value.get(key)
            if not isinstance(ref, dict) or file_ref(Path(ref["path"])) != ref:
                raise RuntimeError(f"HAWOR_STAGE_SHA_DRIFT:{key}")
        stage_status["hawor"] = "SEALED_CAMERA_PREDICTIONS"
        stages["hawor"] = file_ref(hawor_result)
        stages["hawor_native_open_audit"] = file_ref(output / "NATIVE_OPEN_AUDIT.json")
        hawor_seal = {
            "schema_version": "exact78-raw-hawor-stage-seal-v1",
            "session_id": source["session_id"], "result": stages["hawor"],
            "native_open_audit": stages["hawor_native_open_audit"],
            "signature": stage_signature(
                inputs=[*source["files"].values(), file_ref(T0_FREEZE), file_ref(T0_MAP)],
                code=[ref for ref in packet["runtime_code"]
                      if ref["path"].endswith("run_exact78_hawor_raw_v3.py")],
                config={"source_kind": _source_kind(request), "frame_mapping": "T0_EXACT",
                        "infiller": False, "output_coordinate_frame": "CAMERA"},
                weights=packet["inputs"]["assets"][:4],
                calibration=source["files"].get("camera_params", source["files"].get("raw/camera_params.json")),
                schema_version="exact78-hawor-stage-v3"),
            "downstream_use": "SEALED_STAGE_ONLY",
        }
        _write_new(output / "HAWOR_STAGE_SEAL.json", hawor_seal)
        stages["hawor_seal"] = file_ref(output / "HAWOR_STAGE_SEAL.json")

        from chaoyang.pipeline.exact78_fresh_robot_stage_v1 import run_robot_stage

        robot_output = output / "robot"
        robot_result = run_robot_stage(
            hawor_npz=Path(hawor_value["npz"]["path"]),
            frame_provenance=Path(hawor_value["frame_provenance"]["path"]),
            input_audit=Path(hawor_value["input_audit"]["path"]),
            raw_video=Path(source["files"]["raw/vst.h264" if _source_kind(request) == "vst0915"
                                           else "video"]["path"]),
            source_kind=_source_kind(request), session_id=source["session_id"],
            output_root=robot_output,
        )
        robot_result_path = robot_output / "RESULT.json"
        if not robot_result_path.is_file() or not isinstance(robot_result, dict):
            raise RuntimeError("ROBOT_STAGE_RESULT_ABSENT")
        stage_status["robot"] = "SEALED_CAMERA_DIAGNOSTIC"
        stages["robot"] = file_ref(robot_result_path)
        _write_new(output / "ROBOT_STAGE_SEAL.json", {
            "schema_version": "exact78-raw-robot-stage-seal-v1",
            "session_id": source["session_id"], "result": stages["robot"],
            "input_hawor_seal": stages["hawor_seal"],
            "signature": stage_signature(
                inputs=[stages["hawor_seal"], stages["hawor"]],
                code=[ref for ref in packet["runtime_code"] if ref["path"].endswith(
                    ("exact78_fresh_robot_stage_v1.py", "exact78_robot_raster_v1.py"))],
                config={"base_rule": "same_frozen_grid_for_all_sources_v1",
                        "coordinate_frame": "CAMERA_DIAGNOSTIC_FIXED"},
                weights="NO_ROBOT_MODEL_WEIGHTS", calibration="WORLD_POSE_UNVERIFIED",
                schema_version="exact78-robot-stage-v1"),
        })
        stages["robot_seal"] = file_ref(output / "ROBOT_STAGE_SEAL.json")
        status = "CAMERA_DIAGNOSTIC_STAGES_COMPLETE"
        error = None
    except Exception as exc:
        status = "FAILED_RUNTIME_FINAL"
        error = {"type": type(exc).__name__, "message": str(exc)}

    result = {
        "schema_version": "exact78-raw-to-robot-candidate-result-v2",
        "status": status, "session_id": source["session_id"],
        "submission_input_sha256": request["submission_input_sha256"],
        "input_request": file_ref(output / "INPUT_REQUEST.json"),
        "stages": stages, "stage_status": stage_status, "error": error,
        "execution_complete": status == "CAMERA_DIAGNOSTIC_STAGES_COMPLETE",
        "robot_video_complete": stage_status["robot"] == "SEALED_CAMERA_DIAGNOSTIC",
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "general_runnable_operation": None, "control_ground_truth": False,
        "claim_limit": "Single-session RGB-derived camera diagnostic only; Clean, geometry, visual qualification and product qualification remain separate.",
    }
    _write_new(output / "RESULT.json", result)
    return result


def main() -> int:
    authorize_exact78_operation("run_exact78_raw_to_robot_v2", sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_candidate(session=args.session, output=args.output_root)
    print(json.dumps({"status": result["status"],
                      "result": file_ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0 if result["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
