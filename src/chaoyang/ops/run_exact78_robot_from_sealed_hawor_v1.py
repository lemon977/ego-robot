"""Fenced Robot-only retry from one sealed fresh HaWoR v3 stage.

Only the raw RGB session and the explicitly pinned HaWoR stage are inputs.
This candidate never reads the failed predecessor's partial Robot outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_robot_from_sealed_hawor_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SEALED_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _active_packet() -> tuple[dict[str, Any], Path]:
    # main() has already passed the exact-command/live-writer operation gate.
    from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref

    index = _read_object(PROJECT_ROOT / "tasks/current/INDEX.json")
    active = [row for row in index.get("task_packets", [])
              if row.get("execution_allowed") is True]
    if len(active) != 1:
        raise ValueError("CANDIDATE_PACKET_REQUIRED")
    row = active[0]
    packet_path = PROJECT_ROOT / row["packet_path"]
    if file_ref(packet_path)["sha256"] != row.get("packet_sha256"):
        raise ValueError("ACTIVE_PACKET_SHA_DRIFT")
    packet = _read_object(packet_path)
    if packet.get("execution_mode") != "EXACT78_CANDIDATE":
        raise ValueError("CANDIDATE_PACKET_REQUIRED")
    attempts = [(Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw).resolve()
                for raw in packet.get("write_set", []) if Path(raw).name.startswith("attempt_")]
    if (len(attempts) != 1
            or attempts[0].parent != PROJECT_ROOT / "_run/current" / packet["task_id"] / "attempts"):
        raise ValueError("CANDIDATE_ATTEMPT_WRITE_SET")
    return packet, attempts[0]


def _sealed_hawor(root: Path, packet: dict[str, Any], session_id: str,
                  source_kind: str) -> tuple[Path, Path, Path]:
    """Verify the sealed upstream refs without opening any predecessor Robot path."""
    from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref

    root = root.resolve(strict=True)
    project = PROJECT_ROOT.resolve(strict=True)
    if (root.name != "hawor" or project not in root.parents
            or not root.is_dir()):
        raise ValueError("SEALED_HAWOR_ROOT_IDENTITY")
    inputs = packet.get("inputs") or {}
    sealed = inputs.get("sealed_hawor")
    names = {"RESULT.json": root / "RESULT.json",
             "HAWOR_CAMERA_PREDICTIONS.npz": root / "HAWOR_CAMERA_PREDICTIONS.npz",
             "FRAME_PROVENANCE.json": root / "FRAME_PROVENANCE.json",
             "INPUT_AUDIT.json": root / "INPUT_AUDIT.json"}
    if not isinstance(sealed, dict) or set(sealed) != set(names):
        raise ValueError("SEALED_HAWOR_PACKET_REFS_REQUIRED")
    refs = dict(sealed)
    refs["hawor_seal"] = inputs.get("hawor_seal")
    refs["native_open_audit"] = inputs.get("native_open_audit")
    names["hawor_seal"] = root.parent / "HAWOR_STAGE_SEAL.json"
    names["native_open_audit"] = root.parent / "NATIVE_OPEN_AUDIT.json"
    for key, path in names.items():
        if path.is_symlink() or not isinstance(refs[key], dict) or file_ref(path) != refs[key]:
            raise ValueError(f"SEALED_HAWOR_SHA_OR_PATH_DRIFT:{key}")
    seal = _read_object(names["hawor_seal"])
    result = _read_object(names["RESULT.json"])
    native = _read_object(names["native_open_audit"])
    if (seal.get("schema_version") != "exact78-raw-hawor-stage-seal-v1"
            or seal.get("session_id") != session_id
            or seal.get("result") != refs["RESULT.json"]
            or seal.get("native_open_audit") != refs["native_open_audit"]):
        raise ValueError("SEALED_HAWOR_STAGE_SEAL_DRIFT")
    if native.get("status") != "PASS_NATIVE_OPEN_SCOPE":
        raise ValueError("SEALED_HAWOR_NATIVE_READ_SCOPE_FAILED")
    if (result.get("schema_version") != "exact78-hawor-pure-rgb-result-v3"
            or result.get("status") != "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS"
            or result.get("session_id") != session_id
            or result.get("source_kind") != source_kind
            or result.get("npz") != refs["HAWOR_CAMERA_PREDICTIONS.npz"]
            or result.get("frame_provenance") != refs["FRAME_PROVENANCE.json"]
            or result.get("input_audit") != refs["INPUT_AUDIT.json"]):
        raise ValueError("SEALED_HAWOR_RESULT_DRIFT")
    return (names["HAWOR_CAMERA_PREDICTIONS.npz"],
            names["FRAME_PROVENANCE.json"], names["INPUT_AUDIT.json"])


def run_candidate(*, session: Path, sealed_hawor_root: Path,
                  output_root: Path) -> dict[str, Any]:
    """Verify all inputs, then run the unchanged v1 solver with FFmpeg 4.4 shim."""
    from chaoyang.cli.exact78_submit import build_request
    from chaoyang.governance.exact78_cold_start_guard_v1 import validate_output_path

    packet, attempt = _active_packet()
    output_root = validate_output_path(output_root, attempt_root=attempt)
    if output_root.exists():
        raise ValueError("FRESH_OUTPUT_REQUIRED")
    request = build_request(session=session)
    if len(request["sources"]) != 1:
        raise ValueError("ROBOT_RETRY_SINGLE_SESSION_ONLY")
    source = request["sources"][0]
    if packet.get("inputs", {}).get("submission_input_sha256") != request["submission_input_sha256"]:
        raise ValueError("SUBMISSION_INPUT_SHA_DRIFT")
    kind = "tracker" if source["kind"] == "TRACKER_CLIP_RGB" else "vst0915"
    video_role = "video" if kind == "tracker" else "raw/vst.h264"
    raw_video = Path(source["files"][video_role]["path"])
    if packet.get("inputs", {}).get("raw_video") != source["files"][video_role]:
        raise ValueError("PACKET_RAW_VIDEO_SHA_DRIFT")
    npz, provenance, audit = _sealed_hawor(
        sealed_hawor_root, packet, source["session_id"], kind)

    # Import the algorithm only after the CLI's pre-import operation gate and
    # the raw/sealed-input checks above have succeeded.
    from chaoyang.pipeline.exact78_fresh_robot_stage_v2 import run_robot_stage_ffmpeg44

    result = run_robot_stage_ffmpeg44(
        hawor_npz=npz, frame_provenance=provenance, input_audit=audit,
        raw_video=raw_video, source_kind=kind, session_id=source["session_id"],
        output_root=output_root)
    return result


def main() -> int:
    authorize_exact78_operation(OPERATION, sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--sealed-hawor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_candidate(session=args.session, sealed_hawor_root=args.sealed_hawor_root,
                           output_root=args.output_root)
    from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref

    print(json.dumps({"status": result["status"],
                      "result": file_ref(args.output_root / "RESULT.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
