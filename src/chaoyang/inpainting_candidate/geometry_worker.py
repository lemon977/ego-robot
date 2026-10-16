"""Registered CPU Motion/full-link Render candidate; no models, Clean or adoption.

The existing root executor owns the publisher and deadline. This worker seals
stage receipts only, preserving every native frame and unverified provenance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

NATIVE_GEOMETRY_SCOPE = "INPAINTING_NATIVE_MOTION_RENDER_CPU"
NATIVE_GEOMETRY_RESULT_SCHEMA = "inpainting-native-geometry-batch-result-v1"


def _direct_outputs(receipt, signature, receipt_path):
    """Verify all nested bytes, then select only this consumer's own ABI."""
    from .contracts import verify_sealed_stage
    directory = Path(receipt_path).resolve(strict=True).parent
    refs = verify_sealed_stage(receipt, signature, output_root=directory)
    direct = [r for r in refs if Path(r["path"]).parent == directory]
    names = {Path(r["path"]).name: r for r in direct}
    if len(names) != len(direct):
        raise ValueError("NATIVE_GEOMETRY_DIRECT_OUTPUT_COLLISION")
    return names


def _read_verified(reference):
    from .contracts import verify_ref
    verify_ref(reference)
    return json.loads(Path(reference["path"]).read_text(encoding="utf-8-sig"))


def verify_native_source(row):
    """Source identity/map/raw bindings are evidence; qualification stays false."""
    from .contracts import verify_ref
    reference = row["source_receipt"]
    source = _read_verified(reference)
    refs = _direct_outputs(source, row["source_signature"], reference["path"])
    report = _read_verified(refs["SOURCE_REPORT.json"])
    mapping = _read_verified(refs["FRAME_MAP.json"])
    count = row["native_frames"]
    if (source.get("stage") != "Source" or type(count) is not int or count < 1
            or source.get("native_frames") != count
            or source.get("session_id") != row["session_id"]
            or report.get("session_id") != row["session_id"]
            or report.get("native_sidecar_frames") != count
            or not isinstance(mapping, list) or len(mapping) != count
            or row.get("frame_map_ref") != refs["FRAME_MAP.json"]
            or report.get("actual_inputs") != row.get("raw_inputs")
            or not isinstance(row.get("raw_inputs"), dict) or not row["raw_inputs"]):
        raise ValueError("NATIVE_GEOMETRY_SOURCE_IDENTITY_OR_BINDING_DRIFT")
    declared = report.get("frame_mapping")
    if not isinstance(declared, list) or len(declared) != count:
        raise ValueError("NATIVE_GEOMETRY_SOURCE_NATIVE_MAP_INCOMPLETE")
    qpc = []
    for index, frame in enumerate(mapping):
        if (not isinstance(frame, dict)
                or any(type(frame.get(k)) is not int or frame[k] != index for k in
                       ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
                or type(frame.get("qpc_declared_ns")) is not int
                or frame["qpc_declared_ns"] != declared[index].get("qpc_declared_ns")
                or frame.get("physical_eye") != report.get("camera", {}).get("physical_eye")
                or frame.get("sbs_index") != report.get("camera", {}).get("sbs_index")):
            raise ValueError("NATIVE_GEOMETRY_SOURCE_FRAME_MAP_DRIFT")
        qpc.append(frame["qpc_declared_ns"])
    if any(b <= a for a, b in zip(qpc, qpc[1:])):
        raise ValueError("NATIVE_GEOMETRY_QPC_DUPLICATE_OR_BACKWARD")
    for raw in row["raw_inputs"].values():
        if raw not in source.get("actual_inputs", []):
            raise ValueError("NATIVE_GEOMETRY_RAW_REF_NOT_SOURCE_BOUND")
        verify_ref(raw)
    for raw in source.get("actual_inputs", []):
        verify_ref(raw)
    qpc_ref = row["raw_inputs"].get("video_qpc")
    if qpc_ref is None:
        raise ValueError("NATIVE_GEOMETRY_RAW_QPC_REQUIRED")
    actual_qpc = [json.loads(line) for line in Path(qpc_ref["path"]).read_text().splitlines() if line.strip()]
    if (len(actual_qpc) != count or any(type(t) is not int for t in actual_qpc)
            or actual_qpc != qpc):
        raise ValueError("NATIVE_GEOMETRY_RAW_QPC_SOURCE_MAP_DRIFT")
    # The source bridge needs the full decoded map, not a parser-generated ID.
    report = dict(report, frame_mapping=mapping)
    return source, report, mapping


def verify_native_receipt(reference, signature, row, mapping, *, native_only):
    """Verify the existing MotionSource role; do not rewrite its sealed receipt."""
    import numpy as np
    receipt = _read_verified(reference)
    refs = _direct_outputs(receipt, signature, reference["path"])
    count = row["native_frames"]
    if (receipt.get("stage") != "MotionSource" or receipt.get("native_frames") != count
            or receipt.get("session_id") != row["session_id"]
            or receipt.get("kinematic_actions_generated") is not False
            or receipt.get("model_calls") != 0 or receipt.get("retarget_calls") != 0
            or receipt.get("control_ground_truth") is not False or receipt.get("training_eligible") is not False):
        raise ValueError("NATIVE_GEOMETRY_NATIVE_RECEIPT_OR_QUALIFICATION_DRIFT")
    states = _read_verified(refs["FRAME_STATUS.json"])
    report = _read_verified(refs["MOTION_SOURCE_REPORT.json"])
    if (not isinstance(states, list) or len(states) != count
            or report.get("session_id") != row["session_id"] or report.get("native_frames") != count
            or report.get("actual_source_inputs") != row["raw_inputs"]
            or any(pin not in receipt.get("actual_inputs", []) for pin in row["raw_inputs"].values())
            or report.get("model_calls") != 0 or report.get("retarget_calls") != 0):
        raise ValueError("NATIVE_GEOMETRY_NATIVE_CURRENT_RAW_OR_STATUS_DRIFT")
    shapes = {"native__head": (count, 4, 4), "native__controller": (count, 2, 4, 4),
              "native__wrist": (count, 2, 4, 4), "native__hands": (2, count, 21, 3),
              "native__head_valid": (count,), "native__controller_valid": (count, 2),
              "native__manus_valid": (count, 2), "native__hand_observed": (2, count),
              "native__hand_display_valid": (2, count), "native__hand_source": (2, count),
              "qpc_ns": (count,)}
    with np.load(refs["NATIVE_MOTION.npz"]["path"], allow_pickle=False) as archive:
        if native_only and {"q_arm", "q_hand", "K", "c2w", "c2w_factory"} & set(archive.files):
            raise ValueError("NATIVE_GEOMETRY_NATIVE_ONLY_ACTION_OR_CAMERA_FICTION")
        if any(key not in archive or archive[key].shape != shape or archive[key].dtype.hasobject
               for key, shape in shapes.items()):
            raise ValueError("NATIVE_GEOMETRY_NATIVE_ARRAY_SHAPE_DRIFT")
        qpc = archive["qpc_ns"]
        hand_source = archive["native__hand_source"]
        if (qpc.dtype.kind not in "iu" or not np.array_equal(qpc, [r["qpc_declared_ns"] for r in mapping])):
            raise ValueError("NATIVE_GEOMETRY_NATIVE_QPC_DRIFT")
        for index, state in enumerate(states):
            if (type(state.get("native_frame_index")) is not int or state["native_frame_index"] != index
                    or state.get("qpc_declared_ns") != mapping[index]["qpc_declared_ns"]
                    or state.get("timebase_status") != mapping[index].get("timebase_status")
                    or state.get("camera_head_valid") is not bool(archive["native__head_valid"][index])
                    or state.get("training_eligible") is not False
                    or state.get("projection_qualified") is not False
                    or any(state.get("hands", {}).get(name, {}).get("motion_source") != str(hand_source[side,index])
                           for side,name in enumerate(("left", "right")))):
                raise ValueError("NATIVE_GEOMETRY_NATIVE_FRAME_STATUS_DRIFT")
            for side, name in enumerate(("left", "right")):
                hand = state.get("hands", {}).get(name, {})
                validity = {"wrist_position_valid": bool(archive["native__controller_valid"][index, side]),
                    "wrist_orientation_valid": bool(archive["native__controller_valid"][index, side]),
                    "local_handshape_valid": bool(archive["native__manus_valid"][index, side]),
                    "raw_composed_valid": bool(archive["native__hand_observed"][side, index]),
                    "display_valid": bool(archive["native__hand_display_valid"][side, index]),
                    "control_ground_truth": False}
                if any(hand.get(key) is not value for key, value in validity.items()):
                    raise ValueError("NATIVE_GEOMETRY_NATIVE_VALIDITY_ARRAY_STATUS_DRIFT")
    for pin in receipt.get("actual_inputs", []):
        from .contracts import verify_ref
        verify_ref(pin)
    return receipt, refs, report


def execute_native_geometry(packet, *, root, repo_root, authority,
                            native_builder=None, motion_builder=None, render_builder=None):
    """Dispatch an independently signed variable-native CPU contract only."""
    from chaoyang.governance.common import now_iso
    from .contracts import canonical_sha, file_ref
    stages = packet["stages"]
    if stages not in (["Native"], ["Motion", "Render"]):
        raise ValueError("NATIVE_GEOMETRY_STAGE_SET_UNSUPPORTED")
    if stages == ["Native"]:
        if native_builder is None:
            from .motion_source import prepare_motion_source
            native_builder = prepare_motion_source
    else:
        if motion_builder is None:
            from .motion import build_motion_stage
            motion_builder = build_motion_stage
        if render_builder is None:
            from .render import build_render_stage
            render_builder = build_render_stage
    rows = []
    for row in packet["sessions"]:
        authority(full=True)
        source, source_report, mapping = verify_native_source(row)
        session = root / "sessions" / row["session_key"]
        session.mkdir(parents=True, exist_ok=False)
        result = {"session_key": row["session_key"], "session_id": row["session_id"],
            "native_frames": row["native_frames"], "source": row["source_receipt"],
            "frame_map_ref": row["frame_map_ref"], "timebase_status": source["timebase_status"],
            "mask": "NOT_CONSUMED", "clean": "NOT_CONSUMED", "composite": "NOT_EXECUTED",
            "visual_quality": "NOT_REVIEWED", "human_review": "PENDING", "geometry": "UNKNOWN",
            "control_ground_truth": False, "training_eligible": False,
            "render_calls": 0, "zbuffer_passes": 0}
        if stages == ["Native"]:
            bridge = native_builder(source_report, session / "Native",
                component_signature=row["native_signature"], v1_manifest_ref=None)
            if bridge.get("raw_session") is not None:
                raise ValueError("NATIVE_GEOMETRY_NATIVE_ONLY_RAWSESSION_FORBIDDEN")
            pin = file_ref(session / "Native/RECEIPT.json")
            _, _, native_report = verify_native_receipt(pin, row["native_signature"], row, mapping, native_only=True)
            result.update(execution="COMPLETE_NATIVE_ONLY", native_receipt=pin,
                motion_receipt=None, render_receipt=None, render_status="NOT_REQUESTED",
                robot_consumer_status=native_report["robot_consumer_status"])
        else:
            motion_dir = session / "Motion"
            motion = motion_builder(source_receipt=row["source_receipt"],
                source_signature=row["source_signature"], output_dir=motion_dir,
                component_signature=row["motion_signature"], raw_inputs=row["raw_inputs"],
                camera_contract=row["camera_contract"], repo_root=repo_root)
            motion_ref = file_ref(motion_dir / "RECEIPT.json")
            _direct_outputs(motion, row["motion_signature"], motion_ref["path"])
            native_ref = file_ref(motion_dir / "Native/RECEIPT.json")
            native_signature = canonical_sha({"stage": "MotionSource", "Motion": row["motion_signature"], "Source": row["source_signature"]})
            verify_native_receipt(native_ref, native_signature, row, mapping, native_only=False)
            result.update(native_receipt=native_ref, motion_receipt=motion_ref)
            if motion.get("kinematic_actions_generated") is False:
                failure = motion.get("failure_code")
                if not isinstance(failure, str) or not failure:
                    raise ValueError("NATIVE_GEOMETRY_BLOCKED_MOTION_REASON_REQUIRED")
                diagnostic = {"schema": "inpainting-native-render-local-blocked-v1", "status": "BLOCKED",
                    "failure_code": failure, "source_receipt": row["source_receipt"], "motion_receipt": motion_ref,
                    "native_frames": row["native_frames"], "render_calls": 0, "zbuffer_passes": 0,
                    "control_ground_truth": False, "training_eligible": False}
                diagnostic_ref = _write_once(session / "RENDER_BLOCKED.json", diagnostic)
                result.update(execution="COMPLETE_NATIVE_LOCAL_RENDER_BLOCKED", render_status="BLOCKED",
                    render_receipt=None, render_diagnostic=diagnostic_ref, failure_code=failure)
            elif motion.get("kinematic_actions_generated") is True:
                authority(full=True)
                render_dir = session / "Render"
                render = render_builder(source_receipt=row["source_receipt"],
                    source_signature=row["source_signature"], motion_receipt=motion_ref,
                    motion_signature=row["motion_signature"], output_dir=render_dir,
                    component_signature=row["render_signature"], repo_root=repo_root,
                    authorization=authority, asset_closure=packet["inputs"]["asset_closure"])
                render_ref = file_ref(render_dir / "RECEIPT.json")
                _direct_outputs(render, row["render_signature"], render_ref["path"])
                if (render.get("render_calls") != 2 * row["native_frames"]
                        or render.get("zbuffer_passes") != 4 * row["native_frames"]
                        or render.get("render_views") != 2 or render.get("complete_robot") is not True):
                    raise ValueError("NATIVE_GEOMETRY_RENDER_NATIVE_CALL_ACCOUNTING_DRIFT")
                result.update(execution="COMPLETE_MOTION_RENDER", render_status="COMPLETE", render_receipt=render_ref,
                    geometry="VISUAL_ROBOT_ONLY_SCENE_UNKNOWN", render_calls=render["render_calls"],
                    zbuffer_passes=render["zbuffer_passes"])
            else:
                raise ValueError("NATIVE_GEOMETRY_MOTION_ACTION_STATUS_REQUIRED")
        authority(full=True)
        result["session_result"] = _write_once(session / "SESSION_RESULT.json", result)
        rows.append(result)
    result = {"schema": NATIVE_GEOMETRY_RESULT_SCHEMA, "status": "PASSED", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "planned_sessions": len(packet["sessions"]),
        "completed_sessions": len(rows), "completed_native_sessions": len(rows),
        "completed_geometry_sessions": sum(r["render_status"] == "COMPLETE" for r in rows),
        "blocked_render_sessions": sum(r["render_status"] == "BLOCKED" for r in rows),
        "native_frames": sum(r["native_frames"] for r in rows), "sessions": rows,
        "render_calls": sum(r["render_calls"] for r in rows), "zbuffer_passes": sum(r["zbuffer_passes"] for r in rows),
        "model_calls": 0, "gpu_count": 0, "training_steps": 0, "files_deleted": 0,
        "candidate_adopted": False, "control_ground_truth": False, "training_eligible": False,
        "visual_quality": "NOT_REVIEWED", "completed_at": now_iso(),
        "claim_limit": "Native execution and local Robot diagnostics only; sealed local blockers and numeric FAIL remain, not visual/product/training qualification."}
    reference = _write_once(root / "GEOMETRY_RESULT.json", result)
    print(json.dumps(reference, ensure_ascii=False), flush=True)
    return 0


def _write_once(path, value):
    from .contracts import file_ref
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT, load_json, now_iso
    from chaoyang.governance.inpainting_candidate_registration import (
        GEOMETRY_OPERATION, authorize_geometry_operation)
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parsed = parser.parse_args(args)
    # No implementation/asset imports until the real operation gate succeeds.
    packet = authorize_geometry_operation(GEOMETRY_OPERATION, args, repo_root=REPO_ROOT)
    if load_json(parsed.packet) != packet:
        raise ValueError("GEOMETRY_IN_MEMORY_PACKET_DRIFT")
    root = Path(packet["output_dir"])
    if root.resolve(strict=True) != root or not root.is_relative_to(REPO_ROOT) or root == REPO_ROOT:
        raise ValueError("GEOMETRY_PROJECT_OUTPUT_REQUIRED")
    cache = root / "cpu_cache"
    cache.mkdir(exist_ok=False)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"))
    tempfile.tempdir = str(cache)
    sys.dont_write_bytecode = True
    from .contracts import file_ref, verify_ref, verify_sealed_stage

    def authority(*, full=False):
        current = authorize_geometry_operation(GEOMETRY_OPERATION, args, repo_root=REPO_ROOT, full=full)
        if current != packet:
            raise ValueError("GEOMETRY_TASK_OR_WRITER_IDENTITY_DRIFT")
        return {"execution_allowed": True, "algorithm_execution_allowed": True,
            "task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "fencing_token": packet["writer"]["fencing_token"], "gpu_count": 0,
            "model_calls_max": 0, "cpu_threads_per_worker": 2,
            "deadline_at": packet["deadline_at"]}

    if packet.get("experiment_scope") == NATIVE_GEOMETRY_SCOPE:
        os.environ["PYTHONPYCACHEPREFIX"] = str(cache / "bytecode")
        return execute_native_geometry(packet, root=root, repo_root=REPO_ROOT, authority=authority)

    from .motion import build_motion_stage
    from .render import build_render_stage

    rows = []
    for row in packet["sessions"]:
        authority(full=True)
        verify_ref(row["source_receipt"])
        source_path = Path(row["source_receipt"]["path"])
        source = load_json(source_path)
        verify_sealed_stage(source, row["source_signature"], output_root=source_path.parent)
        session_root = root / "sessions" / row["session_key"]
        session_root.mkdir(parents=True, exist_ok=False)
        motion_dir = session_root / "Motion"
        motion = build_motion_stage(source_receipt=row["source_receipt"],
            source_signature=row["source_signature"], output_dir=motion_dir,
            component_signature=row["motion_signature"], raw_inputs=row["raw_inputs"],
            camera_contract=row["camera_contract"], repo_root=REPO_ROOT)
        verify_sealed_stage(motion, row["motion_signature"], output_root=motion_dir)
        authority(full=True)
        render_dir = session_root / "Render"
        render = build_render_stage(source_receipt=row["source_receipt"],
            source_signature=row["source_signature"], motion_receipt=file_ref(motion_dir / "RECEIPT.json"),
            motion_signature=row["motion_signature"], output_dir=render_dir,
            component_signature=row["render_signature"], repo_root=REPO_ROOT,
            authorization=authority, asset_closure=packet["inputs"]["asset_closure"])
        verify_sealed_stage(render, row["render_signature"], output_root=render_dir)
        authority(full=True)
        rows.append({"session_key": row["session_key"], "native_frames": row["native_frames"],
            "source": row["source_receipt"], "motion": file_ref(motion_dir / "RECEIPT.json"),
            "render": file_ref(render_dir / "RECEIPT.json"),
            "motion_receipt": file_ref(motion_dir / "RECEIPT.json"),
            "render_receipt": file_ref(render_dir / "RECEIPT.json"),
            "execution": "COMPLETE_MOTION_RENDER",
            "timebase_status": source["timebase_status"], "geometry": "VISUAL_ROBOT_ONLY_SCENE_UNKNOWN",
            "mask": "NOT_CONSUMED", "clean": "NOT_CONSUMED", "composite": "NOT_EXECUTED",
            "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "control_ground_truth": False, "training_eligible": False})
    result = {"schema": "inpainting-geometry-batch-result-v1",
        "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "status": "PASSED", "completed_sessions": len(rows), "sessions": rows,
        "planned_sessions": len(packet["sessions"]), "completed_geometry_sessions": len(rows),
        "native_frames": sum(row["native_frames"] for row in rows),
        "model_calls": 0, "gpu_count": 0, "training_steps": 0, "files_deleted": 0,
        "candidate_adopted": False, "control_ground_truth": False,
        "visual_quality": "NOT_REVIEWED", "training_eligible": False,
        "completed_at": now_iso(),
        "claim_limit": "Current raw Controller/Head + MANUS to visual Robot and internal depth, not scene occlusion correctness or a complete Raw-to-Robot production qualification."}
    reference = _write_once(root / "GEOMETRY_RESULT.json", result)
    print(json.dumps(reference, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
