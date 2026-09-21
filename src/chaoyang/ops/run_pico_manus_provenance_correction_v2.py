"""Append-only provenance downscope; never rerun or alter numerical motion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SESSIONS = dict(zip((f"play_cards_0916_{s}" for s in ("097", "098", "101")), (165, 179, 122)))
MAX_BYTES = 32 << 20


def ref(path):
    p = Path(path).resolve(strict=True)
    a = p.stat()
    if not p.is_file() or a.st_size > MAX_BYTES:
        raise ValueError("not a bounded regular input")
    raw = p.read_bytes()
    b = p.stat()
    if (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) != (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns):
        raise ValueError("unstable input")
    return {"path": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(binding, roots):
    if set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError("incomplete reference binding")
    p = Path(binding["path"]).resolve(strict=True)
    if not any(p.is_relative_to(r) for r in roots):
        raise ValueError("reference outside allowlist")
    if ref(p) != binding:
        raise ValueError("reference bytes/SHA mismatch")
    return p


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def numerical_digest(data, excluded):
    h = hashlib.sha256()
    for key in sorted(set(data) - set(excluded)):
        a = data[key]
        if a.dtype.hasobject:
            raise ValueError("object arrays prohibited")
        h.update(json.dumps([key, a.dtype.str, a.shape]).encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def correct_arrays(data, kind):
    """Only direct-observation provenance is changed; exact inference is unknown."""
    if kind == "motion":
        key, valid_key, tail = "joint_observed_local", "manus_local_joint_valid", (2, 21)
    elif kind == "robot_input":
        key, valid_key, tail = "source_observed_physical", "finger_valid", (2,)
    else:
        raise ValueError("unsupported correction kind")
    if key not in data or valid_key not in data or "timestamp_ns" not in data:
        raise ValueError("missing required provenance or time fields")
    observed, valid = data[key], data[valid_key]
    shape = (len(data["timestamp_ns"]),) + tail
    if observed.dtype != np.bool_ or valid.dtype != np.bool_ or observed.shape != shape or valid.shape != shape:
        raise ValueError("unexpected provenance mask shape/dtype")
    if any(k.startswith("provenance_v2_") for k in data):
        raise ValueError("already corrected input")
    out = {k: v.copy() for k, v in data.items()}
    out[key] = np.zeros_like(observed)
    changed_keys = [key]
    if kind == "motion":
        cv = data.get("controller_pose_observed")
        if cv is None or cv.dtype != np.bool_ or cv.shape != (len(data["timestamp_ns"]), 2):
            raise ValueError("controller source mask shape/dtype missing")
        out["controller_pose_observed"] = np.zeros_like(cv)
        out["provenance_v2_controller_pose_computable"] = cv.copy()
        out["provenance_v2_controller_direct_observation_unverified"] = cv.copy()
        changed_keys.append("controller_pose_observed")
    out["provenance_v2_upstream_resampled"] = valid.copy()
    out["provenance_v2_direct_observation_unverified"] = valid.copy()
    out["provenance_v2_exact_interpolation_support_known"] = np.zeros_like(valid)
    out["provenance_v2_source_clock"] = np.array("PICO_SENSOR_CLOCK_RESAMPLE_TIMELINE_NOT_RAW_ARRIVAL")
    out["provenance_v2_inferred_scope"] = np.array("ORIGINAL_INFERRED_MASK_IS_ADAPTER_ONLY_NOT_UPSTREAM_SAMPLING")
    out["provenance_v2_status"] = np.array("DIRECT_OBSERVATION_DOWNSCOPED_NO_NUMERICAL_CHANGE")
    before = numerical_digest(data, changed_keys)
    preserved = {k: out[k] for k in data}
    after = numerical_digest(preserved, changed_keys)
    if before != after:
        raise ValueError("numerical or non-target semantic array changed")
    return out, {"changed_original_fields": changed_keys, "preserved_array_count": len(data) - len(changed_keys),
                 "preserved_semantic_sha256": before, "bit_exact_preserved": True,
                 "all_remaining_original_fields_including_inferred_unchanged": True}


def correct_npz(binding, output, roots, kind, count):
    p = checked(binding, roots)
    with np.load(p, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if len(data["timestamp_ns"]) != count:
        raise ValueError("frame count drift")
    out, evidence = correct_arrays(data, kind)
    if ref(p) != binding:
        raise ValueError("input changed during array read")
    with output.open("xb") as f:
        np.savez_compressed(f, **out)
    with np.load(output, allow_pickle=False) as z:
        for k, a in out.items():
            b = z[k]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError("serialized corrected array differs")
    return {"predecessor": binding, "successor": ref(output), "regression": evidence}


def run(spec_path, output):
    import h5py
    spec_ref = ref(spec_path)
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("schema_version") != "PICO_MANUS_PROVENANCE_CORRECTION_SPEC_V2":
        raise ValueError("unsupported frozen binding schema")
    lane = Path(spec["lane_root"]).resolve(strict=True)
    canonical = Path(os.environ["CHAOYANG_REPO_ROOT"]).resolve(strict=True)
    if not lane.is_relative_to(canonical / "_run/current"):
        raise ValueError("lane outside project")
    output = Path(output).resolve()
    if output == lane or not output.is_relative_to(lane) or output.exists():
        raise ValueError("output must be a new own-lane directory")
    entries = spec.get("sessions", [])
    if len(entries) != 3 or set(e.get("session_id") for e in entries) != set(SESSIONS):
        raise ValueError("fixed three-session read-set required")
    roots = [lane]
    audit = spec["source_observation_audit"]
    checked(audit, roots)
    output.mkdir()
    dump(output / "INPUT_BINDINGS.json", spec)
    results = []
    for e in entries:
        sid = e["session_id"]
        motion_result_path = checked(e["motion_result"], roots)
        robot_result_path = checked(e["robot_result"], roots)
        motion = json.loads(motion_result_path.read_text())
        robot = json.loads(robot_result_path.read_text())
        if motion.get("session_id") != sid or robot.get("session_id") != sid:
            raise ValueError("session binding mismatch")
        if robot.get("schema_version") != "FULL_ROBOT_REVIEW_RESULT_V2":
            raise ValueError("unknown robot result schema")
        source_ref = motion["source_hdf5"]
        expected_source = Path("/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916/cards_130_0916") / sid[-3:] / "dataset.hdf5"
        source = checked(source_ref, [expected_source.parent.resolve(strict=True)])
        if source != expected_source.resolve(strict=True):
            raise ValueError("unexpected HDF5 source")
        with h5py.File(source, "r") as f:
            clock = str(f.attrs.get("timestamp_clock", ""))
            if "resample timeline" not in clock or str(f.attrs.get("alignment_clock")) != "host_qpc":
                raise ValueError("source resampling evidence absent")
            source_times = f["timestamp_ns"][:]
            timing = {k: f[k][:] for k in ("recv_qpc_ns", "video_source_recv_qpc_ns", "left_hand_source_recv_qpc_ns", "right_hand_source_recv_qpc_ns")}
        if ref(source) != source_ref:
            raise ValueError("HDF5 changed during read")
        folder = output / sid
        folder.mkdir()
        a = correct_npz(motion["motion_npz"], folder / "CONTROLLER_MANUS_MOTION_CORRECTED.npz", roots, "motion", SESSIONS[sid])
        b = correct_npz(robot["motion_input"], folder / "ROBOT_MOTION_INPUT_CORRECTED.npz", roots, "robot_input", SESSIONS[sid])
        with np.load(a["successor"]["path"], allow_pickle=False) as z:
            if z["timestamp_ns"].dtype != source_times.dtype or z["timestamp_ns"].tobytes() != source_times.tobytes():
                raise ValueError("source HDF5 timeline differs")
        timing_path = folder / "SOURCE_TIME_EVIDENCE.npz"
        with timing_path.open("xb") as f:
            np.savez_compressed(f, timestamp_ns=source_times, **timing)
        for r in [motion["review"]["video"], robot["review_video"], robot["motion_result"]]:
            checked(r, roots)
        for r in [e["motion_result"], e["robot_result"]]:
            checked(r, roots)
        result = {"schema_version": "PICO_MANUS_PROVENANCE_CORRECTION_RESULT_V2", "session_id": sid,
                  "status": "PROVENANCE_DOWNSCOPED_NUMERICAL_ARRAYS_UNCHANGED",
                  "predecessor_results": [e["motion_result"], e["robot_result"]],
                  "corrected_motion": a, "corrected_robot_input": b,
                  "effective_consumer_inputs": {"controller_manus_motion": a["successor"], "robot_motion_input": b["successor"], "robot_motion_result": robot["motion_result"]},
                  "reused_videos": [motion["review"]["video"], robot["review_video"]],
                  "video_semantics": "ORIGINAL_NUMERIC_RENDER_UNCHANGED_ONLY_PROVENANCE_CORRECTED",
                  "source_hdf5": source_ref, "timestamp_clock": clock, "source_time_evidence": ref(timing_path),
                  "source_observation_audit": audit,
                  "authority_unchanged": True, "evidence_downscoped": True,
                  "direct_observation_status": "UNVERIFIED_NOT_INFERRED_TRUE",
                  "training_eligible": False, "control_ground_truth": False, "numeric_quality_pass": False,
                  "visual_review_status": "NOT_USER_ACCEPTED", "numerical_optimizer_rerun": False}
        dump(folder / "RESULT.json", result)
        results.append(ref(folder / "RESULT.json"))
    if ref(spec_path) != spec_ref:
        raise ValueError("frozen bindings changed")
    result = {"schema_version": "PICO_MANUS_PROVENANCE_CORRECTION_V2", "status": "APPEND_ONLY_CORRECTION_COMPLETE",
              "input_bindings": spec_ref, "session_results": results,
              "authority_unchanged": True, "evidence_downscoped": True, "numerical_optimizer_rerun": False,
              "source_data_modified": False, "gpu_used": False, "frames": sum(SESSIONS.values())}
    dump(output / "RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU-only operation requires CUDA_VISIBLE_DEVICES=''")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    print(json.dumps(run(args.bindings, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
