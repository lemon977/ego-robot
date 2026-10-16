"""Variable-native wrapper tests with synthetic producers; no FK/model/render."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from chaoyang.inpainting_candidate import geometry_worker as worker
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/maintenance/inpainting_candidate_geometry_task_v1.py"
loader = importlib.util.spec_from_file_location("_native_geometry_execution_tests", SCRIPT)
runner = importlib.util.module_from_spec(loader)
loader.loader.exec_module(runner)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return file_ref(path)


def load(pin):
    return json.loads(Path(pin["path"]).read_text())


def npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return file_ref(path)


def blob(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Synthetic bytes only, not actual video")
    return file_ref(path)


def seal(folder, stage, signature, outputs, count, identity, **extra):
    return put(folder / "RECEIPT.json", {"schema": "inpainting-candidate-component-v1", "stage": stage,
        "status": "SEALED_COMPLETE", "component_signature": signature, "outputs": outputs,
        "native_frames": count, "session_id": identity, "control_ground_truth": False,
        "training_eligible": False, "model_calls": 0, "actual_inputs": [], **extra})


def source(root, key, count):
    identity = "capture_" + key
    rawdir = root / "raw_fixture" / key
    rawdir.mkdir(parents=True)
    qpc = [1000000000 + i * 33333333 for i in range(count)]
    qpath = rawdir / "qpc.jsonl"
    qpath.write_text("".join(json.dumps(t) + "\n" for t in qpc))
    raw = {"video_qpc": file_ref(qpath), "controller_head": blob(rawdir / "pico.jsonl"),
           "manus": blob(rawdir / "manus.jsonl")}
    mapping = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
                "output_frame": i, "qpc_declared_ns": qpc[i], "acquisition_frame_id": "UNKNOWN",
                "physical_eye": "right", "sbs_index": 0, "timebase_status": "TIMEBASE_UNVERIFIED"}
               for i in range(count)]
    folder = root / "source_fixture" / key
    map_ref = put(folder / "FRAME_MAP.json", mapping)
    report = {"status": "SOURCE_PREFLIGHT_CANDIDATE", "session_id": identity,
              "session_dir": str(rawdir), "native_sidecar_frames": count, "actual_inputs": raw,
              "frame_mapping": mapping, "camera": {"physical_eye": "right", "sbs_index": 0},
              "timebase_status": "TIMEBASE_UNVERIFIED"}
    outputs = [map_ref, put(folder / "SOURCE_REPORT.json", report),
               npz(folder / "RAW_RGB.npz", rgb=np.zeros((count, 3, 4, 3), np.uint8))]
    pin = seal(folder, "Source", "a" * 64, outputs, count, identity,
               actual_inputs=list(raw.values()), timebase_status="TIMEBASE_UNVERIFIED")
    return {"session_key": key, "session_id": identity, "native_frames": count,
            "source_receipt": pin, "source_signature": "a" * 64, "frame_map_ref": map_ref,
            "raw_inputs": raw, "native_signature": "d" * 64}, mapping


def make_native(row, folder, signature):
    count = row["native_frames"]
    mapping = load(row["frame_map_ref"])
    arrays = {"qpc_ns": np.array([r["qpc_declared_ns"] for r in mapping], np.int64),
              "native__head": np.zeros((count, 4, 4)), "native__controller": np.zeros((count, 2, 4, 4)),
              "native__wrist": np.zeros((count, 2, 4, 4)), "native__hands": np.zeros((2, count, 21, 3)),
              "native__head_valid": np.ones(count, bool), "native__controller_valid": np.ones((count, 2), bool),
              "native__manus_valid": np.ones((count, 2), bool), "native__hand_observed": np.ones((2, count), bool),
              "native__hand_display_valid": np.ones((2, count), bool),
              "native__hand_source": np.full((2, count), "RAW_CONTROLLER_MANUS", dtype="U32")}
    states = [{"native_frame_index": i, "qpc_declared_ns": m["qpc_declared_ns"],
               "training_eligible": False, "projection_qualified": False, "timebase_status": "TIMEBASE_UNVERIFIED",
               "camera_head_valid": True,
               "hands": {s: {"motion_source": "RAW_CONTROLLER_MANUS", "control_ground_truth": False,
                   "wrist_position_valid": True, "wrist_orientation_valid": True,
                   "local_handshape_valid": True, "raw_composed_valid": True, "display_valid": True}
                   for s in ("left", "right")}}
              for i, m in enumerate(mapping)]
    report = {"session_id": row["session_id"], "native_frames": count, "actual_source_inputs": row["raw_inputs"],
              "robot_consumer_status": "BLOCKED_CAMERA_AXES", "model_calls": 0, "retarget_calls": 0,
              "camera_binding": {"status": "BLOCKED_CAMERA_AXES", "failure_code": "DEVICE_NOT_IN_ADOPTED_V1_DOMAIN"}}
    outputs = [npz(folder / "NATIVE_MOTION.npz", **arrays), put(folder / "FRAME_STATUS.json", states),
               put(folder / "MOTION_SOURCE_REPORT.json", report)]
    pin = seal(folder, "MotionSource", signature, outputs, count, row["session_id"],
               actual_inputs=list(row["raw_inputs"].values()), kinematic_actions_generated=False, retarget_calls=0)
    return {"receipt": load(pin), "raw_session": None, "report": report}


@pytest.fixture
def finite(tmp_path):
    row, mapping = source(tmp_path, "first", 3)
    packet = {"experiment_scope": worker.NATIVE_GEOMETRY_SCOPE, "stages": ["Native"],
              "task_id": "inpainting_native_motion_render_fixture", "run_signature": "f" * 64,
              "output_dir": str(tmp_path / "output"), "sessions": [row], "inputs": {},
              "config": {"image_size": [4, 3]}, "budgets": {"render_calls_max": 0, "zbuffer_passes_max": 0}}
    Path(packet["output_dir"]).mkdir()
    return tmp_path, packet, mapping


def execute(packet, native_builder=None, motion_builder=None, render_builder=None):
    root = Path(packet["output_dir"])
    worker.execute_native_geometry(packet, root=root, repo_root=root.parent,
        authority=lambda **kw: {"execution_allowed": True}, native_builder=native_builder,
        motion_builder=motion_builder, render_builder=render_builder)
    return load(file_ref(root / "GEOMETRY_RESULT.json"))


def native_builder(packet):
    def build(report, folder, *, component_signature, v1_manifest_ref):
        assert v1_manifest_ref is None
        row = next(row for row in packet["sessions"] if row["session_id"] == report["session_id"])
        assert report["frame_mapping"] == load(row["frame_map_ref"])
        return make_native(row, folder, component_signature)
    return build


def geometry_builders(packet, blocked_keys=()):
    def motion(**kw):
        row = next(r for r in packet["sessions"] if r["source_receipt"] == kw["source_receipt"])
        count = row["native_frames"]
        folder = kw["output_dir"]
        mapping = load(row["frame_map_ref"])
        signature = canonical_sha({"stage": "MotionSource", "Motion": row["motion_signature"], "Source": row["source_signature"]})
        native = make_native(row, folder / "Native", signature)
        blocked = row["session_key"] in blocked_keys
        states = [{"native_frame_index": i, "source_frame_map": m,
                   "left": {"numerical_quality": "FAIL"}, "right": {"numerical_quality": "UNKNOWN"}}
                  for i, m in enumerate(mapping)]
        arrays = {"qpc_ns": np.array([m["qpc_declared_ns"] for m in mapping], np.int64)}
        if not blocked:
            arrays.update(q_arm=np.zeros((count, 2, 7)), q_hand=np.zeros((count, 2, 22)))
        outputs = [put(folder / "FRAME_MAP.json", mapping), put(folder / "FRAME_STATUS.json", states),
                   npz(folder / "MOTION.npz", **arrays), put(folder / "MOTION_REPORT.json", {"status": "FAKE_NUMERICAL_FAIL"}),
                   file_ref(folder / "Native/RECEIPT.json"), *native["receipt"]["outputs"]]
        pin = seal(folder, "Motion", kw["component_signature"], outputs, count, row["session_id"],
                   source_receipt=row["source_receipt"], source_signature=row["source_signature"],
                   kinematic_actions_generated=not blocked,
                   failure_code="DEVICE_NOT_IN_ADOPTED_V1_DOMAIN" if blocked else None)
        return load(pin)

    def render(**kw):
        row = next(r for r in packet["sessions"] if r["source_receipt"] == kw["source_receipt"])
        assert row["session_key"] not in blocked_keys
        folder, count = kw["output_dir"], row["native_frames"]
        mapping = load(row["frame_map_ref"])
        qpc = np.array([m["qpc_declared_ns"] for m in mapping], np.int64)
        buffers = {"rgba": np.zeros((count, 3, 4, 4), np.uint8), "depth_m": np.zeros((count, 3, 4), np.float32),
                   "link_id": np.zeros((count, 3, 4), np.int32), "triangle_id": np.zeros((count, 3, 4), np.int32),
                   "qpc_ns": qpc, "K": np.eye(3), "c2w": np.zeros((count, 4, 4)), "T_world_base": np.eye(4)}
        outputs = [put(folder / "FRAME_MAP.json", mapping),
                   put(folder / "FRAME_STATUS.json", [{"native_frame_index": i, "source_frame_map": m,
                       "numerical_quality": "FAIL"} for i, m in enumerate(mapping)]),
                   put(folder / "RENDER_REPORT.json", {"scene_geometry": "UNKNOWN_NOT_CONSUMED", "complete_robot": True,
                       "render_calls": 2 * count, "zbuffer_passes": 4 * count, "render_views": 2, "model_calls": 0}),
                   npz(folder / "ROBOT_BUFFERS.npz", **buffers), blob(folder / "RAW_ROBOT_FPV_REVIEW.mp4"),
                   blob(folder / "ROBOT_FULL_ARM_THIRD_PERSON_REVIEW.mp4")]
        pin = seal(folder, "Render", kw["component_signature"], outputs, count, row["session_id"],
                   source_receipt=row["source_receipt"], source_signature=row["source_signature"],
                   motion_receipt=kw["motion_receipt"], motion_signature=kw["motion_signature"],
                   complete_robot=True, render_views=2, render_calls=2 * count, zbuffer_passes=4 * count)
        return load(pin)
    return motion, render


def to_geometry(packet):
    packet["stages"] = ["Motion", "Render"]
    packet["inputs"] = {"asset_closure": {"unused_fake": True}}
    for row in packet["sessions"]:
        row.update(motion_signature="b" * 64, render_signature="c" * 64,
                   camera_contract={"unused_fake": True})
    packet["budgets"] = {"render_calls_max": 2 * sum(r["native_frames"] for r in packet["sessions"]),
                         "zbuffer_passes_max": 4 * sum(r["native_frames"] for r in packet["sessions"])}


def decode(packet):
    counts = {r["session_key"]: r["native_frames"] for r in packet["sessions"]}
    return lambda path: counts[path.parents[1].name]


def reseal_output(pin, basename, replacement):
    receipt = load(pin)
    old = next(ref for ref in receipt["outputs"] if Path(ref["path"]).name == basename)
    receipt["outputs"] = [replacement if ref == old else ref for ref in receipt["outputs"]]
    return put(Path(pin["path"]), receipt)


def test_native_only_variable_n_no_q_camera_robot_or_renderer(finite):
    root, packet, _ = finite
    second, _ = source(root, "second", 5)
    packet["sessions"].append(second)
    def forbidden(**kw):
        pytest.fail("Native-only reached Motion or Render")
    result = execute(packet, native_builder=native_builder(packet), motion_builder=forbidden, render_builder=forbidden)
    assert result["completed_native_sessions"] == 2
    assert result["native_frames"] == 8 and result["completed_geometry_sessions"] == 0
    assert result["blocked_render_sessions"] == result["render_calls"] == result["zbuffer_passes"] == 0
    assert runner.validate_geometry_result(packet, decoder=lambda p: pytest.fail("Native-only decoded Robot video"))


def test_local_camera_blocker_continues_an_independent_geometry_row(finite):
    root, packet, _ = finite
    second, _ = source(root, "second", 5)
    packet["sessions"].append(second)
    to_geometry(packet)
    m,r = geometry_builders(packet, blocked_keys={"first"})
    result = execute(packet, motion_builder=m, render_builder=r)
    assert result["completed_native_sessions"] == 2 and result["completed_geometry_sessions"] == 1
    assert result["blocked_render_sessions"] == 1
    assert result["render_calls"] == 10 and result["zbuffer_passes"] == 20
    assert result["sessions"][0]["failure_code"] == "DEVICE_NOT_IN_ADOPTED_V1_DOMAIN"
    assert not (Path(packet["output_dir"]) / "sessions/first/Render").exists()
    assert runner.validate_geometry_result(packet, decoder=decode(packet))


def test_numeric_fail_does_not_cancel_legitimate_render_and_nested_status_is_distinct(finite):
    _, packet, _ = finite
    to_geometry(packet)
    m,r = geometry_builders(packet)
    result = execute(packet, motion_builder=m, render_builder=r)
    assert result["completed_geometry_sessions"] == 1 and result["render_calls"] == 6
    motion, names = runner._native_outputs(result["sessions"][0]["motion_receipt"], "Motion", "b" * 64, 3)
    assert len(motion["outputs"]) == 8
    assert Path(names["FRAME_STATUS.json"]["path"]).parent.name == "Motion"
    assert load(names["FRAME_STATUS.json"])[0]["left"]["numerical_quality"] == "FAIL"
    assert runner.validate_geometry_result(packet, decoder=decode(packet))


@pytest.mark.parametrize("kind", ["count", "identity", "frame_ref", "raw_bind", "raw_sha", "source_partial", "map_index", "qpc_sidecar"])
def test_source_negative_bindings_reject_before_native(finite, kind):
    _, packet, _ = finite
    row = packet["sessions"][0]
    if kind == "count": row["native_frames"] += 1
    elif kind == "identity": row["session_id"] = "another"
    elif kind == "frame_ref": row["frame_map_ref"] = row["source_receipt"]
    elif kind == "raw_bind": row["raw_inputs"] = {}
    elif kind == "raw_sha": row["raw_inputs"]["manus"]["sha256"] = "e" * 64
    elif kind == "source_partial":
        receipt = load(row["source_receipt"]); receipt["status"] = "RUNNING"
        row["source_receipt"] = put(Path(row["source_receipt"]["path"]), receipt)
    elif kind == "map_index":
        mapping = load(row["frame_map_ref"]); mapping[0]["output_frame"] = 1
        row["frame_map_ref"] = put(Path(row["frame_map_ref"]["path"]), mapping)
        row["source_receipt"] = reseal_output(row["source_receipt"], "FRAME_MAP.json", row["frame_map_ref"])
    elif kind == "qpc_sidecar":
        Path(row["raw_inputs"]["video_qpc"]["path"]).write_text("1\n2\n3\n")
    with pytest.raises((ValueError, KeyError)):
        execute(packet, native_builder=lambda *a, **kw: pytest.fail("source blocker reached native builder"))


@pytest.mark.parametrize("kind", ["count", "qpc", "K", "raw", "frame_status", "validity", "time_status", "model", "retarget", "partial", "training"])
def test_native_receipt_negative_contracts(finite, kind):
    _, packet, mapping = finite
    row = packet["sessions"][0]
    folder = Path(packet["output_dir"]) / "standaloneNative"
    make_native(row, folder, row["native_signature"])
    pin = file_ref(folder / "RECEIPT.json")
    if kind in {"count", "qpc", "K"}:
        p = folder / "NATIVE_MOTION.npz"
        with np.load(p) as z: arrays = {k:z[k] for k in z.files}
        if kind == "count": arrays["native__head"] = np.zeros((2, 4, 4))
        elif kind == "qpc": arrays["qpc_ns"][1] += 1
        else: arrays["K"] = np.eye(3)
        pin = reseal_output(pin, p.name, npz(p, **arrays))
    elif kind == "raw":
        p = folder / "MOTION_SOURCE_REPORT.json"; v = load(file_ref(p)); v["actual_source_inputs"] = {}
        pin = reseal_output(pin, p.name, put(p, v))
    elif kind in {"frame_status", "validity", "time_status"}:
        p = folder / "FRAME_STATUS.json"; v = load(file_ref(p))
        if kind == "frame_status": v[0]["hands"]["left"]["motion_source"] = "UNDECLARED"
        elif kind == "validity": v[0]["hands"]["left"]["local_handshape_valid"] = False
        else: v[0]["timebase_status"] = "VERIFIED"
        pin = reseal_output(pin, p.name, put(p, v))
    else:
        v = load(pin)
        if kind == "model": v["model_calls"] = 1
        elif kind == "retarget": v["retarget_calls"] = 1
        elif kind == "partial": v["status"] = "RUNNING"
        elif kind == "training": v["training_eligible"] = True
        pin = put(Path(pin["path"]), v)
    with pytest.raises(ValueError):
        worker.verify_native_receipt(pin, row["native_signature"], row, mapping, native_only=True)


@pytest.mark.parametrize("kind", ["counts", "status", "source", "native_n", "quality", "decode", "zbuffer", "session_ref", "nested_drift", "buffer_qpc"])
def test_geometry_result_negative_integrity(finite, kind):
    _, packet, _ = finite
    to_geometry(packet)
    m,r = geometry_builders(packet)
    result = execute(packet, motion_builder=m, render_builder=r)
    row = result["sessions"][0]
    if kind == "counts": result["completed_native_sessions"] = 0
    elif kind == "status": row["render_status"] = "BLOCKED"
    elif kind == "source": row["source"] = row["native_receipt"]
    elif kind == "native_n": row["native_frames"] = 2
    elif kind == "quality": result["training_eligible"] = True
    elif kind == "zbuffer": row["zbuffer_passes"] = 1
    elif kind == "session_ref": row["session_result"] = row["motion_receipt"]
    elif kind == "nested_drift":
        p = Path(row["motion_receipt"]["path"]).parent / "Native/FRAME_STATUS.json"
        put(p, {"unsealed_changed_bytes": True})
    elif kind == "buffer_qpc":
        rp = row["render_receipt"]; receipt = load(rp)
        ref = next(p for p in receipt["outputs"] if Path(p["path"]).name == "ROBOT_BUFFERS.npz")
        with np.load(ref["path"]) as z: arrays = {k:z[k] for k in z.files}
        arrays["qpc_ns"][0] += 1
        row["render_receipt"] = reseal_output(rp, "ROBOT_BUFFERS.npz", npz(Path(ref["path"]), **arrays))
        session_copy = {k:v for k,v in row.items() if k != "session_result"}
        row["session_result"] = put(Path(row["session_result"]["path"]), session_copy)
    put(Path(packet["output_dir"]) / "GEOMETRY_RESULT.json", result)
    with pytest.raises((ValueError, KeyError)):
        runner.validate_geometry_result(packet, decoder=(lambda path: 2) if kind == "decode" else decode(packet))


def test_worker_refuses_native_only_returned_rawsession(finite):
    _, packet, _ = finite
    def wrong(*a, **kw):
        return {"raw_session": object()}
    with pytest.raises(ValueError, match="RAWSESSION_FORBIDDEN"):
        execute(packet, native_builder=wrong)


def test_illegal_stage_set_rejected_without_any_builder(finite):
    _, packet, _ = finite
    packet["stages"] = ["Native", "Motion", "Render"]
    with pytest.raises(ValueError, match="STAGE_SET_UNSUPPORTED"):
        execute(packet, native_builder=lambda *a, **kw: pytest.fail("illegal stages reached native"))
