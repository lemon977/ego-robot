"""Synthetic Render glue only: no real Robot, FK, raster or model invocation."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import render as r
from chaoyang.inpainting_candidate.contracts import SCHEMA, file_ref, verify_sealed_stage


def json_ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_ref(path)


def pure_v1_clipping():
    """Load only the frozen pure clipping functions, not asset/FK/JIT imports."""
    repo = Path(__file__).resolve().parents[2]
    namespace = {"np": np, "raster": SimpleNamespace(NEAR_M=.05)}
    for relative, name in (("src/chaoyang/pipeline/exact78_robot_raster_v1.py", "clip_triangle_near"),
                           ("src/chaoyang/exact78_baseline/render.py", "near_clip")):
        tree = ast.parse((repo / relative).read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), relative, "exec"), namespace)
        if name == "clip_triangle_near":
            namespace["raster"].clip_triangle_near = namespace[name]
    return namespace["near_clip"]


@pytest.fixture
def case(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(r, "PROJECT_ROOT", tmp_path)
    for path in r.CODE_CLOSURE:
        code = tmp_path / path
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("# synthetic dependency\n")
    count = 2
    mapping = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
        "output_frame": i, "sbs_source_frame": i, "physical_eye": "right", "sbs_index": 1,
        "physical_eye_verified": False, "acquisition_frame_id": "UNKNOWN",
        "qpc_declared_ns": 1_000_000_000 + i * 40_000_000,
        "timebase_status": r.TIMEBASE} for i in range(count)]
    source_dir = tmp_path / "Source"
    source_dir.mkdir()
    rgb = np.full((count, 480, 640, 3), [9, 19, 29], np.uint8)
    np.savez_compressed(source_dir / "RAW_RGB.npz", rgb=rgb)
    source = {"schema": SCHEMA, "stage": "Source", "status": "SEALED_COMPLETE",
        "component_signature": "a" * 64, "native_frames": count, "session_id": "synthetic_001",
        "outputs": [file_ref(source_dir / "RAW_RGB.npz"),
            json_ref(source_dir / "FRAME_MAP.json", mapping),
            json_ref(source_dir / "SOURCE_REPORT.json", {"native_sidecar_frames": count})]}
    source_ref = json_ref(source_dir / "RECEIPT.json", source)
    arrays = {"q_arm": np.zeros((count, 2, 7)), "q_hand": np.zeros((count, 2, 22)),
        "T_world_base": np.eye(4), "c2w": np.tile(np.eye(4), (count, 1, 1)),
        "K": np.asarray([[1000., 0, 640], [0, 1000., 480], [0, 0, 1]]),
        "qpc_ns": np.asarray([row["qpc_declared_ns"] for row in mapping], np.int64)}
    # Moving ego camera must NOT make the fixed external camera move.
    arrays["c2w"][1, 0, 3] = .15
    states = [{"native_frame_index": i, "source_frame_map": deepcopy(row),
        "timebase_status": r.TIMEBASE, "hands": {
            "left": {"robot_status": "TRACK_GATE_FAIL", "training_eligible": False},
            "right": {"robot_status": "CANDIDATE", "training_eligible": False}}}
        for i, row in enumerate(mapping)]
    report = {"session_id": source["session_id"], "native_frames": count,
              "native_image_size": [1280, 960], "kinematic_actions_generated": True}
    motion_dir = tmp_path / "Motion"
    motion_dir.mkdir()

    def write_motion():
        np.savez_compressed(motion_dir / "MOTION.npz", **arrays)
        native_states = json_ref(motion_dir / "Native/FRAME_STATUS.json", [
            {"native_frame_index": i, "native_only_schema": True} for i in range(count)])
        native_receipt = json_ref(motion_dir / "Native/RECEIPT.json", {
            "schema": SCHEMA, "stage": "MotionSource", "status": "SEALED_COMPLETE",
            "component_signature": "f" * 64, "outputs": [native_states]})
        motion["outputs"] = [file_ref(motion_dir / "MOTION.npz"),
            json_ref(motion_dir / "FRAME_MAP.json", mapping),
            json_ref(motion_dir / "FRAME_STATUS.json", states),
            json_ref(motion_dir / "MOTION_REPORT.json", report), native_receipt, native_states]
        return json_ref(motion_dir / "RECEIPT.json", motion)

    motion = {"schema": SCHEMA, "stage": "Motion", "status": "SEALED_COMPLETE",
        "component_signature": "b" * 64, "native_frames": count, "session_id": source["session_id"],
        "source_receipt": source_ref, "source_signature": "a" * 64,
        "kinematic_actions_generated": True,
        "consumer_status": {"Render": "READY_OFFLINE_VISUAL_CANDIDATE"}}
    motion_ref = write_motion()
    asset = tmp_path / "assets/fake_mesh.stl"
    asset.parent.mkdir()
    asset.write_bytes(b"synthetic mesh; never loaded")
    closure = {"status": "VERIFIED_STATIC_ASSET_CLOSURE", "assets": [file_ref(asset)],
               "asset_count": 1, "issues": []}
    asset_ref = json_ref(tmp_path / "ASSET_CLOSURE.json", closure)
    records, video_records = [], []
    state = {"gate_calls": 0, "revoke_at": None, "identity_at": None, "buffer_error": None,
             "decode_count": count, "passes": 2, "loads": 0}
    triangles = np.asarray([[[0, 0, .01], [1, 0, .10], [0, 1, .10]]], float)
    colours = np.asarray([[30, 20, 10]], np.uint8)
    ids = {4000: {"instance_id": "left_flange", "part_id": "flange", "link_id": "flange_left"},
           2000: {"instance_id": "left_kaihand", "part_id": "palm", "link_id": "hand_l_base_link"},
           2001: {"instance_id": "left_kaihand", "part_id": "finger", "link_id": "index_link"},
           0: {"instance_id": "left_tianji", "part_id": "arm", "link_id": "Link1_L"}}

    def zbuffer(tris, cols, labels, fx, fy, cx, cy, width, height):
        assert np.all(tris[:, :, 2] >= .05)
        return None, None, None

    def render_robot(wrapper, assets, q_arm, q_hand, mounts, camera, K, cache, flange, **kwargs):
        assert kwargs == {"complete_robot": True, "return_buffers": True, "width": 640, "height": 480}
        assert q_arm.shape == (2, 7) and q_hand.shape == (2, 22)
        records.append({"camera": camera.copy(), "K": K.copy(), "kwargs": kwargs})
        for labels in (np.asarray([4000]), np.asarray([0]))[:state["passes"]]:
            wrapper.rasterize_zbuffer(triangles, colours, labels,
                K[0, 0], K[1, 1], K[0, 2], K[1, 2], 640, 480)
        bgr = np.zeros((480, 640, 3), np.uint8)
        labels = np.full((480, 640), -1, np.int32)
        depth = np.full((480, 640), np.inf, np.float32)
        triangle = np.full((480, 640), -1, np.int64)
        for x, label in enumerate(ids):
            labels[0, x], depth[0, x], triangle[0, x], bgr[0, x] = label, .75, 42 + x, [30, 20, 10]
        buffers = {"depth_m": depth, "triangle_id": triangle,
                   "label_identity": deepcopy(ids), "one_scene_zbuffer": True}
        error = state["buffer_error"]
        if error == "hand_roots":
            return bgr, labels, np.tile(np.eye(4), (2, 1, 1))
        if error == "depth":
            depth[0, 0] = np.inf
        elif error == "background":
            depth[1, 0] = 0
        elif error == "triangle":
            triangle[0, 0] = -1
        elif error == "identity":
            buffers["label_identity"].pop(4000)
        elif error == "identity_drift" and len(records) > 1:
            buffers["label_identity"][4000]["link_id"] = "changed"
        return bgr, labels, buffers

    def over(background, layer, alpha):
        return np.where(alpha[..., None], layer, background).astype(np.uint8)

    def intrinsics(K, width, height, native):
        assert native == (1280, 960)
        value = K.copy()
        value[0] *= width / native[0]
        value[1] *= height / native[1]
        return value

    display = SimpleNamespace(near_clip=pure_v1_clipping(), scaled_intrinsics=intrinsics,
        external_camera=lambda initial: initial + np.diag([0., 0., 0., 0.]),
        EXTERNAL_K=np.asarray([[575., 0, 319.5], [0, 575., 239.5], [0, 0, 1]]),
        over=over, on_grey=lambda layer, alpha: over(np.full_like(layer, 242), layer, alpha))
    runtime = SimpleNamespace(raster=SimpleNamespace(render_robot=render_robot, rasterize_zbuffer=zbuffer),
        display=display, robot=SimpleNamespace(assets=object(), tool_mounts=np.tile(np.eye(4), (2, 1, 1))),
        cache={}, flange=np.zeros((1, 3, 3)))

    def load(_):
        state["loads"] += 1
        return runtime

    def writer(path, frames, frame_map):
        assert frame_map == mapping
        video_records.append({"path": path, "shape": frames.shape, "frames": frames.copy()})
        path.write_bytes(b"synthetic video; not a real encode")
        return {"video": file_ref(path), "decoded_frames": state["decode_count"],
                "timebase_status": r.TIMEBASE, "visual_quality": "NOT_REVIEWED"}

    def authority():
        state["gate_calls"] += 1
        return {"execution_allowed": state["revoke_at"] != state["gate_calls"],
            "algorithm_execution_allowed": True, "task_id": "synthetic_registered_task",
            "run_signature": "d" * 64,
            "fencing_token": "changed" if state["identity_at"] == state["gate_calls"] else "fence"}

    monkeypatch.setattr(r, "_load_render_runtime", load)
    monkeypatch.setattr(r, "_write_review_video", writer)
    args = dict(source_receipt=source_ref, source_signature="a" * 64,
        motion_receipt=motion_ref, motion_signature="b" * 64,
        output_dir=tmp_path / "Render", component_signature="c" * 64,
        repo_root=tmp_path, authorization=authority, asset_closure=asset_ref)
    return SimpleNamespace(args=args, source=source, motion=motion, arrays=arrays,
        mapping=mapping, states=states, report=report, records=records, state=state,
        videos=video_records, write_motion=write_motion, asset=asset,
        refresh_motion=lambda: args.update(motion_receipt=write_motion()))


def test_full_native_sealed_render_and_fixed_camera(case):
    result = r.build_render_stage(**case.args)
    assert result["stage"] == "Render" and result["status"] == "SEALED_COMPLETE"
    assert result["render_calls"] == 4 and result["zbuffer_passes"] == 8
    assert result["native_frames"] == 2 and result["complete_robot"] is True
    assert result["control_ground_truth"] is False and result["training_eligible"] is False
    assert result["visual_quality"] == "NOT_REVIEWED" and result["human_review"] == "PENDING"
    verify_sealed_stage(result, "c" * 64, output_root=case.args["output_dir"])
    with np.load(case.args["output_dir"] / "ROBOT_BUFFERS.npz", allow_pickle=False) as arrays:
        assert arrays["rgba"].shape == (2, 480, 640, 4)
        assert arrays["rgba"][0, 0, 0].tolist() == [10, 20, 30, 255]
        assert arrays["depth_m"][0, 0, 0] == .75  # not hand-root translation
        assert arrays["triangle_id"][0, 0, 0] == 42
        assert arrays["link_id"][0, 0, 0] == 4000
        assert arrays["K"][0, 0] == 500 and arrays["K"][0, 2] == 320
        assert np.isposinf(arrays["depth_m"][0, 1, 0])
    assert np.array_equal(case.records[1]["camera"], case.records[3]["camera"])
    assert not np.array_equal(case.records[0]["camera"], case.records[2]["camera"])
    frame_status = json.loads((case.args["output_dir"] / "FRAME_STATUS.json").read_text())
    assert frame_status[0]["hands"]["left"]["robot_status"] == "TRACK_GATE_FAIL"
    assert frame_status[0]["render_status"] == "COMPLETE_DIAGNOSTIC_RENDER"
    assert any(Path(ref["path"]).parts[-2:] == ("Native", "FRAME_STATUS.json")
               for ref in result["actual_inputs"])
    assert json.loads((case.args["output_dir"] / "FRAME_MAP.json").read_text()) == case.mapping
    assert [row["shape"] for row in case.videos] == [(2, 480, 1280, 3), (2, 480, 640, 3)]
    assert case.videos[0]["frames"][0, 0, 0].tolist() == [9, 19, 29]
    assert case.videos[0]["frames"][0, 0, 640].tolist() == [10, 20, 30]


def test_near_clip_preserves_original_triangle_id_on_both_passes():
    captured = []
    raster = SimpleNamespace(rasterize_zbuffer=lambda *args: captured.append(args))
    safe = r._NearClippedRaster(raster, pure_v1_clipping())
    triangles = np.asarray([[[0, 0, .01], [1, 0, .10], [0, 1, .10]],
                            [[0, 0, -.2], [1, 0, -.1], [0, 1, -.1]],
                            [[0, 0, .2], [1, 0, .2], [0, 1, .2]]])
    colours = np.full((3, 3), 17, np.uint8)
    safe.rasterize_zbuffer(triangles, colours, np.asarray([4000, 2000, 42]), 1, 1, 0, 0, 2, 2)
    safe.rasterize_zbuffer(triangles, colours, np.arange(3), 1, 1, 0, 0, 2, 2)
    assert safe.calls == 2
    assert captured[0][2].tolist() == [42, 4000, 4000]
    assert captured[1][2].tolist() == [2, 0, 0]  # NOT renumbered after clipping
    assert np.all(captured[0][0][:, :, 2] >= .05)
    assert np.array_equal(captured[0][0], captured[1][0])


@pytest.mark.parametrize("field", ["source_signature", "motion_signature", "component_signature"])
def test_bad_signature_before_runtime(case, field):
    case.args[field] = "bad"
    with pytest.raises(r.RenderStageError, match="SIGNATURE_REQUIRED"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


@pytest.mark.parametrize("at", [1, 2, 3, 4])
def test_gate_revocation_no_final_receipt(case, at):
    case.state["revoke_at"] = at
    with pytest.raises(r.RenderStageError, match="NOT_AUTHORIZED"):
        r.build_render_stage(**case.args)
    assert not (case.args["output_dir"] / "RECEIPT.json").exists()
    if at <= 2:
        assert case.state["loads"] == 0


def test_writer_fence_change_rejected(case):
    case.state["identity_at"] = 3
    with pytest.raises(r.RenderStageError, match="WRITER_IDENTITY_CHANGED"):
        r.build_render_stage(**case.args)
    assert not (case.args["output_dir"] / "RECEIPT.json").exists()


def test_direct_no_packet_authority_rejected(case):
    case.args["authorization"] = None
    with pytest.raises(r.RenderStageError, match="AUTHORIZATION_REQUIRED"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


@pytest.mark.parametrize("field,value", [("status", "PARTIAL"), ("source_signature", "e" * 64),
                                        ("native_frames", 99), ("session_id", "other")])
def test_wrong_upstream_identity_or_partial(case, field, value):
    case.motion[field] = value
    case.refresh_motion()
    with pytest.raises(ValueError):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_unsupported_camera_does_not_construct_robot(case):
    case.motion["kinematic_actions_generated"] = False
    case.motion["consumer_status"]["Render"] = "BLOCKED"
    case.refresh_motion()
    with pytest.raises(r.RenderStageError, match="BLOCKED_MOTION_CAMERA"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


@pytest.mark.parametrize("error", ["hand_roots", "depth", "background", "triangle", "identity", "identity_drift"])
def test_wrong_real_buffer_contract_never_seals(case, error):
    case.state["buffer_error"] = error
    with pytest.raises(r.RenderStageError):
        r.build_render_stage(**case.args)
    assert not (case.args["output_dir"] / "RECEIPT.json").exists()


def test_single_pass_not_accepted_as_triangle_provenance(case):
    case.state["passes"] = 1
    with pytest.raises(r.RenderStageError, match="BOTH_ZBUFFER_PASSES"):
        r.build_render_stage(**case.args)


def test_actual_video_decode_must_match_native_count(case):
    case.state["decode_count"] = 1
    with pytest.raises(r.RenderStageError, match="FULL_DECODE_COUNT"):
        r.build_render_stage(**case.args)
    assert not (case.args["output_dir"] / "RECEIPT.json").exists()


@pytest.mark.parametrize("key", ["q_arm", "q_hand", "K", "c2w"])
def test_nonfinite_motion_rejected_before_fk(case, key):
    case.arrays[key].flat[0] = np.nan
    case.refresh_motion()
    with pytest.raises(r.RenderStageError, match="MOTION_ARRAY_INVALID"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_qpc_drift_rejected_before_runtime(case):
    case.arrays["qpc_ns"][1] += 1
    case.refresh_motion()
    with pytest.raises(r.RenderStageError, match="MOTION_QPC_CHANGED"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_camera_domain_not_scaled_twice(case):
    case.report["native_image_size"] = [640, 480]
    case.refresh_motion()
    with pytest.raises(r.RenderStageError, match="CAMERA_DOMAIN_MISMATCH"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_changed_frame_status_rejected(case):
    case.states[0]["source_frame_map"]["physical_eye"] = "left"
    case.refresh_motion()
    with pytest.raises(r.RenderStageError, match="FRAME_IDENTITY_CHANGED"):
        r.build_render_stage(**case.args)


def test_missing_or_drifting_asset_closure_rejected(case):
    case.asset.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_missing_closure_before_runtime(case):
    case.args["asset_closure"] = None
    with pytest.raises(r.RenderStageError, match="ASSET_CLOSURE_REQUIRED"):
        r.build_render_stage(**case.args)


def test_fresh_attempt_required(case):
    case.args["output_dir"].mkdir()
    with pytest.raises(r.RenderStageError, match="FRESH_OUTPUT_REQUIRED"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_output_symlink_outside_project_rejected(case):
    case.args["output_dir"] = Path("/tmp") / "forbidden-render-output"
    with pytest.raises(r.RenderStageError, match="OUTSIDE_PROJECT"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_nested_frame_status_is_not_substituted_for_motion_frame_status(case):
    receipt, direct = r._sealed(case.args["motion_receipt"], "b" * 64, "Motion", case.args["repo_root"])
    assert len(receipt["outputs"]) == 6  # including Native receipt and state
    assert Path(direct["FRAME_STATUS.json"]["path"]).parent == case.args["repo_root"] / "Motion"
    assert set(direct) == {"MOTION.npz", "FRAME_MAP.json", "FRAME_STATUS.json", "MOTION_REPORT.json"}


def test_nested_output_sha_still_verified_before_runtime(case):
    nested = next(row for row in case.motion["outputs"]
                  if Path(row["path"]).parts[-2:] == ("Native", "FRAME_STATUS.json"))
    Path(nested["path"]).write_text("corrupt Native evidence")
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0


def test_nested_status_cannot_fill_missing_direct_status(case):
    case.motion["outputs"] = [row for row in case.motion["outputs"]
        if not (Path(row["path"]).name == "FRAME_STATUS.json"
                and Path(row["path"]).parent.name == "Motion")]
    case.args["motion_receipt"] = json_ref(Path(case.args["motion_receipt"]["path"]), case.motion)
    with pytest.raises(r.RenderStageError, match="MOTION_OUTPUT_CONTRACT_INCOMPLETE"):
        r.build_render_stage(**case.args)
    assert case.state["loads"] == 0
