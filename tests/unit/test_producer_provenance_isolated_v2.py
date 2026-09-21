"""CPU adapter tests; frozen-fit stubs do NOT certify a fresh solver run."""
import importlib.util
import io
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


def isolated(name):
    path = Path(__file__).resolve().parents[2] / "src/chaoyang/ops" / (name + ".py")
    spec = importlib.util.spec_from_file_location("isolated_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampling_downscope_only_preserves_numeric_and_validity():
    m = isolated("run_pico_manus_motion_v2")
    valid = np.ones((3, 2, 21), bool); valid[1, 0, 8] = False
    motion = dict(joint_observed_local=valid.copy(), controller_pose_observed=valid.all(-1),
                  manus_local_joint_valid=valid, joint_inferred=np.zeros_like(valid),
                  q=np.array([0., -0., np.nan]), timestamp_ns=np.array([2, 3, 5], np.int64))
    before = {k: v.copy() for k, v in motion.items()}
    m.apply_source_sampling_provenance(motion, {"timestamp_clock": "PICO timeStampNs (sensor clock, resample timeline)"})
    for key in set(before) - {"joint_observed_local", "controller_pose_observed"}:
        assert motion[key].dtype == before[key].dtype
        assert motion[key].tobytes() == before[key].tobytes()
    assert not motion["joint_observed_local"].any()
    assert not motion["controller_pose_observed"].any()
    np.testing.assert_array_equal(motion["provenance_v2_controller_pose_computable"], before["controller_pose_observed"])
    assert not motion["provenance_v2_exact_interpolation_support_known"].any()


def test_unknown_sampling_contract_rejected():
    m = isolated("run_pico_manus_motion_v2")
    with pytest.raises(ValueError): m.apply_source_sampling_provenance({}, {})


@pytest.mark.parametrize("kind", ["FROZEN_R0", "PICO_MANUS", "FROZEN_MANO21"])
def test_clipping_claim_branches(kind):
    m = isolated("run_full_robot_review_v2")
    claim = m.clipping_claim(kind)
    if kind == "FROZEN_R0":
        assert "was historically clipped" in claim
    else:
        assert "No posthoc clipping in this operation" in claim
        assert "explicit hand-only reference solver" in claim
        assert "was historically clipped" not in claim


def test_clipping_unknown_rejected():
    m = isolated("run_full_robot_review_v2")
    with pytest.raises(ValueError): m.clipping_claim("UNKNOWN")


def source_fixture():
    v = np.array([[True, True], [False, True], [True, False]])
    t = np.tile(np.eye(4), (3, 2, 1, 1)); t[:, :, 0, 3] = np.arange(3)[:, None] * .01
    source = dict(joints_world_m=np.arange(378, dtype=float).reshape(3, 2, 21, 3) / 1000,
                  joint_world_valid=np.repeat(v[..., None], 21, axis=-1),
                  timestamp_ns=np.array([0, 31_000_000, 67_000_000], np.int64),
                  frame_id=np.arange(3, dtype=np.int64), T_world_wrist=t,
                  wrist_world_valid=v, segment_id=np.zeros(3, np.int64),
                  wrist_position_observed=np.zeros_like(v), wrist_rotation_observed=np.zeros_like(v),
                  surface_valid=np.zeros_like(v))
    source["provenance_v2_upstream_resampled"] = source["joint_world_valid"].copy()
    source["provenance_v2_direct_observation_unverified"] = source["joint_world_valid"].copy()
    return source


def test_public_adapter_keeps_fixed_fit_and_motion_not_direct_authority():
    from chaoyang.ops import run_full_robot_review_v2 as original
    patched = isolated("run_full_robot_review_v2")
    source = source_fixture(); buf = io.BytesIO(); np.savez_compressed(buf, **source)
    physical = source["wrist_world_valid"][:, ::-1].copy()
    fit = dict(q22=np.arange(132, dtype=float).reshape(3, 2, 22) / 500,
               valid=physical, solver_success=physical.copy(), source_observed_physical=physical.copy())
    seen = []
    def frozen_fit(hands, joints, observed, ids, **kwargs):
        seen.append((joints.copy(), observed.copy(), ids.copy()))
        return {k: v.copy() for k, v in fit.items()}
    assets = SimpleNamespace(left_hand=SimpleNamespace(path="frozen_left"), right_hand=SimpleNamespace(path="frozen_right"))
    case = dict(kind="PICO_MANUS", motion={}, expected_frames=3)
    with patch.object(original, "read_frozen", return_value=buf.getvalue()), patch.object(patched, "read_frozen", return_value=buf.getvalue()), patch("chaoyang.pipeline.huro_hand_only_retarget_v1.load_hand_model", return_value=object()), patch("chaoyang.pipeline.huro_hand_frame_v2.solve_aligned_sequence", side_effect=frozen_fit):
        old, old_method = original.prepare_arrays(case, [], assets)
        new, new_method = patched.prepare_arrays(case, [], assets)
    assert old_method == new_method
    assert len(seen) == 2
    for before, after in zip(seen[0], seen[1]): assert before.tobytes() == after.tobytes()
    for key in set(old) - {"source_observed_physical"}:
        assert old[key].dtype == new[key].dtype
        assert old[key].shape == new[key].shape
        assert old[key].tobytes() == new[key].tobytes(), key
    assert not new["source_observed_physical"].any()
    np.testing.assert_array_equal(new["source_computable_physical"], physical)
    np.testing.assert_array_equal(new["provenance_v2_direct_observation_unverified"], physical)
    assert not new["provenance_v2_exact_interpolation_support_known"].any()


def test_uncorrected_source_rejected_before_solver():
    m = isolated("run_full_robot_review_v2")
    source = source_fixture(); del source["provenance_v2_direct_observation_unverified"]
    buf = io.BytesIO(); np.savez_compressed(buf, **source)
    with patch.object(m, "read_frozen", return_value=buf.getvalue()), patch("chaoyang.pipeline.huro_hand_frame_v2.solve_aligned_sequence") as solve:
        with pytest.raises(ValueError, match="PICO_SOURCE_PROVENANCE_CORRECTION_REQUIRED"):
            m.prepare_arrays(dict(kind="PICO_MANUS", motion={}, expected_frames=3), [], object())
        solve.assert_not_called()


@pytest.mark.skipif(not os.environ.get("AI1_PRODUCER_REPLAY_DIR"), reason="Explicit own-lane real-data CPU replay only")
def test_fixed_sessions_replay_and_frozen_fit_consumer():
    """Real motion replay; Robot adapter only with saved fit, NEVER fresh IK."""
    lane = Path("/mnt/workspace/code/chaoyang/_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai1")
    output = Path(os.environ["AI1_PRODUCER_REPLAY_DIR"]).resolve()
    assert output != lane and output.is_relative_to(lane) and not output.exists()
    output.mkdir()
    motion_op = isolated("run_pico_manus_motion_v2")
    robot_op = isolated("run_full_robot_review_v2")
    sessions = []
    for sid, count in zip(("097", "098", "101"), (165, 179, 122)):
        session = "play_cards_0916_" + sid
        motion, video, provenance = motion_op.load_session(
            Path("/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916"),
            Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916"), session, count)
        folder = output / session; folder.mkdir()
        with (folder / "REPLAYED_MOTION.npz").open("xb") as f: np.savez_compressed(f, **motion)
        previous = lane / "motion_adapter_v2_run1" / session / "CONTROLLER_MANUS_MOTION_V2.npz"
        corrected = lane / "provenance_correction_v2_run1" / session / "CONTROLLER_MANUS_MOTION_CORRECTED.npz"
        numeric_keys = []
        with np.load(previous, allow_pickle=False) as before:
            for k in before.files:
                if k in {"joint_observed_local", "controller_pose_observed"}: continue
                assert before[k].dtype == motion[k].dtype and before[k].shape == motion[k].shape
                assert before[k].tobytes() == motion[k].tobytes(), k
                numeric_keys.append(k)
        with np.load(corrected, allow_pickle=False) as expected:
            assert set(expected.files) == set(motion)
            for k in expected.files:
                assert expected[k].dtype == motion[k].dtype and expected[k].shape == motion[k].shape
                assert expected[k].tobytes() == motion[k].tobytes(), k
        assert provenance["observed_manus_hand_frames"] == {"left": 0, "right": 0}
        assert provenance["source_valid_manus_hand_frames"] == {"left": count, "right": count}
        old_input = lane / "full_robot_v2_run1" / session / "MOTION_INPUT.npz"
        with np.load(old_input, allow_pickle=False) as z: frozen = {k: z[k] for k in z.files}
        fit = {"q22": frozen["q22"], "valid": frozen["finger_valid"],
               "solver_success": frozen["finger_solver_success"], "source_observed_physical": frozen["source_observed_physical"]}
        buf = io.BytesIO(); np.savez_compressed(buf, **motion)
        seen = []
        def saved_fit(hands, joints, observed, ids, **kwargs):
            np.testing.assert_array_equal(observed, motion["joint_world_valid"].all(-1).T)
            np.testing.assert_array_equal(ids, motion["frame_id"])
            np.testing.assert_array_equal(joints, motion["joints_world_m"].transpose(1, 0, 2, 3))
            seen.append(True)
            return {k: v.copy() for k, v in fit.items()}
        assets = SimpleNamespace(left_hand=SimpleNamespace(path="frozen_left"), right_hand=SimpleNamespace(path="frozen_right"))
        with patch.object(robot_op, "read_frozen", return_value=buf.getvalue()), patch("chaoyang.pipeline.huro_hand_only_retarget_v1.load_hand_model", return_value=object()), patch("chaoyang.pipeline.huro_hand_frame_v2.solve_aligned_sequence", side_effect=saved_fit):
            robot, method = robot_op.prepare_arrays(dict(kind="PICO_MANUS", motion={}, expected_frames=count), [], assets)
        assert len(seen) == 1
        robot_keys = []
        for k in set(frozen) - {"source_observed_physical"}:
            assert frozen[k].dtype == robot[k].dtype and frozen[k].shape == robot[k].shape
            assert frozen[k].tobytes() == robot[k].tobytes(), k
            robot_keys.append(k)
        assert not robot["source_observed_physical"].any()
        with (folder / "ROBOT_ADAPTER_ONLY_REPLAY.npz").open("xb") as f: np.savez_compressed(f, **robot)
        sessions.append({"session_id": session, "frames": count, "motion_numeric_bit_exact_keys": numeric_keys,
                         "corrected_motion_all_fields_bit_exact": True, "robot_adapter_preserved_keys": sorted(robot_keys),
                         "robot_test_scope": "CONSUMER_ADAPTER_ONLY_USING_PREVIOUSLY_FROZEN_FIT_NO_SOLVER_RUN", "source_method": method})
    def digest(path):
        raw = path.read_bytes(); return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    result = {"schema_version": "ISOLATED_PRODUCER_PROVENANCE_REPLAY_V2", "status": "PASS", "sessions": sessions,
              "gpu_used": False, "source_data_modified": False, "numerical_optimizer_rerun": False,
              "video_rerendered": False, "source_ops": [digest(Path(motion_op.__file__)), digest(Path(robot_op.__file__))]}
    with (output / "RESULT.json").open("x") as f: json.dump(result, f, indent=2); f.write("\n")
