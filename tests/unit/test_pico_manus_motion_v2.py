"""Known-answer CPU tests; no data/model/network/GPU access."""
import copy
import io
import importlib.util
from pathlib import Path
import unittest

import numpy as np

try:
    from chaoyang.pipeline import pico_manus_motion_v2 as m
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("pico_manus_motion_v2", Path(__file__).with_name("pico_manus_motion_v2.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)


def fixture():
    n = 4
    ext = np.eye(4)
    ext[:3, :3] = m.REP103_HEAD_TO_OPENXR.T  # make effective camera=head
    right = ext.copy()
    right[0, 3] = -.06
    calibration = {
        "extrinsic_convention": "head_to_camera_4x4_row_major",
        "hdf5_rep103_head_to_openxr_head": m.REP103_HEAD_TO_OPENXR.tolist(),
        "extrinsics": {"left": ext.tolist(), "right": right.tolist()},
        "left": {"sourceIndex": 1, "intrinsics": {"fx": 1000., "fy": 1000., "cx": 1024., "cy": 768.}},
        "right": {"sourceIndex": 0}, "width": 2048, "height": 1536,
    }
    head = np.zeros((n, 7)); head[:, 6] = 1
    ctrl = np.zeros((n, 2, 7)); ctrl[:, :, 6] = 1; ctrl[:, :, 2] = 1
    ctrl[:, 0, 0] = np.arange(n) * .02
    ctrl[:, 1, 0] = -.2 - np.arange(n) * .01
    local = np.zeros((n, 2, 25, 3)); local[:, :, 1:, 0] = np.arange(1, 25) * .005
    prior = {"schema": "controller_to_wrist_v1", "convention": "wrist_pose = compose_pose(controller_pose, transform)", "controller_pose_frame": "pico_world_rh_x_forward_y_left_z_up", "left": {"pos": [.1, 0, 0], "quat": [0, 0, 0, 1]}, "right": {"pos": [0, .1, 0], "quat": [0, 0, 0, 1]}}
    return dict(head_pose=head, controller_pose=ctrl, manus_local=local, hand_valid=np.ones((n, 2), bool), timestamp_ns=np.array([0, 30, 70, 120], np.int64) * 1_000_000 + 1_700_000_000_000_000_000, frame_id=np.arange(n), segment_id=np.zeros(n, np.int64), calibration=calibration, prior=prior)


class TestPicoManusMotionV2(unittest.TestCase):
    def test_known_composition(self):
        out = m.compose_motion(**fixture())
        np.testing.assert_allclose(out["T_world_wrist"][0, 0, :3, 3], [.1, 0, 1])
        np.testing.assert_allclose(out["joints_world_m"][0, 0, 1], [.105, 0, 1])
        np.testing.assert_allclose(out["T_camera_world"] @ out["T_world_camera"], np.broadcast_to(np.eye(4), (4, 4, 4)))

    def test_rotated_offset_and_wrong_order_negative(self):
        f = fixture(); f["controller_pose"][:, 0, 3:] = [0, 0, np.sqrt(.5), np.sqrt(.5)]
        out = m.compose_motion(**f)
        np.testing.assert_allclose(out["T_world_wrist"][0, 0, :3, 3], [0, .1, 1], atol=1e-12)
        wrong = out["T_controller_wrist_prior"][0] @ out["T_world_controller"][0, 0]
        self.assertGreater(np.linalg.norm(wrong[:3, 3] - out["T_world_wrist"][0, 0, :3, 3]), .1)

    def test_rotation_bias_does_not_move_wrist_origin(self):
        f = fixture(); first = m.compose_motion(**f)
        f["prior"]["left"]["quat"] = [0, 0, np.sqrt(.5), np.sqrt(.5)]
        second = m.compose_motion(**f)
        np.testing.assert_allclose(first["T_world_wrist"][:, :, :3, 3], second["T_world_wrist"][:, :, :3, 3])
        self.assertFalse(second["wrist_rotation_observed"].any())

    def test_head_motion_not_hand_world_motion(self):
        f = fixture(); first = m.compose_motion(**f)
        f["head_pose"][:, 0] = .2
        second = m.compose_motion(**f)
        np.testing.assert_allclose(first["joints_world_m"], second["joints_world_m"])
        np.testing.assert_allclose(first["joints_camera_m"] - second["joints_camera_m"], np.broadcast_to([.2, 0, 0], (4, 2, 21, 3)), atol=1e-12)

    def test_physical_left_not_legacy_named_left(self):
        out = m.compose_motion(**fixture())
        self.assertEqual(int(out["camera_source_index"]), 1)
        np.testing.assert_allclose(out["T_camera_wrist"][:, :, 0, 3] - out["legacy_right_T_camera_wrist"][:, :, 0, 3], .06)
        f = fixture(); f["calibration"]["left"]["sourceIndex"] = 0
        with self.assertRaises(m.MotionContractError): m.compose_motion(**f)

    def test_side_and_unit_mismatch_rejected(self):
        for extra in ({"side_names": ("right", "left")}, {"units": "mm"}):
            with self.assertRaises(m.MotionContractError): m.compose_motion(**fixture(), **extra)
        f = fixture(); f["manus_local"] *= 1000
        with self.assertRaises(m.MotionContractError): m.compose_motion(**f)

    def test_names_and_double_root_rejected(self):
        names = list(m.MANUS_NAMES); names[1], names[2] = names[2], names[1]
        with self.assertRaises(m.MotionContractError): m.compose_motion(**fixture(), joint_names=names)
        f = fixture(); f["manus_local"] += .1
        with self.assertRaises(m.MotionContractError): m.compose_motion(**f)

    def test_one_hand_missing_does_not_copy_or_clear_other(self):
        f = fixture(); f["hand_valid"][1, 0] = False
        out = m.compose_motion(**f)
        self.assertTrue(np.isnan(out["joints_world_m"][1, 0]).all())
        self.assertTrue(out["joint_world_valid"][1, 1].all())
        self.assertTrue(out["wrist_world_valid"][1, 0])  # controller still exists
        self.assertFalse(out["joint_inferred"].any())

    def test_invalid_controller_retains_local_manus(self):
        f = fixture(); f["controller_pose"][2, 0, 3:] = 0
        out = m.compose_motion(**f)
        self.assertFalse(out["wrist_world_valid"][2, 0])
        self.assertTrue(np.isnan(out["T_world_wrist"][2, 0]).all())
        self.assertTrue(out["joint_observed_local"][2, 0, 1:].all())
        self.assertFalse(out["joint_world_valid"][2, 0].any())
        self.assertTrue(out["wrist_world_valid"][2, 1])

    def test_invalid_head_preserves_world(self):
        f = fixture(); f["head_pose"][1, 3:] = 0
        out = m.compose_motion(**f)
        self.assertTrue(out["joint_world_valid"][1].all())
        self.assertFalse(out["joint_camera_valid"][1].any())
        self.assertTrue(np.isnan(out["joints_camera_m"][1]).all())

    def test_timestamp_duplicate_backward_and_wrong_units(self):
        for ts in ([0, 30, 30, 120], [0, 70, 30, 120]):
            f = fixture(); f["timestamp_ns"] = np.array(ts, np.int64) * 1_000_000
            with self.assertRaises(m.MotionContractError): m.compose_motion(**f)
        f = fixture(); f["timestamp_ns"] = np.array([0, 30, 70, 120])
        with self.assertRaises(m.MotionContractError): m.compose_motion(**f)

    def test_real_dt_and_segment_gaps(self):
        f = fixture(); f["segment_id"][2:] = 1; f["frame_id"][-1] = 7
        out = m.compose_motion(**f)
        np.testing.assert_allclose(out["timestamp_s"], [0, .03, .07, .12])
        np.testing.assert_array_equal(out["time_transition_valid"], [False, True, False, False])

    def test_long_time_gap_not_interpolated(self):
        f = fixture(); f["timestamp_ns"][-1] += 1_000_000_000
        out = m.compose_motion(**f)
        self.assertFalse(out["time_transition_valid"][-1])
        self.assertEqual(len(out["frame_id"]), 4)

    def test_known_projection_crop_resize(self):
        _, _, k = m.camera_contract(fixture()["calibration"])
        uv, valid = m.project_points(np.array([[.1, .2, 1], [0, 0, -1]]), k, np.ones(2, bool))
        np.testing.assert_allclose(uv[0], [702.5, 605.])
        self.assertFalse(valid[1]); self.assertTrue(np.isnan(uv[1]).all())

    def test_25_to_21_is_explicit_and_root_not_observed(self):
        out = m.compose_motion(**fixture())
        self.assertEqual(out["joints_world_m"].shape, (4, 2, 21, 3))
        np.testing.assert_array_equal(out["manus25_to_21"], [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24])
        self.assertFalse(out["joint_observed_local"][:, :, 0].any())
        self.assertFalse(out["surface_valid"].any())
        self.assertFalse(out["training_eligible"])

    def test_reflection_and_axis_contract_rejected(self):
        reflected = np.eye(4); reflected[0, 0] = -1
        with self.assertRaises(m.MotionContractError): m.require_transform(reflected)
        f = fixture(); f["calibration"]["hdf5_rep103_head_to_openxr_head"] = np.eye(3).tolist()
        with self.assertRaises(m.MotionContractError): m.compose_motion(**f)

    def test_semantic_replay_is_deterministic(self):
        a, b = m.compose_motion(**fixture()), m.compose_motion(**fixture())
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_future_change_does_not_change_prefix(self):
        f = fixture(); a = m.compose_motion(**f)
        f["controller_pose"][-1, :, :3] += .3
        f["manus_local"][-1, :, 1:, :] *= .8
        b = m.compose_motion(**f)
        np.testing.assert_array_equal(a["joints_world_m"][:-1], b["joints_world_m"][:-1])
        self.assertFalse(np.array_equal(a["joints_world_m"][-1], b["joints_world_m"][-1]))

    def test_npz_roundtrip_preserves_nan_masks_and_clock(self):
        f = fixture(); f["hand_valid"][1, 0] = False
        a = m.compose_motion(**f)
        stream = io.BytesIO(); np.savez_compressed(stream, **a); stream.seek(0)
        with np.load(stream, allow_pickle=False) as z:
            for key in a:
                np.testing.assert_array_equal(a[key], z[key], err_msg=key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
