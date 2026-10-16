"""Model-free checks for the unrouted 0901 Chips001 structural Robot candidate."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_fresh_robot_flange_successor_v1 as flange
from chaoyang.pipeline import exact78_robot_raster_v1 as raster
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction
from chaoyang.pipeline import exact78_newraw_robot_structural_v2_candidate as candidate


class StructuralV2CandidateTests(unittest.TestCase):
    def test_sealed_0901_cpu_check_does_not_solve_or_render(self) -> None:
        with mock.patch.object(legacy, "solve_camera_diagnostic", side_effect=AssertionError("solve")), \
             mock.patch.object(legacy, "run_robot_stage", side_effect=AssertionError("stage")), \
             mock.patch.object(direction, "render_camera_diagnostic", side_effect=AssertionError("render")):
            result = candidate.check_only()
        self.assertEqual(result["status"], "CHECK_ONLY_READY_NOT_ROUTABLE")
        self.assertEqual((result["frame_count"], result["fps_nominal"]), (799, 25.0))
        self.assertEqual((result["model_calls"], result["render_calls"]), (0, 0))
        self.assertFalse(result["old_robot_state_consumed"])
        self.assertEqual(result["source_refs"]["hawor_npz"]["sha256"],
                         candidate._SOURCE_SHA["hawor_npz"])
        self.assertTrue(all("robot" not in role for role in result["source_refs"]))

    def test_wrong_source_sha_or_manifest_fails_closed(self) -> None:
        with mock.patch.object(candidate, "_V1_PACKET_SHA", "0" * 64):
            with self.assertRaisesRegex(legacy.FreshRobotError, "SOURCE_MANIFEST_DRIFT"):
                candidate._source_inputs()
        mutated = dict(candidate._SOURCE_SHA, hawor_npz="0" * 64)
        with mock.patch.object(candidate, "_SOURCE_SHA", mutated):
            with self.assertRaisesRegex(legacy.FreshRobotError, "SOURCE_REF_DRIFT:hawor_npz"):
                candidate._source_inputs()

    def test_two_structural_overrides_are_joint_and_restore_on_error(self) -> None:
        old_tool = legacy._tool_fk
        old_render = legacy.render_camera_diagnostic
        old_raster_fk = raster.forward_kinematics
        old_encoder = legacy._encoder
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with candidate._structural_override():
                self.assertIs(legacy._tool_fk, flange.flange_fk)
                self.assertIs(legacy.render_camera_diagnostic,
                              direction.render_camera_diagnostic)
                self.assertIs(legacy._encoder, flange.ffmpeg44.ffmpeg44_encoder)
                self.assertIsNot(raster.forward_kinematics, old_raster_fk)
                raise RuntimeError("restore")
        self.assertIs(legacy._tool_fk, old_tool)
        self.assertIs(legacy.render_camera_diagnostic, old_render)
        self.assertIs(raster.forward_kinematics, old_raster_fk)
        self.assertIs(legacy._encoder, old_encoder)

    def test_real_flange_not_legacy_tool_and_unchanged_numeric_gates(self) -> None:
        assets = legacy.load_pinned_robot_assets(candidate.ROOT)
        for side in range(2):
            q = np.zeros(7)
            actual = flange.flange_fk(assets, side, q)
            old = legacy._tool_fk(assets, side, q)
            self.assertFalse(np.allclose(actual, old))
            np.testing.assert_allclose((np.linalg.inv(actual) @ old)[:3, 3],
                                       [0, 0, .145], atol=1e-9)
        self.assertEqual((legacy.POSITION_GATE_MM, legacy.ROTATION_GATE_DEG), (10.0, 5.0))
        self.assertEqual((legacy.ARM_VELOCITY_GATE_RAD, legacy.ARM_ACCELERATION_GATE_RAD),
                         (0.12, 0.06))
        self.assertEqual((legacy.HAND_VELOCITY_GATE_RAD, legacy.HAND_ACCELERATION_GATE_RAD),
                         (0.08, 0.06))
        self.assertEqual((legacy.HAND_BONE_GATE_DEG, legacy.HAND_TIP_GATE_DEG),
                         (60.0, 15.0))

    def test_camera_mount_chain_uses_forward_base_and_proxy(self) -> None:
        proxy = json.loads(legacy.MOUNT.read_text(encoding="utf-8"))
        mounts = np.asarray((
            proxy["proxy_transforms"]["left_flange_L_to_hand_l_base_link"],
            proxy["proxy_transforms"]["right_flange_R_to_hand_r_base_link"],
        ))
        self.assertEqual(proxy["proxy_transforms"]["matrix_direction"],
                         "p_flange = T_flange_hand_root_proxy @ p_hand_root")
        with tempfile.TemporaryDirectory(prefix="structural_v2_cpu_", dir=candidate.ROOT / "_run") as temp:
            states = Path(temp) / "states.npz"
            np.savez_compressed(states, q_arm=np.zeros((candidate.FRAMES, 2, 7)),
                                T_camera_base=np.eye(4), T_tool_hand_root=mounts)
            result = candidate._mount_camera_audit(states)
        self.assertEqual(result["camera_rule"], "BASE_TO_CAMERA_NO_INVERSION")
        self.assertEqual(result["hand_root_rule"], "CAMERA_AT_FLANGE_LR_AT_PROXY_MOUNT")
        self.assertLess(result["max_relative_matrix_error"], 1e-8)
        np.testing.assert_allclose(result["flange_to_hand_root_m"],
                                   [.07519998737695313] * 2, atol=1e-12)
        self.assertEqual(result["zero_step_pairs"], [candidate.FRAMES - 1] * 2)
        self.assertFalse(result["continuity_quality_pass"])

    def test_unapproved_output_scope_rejected_before_stage(self) -> None:
        with mock.patch("chaoyang.governance.exact78_operation_gate.authorize_exact78_operation"), \
             mock.patch.object(legacy, "run_robot_stage") as stage:
            with self.assertRaisesRegex(legacy.FreshRobotError, "FRESH_OUTPUT_REQUIRED"):
                candidate.run_candidate(output_root=candidate.ROOT / "_run" / "wrong_attempt")
            stage.assert_not_called()

    def test_unregistered_run_is_denied_before_stage(self) -> None:
        with mock.patch.object(legacy, "run_robot_stage") as stage:
            with self.assertRaisesRegex(SystemExit, "EXACT78_GATE_DENIED"):
                candidate.run_candidate(output_root=(candidate.ROOT / "_run/current" /
                    candidate.TASK_ID / "attempts/attempt_0001/candidate/robot_v2"))
            stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
