"""Model-free checks for the development-only flange-link Robot successor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from chaoyang.pipeline import exact78_fresh_robot_flange_successor_v1 as candidate
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage_v1
from chaoyang.pipeline import exact78_robot_raster_v1 as raster_v1


class FlangeSuccessorTests(unittest.TestCase):
    def test_existing_code_closure_is_exactly_pinned(self) -> None:
        candidate._check_code_pins()

    def test_only_two_development_seals_are_admitted(self) -> None:
        for session_id, spec in candidate._SEALED_DEVELOPMENT.items():
            root = candidate.ROOT / spec["root"]
            audit = json.loads((root / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
            accepted = candidate.validate_sealed_development(
                hawor_npz=root / "HAWOR_CAMERA_PREDICTIONS.npz",
                frame_provenance=root / "FRAME_PROVENANCE.json",
                input_audit=root / "INPUT_AUDIT.json",
                raw_video=Path(audit["raw_source_ref"]["path"]),
                source_kind=spec["source_kind"], session_id=session_id)
            self.assertEqual(accepted["frames"], spec["frames"])
        with self.assertRaisesRegex(stage_v1.FreshRobotError, "DEVELOPMENT_ONLY"):
            candidate.validate_sealed_development(
                hawor_npz=Path("holdout.npz"), frame_provenance=Path("holdout.json"),
                input_audit=Path("holdout_audit.json"), raw_video=Path("holdout.mp4"),
                source_kind="tracker", session_id="play_cards_0901_001")

    def test_wrong_input_path_or_sha_fails_before_robot_stage(self) -> None:
        spec = candidate._SEALED_DEVELOPMENT["play_cards_0902_042"]
        root = candidate.ROOT / spec["root"]
        audit = json.loads((root / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
        kwargs = dict(
            hawor_npz=root / "HAWOR_CAMERA_PREDICTIONS.npz",
            frame_provenance=root / "FRAME_PROVENANCE.json",
            input_audit=root / "INPUT_AUDIT.json",
            raw_video=Path(audit["raw_source_ref"]["path"]),
            source_kind="tracker", session_id="play_cards_0902_042")
        with self.assertRaisesRegex(stage_v1.FreshRobotError, "SEALED_PATH_DRIFT"):
            candidate.validate_sealed_development(**{**kwargs, "hawor_npz": Path("old_hawor.npz")})
        with mock.patch.object(candidate, "_sha", return_value="0" * 64):
            with self.assertRaisesRegex(stage_v1.FreshRobotError, "SEALED_SHA_DRIFT"):
                candidate.validate_sealed_development(**kwargs)
        with mock.patch.object(stage_v1, "run_robot_stage") as forbidden:
            with self.assertRaisesRegex(stage_v1.FreshRobotError, "DEVELOPMENT_ONLY"):
                candidate.run_flange_robot_stage(
                    **{**kwargs, "session_id": "play_cards_0915_002",
                       "output_root": candidate.ROOT / "_run" / "never_created_flange_test"})
            forbidden.assert_not_called()

    def test_ik_flange_fk_not_tool_fk_with_pinned_urdf(self) -> None:
        assets = stage_v1.load_pinned_robot_assets(candidate.ROOT)
        for side in range(2):
            q = np.zeros(7)
            flange = candidate.flange_fk(assets, side, q)
            tool = stage_v1.scene._tool_fk(assets, side, q)
            flange_to_tool = np.linalg.inv(flange) @ tool
            np.testing.assert_allclose(flange_to_tool[:3, 3], [0, 0, .145], atol=1e-9)
            self.assertGreater(np.linalg.norm(flange_to_tool[:3, :3] - np.eye(3)), 1.0)
            self.assertFalse(np.allclose(flange, tool))

    def test_raster_alias_changes_only_hand_root_parent_and_restores(self) -> None:
        before_tool = stage_v1._tool_fk
        before_encoder = stage_v1._encoder
        fake = lambda model, values: {  # noqa: E731 - minimal fake FK table
            "flange_L": np.eye(4), "flange_R": np.eye(4) * 2,
            "left_tool": np.eye(4) * 3, "right_tool": np.eye(4) * 4,
            "unrelated_link": np.eye(4) * 5}
        with mock.patch.object(raster_v1, "forward_kinematics", fake):
            with self.assertRaisesRegex(RuntimeError, "restore"):
                with candidate._flange_link_override():
                    self.assertIs(stage_v1._tool_fk, candidate.flange_fk)
                    self.assertIs(stage_v1._encoder, candidate.ffmpeg44.ffmpeg44_encoder)
                    table = raster_v1.forward_kinematics(None, {})
                    np.testing.assert_array_equal(table["left_tool"], table["flange_L"])
                    np.testing.assert_array_equal(table["right_tool"], table["flange_R"])
                    np.testing.assert_array_equal(table["unrelated_link"], np.eye(4) * 5)
                    raise RuntimeError("restore")
            self.assertIs(raster_v1.forward_kinematics, fake)
        self.assertIs(stage_v1._tool_fk, before_tool)
        self.assertIs(stage_v1._encoder, before_encoder)

    def test_camera_fk_audit_keeps_fixed_flange_offset_without_new_gate(self) -> None:
        contract = json.loads(stage_v1.MOUNT.read_text(encoding="utf-8"))
        mounts = np.asarray((
            contract["proxy_transforms"]["left_flange_L_to_hand_l_base_link"],
            contract["proxy_transforms"]["right_flange_R_to_hand_r_base_link"],
        ))
        self.assertEqual(contract["proxy_transforms"]["matrix_direction"],
                         "p_flange = T_flange_hand_root_proxy @ p_hand_root")
        with tempfile.TemporaryDirectory(prefix="flange_cpu_test_", dir=candidate.ROOT / "_run") as temp:
            path = Path(temp) / "states.npz"
            np.savez_compressed(path, q_arm=np.zeros((2, 2, 7)),
                                T_camera_base=np.eye(4), T_tool_hand_root=mounts)
            audit = candidate._chain_audit(path, 2)
        np.testing.assert_allclose(audit["flange_to_hand_root_m"],
                                   [0.07519998737695313] * 2, atol=1e-12)
        self.assertLess(audit["max_relative_matrix_error"], 1e-8)
        self.assertEqual(audit["zero_step_pairs"], [1, 1])
        self.assertFalse(audit["continuity_quality_pass"])
        self.assertIn("NOT_CHECKED", audit["camera_projection_chain_claim"])


if __name__ == "__main__":
    unittest.main()
