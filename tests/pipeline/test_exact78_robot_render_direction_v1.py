"""CPU-only proof that render direction changes without invoking the raster."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction


STAGE_RESULT = (REPO_ROOT / "_run/current/exact78_raw_to_robot_t2_0902_cards042_v1_20260930"
                "/attempts/attempt_0001/candidate/robot/v1_stage_with_ffmpeg44_encoder/RESULT.json")
OUTPUT = REPO_ROOT / "_run/current/exact78_robot_render_direction_cards042_v1_20260930/attempts/attempt_0001/candidate"
TASK = "exact78_robot_render_direction_cards042_v1_20260930"


def sealed_spec() -> dict:
    stage = json.loads(STAGE_RESULT.read_text(encoding="utf-8"))
    return {"task_id": TASK, "session_id": "play_cards_0902_042", "source_kind": "TRACKER_CLIP_RGB",
            "inputs": {"stage_result": artifact_ref(STAGE_RESULT),
                       "states": stage["outputs"]["states"],
                       **{key: stage["inputs"][key] for key in (
                           "hawor_npz", "frame_provenance", "input_audit", "raw_video")}},
            "output_root": str(OUTPUT)}


class RenderDirectionTests(unittest.TestCase):
    def test_render_body_has_only_one_ast_change(self):
        original = ast.parse(inspect.getsource(legacy.render_camera_diagnostic)).body[0]
        successor = ast.parse(inspect.getsource(direction.render_camera_diagnostic)).body[0]
        expected = copy.deepcopy(original)
        matches = [node for node in ast.walk(expected) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "ego_camera"]
        self.assertEqual(len(matches), 1)
        matches[0].value = ast.parse('states["T_camera_base"]', mode="eval").body
        self.assertEqual(ast.dump(expected, include_attributes=False),
                         ast.dump(successor, include_attributes=False))

    def test_real_sealed_cards042_preflight_reads_q_without_ik_or_render(self):
        spec = sealed_spec()
        with patch.object(legacy, "solve_camera_diagnostic", side_effect=AssertionError("IK")), \
             patch.object(direction, "render_camera_diagnostic", side_effect=AssertionError("render")):
            states, intrinsics, fps = direction.preflight(spec)
        self.assertEqual((states["q_arm"].shape, states["q_hand"].shape),
                         ((171, 2, 7), (171, 2, 22)))
        self.assertEqual(intrinsics.shape, (171, 3, 3))
        self.assertEqual(fps, 30)
        self.assertTrue(np.isfinite(states["T_camera_base"]).all())
        self.assertFalse(OUTPUT.exists())

    def test_dry_run_has_new_attempt_write_set_and_no_render(self):
        with patch.object(direction, "render_camera_diagnostic", side_effect=AssertionError("render")):
            preview = direction.dry_run(sealed_spec())
        self.assertEqual(preview["status"], "DRY_RUN_READY_NOT_ROUTABLE")
        self.assertEqual(preview["frame_count"], 171)
        self.assertEqual(preview["gpu_count"], 0)
        self.assertEqual(preview["attempt_write_set_first"],
                         str(OUTPUT.parent.relative_to(REPO_ROOT)))
        self.assertFalse(preview["old_render_video_consumed"])
        self.assertFalse(OUTPUT.exists())

    def test_third_person_uses_same_base_to_camera_direction(self):
        states, _, _ = direction.preflight(sealed_spec())
        base_to_camera = states["T_camera_base"]
        external = direction._fixed_external_camera(base_to_camera)
        self.assertTrue(np.array_equal(external[:3, :3], base_to_camera[:3, :3]))
        self.assertFalse(np.allclose(external,
                                     direction._fixed_external_camera(np.linalg.inv(base_to_camera))))

    def test_input_sha_drift_and_wrong_output_scope_rejected(self):
        spec = sealed_spec()
        spec["inputs"]["states"] = dict(spec["inputs"]["states"], sha256="0" * 64)
        with self.assertRaisesRegex(direction.FreshRobotError, "RENDER_ONLY_INPUT_SHA_DRIFT:states"):
            direction.preflight(spec)
        spec = sealed_spec()
        spec["output_root"] = "/path/to/private_data/egodata/forbidden-render"
        with self.assertRaisesRegex(direction.FreshRobotError, "RENDER_ONLY_FRESH_PROJECT_OUTPUT_REQUIRED"):
            direction.preflight(spec)

    def test_state_and_stage_identity_rejected(self):
        spec = sealed_spec()
        spec["session_id"] = "wrong_session"
        with self.assertRaisesRegex(direction.FreshRobotError, "RENDER_ONLY_SEALED_STAGE_IDENTITY"):
            direction.preflight(spec)
        spec = sealed_spec()
        spec["inputs"]["hawor_npz"] = spec["inputs"]["states"]
        with self.assertRaisesRegex(direction.FreshRobotError, "RENDER_ONLY_STAGE_INPUT_DRIFT:hawor_npz"):
            direction.preflight(spec)

    def test_flange_and_legacy_code_pin_drift_rejected(self):
        spec = sealed_spec()
        wrong = dict(direction.PINNED_CODE_SHA)
        wrong["src/chaoyang/pipeline/robot_scene_state_cpu.py"] = "0" * 64
        with patch.object(direction, "PINNED_CODE_SHA", wrong):
            with self.assertRaisesRegex(direction.FreshRobotError, "RENDER_ONLY_CODE_SHA_DRIFT"):
                direction.preflight(spec)


if __name__ == "__main__":
    unittest.main()
