from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import run_reverse_right_depth_zbuffer_clean_successor as reverse  # noqa: E402
from tools import launch_reverse_right_depth_zbuffer_clean_once as launcher  # noqa: E402
from tools import project_jsonschema  # noqa: E402
from tools.run_tianji_kai_robot_baseline import _validate_clean_source_manifest  # noqa: E402


class ReverseRightDepthGeometryTests(unittest.TestCase):
    def test_reverse_model_input_is_flip_right_then_flip_left(self) -> None:
        left = np.zeros((reverse.DEPTH_HEIGHT, reverse.DEPTH_WIDTH, 3), np.uint8)
        right = np.zeros_like(left)
        left[:, :, 0] = np.arange(reverse.DEPTH_WIDTH, dtype=np.uint16) % 251
        right[:, :, 1] = np.arange(reverse.DEPTH_WIDTH, dtype=np.uint16) % 239
        first, second = reverse.reverse_inference_images(left, right)
        np.testing.assert_array_equal(first, right[:, ::-1])
        np.testing.assert_array_equal(second, left[:, ::-1])

    def test_right_depth_metric_and_boundary_are_right_view(self) -> None:
        disparity = np.full((reverse.DEPTH_HEIGHT, reverse.DEPTH_WIDTH), 32.0, np.float32)
        depth, valid = reverse.right_depth_from_disparity(disparity, 0.0637716504026918)
        self.assertAlmostEqual(float(depth[100, 607]), 320.0 * 0.0637716504026918 / 32.0, places=6)
        self.assertTrue(bool(valid[100, 607]))
        self.assertFalse(bool(valid[100, 608]))
        self.assertTrue(np.isnan(depth[100, 608]))

    def test_continuity_gate_rejects_depth_and_colour_edges(self) -> None:
        depth = np.ones((reverse.DEPTH_HEIGHT, reverse.DEPTH_WIDTH), np.float32)
        rgb = np.zeros((reverse.DEPTH_HEIGHT, reverse.DEPTH_WIDTH, 3), np.uint8)
        base = np.ones(depth.shape, bool)
        depth[:, 320:] = 2.0
        rgb[240:, :] = 255
        depth_only = reverse.continuous_depth_mask(depth, base)
        surface = reverse.continuous_surface_mask(depth, rgb, base)
        self.assertTrue(bool(depth_only[200, 200]))
        self.assertFalse(bool(depth_only[200, 319]))
        self.assertFalse(bool(depth_only[200, 320]))
        self.assertFalse(bool(surface[239, 200]))
        self.assertFalse(bool(surface[240, 200]))

    def test_nearest_splat_is_z_buffered_and_deterministic(self) -> None:
        cx = np.asarray([2, 2], np.int64)
        cy = np.asarray([2, 2], np.int64)
        z = np.asarray([1.0, 0.5], np.float64)
        source = np.asarray([7, 3], np.int64)
        tx, ty, chosen, tz, _ = reverse.nearest_splat(cx, cy, z, source, width=5, height=5)
        self.assertEqual(len(tx), 9)
        self.assertTrue(np.all(chosen == 3))
        self.assertTrue(np.allclose(tz, 0.5))
        self.assertEqual(set(zip(tx.tolist(), ty.tolist())), {(x, y) for y in range(1, 4) for x in range(1, 4)})

    def test_dynamic_occupancy_projects_left_to_right_and_dilates_five(self) -> None:
        disparity = np.zeros((reverse.DEPTH_HEIGHT, reverse.DEPTH_WIDTH), np.float32)
        disparity[20, 30] = 5.0
        valid = np.zeros(disparity.shape, bool)
        valid[20, 30] = True
        dynamic = np.zeros(disparity.shape, bool)
        dynamic[20, 30] = True
        occupancy = reverse.project_dynamic_occupancy_to_right(disparity, valid, dynamic)
        self.assertTrue(bool(occupancy[20, 25]))
        self.assertTrue(bool(occupancy[15, 20]))
        self.assertTrue(bool(occupancy[25, 30]))
        self.assertEqual(int(occupancy.sum()), 121)


class ExactPixelContractTests(unittest.TestCase):
    def _valid_composite(self):
        frame_id = 7
        raw = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        donor = np.flip(raw, axis=1).copy()
        right = (np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3) + 80).astype(np.uint8)
        removal = np.zeros((3, 4), bool)
        protected = np.zeros((3, 4), bool)
        composed = reverse.initial_composed(raw, removal, protected, frame_id)
        composed["source_kind"][0, 0] = reverse.adaptive.SOURCE_DONOR
        composed["source_frame"][0, 0] = 2
        composed["source_x"][0, 0] = 3
        composed["source_y"][0, 0] = 2
        composed["clean"][0, 0] = donor[2, 3]
        composed["source_kind"][1, 1] = reverse.SOURCE_RIGHT_RAW
        composed["source_frame"][1, 1] = frame_id
        composed["source_x"][1, 1] = 4
        composed["source_y"][1, 1] = 3
        composed["clean"][1, 1] = right[3, 4]
        return frame_id, raw, donor, right, composed

    def test_both_legal_donor_types_verify_byte_exact(self) -> None:
        frame_id, raw, donor, right, composed = self._valid_composite()
        audit = reverse.exact_provenance(raw, composed["clean"], composed, {2: donor}, right, frame_id)
        self.assertTrue(audit["verified"])
        self.assertEqual(audit["mismatch_pixels"], 0)

    def test_adaptive_minus_one_immutable_provenance_is_normalized_to_current_left_identity(self) -> None:
        frame_id = 7
        raw = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        removal = np.zeros((3, 4), bool)
        removal[1, 1] = True
        protected = np.zeros((3, 4), bool)
        protected[2, 2] = True
        composed = reverse.initial_composed(raw, removal, protected, frame_id)
        composed["source_frame"].fill(-1)
        composed["source_x"].fill(-1)
        composed["source_y"].fill(-1)
        composed["source_kind"][0, 0] = reverse.adaptive.SOURCE_DONOR
        reverse.normalize_immutable_provenance(raw, composed, frame_id)
        immutable = composed["source_kind"] != reverse.adaptive.SOURCE_DONOR
        yy, xx = np.indices(raw.shape[:2], dtype=np.int32)
        self.assertTrue(np.all(composed["source_frame"][immutable] == frame_id))
        np.testing.assert_array_equal(composed["source_x"][immutable], xx[immutable])
        np.testing.assert_array_equal(composed["source_y"][immutable], yy[immutable])
        self.assertEqual(int(composed["source_frame"][0, 0]), -1)

    def test_wrong_right_frame_and_wrong_immutable_identity_fail(self) -> None:
        frame_id, raw, donor, right, composed = self._valid_composite()
        composed["source_frame"][1, 1] = frame_id - 1
        composed["source_frame"][2, 2] = frame_id - 1
        audit = reverse.exact_provenance(raw, composed["clean"], composed, {2: donor}, right, frame_id)
        self.assertFalse(audit["verified"])
        self.assertGreaterEqual(audit["mismatch_pixels"], 2)

    def test_missing_or_out_of_bounds_donor_fails_without_indexing_error(self) -> None:
        frame_id, raw, donor, right, composed = self._valid_composite()
        composed["source_x"][0, 0] = 999
        audit = reverse.exact_provenance(raw, composed["clean"], composed, {2: donor}, right, frame_id)
        self.assertFalse(audit["verified"])
        self.assertGreater(audit["mismatch_pixels"], 0)

    def test_saved_source_eye_codes_follow_robot_contract(self) -> None:
        frame_id, raw, _, _, composed = self._valid_composite()
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp) / "staging"
            final = Path(temp) / "final"
            (staging / "clean_frames").mkdir(parents=True)
            (staging / "pixel_sources").mkdir(parents=True)
            reverse.save_frame(staging, final, frame_id, composed)
            with np.load(staging / "pixel_sources" / f"{frame_id:06d}.npz", allow_pickle=False) as bundle:
                self.assertEqual(bundle["source_eye"].dtype, np.uint8)
                self.assertEqual(int(bundle["source_eye"][0, 0]), 0)
                self.assertEqual(int(bundle["source_eye"][1, 1]), 1)
                self.assertEqual(int(bundle["source_eye"][2, 2]), 0)

    def test_source_manifest_declares_dual_eye_lineage_and_decoded_hashes(self) -> None:
        ref = {"path": "/ordinary", "bytes": 1, "sha256": "0" * 64}
        row = {
            "frame_id": 0,
            "selected_left_raw_decoded_sha256": "1" * 64,
            "right_raw_decoded_sha256": "2" * 64,
            "synchronized_right_raw_decoded_sha256": "2" * 64,
        }
        manifest = reverse.source_manifest_payload(
            task="chips",
            session="get_potato_chips_0902_034",
            frame_count=1,
            selected_left_raw_video=ref,
            synchronized_right_stereo_video=ref,
            frames=[row],
        )
        self.assertEqual(manifest["source_lineage"]["selected_left_raw_video"], ref)
        self.assertEqual(manifest["source_kind_codes"]["1"], "SAME_SESSION_LEFT_TEMPORAL_RAW")
        self.assertEqual(manifest["source_kind_codes"]["4"], "SAME_SESSION_SYNCHRONIZED_RIGHT_RAW")
        self.assertEqual(manifest["frames"][0]["selected_left_raw_decoded_sha256"], "1" * 64)
        self.assertEqual(manifest["frames"][0]["right_raw_decoded_sha256"], "2" * 64)
        self.assertEqual(manifest["frames"][0]["synchronized_right_raw_decoded_sha256"], "2" * 64)

    def test_manifest_is_accepted_by_robot_with_fullres_right_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_video = root / "left.mp4"
            stereo_video = root / "stereo.mp4"
            raw_video.write_bytes(b"left")
            stereo_video.write_bytes(b"right")
            yy, xx = np.indices((3, 4), dtype=np.int32)
            source_kind = np.zeros((3, 4), np.uint8)
            source_kind[1, 1] = reverse.SOURCE_RIGHT_RAW
            source_frame = np.zeros((3, 4), np.int32)
            source_x = xx.copy()
            source_y = yy.copy()
            source_x[1, 1] = 1800
            source_y[1, 1] = 1200
            source_map = root / "000000.npz"
            np.savez_compressed(
                source_map,
                frame_id=np.int32(0),
                source_kind=source_kind,
                source_eye=(source_kind == reverse.SOURCE_RIGHT_RAW).astype(np.uint8),
                source_frame=source_frame,
                source_x=source_x,
                source_y=source_y,
            )
            row = {
                "frame_id": 0,
                "pixel_source_map": reverse.artifact(source_map),
                "selected_left_raw_decoded_sha256": "1" * 64,
                "right_raw_decoded_sha256": "2" * 64,
                "synchronized_right_raw_decoded_sha256": "2" * 64,
            }
            manifest = reverse.source_manifest_payload(
                task="chips",
                session="get_potato_chips_0902_034",
                frame_count=1,
                selected_left_raw_video=reverse.artifact(raw_video),
                synchronized_right_stereo_video=reverse.artifact(stereo_video),
                frames=[row],
            )
            manifest_path = root / "SOURCE_MAP_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            observed = _validate_clean_source_manifest(
                manifest_path,
                task="chips",
                session_id="get_potato_chips_0902_034",
                frame_count=1,
                width=4,
                height=3,
                raw_video_ref=reverse.artifact(raw_video),
            )
            self.assertEqual(observed["synchronized_right_stereo_video"], reverse.artifact(stereo_video))


class BoundedExecutionContractTests(unittest.TestCase):
    def test_canary_requires_actual_reverse_candidates_order_domain_and_fill(self) -> None:
        metrics = {
            "left_to_right_to_left_within_2px_fraction": 1.0,
            "right_to_left_to_right_within_2px_fraction": 1.0,
            "stable_depth_consistency_fraction": 1.0,
            "provenance_mismatch_pixels": 0,
            "changed_outside_removal_pixels": 0,
            "changed_protected_object_pixels": 0,
            "right_projected_target_pixels": 0,
            "background_order_candidate_pixels": 0,
            "right_supported_pixels": 0,
        }
        gates = reverse.canary_hard_gates(metrics, 0.8, 0.8, 3)
        self.assertEqual(gates["reverse_right_projected_candidates_nonzero"], "FAIL")
        self.assertEqual(gates["background_order_fixed_denominator_nonzero"], "FAIL")
        self.assertEqual(gates["reverse_right_actual_fill_nonzero"], "FAIL")
        for key in ("right_projected_target_pixels", "background_order_candidate_pixels", "right_supported_pixels"):
            metrics[key] = 1
        self.assertTrue(all(value == "PASS" for value in reverse.canary_hard_gates(metrics, 0.8, 0.8, 3).values()))

    def test_each_session_fails_or_advances_independently(self) -> None:
        actions = reverse.independent_session_actions({
            "chips": {"status": "PASS_CANARY"},
            "poker": {"status": "TERMINAL_GRADE_C_CANARY_HARD_GATE_FAILURE"},
        })
        self.assertEqual(actions, {"chips": "FULL_ONCE", "poker": "TERMINAL_C_NO_FULL"})

    def test_canary_right_depth_artifact_is_reused_no_clobber(self) -> None:
        disparity = np.full((3, 4), 2.0, np.float32)
        depth = np.full((3, 4), 1.0, np.float32)
        valid = np.ones((3, 4), bool)
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp) / "staging"
            final = Path(temp) / "final"
            first = reverse.write_right_depth(staging, final, 5, disparity, depth, valid)
            second = reverse.write_right_depth(staging, final, 5, disparity.copy(), depth.copy(), valid.copy())
            self.assertEqual(first, second)
            different = disparity.copy()
            different[0, 0] = 3.0
            with self.assertRaises(reverse.ReverseCleanError):
                reverse.write_right_depth(staging, final, 5, different, depth, valid)


class LauncherContractTests(unittest.TestCase):
    def test_cuda_runtime_initializes_single_visible_device_before_peak_reset(self) -> None:
        calls: list[tuple[str, int | None]] = []
        cuda = mock.Mock()
        cuda.is_available.side_effect = lambda: calls.append(("is_available", None)) or True
        cuda.device_count.side_effect = lambda: calls.append(("device_count", None)) or 1
        cuda.set_device.side_effect = lambda device: calls.append(("set_device", device))
        cuda.init.side_effect = lambda: calls.append(("init", None))
        cuda.reset_peak_memory_stats.side_effect = lambda device: calls.append(("reset", device))
        reverse.initialize_cuda_runtime(mock.Mock(cuda=cuda))
        self.assertEqual(
            calls,
            [
                ("is_available", None),
                ("device_count", None),
                ("set_device", 0),
                ("init", None),
                ("reset", 0),
            ],
        )

    def test_environment_is_project_local_offline_and_gpu_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
            os.environ,
            {"PYTHONPATH": "/external", "CONDA_PREFIX": "/external/conda", "CUDA_VISIBLE_DEVICES": "7"},
        ):
            environment = launcher.clean_environment(Path(temp))
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("CONDA_PREFIX", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertTrue(environment["PATH"].startswith(str(launcher.LOCAL_PYTHON.resolve().parents[1] / "bin")))

    def test_worker_exception_releases_lease_and_writes_terminal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = root / "SPEC.json"
            spec.write_text(json.dumps({
                "output_root": str(root / "output"),
                "quality_gates": {"minimum_free_vram_mib": 61440},
            }), encoding="utf-8")
            receipt = root / "RECEIPT.json"
            preflight = subprocess.CompletedProcess(["preflight"], 0, "pass", "")
            gpu = {"free_mib": 96000}
            with (
                mock.patch.object(launcher, "run_checked", side_effect=[preflight, preflight, RuntimeError("worker failed")]),
                mock.patch.object(launcher, "gpu_state", return_value=gpu),
                mock.patch.object(launcher, "acquire", return_value={"status": "ACQUIRED"}),
                mock.patch.object(launcher, "release", return_value={"status": "RELEASED"}) as release,
            ):
                payload = launcher.execute(spec, "holder", receipt, 60)
            release.assert_called_once_with("holder", "WORKER_EXCEPTION_RuntimeError")
            self.assertEqual(payload["status"], "TERMINAL_WORKER_FAILURE_GPU_RELEASED")
            self.assertEqual(payload["worker_error"]["type"], "RuntimeError")
            self.assertTrue(receipt.is_file())


class ProjectSchemaFallbackTests(unittest.TestCase):
    def test_actual_frozen_spec_validates_and_unknown_keyword_fails_closed(self) -> None:
        schema = reverse.load_json(reverse.SCHEMA)
        v2_spec = PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/clean_fresh_baseline_v1/reverse_right_depth_zbuffer_successor_v2/RUN_SPEC.json"
        spec = reverse.load_json(v2_spec)
        project_jsonschema.Draft202012Validator(schema).validate(spec)
        invalid_schema = dict(schema)
        invalid_schema["unknownKeywordMustFail"] = True
        with self.assertRaises(project_jsonschema.ValidationError):
            project_jsonschema.Draft202012Validator(invalid_schema).validate(spec)


if __name__ == "__main__":
    unittest.main()
