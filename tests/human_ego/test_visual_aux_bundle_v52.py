from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from chaoyang.human_ego.tools.validate_visual_aux_bundle_v52 import (
    ContractError,
    eligible_starts,
    validate_bundle,
)
from chaoyang.human_ego.tools.build_visual_retarget_projection_candidate_v52 import future_arrays
from chaoyang.human_ego.tools.build_visual_aux_session_bundle_v54 import (
    causal_real_donor_validity,
    ref as absolute_ref,
    scale_uv_to_training_domain,
    validate_silver_compositor_receipts,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


class VisualAuxBundleV52Test(unittest.TestCase):
    def build_occlusion_receipts(self, root: Path) -> tuple[Path, Path, dict, dict]:
        robot = root / "ROBOT_RESULT.json"
        clean = root / "CLEAN_RESULT.json"
        robot.write_text("{}\n", encoding="utf-8")
        clean.write_text("{}\n", encoding="utf-8")
        robot_ref = absolute_ref(robot)
        clean_ref = absolute_ref(clean)
        silver = root / "OCCLUSION_SILVER_RESULT.json"
        silver.write_text(
            json.dumps(
                {
                    "schema_version": "OCCLUSION_SILVER_R3",
                    "status": "VISUAL_OCCLUSION_SILVER_READY",
                    "authority_level": "SILVER",
                    "task": "chips",
                    "session": "fixture",
                    "frame_count": 60,
                    "accuracy_reported": False,
                    "external_accuracy": "UNKNOWN",
                    "known_decision_coverage": 0.75,
                    "unknown_pixel_ratio": 0.25,
                    "unknown_contact_frame_ratio": 0.10,
                    "max_unknown_run": 3,
                    "protected_retention": 0.995,
                    "pixel_provenance_coverage": 0.80,
                    "temporal_consistency_pass": True,
                    "zbuffer_consistency_pass": True,
                    "byte_exact_outside_authorized_band": True,
                    "causal_donor_pass": True,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        compositor = root / "COMPOSITOR_RESULT.json"
        compositor.write_text(
            json.dumps(
                {
                    "schema_version": "ROBOTIZED_COMPOSITOR_CAUSAL_V1_RESULT",
                    "status": "PASS_CAUSAL_TRAINING_COMPOSITOR",
                    "task": "chips",
                    "session": "fixture",
                    "frame_count": 60,
                    "input_mode": "CAUSAL_TRAINING_INPUT",
                    "training_authorized": True,
                    "pixel_source_gate_pass": True,
                    "control_ground_truth": False,
                    "occlusion_silver": absolute_ref(silver),
                    "inputs": {
                        "robot_result": robot_ref,
                        "clean_result": clean_ref,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return silver, compositor, robot_ref, clean_ref

    def test_silver_and_causal_compositor_must_be_exactly_bound(self) -> None:
        root = Path(tempfile.mkdtemp())
        silver, compositor, robot_ref, clean_ref = self.build_occlusion_receipts(root)
        receipts = validate_silver_compositor_receipts(
            silver_path=silver,
            compositor_path=compositor,
            task="chips",
            session="fixture",
            frame_count=60,
            robot_ref=robot_ref,
            clean_ref=clean_ref,
        )
        self.assertEqual(receipts["occlusion_silver_result"], absolute_ref(silver))
        self.assertEqual(receipts["compositor_result"], absolute_ref(compositor))

    def test_silver_gate_and_compositor_input_mismatch_fail_closed(self) -> None:
        root = Path(tempfile.mkdtemp())
        silver, compositor, robot_ref, clean_ref = self.build_occlusion_receipts(root)
        silver_payload = json.loads(silver.read_text(encoding="utf-8"))
        silver_payload["known_decision_coverage"] = 0.69
        silver.write_text(json.dumps(silver_payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "known_decision_coverage"):
            validate_silver_compositor_receipts(
                silver_path=silver,
                compositor_path=compositor,
                task="chips",
                session="fixture",
                frame_count=60,
                robot_ref=robot_ref,
                clean_ref=clean_ref,
            )

        silver, compositor, robot_ref, clean_ref = self.build_occlusion_receipts(root)
        compositor_payload = json.loads(compositor.read_text(encoding="utf-8"))
        compositor_payload["inputs"]["clean_result"] = robot_ref
        compositor.write_text(json.dumps(compositor_payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Clean result binding mismatch"):
            validate_silver_compositor_receipts(
                silver_path=silver,
                compositor_path=compositor,
                task="chips",
                session="fixture",
                frame_count=60,
                robot_ref=robot_ref,
                clean_ref=clean_ref,
            )

    def test_any_endpoint_mode_preserves_per_side_validity(self) -> None:
        uv = np.zeros((60, 2, 2), dtype=np.float32)
        point_valid = np.zeros((60, 2), dtype=bool)
        point_valid[:, 1] = True
        _, future, current, starts = future_arrays(
            uv, point_valid, "ANY_ENDPOINT_40_OF_50"
        )
        self.assertEqual(starts, list(range(15)))
        self.assertTrue(current.all())
        self.assertFalse(future[:, :, 0].any())
        self.assertTrue(future[0, :, 1].all())
        rgb = np.ones((60, 2, 2), dtype=bool)
        self.assertEqual(
            eligible_starts(current, future, rgb, "ANY_ENDPOINT_40_OF_50"),
            starts,
        )
        self.assertEqual(eligible_starts(current, future, rgb), [])

    def test_causal_donor_filter_rejects_future_and_unsupported_pixels(self) -> None:
        kind = np.asarray([[0, 1, 1, 2, 3]], dtype=np.uint8)
        frames = np.asarray([[0, 3, 8, 0, 0]], dtype=np.int32)
        valid, donor = causal_real_donor_validity(kind, frames, target_frame=4)
        np.testing.assert_array_equal(donor, [[False, True, False, False, False]])
        np.testing.assert_array_equal(valid, [[True, True, False, True, False]])

    def test_source_edge_maps_to_training_edge(self) -> None:
        uv = np.asarray([[[1279.0, 959.0], [0.0, 0.0]]], dtype=np.float32)
        mapped = scale_uv_to_training_domain(uv, 1280, 960)
        np.testing.assert_allclose(mapped[0, 0], (639.0, 479.0), atol=1e-6)
        np.testing.assert_allclose(mapped[0, 1], (0.0, 0.0), atol=1e-6)

    def build(self) -> Path:
        root = Path(tempfile.mkdtemp())
        t, h, w = 60, 4, 5
        future = np.zeros((t, 50, 2), dtype=bool)
        original = np.full((t, 50, 2, 2), np.nan, dtype=np.float32)
        for current in range(t):
            for offset in range(50):
                if current + offset + 1 < t:
                    future[current, offset] = True
                    original[current, offset, 0] = (1.0, 1.0)
                    original[current, offset, 1] = (3.0, 2.0)
        normalized = original.copy()
        normalized[..., 0] /= w - 1
        normalized[..., 1] /= h - 1
        current = np.ones(t, dtype=bool)
        rgb_mask = np.ones((t, h, w), dtype=bool)
        labels = root / "LABELS.npz"
        np.savez_compressed(
            labels,
            schema_version_utf8=np.frombuffer(b"exact78-visual-aux-labels-v52-v1", dtype=np.uint8),
            frame_ids=np.arange(t, dtype=np.int64),
            future_2d_xy_original=original,
            future_2d_xy_normalized=normalized,
            future_2d_valid=future,
            current_frame_valid=current,
            rgb_training_valid_mask=rgb_mask,
            label_source_utf8=np.frombuffer(b"VISUAL_RETARGET_PROJECTION", dtype=np.uint8),
            control_ground_truth=np.asarray(False, dtype=bool),
            image_width=np.asarray(w, dtype=np.int64),
            image_height=np.asarray(h, dtype=np.int64),
            normalization_utf8=np.frombuffer(b"x/(W-1),y/(H-1); range=[0,1]", dtype=np.uint8),
        )
        branches = {}
        for branch, value in (("HUMAN_RAW_RGB", 20), ("ROBOTIZED_RGB", 40)):
            image_dir = root / branch
            image_dir.mkdir()
            frames = []
            for index in range(t):
                image = np.full((h, w, 3), value + index % 10, dtype=np.uint8)
                path = image_dir / f"{index:05d}.png"
                assert cv2.imwrite(str(path), image)
                frames.append({"frame_id": index, "rgb": ref(root, path)})
            selector = root / f"{branch}_SELECTOR.json"
            selector.write_text(json.dumps({
                "schema_version": "exact78-visual-aux-rgb-selector-v1",
                "branch": branch, "session_id": "fixture", "frame_ids": list(range(t)), "frames": frames,
            }, sort_keys=True) + "\n")
            branches[branch] = {"selector": ref(root, selector)}
        manifest = {
            "schema_version": "exact78-visual-aux-session-bundle-v52-v1",
            "task": "chips", "session_id": "fixture", "split": "train", "seed": 7,
            "pred_horizon": 50, "frame_count": t, "image_width": w, "image_height": h,
            "label_source": "VISUAL_RETARGET_PROJECTION", "control_ground_truth": False,
            "only_branch_variable": "RGB_BYTES", "labels": ref(root, labels),
            "labels_sha_shared_by_branches": sha(labels), "eligible_h50_starts": list(range(15)),
            "branches": branches,
        }
        (root / "VISUAL_AUX_SESSION_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return root

    def test_valid_fixture(self) -> None:
        root = self.build()
        report = validate_bundle(root)
        self.assertEqual(report["eligible_h50_window_count"], 15)
        self.assertFalse(report["control_ground_truth"])

    def test_control_truth_rejected(self) -> None:
        root = self.build()
        p = root / "VISUAL_AUX_SESSION_MANIFEST.json"
        value = json.loads(p.read_text())
        value["control_ground_truth"] = True
        p.write_text(json.dumps(value) + "\n")
        with self.assertRaises(ContractError):
            validate_bundle(root)

    def test_branch_frame_drift_rejected(self) -> None:
        root = self.build()
        p = root / "ROBOTIZED_RGB_SELECTOR.json"
        value = json.loads(p.read_text())
        value["frames"][2]["frame_id"] = 3
        p.write_text(json.dumps(value) + "\n")
        manifest_path = root / "VISUAL_AUX_SESSION_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["branches"]["ROBOTIZED_RGB"]["selector"] = ref(root, p)
        manifest_path.write_text(json.dumps(manifest) + "\n")
        with self.assertRaises(ContractError):
            validate_bundle(root)

    def test_invalid_coordinate_must_be_nan(self) -> None:
        root = self.build()
        labels = root / "LABELS.npz"
        with np.load(labels, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        arrays["future_2d_xy_original"][59, 0, 0] = (1.0, 1.0)
        np.savez_compressed(labels, **arrays)
        manifest_path = root / "VISUAL_AUX_SESSION_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["labels"] = ref(root, labels)
        manifest["labels_sha_shared_by_branches"] = sha(labels)
        manifest_path.write_text(json.dumps(manifest) + "\n")
        with self.assertRaises(ContractError):
            validate_bundle(root)


if __name__ == "__main__":
    unittest.main()
