from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from HumanEgo.tools.validate_visual_aux_bundle_v52 import ContractError, validate_bundle


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


class VisualAuxBundleV52Test(unittest.TestCase):
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
