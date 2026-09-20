#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "d2", HERE / "run_d2_fresh_propainter_offline_v1.py"
)
assert SPEC and SPEC.loader
d2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d2)
PUB_SPEC = importlib.util.spec_from_file_location(
    "d2pub", HERE / "publish_d2_shallow_visuals_v1.py"
)
assert PUB_SPEC and PUB_SPEC.loader
d2pub = importlib.util.module_from_spec(PUB_SPEC)
PUB_SPEC.loader.exec_module(d2pub)


def frozen_config_path() -> Path:
    adjacent = HERE / "D2_FRESH_PROPAINTER_OFFLINE_V1.json"
    if adjacent.is_file():
        return adjacent
    # Deployed layout: <repo>/src/chaoyang/ops/test_*.py with the canonical
    # immutable config under <repo>/contracts/robot_recovery/.
    repository = HERE.parents[2]
    canonical = repository / "contracts/robot_recovery/D2_FRESH_PROPAINTER_OFFLINE_V1.json"
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(
        f"D2 frozen config missing from adjacent={adjacent} and canonical={canonical}"
    )


class D2ContractTest(unittest.TestCase):
    def fixture(self):
        raw = np.zeros((8, 10, 3), np.uint8)
        raw[..., 1] = 10
        generated = np.full_like(raw, 220)
        m_write = np.zeros((8, 10), bool)
        m_write[2:6, 3:7] = True
        m_remove = m_write.copy()
        m_flow = m_write.copy()
        m_flow[1:7, 2:8] = True
        unknown = m_write.copy()
        source = np.zeros((8, 10), np.uint8)
        source[0, 0] = 10
        return raw, generated, m_remove, m_write, m_flow, unknown, source

    def test_only_m_write_changes_and_unknown_is_retained(self):
        values = self.fixture()
        raw, generated, _m_remove, m_write, _m_flow, _unknown, _source = values
        candidate, source, unknown, stats = d2.composite_visual_candidate(*values)
        self.assertTrue(np.array_equal(candidate[~m_write], raw[~m_write]))
        self.assertTrue(np.array_equal(candidate[m_write], generated[m_write]))
        self.assertTrue(np.array_equal(unknown, m_write))
        self.assertTrue(np.all(source[m_write] == 3))
        self.assertEqual(stats["changed_outside_m_write_pixels"], 0)
        self.assertEqual(stats["changed_protected_pixels"], 0)

    def test_visible_instance_is_byte_exact(self):
        values = self.fixture()
        raw = values[0]
        candidate, source, _unknown, _stats = d2.composite_visual_candidate(*values)
        self.assertTrue(np.array_equal(candidate[0, 0], raw[0, 0]))
        self.assertEqual(int(source[0, 0]), 10)

    def test_rejects_write_outside_flow(self):
        values = list(self.fixture())
        values[4] = np.zeros_like(values[4])
        with self.assertRaisesRegex(d2.ContractError, "subset of M_flow"):
            d2.composite_visual_candidate(*values)

    def test_rejects_object_write_overlap(self):
        values = list(self.fixture())
        values[6][3, 4] = 11
        with self.assertRaisesRegex(d2.ContractError, "protected object overlaps"):
            d2.composite_visual_candidate(*values)

    def test_rejects_unknown_not_equal_write(self):
        values = list(self.fixture())
        values[5] = np.zeros_like(values[5])
        with self.assertRaisesRegex(d2.ContractError, "UNKNOWN must equal M_write"):
            d2.composite_visual_candidate(*values)

    def test_frozen_policy_has_no_sweep_or_fallback(self):
        config = json.loads(frozen_config_path().read_text())
        d2.validate_policy(config)
        self.assertFalse(config["parameter_sweep_allowed"])
        self.assertFalse(config["model_fallback_allowed"])
        self.assertFalse(config["training_eligible"])
        self.assertEqual(config["execution_semantics"], "OFFLINE_BIDIRECTIONAL_VISUAL_ONLY")
        self.assertEqual(config["write_contract"]["input_mask_domain"], "M_flow")
        self.assertEqual(
            config["write_contract"]["effective_internal_model_mask"],
            "DILATE(RESIZED_M_flow,4)_INSIDE_PROPAINTER",
        )
        self.assertEqual(config["write_contract"]["publish_write_domain"], "M_write")

    def test_no_forbidden_input_key_in_config(self):
        text = frozen_config_path().read_text()
        for forbidden in ("donor_manifest", "object6d_input", "contact_input", "robot_input", "old_clean_path"):
            self.assertNotIn(forbidden, text.lower())

    def test_projected_ref_hashes_staging_but_publishes_final_path(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            staging = parent / ".result.staging.fixture"
            final = parent / "result"
            artifact = staging / "nested" / "file.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"payload")
            value = d2.projected_ref(artifact, staging, final)
            self.assertEqual(value["path"], str(final / "nested" / "file.bin"))
            self.assertNotIn(".staging.", value["path"])
            self.assertEqual(value["sha256"], d2.sha256(artifact))

    def test_atomic_publish_success_has_no_partial_or_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            final = parent / "published"

            def build(staging, projected_final):
                artifact = staging / "artifact.txt"
                artifact.write_text("complete", encoding="utf-8")
                receipt = {"artifact": d2.projected_ref(artifact, staging, projected_final)}
                d2.write_json(staging / "RESULT.json", receipt)
                self.assertFalse(final.exists())
                return receipt

            with mock.patch.object(d2.os, "replace", wraps=d2.os.replace) as replace:
                result = d2.publish_directory_atomically(final, build)
            replace.assert_called_once()
            self.assertTrue((final / "artifact.txt").is_file())
            self.assertEqual(result["artifact"]["path"], str(final / "artifact.txt"))
            self.assertEqual(list(parent.glob(".published.staging.*")), [])

    def test_atomic_publish_failure_cleans_staging_and_leaves_no_final(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            final = parent / "published"

            def fail(staging, _projected_final):
                (staging / "partial.bin").write_bytes(b"partial")
                raise RuntimeError("injected failure")

            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                d2.publish_directory_atomically(final, fail)
            self.assertFalse(final.exists())
            self.assertEqual(list(parent.glob(".published.staging.*")), [])

    def test_atomic_publish_rejects_staging_path_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            final = parent / "published"

            def leak(staging, _projected_final):
                d2.write_json(staging / "RESULT.json", {"bad": str(staging / "x")})
                return {}

            with self.assertRaisesRegex(d2.ContractError, "staging path leaked"):
                d2.publish_directory_atomically(final, leak)
            self.assertFalse(final.exists())
            self.assertEqual(list(parent.glob(".published.staging.*")), [])

    def test_shallow_publisher_uses_real_copies_final_refs_and_atomic_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deep = root / "deep"
            deep.mkdir()
            session_refs = []
            for sid, frame_count in d2pub.EXPECTED.items():
                video = deep / f"{sid}.mp4"
                video.write_bytes((sid + "-video").encode())
                session_path = deep / f"{sid}.json"
                d2.write_json(session_path, {
                    "session_id": sid,
                    "frame_count": frame_count,
                    "review_video": d2.file_ref(video),
                })
                session_refs.append(d2.file_ref(session_path))
            deep_result = deep / "RESULT.json"
            d2.write_json(deep_result, {
                "status": "COMPLETED_OFFLINE_VISUAL_CANDIDATE_UNKNOWN_RETAINED",
                "training_eligible": False,
                "clean_terminal": False,
                "sessions": session_refs,
            })
            gpu_receipt = root / "GPU_RECEIPT.json"
            d2.write_json(gpu_receipt, {
                "schema_version": "v71-gpu-command-receipt-v1",
                "status": "PASSED",
                "task_id": d2pub.TASK_ID,
                "attempt_id": d2pub.ATTEMPT_ID,
                "command": [
                    "python", "-m", "chaoyang.cli", "run",
                    "run_d2_fresh_propainter_offline_v1",
                    "--config", str(d2pub.CONFIG_PATH),
                    "--output-root", str(deep_result.parent),
                ],
            })
            shallow = root / "visuals"
            with mock.patch.object(
                d2pub, "decode_count",
                side_effect=lambda p: d2pub.EXPECTED[p.name.split("_D2_")[0]],
            ):
                result = d2pub.publish(deep_result, gpu_receipt, shallow)
            self.assertTrue(shallow.is_dir())
            self.assertFalse(shallow.is_symlink())
            self.assertEqual(list(root.glob(".visuals.staging.*")), [])
            for row in result["videos"]:
                path = Path(row["video"]["path"])
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.parent, shallow)
                self.assertNotIn(".staging.", str(path))
            self.assertEqual(
                result["gpu_lease_receipt"]["sha256"], d2.sha256(gpu_receipt)
            )

    def test_generated_frames_require_exact_six_digit_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(4):
                (root / f"{index:06d}.png").write_bytes(bytes([index]))
            paths = d2.require_exact_png_sequence(root, expected=4, digits=6)
            self.assertEqual([p.name for p in paths], [f"{i:06d}.png" for i in range(4)])
            (root / "000002.png").unlink()
            (root / "0002.png").write_bytes(b"wrong-width")
            with self.assertRaisesRegex(d2.ContractError, "filename sequence mismatch"):
                d2.require_exact_png_sequence(root, expected=4, digits=6)

    def test_vendor_four_digit_sequence_normalizes_byte_exactly_to_six_digits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            upstream = root / "model" / "frames" / "frames"
            upstream.mkdir(parents=True)
            for index in range(3):
                (upstream / f"{index:04d}.png").write_bytes(f"frame-{index}".encode())
            normalized = d2.normalize_generated_frames(root / "model", root / "normalized", 3)
            self.assertEqual([p.name for p in normalized], [f"{i:06d}.png" for i in range(3)])
            for index, path in enumerate(normalized):
                self.assertEqual(path.read_bytes(), f"frame-{index}".encode())

    def test_weight_provenance_accepts_exact_symlink_layout_and_pins_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vendor = root / "vendor" / "ProPainter"
            assets = root / "assets" / "models" / "vendor" / "propainter"
            vendor.mkdir(parents=True)
            assets.mkdir(parents=True)
            (vendor / "weights").symlink_to(assets, target_is_directory=True)
            target = assets / "ProPainter.pth"
            target.write_bytes(b"pinned-weight")
            lexical = vendor / "weights" / "ProPainter.pth"
            value = {
                "lexical_path": str(lexical),
                "resolved_path": str(target),
                "bytes": target.stat().st_size,
                "sha256": d2.sha256(target),
            }
            checked = d2.verify_lexical_resolved_ref(
                value, lexical, "fixture weight"
            )
            self.assertEqual(checked["lexical_path"], str(lexical))
            self.assertEqual(checked["resolved_path"], str(target))
            self.assertTrue(checked["lexical_parent_is_symlink"])
            wrong = dict(value, resolved_path=str(root / "wrong.pth"))
            with self.assertRaisesRegex(d2.ContractError, "resolved target mismatch"):
                d2.verify_lexical_resolved_ref(wrong, lexical, "fixture weight")

    def test_shallow_publisher_rejects_bad_gpu_receipt_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deep = root / "RESULT.json"
            d2.write_json(deep, {
                "status": "COMPLETED_OFFLINE_VISUAL_CANDIDATE_UNKNOWN_RETAINED",
                "training_eligible": False,
                "clean_terminal": False,
                "sessions": [],
            })
            cases = [
                ("FAILED_RUNTIME_FINAL", d2pub.TASK_ID, d2pub.ATTEMPT_ID, deep.parent),
                ("PASSED", "wrong_task", d2pub.ATTEMPT_ID, deep.parent),
                ("PASSED", d2pub.TASK_ID, "wrong_attempt", deep.parent),
                ("PASSED", d2pub.TASK_ID, d2pub.ATTEMPT_ID, root / "wrong_deep"),
            ]
            for index, (status, task, attempt, command_root) in enumerate(cases):
                with self.subTest(index=index):
                    receipt = root / f"receipt_{index}.json"
                    d2.write_json(receipt, {
                        "schema_version": "v71-gpu-command-receipt-v1",
                        "status": status,
                        "task_id": task,
                        "attempt_id": attempt,
                        "command": [
                            "python", "-m", "chaoyang.cli", "run",
                            "run_d2_fresh_propainter_offline_v1",
                            "--config", str(d2pub.CONFIG_PATH),
                            "--output-root", str(command_root),
                        ],
                    })
                    with self.assertRaises(d2pub.ContractError):
                        d2pub.validate_gpu_receipt(receipt, deep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
