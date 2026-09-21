import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chaoyang.ops.run_pico_manus_provenance_correction_v2 import (
    checked, correct_arrays, correct_npz, numerical_digest, ref,
)


class ProvenanceCorrectionTests(unittest.TestCase):
    def fixture(self, kind="motion"):
        d = {"timestamp_ns": np.array([1, 7], dtype=np.int64),
             "values": np.array([0., -0., np.nan, np.inf]),
             "T_world_wrist": np.arange(64, dtype=np.float64).reshape(2, 2, 4, 4),
             "q22": np.arange(88, dtype=np.float64).reshape(2, 2, 22)}
        if kind == "motion":
            d.update(joint_observed_local=np.ones((2, 2, 21), dtype=bool),
                     manus_local_joint_valid=np.ones((2, 2, 21), dtype=bool),
                     controller_pose_observed=np.array([[True, False], [True, True]]),
                     joint_inferred=np.zeros((2, 2, 21), dtype=bool))
            d["manus_local_joint_valid"][1, 0, 3] = False
        else:
            d.update(source_observed_physical=np.ones((2, 2), dtype=bool),
                     finger_valid=np.array([[True, False], [True, True]]))
        return d

    def test_motion_downscope_preserves_all_other_bytes(self):
        d = self.fixture()
        out, evidence = correct_arrays(d, "motion")
        self.assertFalse(out["joint_observed_local"].any())
        self.assertTrue(d["joint_observed_local"].all())
        self.assertFalse(out["joint_inferred"].any())
        self.assertFalse(out["controller_pose_observed"].any())
        np.testing.assert_array_equal(out["provenance_v2_controller_pose_computable"], d["controller_pose_observed"])
        for k in set(d) - {"joint_observed_local", "controller_pose_observed"}:
            self.assertEqual(d[k].tobytes(), out[k].tobytes(), k)
        np.testing.assert_array_equal(out["provenance_v2_upstream_resampled"], d["manus_local_joint_valid"])
        self.assertTrue(evidence["bit_exact_preserved"])

    def test_robot_downscope_does_not_copy_side_or_change_q(self):
        d = self.fixture("robot_input")
        out, _ = correct_arrays(d, "robot_input")
        self.assertFalse(out["source_observed_physical"].any())
        np.testing.assert_array_equal(out["provenance_v2_direct_observation_unverified"], d["finger_valid"])
        self.assertEqual(out["q22"].tobytes(), d["q22"].tobytes())
        self.assertFalse(out["provenance_v2_exact_interpolation_support_known"].any())

    def test_wrong_schema_and_repeat_rejected(self):
        d = self.fixture()
        with self.assertRaises(ValueError): correct_arrays(d, "unsupported")
        d["joint_observed_local"] = d["joint_observed_local"].astype(int)
        with self.assertRaises(ValueError): correct_arrays(d, "motion")
        out, _ = correct_arrays(self.fixture(), "motion")
        with self.assertRaises(ValueError): correct_arrays(out, "motion")

    def test_signed_zero_mutation_detected_by_semantic_digest(self):
        d = self.fixture(); changed = {k: v.copy() for k, v in d.items()}
        changed["values"][1] = 0.
        self.assertNotEqual(numerical_digest(d, []), numerical_digest(changed, []))

    def test_bad_sha_escape_and_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p = root / "source.npz"
            with p.open("xb") as f: np.savez_compressed(f, **self.fixture())
            binding = ref(p); bad = dict(binding, sha256="0" * 64)
            with self.assertRaises(ValueError): checked(bad, [root])
            with self.assertRaises(ValueError): checked(binding, [root / "other"])
            out = correct_npz(binding, root / "corrected.npz", [root], "motion", 2)
            self.assertEqual(ref(p), binding)
            self.assertTrue(out["regression"]["bit_exact_preserved"])
            with self.assertRaises(FileExistsError): correct_npz(binding, root / "corrected.npz", [root], "motion", 2)
            with self.assertRaises(ValueError): correct_npz(binding, root / "bad.npz", [root], "motion", 3)


if __name__ == "__main__":
    unittest.main()
