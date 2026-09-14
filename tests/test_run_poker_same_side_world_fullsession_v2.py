from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "tools" / "run_poker_same_side_world_fullsession_v2.py"
spec = importlib.util.spec_from_file_location("poker_fullsession_v2", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_v2_does_not_change_published_thresholds_or_temporal_limits() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert module.ACCEPTED_V3_FRAME_COUNT == 48
    assert module.EXPECTED_V1_FAILED_GATES == {
        "arm_world_position_error_le_10mm",
        "hand_tip_direction_error_le_15deg",
        "hand_bone_error_le_60deg",
        "upper_arms_elbows_forearms_keep_accepted_branches",
    }
    assert 'position_mm - 10.0' in source
    assert 'rotation_deg - 5.0' in source
    assert 'bone - 60.0' in source
    assert 'tip - 15.0' in source
    assert "temporal.HAND_SOLVER_STEP_LIMIT" in source
    assert "temporal.ARM_SOLVER_STEP_LIMIT" in source


def test_branch_margins_preserve_original_left_and_right_gate_semantics() -> None:
    left = module.branch_margins_from_points(
        0, np.asarray((0.0, 0.21, 0.0)), np.asarray((0.041, 0.05, 0.0))
    )
    right = module.branch_margins_from_points(
        1, np.asarray((0.0, -0.21, 0.0)), np.asarray((0.041, -0.05, 0.0))
    )
    np.testing.assert_allclose(left, (0.01, 0.01, 0.001), atol=1e-12)
    np.testing.assert_allclose(right, (0.01, 0.01, 0.001), atol=1e-12)
    assert module.branch_margins_from_points(
        1, np.asarray((0.0, -0.21, 0.0)), np.asarray((0.039, -0.05, 0.0))
    )[2] < 0.0


def test_nonthumb_semantics_require_four_direct_physical_edges() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    function = source[
        source.index("def robot_finger_features_chain_semantic") :
        source.index("def _feature_score")
    ]
    assert "unit(np.diff(points, axis=0))" in function
    assert 'if finger == "thumb"' in function
    assert "resample_polyline_unit_directions" in function
    nonthumb = function.split("else:", 1)[1]
    assert "resample_polyline_unit_directions" not in nonthumb


def test_hybrid_audit_keeps_both_raw_errors_and_takes_per_bone_envelope() -> None:
    class FakeHandfit:
        @staticmethod
        def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            dot = np.sum(a * b, axis=-1)
            return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    bones = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    target = {"mcp": np.zeros(3), "tip": np.asarray((1.0, 0.0, 0.0)), "bones": bones}
    direct = {"mcp": np.zeros(3), "tip": target["tip"], "bones": target["bones"].copy()}
    legacy = {"mcp": np.zeros(3), "tip": target["tip"], "bones": -target["bones"]}
    row = module.hybrid_feature_row(FakeHandfit, direct, legacy, target)
    assert row["bone_error_deg_max"] == 0.0
    assert row["bone_error_deg_direct_max"] == 0.0
    assert row["bone_error_deg_legacy_max"] == 180.0
    assert row["bone_audit_policy"] == "PER_BONE_MIN_LEGACY_ARCLENGTH_OR_DIRECT_PHYSICAL_CHAIN"


def test_execution_uses_fresh_working_directory_and_atomic_promotion() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    execution = source[source.index("def execute(") : source.index("def main()")]
    assert "if output_dir.exists() or output_dir.is_symlink()" in execution
    assert 'f".{output_dir.name}.working-{os.getpid()}"' in execution
    assert "working.replace(output_dir)" in execution
    assert "failed_v1_result" in execution


def test_v3_trajectory_is_locked_as_a_prefix_not_only_frame_zero() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert source.count("q_rows[:ACCEPTED_V3_FRAME_COUNT] = accepted_short_q") == 2
    assert source.count("ACCEPTED_V3_FRAME_0_47_BIT_EXACT_LOCK") == 2
