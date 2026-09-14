from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.preflight_robot_chain_qarm_base import (
    PreflightError,
    build_preflight,
    implied_tool_targets,
    read_ordinary_file,
    validate_joint_limits,
    validate_session_constant_base,
)


def test_session_constant_base_accepts_single_or_repeated_transform() -> None:
    base = np.eye(4)
    base[:3, 3] = [0.2, -0.1, 0.8]
    validate_session_constant_base(base)
    validate_session_constant_base(np.repeat(base[None], 4, axis=0))


def test_session_constant_base_rejects_per_frame_motion() -> None:
    bases = np.repeat(np.eye(4)[None], 3, axis=0)
    bases[2, 0, 3] = 1e-3
    with pytest.raises(PreflightError, match="per-frame"):
        validate_session_constant_base(bases)


def test_session_constant_base_rejects_reflection() -> None:
    base = np.eye(4)
    base[0, 0] = -1
    with pytest.raises(PreflightError, match="determinant"):
        validate_session_constant_base(base)


def test_joint_limits_fail_closed() -> None:
    validate_joint_limits(np.array([[0.0, 0.5]]), np.array([-1.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(PreflightError, match="violates"):
        validate_joint_limits(np.array([[0.0, 1.01]]), np.array([-1.0, 0.0]), np.array([1.0, 1.0]))


def test_mount_is_not_identifiable_from_hand_targets_alone() -> None:
    hands = np.repeat(np.eye(4)[None], 2, axis=0)
    hands[1, :3, 3] = [0.1, 0.2, 0.5]
    mount_a = np.eye(4)
    mount_b = np.eye(4)
    mount_b[2, 3] = 0.05
    tools_a = implied_tool_targets(hands, mount_a)
    tools_b = implied_tool_targets(hands, mount_b)
    assert not np.allclose(tools_a, tools_b)
    assert np.allclose(tools_a @ mount_a, hands)
    assert np.allclose(tools_b @ mount_b, hands)


def test_secure_reader_rejects_symlink(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"fixed")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(PreflightError, match="securely open"):
        read_ordinary_file(link)


def test_actual_project_preflight_stops_on_missing_mount() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = build_preflight(project_root)
    assert report["status"] == "HOLD_MISSING_PINNED_TIANJI_TOOL_TO_KAIHAND_ROOT_TRANSFORM"
    assert report["inputs"]["r2"]["q_arm_present"] is False
    assert report["verified"]["asset_pin_calibration_files"] == 0
    assert report["verified"]["pinned_adapter_or_mount_named_assets"] == []
    assert report["t1_candidate_card_ready"] is False
