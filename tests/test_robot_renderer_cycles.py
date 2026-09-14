from pathlib import Path

import numpy as np
import pytest

from pipeline.robot_renderer_cycles import (
    RendererError,
    forward_kinematics,
    full_chain_gaps,
    parse_urdf,
    range_to_metric_z,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEFT_URDF = PROJECT_ROOT / (
    "assets/robot/kaihand/packages/"
    "KaiBot-Dexhand shell-URDF-L-260624(1620)/urdf/"
    "KaiBot-Dexhand shell-URDF-L-260624(1620).urdf"
)


def test_actual_kaihand_urdf_fk_resolves_all_links() -> None:
    model = parse_urdf(LEFT_URDF)
    q = {
        joint.name: 0.0
        for joint in model.joints
        if joint.joint_type != "fixed"
    }
    transforms = forward_kinematics(model, q)
    assert model.root_link == "hand_l_base_link"
    assert len(model.links) == 23
    assert len(model.joints) == 22
    assert len(model.visuals) == 23
    assert set(transforms) == set(model.links)
    assert all(np.isfinite(transform).all() for transform in transforms.values())


def test_fk_rejects_missing_joint_instead_of_defaulting() -> None:
    model = parse_urdf(LEFT_URDF)
    q = {
        joint.name: 0.0
        for joint in model.joints
        if joint.joint_type != "fixed"
    }
    q.pop(next(iter(q)))
    with pytest.raises(RendererError, match="joint state identity mismatch"):
        forward_kinematics(model, q)


def test_range_to_metric_z_recovers_fronto_parallel_plane() -> None:
    height, width = 5, 7
    fx, fy, cx, cy = 6.0, 8.0, 3.0, 2.0
    u = np.arange(width)[None, :]
    v = np.arange(height)[:, None]
    z_expected = np.full((height, width), 2.0)
    ranges = z_expected * np.sqrt(
        1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2
    )
    assert np.allclose(range_to_metric_z(ranges, fx, fy, cx, cy), z_expected)


def test_full_chain_gap_audit_fails_loud_for_current_inputs() -> None:
    gaps = full_chain_gaps(
        ["q", "wrist_T_camera", "valid"],
        calibration_file_count=0,
        placement_ref=None,
    )
    assert gaps == [
        "R2_SIDECAR_HAS_NO_Q_ARM",
        "TIANJI_CALIBRATION_EMPTY",
        "UNIFORM_BASE_PLACEMENT_CONTRACT_MISSING",
    ]
