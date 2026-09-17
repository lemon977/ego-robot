import numpy as np

from chaoyang.ops.audit_object6d_coordinate_domain_v71 import rotation_angle_deg


def test_rotation_angle_reports_registration_rotation() -> None:
    angle = np.deg2rad(5.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    assert abs(rotation_angle_deg(rotation) - 5.0) < 1e-8
