import importlib.util
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "run_hawor_stereo_surface_consistency_qa.py"
SPEC = importlib.util.spec_from_file_location("surface_qa", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_triangle_zbuffer_uses_visible_nearest_surface():
    triangles = np.asarray([
        [[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.0, 0.1, 1.0]],
        [[-0.1, -0.1, 0.5], [0.1, -0.1, 0.5], [0.0, 0.1, 0.5]],
    ], dtype=np.float64)
    depth, labels = MODULE.rasterize_visible_surface(triangles, np.asarray([0, 1], np.int16), 40.0, 40.0, 20.0, 20.0, 40, 40)
    assert np.isclose(depth[20, 20], 0.5, atol=1e-9)
    assert labels[20, 20] == 1


def test_registered_depth_identity_contract():
    depth = np.full((4, 5), 0.5, np.float32)
    valid = np.ones_like(depth, bool)
    registration = {
        "stereo_rectified_depth_formula_intrinsics": np.asarray([[10.0, 0, 2.0], [0, 10.0, 1.5], [0, 0, 1.0]]),
        "T_stereo_rectified_camera_to_selected_camera": np.eye(4),
        "H_depth_pixel_to_selected_rgb": np.eye(3),
    }
    selected, selected_valid, local = MODULE.registered_selected_depth(depth, valid, registration, (5, 4), 0.02)
    assert np.allclose(selected, 0.5)
    assert selected_valid.all()
    assert local.all()


def test_metrics_signed_and_absolute_are_not_mixed():
    result = MODULE.metrics(np.asarray([-0.002, 0.001, 0.003]))
    assert np.isclose(result["signed_bias_mean_mm"], 2.0 / 3.0)
    assert np.isclose(result["signed_p50_mm"], 1.0)
    assert np.isclose(result["abs_mae_mm"], 2.0)
    assert np.isclose(result["abs_p50_mm"], 2.0)


def test_same_session_binding_fails_closed():
    try:
        MODULE.validate_session_binding("expected", {"session": "other"}, {"session_id": "expected"})
    except RuntimeError as error:
        assert "session mismatch" in str(error)
    else:
        raise AssertionError("cross-session input did not fail closed")
