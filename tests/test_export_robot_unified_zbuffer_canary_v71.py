import numpy as np
import pytest

from chaoyang.ops.export_robot_unified_zbuffer_canary_v71 import source_image_size, uniform_valid_frames


def test_uniform_valid_frames_requires_both_sides_and_covers_span() -> None:
    valid = np.ones((2, 10), dtype=np.bool_)
    valid[1, 4] = False
    assert uniform_valid_frames(valid, 4).tolist() == [0, 3, 6, 9]


def test_uniform_valid_frames_fails_closed() -> None:
    with pytest.raises(ValueError, match="bilateral valid"):
        uniform_valid_frames(np.asarray([[True], [False]], dtype=np.bool_), 1)


def test_source_domain_uses_centered_k_instead_of_2048_fallback() -> None:
    hawor = {"intrinsics": np.asarray([[[640.0, 0.0, 639.5], [0.0, 640.0, 479.5], [0.0, 0.0, 1.0]]])}
    assert source_image_size(hawor, 0) == (1280.0, 960.0, "CENTERED_PRINCIPAL_POINT_INFERENCE")


def test_source_domain_honors_explicit_image_size() -> None:
    hawor = {"intrinsics": np.eye(3)[None], "image_size": np.asarray([2048, 1536])}
    assert source_image_size(hawor, 0) == (2048.0, 1536.0, "HAWOR_IMAGE_SIZE")
