from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.robot_review_video import ReviewVideoError, alpha_compose, require_new_output_root


def test_alpha_compose_exact_endpoints() -> None:
    raw = np.full((2, 3, 3), 10, dtype=np.uint8)
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[0, 0, 3] = 255
    result = alpha_compose(raw, rgba)
    assert np.array_equal(result[0, 0], [200, 200, 200])
    assert np.array_equal(result[1, 1], [10, 10, 10])


def test_alpha_compose_rejects_shape_mismatch() -> None:
    with pytest.raises(ReviewVideoError, match="shape mismatch"):
        alpha_compose(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2, 4), np.uint8))


def test_output_requires_new_direct_review_child(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    good = project / "_run" / "004_r2_kaihand_regenerated_review_v1"
    require_new_output_root(project, good)
    with pytest.raises(ReviewVideoError, match="direct child"):
        require_new_output_root(project, project / "processed" / good.name)
    with pytest.raises(ReviewVideoError, match="wrong review namespace"):
        require_new_output_root(project, project / "_run" / "unscoped")
    good.mkdir()
    with pytest.raises(ReviewVideoError, match="refusing to overwrite"):
        require_new_output_root(project, good)
