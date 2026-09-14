from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_poker042_human_contact_v1_canary import (  # noqa: E402
    CanaryError,
    PROJECT,
    run,
)


def test_real_poker042_build_is_independent_and_fail_closed(tmp_path: Path) -> None:
    wave = PROJECT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
    if not wave.exists():
        pytest.skip("frozen Wave0 selection unavailable")
    output = tmp_path / "poker042_v3"
    result = run(wave, output)
    gate = result["human_contact_gate"]
    assert gate["status"] == "COMPLETE_HYPOTHESIS_ONLY"
    assert gate["full_session_frame_count"] == 171
    assert gate["direct_object6d_frames"] == 73
    assert gate["unknown_object_pose_frames"] == 94
    assert gate["inferred_object_pose_frames"] == 4
    assert gate["formal_object6d_mutated"] is False
    assert gate["clean_required"] is False
    assert gate["robot_render_required"] is False
    assert result["four_keyframes"]["selected_frame_ids"] == [99, 100, 104, 114]
    assert result["retarget_gate"]["status"] == "NO_GO_BLOCKED_PREREQ"
    assert result["retarget_gate"]["solver_executed"] is False
    assert "CAMERA_WORLD_TO_TIANJI_BASE_CALIBRATION_MISSING" in result["retarget_gate"]["hard_blockers"]
    assert result["compositor_gate"]["feeds_back_to_human_contact"] is False
    assert result["compositor_gate"]["feeds_back_to_retarget"] is False
    assert result["authority_promoted"] is False
    with np.load(output / "HUMAN_CONTACT_HYPOTHESIS_V1.npz", allow_pickle=False) as data:
        assert data["frame_indices"].shape == (171,)
        assert np.count_nonzero(data["formal_object6d_valid"]) == 73
        assert np.count_nonzero(data["pose_hypothesis"]) == 4
        assert np.isnan(data["pose_object_to_camera"][~data["pose_valid"]]).all()
        assert data["frozen_four_frame_indices"].tolist() == [99, 100, 104, 114]
    assert json.loads((output / "HUMAN_CONTACT_HYPOTHESIS_V1_CONTRACT.json").read_text())["output"]["authority_promotable"] is False


def test_real_poker042_bundle_is_no_clobber(tmp_path: Path) -> None:
    wave = PROJECT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
    if not wave.exists():
        pytest.skip("frozen Wave0 selection unavailable")
    output = tmp_path / "poker042_v3"
    run(wave, output)
    with pytest.raises(CanaryError, match="refusing to overwrite"):
        run(wave, output)
