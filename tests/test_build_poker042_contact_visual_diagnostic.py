from __future__ import annotations

from pathlib import Path
import sys

import cv2
import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.build_poker042_contact_visual_diagnostic import (  # noqa: E402
    SOURCE,
    VisualDiagnosticError,
    run,
)


def test_real_visual_diagnostic_is_non_robot_and_no_clobber(tmp_path: Path) -> None:
    if not SOURCE.exists():
        pytest.skip("Poker042 hypothesis bundle unavailable")
    output = tmp_path / "visual"
    result = run(SOURCE, output)
    assert result["status"] == "PENDING_MANUAL_VISUAL_REVIEW"
    assert result["selected_frame_ids"] == [99, 100, 104, 114]
    assert result["robot_pixels_drawn"] is False
    assert result["robot_solver_run"] is False
    assert result["formal_object6d_mutated"] is False
    assert result["authority_promoted"] is False
    image = cv2.imread(result["outputs"]["four_frame_sheet"]["path"])
    assert image.shape == (1800, 2560, 3)
    capture = cv2.VideoCapture(result["outputs"]["short_video"]["path"])
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 80
    capture.release()
    with pytest.raises(VisualDiagnosticError, match="refusing to overwrite"):
        run(SOURCE, output)
