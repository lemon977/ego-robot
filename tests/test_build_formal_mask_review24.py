from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.build_formal_mask_review24 import ReviewError, validate_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    PROJECT_ROOT / "tasks/chips/runs/mask/20260903_chips001_canonical_review24_v1/ANCHOR_REVIEW_SPEC.json",
    PROJECT_ROOT / "tasks/poker/runs/mask/20260903_poker001_canonical_review24_v1/ANCHOR_REVIEW_SPEC.json",
)


@pytest.mark.parametrize("path", SPECS)
def test_formal_specs_pin_exactly_24_sorted_canonical_frames(path: Path) -> None:
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate_spec(spec)
    frames = [row["frame_index"] for row in spec["review_frames"]]
    assert len(frames) == len(set(frames)) == 24
    assert spec["canonical_session_path"].startswith(
        "/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/"
    )
    assert spec["shared_model"]["weight_id"] == "sam31.multiplex.0567debe"


def test_spec_rejects_non_24_or_out_of_image_anchor() -> None:
    spec = json.loads(SPECS[0].read_text(encoding="utf-8"))
    wrong_count = deepcopy(spec)
    wrong_count["review_frames"].pop()
    with pytest.raises(ReviewError, match="exactly 24"):
        validate_spec(wrong_count)

    bad_anchor = deepcopy(spec)
    bad_anchor["manual_anchors"][0]["xy"] = [1280, 0]
    with pytest.raises(ReviewError, match="outside image"):
        validate_spec(bad_anchor)
