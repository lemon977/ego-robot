from __future__ import annotations

from chaoyang.ops.run_sam31_lowmem_equivalence_v1 import compare_rows


def row(*, mask: str = "abc", score: float = 0.7) -> dict:
    return {
        "frame_index": 0,
        "present": True,
        "area_pixels": 12,
        "output_object_ids": [0],
        "reason": "FROZEN_ID_PRESENT",
        "mask_sha256": mask,
        "score": score,
    }


def test_exact_mask_id_and_score_match() -> None:
    result = compare_rows([row()], [row(score=0.7000005)])
    assert result["equivalent"] is True
    assert result["mask_sha_equal_count"] == 1


def test_pixel_change_cannot_be_hidden_by_presence() -> None:
    result = compare_rows([row(mask="new")], [row(mask="old")])
    assert result["equivalent"] is False
    assert result["mismatches"][0]["fields"] == ["mask_sha256"]


def test_score_outside_frozen_tolerance_fails() -> None:
    result = compare_rows([row(score=0.7)], [row(score=0.71)])
    assert result["equivalent"] is False
    assert "score" in result["mismatches"][0]["fields"]


def test_missing_stream_frame_fails() -> None:
    result = compare_rows([row()], [row(), row()])
    assert result == {"equivalent": False, "reason": "FRAME_COUNT_MISMATCH"}
