from __future__ import annotations

import pytest

from chaoyang.ops.build_0916_cleaning_visuals_v1 import selected_clean_rows, summarize


def _result() -> dict:
    rows = []
    for index in range(240):
        task = "playing_cards" if index < 130 else "potato_chips"
        classification = "REJECTED" if index == 35 else "CLEANED"
        rows.append({
            "task": task, "session_id": f"{index + 1:03d}",
            "classification": classification,
            "reason": "VISUAL_OR_TRACKING_CONTENT" if classification == "REJECTED" else None,
            "target": f"/tmp/{index}", "source": f"/source/{index}",
        })
    return {
        "state": "COMMITTED", "session_count": 240,
        "cleaned": 239, "rejected": 1, "failed": 0, "results": rows,
    }


def test_summary_is_derived_from_fixed_240_rows() -> None:
    summary = summarize(_result())
    assert summary["by_task"]["playing_cards"] == {"CLEANED": 129, "REJECTED": 1}
    assert summary["by_task"]["potato_chips"] == {"CLEANED": 110}
    assert summary["rejection_reasons"] == {"VISUAL_OR_TRACKING_CONTENT": 1}


def test_representative_selection_is_bounded() -> None:
    rows = selected_clean_rows(_result())
    assert len(rows) == 6
    assert {row["task"] for row in rows} == {"playing_cards", "potato_chips"}


def test_incomplete_dataset_is_rejected() -> None:
    value = _result()
    value["failed"] = 1
    with pytest.raises(ValueError, match="zero-runtime-failure"):
        summarize(value)
