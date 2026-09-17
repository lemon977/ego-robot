"""Robot30 published counts must be derived from the fixed 60 rows."""

from __future__ import annotations

import pytest

from chaoyang.ops import build_rc1_robot30_research_delivery_revision_v7 as revision


def _rows() -> list[dict]:
    rows = []
    for task in ("chips", "poker"):
        for rank in range(1, 31):
            rows.append({"task": task, "rank": rank,
                         "session_id": f"{task}_fixture_{rank:03d}",
                         "delivery_full_video": {"source_kind": "PRIOR_VERIFIED"} if rank <= (30 if task == "chips" else 21) else None,
                         "delivery_class": "PRIOR_VERIFIED_FULL_VIDEO",
                         "robot30_hard_geometry_pass": False})
    return rows


def test_frozen_30_counts_are_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(revision, "delivered", lambda row: row["delivery_full_video"] is not None)
    counts, invariant = revision.rebuild_counts(_rows())
    assert counts["chips"]["all_watchable_full_video"] == 30
    assert counts["poker"]["all_watchable_full_video"] == 21
    assert counts["poker"]["missing_full_video"] == 9
    assert invariant["poker"]["row_boolean_sum_equals_counts"] is True


def test_duplicate_fixed_session_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(revision, "delivered", lambda row: row["delivery_full_video"] is not None)
    rows = _rows()
    rows[31]["session_id"] = rows[30]["session_id"]
    with pytest.raises(RuntimeError, match="60 unique"):
        revision.rebuild_counts(rows)
