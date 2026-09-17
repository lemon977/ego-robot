from __future__ import annotations

import inspect

from chaoyang.ops import batch_clean_handle_content_v4 as subject


def test_v4_is_bounded_to_two_workers_and_reuses_v3_converter() -> None:
    assert subject.MAX_WORKERS == 2
    source = inspect.getsource(subject)
    assert "v3.convert_one" in source
    assert "convert_handle_egodex_v3.py" in source
    assert "max_workers=args.workers" in source


def test_deterministic_result_order_and_counts() -> None:
    slots = [
        {"classification": "CLEANED", "session_id": "001"},
        None,
        {"classification": "REJECTED", "session_id": "003"},
        {"classification": "FAILED", "session_id": "004"},
    ]
    completed = subject._ordered_completed(slots)
    assert [row["session_id"] for row in completed] == ["001", "003", "004"]
    assert subject._counts(completed) == {
        "completed": 3,
        "cleaned": 1,
        "rejected": 1,
        "failed": 1,
    }


def test_v4_claim_scope_stops_at_cleaning() -> None:
    source = inspect.getsource(subject)
    assert "no HaWoR, Mask, Depth, Contact or Robot authority" in source
    assert "UNCHANGED_FROM_V3" in source

