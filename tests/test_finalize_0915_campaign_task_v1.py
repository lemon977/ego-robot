from __future__ import annotations

from chaoyang.governance import finalize_0915_campaign_task_v1 as subject


def test_finalizer_has_explicit_bounded_terminal_sets() -> None:
    assert subject.TERMINAL == {
        "PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE",
        "BLOCKED_EXTERNAL", "CANCELLED",
    }
    assert "RUNNING" in subject.LIVE
    assert not (subject.TERMINAL & subject.LIVE)
