from __future__ import annotations

import pytest

from chaoyang.ops.record_three_stream_lane_result_v31 import _derived_status


@pytest.mark.parametrize(
    ("machine", "lane"),
    [
        ("PASS_CPU_PREFLIGHT", "CPU_PREFLIGHT_COMPLETE"),
        ("PASSED", "CPU_PREFLIGHT_COMPLETE"),
        ("BLOCKED_INDEPENDENT_OBSERVABILITY", "BLOCKED_CPU_PREFLIGHT"),
        ("DIAGNOSTIC_DISPARITY_ONLY", "BLOCKED_CPU_PREFLIGHT"),
        ("REJECTED_IMAGE_DOMAIN", "REJECTED_CPU_PREFLIGHT"),
        ("FAILED_RUNTIME", "REJECTED_CPU_PREFLIGHT"),
    ],
)
def test_lane_state_is_derived_from_machine_status(machine: str, lane: str) -> None:
    assert _derived_status(machine)[0] == lane


def test_unknown_status_cannot_be_recorded() -> None:
    with pytest.raises(RuntimeError, match="unsupported machine result status"):
        _derived_status("LOOKS_GOOD_TO_ME")
