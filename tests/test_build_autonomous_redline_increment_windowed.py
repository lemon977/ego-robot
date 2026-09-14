from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path("/mnt/workspace/code/chaoyang")
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import build_autonomous_redline_increment_windowed as target  # noqa: E402


def _args(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "start": "2026-08-29T08:00:00+08:00",
        "cutoff": "2026-08-29T10:00:00+08:00",
        "checkpoint_target": "2026-08-29T12:00:00+08:00",
        "snapshot_kind": "preflight",
        "decision_packet": target.RUN_ROOT / "A_CLASS_DECISION_QUEUE_1200_V2.json",
        "scope_deviation_ref": (
            target.CHECKPOINT_ROOT / "CHECKPOINT_1200_PREFLIGHT_SCOPE_DEVIATION_T0_V1.json"
        ),
        "output": target.CHECKPOINT_ROOT / "CHECKPOINT_1200_RED_LINE_PREFLIGHT_T0_V1.json",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_accepts_0800_1200_prefix() -> None:
    start, cutoff, target_time, *_ = target.validate_request(_args())
    assert start.isoformat() == "2026-08-29T08:00:00+08:00"
    assert cutoff.isoformat() == "2026-08-29T10:00:00+08:00"
    assert target_time.isoformat() == "2026-08-29T12:00:00+08:00"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cutoff", "2026-08-29T08:00:00+08:00", "start < cutoff"),
        ("cutoff", "2026-08-29T12:00:01+08:00", "cutoff <= checkpoint-target"),
        ("start", "2026-08-29T04:00:00+08:00", "exactly four hours"),
        ("cutoff", "2026-08-29T02:00:00+00:00", "same UTC offset"),
    ],
)
def test_rejects_wrong_window(field: str, value: str, message: str) -> None:
    with pytest.raises(target.WindowedAuditError, match=message):
        target.validate_request(_args(**{field: value}))


def test_rejects_noncanonical_decision_path() -> None:
    with pytest.raises(target.WindowedAuditError, match="decision packet is not a canonical named path"):
        target.validate_request(_args(decision_packet=PROJECT_ROOT / "archive/legacy/task_cards/06_HISTORY.md"))


def test_rejects_noncheckpoint_output_path() -> None:
    with pytest.raises(target.WindowedAuditError, match="output is not a canonical checkpoint JSON path"):
        target.validate_request(_args(output=target.RUN_ROOT / "bad.json"))
