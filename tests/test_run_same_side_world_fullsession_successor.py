from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.run_same_side_world_fullsession_successor import (
    CONFIG,
    FullSessionError,
    _motion_metrics,
    verified_artifact,
)


def test_fullsession_contract_is_same_side_and_exact_length() -> None:
    assert CONFIG["poker"]["frame_count"] == 171
    assert CONFIG["chips"]["frame_count"] == 293
    assert CONFIG["poker"]["decision"].startswith("ACCEPT_POKER_WORLD_FIRST_SAME_SIDE")
    assert CONFIG["chips"]["decision"].startswith("ACCEPT_CHIPS_WORLD_FIRST_SAME_SIDE")


def test_verified_artifact_fails_closed_on_byte_drift(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    reference = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert verified_artifact(reference, "value") == path.resolve()
    path.write_text(json.dumps({"ok": False}), encoding="utf-8")
    with pytest.raises(FullSessionError, match="byte identity drift"):
        verified_artifact(reference, "value")


def test_motion_metrics_keep_velocity_and_acceleration_separate() -> None:
    values = np.asarray([0.0, 0.05, 0.12], dtype=np.float64)[:, None, None]
    metrics = _motion_metrics(values)
    assert metrics["velocity_max_rad_per_frame"] == pytest.approx(0.07)
    assert metrics["acceleration_max_rad_per_frame2"] == pytest.approx(0.02)
