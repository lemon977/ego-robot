from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.governance.robot15h_task_specs_v1 import build_packet
from chaoyang.ops import build_0915_robot15h_final_evidence_v1 as final_evidence
from chaoyang.ops import run_0915_robot15h_foundationstereo_waves_v1 as depth_w1
from chaoyang.ops import run_0915_robot15h_wave_scope_gate_v1 as scope_gate


def test_w1_scope_gate_uses_frozen_eight_session_denominator() -> None:
    rows = scope_gate.w1_rows()
    assert len(rows) == 8
    assert sum(int(row["frame_count"]) for row in rows) == 2332
    assert len({row["source_group"] for row in rows}) == 8


@pytest.mark.parametrize(
    ("task_id", "weights"),
    [
        ("0915_robot15h_hawor_waves_v1", [
            "assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json"
        ]),
        ("0915_robot15h_sam31_waves_v1", [
            "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
        ]),
        ("0915_robot15h_foundationstereo_waves_v1", [
            "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
        ]),
        ("0915_robot15h_robot_waves_v1", "ABSENT"),
    ],
)
def test_w1_packets_are_real_bounded_specs(task_id: str, weights: object) -> None:
    packet = build_packet(task_id)
    assert packet["weights"] == weights
    assert "bounded future-node packet" not in packet["objective"]
    assert "BATCH_RESULT.json" in packet["required_outputs"]
    assert len(packet["read_set"]) <= 8


def test_foundation_w1_refuses_final_holdout_before_h9(monkeypatch: pytest.MonkeyPatch) -> None:
    class BeforeH9(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return cls.fromisoformat("2026-09-19T08:59:59+08:00")

    monkeypatch.setattr(depth_w1, "datetime", BeforeH9)
    with pytest.raises(RuntimeError, match="still sealed until"):
        depth_w1.w1_rows()


def test_foundation_w1_binds_sha_on_copies_after_h9(monkeypatch: pytest.MonkeyPatch) -> None:
    class AfterH9(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return cls.fromisoformat("2026-09-19T09:10:00+08:00")

    monkeypatch.setattr(depth_w1, "datetime", AfterH9)
    monkeypatch.setattr(depth_w1.base, "sha256", lambda _path: "a" * 64)
    rows = depth_w1.w1_rows()
    assert len(rows) == 8
    assert all(row["source_stereo"]["sha256"] == "a" * 64 for row in rows)
    assert all(
        row["source_stereo"]["sha_policy"]
        == "BOUND_AT_W1_SCHEDULING_WITHOUT_MUTATING_FROZEN_MANIFEST"
        for row in rows
    )


def test_foundation_w1_corrects_batch_denominator_from_session_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(depth_w1, "OUTPUT", tmp_path)
    depth_w1.base.atomic_json(tmp_path / "BATCH_RESULT.json", {
        "schema_version": "old-wave0-name",
        "counts": {"attempted": 4},
        "sessions": [
            {"status": "PASSED"},
            {"status": "PASSED"},
            {"status": "REJECTED_QUALITY"},
            {"status": "FAILED_RUNTIME"},
        ],
    })
    batch = depth_w1._correct_batch()
    assert batch is not None
    assert batch["schema_version"] == "0915-robot15h-foundationstereo-w1-batch-v1"
    assert batch["counts"] == {
        "attempted": 4,
        "passed": 2,
        "rejected_quality": 1,
        "failed_runtime": 1,
        "unrun": 0,
    }


def test_release_matrix_allows_only_foundation_for_real_w1_execution() -> None:
    allowed = depth_w1._validate_release_scope()
    assert {row["task"] for row in allowed} == {"playing_cards", "potato_chips"}
    assert all(row["w1_expansion_allowed"] is True for row in allowed)
    assert not any(
        row.get("w1_expansion_allowed")
        for row in scope_gate.matrix_rows("hawor_direct_observed")
    )


def test_final_evidence_full_decode_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "review.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24),
    )
    assert writer.isOpened()
    for value in range(5):
        writer.write(np.full((24, 32, 3), value * 20, np.uint8))
    writer.release()
    passed = final_evidence.full_decode(path, 5)
    assert passed["full_decode"] is True
    assert passed["decoded_frames"] == 5
    failed = final_evidence.full_decode(path, 6)
    assert failed["full_decode"] is False


def test_final_source_integrity_is_limited_to_frozen_twelve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "processed" / "chips_cards_hands_0915"
    allowed.mkdir(parents=True)
    sessions = []
    binding = []
    for index in range(12):
        source = allowed / f"session_{index:03d}.mp4"
        source.write_bytes(f"source-{index}".encode())
        item = {
            "session_id": f"session_{index:03d}",
            "wave": "W0" if index < 4 else "W1",
            "split": "development" if index < 4 else "validation",
            "source_stereo": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": final_evidence.sha256(source) if index < 4 else None,
            },
        }
        sessions.append(item)
        if index >= 4:
            binding.append({
                "session_id": item["session_id"],
                "source_stereo": {
                    "path": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": final_evidence.sha256(source),
                },
            })
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    w1_binding = tmp_path / "binding.json"
    w1_binding.write_text(json.dumps({"sessions": binding}), encoding="utf-8")
    monkeypatch.setattr(final_evidence, "INVENTORY", inventory)
    monkeypatch.setattr(final_evidence, "W1_INPUT_BINDING", w1_binding)
    monkeypatch.setattr(final_evidence, "ALLOWED_DATASET_ROOT", allowed)
    result = final_evidence.selected_source_integrity()
    assert result["status"] == "PASSED"
    assert result["sessions_checked"] == 12
    assert result["content_drift_count"] == 0
    assert result["0916_consumed"] is False
