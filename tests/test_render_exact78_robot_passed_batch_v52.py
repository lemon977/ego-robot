from __future__ import annotations

import pytest

from tools import render_exact78_robot_passed_batch_v52 as render


REF_A = {"path": "/tmp/a", "bytes": 1, "sha256": "a" * 64}
REF_B = {"path": "/tmp/b", "bytes": 2, "sha256": "b" * 64}


def test_arm_payload_only_accepts_final_contract() -> None:
    assert render.arm_payload(
        {
            "status": "PASS_ARM_METHOD1_CARRIED_NO_ROUND2",
            "method1_result": REF_A,
            "method1_states": REF_B,
            "selected_backoff_m": 0.24,
        }
    ) == (REF_A, REF_B, "fixed placement=0.2400 m", True)
    assert render.arm_payload(
        {
            "status": "PASS_ARM_ROUND2_BIDIRECTIONAL",
            "result": REF_A,
            "states": REF_B,
            "method1_best_hold_backoff_m": 0.258,
        }
    ) == (REF_A, REF_B, "fixed placement=0.2580 m; arm method2 bidirectional", True)
    assert render.arm_payload(
        {
            "status": "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C",
            "result": REF_A,
            "states": REF_B,
            "method1_best_hold_backoff_m": 0.24,
        }
    ) == (REF_A, REF_B, "fixed placement=0.2400 m; arm method2 HOLD", False)
    with pytest.raises(ValueError, match="unexpected"):
        render.arm_payload({"status": "FAILED_RUNTIME_RETRYABLE"})


def test_hand_payload_never_promotes_hold_or_arm_block() -> None:
    assert render.hand_payload(
        {"status": "PASS_HAND_METHOD1_CARRIED_NO_ROUND2", "method1_result": REF_A, "method1_states": REF_B}
    ) == (REF_A, REF_B, True)
    assert render.hand_payload({"status": "PASS_HAND_ROUND2_BIDIRECTIONAL", "result": REF_A, "states": REF_B}) == (
        REF_A,
        REF_B,
        True,
    )
    assert render.hand_payload(
        {"status": "HOLD_HAND_AFTER_TWO_METHODS_FAILED_QUALITY_C", "result": REF_A, "states": REF_B}
    ) == (REF_A, REF_B, False)
    with pytest.raises(ValueError, match="unexpected"):
        render.hand_payload({"status": "FAILED_RUNTIME_RETRYABLE"})


def test_unique_index_rejects_duplicate_identity() -> None:
    assert set(render.unique_index([{"session": "a"}, {"session": "b"}], "rows")) == {"a", "b"}
    with pytest.raises(ValueError, match="duplicate"):
        render.unique_index([{"session": "a"}, {"session": "a"}], "rows")


def test_media_artifact_ref_allows_audited_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"path": "/tmp/a", "bytes": 1, "sha256": "a" * 64}
    monkeypatch.setattr(render, "ref", lambda path: expected)
    value = {**expected, "decoded_frames": 12, "width": 1920, "height": 480}
    assert render.verify_artifact_ref(value, "video") == render.Path("/tmp/a")
    with pytest.raises(ValueError, match="artifact ref"):
        render.verify_artifact_ref({"path": "/tmp/a"}, "video")
