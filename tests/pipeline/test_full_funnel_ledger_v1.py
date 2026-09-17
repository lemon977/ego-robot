from __future__ import annotations

import copy

import pytest

from chaoyang.pipeline.full_funnel_ledger_v1 import (
    FullFunnelLedgerError,
    initialize,
    update_stage,
    validate,
)


def sessions() -> list[dict[str, str]]:
    return [
        {"task": "playing_cards" if index < 120 else "potato_chips",
         "session_id": f"session_{index:03d}"}
        for index in range(220)
    ]


def test_fixed_220_and_generated_counts() -> None:
    ledger = initialize(sessions())
    update_stage(
        ledger, task="playing_cards", session_id="session_000", stage="Raw",
        status="PASS", reason=None, input_signature="a" * 64,
        weight_identity="ABSENT", calibration_identity="camera-sha",
        artifact={"path": "/tmp/receipt.json", "bytes": 12, "sha256": "b" * 64},
    )
    assert ledger["summary"]["Raw"]["PASS"] == 1
    assert ledger["summary"]["Raw"]["NOT_RUN"] == 219
    assert ledger["sessions"][0]["first_blocker"] is None


def test_first_blocker_is_derived_in_stage_order() -> None:
    ledger = initialize(sessions())
    update_stage(
        ledger, task="playing_cards", session_id="session_000", stage="Mask",
        status="REJECTED_QUALITY", reason="identity switch", input_signature="a" * 64,
        weight_identity="SAM3.1:sha", calibration_identity=None, artifact=None,
    )
    assert ledger["sessions"][0]["first_blocker"] == "Mask"


def test_tampered_summary_is_rejected() -> None:
    ledger = initialize(sessions())
    tampered = copy.deepcopy(ledger)
    tampered["summary"]["Raw"]["PASS"] = 220
    with pytest.raises(FullFunnelLedgerError, match="summary"):
        validate(tampered)


def test_wrong_denominator_is_rejected() -> None:
    with pytest.raises(FullFunnelLedgerError, match="denominator"):
        initialize(sessions()[:-1])
