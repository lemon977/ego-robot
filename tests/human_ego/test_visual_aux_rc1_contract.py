from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.human_ego.tools.visual_aux_rc1_contract import (
    CURRENT_CONTRACT_PATH,
    RC1ContractError,
    validate_rc1_eligibility,
)


def _ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, input_mode: str = "CAUSAL_TRAINING_INPUT") -> Path:
    raw = tmp_path / "raw.png"
    robot = tmp_path / "robot.png"
    raw.write_bytes(b"raw")
    robot.write_bytes(b"robot")
    raw_selector = _json(
        tmp_path / "raw_selector.json",
        {"frames": [{"rgb": _ref(raw)}]},
    )
    robot_selector = _json(
        tmp_path / "robot_selector.json",
        {"frames": [{"rgb": _ref(robot)}]},
    )
    labels = tmp_path / "labels.npz"
    labels.write_bytes(b"labels")
    pixel_source = tmp_path / "pixel_source.npz"
    pixel_source.write_bytes(b"pixel-source")
    unknown = tmp_path / "unknown.npz"
    unknown.write_bytes(b"unknown")
    proof = _json(
        tmp_path / "input_payload_receipt.json",
        {
            "status": "PASSED",
            "input_mode": input_mode,
            "repeatability_pass": True,
            "future_mutation_test_pass": True,
            "source_max_frame_le_target": True,
            "shared_neutralization_pass": True,
        },
    )
    manifest = _json(
        tmp_path / "VISUAL_AUX_SESSION_MANIFEST.json",
        {
            "task": "chips",
            "session_id": "chips_fixture",
            "split": "train",
            "input_mode": input_mode,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
        },
    )
    eligibility = {
        "schema_version": "VISUAL_AUX_RC1_ELIGIBILITY_V1",
        "release_id": "chaoyang-visual-aux-rc1-final",
        "task": "chips",
        "session_id": "chips_fixture",
        "source_group_id": "capture-fixture",
        "split": "train",
        "execution_status": "COMPLETED",
        "visual_quality": "PARTIAL_TRAINABLE",
        "train_eligible": True,
        "consumer_eligibility": {
            "visual_aux_rc1": {
                "eligible": True,
                "reasons": [],
                "evaluated_at": "2026-09-16T00:00:00+08:00",
            }
        },
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "contract": _ref(CURRENT_CONTRACT_PATH),
        "artifacts": {
            "manifest": _ref(manifest),
            "labels": _ref(labels),
            "raw_selector": _ref(raw_selector),
            "robotized_selector": _ref(robot_selector),
            "pixel_source": _ref(pixel_source),
            "final_input_unknown_mask": _ref(unknown),
            "input_payload_receipt": _ref(proof),
        },
        "counts": {"eligible_h50_starts": 1},
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "fixture",
    }
    return _json(tmp_path / "VISUAL_AUX_RC1_ELIGIBILITY.json", eligibility)


def test_rc1_eligible_real_bundle_contract_passes(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    result = validate_rc1_eligibility(path)
    assert result["status"] == "PASS_VISUAL_AUX_RC1_ELIGIBILITY"


def test_rc1_offline_bidirectional_bundle_is_rejected(tmp_path: Path) -> None:
    path = _fixture(tmp_path, input_mode="OFFLINE_BIDIRECTIONAL_VISUALIZATION")
    with pytest.raises((RC1ContractError, Exception)):
        validate_rc1_eligibility(path)


def test_rc1_review_video_is_rejected(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    value = json.loads(path.read_text())
    selector_path = Path(value["artifacts"]["raw_selector"]["path"])
    video = tmp_path / "review.mp4"
    video.write_bytes(b"review")
    _json(selector_path, {"frames": [{"rgb": _ref(video)}]})
    value["artifacts"]["raw_selector"] = _ref(selector_path)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RC1ContractError, match="forbidden"):
        validate_rc1_eligibility(path)
