from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from chaoyang.ops.evaluate_s1_modal_mask_canaries_v71 import build_comparison


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _receipt(root: Path, backend: str, masks: list[bool]) -> Path:
    frames = []
    for frame_id, known in enumerate(masks):
        record = None
        if known:
            mask = np.zeros((3, 3), dtype=np.uint8)
            mask[1, frame_id % 3] = 1
            path = root / backend / f"{frame_id:06d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, mask, allow_pickle=False)
            record = {**_ref(path), "encoding": "NPY_UINT8_MODAL_BINARY"}
        frames.append({
            "frame_id": frame_id,
            "prompt_instance_ids": [],
            "instances": [{
                "instance_id": "poker_0",
                "visibility": "VISIBLE" if known else "TRACK_LOST_UNKNOWN",
                "mask": record,
            }],
            "identity_qa": {"overlap_pixels": 0, "warnings": []},
        })
    output = {
        "schema_version": "causal-modal-mask-output-v71",
        "session_id": "play_cards_test",
        "execution_mode": "CAUSAL_PROCESSING",
        "mask_semantics": "VISIBLE_MODAL_SURFACE_ONLY",
        "allow_instance_union": False,
        "frames": frames,
        "identity_qa": {"warning_count": 0, "warnings": [], "instance_ids": ["poker_0"]},
    }
    output_path = root / backend / "MASK_OUTPUT.json"
    output_path.write_text(json.dumps(output))
    receipt = {
        "payload": {
            "backend": backend,
            "status": "PASSED",
            "execution_performed": True,
            "runtime_error": None,
            "output": _ref(output_path),
        }
    }
    receipt_path = root / backend / "RESULT.json"
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path


def test_comparison_reports_coverage_not_accuracy(tmp_path):
    audit = {
        "rows": [{
            "session_id": "play_cards_test",
            "event": {
                "event_id": "event_0",
                "empty_or_unobserved_start": 1,
                "empty_or_unobserved_end": 2,
                "audit_frames": [0, 1, 2, 3],
            },
        }]
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    first = _receipt(tmp_path, "cutie", [True, False, True, True])
    second = _receipt(tmp_path, "sam2_1", [True, True, True, True])
    value = build_comparison([first, second], audit_path)
    assert value["status"] == "COMPARISON_COMPLETE_REVIEW_REQUIRED"
    assert value["gold_accuracy_computed"] is False
    assert value["accuracy"] is None
    assert value["backends"]["cutie"]["internal_metrics"]["known_decision_coverage"] == 0.75
    assert value["cross_backend"]["common_known_masks"] == 3
    assert value["cross_backend"]["mask_iou_mean"] == 1.0
    event = value["backends"]["cutie"]["reentry_development_coverage"]["events"][0]
    assert event["candidate_known_coverage_in_predecessor_unobserved_region"] == 0.5


def test_failed_backend_is_bounded_no_go(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"rows": []}))
    result = tmp_path / "failed.json"
    result.write_text(json.dumps({"payload": {"backend": "cutie", "status": "FAILED_RUNTIME_FINAL", "execution_performed": False, "runtime_error": "boom"}}))
    value = build_comparison([result], audit_path)
    assert value["status"] == "NO_GO_BACKEND_EXECUTION"
    assert value["next_action"] == "FIX_EXECUTION_WITHIN_ATTEMPT_BUDGET"
