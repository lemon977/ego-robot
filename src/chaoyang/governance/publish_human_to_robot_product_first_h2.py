"""Publish the next immutable product-first evidence checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_product_first_cleanup_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
H1 = ROOT / "PROGRESS_H1.json"
H2 = ROOT / "PROGRESS_H2.json"
CARD = ROOT / "lanes/scene/card_031/review_v1/RESULT.json"
FORMAL = ROOT / "lanes/motion_product/formal_007_current/attempt_0001/PRODUCT_RESULT.json"
RESUME = ROOT / "lanes/motion_product/formal_007_current/RESUME_RECEIPT.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    if H2.exists():
        raise FileExistsError(H2)
    previous = load_json(H1)
    card, formal, resume = (load_json(path) for path in (CARD, FORMAL, RESUME))
    if card.get("quality") != "REJECTED_QUALITY_CARD_EDGE_BLUR":
        raise RuntimeError("CARD_QUALITY_MISMATCH")
    if formal.get("session_id") != "get_potato_chips_0915_007" or formal.get("decoded_frames") != 378:
        raise RuntimeError("FORMAL_007_COVERAGE_MISMATCH")
    if formal.get("quality") != "REJECTED_QUALITY" or resume.get("quality") != "REJECTED_QUALITY":
        raise RuntimeError("PRODUCT_QUALITY_MISMATCH")
    old = REPO_ROOT / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_007/attempt_0002/PRODUCT_RESULT.json"
    old_product = load_json(old)
    if formal["product_video"]["sha256"] != old_product["product_video"]["sha256"]:
        raise RuntimeError("UNEXPECTED_NEW_PRODUCT_PIXEL_DIFFERENCE")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    # Keep the lane STATE bytes pinned by H1 immutable. New evidence is
    # appended in this checkpoint rather than invalidating H1's SHA refs.
    counts = {**previous["counts"], "031_card_fixed_window_rejected": True,
              "007_current_formal_full_frames": 378,
              "007_current_formal_resume_no_mutation": True,
              "007_formal_pixel_improvement": False}
    atomic_json(H2, {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_PROGRESS_V1",
        "task_id": TASK, "created_at": now_iso(),
        "checkpoint": "SECOND_EVIDENCE_WAVE_NOT_TERMINAL",
        "previous": artifact_ref(H1), "counts": counts,
        "new_evidence": {"card": artifact_ref(CARD), "formal_007": artifact_ref(FORMAL),
                         "resume_007": artifact_ref(RESUME), "old_product": artifact_ref(old)},
        "lane_states": {name: artifact_ref(ROOT / "lanes" / name / "STATE.json")
                        for name in ("scene", "motion_product", "sensor", "compare")},
        "authority": {"training_eligible": False, "control_ground_truth": False,
                      "physical_deployable": False, "external_metric_authority": False},
    })
    task = next(row for row in state["tasks"] if row.get("task_id") == TASK)
    task.update(status="PENDING", attempt=1, updated_at=now_iso(), heartbeat_at=None,
                pid=None, proc_start_ticks=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PENDING", "created_at": now_iso(),
        "message": "031 card fixed window rejected; current-signature 007 full candidate and resume validated, with identical visual SHA to old rejected product.",
    }])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_H2_PUBLISHED",
        expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "PASS", "revision": published["governance_revision"],
                      "progress": str(H2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
