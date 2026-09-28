#!/usr/bin/env python3
"""Publish only receipt-backed V5 lane progress through the sole governance publisher.

Usage: python -m chaoyang.governance.publish_four_stream_v5_progress
       --expected-revision N
No algorithm success or product quality is inferred from process completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "four_stream_visual_delivery_v5"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"


def _maybe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"reference": artifact_ref(path), "result": load_json(path)}


def _lane(name: str, status: str, action: str, evidence: list[dict], blocker: str | None = None) -> None:
    path = ROOT / "lanes" / name / "STATE.json"
    old = load_json(path)
    latest = [entry["reference"] for entry in evidence]
    claims = {
        "algorithm_quality_pass": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "evidence_statuses": [entry["result"].get("status", "UNKNOWN") for entry in evidence],
    }
    if (old.get("status") == status and old.get("current_action") == action
            and old.get("blocker") == blocker and old.get("latest_artifacts") == latest
            and old.get("claims") == claims):
        return
    old.update(status=status, current_action=action, blocker=blocker, updated_at=now_iso(),
               latest_artifacts=latest, claims=claims)
    atomic_json(path, old)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("V5 is not current")
    sensor = _maybe(ROOT / "lanes/sensor/run_0001/RESULT.json")
    motion = [item for item in (
        _maybe(ROOT / "lanes/motion/recovered_007_v1/RESULT.json"),
        _maybe(ROOT / "lanes/motion/recovered_031_v1/RESULT.json"),
        _maybe(ROOT / "lanes/motion/recovered_0902_103_v1/RESULT.json"),
        _maybe(ROOT / "lanes/motion/recovered_0902_042_v1/RESULT.json"),
        _maybe(ROOT / "lanes/motion/LEFT_DIAGNOSTIC_031_V1.json"),
        _maybe(ROOT / "lanes/motion/MOTION_HANDOFF_V5.json"),
        _maybe(ROOT / "lanes/motion/review_0902_103_v1/RESULT.json"),
        _maybe(ROOT / "lanes/motion/review_0902_042_v1/RESULT.json")) if item]
    scene = [item for item in (
        _maybe(ROOT / "lanes/scene/review_007_v1/RESULT.json"),
        _maybe(ROOT / "lanes/scene/review_031_v1/RESULT.json"),
        _maybe(ROOT / "lanes/scene/review_103_v1/RESULT.json"),
        _maybe(ROOT / "lanes/scene/review_042_v1/RESULT.json"),
        _maybe(ROOT / "lanes/scene/GEOMETRY_READINESS.json"),
        _maybe(ROOT / "lanes/scene/diagnostic_007_v1/WRITE_QA_RESULT.json"),
        _maybe(ROOT / "lanes/scene/diagnostic_031_v1/WRITE_QA_RESULT.json")) if item]
    huro = [item for item in (
        _maybe(ROOT / "lanes/huro/canary_0001/RESULT.json"),
        _maybe(ROOT / "lanes/huro/full_0001/RESULT.json"),
        _maybe(ROOT / "lanes/huro/HURO_TERMINAL.json"),
        _maybe(ROOT / "lanes/huro/review_get_potato_chips_0915_007/RESULT.json"),
        _maybe(ROOT / "lanes/huro/review_play_cards_0915_031/RESULT.json")) if item]
    if sensor:
        _lane("sensor", "EXECUTED_PENDING_VISUAL_REVIEW", "Three 0916 sensor replays and HandMotion exported; QA not adopted.", [sensor])
    if motion:
        complete = (ROOT / "lanes/motion/MOTION_HANDOFF_V5.json").is_file()
        _lane("motion", "DIAGNOSTIC_COMPLETE_QUALITY_C" if complete else "DIAGNOSTIC_IN_PROGRESS",
              "Four same-domain motion packages and two full Raw R0 reviews; no product-quality pass." if complete
              else "Motion source and R0 inspection in progress.", motion,
              "031_LEFT_NOT_VISIBLE_OR_UNKNOWN_NO_LEGAL_MODEL_INPUT")
    if scene:
        complete = all((ROOT / f"lanes/scene/review_{sid}_v1/RESULT.json").is_file()
                       for sid in ("007", "031", "103", "042"))
        _lane("scene", "FAILED_QUALITY_C" if complete else "MASK_PRODUCTION_IN_PROGRESS",
              "Four full-session Mask reviews fail device/object protection; no product Clean/ProPainter adopted."
              if complete else "Four-session role Mask and independent QA in progress.", scene,
              "DEVICE_MASK_AND_TRUSTED_VISIBLE_OBJECT_PROTECTION_FAILED" if complete else None)
    if huro:
        complete = (ROOT / "lanes/huro/HURO_TERMINAL.json").is_file()
        _lane("huro", "FAILED_QUALITY_LIMITS" if complete else "CORE_TEST_IN_PROGRESS",
              "Two same-input full Raw comparisons completed; core result rejected by joint-limit gate."
              if complete else "Core wrist objective and fair comparison in progress.", huro,
              "PINNED_JOINT_LIMIT_VIOLATION" if complete else None)
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_V5_RECEIPT_BACKED_PROGRESS",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "PUBLISHED", "revision": published["governance_revision"],
                      "sensor": bool(sensor), "motion": bool(motion),
                      "scene": len(scene), "huro": len(huro)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
