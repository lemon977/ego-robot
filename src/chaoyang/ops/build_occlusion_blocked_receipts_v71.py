#!/usr/bin/env python3
"""Publish immutable fail-closed V7.1 occlusion prerequisite receipts.

These receipts deliberately contain no accuracy.  They are replaced only by a
new immutable artifact revision after causal Robotized candidates or an
independent frozen goldset actually exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"no-clobber conflict: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    contact = RUN / "contact/RESULT.json"
    plan = ROOT / "docs/governance/PLAN_REVISION.json"
    common = {
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "generated_at": generated_at,
        "inputs": [
            {"path": str(contact), "bytes": contact.stat().st_size, "sha256": _sha(contact)},
            {"path": str(plan), "bytes": plan.stat().st_size, "sha256": _sha(plan)},
        ],
        "accuracy_reported": False,
        "physical_accuracy_claimed": False,
    }
    silver_dir = RUN / "occlusion_silver"
    silver = {
        "schema_version": "OCCLUSION_SILVER_V1_BLOCKED_RECEIPT",
        **common,
        "task_id": "occlusion_silver_v1",
        "status": "BLOCKED_PREREQ",
        "blocker": "NO_CAUSAL_ROBOTIZED_CANDIDATE_WITH_PIXEL_PROVENANCE",
        "resolved_contract_tests": {
            "silver_separate_from_gold": True,
            "unknown_is_training_invalid": True,
            "clean_background_cannot_supply_hidden_object": True,
        },
        "resume_condition": "Publish causal Robotized candidate with robot/object z-buffer and pixel provenance.",
        "claim_limit": "Contract evidence only; no session Silver authority and no accuracy.",
    }
    _publish(silver_dir / "OCCLUSION_SILVER_REPORT.json", silver)
    _publish(silver_dir / "RESULT.json", silver)

    gold_dir = RUN / "occlusion_gold"
    gold = {
        "schema_version": "CONTACT_OCCLUSION_GOLDSET_V1_BLOCKED_RECEIPT",
        **common,
        "task_id": "occlusion_goldset_v1",
        "status": "BLOCKED_PREREQ",
        "blocker": "INDEPENDENT_240_FRAME_DOUBLE_REVIEWED_GOLDSET_ABSENT",
        "required": {
            "poker_frames": 120,
            "chips_frames": 120,
            "minimum_sessions": 4,
            "independent_reviewers": 2,
            "freeze_before_scoring": True,
        },
        "resume_condition": "Freeze independent labels before algorithm scoring and resolve reviewer disagreements.",
        "claim_limit": "No gold accuracy exists; Silver evidence cannot substitute for independent labels.",
    }
    _publish(gold_dir / "CONTACT_OCCLUSION_GOLDSET_V1.json", gold)
    _publish(gold_dir / "RESULT.json", gold)
    print(json.dumps({"silver": str(silver_dir / 'RESULT.json'), "gold": str(gold_dir / 'RESULT.json')}))


if __name__ == "__main__":
    main()
