"""Append an immutable write-scope amendment for the registered quality task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "human_to_robot_quality_acceptance_20260924"
OLD = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
NEW = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET_V2.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ADDED = (
    "src/chaoyang/ops/run_human_to_robot_baseline_v1.py",
    "src/chaoyang/ops/run_r2_scene_candidate.py",
    "src/chaoyang/pipeline/v5_scene.py",
    "src/chaoyang/governance/amend_human_to_robot_quality_acceptance_packet.py",
    "src/chaoyang/governance/publish_human_to_robot_quality_acceptance_impl.py",
    "tests/unit/test_product_scene_gate_quality.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task", {}).get("task_id") != TASK or NEW.exists():
        raise RuntimeError("TASK_NOT_ACTIVE_OR_ALREADY_AMENDED")
    old = load_json(OLD)
    row = next(item for item in index["task_packets"] if item["task_id"] == TASK)
    if row["packet_sha256"] != artifact_ref(OLD)["sha256"]:
        raise RuntimeError("OLD_PACKET_SHA_DRIFT")
    packet = dict(old)
    packet["write_set"] = sorted(set(old["write_set"]) | set(ADDED))
    packet["supersedes_packet"] = artifact_ref(OLD)
    packet["scope_amendment_reason"] = "Formal product gate and deterministic Scene producer tests are in the authorized quality-acceptance plan."
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_AMENDMENT:" + ";".join(errors))
    atomic_json(NEW, packet)
    ref = artifact_ref(NEW)
    row.update(packet_path=str(NEW.relative_to(REPO_ROOT)), packet_sha256=ref["sha256"],
               supersedes_packet=artifact_ref(OLD))
    state["tasks"][-1]["task_packet"] = ref
    index["supersedes_index"] = artifact_ref(INDEX)
    index["packet_revision"] = "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_SCOPE_V2"
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_PACKET_AMENDED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    print(json.dumps({"status": "AMENDED", "governance_revision": published["governance_revision"],
                      "packet": str(NEW)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
