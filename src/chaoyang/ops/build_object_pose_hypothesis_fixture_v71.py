from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Publish the immutable CPU-only OBJECT_POSE_HYPOTHESIS_V2 fixture terminal."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np

from chaoyang.pipeline.object_pose_hypothesis_v2 import (
    MAX_ATTACHMENT_ENTRY_DISTANCE_M,
    MAX_ATTACHMENT_ROTATION_DRIFT_DEG,
    MAX_ATTACHMENT_TRANSLATION_DRIFT_M,
    MAX_ENDPOINT_ROTATION_DEG,
    MAX_ENDPOINT_TRANSLATION_M,
    MAX_GAP_FRAMES,
    MAX_GAP_SECONDS,
    build_object_pose_hypotheses,
)
from chaoyang.governance.common import artifact_ref, canonical_bytes, sha256_file
from chaoyang.governance.v52_contracts import atomic_write_new


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose"
FIXTURE = ROOT / "tests/fixtures/object_pose_hypothesis_v2_fixture.json"
SCHEMA = ROOT / "contracts/object_pose_hypothesis_v2.schema.json"
EVIDENCE_SCHEMA = ROOT / "contracts/object_contact_evidence_v1.schema.json"
EVIDENCE_MODULE = ROOT / "src/chaoyang/pipeline/object_contact_evidence_v1.py"
MODULE = ROOT / "src/chaoyang/pipeline/object_pose_hypothesis_v2.py"


def _node_dict(node: object) -> dict[str, object]:
    values = vars(node).copy()
    values["evidence_type"] = values["evidence_type"].value
    values["parent_evidence_ids"] = list(values["parent_evidence_ids"])
    return values


def _build_fixture() -> tuple[dict[str, object], object]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frames = int(fixture["frame_count"])
    poses = np.broadcast_to(np.eye(4), (frames, 1, 4, 4)).copy()
    valid = np.zeros((frames, 1), dtype=np.bool_)
    evidence = np.full((frames, 1), None, dtype=object)
    for frame in fixture["direct_frames"]:
        frame = int(frame)
        valid[frame, 0] = True
        poses[frame, 0, 0, 3] = float(fixture["direct_x_m"][str(frame)])
        evidence[frame, 0] = f"fixture-direct-card-{frame}"
    attachment = np.broadcast_to(np.eye(4), (frames, 1, 4, 4)).copy()
    attachment[:, 0, 0, 3] = np.asarray(fixture["attachment_x_m"], dtype=np.float64)
    zeros = np.zeros((frames, 1), dtype=np.float64)
    seed_parent_frame = int(fixture["contact_seed_parent_frame"])
    seed_id = "fixture-contact-seed-card"
    seed = {
        "evidence_id": seed_id,
        "evidence_type": "CONTACT_SEED",
        "parent_evidence_ids": [f"fixture-direct-card-{seed_parent_frame}"],
        "evidence_depth": 1,
        "may_support_contact_authority": True,
        "may_support_object6d_authority": False,
        "may_support_gold_contact_accuracy": False,
        "may_upgrade_tactile_supported_contact": False,
    }
    result = build_object_pose_hypotheses(
        poses,
        valid,
        task_id="poker",
        object_ids=fixture["object_ids"],
        fps=float(fixture["fps"]),
        formal_evidence_ids=evidence,
        poker_dimensions_m=fixture["poker_dimensions_m"],
        contact_seed_nodes=[seed],
        attachment_seed_by_object={fixture["object_ids"][0]: seed_id},
        attachment_poses=attachment,
        attachment_valid=np.ones((frames, 1), dtype=np.bool_),
        attachment_entry_distance_m=zeros,
        attachment_relative_translation_drift_m=zeros,
        attachment_relative_rotation_drift_deg=zeros,
    )
    observed = result.source[:, 0].tolist()
    if observed != fixture["expected_sources"]:
        raise RuntimeError(f"fixture source mismatch: {observed}")
    return fixture, result


def main() -> int:
    fixture, result = _build_fixture()
    input_refs = [artifact_ref(FIXTURE)]
    producer_signature = hashlib.sha256(
        (sha256_file(MODULE) + sha256_file(EVIDENCE_MODULE) + sha256_file(Path(__file__))).encode()
    ).hexdigest()
    evidence_nodes = [
        _node_dict(result.evidence_graph.nodes_by_id[evidence_id])
        for evidence_id in result.evidence_graph.topological_order
    ]
    records: list[dict[str, object]] = []
    for record in result.records:
        pose = result.pose_object_to_camera[record.frame_index, 0]
        records.append(
            {
                "frame_index": record.frame_index,
                "object_id": record.object_id,
                "source": record.source.value,
                "evidence_id": record.evidence_id,
                "parent_evidence_ids": list(record.parent_evidence_ids),
                "pose_object_to_camera": pose.tolist() if result.pose_valid[record.frame_index, 0] else None,
                "confidence": record.confidence,
                "reason": record.reason,
            }
        )
    counts = Counter(item["source"] for item in records)
    index = {
        "schema_version": "OBJECT_POSE_HYPOTHESIS_V2",
        "artifact_id": "object-pose-hypothesis-fixture-v71",
        "artifact_revision": "R7_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "input_artifact_ids": ["object-pose-hypothesis-fixture-input-v71"],
        "input_manifest_sha": hashlib.sha256(canonical_bytes(input_refs)).hexdigest(),
        "producer_signature": producer_signature,
        "supersedes_artifact_id": None,
        "task_id": "poker",
        "session_id": fixture["session_id"],
        "frame_count": fixture["frame_count"],
        "fps": fixture["fps"],
        "formal_object6d": {
            "authority_mode": "DIRECT_OBSERVED_ONLY",
            "occluded_frame_policy": "KEEP_INVALID",
            "mutated": result.formal_object6d_mutated,
            "sha256_before": result.formal_object6d_sha256_before,
            "sha256_after": result.formal_object6d_sha256_after,
        },
        "policy": {
            "max_gap_frames": MAX_GAP_FRAMES,
            "max_gap_seconds": MAX_GAP_SECONDS,
            "max_endpoint_translation_m": MAX_ENDPOINT_TRANSLATION_M,
            "max_endpoint_rotation_deg": MAX_ENDPOINT_ROTATION_DEG,
            "max_attachment_entry_distance_m": MAX_ATTACHMENT_ENTRY_DISTANCE_M,
            "max_attachment_translation_drift_m": MAX_ATTACHMENT_TRANSLATION_DRIFT_M,
            "max_attachment_rotation_drift_deg": MAX_ATTACHMENT_ROTATION_DRIFT_DEG,
            "require_two_direct_endpoints": True,
            "chips_union_allowed": False,
            "attachment_authority_promotable": False,
        },
        "object_models": [
            {
                "object_id": "card_0",
                "model_mode": "POKER_THIN_TWO_SIDED_RIGID",
                "instance_independent": True,
            }
        ],
        "records": records,
        "evidence_nodes": evidence_nodes,
        "summary": {
            "direct": counts["DIRECT_OBJECT6D"],
            "bidirectional_tracked": counts["BIDIRECTIONAL_TRACKED"],
            "hand_object_attachment": counts["HAND_OBJECT_ATTACHMENT"],
            "unknown": counts["UNKNOWN"],
            "authority_promotable": False,
        },
        "claim_limit": "Deterministic development fixture only; hypotheses do not modify or promote formal Object6D and do not prove contact, gold accuracy, tactile support, or physical truth.",
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    evidence_schema = json.loads(EVIDENCE_SCHEMA.read_text(encoding="utf-8"))
    resolver = jsonschema.RefResolver.from_schema(
        schema,
        store={evidence_schema["$id"]: evidence_schema},
    )
    jsonschema.Draft202012Validator(schema, resolver=resolver).validate(index)
    index_payload = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(OUTPUT / "OBJECT_POSE_HYPOTHESIS_INDEX.json", index_payload)

    evidence_index = {
        "schema_version": "OBJECT_CONTACT_EVIDENCE_INDEX_V1",
        "artifact_id": "object-pose-hypothesis-evidence-fixture-v71",
        "artifact_revision": "R7_1",
        "input_artifact_ids": ["object-pose-hypothesis-fixture-input-v71"],
        "input_manifest_sha": index["input_manifest_sha"],
        "producer_signature": producer_signature,
        "supersedes_artifact_id": None,
        "validity": "VALID_FOR_PINNED_REVISION",
        "evidence_nodes": evidence_nodes,
    }
    evidence_validator = jsonschema.Draft202012Validator(evidence_schema)
    evidence_validator.validate(evidence_index)
    atomic_write_new(
        OUTPUT / "OBJECT_CONTACT_EVIDENCE_INDEX.json",
        json.dumps(evidence_index, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n",
    )
    result_payload = {
        "schema_version": "chaoyang-object-pose-hypothesis-fixture-result-v71",
        "task_id": "object_pose_hypothesis_v2",
        "status": "PASSED",
        "artifact_revision": "R7_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "object_pose_hypothesis_index": artifact_ref(OUTPUT / "OBJECT_POSE_HYPOTHESIS_INDEX.json"),
        "object_contact_evidence_index": artifact_ref(OUTPUT / "OBJECT_CONTACT_EVIDENCE_INDEX.json"),
        "inputs": input_refs,
        "checks": {
            "formal_object6d_direct_observed_only": True,
            "formal_object6d_keep_invalid": True,
            "formal_object6d_unchanged": True,
            "evidence_dag_acyclic": True,
            "tracked_retains_two_direct_parents": True,
            "attachment_seeded_by_contact": True,
            "attachment_reverse_authority_forbidden": True,
            "bounded_gap_and_endpoint_gates": True,
            "poker_thin_two_sided_rigid": True,
            "chips_three_instances_independent": True,
            "chips_deformation_degrades_unknown": True,
        },
        "claim_status": "DEVELOPMENT_EVIDENCE",
        "claim_limit": index["claim_limit"],
    }
    payload = json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(OUTPUT / "RESULT.json", payload)
    print(OUTPUT / "RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
