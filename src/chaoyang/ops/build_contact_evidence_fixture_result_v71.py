from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Publish the deterministic V7.1 contact evidence and geometry fixture terminal."""

import hashlib
import json
from pathlib import Path

import jsonschema

from chaoyang.pipeline.object_contact_evidence_v1 import (
    validate_chips_three_instance_fixture,
    validate_evidence_dag,
    validate_poker_thin_card_fixture,
)
from chaoyang.governance.common import artifact_ref, canonical_bytes, sha256_file
from chaoyang.governance.v52_contracts import atomic_write_new


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact"
POKER = ROOT / "tests/fixtures/poker_thin_card_v71.json"
CHIPS = ROOT / "tests/fixtures/chips_three_instance_v71.json"
SCHEMA = ROOT / "contracts/object_contact_evidence_v1.schema.json"
MODULE = ROOT / "src/chaoyang/pipeline/object_contact_evidence_v1.py"


def nodes() -> list[dict[str, object]]:
    return [
        {
            "evidence_id": "direct-object6d-fixture",
            "evidence_type": "DIRECT_OBJECT6D",
            "parent_evidence_ids": [],
            "evidence_depth": 0,
            "may_support_contact_authority": True,
            "may_support_object6d_authority": True,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
        {
            "evidence_id": "tracked-fixture",
            "evidence_type": "BIDIRECTIONAL_TRACKED",
            "parent_evidence_ids": [],
            "evidence_depth": 0,
            "may_support_contact_authority": True,
            "may_support_object6d_authority": True,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
        {
            "evidence_id": "contact-seed-fixture",
            "evidence_type": "CONTACT_SEED",
            "parent_evidence_ids": ["direct-object6d-fixture"],
            "evidence_depth": 1,
            "may_support_contact_authority": True,
            "may_support_object6d_authority": False,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
        {
            "evidence_id": "attachment-fixture",
            "evidence_type": "HAND_OBJECT_ATTACHMENT",
            "parent_evidence_ids": ["contact-seed-fixture"],
            "evidence_depth": 2,
            "may_support_contact_authority": False,
            "may_support_object6d_authority": False,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
        {
            "evidence_id": "unknown-fixture",
            "evidence_type": "UNKNOWN",
            "parent_evidence_ids": [],
            "evidence_depth": 0,
            "may_support_contact_authority": False,
            "may_support_object6d_authority": False,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
    ]


def main() -> int:
    poker = json.loads(POKER.read_text(encoding="utf-8"))
    chips = json.loads(CHIPS.read_text(encoding="utf-8"))
    poker_regions = validate_poker_thin_card_fixture(poker)
    chips_ids = validate_chips_three_instance_fixture(chips)
    graph = validate_evidence_dag(nodes())
    input_refs = [artifact_ref(POKER), artifact_ref(CHIPS)]
    index = {
        "schema_version": "OBJECT_CONTACT_EVIDENCE_INDEX_V1",
        "artifact_id": "contact-evidence-fixtures-v71",
        "artifact_revision": "R7_1",
        "input_artifact_ids": ["poker-thin-card-v71", "chips-three-instance-v71"],
        "input_manifest_sha": hashlib.sha256(canonical_bytes(input_refs)).hexdigest(),
        "producer_signature": hashlib.sha256(
            (sha256_file(MODULE) + sha256_file(Path(__file__))).encode()
        ).hexdigest(),
        "supersedes_artifact_id": None,
        "validity": "VALID_FOR_PINNED_REVISION",
        "evidence_nodes": nodes(),
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(index)
    index_payload = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(OUTPUT / "OBJECT_CONTACT_EVIDENCE_INDEX.json", index_payload)
    result = {
        "schema_version": "chaoyang-contact-evidence-fixture-result-v71",
        "status": "PASSED",
        "task_id": "contact_evidence_dag_v1",
        "artifact_revision": "R7_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "checks": {
            "evidence_dag_acyclic": True,
            "topological_order": list(graph.topological_order),
            "attachment_may_support_contact_authority": False,
            "attachment_may_support_object6d_authority": False,
            "attachment_may_support_gold_accuracy": False,
            "attachment_may_upgrade_tactile": False,
            "poker_regions": sorted({item.value for item in poker_regions}),
            "chips_instance_ids": list(chips_ids),
            "chips_union_forbidden": True,
            "chips_deformation_degrades_unknown": True,
        },
        "inputs": input_refs,
        "evidence_index": artifact_ref(OUTPUT / "OBJECT_CONTACT_EVIDENCE_INDEX.json"),
        "claim_limit": "Deterministic geometry and provenance contract only; no real-session contact, Object6D, occlusion accuracy or physical truth.",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(OUTPUT / "CONTACT_EVIDENCE_RESULT.json", payload)
    atomic_write_new(OUTPUT / "RESULT.json", payload)
    print(OUTPUT / "RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
