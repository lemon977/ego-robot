from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from chaoyang.pipeline.object_contact_evidence_v1 import (
    CardRegion,
    EvidenceGraphError,
    classify_poker_card_local_point,
    validate_chips_three_instance_fixture,
    validate_evidence_dag,
    validate_poker_thin_card_fixture,
)


PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT / "tests" / "fixtures"


def node(
    evidence_id: str,
    evidence_type: str,
    parents: list[str],
    depth: int,
    capabilities: tuple[bool, bool, bool, bool],
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "parent_evidence_ids": parents,
        "evidence_depth": depth,
        "may_support_contact_authority": capabilities[0],
        "may_support_object6d_authority": capabilities[1],
        "may_support_gold_contact_accuracy": capabilities[2],
        "may_upgrade_tactile_supported_contact": capabilities[3],
    }


def legal_graph() -> list[dict[str, object]]:
    return [
        node("attachment", "HAND_OBJECT_ATTACHMENT", ["seed"], 2, (False, False, False, False)),
        node("direct", "DIRECT_OBJECT6D", [], 0, (True, True, False, False)),
        node("unknown", "UNKNOWN", [], 0, (False, False, False, False)),
        node("tracked", "BIDIRECTIONAL_TRACKED", [], 0, (True, True, False, False)),
        node("seed", "CONTACT_SEED", ["tracked"], 1, (True, False, False, False)),
    ]


def test_schema_is_valid_and_accepts_immutable_index() -> None:
    schema = json.loads(
        (PROJECT / "contracts" / "object_contact_evidence_v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    index = {
        "schema_version": "OBJECT_CONTACT_EVIDENCE_INDEX_V1",
        "artifact_id": "object-contact-fixture-v1",
        "artifact_revision": "R7_1",
        "input_artifact_ids": ["direct-object6d-r7-0"],
        "input_manifest_sha": "0" * 64,
        "producer_signature": "1" * 64,
        "supersedes_artifact_id": None,
        "validity": "VALID_FOR_PINNED_REVISION",
        "evidence_nodes": legal_graph(),
    }
    jsonschema.Draft202012Validator(schema).validate(index)


def test_legal_graph_has_deterministic_parent_before_child_order() -> None:
    graph = validate_evidence_dag(legal_graph())
    order = graph.topological_order
    assert order.index("tracked") < order.index("seed") < order.index("attachment")


def test_cycle_and_depth_mismatch_fail_closed() -> None:
    graph = legal_graph()
    graph[-1]["parent_evidence_ids"] = ["attachment"]
    with pytest.raises(EvidenceGraphError, match="cycle detected"):
        validate_evidence_dag(graph)

    graph = legal_graph()
    graph[-1]["evidence_depth"] = 8
    with pytest.raises(EvidenceGraphError, match="evidence_depth"):
        validate_evidence_dag(graph)


def test_only_direct_or_tracked_evidence_can_seed_contact() -> None:
    graph = legal_graph()
    graph[-1]["parent_evidence_ids"] = ["unknown"]
    graph[-1]["evidence_depth"] = 1
    with pytest.raises(EvidenceGraphError, match="illegal parent type"):
        validate_evidence_dag(graph)


@pytest.mark.parametrize("capability_index", range(4))
def test_attachment_cannot_escalate_any_authority(capability_index: int) -> None:
    graph = legal_graph()
    capability_keys = (
        "may_support_contact_authority",
        "may_support_object6d_authority",
        "may_support_gold_contact_accuracy",
        "may_upgrade_tactile_supported_contact",
    )
    graph[0][capability_keys[capability_index]] = True
    with pytest.raises(EvidenceGraphError, match="capability escalation"):
        validate_evidence_dag(graph)


def test_poker_fixture_covers_two_surfaces_edge_and_penetration() -> None:
    fixture = json.loads((FIXTURES / "poker_thin_card_v71.json").read_text())
    regions = validate_poker_thin_card_fixture(fixture)
    assert set(regions) == {
        CardRegion.FRONT,
        CardRegion.BACK,
        CardRegion.EDGE,
        CardRegion.PENETRATION,
    }
    assert classify_poker_card_local_point([0.1, 0, 0], fixture["dimensions_m"]) is CardRegion.OUTSIDE


def test_poker_fixture_rejects_single_surface_or_thick_card() -> None:
    fixture = json.loads((FIXTURES / "poker_thin_card_v71.json").read_text())
    fixture["two_sided"] = False
    with pytest.raises(EvidenceGraphError, match="two-sided"):
        validate_poker_thin_card_fixture(fixture)
    fixture = json.loads((FIXTURES / "poker_thin_card_v71.json").read_text())
    fixture["dimensions_m"] = [0.088, 0.063, 0.01]
    with pytest.raises(EvidenceGraphError, match="thin-card"):
        validate_poker_thin_card_fixture(fixture)


def test_chips_fixture_preserves_three_instances_and_degrades_deformation() -> None:
    fixture = json.loads((FIXTURES / "chips_three_instance_v71.json").read_text())
    assert validate_chips_three_instance_fixture(fixture) == (
        "chips_left",
        "chips_center",
        "chips_right",
    )


def test_chips_fixture_rejects_union_and_rigid_pose_for_deformation() -> None:
    fixture = json.loads((FIXTURES / "chips_three_instance_v71.json").read_text())
    fixture["instance_ids"] = ["chips_union"]
    with pytest.raises(EvidenceGraphError, match="exactly three"):
        validate_chips_three_instance_fixture(fixture)

    fixture = json.loads((FIXTURES / "chips_three_instance_v71.json").read_text())
    bad = copy.deepcopy(fixture)
    bad["frames"][1]["instances"][1]["evidence_type"] = "BIDIRECTIONAL_TRACKED"
    with pytest.raises(EvidenceGraphError, match="degrade to UNKNOWN"):
        validate_chips_three_instance_fixture(bad)
