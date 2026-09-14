import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import lr_distributed_side_evidence as d1
from tools import run_002_012_d4_contact_from_shared_inventory as runner


PROJECT = Path(__file__).parents[1]
ARTIFACT = PROJECT / "archive/legacy_runs/unclassified/cpu_b1_d4_production_wiring_20260830_v1"


def test_production_connector_passes_the_exact_inventory_once(monkeypatch, tmp_path):
    inventory = []
    geometry = {"left": object(), "right": object()}
    provider = SimpleNamespace(pair=lambda session, frame: geometry)
    authorities = {
        side: SimpleNamespace(session_id="grap_a_cap_012", frame_index=7)
        for side in runner.SIDES
    }
    calls = []
    sentinel = object()

    def fake_select(authority_arg, raw_arg, geometry_arg, **kwargs):
        calls.append((authority_arg, raw_arg, geometry_arg, kwargs))
        return sentinel

    monkeypatch.setattr(runner.d4, "select_frame", fake_select)
    result = runner.select_d1_then_d4(
        authorities,
        inventory,
        provider,
        evidence_root=tmp_path,
        image_shape=(960, 1280),
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][0] is authorities
    assert calls[0][1] is inventory
    assert calls[0][2] is geometry
    assert calls[0][3]["thresholds"] == d1.SelectorThresholds(0.20, 0.05, 0.12)


def test_connector_rejects_threshold_or_frame_specific_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.d4,
        "select_frame",
        lambda *args, **kwargs: pytest.fail("D4 must not run after preflight failure"),
    )
    provider = SimpleNamespace(pair=lambda session, frame: {})
    authorities = {
        "left": SimpleNamespace(session_id="grap_a_cap_012", frame_index=0),
        "right": SimpleNamespace(session_id="grap_a_cap_012", frame_index=0),
    }
    with pytest.raises(runner.D4WiringError, match="thresholds changed"):
        runner.select_d1_then_d4(
            authorities,
            [],
            provider,
            evidence_root=tmp_path,
            image_shape=(960, 1280),
            thresholds=d1.SelectorThresholds(0.20, 0.05, 0.1200001),
        )
    authorities["right"] = SimpleNamespace(
        session_id="grap_a_cap_012", frame_index=1
    )
    with pytest.raises(runner.D4WiringError, match="cross-frame"):
        runner.select_d1_then_d4(
            authorities,
            [],
            provider,
            evidence_root=tmp_path,
            image_shape=(960, 1280),
        )


def test_decision_audit_record_preserves_rejected_candidate_taxonomy():
    candidate = d1.CandidateAudit(
        raw_source_sha256="a" * 64,
        raw_instance_offset=2,
        instance_id=7,
        assigned_pool="left",
        considered_for_side=True,
        eligible=False,
        rejection_reasons=("OBJECT6D_PROTECTION_REJECTION",),
        in_image_joint_count=21,
        supported_in_image_joint_count=17,
        joint_support_ratio=17 / 21,
        non_target_joint_support_ratio=0.0,
        side_difference=17 / 21,
        wrist_supported_diagnostic=True,
        authority_wrist_outside_image_diagnostic=False,
        boundary_supported_diagnostic=True,
        object6d_overlap_over_instance=0.5,
    )
    decision = d1.SideDecision(
        identity="left",
        authority_lineage_id="left-lineage",
        authority_state="AVAILABLE",
        status="REJECT",
        raw_instance_offset=None,
        instance_id=None,
        mask=None,
        failure_reason="OBJECT6D_PROTECTION_REJECTION",
        authority_wrist_outside_image_diagnostic=False,
        candidate_audit=(candidate,),
    )
    record = runner.decision_audit_record(decision)
    assert record["status"] == "REJECT"
    assert record["failure_reason"] == "OBJECT6D_PROTECTION_REJECTION"
    assert record["selected_mask_area"] is None
    assert record["candidate_audit"][0]["rejection_reasons"] == [
        "OBJECT6D_PROTECTION_REJECTION"
    ]


def test_decision_audit_record_rejects_cross_side_application():
    decision = d1.SideDecision(
        identity="left",
        authority_lineage_id="left-lineage",
        authority_state="AVAILABLE",
        status="ACCEPT",
        raw_instance_offset=0,
        instance_id=1,
        mask=np.ones((2, 2), dtype=np.bool_),
        failure_reason=None,
        authority_wrist_outside_image_diagnostic=False,
        candidate_audit=tuple(),
    )
    application = SimpleNamespace(identity="right")
    with pytest.raises(runner.D4WiringError, match="side mismatch"):
        runner.decision_audit_record(decision, application)


def test_frozen_geometry_provider_uses_left_axis0_right_axis1():
    value, _ = runner.artifact_io.read_json_nofollow(
        runner.DEFAULT_COUNTERFACTUAL, allowed_root=PROJECT
    )
    provider = runner.FrozenContactGeometryProvider.from_counterfactual(
        value, allowed_root=PROJECT
    )
    pair = provider.pair("grap_a_cap_012", 75)
    assert pair["left"].physical_hand_axis == 0
    assert pair["right"].physical_hand_axis == 1
    assert pair["left"].hawor_source_sha256 == pair["right"].hawor_source_sha256
    assert pair["left"].object6d_source_sha256 == pair["right"].object6d_source_sha256
    assert pair["left"].cylinder_source_sha256 == pair["right"].cylinder_source_sha256


def test_real_cpu_artifact_has_all_rows_and_expected_transitions():
    manifest = json.loads((ARTIFACT / "RUN_MANIFEST.json").read_bytes())
    assert manifest["status"] == "ARTIFACT_EXISTS"
    assert manifest["population"]["frame_count"] == 757
    assert manifest["population"]["side_row_count"] == 1514
    observed = {
        (row["session_id"], row["side"]): (
            row["d1_accept"],
            row["strict_negative_flips"],
            row["d4_accept"],
            row["object_only_rejections_retained"],
        )
        for row in manifest["population"]["summary"]
    }
    assert observed == runner.EXPECTED_TRANSITIONS
    assert manifest["semantics"]["contact_rule"] == "minimum signed distance < 0"
    assert manifest["semantics"]["exact_zero_is_contact"] is False
    assert manifest["shared_raw_inventory"]["second_inference"] is False

    with (ARTIFACT / "D4_PER_FRAME_SIDE_DECISIONS.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1514
    assert sum(
        row["d4_application"] == "APPLIED_OBJECT_OVERLAP_AS_CONTACT_EVIDENCE"
        for row in rows
    ) == 180
    assert all(row["model_inference_calls"] == "0" for row in rows)


def test_overlap_counts_answer_zero_means_no_high_overlap_inventory():
    value = json.loads(
        (ARTIFACT / "D4_OVERLAP_INSTANCE_FRAME_COUNTS.json").read_bytes()
    )["counts"]
    for side in runner.SIDES:
        assert value["grap_a_cap_002"]["by_side"][side] == {
            "d1_routed_high_overlap_frame_count": 0,
            "d1_routed_high_overlap_instance_count": 0,
            "d4_exact_object_only_candidate_frame_count": 0,
            "d4_exact_object_only_candidate_instance_count": 0,
        }
    assert value["grap_a_cap_002"]["shared_inventory_high_overlap"] == {
        "frame_count": 0,
        "instance_count": 0,
    }
    assert value["grap_a_cap_012"]["by_side"]["right"] == {
        "d1_routed_high_overlap_frame_count": 0,
        "d1_routed_high_overlap_instance_count": 0,
        "d4_exact_object_only_candidate_frame_count": 0,
        "d4_exact_object_only_candidate_instance_count": 0,
    }
