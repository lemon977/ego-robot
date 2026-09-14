from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pipeline import lr_distributed_side_evidence as selector
from pipeline.lr_candidate_audit_observability_v1 import (
    CandidateAuditObservabilityError,
    build_pre_enumeration_record,
    decision_sha256,
    select_frame_observed,
    write_audit_exclusive,
)


PROJECT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = PROJECT / "pipeline/lr_distributed_side_evidence.py"


def ref(root: Path, path: str, value: object, kind: str) -> selector.EvidenceRef:
    data = selector.canonical_json(value)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return selector.EvidenceRef(str(target), len(data), hashlib.sha256(data).hexdigest(), kind)


def authorities(root: Path, *, left_state="AVAILABLE", same_lineage=False, left_wrist=(5.0, 5.0)):
    payload = {
        "frame_index": 12,
        "sides": {
            "left": {"authority": {"identity": "left", "lineage_id": "same" if same_lineage else "L", "state": left_state, "wrist_xy": list(left_wrist) if left_wrist else None}},
            "right": {"authority": {"identity": "right", "lineage_id": "same" if same_lineage else "R", "state": "AVAILABLE", "wrist_xy": [25.0, 5.0]}},
        },
    }
    evidence = ref(root, "authority.json", payload, "A_PRIME_V3_FRAME_SELECTION")
    return {
        "left": selector.SideAuthority("left", "grap_a_cap_004", 12, "same" if same_lineage else "L", left_state, left_wrist, evidence),
        "right": selector.SideAuthority("right", "grap_a_cap_004", 12, "same" if same_lineage else "R", "AVAILABLE", (25.0, 5.0), evidence),
    }


def support(count: int, *, wrist=True):
    return selector.JointSupportEvidence(10, count, count / 10, 2.0, wrist)


def raw(root: Path, offset: int, *, left=8, right=0, overlap=0.0, wrist=False, boundary=False):
    mask = np.zeros((24, 32), dtype=np.bool_)
    mask[3 + offset:12 + offset, 4:18] = True
    payload = {"source_kind": "SAM_RAW_INSTANCE", "frame_id": "grap_a_cap_004:12", "instance_id": 100 + offset, "mask_sha256": selector.mask_sha256(mask)}
    evidence = ref(root, f"raw_{offset}.json", payload, "SAM_RAW_INSTANCE")
    return selector.RawInstance(
        "grap_a_cap_004", 12, offset, 100 + offset, evidence, mask, 0.9 - offset * 0.01,
        {"left": support(left, wrist=wrist), "right": support(right, wrist=wrist)},
        {"left": boundary, "right": boundary}, overlap, float(mask.mean()),
    )


def observe(root: Path, auth, raws):
    return select_frame_observed(auth, raws, evidence_root=root, image_shape=(24, 32), selector_path=SELECTOR_PATH)


def test_old_vs_observed_decision_is_bit_exact_counterexample(tmp_path: Path):
    auth = authorities(tmp_path)
    raws = (raw(tmp_path, 0, left=8), raw(tmp_path, 1, left=7))
    old = selector.select_frame(auth, raws, evidence_root=tmp_path, image_shape=(24, 32))
    new = observe(tmp_path, auth, raws)
    assert decision_sha256(old) == decision_sha256(new.selection)
    assert new.audit["semantic_changes"] == {"prompt": 0, "threshold": 0, "selection": 0, "pixels": 0, "status": 0}


def test_all_rejected_still_persists_every_raw_candidate(tmp_path: Path):
    auth = authorities(tmp_path)
    raws = (raw(tmp_path, 0, left=1, right=0), raw(tmp_path, 1, left=8, right=0, overlap=0.5))
    out = observe(tmp_path, auth, raws)
    assert out.selection.left.status == "REJECT"
    assert out.audit["stage"] == "CANDIDATES_ENUMERATED"
    assert out.audit["candidate_count"] == 2
    assert len(out.audit["candidate_audit"]) == 2
    for item in out.audit["candidate_audit"]:
        left = item["side_audit"]["left"]
        assert set(left) >= {"joint_support_ratio", "side_margin", "object6d_overlap_over_instance", "wrist_supported_diagnostic", "boundary_supported_diagnostic", "final_veto_items"}
        assert left["final_veto_items"]


def test_oob_is_recorded_but_does_not_change_decision(tmp_path: Path):
    auth = authorities(tmp_path, left_state="OUTSIDE_IMAGE", left_wrist=(5.0, 30.0))
    raws = (raw(tmp_path, 0, left=8), raw(tmp_path, 1, left=0, right=8))
    out = observe(tmp_path, auth, raws)
    assert out.selection.left.status == out.selection.right.status == "ACCEPT"
    assert all(item["side_audit"]["left"]["authority_wrist_outside_image_diagnostic"] for item in out.audit["candidate_audit"])
    assert out.audit["stage"] == "CANDIDATES_ENUMERATED"


def test_missing_authority_after_enumeration_gets_explicit_veto(tmp_path: Path):
    auth = authorities(tmp_path, left_state="MISSING", left_wrist=None)
    out = observe(tmp_path, auth, (raw(tmp_path, 0, left=0, right=8),))
    assert out.selection.left.status == "HOLD"
    record = out.audit["candidate_audit"][0]["side_audit"]["left"]
    assert record["final_veto_items"] == ["SIDE_PRECHECK_HOLD:SIDE_AUTHORITY_EVIDENCE_MISSING"]


def test_identity_abort_is_explicitly_pre_enumeration(tmp_path: Path):
    out = observe(tmp_path, authorities(tmp_path, same_lineage=True), (raw(tmp_path, 0),))
    assert out.selection.frame_status == "IDENTITY_HOLD"
    assert out.audit["stage"] == "NOT_SCORED_BEFORE_CANDIDATE_ENUMERATION"
    assert out.audit["candidate_count"] == 0 and out.audit["candidate_audit"] == []


def test_manual_pre_enumeration_abort_and_access_zero():
    record = build_pre_enumeration_record("grap_a_cap_004", 12, abort_reason="RAW_ENUMERATION_FAILED")
    assert record["stage"] == "NOT_SCORED_BEFORE_CANDIDATE_ENUMERATION"
    assert all(value == 0 for value in record["access"].values())


def test_o_excl_and_symlink_rejection(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    record = build_pre_enumeration_record("grap_a_cap_004", 12, abort_reason="X")
    target = output / "audit.json"
    write_audit_exclusive(target, output, record)
    with pytest.raises(CandidateAuditObservabilityError, match="exclusive"):
        write_audit_exclusive(target, output, record)
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(CandidateAuditObservabilityError, match="root invalid"):
        write_audit_exclusive(linked / "x.json", linked, record)


def test_selector_pin_drift_fails_before_selection(tmp_path: Path):
    fake = tmp_path / "selector.py"
    fake.write_text("# drift\n")
    with pytest.raises(CandidateAuditObservabilityError, match="SHA drift"):
        select_frame_observed(authorities(tmp_path), (), evidence_root=tmp_path, image_shape=(24, 32), selector_path=fake)
