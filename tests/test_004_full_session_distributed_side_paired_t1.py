import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


paired = _load(
    "task28_full460_paired_test_module",
    "tools/run_004_full_session_distributed_side_paired_t1.py",
)
selector = _load("task28_selector_test_module", "pipeline/lr_distributed_side_evidence.py")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _frame_source(
    root: Path,
    *,
    frame: int = 7,
    left_state: str = "AVAILABLE",
    left_wrist=(2.0, 2.0),
    right_state: str = "MISSING",
    right_wrist=None,
):
    path = root / "frame.json"
    payload = {
        "frame_index": frame,
        "sides": {
            "left": {
                "authority": {
                    "identity": "left",
                    "lineage_id": "session:left",
                    "state": left_state,
                    "wrist_xy": None if left_wrist is None else list(left_wrist),
                }
            },
            "right": {
                "authority": {
                    "identity": "right",
                    "lineage_id": "session:right",
                    "state": right_state,
                    "wrist_xy": None if right_wrist is None else list(right_wrist),
                }
            },
        },
    }
    _write_json(path, payload)
    ref = paired.identity(path)
    evidence = selector.EvidenceRef(
        ref["path"], ref["bytes"], ref["sha256"], "A_PRIME_V3_FRAME_SELECTION"
    )
    return path, evidence


def _authority(evidence, side: str, state: str, wrist, frame: int = 7):
    return selector.SideAuthority(
        side,
        "session",
        frame,
        f"session:{side}",
        state,
        wrist,
        evidence,
        True,
    )


def _support(ratio: float, *, wrist: bool, total: int = 10):
    supported = int(round(ratio * total))
    return selector.JointSupportEvidence(total, supported, supported / total, 1.0, wrist)


def _raw(root: Path, mask: np.ndarray, left, right, *, frame: int = 7):
    payload = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": f"session:{frame}",
        "instance_id": 3,
        "mask_sha256": selector.mask_sha256(mask),
    }
    path = root / "raw.json"
    _write_json(path, payload)
    ref = paired.identity(path)
    evidence = selector.EvidenceRef(
        ref["path"], ref["bytes"], ref["sha256"], "SAM_RAW_INSTANCE"
    )
    return selector.RawInstance(
        "session",
        frame,
        0,
        3,
        evidence,
        mask,
        0.8,
        {"left": left, "right": right},
        {"left": False, "right": False},
        0.0,
        float(mask.mean()),
    )


def test_exact_task26_and_task27_pair_identities_are_frozen():
    assert paired.TASK_SHA == "e3b92340b207e0117109775af6c2ede0566279dab28fcfef4035686d50a5c7d0"
    assert paired.SELECTOR_SHA == "bca51f866de925a2e795c5257540abc9c06fb8d298f26dd049196fda14fb9ff3"
    assert paired.TASK26_TERMINAL_SHA == "7ecee41b95716c81997830923a616b91f0bbc6a626aaa3f4786598907408c040"
    assert paired.BASE_MANIFEST_SHA == "3f9104c98d439cf0e31fec9472095bd4fae7ad571d846ca015328fe9743ad2f6"
    assert paired.BASE_ARTIFACT_SHA == "7b7cb34f34903ae5846284f8555bbd4029a746cdbcb781d53de8ba5d663db4c7"
    assert paired.BASE_VIDEO_SHA == "8d4ebd03d20bd398b326429c3cc989ecea2fa6344068d4d4cfe29206297b3622"
    assert paired.BASE_RESULT_QA_SHA == "fbb02d17bda0401d98aff85960bb28034fd8bfb2fcbfe6487803ecf7e2fc6ac3"
    assert paired.TAXONOMY == (
        "AUTHORITY_POINT_OUT_OF_IMAGE",
        "SINGLE_JOINT_VETO",
        "NO_INSTANCE_SUPPORTS_SIDE",
        "AUTHORITY_SIDE_ALIAS",
        "WEAK_SUPPORT",
        "MISSING_OR_UNVERIFIED_IDENTITY",
    )


def test_task26_thresholds_and_prompt_are_exact():
    assert selector.MIN_JOINT_SUPPORT_RATIO == 0.20
    assert selector.MIN_SIDE_MARGIN == 0.05
    assert selector.MAX_OBJECT_OVERLAP_OVER_INSTANCE == 0.12
    assert selector.TEXT_PROMPT == "an arm"
    with pytest.raises(selector.DistributedSideEvidenceError, match="threshold drift"):
        selector.SelectorThresholds(min_joint_support_ratio=0.21).validate()


def test_s1_wrist_support_false_is_diagnostic_only(tmp_path: Path):
    _, evidence = _frame_source(tmp_path)
    authorities = {
        "left": _authority(evidence, "left", "AVAILABLE", (2.0, 2.0)),
        "right": _authority(evidence, "right", "MISSING", None),
    }
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:4, 1:4] = True
    raw = _raw(
        tmp_path,
        mask,
        _support(0.6, wrist=False),
        _support(0.0, wrist=False),
    )
    decision = selector.select_frame(
        authorities,
        [raw],
        evidence_root=tmp_path,
        image_shape=(8, 8),
    )
    assert decision.left.status == "ACCEPT"
    assert decision.left.candidate_audit[0].wrist_supported_diagnostic is False
    assert "EGO_WRIST_NOT_SUPPORTED" not in decision.left.candidate_audit[0].rejection_reasons


def test_s2_outside_image_authority_is_diagnostic_only(tmp_path: Path):
    _, evidence = _frame_source(
        tmp_path,
        left_state="OUTSIDE_IMAGE",
        left_wrist=(-3.0, 2.0),
    )
    authorities = {
        "left": _authority(evidence, "left", "OUTSIDE_IMAGE", (-3.0, 2.0)),
        "right": _authority(evidence, "right", "MISSING", None),
    }
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:4, 1:4] = True
    raw = _raw(
        tmp_path,
        mask,
        _support(0.6, wrist=False),
        _support(0.0, wrist=False),
    )
    decision = selector.select_frame(
        authorities,
        [raw],
        evidence_root=tmp_path,
        image_shape=(8, 8),
    )
    assert decision.left.status == "ACCEPT"
    assert decision.left.authority_wrist_outside_image_diagnostic is True


def test_missing_authority_still_holds_only_that_side(tmp_path: Path):
    _, evidence = _frame_source(tmp_path, left_state="MISSING", left_wrist=None)
    authorities = {
        "left": _authority(evidence, "left", "MISSING", None),
        "right": _authority(evidence, "right", "MISSING", None),
    }
    selection = selector.select_frame(
        authorities,
        [],
        evidence_root=tmp_path,
        image_shape=(8, 8),
    )
    assert selection.left.failure_reason == "SIDE_AUTHORITY_EVIDENCE_MISSING"
    assert selection.right.failure_reason == "SIDE_AUTHORITY_EVIDENCE_MISSING"


def _inventory_fixture(tmp_path: Path):
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:4, 1:4] = True
    source_payload = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": "grap_a_cap_004:7",
        "instance_id": 3,
        "mask_sha256": selector.mask_sha256(mask),
    }
    relative = "raw_evidence/grap_a_cap_004/frame_00007_offset_000.json"
    source = tmp_path / paired.BASE_COMPONENT_REL / relative
    _write_json(source, source_payload)
    source_ref = paired.identity(source)
    baseline_row = {
        "raw_instance_evidence": {
            "0": {
                "raw_source": {**source_ref, "source_kind": "SAM_RAW_INSTANCE"},
                "instance_id": 3,
                "score": 0.8,
                "area_pixels": 9,
                "area_ratio": 9 / (960 * 1280),
                "object6d_overlap_pixels": 0,
                "object6d_overlap_over_instance": 0.0,
                "sides": {
                    "left": {"hand_scale_pixels": None, "boundary_supported": False},
                    "right": {"hand_scale_pixels": None, "boundary_supported": False},
                },
            }
        }
    }
    artifact_index = {
        relative: {"path": relative, "bytes": source_ref["bytes"], "sha256": source_ref["sha256"]}
    }
    instances = SimpleNamespace(
        masks=[mask],
        scores=[0.8],
        instance_ids=[3],
    )
    component = tmp_path / "new_component"
    component.mkdir()
    return mask, baseline_row, artifact_index, instances, component


def test_raw_inventory_exact_pair_accepts_identical_and_rejects_score_drift(tmp_path: Path):
    mask, row, index, instances, component = _inventory_fixture(tmp_path)
    raws, evidence = paired.paired_raw_instances(
        selector,
        tmp_path,
        component,
        7,
        instances,
        row,
        {"left": None, "right": None},
        np.zeros_like(mask),
        index,
    )
    assert len(raws) == 1
    assert evidence[0]["paired_exact"] is True

    _, row2, index2, instances2, component2 = _inventory_fixture(tmp_path / "drift")
    instances2.scores = [0.81]
    with pytest.raises(paired.PairedRunError, match="HOLD_NOT_PAIRED"):
        paired.paired_raw_instances(
            selector,
            tmp_path / "drift",
            component2,
            7,
            instances2,
            row2,
            {"left": None, "right": None},
            np.zeros_like(mask),
            index2,
        )


def test_raw_inventory_rejects_mask_or_instance_drift(tmp_path: Path):
    mask, row, index, instances, component = _inventory_fixture(tmp_path)
    changed = mask.copy()
    changed[7, 7] = True
    instances.masks = [changed]
    with pytest.raises(paired.PairedRunError, match="HOLD_NOT_PAIRED"):
        paired.paired_raw_instances(
            selector,
            tmp_path,
            component,
            7,
            instances,
            row,
            {"left": None, "right": None},
            np.zeros_like(mask),
            index,
        )


def test_delta_rejects_unexplained_change():
    empty = np.zeros((4, 4), dtype=np.bool_)
    baseline_side = {
        "status": "HOLD",
        "selected_raw_instance_offset": None,
        "formal_mask_raw_bool_sha256": selector.mask_sha256(empty),
    }
    mask = empty.copy()
    mask[0, 0] = True
    decision = SimpleNamespace(
        status="ACCEPT",
        raw_instance_offset=0,
        authority_wrist_outside_image_diagnostic=False,
        candidate_audit=(),
    )
    baseline_row = {
        "raw_pool_routing": [{"raw_instance_offset": 0, "assigned_pool": None}]
    }
    routing = (SimpleNamespace(raw_instance_offset=0, assigned_pool=None),)
    with pytest.raises(paired.PairedRunError, match="without frozen S1/S2"):
        paired.delta_record(
            selector, baseline_side, decision, mask, baseline_row, routing
        )


def test_delta_allows_change_with_s2_diagnostic():
    empty = np.zeros((4, 4), dtype=np.bool_)
    baseline_side = {
        "status": "HOLD",
        "selected_raw_instance_offset": None,
        "formal_mask_raw_bool_sha256": selector.mask_sha256(empty),
    }
    mask = empty.copy()
    mask[0, 0] = True
    decision = SimpleNamespace(
        status="ACCEPT",
        raw_instance_offset=0,
        authority_wrist_outside_image_diagnostic=True,
        candidate_audit=(),
    )
    baseline_row = {
        "raw_pool_routing": [{"raw_instance_offset": 0, "assigned_pool": None}]
    }
    routing = (SimpleNamespace(raw_instance_offset=0, assigned_pool=None),)
    delta = paired.delta_record(
        selector, baseline_side, decision, mask, baseline_row, routing
    )
    assert delta["changed"] is True
    assert delta["authority_wrist_outside_image_diagnostic"] is True


@pytest.mark.parametrize(
    ("status", "state", "reason", "expected"),
    [
        ("ACCEPT", "AVAILABLE", None, None),
        ("HOLD", "MISSING", "SIDE_AUTHORITY_EVIDENCE_MISSING", "MISSING_OR_UNVERIFIED_IDENTITY"),
        ("REJECT", "AVAILABLE", "NO_INSTANCE_SUPPORTS_SIDE", "NO_INSTANCE_SUPPORTS_SIDE"),
        ("REJECT", "AVAILABLE", "AUTHORITY_SIDE_ALIAS", "AUTHORITY_SIDE_ALIAS"),
        ("REJECT", "AVAILABLE", "MULTIPLE_NUMERIC_REJECTIONS", "WEAK_SUPPORT"),
    ],
)
def test_taxonomy_is_total_for_task26_decisions(status, state, reason, expected):
    decision = SimpleNamespace(status=status, authority_state=state, failure_reason=reason)
    assert paired.taxonomy(decision) == expected


def test_failed_formal_mask_is_empty_and_accept_is_exact():
    raw = np.zeros((960, 1280), dtype=np.bool_)
    raw[3:8, 4:9] = True
    assert np.array_equal(paired.formal_mask(SimpleNamespace(status="ACCEPT", mask=raw)), raw)
    assert not paired.formal_mask(SimpleNamespace(status="REJECT", mask=raw)).any()
    with pytest.raises(paired.PairedRunError, match="lacks raw mask"):
        paired.formal_mask(SimpleNamespace(status="ACCEPT", mask=None))


def test_governance_rejects_bool_p0_and_unbound_pin(tmp_path: Path):
    qa = tmp_path / paired.INDEPENDENT_QA_REL
    payload = tmp_path / "payload.json"
    _write_json(payload, {"x": 1})
    pin = paired.identity(payload)
    _write_json(qa, {"status": "PASS", "p0_findings": False, "refs": [pin]})
    with pytest.raises(paired.PairedRunError, match="exact integer zero"):
        paired.require_governance(tmp_path, {"payload": pin})
    _write_json(qa, {"status": "PASS", "p0_findings": 0})
    with pytest.raises(paired.PairedRunError, match="does not bind payload"):
        paired.require_governance(tmp_path, {"payload": pin})


def test_governance_accepts_exact_ordinary_file_ref(tmp_path: Path):
    qa = tmp_path / paired.INDEPENDENT_QA_REL
    payload = tmp_path / "payload.json"
    _write_json(payload, {"x": 1})
    pin = paired.identity(payload)
    _write_json(qa, {"status": "PASS_CPU_EXACT", "p0_findings": 0, "refs": [pin]})
    result = paired.require_governance(tmp_path, {"payload": pin})
    assert result["bound_pins"]["payload"] == pin


def test_identity_rejects_intermediate_symlink(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    payload = real / "evidence.json"
    payload.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(paired.PairedRunError, match="symlink component"):
        paired.identity(link / "evidence.json")


def test_disk_gate_and_namespace_are_fixed(tmp_path: Path):
    assert paired.MAX_RUN_BYTES == int(2.5 * 1024**3)
    assert paired.ABSOLUTE_RESERVE_BYTES == 8 * 1024**3
    assert paired.ADMISSION_REQUIRED_FREE == paired.MAX_RUN_BYTES + paired.ABSOLUTE_RESERVE_BYTES
    with pytest.raises(paired.PairedRunError, match="namespace"):
        paired.create_run(tmp_path, "not_task28")


def test_frame_json_write_is_delayed_until_next_iou_exists():
    source = (PROJECT / "tools/run_004_full_session_distributed_side_paired_t1.py").read_text()
    update = source.index('frame_rows[-1]["sides"][side]["raw_iou_to_next"] = adjacent')
    prior_write = source.index(
        'exclusive_json(component / "frames" / f"frame_{prior_frame:05d}.json", frame_rows[-1])'
    )
    assert update < prior_write
    assert 'exclusive_json(component / "frames" / f"frame_{last_frame:05d}.json", frame_rows[-1])' in source


def test_terminal_manifest_is_after_decode_and_artifact_manifest():
    source = (PROJECT / "tools/run_004_full_session_distributed_side_paired_t1.py").read_text()
    terminal = source.index("exclusive_json(manifest_path, result)")
    decode = source.index("mask_png_full_decode_count")
    artifact = source.index("exclusive_json(\n            artifact_path")
    assert terminal > decode
    assert terminal > artifact
    assert 'failure = component / "FAILURE_MANIFEST.json"' in source
