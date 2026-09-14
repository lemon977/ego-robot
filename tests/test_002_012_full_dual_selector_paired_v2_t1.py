from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task29_paired_runner_test",
    PROJECT / "tools/run_002_012_full_dual_selector_paired_v2_t1.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
helper = runner.context_helper(PROJECT)


def load_selector():
    return runner.load_module(
        "task29_selector_test", PROJECT / runner.NEW_MODULE_REL
    )


def test_frozen_context_and_storage_pins_are_exact():
    pins = runner.verify_pins(PROJECT)
    assert pins["context_result"]["sha256"] == runner.CONTEXT_RESULT_SHA
    assert pins["v1_stale_marker"]["sha256"] == runner.V1_STALE_MARKER_SHA
    assert pins["v2_stale_marker"]["sha256"] == runner.V2_STALE_MARKER_SHA
    assert runner.MAX_RUN_BYTES == int(1.5 * 1024**3)
    assert runner.ADMISSION_REQUIRED_FREE == (
        8 * 1024**3 + int(1.5 * 1024**3) + 8 * 1024**2
    )


def test_transient_raw_index_references_but_never_copies(tmp_path: Path):
    source = tmp_path / "raw.png"
    source.write_bytes(b"raw-evidence-bytes")
    ref = helper.identity(source)
    temporary, index, evidence = runner.prepare_transient_raw_index(
        helper,
        [
            {
                "local_index": 0,
                "source_frame_index": 7,
                "raw_path": ref["path"],
                "raw_bytes": ref["bytes"],
                "raw_sha256": ref["sha256"],
            }
        ],
    )
    try:
        entry = index / "00000.png"
        assert entry.is_symlink()
        assert entry.resolve() == source
        assert evidence["persistent_raw_copies"] == 0
        assert evidence["source_bytes_referenced_not_written"] == len(
            b"raw-evidence-bytes"
        )
    finally:
        temporary.cleanup()
    assert not index.exists()


def test_stream_prompt_once_is_one_prompt_and_contiguous(monkeypatch: pytest.MonkeyPatch):
    instances = SimpleNamespace(
        masks=np.zeros((1, 4, 4), dtype=np.bool_),
        scores=np.asarray([0.7]),
        instance_ids=np.asarray([8]),
    )

    class Base:
        @staticmethod
        def outputs_to_instances(_outputs, _height, _width):
            return instances

    class Old:
        base = Base()

        @staticmethod
        def raw_mask_digest(mask):
            return load_selector().mask_sha256(mask)

    class Adapter:
        def __init__(self):
            self.prompts = 0
            self.closed = 0

        def handle_request(self, value):
            if value["type"] == "start_session":
                return {"parameter_mapping": {"text": "an arm"}}
            if value["type"] == "add_prompt":
                self.prompts += 1
                return {"outputs": object()}
            if value["type"] == "close_session":
                self.closed += 1
                return {}
            raise AssertionError(value)

        @staticmethod
        def handle_stream_request(_value):
            for frame in range(3):
                yield {"frame_index": frame, "outputs": object()}

    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    adapter = Adapter()
    timing = {}
    rows = list(
        runner.stream_prompt_once(
            adapter,
            Old(),
            Path("/unused"),
            session="grap_a_cap_002",
            frame_count=3,
            timing=timing,
        )
    )
    assert [frame for frame, _ in rows] == [0, 1, 2]
    assert adapter.prompts == 1
    assert adapter.closed == 1
    assert timing["frame_count"] == 3
    assert timing["materialized_full_session_output_list"] is False
    duplicate = timing["propagation_frame0_duplicate_delta"]
    assert duplicate["provenance_identity_match"] is True
    assert duplicate["consumed_by_selectors"] == "ADD_PROMPT_FRAME0"
    assert duplicate["discarded_without_pixel_use"] == (
        "PROPAGATION_DUPLICATE_FRAME0"
    )


def frame0_stream_fixture(
    monkeypatch: pytest.MonkeyPatch,
    initial_instances,
    propagated_instances,
):
    later_instances = SimpleNamespace(
        masks=np.zeros_like(initial_instances.masks),
        scores=np.array(initial_instances.scores, copy=True),
        instance_ids=np.array(initial_instances.instance_ids, copy=True),
    )

    class Base:
        @staticmethod
        def outputs_to_instances(outputs, _height, _width):
            return outputs

    class Old:
        base = Base()

        @staticmethod
        def raw_mask_digest(mask):
            return load_selector().mask_sha256(mask)

    class Adapter:
        def __init__(self):
            self.closed = 0

        def handle_request(self, value):
            if value["type"] == "start_session":
                return {"parameter_mapping": {"text": "an arm"}}
            if value["type"] == "add_prompt":
                return {"outputs": initial_instances}
            if value["type"] == "close_session":
                self.closed += 1
                return {}
            raise AssertionError(value)

        @staticmethod
        def handle_stream_request(_value):
            yield {"frame_index": 0, "outputs": propagated_instances}
            yield {"frame_index": 1, "outputs": later_instances}

    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    return Adapter(), Old()


def test_stream_prompt_once_discards_boundary_raster_delta_and_keeps_add_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    initial_mask = np.zeros((1, 5, 5), dtype=np.bool_)
    initial_mask[0, 1:4, 1:4] = True
    propagated_mask = initial_mask.copy()
    propagated_mask[0, 1, 1] = False
    propagated_mask[0, 1, 2] = False
    propagated_mask[0, 0, 1] = True
    propagated_mask[0, 0, 2] = True
    initial = SimpleNamespace(
        masks=initial_mask,
        scores=np.asarray([0.7], dtype=np.float32),
        instance_ids=np.asarray([8]),
    )
    propagated = SimpleNamespace(
        masks=propagated_mask,
        scores=np.asarray([0.7], dtype=np.float32),
        instance_ids=np.asarray([8]),
    )
    adapter, old = frame0_stream_fixture(monkeypatch, initial, propagated)
    timing = {}

    rows = list(
        runner.stream_prompt_once(
            adapter,
            old,
            Path("/unused"),
            session="grap_a_cap_002",
            frame_count=2,
            timing=timing,
        )
    )

    assert [frame for frame, _ in rows] == [0, 1]
    assert rows[0][1] is initial
    assert all(item[1] is not propagated for item in rows)
    assert adapter.closed == 1
    diagnostic = timing["propagation_frame0_duplicate_delta"]
    assert diagnostic["provenance_identity_match"] is True
    assert diagnostic["mask_bytes_required_equal"] is False
    delta = diagnostic["instances"][0]
    assert delta["instance_id_match"] is True
    assert delta["score_exact_match"] is True
    assert delta["mask_sha256_match"] is False
    assert delta["pixels_only_add_prompt"] == 2
    assert delta["pixels_only_propagation"] == 2
    assert delta["mask_xor_pixels"] == 4


@pytest.mark.parametrize("drift", ["instance_id", "score"])
def test_stream_prompt_once_rejects_true_frame0_identity_or_score_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
):
    mask = np.zeros((1, 4, 4), dtype=np.bool_)
    mask[0, 1:3, 1:3] = True
    initial = SimpleNamespace(
        masks=mask,
        scores=np.asarray([0.7], dtype=np.float64),
        instance_ids=np.asarray([8]),
    )
    propagated = SimpleNamespace(
        masks=mask.copy(),
        scores=np.asarray([0.7000001 if drift == "score" else 0.7]),
        instance_ids=np.asarray([9 if drift == "instance_id" else 8]),
    )
    adapter, old = frame0_stream_fixture(monkeypatch, initial, propagated)
    timing = {}

    with pytest.raises(runner.PairedError, match="instance provenance drift"):
        list(
            runner.stream_prompt_once(
                adapter,
                old,
                Path("/unused"),
                session="grap_a_cap_002",
                frame_count=2,
                timing=timing,
            )
        )

    assert adapter.closed == 1
    diagnostic = timing["propagation_frame0_duplicate_delta"]
    assert diagnostic["provenance_identity_match"] is False
    match_field = "instance_id_match" if drift == "instance_id" else "score_exact_match"
    assert diagnostic["instances"][0][match_field] is False


def test_shared_raw_instance_object_is_consumed_without_copy(tmp_path: Path):
    selector = load_selector()
    mask = np.zeros((1, 8, 8), dtype=np.bool_)
    mask[0, 1:5, 2:6] = True
    instances = SimpleNamespace(
        masks=mask,
        scores=np.asarray([0.8]),
        instance_ids=np.asarray([31]),
    )
    evidence_path = tmp_path / "raw.json"
    evidence_payload = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": "grap_a_cap_002:0",
        "instance_id": 31,
        "mask_sha256": selector.mask_sha256(mask[0]),
    }
    helper.exclusive_json(evidence_path, evidence_payload)
    ref = helper.identity(evidence_path)
    old_evidence = {
        0: {
            "raw_source": {**ref, "source_kind": "SAM_RAW_INSTANCE"},
            "score": 0.8,
            "instance_id": 31,
            "area_ratio": float(mask[0].mean()),
            "object6d_overlap_over_instance": 0.0,
            "sides": {
                side: {"hand_scale_pixels": None, "boundary_supported": False}
                for side in runner.SIDES
            },
        }
    }
    raws, audit = runner.task26_raws(
        selector,
        tmp_path,
        "grap_a_cap_002",
        0,
        instances,
        old_evidence,
        {side: None for side in runner.SIDES},
    )
    assert np.shares_memory(raws[0].mask, instances.masks[0])
    assert audit[0]["same_object_consumed_by_both_selectors"] is True
    assert audit[0]["mask_sha256"] == selector.mask_sha256(mask[0])


def test_area_formula_and_failed_empty_mask_include_every_adjacent_pair():
    assert runner.relative_area_change(50, 100) == (1.0, False)
    assert runner.relative_area_change(0, 100) == (None, True)
    assert runner.relative_area_change(0, None) == (None, True)
    rejected = SimpleNamespace(status="HOLD", mask=None)
    empty = runner.formal_mask(rejected)
    accepted = SimpleNamespace(status="ACCEPT", mask=np.ones((runner.HEIGHT, runner.WIDTH), bool))
    full = runner.formal_mask(accepted)
    assert empty.sum() == 0
    assert runner.iou(empty, empty) == 1.0
    assert runner.iou(empty, full) == 0.0


def test_real_v3_oob_denominators_reproduce_all_conventions():
    authority, _ = runner.verified_json(helper, PROJECT / runner.OOB_CONTEXT_REL)
    frames = [
        {"frame_index": row["frame_index"], "authority": row["sides"]}
        for row in authority["frame_side_rows"]
        if row["session_id"] == "grap_a_cap_012"
    ]
    report = runner.oob_denominator_report(frames, "grap_a_cap_012")
    right = report["sides"]["right"]
    assert right["selector_operational_total"]["numerator"] == 113
    assert right["selector_operational_total"]["denominator"] == 364
    assert right["valid_finite_geometry"]["numerator"] == 113
    assert right["measurement_accepted_observed_only"]["numerator"] == 112
    assert right["measurement_accepted_observed_only"]["denominator"] == 363
    assert right["last_integer_pixel_center_total"]["numerator"] == 114
    assert report["left_oob_right_missing_overlap"]["intersection_count"] == 0


def test_runtime_geometry_parses_same_verified_bytes():
    authority, _ = runner.verified_json(helper, PROJECT / runner.OOB_CONTEXT_REL)
    geometry = runner.load_runtime_geometry(
        helper,
        PROJECT,
        "grap_a_cap_012",
        authority["sessions"]["grap_a_cap_012"]["geometry"],
    )
    assert geometry["joints"].shape == (2, 364, 21, 2)
    assert geometry["object_transform"].shape[0] == 364
    assert geometry["radius_m"] > 0


def test_physical_authority_ignores_source_slot_alias(tmp_path: Path):
    old = runner.load_module("task29_old_test", PROJECT / runner.OLD_RUNNER_REL)
    arrays, sources = helper.load_geometry(PROJECT, "grap_a_cap_012", 364)
    frozen_payload, _ = runner.verified_json(helper, PROJECT / runner.OOB_CONTEXT_REL)
    frozen = {
        (row["session_id"], row["frame_index"]): row
        for row in frozen_payload["frame_side_rows"]
    }[("grap_a_cap_012", 312)]["sides"]["left"]
    component = tmp_path / "component"
    (component / "authority/grap_a_cap_012").mkdir(parents=True)
    authority, joints, diagnostic = runner.write_physical_axis_authority(
        helper,
        old,
        component,
        session="grap_a_cap_012",
        frame=312,
        side="left",
        arrays=arrays,
        confidence_by_axis=arrays["confidence"],
        quality=sources["quality"],
        frozen=frozen,
    )
    assert diagnostic["source_track_index"] == 0
    assert diagnostic["source_slot_observation"] == -1
    assert joints is not None
    assert np.allclose(joints[0], frozen["wrist_xy"])
    assert authority.lineage_id == frozen["lineage_id"]


def qa_payload(pins):
    return {
        "schema_version": "task29-independent-cpu-qa-v2",
        "status": "PASS_TASK29_CPU_QA_P0_ZERO_GPU_ADMISSION_ALLOWED",
        "producer": "INDEPENDENT_QA",
        "AUTH_TIER": "T1_INDEPENDENT_CPU_QA",
        "WHY_NOT_BLOCKED": "READ_ONLY_QA_OF_FROZEN_TASK29_CPU_IMPLEMENTATION",
        "p0_findings": 0,
        "p1_findings": 0,
        "access_counters": {key: 0 for key in runner.QA_ACCESS_KEYS},
        "frozen_refs": pins,
        "checks": {},
        "findings": [],
        "claim_limit": "CPU_QA_ONLY_GPU_ADMISSION_ALLOWED_CANDIDATE_STILL_UNREVIEWED",
        "created_at": "2026-08-28T00:00:00+08:00",
    }


def write_qa_root(tmp_path: Path, value):
    path = tmp_path / runner.INDEPENDENT_QA_REL
    path.parent.mkdir(parents=True)
    helper.exclusive_json(path, value)


def test_governance_rejects_pass_prefix_and_nested_ref_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    pins = {"only": helper.identity(evidence)}
    qa = qa_payload(pins)
    qa["status"] += "_ATTACKER"
    qa["nested_ref"] = {"only": pins["only"]}
    write_qa_root(tmp_path, qa)
    monkeypatch.setattr(runner, "context_helper", lambda _project: helper)
    with pytest.raises(runner.PairedError):
        runner.require_governance(tmp_path, pins)


def test_governance_accepts_only_exact_top_level_ref_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    pins = {"only": helper.identity(evidence)}
    write_qa_root(tmp_path, qa_payload(pins))
    monkeypatch.setattr(runner, "context_helper", lambda _project: helper)
    result = runner.require_governance(tmp_path, pins)
    assert result["qa"]["p0_findings"] == 0


def test_admission_rejects_intermediate_run_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "_run").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "context_helper", lambda _project: helper)
    monkeypatch.setattr(runner, "verify_pins", lambda _project: {})
    monkeypatch.setattr(runner, "require_governance", lambda _project, _pins: {})
    with pytest.raises(OSError):
        runner.admit(tmp_path, "002_012_full_dual_selector_paired_attack", Path("/x"), Path("/y"))
    assert list(outside.iterdir()) == []


def test_disk_gate_checks_cap_and_treats_cpfs_zero_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_fd = 9
    monkeypatch.setattr(
        runner,
        "run_fs_state",
        lambda _fd: {"st_dev": 1, "st_ino": 2, "free_bytes": 0, "block_size": 4096},
    )
    monkeypatch.setattr(runner, "tree_bytes_nofollow", lambda *_args: 10)
    assert runner.disk_violation(run_fd, tmp_path, helper) is None
    monkeypatch.setattr(
        runner, "tree_bytes_nofollow", lambda *_args: runner.MAX_RUN_BYTES + 1
    )
    assert runner.disk_violation(run_fd, tmp_path, helper)["run_bytes"] == runner.MAX_RUN_BYTES + 1


def early_observability_fixture(failure_reason: str, authority_state: str):
    old = runner.load_module("task29_old_early_observe", PROJECT / runner.OLD_MODULE_REL)
    selector = load_selector()
    decision = old.SideDecision(
        "right", "session:axis1:right", "HOLD", None, None, None,
        failure_reason, (),
    )
    old_evidence = {
        0: {
            "raw_source": {
                "path": "/evidence/raw.json",
                "bytes": 10,
                "sha256": "a" * 64,
                "source_kind": "SAM_RAW_INSTANCE",
            },
            "score": 0.9,
            "instance_id": 3,
            "object6d_overlap_over_instance": 0.01,
            "sides": {
                "left": {
                    "joint_support_ratio": 0.05,
                    "wrist_supported": False,
                    "boundary_supported": False,
                },
                "right": {
                    "joint_support_ratio": 0.75,
                    "wrist_supported": True,
                    "boundary_supported": True,
                },
            },
        }
    }
    shared = [
        {
            "raw_instance_offset": 0,
            "instance_id": 3,
            "shared_raw_source": old_evidence[0]["raw_source"],
            "support": {
                "left": asdict(selector.JointSupportEvidence(20, 1, 0.05, 2.0, False)),
                "right": asdict(selector.JointSupportEvidence(20, 15, 0.75, 2.0, True)),
            },
        }
    ]
    old_routing = [{"raw_instance_offset": 0, "assigned_pool": "right"}]
    new_routing = [
        selector.RoutingAudit(0, 3, "right", "RIGHT_DOMINANT", 0.05, 0.75)
    ]
    authority = {
        "state": authority_state,
        "wrist_xy": [1100.0, 980.0] if authority_state == "OUTSIDE_IMAGE" else None,
        "outside_image_distance_px": 21.0 if authority_state == "OUTSIDE_IMAGE" else None,
    }
    return decision, authority, old_evidence, old_routing, shared, new_routing


@pytest.mark.parametrize(
    ("failure_reason", "authority_state"),
    [
        ("SIDE_AUTHORITY_EVIDENCE_OUTSIDE_IMAGE", "OUTSIDE_IMAGE"),
        ("SIDE_AUTHORITY_EVIDENCE_MISSING", "MISSING"),
        ("SIDE_IDENTITY_LINEAGE_UNVERIFIED", "UNVERIFIED_IDENTITY"),
        ("RAW_INSTANCE_CROSS_POOL_ALIAS", "AVAILABLE"),
    ],
)
def test_early_hold_retains_every_raw_side_audit_without_changing_decision_or_mask(
    failure_reason: str, authority_state: str
):
    decision, authority, old_evidence, old_routing, shared, new_routing = (
        early_observability_fixture(failure_reason, authority_state)
    )
    before_decision = json.dumps(asdict(decision), sort_keys=True)
    before_mask = runner.formal_mask(decision)
    before_mask_sha = load_selector().mask_sha256(before_mask)
    audit = runner.observability_candidate_audit(
        selector_name="old_v3",
        side="right",
        decision=decision,
        authority=authority,
        old_evidence=old_evidence,
        old_routing=old_routing,
        shared_inventory=shared,
        new_routing=new_routing,
    )
    assert len(audit) == 1
    item = audit[0]
    assert item["assigned_pool"] == "right"
    assert item["considered_for_side"] is True
    assert item["eligible"] is False
    assert item["candidate_eligible_before_final_veto"] is True
    assert item["ordered_rejection_reasons"][-1] == f"FINAL_VETO:{failure_reason}"
    assert item["joint_support_ratio"] == 0.75
    assert item["side_margin"] == pytest.approx(0.70)
    assert item["object6d_overlap_over_instance"] == 0.01
    assert item["wrist_supported_diagnostic"] is True
    assert item["boundary_supported_diagnostic"] is True
    assert item["final_veto"] == failure_reason
    assert item["diagnostic_only_no_selector_reentry"] is True
    assert json.dumps(asdict(decision), sort_keys=True) == before_decision
    after_mask = runner.formal_mask(decision)
    assert load_selector().mask_sha256(after_mask) == before_mask_sha
    assert np.array_equal(after_mask, before_mask)


def test_task26_early_hold_observability_uses_in_image_distribution_metrics():
    decision, authority, old_evidence, old_routing, shared, new_routing = (
        early_observability_fixture("SIDE_AUTHORITY_EVIDENCE_MISSING", "MISSING")
    )
    audit = runner.observability_candidate_audit(
        selector_name="task26",
        side="right",
        decision=decision,
        authority=authority,
        old_evidence=old_evidence,
        old_routing=old_routing,
        shared_inventory=shared,
        new_routing=new_routing,
    )
    assert audit[0]["in_image_joint_count"] == 20
    assert audit[0]["supported_in_image_joint_count"] == 15
    assert audit[0]["eligible"] is False
    assert audit[0]["final_veto"] == "SIDE_AUTHORITY_EVIDENCE_MISSING"
