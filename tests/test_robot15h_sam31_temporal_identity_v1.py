from __future__ import annotations

import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import SAM31_WEIGHT, build_packet
from chaoyang.ops import run_0915_robot15h_sam31_temporal_identity_v1 as runner
from chaoyang.ops.run_0915_robot15h_sam31_temporal_identity_v1 import (
    HEIGHT,
    PRIMARY_TASK_ID,
    QUALITY_TASK_ID,
    ROLE_BLOCKERS,
    WIDTH,
    assess_hand_temporal_proxy,
    choose_primary_fallback,
    choose_seed_candidate,
    evaluate_stream,
    merge_tracks,
    save_initial_seed_candidates,
    save_packed,
    select_anchor_candidates,
    should_reseed,
)


def test_task_is_sam31_only_and_unanchored_roles_fail_closed() -> None:
    packet = build_packet("0915_robot15h_sam31_temporal_identity_v1")
    assert packet["weights"] == [SAM31_WEIGHT]
    assert "roles_without_independent_visual_anchor_fail_closed" in packet["prerequisites"]
    assert "MULTIPLEX_GEOMETRIC_BOX_not_visual_exemplar" in packet["prerequisites"]
    assert "task_object" in ROLE_BLOCKERS


def test_runtime_recovery_only_changes_fixed_import_binding() -> None:
    recovery = build_packet("0915_robot15h_sam31_temporal_identity_recovery_v1")
    assert recovery["weights"] == [SAM31_WEIGHT]
    assert recovery["dag_dependencies"] == ["0915_robot15h_sam31_temporal_identity_v1"]
    assert "fixed_vendor_SAM3_path_only_no_dependency_install" in recovery["prerequisites"]
    quality = build_packet("0915_robot15h_sam31_temporal_identity_quality_v2")
    assert quality["weights"] == [SAM31_WEIGHT]
    assert quality["dag_dependencies"] == ["0915_robot15h_sam31_temporal_identity_recovery_v1"]
    task_object = build_packet("0915_robot15h_sam31_task_object_wave0_v1")
    assert task_object["dag_dependencies"] == ["0915_robot15h_sam31_temporal_identity_quality_v2"]
    task_object_recovery = build_packet("0915_robot15h_sam31_task_object_wave0_recovery_v1")
    assert task_object_recovery["dag_dependencies"] == ["0915_robot15h_sam31_task_object_wave0_v1"]
    geometry = build_packet("0915_robot15h_geometry_object6d_wave0_v1")
    assert "0915_robot15h_sam31_task_object_wave0_recovery_v1" in geometry["dag_dependencies"]
    assert "0915_robot15h_sam31_temporal_identity_v1" not in geometry["dag_dependencies"]


def test_anchor_selection_uses_only_direct_observed_side_frames() -> None:
    joints = np.full((2, 20, 21, 2), np.nan, np.float64)
    observed = np.zeros((2, 20), bool)
    confidence = np.zeros((2, 20), np.float64)
    base = np.stack((np.linspace(200, 300, 21), np.linspace(300, 380, 21)), axis=1)
    joints[0, 3] = base
    joints[0, 12] = base + 20
    observed[0, [3, 12]] = True
    confidence[0, 3], confidence[0, 12] = 0.9, 0.4
    # Finite but not direct-observed must not become an anchor.
    joints[0, 7] = base + 50
    rows = select_anchor_candidates(joints, observed, confidence, 0)
    assert {row["frame_index"] for row in rows} == {3, 12}
    assert select_anchor_candidates(joints, observed, confidence, 1) == []


def test_fallback_requires_frozen_temporal_separation() -> None:
    rows = [
        {"frame_index": 10, "score_tuple": [1]},
        {"frame_index": 20, "score_tuple": [0]},
        {"frame_index": 30, "score_tuple": [-1]},
    ]
    primary, fallback = choose_primary_fallback(rows, 100)
    assert primary["frame_index"] == 10
    assert fallback["frame_index"] == 30  # minimum separation is 15, so frame 20 is rejected


def test_opposite_palm_hit_rejects_seed_identity() -> None:
    masks = np.zeros((1, HEIGHT, WIDTH), bool)
    masks[0, 100:250, 100:250] = True
    ids = np.asarray([7])
    scores = np.asarray([0.99])
    own = np.repeat(np.asarray([[150.0, 150.0]]), 21, axis=0)
    opposite = np.repeat(np.asarray([[160.0, 160.0]]), 21, axis=0)
    anchor = {"box_xywh": [100.0, 100.0, 150.0, 150.0]}
    try:
        choose_seed_candidate(masks, scores, ids, anchor, own, opposite)
    except Exception as error:
        assert "side-safe" in str(error)
    else:
        raise AssertionError("opposite palm overlap must reject the candidate")


def test_seed_area_gate_scales_with_anchor_box_for_small_valid_hand() -> None:
    masks = np.zeros((1, HEIGHT, WIDTH), bool)
    masks[0, 110:140, 110:140] = True  # 900 px: below the former fixed 1,200 floor.
    own = np.repeat(np.asarray([[125.0, 125.0]]), 21, axis=0)
    raw_id, rows = choose_seed_candidate(
        masks, np.asarray([0.8]), np.asarray([13]),
        {"box_xywh": [100.0, 100.0, 60.0, 60.0]}, own, None,
    )
    assert raw_id == 13
    assert rows[0]["minimum_area_pixels"] == 288
    assert rows[0]["minimum_area_policy"] == "CLAMP_128_1200_OF_0_08_ANCHOR_BOX_AREA"
    assert rows[0]["eligible"] is True


def test_seed_area_gate_preserves_strict_area_upper_bound() -> None:
    masks = np.ones((1, HEIGHT, WIDTH), bool)
    own = np.repeat(np.asarray([[125.0, 125.0]]), 21, axis=0)
    try:
        choose_seed_candidate(
            masks, np.asarray([0.99]), np.asarray([1]),
            {"box_xywh": [100.0, 100.0, 60.0, 60.0]}, own, None,
        )
    except Exception as error:
        assert "side-safe" in str(error)
    else:
        raise AssertionError("scale-aware minimum must not relax the area upper gate")


def test_stream_area_floor_is_anchor_scale_aware() -> None:
    masks = np.zeros((3, HEIGHT, WIDTH), bool)
    masks[:, 110:140, 110:140] = True
    joints = np.full((2, 3, 21, 2), np.nan, np.float64)
    observed = np.zeros((2, 3), bool)
    valid, rows = evaluate_stream(masks, 0, joints, observed, 0)
    assert valid.tolist() == [True, True, True]
    assert {row["minimum_area_pixels"] for row in rows} == {128}


def test_reseed_is_quality_triggered_not_periodic() -> None:
    valid = np.ones(100, bool)
    assert should_reseed(valid, True) == (False, [])
    valid[20:23] = False
    triggered, reasons = should_reseed(valid, True)
    assert triggered is True
    assert "THREE_OR_MORE_CONSECUTIVE_UNKNOWN" in reasons
    assert should_reseed(valid, False)[0] is False


def test_unknown_semantic_is_separate_from_raw_candidate() -> None:
    primary = np.zeros((2, 4, 4), bool)
    primary[:, 1:3, 1:3] = True
    primary_valid = np.asarray([True, False])
    raw, semantic, ledger = merge_tracks(primary, primary_valid, None, None, 0, None)
    assert raw[1].any()
    assert not semantic[1].any()
    assert ledger[1]["raw_present"] is True
    assert ledger[1]["semantic_admitted"] is False
    assert ledger[1]["tracking_state"] == "unknown"


def test_direct_admission_is_not_failed_by_whole_video_unknown_diagnostic() -> None:
    passed, status, diagnostics = assess_hand_temporal_proxy(
        direct_count=40, direct_fraction=0.90, unknown_fraction=0.80,
        identity_conflicts=0,
    )
    assert passed is True
    assert status == "PASS_HAND_DIRECT_OBSERVED_PROXY"
    assert diagnostics == ["WHOLE_VIDEO_UNKNOWN_FRACTION_ABOVE_0_35"]


def test_identity_conflict_remains_a_hard_rejection() -> None:
    passed, status, _ = assess_hand_temporal_proxy(
        direct_count=40, direct_fraction=0.90, unknown_fraction=0.0,
        identity_conflicts=1,
    )
    assert passed is False
    assert status == "REJECTED_IDENTITY_CONFLICT"


def test_staged_archives_publish_final_paths(tmp_path) -> None:
    staging = tmp_path / ".session.staging" / "semantic" / "left_hand.npz"
    final = tmp_path / "session" / "semantic" / "left_hand.npz"
    masks = np.zeros((2, 4, 4), bool)
    masks[0, 1:3, 1:3] = True
    artifact = save_packed(staging, masks, published_path=final)
    assert artifact["path"] == str(final.resolve())
    assert ".staging" not in artifact["path"]
    assert artifact["sha256"] == runner.sha256(staging)


def test_publication_audit_accepts_only_resolved_final_refs(tmp_path) -> None:
    original_output = runner.OUTPUT
    try:
        runner.OUTPUT = tmp_path / "attempt_0001"
        runner.OUTPUT.mkdir()
        final = tmp_path / "sessions" / "session_001" / "semantic" / "left_hand.npz"
        final.parent.mkdir(parents=True)
        final.write_bytes(b"published")
        audit = runner.audit_publication_refs([{"semantic_archive": runner.ref(final)}])
        assert audit["status"] == "PASSED"
        assert audit["refs_checked"] == 1
        assert audit["transient_paths_serialized"] == 0
    finally:
        runner.OUTPUT = original_output


def test_rejected_seed_candidates_are_preserved_as_nonsemantic_evidence(tmp_path) -> None:
    path = tmp_path / "raw_candidate" / "left_hand_primary_seed_candidates.npz"
    masks = np.zeros((2, 8, 8), bool)
    masks[0, 1:4, 1:4] = True
    masks[1, 4:7, 4:7] = True
    artifact = save_initial_seed_candidates(path, {
        "masks": masks, "scores": np.asarray([0.7, 0.5]),
        "raw_ids": np.asarray([2, 9]), "frame_index": 12,
    })
    assert artifact["path"] == str(path.resolve())
    with np.load(path, allow_pickle=False) as archive:
        assert int(archive["candidate_count"]) == 2
        assert archive["raw_ids"].tolist() == [2, 9]
        assert bool(archive["semantic_admitted"]) is False
        assert bool(archive["persistent_identity_selected"]) is False


def test_quality_v2_has_an_isolated_namespace() -> None:
    try:
        runner.configure_task(QUALITY_TASK_ID)
        assert runner.TASK_ID == QUALITY_TASK_ID
        assert runner.OUTPUT.parts[-3:] == (QUALITY_TASK_ID, "attempts", "attempt_0001")
        assert runner.VISUAL.name.endswith("QUALITY_V2")
    finally:
        runner.configure_task(PRIMARY_TASK_ID)
