from __future__ import annotations

import inspect

import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import SAM31_WEIGHT, build_packet
from chaoyang.ops import run_0915_robot15h_sam31_task_object_wave0_v1 as runner


def rectangle(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((runner.HEIGHT, runner.WIDTH), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_packet_binds_only_sam31_and_fixed_short_prompts() -> None:
    packet = build_packet(runner.TASK_ID)
    assert packet["weights"] == [SAM31_WEIGHT]
    assert runner.prompt_for_task("playing_cards") == "playing card"
    assert runner.prompt_for_task("potato_chips") == "potato chip"


def test_empty_candidate_archive_is_valid_unknown_evidence(tmp_path) -> None:
    staging = tmp_path / ".session.staging-abc" / "raw_candidate" / "seed.npz"
    final = tmp_path / "session" / "raw_candidate" / "seed.npz"
    masks = np.zeros((0, runner.HEIGHT, runner.WIDTH), bool)
    artifact = runner.save_initial_candidates(
        staging, masks, np.zeros(0, np.float32), np.zeros(0, np.int64), 7,
        published_path=final,
    )
    assert artifact["path"] == str(final.resolve())
    with np.load(staging, allow_pickle=False) as archive:
        assert archive["packed"].shape == (0, runner.HEIGHT * runner.WIDTH // 8)
        assert int(archive["candidate_count"]) == 0
        assert bool(archive["semantic_admitted"]) is False


def test_empty_candidate_recovery_is_bounded_successor() -> None:
    packet = build_packet(runner.RECOVERY_TASK_ID)
    assert packet["weights"] == [SAM31_WEIGHT]
    assert packet["dag_dependencies"] == [runner.PRIMARY_TASK_ID]
    assert "empty_text_candidate_serializes_as_raw_UNKNOWN_not_absent" in packet["prerequisites"]
    geometry = build_packet("0915_robot15h_geometry_object6d_wave0_v1")
    assert runner.RECOVERY_TASK_ID in geometry["dag_dependencies"]


def test_runner_has_no_manual_spatial_prompt() -> None:
    source = inspect.getsource(runner.run_text_seed)
    assert "boxes_xywh" not in source
    assert "point_coords" not in source
    assert 'text_str=prompt_for_task(task)' in source
    assert runner.deterministic_anchor_schedule(150)[0] == 0
    assert len(runner.deterministic_anchor_schedule(150)) == 4


def test_cards_are_selected_as_separate_instances_not_union() -> None:
    masks = np.stack([
        rectangle(300, 350, 200, 270),
        rectangle(300, 350, 500, 570),
        rectangle(300, 350, 800, 870),
    ])
    selected, rows = runner.select_separate_instances(
        masks, np.asarray([0.8, 0.9, 0.7]), np.asarray([12, 4, 9]), "playing_cards",
    )
    assert set(selected) == {4, 9, 12}
    assert all(row["selected_as_separate_instance"] for row in rows)
    assert sum(row["area_pixels"] for row in rows) == int(masks.sum())


def test_overlapping_card_proposal_is_not_a_second_physical_instance() -> None:
    masks = np.stack([
        rectangle(300, 350, 200, 270),
        rectangle(302, 352, 202, 272),
    ])
    selected, rows = runner.select_separate_instances(
        masks, np.asarray([0.9, 0.8]), np.asarray([1, 2]), "playing_cards",
    )
    assert selected == [1]
    rejected = next(row for row in rows if row["raw_id"] == 2)
    assert "OVERLAPS_SELECTED_PHYSICAL_INSTANCE" in rejected["reasons"]


def test_support_scale_candidate_is_rejected() -> None:
    # A broad tray/bowl-like foreground cannot become a task-object instance.
    support = rectangle(200, 600, 200, 1000)
    row = runner.candidate_row(support, 0.99, 5, "playing_cards")
    assert row["eligible_shape_candidate"] is False
    assert "SUPPORT_SCALE_AREA_REJECT" in row["reasons"]
    assert row["support_tray_or_bowl_admitted"] is False


def test_chip_instance_count_is_bounded_and_separate() -> None:
    masks = np.stack([rectangle(200 + i * 100, 235 + i * 100, 200, 245) for i in range(5)])
    selected, rows = runner.select_separate_instances(
        masks, np.linspace(0.95, 0.55, 5), np.arange(5), "potato_chips",
    )
    assert len(selected) == runner.TASK_POLICIES["potato_chips"]["maximum_instances"] == 3
    assert sum(row["selected_as_separate_instance"] for row in rows) == 3


def test_track_loss_is_unknown_not_absence() -> None:
    track = np.zeros((20, runner.HEIGHT, runner.WIDTH), bool)
    track[:5, 300:350, 200:270] = True
    valid, rows, summary = runner.evaluate_identity(track, 0, "playing_cards")
    assert valid[:5].all()
    assert not valid[5:].any()
    assert all(row["tracking_state"] == "unknown" for row in rows[5:])
    assert all(row["visibility_state"] == "UNKNOWN" for row in rows[5:])
    assert summary["unknown_frames"] == 15


def test_temporal_instance_collapse_becomes_unknown_not_merged_identity() -> None:
    first = np.zeros((60, runner.HEIGHT, runner.WIDTH), bool)
    second = np.zeros_like(first)
    first[:, 300:350, 200:270] = True
    second[:, 300:350, 500:570] = True
    # One tracker-collapse frame is locally rejected while the persistent
    # identities remain separate and can still pass bounded temporal QA.
    second[20] = first[20]
    evaluations = {
        1: runner.evaluate_identity(first, 0, "playing_cards"),
        2: runner.evaluate_identity(second, 0, "playing_cards"),
    }
    runner.apply_inter_instance_exclusion({1: first, 2: second}, evaluations)
    for raw_id in (1, 2):
        valid, rows, quality = evaluations[raw_id]
        assert valid[20] is np.False_ or not bool(valid[20])
        assert rows[20]["tracking_state"] == "unknown"
        assert "INTER_INSTANCE_OVERLAP_IDENTITY_CONFLICT" in rows[20]["reasons"]
        assert quality["inter_instance_identity_conflict_frames"] == 1
        assert quality["stable_instance"] is True


def test_packed_archive_publishes_final_target_reference(tmp_path) -> None:
    staging = tmp_path / ".session.staging-abc" / "semantic" / "playing_card_00.npz"
    final = tmp_path / "session" / "semantic" / "playing_card_00.npz"
    masks = np.zeros((2, 8, 8), bool)
    masks[0, 1:4, 1:4] = True
    artifact = runner.save_packed(staging, masks, published_path=final)
    assert artifact["path"] == str(final.resolve())
    assert ".staging-" not in artifact["path"]
    assert artifact["sha256"] == runner.sha256(staging)


def test_raw_initial_candidates_are_preserved_without_semantic_upgrade(tmp_path) -> None:
    staging = tmp_path / ".session.staging-abc" / "raw_candidate" / "seed.npz"
    final = tmp_path / "session" / "raw_candidate" / "seed.npz"
    masks = np.stack([rectangle(300, 350, 200, 270), rectangle(300, 350, 500, 570)])
    artifact = runner.save_initial_candidates(
        staging, masks, np.asarray([0.9, 0.8]), np.asarray([2, 7]), 0, published_path=final,
    )
    assert artifact["path"] == str(final.resolve())
    with np.load(staging, allow_pickle=False) as archive:
        assert int(archive["candidate_count"]) == 2
        assert bool(archive["semantic_admitted"]) is False
        assert bool(archive["union_mask_created"]) is False
