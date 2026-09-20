from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chaoyang.ops import build_exact78_same_session_hand_initializer_v32 as producer


PROJECT = Path(__file__).resolve().parents[1]


def test_archive_inputs_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "archive" / "legacy.npz"
    path.parent.mkdir()
    path.write_bytes(b"not consumed")
    with pytest.raises(producer.SameSessionHandError, match="archive input forbidden"):
        producer.reject_archive_inputs((path,))


def test_runner_path_alias_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "play_cards_0901_042" / "HAWOR.npz"
    source.write_bytes(b"same-session-hawor-bytes")
    producer.atomic_byte_alias(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert producer.sha256(destination) == producer.sha256(source)


def test_frozen_v32_membership_is_same_session_train() -> None:
    row = producer.validate_session_membership(
        PROJECT / "manifests/human_ego/exact78_cohort_split_v32.json",
        "play_cards_0901_042",
        "poker",
    )
    assert row["split"] == "train"
    assert row["source_group_id"] == "merged-00-20260901-213104:get_potato_chips_0901"


def test_initializer_contains_hand_seed_but_no_arm_or_installation_geometry(
    tmp_path: Path,
) -> None:
    rows, _ = producer.hand_contract(PROJECT)
    arrays = producer.initializer_arrays(rows, "play_cards_0901_042")
    assert arrays["q_hand"].shape == (1, 2, 22)
    assert np.all(arrays["q_hand"][0] >= arrays["joint_lower_rad"])
    assert np.all(arrays["q_hand"][0] <= arrays["joint_upper_rad"])
    assert not {
        "q_arm",
        "T_world_base",
        "T_camera_base",
        "T_tool_hand_root",
        "T_target_hand_root_world",
    }.intersection(arrays)
    output = tmp_path / "ACCEPTED_HAND_STATES.npz"
    producer.atomic_npz(output, arrays)
    reloaded = producer.validate_initializer_npz(output, "play_cards_0901_042")
    np.testing.assert_array_equal(reloaded["q_hand"], arrays["q_hand"])


def test_full_fk_preserves_missing_as_nan_and_never_claims_collision() -> None:
    rows, _ = producer.hand_contract(PROJECT)
    count = 3
    observed = np.asarray([[True, False, True], [True, True, True]], dtype=bool)
    neutral = np.stack([row["neutral_q22_rad"] for row in rows])
    q = np.repeat(neutral[None], count, axis=0)
    q[~observed.T] = np.nan
    hawor = {
        "original_frame_indices": np.arange(count, dtype=np.int64),
        "fps": np.asarray(25.0),
        "observed": observed,
    }
    arrays, metrics = producer.full_fk_arrays(
        session_id="play_cards_0901_042",
        hawor=hawor,
        q_hand=q,
        q_hand_raw=q.copy(),
        valid_side_frame=observed,
        rows=rows,
    )
    assert arrays["q22"].shape == (count, 2, 22)
    assert arrays["fk_root_relative"].shape[:2] == (count, 2)
    assert np.array_equal(arrays["q22_valid"], observed.T)
    assert np.isnan(arrays["fk_root_relative"][1, 0]).all()
    assert arrays["fk_finite"][observed.T].all()
    assert arrays["hard_limit_pass"][observed.T].all()
    assert not arrays["self_collision_diagnostic_known"].any()
    assert metrics["full_fk_pass_valid_side_frames"] == int(observed.sum())
    assert metrics["self_collision_diagnostic_known_valid_side_frames"] == 0
