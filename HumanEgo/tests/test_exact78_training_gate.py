from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np


TRAIN_ENTRY = Path(__file__).resolve().parents[1] / "tools/train_newtask_robot.py"
SPEC = importlib.util.spec_from_file_location("exact78_train_entry", TRAIN_ENTRY)
assert SPEC is not None and SPEC.loader is not None
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)

GATE_DAEMON = Path(__file__).resolve().parents[2] / "tools/run_exact78_training_gate_daemon.py"
GATE_SPEC = importlib.util.spec_from_file_location("exact78_training_gate_daemon", GATE_DAEMON)
assert GATE_SPEC is not None and GATE_SPEC.loader is not None
gate_daemon = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(gate_daemon)


def _canonical_split(task: str) -> dict:
    cohort = json.loads(entry.EXACT78_COHORT.read_text(encoding="utf-8"))
    values = {
        role: [
            row["session_id"] for row in cohort["sessions"]
            if row["task"] == task and row["split"] == role
        ]
        for role in ("train", "validation", "test", "heldout")
    }
    values.update({
        "task": task,
        "admitted_splits": {
            "train": values["train"][:16],
            "validation": values["validation"][:4],
            "test": [],
        },
    })
    return values


def test_exact78_minimum_subset_preserves_canonical_roles() -> None:
    report = entry.exact78_split_binding(_canonical_split("chips"))
    assert report["canonical_counts"] == {
        "train": 60, "validation": 8, "test": 5, "heldout": 5,
    }
    assert report["admitted_counts"] == {
        "train": 16, "validation": 4, "test": 0,
    }
    assert report["heldout_in_training"] is False


def test_exact78_rejects_heldout_moved_into_training() -> None:
    split = _canonical_split("poker")
    split["admitted_splits"]["train"][0] = split["heldout"][0]
    with pytest.raises(RuntimeError, match="ROLE_LEAKAGE|HELDOUT_LEAKAGE"):
        entry.exact78_split_binding(split)


def test_legacy_hand_only_bundle_schema_is_rejected(tmp_path: Path) -> None:
    split = _canonical_split("chips")
    split.update({
        "schema_version": "humanego-newtask-robot-split-v2-hawor-ict",
        "ict_contract": {
            "hand_tracking_method": "hawor_v3",
            "single_hand": False,
            "ict_dim": 29,
            "frame_mode": "camera_frame",
            "translation_unit": "metre",
            "camera_coordinate_system": "x_right_y_down_z_forward",
            "object_token_policy": "ABSENT_NO_ADMITTED_OBJECT6D_AUTHORITY",
        },
    })
    with pytest.raises(RuntimeError, match="legacy/hand-only bundles are forbidden"):
        entry.hawor_binding(tmp_path, split)


def _robot_window_fixture(tmp_path: Path, lengths: dict[str, int]) -> tuple[dict, dict]:
    sidecar_root = tmp_path / "sidecars"
    production_root = tmp_path / "production"
    split = {
        "sidecar_root": str(sidecar_root),
        "production_root": str(production_root),
        "hawor_v3_sidecar_root": str(tmp_path / "hawor"),
        "admitted_splits": {
            "train": ["session_train"],
            "validation": ["session_validation"],
            "test": ["session_test"],
        },
        "motion_label_contract": {
            "consumption_mode": "MOTION_TRAINING_LABEL_GRADE_B",
            "provenance_class": "MOTION_TRAINING_LABEL_GRADE_B",
            "target": "fresh_wrist_T_camera_plus_q_hand",
            "arm_consumed": False,
            "achieved_robot_fk_required": False,
            "motion_weight_by_session": {session: 0.5 for session in lengths},
            "contact_aux_weight_by_session": {session: 0.0 for session in lengths},
            "sidecar_sha256_by_session": {session: None for session in lengths},
            "receipt_sha256_by_session": {session: "0" * 64 for session in lengths},
        },
    }
    windows = {}
    for session, count in lengths.items():
        adapter = production_root / session / "09_humanego_adapter/preprocess/all_data"
        for index in range(count):
            frame = adapter / f"{index:05d}"
            frame.mkdir(parents=True)
            (frame / "training_data.json").write_text("{}")
        sidecar = sidecar_root / "kai22" / session
        sidecar.mkdir(parents=True)
        np.savez_compressed(
            sidecar / "sidecar.npz",
            frame_names=np.asarray([f"{index:05d}" for index in range(count)]),
            q=np.zeros((count, 2, 22), dtype=np.float32),
            wrist_T_camera=np.repeat(
                np.eye(4, dtype=np.float64)[None, None], count, axis=0
            ).repeat(2, axis=1),
            valid=np.ones((count, 2), dtype=bool),
            confidence=np.ones((count, 2), dtype=np.float32),
            provenance_class=np.asarray("MOTION_TRAINING_LABEL_GRADE_B"),
            legacy_wrist_discarded=np.asarray(True),
            empirical_continuity_verified=np.asarray(True),
            joint_lower=np.full((2, 22), -1.0, dtype=np.float32),
            joint_upper=np.full((2, 22), 1.0, dtype=np.float32),
            translation_unit=np.asarray("metre"),
            q_unit=np.asarray("radian"),
        )
        split["motion_label_contract"]["sidecar_sha256_by_session"][session] = (
            entry._sha256_file(sidecar / "sidecar.npz")
        )
        fresh = tmp_path / "hawor" / session
        fresh.mkdir(parents=True)
        np.savez_compressed(
            fresh / "entities_hawor_v3.npz",
            frame_names=np.asarray([f"{index:05d}" for index in range(count)]),
            T_hand_to_camera=np.repeat(
                np.eye(4, dtype=np.float64)[None, None], count, axis=0
            ).repeat(2, axis=1),
            valid=np.ones((count, 2), dtype=bool),
        )
        windows[session] = set(range(max(0, count - 49)))
    return split, {"window_starts": windows}


def test_robot_window_gate_rejects_24_frame_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, binding = _robot_window_fixture(tmp_path, {
        "session_train": 24, "session_validation": 64, "session_test": 64,
    })
    monkeypatch.setattr(
        entry, "MINIMUM_VALID_WINDOWS",
        {"train": 1, "validation": 1, "test": 1},
    )
    with pytest.raises(RuntimeError, match="ZERO_H50_WINDOWS"):
        entry.robot_window_gate(split, binding)


def test_motion_label_gate_accepts_fresh_bound_64_frame_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, binding = _robot_window_fixture(tmp_path, {
        "session_train": 64, "session_validation": 64, "session_test": 64,
    })
    monkeypatch.setattr(
        entry, "MINIMUM_VALID_WINDOWS",
        {"train": 1, "validation": 1, "test": 1},
    )
    report = entry.robot_window_gate(split, binding)
    assert report["status"] == "PASS_MOTION_LABEL_H50_WINDOW_GATE"
    assert report["counts"] == {"train": 15, "validation": 15, "test": 15}


def test_motion_label_gate_rejects_wrist_not_bound_to_fresh_hawor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, binding = _robot_window_fixture(tmp_path, {
        "session_train": 64, "session_validation": 64, "session_test": 64,
    })
    path = Path(split["sidecar_root"]) / "kai22/session_train/sidecar.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["wrist_T_camera"] = payload["wrist_T_camera"].copy()
    payload["wrist_T_camera"][0, 0, 0, 3] = 0.01
    np.savez_compressed(path, **payload)
    split["motion_label_contract"]["sidecar_sha256_by_session"]["session_train"] = (
        entry._sha256_file(path)
    )
    monkeypatch.setattr(
        entry, "MINIMUM_VALID_WINDOWS",
        {"train": 1, "validation": 1, "test": 1},
    )
    with pytest.raises(RuntimeError, match="WRIST_NOT_FRESH_HAWOR"):
        entry.robot_window_gate(split, binding)


def test_motion_label_gate_rejects_formal_robot_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, binding = _robot_window_fixture(tmp_path, {
        "session_train": 64, "session_validation": 64, "session_test": 64,
    })
    path = Path(split["sidecar_root"]) / "kai22/session_train/sidecar.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["provenance_class"] = np.asarray("FORMAL_ROBOT_PRODUCT")
    np.savez_compressed(path, **payload)
    split["motion_label_contract"]["sidecar_sha256_by_session"]["session_train"] = (
        entry._sha256_file(path)
    )
    monkeypatch.setattr(
        entry, "MINIMUM_VALID_WINDOWS",
        {"train": 1, "validation": 1, "test": 1},
    )
    with pytest.raises(RuntimeError, match="PROVENANCE"):
        entry.robot_window_gate(split, binding)


def test_detached_resume_is_bound_to_same_bundle_and_stats() -> None:
    manifest = {"schema_version": "v3", "bundle": "/fresh/chips"}
    checkpoint = {
        "run_manifest": manifest,
        "dataset_stats_sha256": "stats-sha",
    }
    entry.validate_resume_checkpoint(
        checkpoint,
        run_manifest=manifest,
        dataset_stats_sha256="stats-sha",
    )
    with pytest.raises(RuntimeError, match="RUN_MANIFEST_MISMATCH"):
        entry.validate_resume_checkpoint(
            checkpoint,
            run_manifest={**manifest, "bundle": "/stale/chips"},
            dataset_stats_sha256="stats-sha",
        )
    with pytest.raises(RuntimeError, match="DATASET_STATS_MISMATCH"):
        entry.validate_resume_checkpoint(
            checkpoint,
            run_manifest=manifest,
            dataset_stats_sha256="different-stats-sha",
        )


def test_terminal_completion_supersedes_retained_finalizing_evidence() -> None:
    assert not gate_daemon.has_active_training(
        running=False,
        finalizing=True,
        awaiting_completion_verification=False,
        complete=True,
    )
    assert gate_daemon.has_active_training(
        running=False,
        finalizing=True,
        awaiting_completion_verification=False,
        complete=False,
    )
