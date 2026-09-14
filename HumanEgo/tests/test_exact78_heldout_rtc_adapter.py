from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def adapter_module():
    path = Path(__file__).resolve().parents[1] / "tools/run_exact78_heldout_rtc.py"
    spec = importlib.util.spec_from_file_location("exact78_heldout_adapter_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_rtc_values_and_each_task_has_exactly_five_heldout() -> None:
    module = adapter_module()
    contract = module.read_json(module.CONTRACT)
    assert contract["rtc"] == {
        "pred_horizon": 50,
        "execution_horizon": 10,
        "inference_delay": 4,
        "attention_schedule": "exp",
        "required": True,
    }
    for task in ("chips", "poker"):
        eligible = contract["tasks"][task]["eligible_heldout_sessions"]
        canonical = [row["session_id"] for row in module.cohort_rows(task)
                     if row["split"] == "heldout"]
        assert eligible == canonical
        assert len(eligible) == 5 == len(set(eligible))


def test_check_only_waits_without_deserializing_missing_checkpoint() -> None:
    module = adapter_module()
    for task in ("chips", "poker"):
        report = module.preflight(task, None)
        assert report["status"] in {"WAIT_CHECKPOINT", "READY"}
        assert report["canonical_role"] == "heldout"
        assert report["training_isolation"]["heldout_in_training"] is False
        assert report["input_policy"]["heldout_robot_or_gt_q_loaded"] is False


def test_noncanonical_requested_session_is_rejected() -> None:
    module = adapter_module()
    with pytest.raises(RuntimeError, match="NOT_CANONICAL_HELDOUT"):
        module.preflight("chips", "get_potato_chips_0901_001")


def test_training_bundle_materialization_of_heldout_is_rejected(tmp_path: Path) -> None:
    module = adapter_module()
    root = tmp_path / "artifacts/newtask_robot_bundles/bad"
    root.mkdir(parents=True)
    (root / "split.json").write_text(json.dumps({
        "task": "chips", "admitted_splits": {
            "train": ["get_potato_chips_0903_013"],
            "validation": [], "test": [],
        }, "selector_sessions": ["get_potato_chips_0903_013"],
    }), encoding="utf-8")
    module.HUMANEGO = tmp_path
    with pytest.raises(RuntimeError, match="HELDOUT_IN_TRAINING_BUNDLE"):
        module.training_leakage_audit("chips", "get_potato_chips_0903_013")


def test_prediction_npz_schema_contains_predictions_only(tmp_path: Path) -> None:
    module = adapter_module()
    path = tmp_path / "heldout_checkpoint_predictions.npz"
    np.savez_compressed(
        path,
        checkpoint_full_plan_action=np.zeros((2, 50, 62), np.float32),
        causally_executed_prefix=np.zeros((2, 10, 62), np.float32),
        observation_frame=np.asarray([0, 10]), rtc_applied=np.asarray([False, True]),
        contains_ground_truth=np.asarray(False),
        contains_heldout_robot_q=np.asarray(False),
    )
    assert module.validate_prediction_npz_schema(path)["status"] == "PASS_PREDICTION_ONLY_SCHEMA"


def test_v2_prediction_schema_requires_and_validates_safe_prefix(tmp_path: Path) -> None:
    module = adapter_module()
    path = tmp_path / "heldout_checkpoint_predictions.npz"
    lower = np.full((2, 22), -1.0, np.float32)
    upper = np.full((2, 22), 1.0, np.float32)
    np.savez_compressed(
        path,
        schema_version=np.asarray("exact78-heldout-prediction-only-v2"),
        checkpoint_full_plan_action=np.zeros((2, 50, 62), np.float32),
        raw_model_prefix=np.zeros((2, 10, 62), np.float32),
        causally_executed_prefix=np.zeros((2, 10, 62), np.float32),
        observation_frame=np.asarray([0, 10]), rtc_applied=np.asarray([False, True]),
        contains_ground_truth=np.asarray(False), contains_heldout_robot_q=np.asarray(False),
        execution_safety_applied=np.asarray(True), joint_lower=lower,
        joint_upper=upper,
        physical_to_human_side=np.asarray(["right", "left"]),
    )
    assert module.validate_prediction_npz_schema(path)["status"] == "PASS_PREDICTION_ONLY_SCHEMA"


def test_hawor_bootstrap_is_crossed_into_physical_kai_slots() -> None:
    module = adapter_module()
    identity = np.eye(4, dtype=np.float64)
    left = identity.copy(); left[0, 3] = -0.2
    right = identity.copy(); right[0, 3] = 0.3

    class Dataset:
        samples = ["frame"]
        pos_mean = np.zeros(3, dtype=np.float32)
        pos_std = np.ones(3, dtype=np.float32)

        @staticmethod
        def _read_frame(_):
            return {"entities": {"hands_hawor_v3": {
                "left": {"T_hand_to_world": left},
                "right": {"T_hand_to_world": right},
            }}}

        @staticmethod
        def _get_T_w2ref(_):
            return np.eye(4, dtype=np.float64)

    state, valid = module.state_from_hawor(
        Dataset(), 0, np.zeros((2, 22), dtype=np.float32),
    )
    assert valid.tolist() == [True, True]
    assert state[0, 0] == pytest.approx(0.3)  # physical Kai left <- human right
    assert state[1, 0] == pytest.approx(-0.2)  # physical Kai right <- human left


def test_safety_filter_bounds_pose_steps_and_raw_radian_q() -> None:
    module = adapter_module()

    class Dataset:
        pos_mean = np.zeros(3, dtype=np.float32)
        pos_std = np.ones(3, dtype=np.float32)

    identity_o6d = np.eye(3, dtype=np.float32)[:, :2].reshape(6)
    state = np.zeros((2, 33), dtype=np.float32)
    state[:, 3:9] = identity_o6d
    plan = np.zeros((50, 62), dtype=np.float32)
    plan[:, :6] = 0.5
    plan[:, 6:12] = identity_o6d
    plan[:, 12:18] = identity_o6d
    plan[:, 18:] = 5.0
    lower = np.full((2, 22), -1.0, dtype=np.float32)
    upper = np.full((2, 22), 1.0, dtype=np.float32)
    q_controller = module.RecedingH50QController(
        22, execute_steps=10, ensemble_decay=0.5,
        max_q_step_rad=module.MAX_Q_STEP_RAD,
    )
    pose_controller = module.CausalPoseRateLimiter(
        max_translation_step_m=module.MAX_WRIST_STEP_M,
        max_rotation_step_rad=module.MAX_WRIST_ROTATION_STEP_RAD,
        ema_alpha=module.WRIST_EMA_ALPHA,
    )
    executed, diagnostics, initialized = module.safety_filter_prefix(
        plan=plan, frame_number=0, current_state=state,
        current_mask=np.ones(2, dtype=bool), c2w=np.eye(4), dataset=Dataset(),
        schema={"joint_lower": lower, "joint_upper": upper},
        q_controller=q_controller, pose_controller=pose_controller,
        pose_initialized=False,
    )
    assert initialized
    q = executed[:, 18:]
    assert np.all(q >= lower.reshape(1, -1))
    assert np.all(q <= upper.reshape(1, -1))
    assert np.max(np.abs(np.diff(np.vstack([np.zeros((1, 44)), q]), axis=0))) <= module.MAX_Q_STEP_RAD + 1e-6
    positions = executed[:, :6].reshape(10, 2, 3)
    steps = np.linalg.norm(np.diff(np.concatenate([
        np.zeros((1, 2, 3)), positions,
    ], axis=0), axis=0), axis=-1)
    assert float(steps.max()) <= module.MAX_WRIST_STEP_M + 1e-6
    assert diagnostics["raw_joint_limit_ratio"] == pytest.approx(1.0)


def test_prediction_npz_schema_rejects_target_or_gt_q(tmp_path: Path) -> None:
    module = adapter_module()
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        checkpoint_full_plan_action=np.zeros((2, 50, 62), np.float32),
        causally_executed_prefix=np.zeros((2, 10, 62), np.float32),
        observation_frame=np.asarray([0, 10]), rtc_applied=np.asarray([False, True]),
        contains_ground_truth=np.asarray(False), contains_heldout_robot_q=np.asarray(False),
        target_action=np.zeros((2, 50, 62), np.float32),
    )
    with pytest.raises(RuntimeError, match="HOLD_PREDICTION_SCHEMA"):
        module.validate_prediction_npz_schema(path)


def write_checkpoint_gate_fixture(
    module, root: Path, completion_status: str,
    *, payload_manifest_drift: bool = False, payload_stats_drift: bool = False,
) -> Path:
    root.mkdir(parents=True)
    stats = root / "dataset_stats.json"
    stats.write_text('{"pos":{"mean":[0,0,0],"std":[1,1,1]}}\n', encoding="utf-8")
    session = "get_potato_chips_0903_013"
    manifest = {
        "schema_version": "humanego-newtask-robot-run-v3-exact78-object-ict",
        "task": "chips",
        "exact78_split_audit": {
            "cohort_sha256": module.COHORT_SHA256,
            "heldout_in_training": False,
        },
        "heldout_sessions": [session],
        "train_sessions": ["train"],
        "validation_sessions": ["validation"],
        "test_sessions": [],
        "dataset_stats_sha256": module.sha256(stats),
    }
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    checkpoint_manifest = dict(manifest)
    if payload_manifest_drift:
        checkpoint_manifest["task"] = "poker"
    checkpoint = root / "best.pt"
    torch.save({
        "run_manifest": checkpoint_manifest,
        "dataset_stats_sha256": (
            "0" * 64 if payload_stats_drift else module.sha256(stats)
        ),
    }, checkpoint)
    (root / "training_complete.json").write_text(json.dumps({
        "status": completion_status,
        "verification": {
            "best_sha256": module.sha256(checkpoint),
            "run_manifest_sha256": module.sha256(manifest_path),
            "dataset_stats_sha256": module.sha256(stats),
        },
    }), encoding="utf-8")
    return checkpoint


@pytest.mark.parametrize(
    "status", ["COMPLETE", "TIME_BOUNDED_COMPLETE", "NATURAL_EARLY_STOP_VERIFIED"],
)
def test_checkpoint_gate_accepts_only_reviewed_bounded_terminal_statuses(
    tmp_path: Path, status: str,
) -> None:
    module = adapter_module()
    checkpoint = write_checkpoint_gate_fixture(module, tmp_path / status, status)
    report = module.validate_checkpoint_files(
        "chips", checkpoint, "get_potato_chips_0903_013",
    )
    assert report["ready"] is True
    assert report["completion_status"] == status
    assert report["dataset_stats"]["sha256"] == module.sha256(
        checkpoint.parent / "dataset_stats.json"
    )


@pytest.mark.parametrize("status", ["complete", "RUNNING", "TRUNCATED", None])
def test_checkpoint_gate_rejects_legacy_or_nonterminal_status(
    tmp_path: Path, status: str | None,
) -> None:
    module = adapter_module()
    checkpoint = write_checkpoint_gate_fixture(module, tmp_path / str(status), status)
    with pytest.raises(RuntimeError, match="HOLD_TRAINING_COMPLETE_STATUS"):
        module.validate_checkpoint_files(
            "chips", checkpoint, "get_potato_chips_0903_013",
        )


@pytest.mark.parametrize("drift", ["manifest", "stats"])
def test_checkpoint_gate_rejects_best_payload_binding_drift(
    tmp_path: Path, drift: str,
) -> None:
    module = adapter_module()
    checkpoint = write_checkpoint_gate_fixture(
        module, tmp_path / drift, "TIME_BOUNDED_COMPLETE",
        payload_manifest_drift=drift == "manifest",
        payload_stats_drift=drift == "stats",
    )
    with pytest.raises(RuntimeError, match="CHECKPOINT_.*MISMATCH"):
        module.validate_checkpoint_files(
            "chips", checkpoint, "get_potato_chips_0903_013",
        )


def test_checkpoint_sha_mismatch_rejected_before_unsafe_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = adapter_module()
    checkpoint = write_checkpoint_gate_fixture(
        module, tmp_path / "bad_sha", "TIME_BOUNDED_COMPLETE",
    )
    completion = json.loads((checkpoint.parent / "training_complete.json").read_text())
    completion["verification"]["best_sha256"] = "0" * 64
    (checkpoint.parent / "training_complete.json").write_text(
        json.dumps(completion), encoding="utf-8",
    )
    monkeypatch.setattr(
        module.torch, "load",
        lambda *args, **kwargs: pytest.fail("torch.load reached before SHA rejection"),
    )
    with pytest.raises(RuntimeError, match="TRUSTED_CHECKPOINT_SHA_MISMATCH"):
        module.validate_checkpoint_files(
            "chips", checkpoint, "get_potato_chips_0903_013",
        )


def test_sha_bound_local_checkpoint_uses_explicit_weights_only_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = adapter_module()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"safe": True}, checkpoint)
    original = torch.load
    observed: dict[str, object] = {}

    def checked_load(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.torch, "load", checked_load)
    payload, reference = module.load_sha_bound_local_checkpoint(
        checkpoint, module.sha256(checkpoint),
    )
    assert payload == {"safe": True}
    assert reference["sha256"] == module.sha256(checkpoint)
    assert observed["weights_only"] is False
