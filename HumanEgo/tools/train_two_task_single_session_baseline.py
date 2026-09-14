#!/usr/bin/env python3
"""Train one current two-task baseline session with the frozen HumanEgo recipe.

The input bundle contains one task/session.  Train and validation examples are
different, non-overlapping temporal H50 supports from that same session.
Validation is therefore an optimization monitor only, never a cross-session
generalization or heldout result.

``--preflight-only`` is CPU-only and must be run first.  A real training launch
requires its immutable preflight receipt plus an already-owned central GPU
lease.  The 400-epoch recipe is preserved while durable execution is capped at
epoch 180, matching the previous time-bounded delivery policy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

HUMANEGO = Path(__file__).resolve().parents[1]
PROJECT = HUMANEGO.parent
CONTROL = PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
LIVE_AUTHORITY = CONTROL / "BASELINE_AUTHORITY.json"
GPU_LEASE = PROJECT / "_run/GPU_LEASE.json"
sys.path.insert(0, str(HUMANEGO))
sys.path.insert(0, str(PROJECT))

from preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar  # noqa: E402

_PREVIOUS_SPEC = importlib.util.spec_from_file_location(
    "frozen_previous_training_entry", HUMANEGO / "tools/train_newtask_robot.py"
)
if _PREVIOUS_SPEC is None or _PREVIOUS_SPEC.loader is None:
    raise RuntimeError("cannot import frozen previous training entry")
previous = importlib.util.module_from_spec(_PREVIOUS_SPEC)
_PREVIOUS_SPEC.loader.exec_module(previous)


HORIZON = 50
MAXIMUM_DURABLE_EPOCH = 180
TASKS = {
    "chips": "get_potato_chips_0902_034",
    "poker": "play_cards_0902_042",
}
FORBIDDEN_DATA_LINEAGE_MARKERS = (
    "exact78",
    "fresh78",
    "/humanego/artifacts/newtask_robot_bundles/",
    "fail_forward_prod",
    "fallback",
    "robot c",
    "robot_c",
)


class TrainingHold(RuntimeError):
    """Fail-closed training/package admission hold."""


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise TrainingHold(f"HOLD_NOT_ORDINARY_FILE:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingHold(f"HOLD_JSON_OBJECT_REQUIRED:{path}")
    return value


def verify_ref(value: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(str(value["path"])).resolve(strict=True)
        expected_size = int(value["bytes"])
        expected_sha = str(value["sha256"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise TrainingHold(f"HOLD_INVALID_REFERENCE:{label}") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected_size
        or sha256(path) != expected_sha
    ):
        raise TrainingHold(f"HOLD_REFERENCE_DRIFT:{label}:{path}")
    return path


def verify_pin(value: Mapping[str, Any], label: str) -> Path:
    """Verify a recipe/provenance pin that may omit a redundant byte count."""
    try:
        path = Path(str(value["path"])).resolve(strict=True)
        expected_sha = str(value["sha256"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise TrainingHold(f"HOLD_INVALID_PIN:{label}") from exc
    if not path.is_file() or path.is_symlink() or sha256(path) != expected_sha:
        raise TrainingHold(f"HOLD_PIN_DRIFT:{label}:{path}")
    if "bytes" in value and path.stat().st_size != int(value["bytes"]):
        raise TrainingHold(f"HOLD_PIN_SIZE_DRIFT:{label}:{path}")
    return path


def atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TrainingHold(f"HOLD_NO_CLOBBER:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _support(split: Mapping[str, Any], role: str) -> set[int]:
    row = split["temporal_split"][role]
    return set(range(int(row["frame_start"]), int(row["frame_stop_exclusive"])))


def _assert_current_data_lineage(paths: list[str]) -> None:
    for path in paths:
        lowered = path.lower()
        marker = next(
            (value for value in FORBIDDEN_DATA_LINEAGE_MARKERS if value in lowered),
            None,
        )
        if marker is not None:
            raise TrainingHold(
                f"HOLD_OLD_EXACT78_C_OR_FALLBACK_DATA_LINEAGE:{marker}:{path}"
            )


def _assert_recipe(recipe: Mapping[str, Any]) -> None:
    required = {
        "embodiment": "kai22",
        "hand_tracking_method": "hawor_v3",
        "hand_action_representation": "kaihand_joint_state",
        "pred_horizon": 50,
        "max_ict": 8,
        "image_size": [240, 320],
        "img_name": "rgb.png",
        "centric_mode": "object_centric",
        "frame_mode": "camera_frame",
        "action_mode": "absolute",
        "batch_size": 64,
        "eval_batch_size": 64,
        "epochs_in_recipe": 400,
        "maximum_durable_epoch_per_task": 180,
        "seed": 7,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "use_amp": True,
        "use_lr_schedule": True,
        "warmup_steps": 200,
        "min_lr_ratio": 0.05,
        "use_ema": True,
        "ema_decay": 0.999,
        "w_flow": 1.0,
        "w_pos": 5.0,
        "w_rot": 1.0,
        "w_hand_joint": 5.0,
        "w_joint_limit": 0.1,
        "w_velocity": 0.1,
        "w_done": 0.2,
        "eval_every": 5,
        "early_stop_patience": 12,
        "num_inference_steps": 20,
        "persistent_workers": True,
        "worker_cap": 8,
        "prefetch_factor": 4,
        "object_training_weight_for_grade_b": 0.25,
        "motion_training_weight_for_grade_b": 0.5,
        "contact_aux_weight": 0.0,
    }
    for key, expected in required.items():
        if recipe.get(key) != expected:
            raise TrainingHold(f"HOLD_EFFECTIVE_RECIPE_FIELD:{key}")


def verify_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve(strict=True)
    freeze_path = bundle / "freeze.json"
    freeze = read_json(freeze_path)
    if freeze.get("schema_version") != "humanego-two-task-single-session-freeze-v1":
        raise TrainingHold("HOLD_BUNDLE_FREEZE_SCHEMA")
    for label, value in freeze.get("bundle_files", {}).items():
        path = verify_ref(value, f"bundle freeze {label}")
        if path != (bundle / label).resolve(strict=True):
            raise TrainingHold(f"HOLD_BUNDLE_FILE_ESCAPES_ROOT:{label}")
    for label, value in freeze.get("payload_artifacts", {}).items():
        path = verify_ref(value, f"bundle payload {label}")
        if bundle not in path.parents:
            raise TrainingHold(f"HOLD_BUNDLE_PAYLOAD_ESCAPES_ROOT:{label}")

    split = read_json(bundle / "split.json")
    selector = read_json(bundle / "selector_records.json")
    paired = read_json(bundle / "paired_windows.json")
    sidecars = read_json(bundle / "sidecars.json")
    hawor_sidecars = read_json(bundle / "hawor_v3_sidecars.json")
    object_sidecars = read_json(bundle / "object_state_sidecars.json")
    lineage = read_json(bundle / "SOURCE_LINEAGE.json")
    recipe_payload = read_json(bundle / "RECIPE.json")
    task = str(split.get("task"))
    session = str(split.get("session"))
    if task not in TASKS or session != TASKS[task]:
        raise TrainingHold("HOLD_BUNDLE_TASK_SESSION")
    if freeze.get("task") != task or freeze.get("session") != session:
        raise TrainingHold("HOLD_FREEZE_TASK_SESSION")
    if (
        split.get("schema_version")
        != "humanego-two-task-single-session-temporal-split-v1"
        or split.get("seed") != 7
        or split.get("split_unit")
        != "non_overlapping_temporal_h50_support_within_one_session"
    ):
        raise TrainingHold("HOLD_TEMPORAL_SPLIT_CONTRACT")
    if paired.get("horizon") != HORIZON or paired.get("session") != session:
        raise TrainingHold("HOLD_PAIRED_WINDOW_CONTRACT")
    train_support = _support(split, "train")
    validation_support = _support(split, "validation")
    if train_support & validation_support:
        raise TrainingHold("HOLD_TRAIN_VALIDATION_TEMPORAL_OVERLAP")
    if len(train_support) < 51 or len(validation_support) < 51:
        raise TrainingHold("HOLD_TEMPORAL_SUPPORT_SHORTER_THAN_H50_PLUS_FUTURE")
    role_starts: dict[str, list[int]] = {}
    for role, support in (("train", train_support), ("validation", validation_support)):
        starts = paired.get("roles", {}).get(role, {}).get("window_starts")
        declared_support = paired.get("roles", {}).get(role, {}).get("support_frames")
        if not isinstance(starts, list) or not starts:
            raise TrainingHold(f"HOLD_EMPTY_{role.upper()}_WINDOWS")
        if set(declared_support or []) != support:
            raise TrainingHold(f"HOLD_{role.upper()}_SUPPORT_DECLARATION")
        for start in starts:
            if not set(range(int(start), int(start) + HORIZON + 1)).issubset(support):
                raise TrainingHold(f"HOLD_{role.upper()}_WINDOW_ESCAPES_SUPPORT:{start}")
        role_starts[role] = [int(value) for value in starts]
    if set(role_starts["train"]) & set(role_starts["validation"]):
        raise TrainingHold("HOLD_ROLE_WINDOW_START_OVERLAP")

    frames = selector.get("sessions", {}).get(session, {}).get("frames")
    if not isinstance(frames, dict) or set(frames) != {
        f"{value:05d}" for value in train_support | validation_support
    }:
        raise TrainingHold("HOLD_SELECTOR_FRAME_SET")
    for name, row in frames.items():
        for leaf in ("metadata", "image"):
            path = verify_ref(row[leaf], f"selector {name}/{leaf}")
            if bundle not in path.parents or path.stat().st_nlink != 1:
                raise TrainingHold(f"HOLD_SELECTOR_NOT_BUNDLE_ORDINARY:{path}")
    if set(sidecars) != {session} or set(hawor_sidecars) != {session} or set(object_sidecars) != {session}:
        raise TrainingHold("HOLD_SIDECAR_SESSION_SET")
    robot_sidecar = bundle / "sidecars/kai22" / session / "sidecar.npz"
    hawor_sidecar = bundle / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
    object_npz = bundle / "object_state_sidecars" / session / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    object_json = bundle / "object_state_sidecars" / session / "AUTO_ESTIMATED_OBJECT_STATE.json"
    expected = {
        robot_sidecar: sidecars[session],
        hawor_sidecar: hawor_sidecars[session],
        object_npz: object_sidecars[session]["npz_sha256"],
        object_json: object_sidecars[session]["json_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != digest or path.stat().st_nlink != 1:
            raise TrainingHold(f"HOLD_SIDECAR_DRIFT_OR_ALIAS:{path}")
    robot_validation = validate_sidecar(robot_sidecar, EMBODIMENTS["kai22"])
    object_manifest = read_json(object_json)
    if (
        object_manifest.get("quality_grade") != "B"
        or object_manifest.get("training_weight") != 0.25
        or object_manifest.get("consumption_authorized") is not True
        or object_manifest.get("claims_formal_object6d") is not False
    ):
        raise TrainingHold("HOLD_OBJECT_GRADE_B_TRAINING_CONTRACT")

    if lineage.get("task") != task or lineage.get("session") != session:
        raise TrainingHold("HOLD_LINEAGE_TASK_SESSION")
    for label in (
        "bundle_builder", "authority", "robot_result", "robot_agent_review",
        "robot_preflight", "robot_spec", "robot_kinematic_result",
        "robot_scene", "raw_video", "hawor_result", "hawor_npz",
        "mask_result", "object6d_result", "object6d_npz", "clean_result",
    ):
        verify_ref(lineage[label], f"lineage {label}")
    consumed_lineage_paths = [
        str(lineage[label]["path"])
        for label in (
            "robot_result", "robot_agent_review", "robot_scene", "raw_video",
            "hawor_result", "hawor_npz", "mask_result", "object6d_result",
            "object6d_npz", "clean_result",
        )
    ]
    _assert_current_data_lineage(consumed_lineage_paths)

    # A frozen snapshot proves build-time authority; current authority is also
    # rechecked so a later user/agent revocation stops training.
    current_authority = read_json(LIVE_AUTHORITY.resolve(strict=True))
    builder_spec = importlib.util.spec_from_file_location(
        "single_session_bundle_builder",
        HUMANEGO / "tools/build_two_task_single_session_e2e_bundle.py",
    )
    if builder_spec is None or builder_spec.loader is None:
        raise TrainingHold("HOLD_CANNOT_IMPORT_BUNDLE_AUTHORITY_GATE")
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    builder._authority_robot_row(
        current_authority,
        task=task,
        robot_result=Path(lineage["robot_result"]["path"]),
    )
    source_contract = verify_ref(
        recipe_payload["source_rebind_contract"], "recipe rebind snapshot"
    )
    contract = read_json(source_contract)
    recipe = recipe_payload.get("effective_recipe")
    if not isinstance(recipe, dict) or recipe != contract.get("frozen_effective_recipe"):
        raise TrainingHold("HOLD_RECIPE_SNAPSHOT_MISMATCH")
    _assert_recipe(recipe)
    for group in ("source_previous_manifests", "implementation_pins"):
        for label, value in recipe_payload[group].items():
            if isinstance(value, dict) and "path" in value:
                verify_pin(value, f"recipe {group}.{label}")

    return {
        "bundle": bundle,
        "freeze_path": freeze_path,
        "freeze": freeze,
        "task": task,
        "session": session,
        "split": split,
        "selector": selector,
        "selector_frames": frames,
        "role_starts": role_starts,
        "sidecar_sha": sidecars[session],
        "hawor_sha": hawor_sidecars[session],
        "object_npz_sha": object_sidecars[session]["npz_sha256"],
        "object_json_sha": object_sidecars[session]["json_sha256"],
        "object_manifest": object_manifest,
        "lineage": lineage,
        "recipe": recipe,
        "robot_validation": robot_validation,
    }


def effective_config(verified: Mapping[str, Any], out_dir: Path):
    split = verified["split"]
    recipe = verified["recipe"]
    yaml_path = Path(
        read_json(verified["bundle"] / "RECIPE.json")["implementation_pins"]
        ["actual_base_yaml"]["path"]
    ).resolve(strict=True)
    adapter = (
        verified["bundle"] / "production" / verified["session"]
        / "09_humanego_adapter"
    )
    cfg = previous.load_config(yaml_path, {
        "task": verified["task"],
        "MPS_PATHS_TRAIN": [str(adapter)],
        "MPS_PATHS_EVAL": [str(adapter)],
        "pretrained_checkpoint": str(Path(
            read_json(verified["bundle"] / "RECIPE.json")["implementation_pins"]
            ["pretrained_checkpoint"]["path"]
        )),
        "robot_sidecar_root": split["sidecar_root"],
        "batch_size": recipe["batch_size"],
        "eval_batch_size": recipe["eval_batch_size"],
        "epochs": recipe["epochs_in_recipe"],
        "seed": recipe["seed"],
        "lr": recipe["lr"],
        "image_size": recipe["image_size"],
        "img_name": recipe["img_name"],
        "use_legacy_image_loading": False,
        "persistent_workers": True,
        "out_dir": str(out_dir),
        "run_root": str(out_dir.parent),
        "device": "cuda",
        "make_video": False,
    })
    checks = {
        "pred_horizon": recipe["pred_horizon"],
        "max_ict": recipe["max_ict"],
        "hand_tracking_method": recipe["hand_tracking_method"],
        "hand_action_representation": recipe["hand_action_representation"],
        "centric_mode": recipe["centric_mode"],
        "frame_mode": recipe["frame_mode"],
        "action_mode": recipe["action_mode"],
        "batch_size": recipe["batch_size"],
        "eval_batch_size": recipe["eval_batch_size"],
        "epochs": recipe["epochs_in_recipe"],
        "seed": recipe["seed"],
        "lr": recipe["lr"],
        "weight_decay": recipe["weight_decay"],
        "grad_clip": recipe["grad_clip"],
        "use_amp": recipe["use_amp"],
        "use_lr_schedule": recipe["use_lr_schedule"],
        "warmup_steps": recipe["warmup_steps"],
        "min_lr_ratio": recipe["min_lr_ratio"],
        "use_ema": recipe["use_ema"],
        "ema_decay": recipe["ema_decay"],
        "w_flow": recipe["w_flow"],
        "w_pos": recipe["w_pos"],
        "w_rot": recipe["w_rot"],
        "w_hand_joint": recipe["w_hand_joint"],
        "w_joint_limit": recipe["w_joint_limit"],
        "w_velocity": recipe["w_velocity"],
        "w_done": recipe["w_done"],
        "eval_every": recipe["eval_every"],
        "early_stop_patience": recipe["early_stop_patience"],
        "num_inference_steps": recipe["num_inference_steps"],
    }
    for field, expected in checks.items():
        if getattr(cfg, field) != expected:
            raise TrainingHold(f"HOLD_EFFECTIVE_CONFIG_DRIFT:{field}")
    augmentation = {
        key: getattr(cfg, key)
        for key in (
            "enable_augmentation", "enable_aug_img", "enable_aug_rrc",
            "enable_aug_target_jittering", "enable_aug_cutout",
            "enable_aug_temporal_stride", "enable_aug_interpolation",
        )
    }
    if augmentation != {
        "enable_augmentation": True,
        "enable_aug_img": True,
        "enable_aug_rrc": True,
        "enable_aug_target_jittering": True,
        "enable_aug_cutout": True,
        "enable_aug_temporal_stride": False,
        "enable_aug_interpolation": False,
    }:
        raise TrainingHold("HOLD_AUGMENTATION_RECIPE_DRIFT")
    return cfg, yaml_path, augmentation


def make_dataset(
    *, verified: Mapping[str, Any], cfg, role: str, stats: Mapping[str, Any],
    images: bool, augmentation: bool,
):
    session = verified["session"]
    bundle = verified["bundle"]
    adapter = bundle / "production" / session / "09_humanego_adapter"
    return previous.FlowMatchingDataloader(
        sessions=[previous.MPSSessions(str(adapter))],
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand,
        single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name if images else None,
        centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        hawor_v3_sidecar_root=verified["split"]["hawor_v3_sidecar_root"],
        hawor_v3_sha256_by_session={session: verified["hawor_sha"]},
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=augmentation,
        enable_aug_img=cfg.enable_aug_img if augmentation else False,
        enable_aug_rrc=cfg.enable_aug_rrc if augmentation else False,
        enable_aug_target_jittering=(
            cfg.enable_aug_target_jittering if augmentation else False
        ),
        enable_aug_cutout=cfg.enable_aug_cutout if augmentation else False,
        enable_aug_temporal_stride=(
            cfg.enable_aug_temporal_stride if augmentation else False
        ),
        enable_aug_interpolation=(
            cfg.enable_aug_interpolation if augmentation else False
        ),
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=False,
        cache_json_in_memory=cfg.cache_json_in_memory,
        cache_image_bytes_in_memory=cfg.cache_image_bytes_in_memory if images else False,
        seed=cfg.seed,
        stats=stats,
        allowed_window_starts={session: set(verified["role_starts"][role])},
        selector_records={session: verified["selector_frames"]},
        selector_root=verified["selector"]["selector_root"],
        sidecar_sha256_by_session={session: verified["sidecar_sha"]},
        object_state_sidecar_root=verified["split"]["object_state_sidecar_root"],
        object_state_npz_sha256_by_session={session: verified["object_npz_sha"]},
        object_state_json_sha256_by_session={session: verified["object_json_sha"]},
        object_state_consumption_mode="training_estimated_grade_b",
        object_state_confidence_thresholds=verified["object_manifest"]
        ["confidence_thresholds"],
    )


def cpu_preflight(bundle: Path, output: Path) -> dict[str, Any]:
    verified = verify_bundle(bundle)
    cfg, yaml_path, augmentation = effective_config(verified, output.parent / "unused")
    unit_stats = {"pos": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}}
    datasets = {
        role: make_dataset(
            verified=verified,
            cfg=cfg,
            role=role,
            stats=unit_stats,
            images=False,
            augmentation=False,
        )
        for role in ("train", "validation")
    }
    rows = {}
    for role, dataset in datasets.items():
        if len(dataset) != len(verified["role_starts"][role]):
            raise TrainingHold(f"HOLD_{role.upper()}_LOADER_WINDOW_COUNT")
        sample = dataset[0]
        if not isinstance(sample, dict):
            raise TrainingHold(f"HOLD_{role.upper()}_LOADER_SAMPLE_TYPE")
        rows[role] = {
            "windows": len(dataset),
            "first_window": verified["role_starts"][role][0],
            "last_window": verified["role_starts"][role][-1],
            "sample_tensor_shapes": {
                key: list(value.shape)
                for key, value in sample.items()
                if hasattr(value, "shape")
            },
            "object_training_weight": float(sample["object_training_weight"]),
        }
    if rows["train"]["object_training_weight"] != 0.25 or rows["validation"]["object_training_weight"] != 0.25:
        raise TrainingHold("HOLD_OBJECT_WEIGHT_NOT_0_25")
    report = {
        "schema_version": "humanego-two-task-single-session-epoch0-preflight-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PASS_CPU_EPOCH0_NO_TRAINING",
        "task": verified["task"],
        "session": verified["session"],
        "bundle_freeze": ref(verified["freeze_path"]),
        "training_entry": ref(Path(__file__)),
        "base_training_entry": ref(Path(previous.__file__)),
        "base_yaml": ref(yaml_path),
        "pretrained_checkpoint": ref(Path(cfg.pretrained_checkpoint)),
        "effective_recipe": verified["recipe"],
        "augmentation": augmentation,
        "temporal_split": verified["split"]["temporal_split"],
        "support_intersection": sorted(
            _support(verified["split"], "train")
            & _support(verified["split"], "validation")
        ),
        "loader": rows,
        "same_session_validation_claim": (
            "OPTIMIZATION_MONITORING_ONLY_NOT_CROSS_SESSION_GENERALIZATION"
        ),
        "independent_checkpoint_per_task_required": True,
        "legacy_exact78_training_data_consumed": False,
        "grade_c_or_fallback_consumed": False,
        "gpu_used": False,
        "training_started": False,
        "maximum_durable_epoch": MAXIMUM_DURABLE_EPOCH,
        "claim_limit": (
            "CPU loader and lineage admission only; no checkpoint exists and "
            "within-session validation is not heldout evidence."
        ),
    }
    if report["support_intersection"]:
        raise TrainingHold("HOLD_PREFLIGHT_TEMPORAL_SUPPORT_INTERSECTION")
    atomic_new_json(output, report)
    return report


def verify_preflight(path: Path, verified: Mapping[str, Any]) -> dict[str, Any]:
    report = read_json(path.resolve(strict=True))
    if (
        report.get("status") != "PASS_CPU_EPOCH0_NO_TRAINING"
        or report.get("task") != verified["task"]
        or report.get("session") != verified["session"]
        or report.get("support_intersection") != []
        or report.get("legacy_exact78_training_data_consumed") is not False
        or report.get("grade_c_or_fallback_consumed") is not False
        or report.get("training_started") is not False
    ):
        raise TrainingHold("HOLD_CPU_PREFLIGHT_NOT_ADMISSIBLE")
    if verify_ref(report["bundle_freeze"], "preflight bundle freeze") != verified["freeze_path"]:
        raise TrainingHold("HOLD_PREFLIGHT_BUNDLE_MISMATCH")
    if verify_ref(report["training_entry"], "preflight training entry") != Path(__file__).resolve(strict=True):
        raise TrainingHold("HOLD_PREFLIGHT_ENTRY_CODE_DRIFT")
    return report


def verify_gpu_lease(holder: str) -> dict[str, Any]:
    lease = read_json(GPU_LEASE.resolve(strict=True))
    if (
        lease.get("status") != "ACQUIRED"
        or lease.get("holder") != holder
        or lease.get("scope", {}).get("gpu_indices") != [0]
    ):
        raise TrainingHold("HOLD_CENTRAL_GPU_LEASE_NOT_OWNED")
    return lease


def run_training(
    *, bundle: Path, run_root: Path, tag: str, preflight_path: Path,
    lease_holder: str, resume: bool,
) -> int:
    verified = verify_bundle(bundle)
    verify_preflight(preflight_path, verified)
    lease = verify_gpu_lease(lease_holder)
    out_dir = run_root.absolute() / tag
    if resume:
        if not out_dir.is_dir():
            raise TrainingHold("HOLD_RESUME_RUN_MISSING")
    elif out_dir.exists() or out_dir.is_symlink():
        raise TrainingHold(f"HOLD_FRESH_RUN_NO_CLOBBER:{out_dir}")
    else:
        out_dir.mkdir(parents=True)
    cfg, yaml_path, augmentation = effective_config(verified, out_dir)
    session = verified["session"]
    run_manifest = {
        "schema_version": "humanego-two-task-single-session-run-v1",
        "task": verified["task"],
        "session": session,
        "embodiment": "kai22",
        "bundle": str(verified["bundle"]),
        "bundle_freeze": ref(verified["freeze_path"]),
        "source_lineage": ref(verified["bundle"] / "SOURCE_LINEAGE.json"),
        "cpu_epoch0_preflight": ref(preflight_path),
        "temporal_split": verified["split"]["temporal_split"],
        "validation_claim": (
            "WITHIN_SESSION_OPTIMIZATION_MONITORING_ONLY_NOT_CROSS_SESSION_"
            "GENERALIZATION_OR_CANONICAL_HELDOUT"
        ),
        "canonical_heldout_consumed": False,
        "legacy_exact78_training_data_consumed": False,
        "grade_c_or_fallback_consumed": False,
        "task_specific_checkpoint": True,
        "pretrained": ref(Path(cfg.pretrained_checkpoint)),
        "base_yaml": ref(yaml_path),
        "base_training_entry": ref(Path(previous.__file__)),
        "training_entry": ref(Path(__file__)),
        "effective_recipe": verified["recipe"],
        "augmentation": augmentation,
        "epochs_in_recipe": 400,
        "maximum_durable_epoch": MAXIMUM_DURABLE_EPOCH,
        "effective_batch_size": 64,
        "seed": 7,
        "central_gpu_lease_at_launch": lease,
        "contact_aux_weight": 0.0,
        "object_training_weight": 0.25,
        "claim_limit": (
            "Single-session baseline optimization checkpoint only; no "
            "cross-session generalization, canonical heldout, contact, or "
            "real-robot deployment claim."
        ),
    }
    manifest_path = out_dir / "run_manifest.json"
    if resume:
        existing_manifest = read_json(manifest_path)
        # The lease payload may be renewed; every other field must be exact.
        left = dict(existing_manifest)
        right = dict(run_manifest)
        left.pop("central_gpu_lease_at_launch", None)
        right.pop("central_gpu_lease_at_launch", None)
        left.pop("dataset_stats_sha256", None)
        right.pop("dataset_stats_sha256", None)
        if left != right:
            raise TrainingHold("HOLD_RESUME_RUN_MANIFEST_MISMATCH")
        run_manifest = existing_manifest
    else:
        atomic_new_json(manifest_path, run_manifest)
        atomic_new_json(out_dir / "split.json", verified["split"])

    stats_path = out_dir / "dataset_stats.json"
    if resume:
        stats = read_json(stats_path)
    else:
        # The exact previous stats implementation is retained; only the
        # allowed train temporal starts differ.
        stats = previous.get_dataset_stats(
            cfg.MPS_PATHS_TRAIN,
            cfg,
            {session: set(verified["role_starts"]["train"])},
            {session: verified["selector_frames"]},
            verified["selector"]["selector_root"],
            {session: verified["sidecar_sha"]},
            verified["split"]["hawor_v3_sidecar_root"],
            {session: verified["hawor_sha"]},
        )
        atomic_new_json(stats_path, stats)
        run_manifest["dataset_stats_sha256"] = sha256(stats_path)
        replace_json(manifest_path, run_manifest)
    if run_manifest.get("dataset_stats_sha256") != sha256(stats_path):
        raise TrainingHold("HOLD_DATASET_STATS_SHA_DRIFT")

    ds_train = make_dataset(
        verified=verified,
        cfg=cfg,
        role="train",
        stats=stats,
        images=True,
        augmentation=True,
    )
    ds_eval = make_dataset(
        verified=verified,
        cfg=cfg,
        role="validation",
        stats=stats,
        images=True,
        augmentation=False,
    )
    if len(ds_train) == 0 or len(ds_eval) == 0:
        raise TrainingHold("HOLD_EMPTY_TRAIN_OR_VALIDATION_LOADER")
    previous._tune_cuda()
    workers = min(int(cfg.num_workers or 8), int(verified["recipe"]["worker_cap"]))
    loader_kwargs = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": min(
                int(getattr(cfg, "prefetch_factor", 4) or 4),
                int(verified["recipe"]["prefetch_factor"]),
            ),
        })
    train_loader = previous._CudaPrefetch(
        previous.DataLoader(
            ds_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            **loader_kwargs,
        ),
        cfg.device,
    )
    eval_loader = previous._CudaPrefetch(
        previous.DataLoader(
            ds_eval,
            batch_size=cfg.eval_batch_size or cfg.batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        cfg.device,
    )
    model = previous.FlowMatchingModel(
        single_hand=cfg.single_hand,
        pred_horizon=cfg.pred_horizon,
        max_ict=cfg.max_ict,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        vision_embed_dim=cfg.vision_embed_dim,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        horizon_weighting=getattr(
            cfg, "model_horizon_weighting",
            getattr(cfg, "model_h_weighting", "uniform"),
        ),
        horizon_beta=getattr(
            cfg, "model_horizon_beta", getattr(cfg, "model_h_beta", 0.0)
        ),
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn,
        use_pre_norm=cfg.use_pre_norm,
        use_ctx_norm=cfg.use_ctx_norm,
        use_done_in_flow=cfg.use_done_in_flow,
        hand_action_representation=cfg.hand_action_representation,
    ).to(cfg.device)
    latest_path = out_dir / "latest.pt"
    if not resume:
        transfer = previous.load_compatible_pretrained(
            model,
            str(Path(cfg.pretrained_checkpoint)),
            str(out_dir / "pretrained_load_report.json"),
        )
        if float(transfer["loaded_parameter_fraction"]) <= 0.0:
            raise TrainingHold("HOLD_PRETRAINED_ZERO_PARAMETER_TRANSFER")

    optimizer = previous.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    ema = previous.EMAModel(model, decay=cfg.ema_decay) if cfg.use_ema else None
    scaler = previous.torch.amp.GradScaler("cuda", enabled=bool(cfg.use_amp))
    best = {"score": 1e9, "epoch": -1}
    no_improve = 0
    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []
    if resume:
        checkpoint = previous.torch.load(latest_path, map_location=cfg.device)
        previous.validate_resume_checkpoint(
            checkpoint,
            run_manifest=run_manifest,
            dataset_stats_sha256=run_manifest["dataset_stats_sha256"],
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best = checkpoint.get("best", best)
        no_improve = int(checkpoint.get("evaluations_without_improvement", 0))
        if ema is not None and checkpoint.get("ema_shadow") is not None:
            ema.load_state_dict(checkpoint["ema_shadow"])
        history_payload = read_json(out_dir / "train_history.json")
        history = history_payload.get("rows", [])
        if not isinstance(history, list):
            raise TrainingHold("HOLD_RESUME_TRAIN_HISTORY_SCHEMA")
    if start_epoch > MAXIMUM_DURABLE_EPOCH:
        raise TrainingHold("HOLD_RESUME_ALREADY_AT_DURABLE_EPOCH_LIMIT")

    total_steps = int(cfg.epochs * max(1, len(train_loader)))
    started = time.time()
    terminal = "TIME_BOUNDED_COMPLETE"
    terminal_epoch = start_epoch - 1
    for epoch in range(start_epoch, MAXIMUM_DURABLE_EPOCH + 1):
        verify_gpu_lease(lease_holder)
        ds_train.set_epoch(epoch)
        train_row, global_step = previous.train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            epoch,
            global_step,
            total_steps,
            ema,
        )
        eval_row: dict[str, Any] = {}
        if epoch % cfg.eval_every == 0:
            if ema is not None:
                ema.apply_shadow()
            try:
                eval_row = previous.eval_ode_inference(
                    model, eval_loader, cfg=cfg, max_batches=cfg.max_eval_batches
                )
            finally:
                if ema is not None:
                    ema.restore()
            score = (
                eval_row["pos_err_w_m"]
                + eval_row["rot_err_w_deg"] / 100.0
                + eval_row.get("joint_mae_w_normalized", 0.0) * 0.10
            )
            if score < best["score"]:
                best = {"score": float(score), "epoch": int(epoch)}
                no_improve = 0
                if ema is not None:
                    ema.apply_shadow()
                previous.save_checkpoint_atomic(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "cfg": cfg.__dict__,
                        "best": best,
                        "global_step": global_step,
                        "run_manifest": run_manifest,
                        "model_weights": "ema",
                        "dataset_stats_sha256": run_manifest["dataset_stats_sha256"],
                    },
                    str(out_dir / "best.pt"),
                )
                if ema is not None:
                    ema.restore()
            else:
                no_improve += 1
        previous.save_checkpoint_atomic(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "best": best,
                "global_step": global_step,
                "run_manifest": run_manifest,
                "model_weights": "train",
                "dataset_stats_sha256": run_manifest["dataset_stats_sha256"],
                "evaluations_without_improvement": no_improve,
                "ema_shadow": ema.state_dict() if ema is not None else None,
            },
            str(latest_path),
        )
        terminal_epoch = epoch
        history.append({"epoch": epoch, **train_row, **eval_row, "best": best})
        replace_json(out_dir / "train_history.json", {"rows": history})
        if (
            cfg.early_stop_patience is not None
            and epoch % cfg.eval_every == 0
            and no_improve >= cfg.early_stop_patience
        ):
            terminal = "EARLY_STOP_COMPLETE_WITHIN_SESSION_MONITORING"
            break
    complete = {
        "schema_version": "humanego-two-task-single-session-training-terminal-v1",
        "status": terminal,
        "task": verified["task"],
        "session": session,
        "terminal_epoch": terminal_epoch,
        "epochs_in_recipe": 400,
        "maximum_durable_epoch": MAXIMUM_DURABLE_EPOCH,
        "best": best,
        "elapsed_s": time.time() - started,
        "run_root": str(out_dir),
        "best_checkpoint": ref(out_dir / "best.pt"),
        "latest_checkpoint": ref(latest_path),
        "heldout_claim": "NONE_WITHIN_SESSION_VALIDATION_ONLY",
        "claim_limit": (
            "Not 400-epoch completion unless terminal_epoch is 400; not "
            "cross-session generalization or deployment authority."
        ),
    }
    atomic_new_json(out_dir / "training_complete.json", complete)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--lease-holder")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        if (
            args.preflight_output is None
            or args.preflight_report is not None
            or args.run_root is not None
            or args.tag is not None
            or args.lease_holder is not None
            or args.resume
        ):
            parser.error(
                "--preflight-only requires only --bundle and --preflight-output"
            )
        report = cpu_preflight(args.bundle, args.preflight_output.absolute())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if any(value is None for value in (
        args.preflight_report, args.run_root, args.tag, args.lease_holder
    )) or args.preflight_output is not None:
        parser.error(
            "training requires --preflight-report --run-root --tag --lease-holder"
        )
    return run_training(
        bundle=args.bundle,
        run_root=args.run_root,
        tag=args.tag,
        preflight_path=args.preflight_report,
        lease_holder=args.lease_holder,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
