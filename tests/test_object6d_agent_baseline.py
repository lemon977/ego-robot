from __future__ import annotations

from pathlib import Path
import json
import sys

import cv2
import numpy as np


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import promote_object6d_agent_baseline as promoter  # noqa: E402
import run_visual_object6d_with_review as orchestrator  # noqa: E402


def test_agent_baseline_object6d_transaction_and_promotion(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    assert orchestrator.synthetic_test(synthetic_root)["status"] == "PASS"
    spec = synthetic_root / "source_chips/chips/INPUT_SPEC.json"
    output = tmp_path / "agent_output"
    result = orchestrator.orchestrate(spec, output, agent_baseline_review=True)
    assert result["agent_baseline_review"] is True
    review, baseline = promoter.audit(output, "chips", "synthetic_chips")
    assert review["grade"] == "B"
    assert review["downstream_authorized"] is True
    assert baseline["authorized_scopes"] == ["CLEAN_BASELINE", "ROBOT_VISUAL_BASELINE"]
    assert baseline["robot_contact_authorized"] is False
    numeric_result = json.loads(
        (output / "numeric/RESULT.json").read_text(encoding="utf-8")
    )
    for reference in numeric_result["artifacts"].values():
        assert ".partial" not in reference["path"]
        assert Path(reference["path"]).is_file()


def test_numeric_object6d_marks_occlusion_without_fabricating_direct_depth(
    tmp_path: Path,
) -> None:
    spec_path, output = orchestrator.numeric.make_synthetic_case(tmp_path, "chips")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest_path = Path(spec["rgb_object_mask_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mask_path = Path(manifest["frames"][4]["mask"]["path"])
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert cv2.imwrite(str(mask_path), np.zeros_like(mask))
    manifest["frames"][4]["mask"] = orchestrator.numeric.artifact(mask_path)
    manifest_path.unlink()
    orchestrator.numeric.atomic_json(manifest_path, manifest)
    spec["rgb_object_mask_manifest"] = orchestrator.numeric.artifact(manifest_path)
    spec_path.unlink()
    orchestrator.numeric.atomic_json(spec_path, spec)

    result = orchestrator.numeric.produce(spec_path, output)
    assert result["status"].startswith("PASS")
    assert result["observed_frames"] == 9
    assert result["propagated_frames"] == 1
    assert result["max_propagated_gap_frames"] == 1
    assert result["dense_pose_rate_metrics"]["translation_step_max_m"] <= 0.030001
    assert result["dense_pose_rate_metrics"]["rotation_step_max_deg"] <= 12.001
    with np.load(output / "VISUAL_FIXED_INSTANCE_OBJECT6D.npz", allow_pickle=False) as values:
        assert values["valid"].all()
        assert not bool(values["observed"][4])
        assert values["visibility"][4] == 0.0
        assert np.isnan(values["observed_near_far_optical_z_m"][4]).all()
        assert np.isfinite(values["analytic_near_far_optical_z_m"][4]).all()


def test_numeric_object6d_keeps_preidentity_prefix_invalid(tmp_path: Path) -> None:
    spec_path, output = orchestrator.numeric.make_synthetic_case(tmp_path, "poker")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    mask_manifest_path = Path(spec["rgb_object_mask_manifest"]["path"])
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))

    onset = 4
    for row in mask_manifest["frames"][:onset]:
        mask_path = Path(row["mask"]["path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert cv2.imwrite(str(mask_path), np.zeros_like(mask))
        row["valid"] = False
        row["mask"] = orchestrator.numeric.artifact(mask_path)
    mask_manifest_path.unlink()
    orchestrator.numeric.atomic_json(mask_manifest_path, mask_manifest)
    spec["rgb_object_mask_manifest"] = orchestrator.numeric.artifact(
        mask_manifest_path
    )
    spec["leading_unobserved_policy"] = "INVALID_UNTIL_VISUAL_IDENTITY_ONSET"
    spec_path.unlink()
    orchestrator.numeric.atomic_json(spec_path, spec)

    result = orchestrator.numeric.produce(spec_path, output)
    assert result["status"].startswith("PASS")
    assert result["active_start_local_frame"] == onset
    assert result["invalid_pre_identity_frames"] == onset
    assert result["valid_frames"] == result["frame_count"] - onset
    assert result["propagated_frames"] == 0
    assert result["observed_fraction"] == 1.0
    with np.load(output / "VISUAL_FIXED_INSTANCE_OBJECT6D.npz", allow_pickle=False) as values:
        valid = values["valid"].astype(bool)
        observed = values["observed"].astype(bool)
        instances = values["physical_instance_id"].astype(np.int64)
        assert not valid[:onset].any()
        assert not observed[:onset].any()
        assert np.all(instances[:onset] == -1)
        assert valid[onset:].all()
        assert observed[onset:].all()
        assert np.all(instances[onset:] == 0)


def test_numeric_object6d_bounds_direct_world_pose_steps() -> None:
    numeric = orchestrator.numeric
    observed = np.ones(3, dtype=bool)
    world = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    world[:, 0, 3] = [0.0, 0.10, 0.20]
    for index, angle in enumerate((0.0, 90.0, 180.0)):
        world[index, :3, :3], _ = cv2.Rodrigues(
            np.asarray([0.0, 0.0, np.radians(angle)], dtype=np.float64)
        )
    camera = world.copy()
    analytic = np.zeros((3, 2), dtype=np.float64)
    c2w = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    rows = [{}, {}, {}]
    metrics = numeric._bound_direct_pose_steps(
        observed,
        world,
        camera,
        analytic,
        c2w,
        np.arange(3, dtype=np.int64),
        np.asarray([0.05, 0.038, 0.003], dtype=np.float64),
        rows,
    )
    assert metrics["raw_translation_step_max_m"] >= 0.099
    assert metrics["raw_rotation_step_max_deg"] >= 89.9
    assert metrics["bounded_translation_step_max_m"] <= 0.030001
    assert metrics["bounded_rotation_step_max_deg"] <= 12.001
    dense = numeric._dense_world_pose_step_metrics(world)
    assert dense["translation_step_max_m"] <= 0.030001
    assert dense["rotation_step_max_deg"] <= 12.001

    gap_observed = np.asarray([True, False, False, True])
    gap_world = np.tile(np.eye(4, dtype=np.float64), (4, 1, 1))
    gap_world[3, :3, :3], _ = cv2.Rodrigues(
        np.asarray([0.0, 0.0, np.radians(36.0)], dtype=np.float64)
    )
    gap_camera = gap_world.copy()
    gap_analytic = np.zeros((4, 2), dtype=np.float64)
    numeric._fill_occluded_poses(
        gap_observed,
        gap_world,
        gap_camera,
        gap_analytic,
        np.tile(np.eye(4, dtype=np.float64), (4, 1, 1)),
        np.arange(4, dtype=np.int64),
        np.asarray([0.05, 0.038, 0.003], dtype=np.float64),
    )
    gap_dense = numeric._dense_world_pose_step_metrics(gap_world)
    assert gap_dense["rotation_step_max_deg"] <= 12.001
