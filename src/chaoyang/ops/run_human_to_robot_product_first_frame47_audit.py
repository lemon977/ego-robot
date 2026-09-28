"""Independent 031 frame-47 q/FK/target/render consumption audit; no new IK."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets
from chaoyang.pipeline.v5_product import ProductRobotRenderer

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/frame47_audit_031/v1"
R2 = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
MOTION = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
PREP = R2 / "lanes/lane1_scene/clean_candidate_031_wave5/SCENE_PREP_MANIFEST.json"
MOUNT = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
FRAMES = (46, 47, 48, 49, 80)


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(MOTION, allow_pickle=False) as archive:
        motion = {name: np.asarray(archive[name]) for name in archive.files}
    prep = load_json(PREP)
    domain = load_json(Path(prep["source_domain"]["path"]))
    mount = load_json(MOUNT)
    contract = np.asarray([mount["transforms"]["left_flange_to_hand_root"],
                           mount["transforms"]["right_flange_to_hand_root"]], np.float64)
    if not np.allclose(contract, motion["T_flange_hand"], rtol=0, atol=1e-10):
        raise ValueError("MOUNT_CONTRACT_MISMATCH")
    assets = load_pinned_robot_assets(ROOT)
    renderer = ProductRobotRenderer(ROOT, motion, domain, include_adapter=True)
    OUT.mkdir(parents=True)
    rows = []
    try:
        for frame in FRAMES:
            layer = renderer.frame_layers(frame)
            frame_row = {"source_frame_id": frame, "per_side": []}
            for side in range(2):
                valid = bool(motion["wrist_valid"][frame, side])
                if not valid:
                    frame_row["per_side"].append({"side": side, "valid": False,
                                                  "reason": "FROZEN_INPUT_INVALID_NO_EVALUATION"})
                    continue
                q = np.asarray(motion["q_arm"][frame], np.float64).copy()
                names = ARM_JOINT_NAMES[side]
                if not np.isfinite(q[side]).all():
                    raise ValueError(f"VALID_SIDE_Q_NONFINITE:{frame}:{side}")
                # The other missing side has no source authority; neutral is
                # an FK API placeholder only, never a rendered observation.
                values = {name: float(q[s, joint]) for s, group in enumerate(ARM_JOINT_NAMES)
                          for joint, name in enumerate(group) if np.isfinite(q[s, joint])}
                for s, group in enumerate(ARM_JOINT_NAMES):
                    for joint, name in enumerate(group):
                        values.setdefault(name, 0.0)
                fk = forward_kinematics(assets.tianji, values)
                flange = fk[("flange_L", "flange_R")[side]]
                analytic = motion["T_cam_base"] @ flange @ contract[side]
                rendered = motion["T_cam_base"] @ layer.T_world_hand[side]
                saved = motion["T_actual_root_cam"][frame, side]
                target = motion["T_target_root_cam"][frame, side]
                errors = {"analytic_vs_saved_mm": float(np.linalg.norm(analytic[:3, 3] - saved[:3, 3]) * 1000),
                          "render_vs_saved_mm": float(np.linalg.norm(rendered[:3, 3] - saved[:3, 3]) * 1000),
                          "target_residual_independent_mm": float(np.linalg.norm(analytic[:3, 3] - target[:3, 3]) * 1000)}
                saved_residual = float(np.linalg.norm(saved[:3, 3] - target[:3, 3]) * 1000)
                frame_row["per_side"].append({"side": side, "valid": True,
                    "q_arm_rad": q[side].tolist(), "q_joint_names": list(names),
                    "target_camera_m": target[:3, 3].tolist(),
                    "analytic_fk_camera_m": analytic[:3, 3].tolist(),
                    "render_camera_m": rendered[:3, 3].tolist(),
                    "saved_actual_camera_m": saved[:3, 3].tolist(),
                    "saved_residual_recomputed_mm": saved_residual,
                    **errors})
            rows.append(frame_row)
    finally:
        renderer.close()
    right = {row["source_frame_id"]: next(item for item in row["per_side"] if item["side"] == 1)
             for row in rows}
    target_jump = None
    if right[47]["valid"] and right[48]["valid"]:
        target_jump = float(np.linalg.norm(np.asarray(right[48]["target_camera_m"]) -
                                           np.asarray(right[47]["target_camera_m"])) * 1000)
    result = {"schema_version": "HUMAN_TO_ROBOT_031_FRAME47_INDEPENDENT_FK_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031", "frames": list(FRAMES),
              "rows": rows, "frame47_to48_target_jump_mm": target_jump,
              "classification": "TARGET_OR_INPUT_OUTLIER_IF_FK_AND_RENDER_CLOSE;NO_NEW_IK",
              "inputs": {"motion": artifact_ref(MOTION), "mount": artifact_ref(MOUNT),
                         "prep": artifact_ref(PREP)},
              "claim_limit": "Independent fixed-URDF/camera/mount FK and actual renderer consumption; no external target truth or reachability certificate.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": "AUDITED", "frame47": right[47],
                      "target_jump_47_48_mm": target_jump, "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
