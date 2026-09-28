"""Sensor097 raw-to-Kai22 provenance and field-scoped learning qualification."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.huro_hand_only_retarget_v1 import keypoints_from_q, load_hand_model
from chaoyang.pipeline.pico_manus_motion_v2 import MANUS_NAMES, MAP_25_TO_21
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

TASK = "human_to_robot_result_breakthrough_20260924"
OUT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/data"
RAW = Path("/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916/cards_130_0916/097/dataset.hdf5")
HAND = REPO_ROOT / ("_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/"
                    "sensor/run_0001/play_cards_0916_097/HAND_MOTION_V1.npz")
BACKEND = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/"
                       "attempt_0001/lanes/lane3_sensor/run_097_wave0/KAI22_COMMON_BACKEND_V1.npz")
OLD = REPO_ROOT / ("_run/current/human_to_robot_quality_acceptance_20260924/attempts/"
                   "attempt_0001/lanes/sensor")


def _npz(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def main():
    if (OUT / "SENSOR_097_LOCAL_KAI22_QUALIFICATION.json").exists():
        raise FileExistsError("QUALIFICATION_EXISTS")
    packet = load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    if packet["task_id"] != TASK:
        raise RuntimeError("TASK_NOT_REGISTERED")
    hand, backend = _npz(HAND), _npz(BACKEND)
    with h5py.File(RAW, "r") as f:
        if f.attrs["hand_frame"] != "wrist-local (relative to MANUS wrist root)":
            raise RuntimeError("RAW_MANUS_COORDINATE_CONVENTION_CHANGED")
        if tuple(json.loads(f.attrs["joint_names"])) != MANUS_NAMES:
            raise RuntimeError("RAW_MANUS_NAMES_CHANGED")
        calibration = json.loads(f.attrs["manus_calibration"])
        raw_local = np.stack([f["left_hand_joints"][:], f["right_hand_joints"][:]], axis=1)
        raw_valid = np.stack([f["left_hand_valid"][:], f["right_hand_valid"][:]], axis=1).astype(bool)
    if (raw_local.shape != (165, 2, 25, 3)
            or not np.array_equal(raw_local, hand["manus_local_25_m"], equal_nan=True)
            or not np.array_equal(raw_valid, hand["manus_hand_valid"])):
        raise RuntimeError("RAW_TO_HANDMOTION_DRIFT")
    if (not np.array_equal(hand["manus25_to_21"], MAP_25_TO_21)
            or not np.array_equal(backend["human_to_physical"], [1, 0])):
        raise RuntimeError("NODE_OR_SIDE_MAPPING_DRIFT")
    assets = load_pinned_robot_assets(REPO_ROOT)
    hands = (load_hand_model(assets.left_hand.path, "left"),
             load_hand_model(assets.right_hand.path, "right"))
    valid = backend["valid"].astype(bool)
    if valid.shape != (165, 2) or not valid.any():
        raise RuntimeError("BACKEND_VALIDITY_DRIFT")
    rows = []
    for anatomical in range(2):
        physical = int(backend["human_to_physical"][anatomical])
        model = hands[physical]
        if not np.array_equal(backend["joint_names"][physical], model.joint_names):
            raise RuntimeError("JOINT_ORDER_DRIFT")
        source = raw_local[:, anatomical]
        root_mcp = np.linalg.norm(source[:, 5] - source[:, 0], axis=1)
        index_proximal = np.linalg.norm(source[:, 6] - source[:, 5], axis=1)
        fk_errors, limit = [], []
        robot_root_mcp, robot_proximal = [], []
        for frame in np.flatnonzero(valid[:, physical]):
            q = backend["q22"][frame, physical]
            fk = keypoints_from_q(model, q)
            fk -= fk[0]
            fk_errors.append(float(np.max(np.abs(fk - backend["fk21_root_relative"][frame, physical]))))
            limit.append(bool(np.all(q >= model.lower - 1e-9) and np.all(q <= model.upper + 1e-9)))
            robot_root_mcp.append(float(np.linalg.norm(fk[5] - fk[0])))
            robot_proximal.append(float(np.linalg.norm(fk[6] - fk[5])))
        if not all(limit) or max(fk_errors) > 1e-8:
            raise RuntimeError("Q_LIMIT_OR_STORED_FK_DRIFT")
        time = hand["timestamp_ns"].astype(np.int64)
        delta = np.diff(time)
        edge = (np.diff(hand["frame_id"]) == 1) & (delta > 0) & (delta <= 2.5 * np.median(delta))
        anchors = [start for start in range(165-50)
                   if valid[start:start+51, physical].all() and edge[start:start+50].all()]
        rows.append({"anatomical_side": str(hand["anatomical_side_names"][anatomical]),
                     "physical_side": model.side, "valid_frames": int(valid[:, physical].sum()),
                     "structural_h50_windows": len(anchors),
                     "source_root_index_mcp_median_m": float(np.median(root_mcp)),
                     "source_index_mcp_pip_median_m": float(np.median(index_proximal)),
                     "robot_root_index_mcp_median_m": float(np.median(robot_root_mcp)),
                     "robot_index_mcp_pip_median_m": float(np.median(robot_proximal)),
                     "source_to_robot_root_ratio": float(np.median(root_mcp) / np.median(robot_root_mcp)),
                     "source_to_robot_proximal_ratio": float(np.median(index_proximal) / np.median(robot_proximal)),
                     "independent_fk_max_abs_m": max(fk_errors),
                     "joint_limit_pass_frames": int(sum(limit))})
    forward = load_json(OLD / "SENSOR_097_QONLY_REAL_RGB_FORWARD.json")
    kernel = load_json(OLD / "SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER.json")
    if (forward["optimizer_steps"] != 0 or forward["training_admitted_elements"] != 0
            or not kernel["invalid_payload_and_flow_input_invariant"]
            or not kernel["zero_supervision_rejected"]):
        raise RuntimeError("FROZEN_CONSUMER_EVIDENCE_DRIFT")
    result = {"schema_version": "SENSOR_097_LOCAL_KAI22_FIELD_QUALIFICATION_V1",
              "task_id": TASK, "session_id": "play_cards_0916_097",
              "source_hdf5": artifact_ref(RAW), "hand_motion": artifact_ref(HAND),
              "backend": artifact_ref(BACKEND),
              "prior_real_rgb_forward": artifact_ref(OLD / "SENSOR_097_QONLY_REAL_RGB_FORWARD.json"),
              "prior_mask_loss_check": artifact_ref(OLD / "SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER.json"),
              "source_coordinate_frame": "wrist-local MANUS node positions, not Kai22 joint angles",
              "manus_calibration_sha256": {side: calibration[side]["sha256"] for side in ("left", "right")},
              "raw_to_saved_max_abs_m": 0.0, "rows": rows,
              "interface_pass": True, "label_quality_pass": False,
              "qualified_h50_windows": 0, "optimizer_steps": 0,
              "reason": "Raw MANUS is faithfully copied and q/FK is structurally valid, but root-to-MCP and MCP-to-PIP require incompatible scale factors; no independent per-finger label-quality gate or visual correspondence proves this Kai22 q is task-valid.",
              "training_eligible": False, "control_ground_truth": False,
              "next_action": "METHOD_LEVEL_MANUS_CALIBRATION_OR_INDEPENDENT_FINGER_LABEL_AUTHORITY_REQUIRED; DO_NOT_REBRAND_EXISTING_Q"}
    atomic_json(OUT / "SENSOR_097_LOCAL_KAI22_QUALIFICATION.json", result)
    print(json.dumps({"interface_pass": True, "label_quality_pass": False,
                      "qualified_h50_windows": 0, "structural_h50_windows": [r["structural_h50_windows"] for r in rows],
                      "receipt": str(OUT / "SENSOR_097_LOCAL_KAI22_QUALIFICATION.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
