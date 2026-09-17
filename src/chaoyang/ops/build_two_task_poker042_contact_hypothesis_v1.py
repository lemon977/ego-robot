#!/usr/bin/env python3
"""Poker042 single-instance, selected-camera, hypothesis-only contact QA.

Usage: python src/chaoyang/ops/build_two_task_poker042_contact_hypothesis_v1.py \
  --output-dir archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1/stage_qa/poker042_contact/attempt_0001

The physical card remains a single instance. Only directly observed Object6D frames
are evaluated; missing evidence is UNKNOWN, never filled from Clean or attachment.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chaoyang.ops.run_contact10_selected_camera_r23 import TemporalContact

SESSION = "play_cards_0902_042"
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES = (4, 8, 12, 16, 20)
HAWOR_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/play_cards_0902_042/RESULT.json"
OBJECT_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260910_exact78_depth_object6d_expansion_v2/object6d_observed_only_v1/poker/play_cards_0902_042"
ADAPTER_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/selected_camera_adapter_batch_R7_3/poker/play_cards_0902_042"


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verified_ref(ref: dict) -> Path:
    path = Path(ref["path"])
    actual = artifact(path)
    if actual["sha256"] != ref["sha256"] or actual["bytes"] != ref["bytes"]:
        raise ValueError(f"SHA closure failed: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")

    hawor = read_json(HAWOR_RESULT)
    if hawor.get("session_id", hawor.get("session")) != SESSION:
        raise ValueError("HaWoR session mismatch")
    hawor_npz = verified_ref(hawor["outputs"]["npz"])
    with np.load(hawor_npz, allow_pickle=False) as h:
        frames = h["original_frame_indices"].astype(np.int64)
        joints = h["joints_3d_camera"].astype(np.float64)
        hand_valid = h["observed"].astype(bool)
        sides = tuple(str(x) for x in h["anatomical_side_names"].tolist())
        fps = float(h["fps"])
    if frames.shape != (171,) or not np.array_equal(frames, np.arange(171)):
        raise ValueError("Poker042 full 171-frame closure failed")
    if joints.shape != (2, 171, 21, 3) or hand_valid.shape != (2, 171) or sides != SIDES or fps <= 0:
        raise ValueError("HaWoR joint/side/fps contract failed")

    instances = []
    inputs = {"hawor_result": artifact(HAWOR_RESULT), "hawor_npz": artifact(hawor_npz)}
    for instance_id in range(1):
        object_result_path = OBJECT_ROOT / f"physical_object_{instance_id}/RESULT.json"
        adapter_result_path = ADAPTER_ROOT / f"physical_object_{instance_id}/RESULT.json"
        obj, adapter = read_json(object_result_path), read_json(adapter_result_path)
        if obj.get("session_id") != SESSION or adapter.get("session_id") != SESSION:
            raise ValueError(f"instance {instance_id} session mismatch")
        if obj.get("unobserved_pose_policy") != "KEEP_INVALID" or obj.get("propagated_frames") != 0:
            raise ValueError(f"instance {instance_id} must be directly observed only")
        spec_path = verified_ref(obj["inputs"]["spec"])
        spec = read_json(spec_path)
        if spec.get("global_physical_instance_id") != instance_id or spec.get("multi_instance_union_used") is not False:
            raise ValueError(f"instance {instance_id} global identity/union contract failed")
        if adapter.get("status") != "PASS_DEVELOPMENT_COORDINATE_ADAPTER" or adapter.get("authorized_scopes") != ["ROBOT_CONTACT_DEVELOPMENT_COORDINATE_INPUT"]:
            raise ValueError(f"instance {instance_id} selected-camera adapter scope failed")
        object_npz = verified_ref(obj["artifacts"]["trajectory"])
        adapter_npz = verified_ref(adapter["outputs"][0])
        if not any(ref["sha256"] == artifact(object_npz)["sha256"] for ref in adapter["inputs"]):
            raise ValueError(f"instance {instance_id} adapter Object6D input SHA mismatch")
        with np.load(object_npz, allow_pickle=False) as o:
            object_frames = o["frame_indices"].astype(np.int64)
            direct = o["valid"].astype(bool) & o["observed"].astype(bool)
            rectified = o["T_object_to_camera"].astype(np.float64)
            size_m = o["object_size_m"].astype(np.float64)
            physical_ids = o["physical_instance_id"].astype(np.int64)
        with np.load(adapter_npz, allow_pickle=False) as a:
            adapter_frames = a["frame_indices"].astype(np.int64)
            adapter_valid = a["valid"].astype(bool)
            selected = a["T_object_to_selected_camera"].astype(np.float64)
            copied_rectified = a["T_object_to_rectified_camera"].astype(np.float64)
            registration = a["T_stereo_rectified_camera_to_selected_camera"].astype(np.float64)
            source_domain = str(a["source_coordinate_domain"].item())
            target_domain = str(a["target_coordinate_domain"].item())
        if not (np.array_equal(frames, object_frames) and np.array_equal(frames, adapter_frames)):
            raise ValueError(f"instance {instance_id} frame mapping mismatch")
        if not np.array_equal(direct, adapter_valid) or not np.array_equal(rectified, copied_rectified):
            raise ValueError(f"instance {instance_id} adapter changes Object6D observations")
        if not np.allclose(selected[direct], registration @ rectified[direct], rtol=0, atol=1e-9):
            raise ValueError(f"instance {instance_id} selected-camera transform mismatch")
        if source_domain != "STEREO_RECTIFIED_DEPTH_CAMERA" or target_domain != "SELECTED_LEFT_RGB_CAMERA":
            raise ValueError(f"instance {instance_id} camera-domain mismatch")
        # Each per-instance NPZ is locally indexed as object 0. The immutable
        # spec's global_physical_instance_id fixes the target card instance.
        if direct.any() and not np.all(physical_ids[direct] == 0):
            raise ValueError(f"instance {instance_id} local physical index mismatch")
        if size_m.shape != (3,) or not np.isfinite(size_m).all() or np.any(size_m <= 0):
            raise ValueError(f"instance {instance_id} proxy size invalid")
        instances.append({"id": instance_id, "direct": direct, "selected": selected, "size_m": size_m})
        inputs[f"object{instance_id}_result"] = artifact(object_result_path)
        inputs[f"object{instance_id}_spec"] = artifact(spec_path)
        inputs[f"object{instance_id}_npz"] = artifact(object_npz)
        inputs[f"adapter{instance_id}_result"] = artifact(adapter_result_path)
        inputs[f"adapter{instance_id}_npz"] = artifact(adapter_npz)

    output.mkdir(parents=True)
    contact_file = output / "CONTACT_HYPOTHESES.jsonl"
    states = Counter()
    per_instance = {}
    with contact_file.open("w", encoding="utf-8") as sink:
        for instance in instances:
            tracker = TemporalContact()
            count = Counter()
            for local, frame_id in enumerate(frames.tolist()):
                direct = bool(instance["direct"][local])
                for side_i, side in enumerate(SIDES):
                    for finger_i, finger in enumerate(FINGERS):
                        available = direct and bool(hand_valid[side_i, local])
                        result = tracker.update(
                            (side_i, finger_i), frame_id, fps,
                            joints[side_i, local, TIP_INDICES[finger_i]] if available else None,
                            instance["selected"][local] if available else None,
                            instance["size_m"] if available else None,
                        )
                        row = {
                            "frame_id": frame_id, "physical_object_instance_id": "playing_card_0",
                            "hand_side": side, "finger_id": finger,
                            "direct_object_observation": direct,
                            "hand_observed": bool(hand_valid[side_i, local]),
                            "coordinate_domain": "SELECTED_LEFT_RGB_CAMERA",
                            "state": result["state"],
                            "distance_mm": result["distance_mm"],
                            "radial_velocity_mm_s": result["radial_velocity_mm_s"],
                            "tangential_velocity_mm_s": result["tangential_velocity_mm_s"],
                            "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN",
                            "contact_ground_truth": False,
                        }
                        sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        count[result["state"]] += 1
                        states[result["state"]] += 1
            per_instance[str(instance["id"])] = {
                "direct_object_frames": int(instance["direct"].sum()),
                "state_counts": dict(count),
            }
    result = {
        "schema_version": "two-task-poker042-contact-hypothesis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASSED_DEVELOPMENT_HYPOTHESIS_ONLY",
        "session_id": SESSION, "task": "poker", "frame_count": len(frames), "fps": fps,
        "object_instances": per_instance,
        "total_rows": len(frames) * 1 * 2 * len(FINGERS),
        "state_counts": dict(states),
        "coordinate_domain": "SELECTED_LEFT_RGB_CAMERA",
        "hand_geometry": "HaWoR MANO fingertip joint center proxy, not fingertip surface",
        "object_geometry": "single physical-card direct-observed Object6D shape proxy, no propagation; face ID UNKNOWN",
        "invalid_policy": "UNKNOWN; temporal state reset at every invalid frame or hand loss",
        "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN",
        "contact_ground_truth": False, "control_ground_truth": False,
        "training_eligible": False, "authority_promoted": False,
        "forbidden_inputs": ["Clean", "HAND_OBJECT_ATTACHMENT", "future frames"],
        "inputs": inputs,
        "outputs": {"contact_hypotheses": artifact(contact_file)},
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps({"result": str(output / "RESULT.json"), "status": result["status"], "instances": per_instance, "sha256": artifact(output / "RESULT.json")["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
