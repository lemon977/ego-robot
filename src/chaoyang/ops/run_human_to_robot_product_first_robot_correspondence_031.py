"""Real 031 66–81 rigid Robot surface correspondences from saved R0 q."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.pipeline.robot_rigid_correspondence_v1 import correspond_robot_pixel
from chaoyang.pipeline.v5_product import ProductRobotRenderer

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/robot_correspondence_031/window_v1"
R2 = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
MOTION = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
PREP = R2 / "lanes/lane1_scene/clean_candidate_031_wave5/SCENE_PREP_MANIFEST.json"
MOUNT = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _key(body: np.ndarray, link: np.ndarray) -> np.ndarray:
    return np.where(body >= 0, body.astype(np.int64) * 1024 + link.astype(np.int64) + 1, -1)


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    prep = load_json(PREP)
    domain = load_json(Path(prep["source_domain"]["path"]))
    with np.load(MOTION, allow_pickle=False) as archive:
        motion = {name: np.asarray(archive[name]) for name in archive.files}
    OUT.mkdir(parents=True)
    renderer = ProductRobotRenderer(ROOT, motion, domain, include_adapter=True)
    layers = []
    try:
        for frame in range(66, 82):
            layers.append(renderer.frame_layers(frame))
    finally:
        renderer.close()
    rows = []
    frame_rows = []
    mapping: dict[str, dict] = {}
    for frame, layer in zip(range(66, 82), layers, strict=True):
        for (body, link), name in layer.link_names.items():
            key = body * 1024 + link + 1
            mapping[str(key)] = {"body_id": body, "link_index": link, "urdf_link_name": name}
    for offset in range(15):
        frame = 66 + offset
        a, b = layers[offset], layers[offset + 1]
        ka, kb = _key(a.body_id, a.link_index), _key(b.body_id, b.link_index)
        fa = {body * 1024 + link + 1: matrix for (body, link), matrix in a.link_frames_camera.items()}
        fb = {body * 1024 + link + 1: matrix for (body, link), matrix in b.link_frames_camera.items()}
        ys, xs = np.nonzero(a.depth_valid)
        selected = (xs % 8 == 0) & (ys % 8 == 0)
        xs, ys = xs[selected], ys[selected]
        counts: dict[str, int] = {}
        overlay = a.rgb[..., ::-1].copy()
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            result = correspond_robot_pixel((x, y), a.optical_depth_m, ka,
                                            b.optical_depth_m, kb, renderer.k, fa, fb)
            result["source_frame_id"] = frame
            result["target_frame_id"] = frame + 1
            rows.append(result)
            reason = result["reason"]
            counts[reason] = counts.get(reason, 0) + 1
            if result["valid"]:
                cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)
            elif reason not in {"OTHER_LINK_OR_DISOCCLUDED", "TARGET_OUT_OF_FRAME"}:
                cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)
        if not cv2.imwrite(str(OUT / f"correspondence_{frame:06d}.png"), overlay):
            raise RuntimeError("OVERLAY_WRITE")
        frame_rows.append({"source_frame_id": frame, "attempted": len(xs), "reasons": counts})
    total = len(rows)
    matched = sum(bool(row["valid"]) for row in rows)
    result = {"schema_version": "HUMAN_TO_ROBOT_ROBOT_RIGID_CORRESPONDENCE_031_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031", "source_frames": list(range(66, 82)),
              "pairs": 15, "sample_stride_px": 8, "attempted": total, "matched_same_link": matched,
              "reason_counts": {reason: sum(row["reason"] == reason for row in rows)
                                for reason in sorted(set(row["reason"] for row in rows))},
              "frame_rows": frame_rows, "numeric_body_link_map": mapping,
              "rows": rows, "inputs": {"motion": artifact_ref(MOTION), "prep": artifact_ref(PREP),
                                        "mount_contract": artifact_ref(MOUNT)},
              "depth_semantics": "OPTICAL_Z_METRES", "correspondence_semantics": "ROBOT_RIGID_SURFACE_ONLY_NOT_MOVING_SCENE",
              "quality": "INTERNAL_RIGID_REGISTRATION_DIAGNOSTIC_NOT_EXTERNAL_TRUTH",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "attempted": total, "matched": matched,
                      "reason_counts": result["reason_counts"], "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
