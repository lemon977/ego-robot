#!/usr/bin/env python3
"""Development-only CONTACT-10 successor with explicit camera-domain and gap gates.

This does not change observed-only Object6D or promote contact authority.  It
compares the archived R22 hypothesis with a selected-camera calculation.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.contact_geometry_v2 import oriented_box_sdf  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES = (4, 8, 12, 16, 20)
SIDES = ("left", "right")


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(reference: dict[str, object]) -> Path:
    path = Path(str(reference["path"]))
    observed = artifact(path)
    if observed["bytes"] != reference.get("bytes") or observed["sha256"] != reference.get("sha256"):
        raise ValueError(f"artifact closure mismatch: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def contact_state(distance_mm: float, radial_mm_s: float | None, tangent_mm_s: float | None, touching: bool) -> str:
    if abs(distance_mm) <= 8.0:
        if touching and tangent_mm_s is not None and tangent_mm_s >= 20.0:
            return "SLIDE_CANDIDATE"
        return "TOUCH_CANDIDATE"
    if 8.0 < distance_mm <= 30.0 and radial_mm_s is not None:
        if radial_mm_s <= -5.0:
            return "APPROACH"
        if touching and radial_mm_s >= 5.0:
            return "RELEASE"
    return "UNKNOWN"


@dataclass
class TemporalContact:
    previous: dict[tuple[int, int], tuple[int, float, np.ndarray, bool]] = field(default_factory=dict)

    def update(
        self, key: tuple[int, int], frame_id: int, fps: float, point: np.ndarray | None,
        transform: np.ndarray | None, size_m: np.ndarray | None,
    ) -> dict[str, object]:
        if point is None or transform is None or size_m is None:
            self.previous.pop(key, None)
            return {"state": "UNKNOWN", "distance_mm": None, "radial_velocity_mm_s": None,
                    "tangential_velocity_mm_s": None, "dt_s": None, "valid": False}
        if not np.isfinite(point).all() or not np.isfinite(transform).all():
            self.previous.pop(key, None)
            return {"state": "UNKNOWN", "distance_mm": None, "radial_velocity_mm_s": None,
                    "tangential_velocity_mm_s": None, "dt_s": None, "valid": False}
        distance = float(oriented_box_sdf(point[None], transform, size_m)[0] * 1000.0)
        local = (point - transform[:3, 3]) @ transform[:3, :3]
        radial: float | None = None
        tangent: float | None = None
        dt: float | None = None
        prior = self.previous.get(key)
        if prior is not None and frame_id == prior[0] + 1:
            dt = (frame_id - prior[0]) / fps
            if dt > 0 and math.isfinite(dt):
                radial = (distance - prior[1]) / dt
                tangent = float(np.linalg.norm(local[:2] - prior[2]) * 1000.0 / dt)
        state = contact_state(distance, radial, tangent, prior[3] if prior is not None and dt is not None else False)
        self.previous[key] = (frame_id, distance, local[:2].copy(), state in {"TOUCH_CANDIDATE", "SLIDE_CANDIDATE"})
        return {"state": state, "distance_mm": distance, "radial_velocity_mm_s": radial,
                "tangential_velocity_mm_s": tangent, "dt_s": dt, "valid": state != "UNKNOWN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--object6d-result", type=Path, required=True)
    parser.add_argument("--adapter-result", type=Path, required=True)
    parser.add_argument("--old-contact-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable attempt exists: {output}")
    hawor_path, object_path = args.hawor_result.resolve(strict=True), args.object6d_result.resolve(strict=True)
    adapter_path, old_path = args.adapter_result.resolve(strict=True), args.old_contact_result.resolve(strict=True)
    hawor, obj, adapter, old = (read_json(p) for p in (hawor_path, object_path, adapter_path, old_path))
    if any(row.get("session_id") != args.session_id for row in (obj, adapter, old)):
        raise ValueError("full session identity mismatch")
    if hawor.get("session_id", hawor.get("session")) != args.session_id:
        raise ValueError("HaWoR session mismatch")
    if obj.get("unobserved_pose_policy") != "KEEP_INVALID" or obj.get("propagated_frames") != 0:
        raise ValueError("Object6D is not direct-observed only")
    if adapter.get("status") != "PASS_DEVELOPMENT_COORDINATE_ADAPTER":
        raise ValueError("selected-camera registration adapter not passed")
    if adapter.get("authorized_scopes") != ["ROBOT_CONTACT_DEVELOPMENT_COORDINATE_INPUT"]:
        raise ValueError("adapter scope mismatch")
    hawor_npz = verify(hawor["outputs"]["npz"])
    object_npz = verify(obj["artifacts"]["trajectory"])
    adapter_npz = verify(adapter["outputs"][0])
    if not any(ref["sha256"] == artifact(object_npz)["sha256"] for ref in adapter["inputs"]):
        raise ValueError("adapter does not bind source Object6D SHA")
    old_hypothesis_path = verify(old["contact_hypothesis"])
    old_rows = read_json(old_hypothesis_path)["hypotheses"]
    old_by_key = {(int(row["frame_id"]), row["hand_side"], row["finger_id"]): row for row in old_rows}
    with np.load(hawor_npz, allow_pickle=False) as h:
        joints = h["joints_3d_camera"].astype(np.float64)
        points_2d = h["joints_2d"].astype(np.float64)
        hand_valid = h["observed"].astype(bool)
        frames = h["original_frame_indices"].astype(np.int64)
        fps = float(h["fps"])
        sides = tuple(str(x) for x in h["anatomical_side_names"].tolist())
    with np.load(object_npz, allow_pickle=False) as o:
        object_frames = o["frame_indices"].astype(np.int64)
        object_valid = o["valid"].astype(bool) & o["observed"].astype(bool)
        old_transform = o["T_object_to_camera"].astype(np.float64)
        size = o["object_size_m"].astype(np.float64)
    with np.load(adapter_npz, allow_pickle=False) as a:
        selected = a["T_object_to_selected_camera"].astype(np.float64)
        rectified = a["T_object_to_rectified_camera"].astype(np.float64)
        registration = a["T_stereo_rectified_camera_to_selected_camera"].astype(np.float64)
        adapter_valid = a["valid"].astype(bool)
        adapter_frames = a["frame_indices"].astype(np.int64)
        source_domain = str(a["source_coordinate_domain"].item())
        target_domain = str(a["target_coordinate_domain"].item())
    if not (sides == SIDES and np.array_equal(frames, object_frames) and np.array_equal(frames, adapter_frames)):
        raise ValueError("side/frame closure failed")
    if fps <= 0 or not math.isfinite(fps) or np.any(np.diff(frames) <= 0):
        raise ValueError("source-frame timebase invalid")
    if source_domain != "STEREO_RECTIFIED_DEPTH_CAMERA" or target_domain != "SELECTED_LEFT_RGB_CAMERA":
        raise ValueError("camera-domain labels missing or wrong")
    if not np.array_equal(old_transform, rectified) or not np.array_equal(adapter_valid, o_valid := object_valid):
        raise ValueError("adapter does not preserve Object6D observations")
    if not np.allclose(selected[o_valid], registration @ rectified[o_valid], rtol=0, atol=1e-9):
        raise ValueError("registration transform closure failed")
    if joints.shape != (2, len(frames), 21, 3) or len(old_by_key) != 10 * len(frames):
        raise ValueError("joint/old comparison closure failed")
    raw_ref = obj["inputs"]["rgb"]
    rgb_path = verify(raw_ref)
    output.mkdir(parents=True)
    config = {"schema_version": "contact10-selected-camera-r23", "fps_source": "HaWoR source frame indices/fps",
              "max_temporal_gap_frames": 0, "object_id": "playing_card_0", "face_id": "UNKNOWN",
              "claim_status": "HYPOTHESIS_ONLY"}
    write_json(output / "CONFIG.json", config)
    tracker = TemporalContact()
    rows: list[dict[str, object]] = []
    state_counts: Counter[str] = Counter()
    delta: list[float] = []
    invalid_reentry = 0
    previous_direct = False
    for local, frame_id in enumerate(frames.tolist()):
        direct = bool(o_valid[local])
        if direct and not previous_direct and local > 0:
            invalid_reentry += 1
        previous_direct = direct
        for side_index, side in enumerate(SIDES):
            for finger_index, finger in enumerate(FINGERS):
                tip_index = TIP_INDICES[finger_index]
                key = (side_index, finger_index)
                available = direct and bool(hand_valid[side_index, local])
                new = tracker.update(key, frame_id, fps,
                                     joints[side_index, local, tip_index] if available else None,
                                     selected[local] if available else None, size if available else None)
                old_row = old_by_key[(frame_id, side, finger)]
                old_distance = old_row.get("surface_distance_mm")
                if old_distance is not None and new["distance_mm"] is not None:
                    delta.append(float(new["distance_mm"]) - float(old_distance))
                state_counts[str(new["state"])] += 1
                rows.append({"frame_id": frame_id, "side": side, "finger": finger,
                             "physical_object_instance_id": "playing_card_0" if direct else None,
                             "face_id": "UNKNOWN", "direct_object_observation": direct,
                             "old_rectified_mixed": {"state": old_row["state"], "distance_mm": old_distance,
                                                     "relative_velocity_mm_s": old_row.get("relative_velocity")},
                             "new_selected": new, "coordinate_domain": target_domain,
                             "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN"})
    rows_path = output / "OLD_NEW_CONTACT_ROWS.json"
    write_json(rows_path, rows)
    metrics = {"session_id": args.session_id, "frames": len(frames), "direct_object_frames": int(o_valid.sum()),
               "reentry_after_invalid_count": invalid_reentry, "state_counts": dict(state_counts),
               "old_vs_new_distance_delta_mm": {"n": len(delta), "p50_abs": float(np.median(np.abs(delta))) if delta else None,
                                                "p95_abs": float(np.percentile(np.abs(delta), 95)) if delta else None,
                                                "max_abs": float(np.max(np.abs(delta))) if delta else None},
               "joint_semantics": "MANO fingertip JOINT CENTER proxy, not pad surface",
               "object_semantics": "observed thin oriented box proxy, not verified card contact surface",
               "physical_instance_id": "playing_card_0", "face_id": "UNKNOWN",
               "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN"}
    write_json(output / "METRICS.json", metrics)
    video_path = output / f"{args.session_id}_CONTACT_R22_vs_R23_OFFLINE_VISUAL_FULLSESSION.mp4"
    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        raise RuntimeError("Raw RGB decode open failed")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError("video writer open failed")
    try:
        for local, frame_id in enumerate(frames.tolist()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, rgb = cap.read()
            if not ok:
                raise RuntimeError(f"Raw decode failed at {frame_id}")
            panels = [rgb.copy(), rgb.copy()]
            for side_index, side in enumerate(SIDES):
                for finger_index, finger in enumerate(FINGERS):
                    uv = points_2d[side_index, local, TIP_INDICES[finger_index]]
                    if not np.isfinite(uv).all():
                        continue
                    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
                    if not (0 <= x < width and 0 <= y < height):
                        continue
                    row = rows[local * 10 + side_index * 5 + finger_index]
                    for panel, item in zip(panels, (row["old_rectified_mixed"], row["new_selected"])):
                        status = str(item["state"])
                        color = (100, 100, 100) if status == "UNKNOWN" else ((0, 220, 220) if side == "left" else (60, 60, 255))
                        cv2.circle(panel, (x, y), 5, color, 2)
                        if finger in {"thumb", "index"}:
                            distance = item["distance_mm"]
                            label = f"{side[0]}{finger[0]}:{distance:+.1f}mm" if distance is not None else f"{side[0]}{finger[0]}:UNKNOWN"
                            cv2.putText(panel, label, (max(0, x - 30), max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
            for panel, title in zip(panels, ("R22 RECTIFIED MIXED (OLD)", "R23 SELECTED + GAP RESET (NEW)")):
                cv2.rectangle(panel, (0, 0), (width, 60), (0, 0, 0), -1)
                cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(panel, f"{args.session_id} frame={frame_id} direct={int(o_valid[local])} CONTACT HYPOTHESIS ONLY", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            writer.write(np.hstack(panels))
    finally:
        cap.release()
        writer.release()
    check = cv2.VideoCapture(str(video_path))
    decoded = 0
    while True:
        ok, _ = check.read()
        if not ok:
            break
        decoded += 1
    check.release()
    if decoded != len(frames):
        raise RuntimeError(f"review full decode mismatch: {decoded}!={len(frames)}")
    input_refs = [artifact(p) for p in (hawor_path, object_path, adapter_path, old_path, hawor_npz, object_npz, adapter_npz, old_hypothesis_path, rgb_path)]
    run_receipt = {"task_id": "RC1-ML-D-CONTACT-R23-POKER245", "session_id": args.session_id,
                   "generated_at": datetime.now(timezone.utc).isoformat(), "inputs": input_refs,
                   "producer": artifact(Path(__file__)), "geometry_code": artifact(ROOT / "src/chaoyang/pipeline/contact_geometry_v2.py"),
                   "config": artifact(output / "CONFIG.json"), "model_weights_sha": "UNKNOWN_VERIFICATION_REQUIRED",
                   "registration_adapter": artifact(adapter_npz), "authority_promoted": False}
    write_json(output / "RUN_RECEIPT.json", run_receipt)
    manifest = {"outputs": [artifact(p) for p in (rows_path, output / "METRICS.json", output / "CONFIG.json", video_path, output / "RUN_RECEIPT.json")],
                "video_decoded_frames": decoded, "video_expected_frames": len(frames)}
    write_json(output / "ARTIFACT_MANIFEST.json", manifest)
    result = {"task_id": "RC1-ML-D-CONTACT-R23-POKER245", "session_id": args.session_id,
              "terminal_status": "PASSED_DEVELOPMENT", "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
              "claim_status": "HYPOTHESIS_ONLY", "external_accuracy": "UNKNOWN", "authority_promoted": False,
              "metrics": artifact(output / "METRICS.json"), "video": artifact(video_path),
              "rows": artifact(rows_path), "run_receipt": artifact(output / "RUN_RECEIPT.json"),
              "claim_limit": "Selected-camera coordinate and temporal-state correction of MANO joint-center to observed card-box proximity only; not physical contact or causal training proof."}
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
