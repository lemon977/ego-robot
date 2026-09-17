#!/usr/bin/env python3
"""Causal, two-frame Poker245 card-order/visible-face development check.

This deliberately does not update Object Mask, Clean, or training authority.
It scores an independently implemented colour-component cue against two
already-frozen V4 proposals, and previews their pixels only when all cues pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SESSION = "play_cards_0903_245"
TARGETS = (35, 39)
SOURCE_ANCHOR = 12


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(pin: dict, label: str) -> Path:
    path = Path(pin["path"]).resolve(strict=True)
    actual = ref(path)
    if actual["bytes"] != pin["bytes"] or actual["sha256"] != pin["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA changed")
    return path


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_prefix(video: Path, final_frame: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(final_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Raw decode stopped at frame {len(frames)}")
        frames.append(frame)
    capture.release()
    return frames


def colour_components(frame: np.ndarray, hue: int, config: dict) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= hue - config["hue_half_width"])
            & (hsv[:, :, 0] <= hue + config["hue_half_width"])
            & (hsv[:, :, 1] >= config["sat_min"])
            & (hsv[:, :, 2] >= config["value_min"])
            & (hsv[:, :, 2] <= config["value_max"]))
    mask[: config["table_roi_y_min"]] = False
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [
        {"centroid_xy": [round(float(centers[i, 0]), 2), round(float(centers[i, 1]), 2)],
         "area_px": int(stats[i, cv2.CC_STAT_AREA]),
         "bbox_xywh": [int(v) for v in stats[i, :4]]}
        for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= config["min_component_px"]
    ]


def track_order(frames: list[np.ndarray], source_mask: np.ndarray, config: dict, target: int) -> tuple[list[dict], dict]:
    if source_mask.shape != frames[SOURCE_ANCHOR].shape[:2] or not source_mask.any():
        raise RuntimeError("source instance mask missing or shape mismatch")
    hsv = cv2.cvtColor(frames[SOURCE_ANCHOR], cv2.COLOR_BGR2HSV)
    seed_hues = hsv[:, :, 0][source_mask & (hsv[:, :, 1] >= 80)]
    if len(seed_hues) < 100:
        raise RuntimeError("source back appearance has too few chromatic pixels")
    # The modal hue is derived only from the frozen observed source card.
    hue = int(np.bincount(seed_hues, minlength=180).argmax())
    ledger = []
    previous = None
    for frame_id in range(SOURCE_ANCHOR, target + 1):
        candidates = colour_components(frames[frame_id], hue, config)
        # The three original backs have non-overlapping left/middle/right bands.
        bands = ((600, 745), (745, 865), (865, 1030))
        chosen = []
        for x_min, x_max in bands:
            possible = [item for item in candidates if x_min < item["centroid_xy"][0] < x_max]
            chosen.append(max(possible, key=lambda item: item["area_px"]) if possible else None)
        areas = [item["area_px"] if item else 0 for item in chosen]
        x_values = [item["centroid_xy"][0] if item else None for item in chosen]
        ordering = all(item is not None for item in chosen) and all(
            x_values[index + 1] - x_values[index] >= config["min_order_gap_px"]
            for index in (0, 1)
        )
        stable = ordering and (previous is None or all(
            abs(x_values[index] - previous[index]) <= config["max_center_step_px"]
            for index in range(3)
        ))
        area_pass = (areas[0] >= config["min_unoccluded_area_px"]
                     and areas[1] >= config["min_unoccluded_area_px"]
                     and areas[2] >= config["min_rightmost_visible_px"])
        row = {"frame": frame_id, "left": chosen[0], "middle": chosen[1], "rightmost": chosen[2],
               "ordered": bool(ordering), "center_stable": bool(stable),
               "three_visible_back_components": bool(area_pass),
               "cue_pass": bool(stable and area_pass)}
        ledger.append(row)
        previous = x_values if ordering else None
    return ledger, {"source_hue_mode": hue, "prefix_rows": len(ledger),
                    "all_prefix_cue_pass": all(item["cue_pass"] for item in ledger),
                    "target_rightmost_visible_px": ledger[-1]["rightmost"]["area_px"] if ledger[-1]["rightmost"] else 0}


def draw_panel(frame: np.ndarray, title: str, track: dict | None = None) -> np.ndarray:
    panel = frame.copy()
    if track is not None:
        colours = ((255, 120, 0), (0, 255, 255), (0, 80, 255))
        for name, colour in zip(("left", "middle", "rightmost"), colours):
            item = track[name]
            if item:
                x, y, w, h = item["bbox_xywh"]
                cv2.rectangle(panel, (x, y), (x + w, y + h), colour, 3)
                cv2.putText(panel, f"{name}:{item['area_px']}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    panel = cv2.resize(panel, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (640, 46), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.packet.resolve(strict=True)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("session_id") != SESSION or packet.get("task_id") != "poker245_donor_observability_two_events":
        raise RuntimeError("packet identity mismatch")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(packet["write_root"]).resolve()):
        raise RuntimeError("new attempt under declared write_root required")
    paths = {key: verify(pin, key) for key, pin in packet["inputs"].items()}
    for key, pin in packet["code"].items():
        verify(pin, f"code:{key}")
    signature_payload = {
        "schema": packet["schema_version"],
        "inputs": {key: pin["sha256"] for key, pin in packet["inputs"].items()},
        "code": {key: pin["sha256"] for key, pin in packet["code"].items()},
        "config": packet["config"], "model_weights": "ABSENT_CPU_COLOUR_CUE",
        "calibration": "ABSENT_2D_OBSERVABILITY_ONLY",
    }
    run_signature = hashlib.sha256(json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    strict = json.loads(paths["strict_decisions"].read_text())["rows"]
    old = json.loads(paths["v4_decisions"].read_text())["rows"]
    objects = json.loads(paths["object_manifest"].read_text())
    strict_rows = {row["target_frame"]: row for row in strict}
    v4_rows = {row["target_frame"]: row for row in old}
    if tuple(packet["targets"]) != TARGETS or any(t not in strict_rows or t not in v4_rows for t in TARGETS):
        raise RuntimeError("two frozen target rows are missing")
    frames = read_prefix(paths["raw_video"], max(TARGETS))
    source_mask_path = Path(objects["frames"][SOURCE_ANCHOR]["physical_instances"]["0"]["mask"]["path"])
    if ref(source_mask_path)["sha256"] != objects["frames"][SOURCE_ANCHOR]["physical_instances"]["0"]["mask"]["sha256"]:
        raise RuntimeError("source physical-mask SHA mismatch")
    source_mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE) > 0
    output.mkdir(parents=True)
    write_json(output / "RUN_START.json", {"task_id": packet["task_id"], "session_id": SESSION,
               "packet": ref(packet_path), "started_at": datetime.now().astimezone().isoformat(),
               "pid": os.getpid(), "process_startticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
               "executor_epoch": packet["executor_epoch"], "fencing_token": packet["fencing_token"],
               "run_signature": run_signature,
               "input_mode": "OFFLINE_VISUAL", "gpu_required": False,
               "control_ground_truth": False, "authority_promoted": False})
    config = packet["config"]
    rows = []
    all_video_frames = []
    for target in TARGETS:
        source = int(strict_rows[target]["source_frame"])
        if source >= target or source != int(v4_rows[target]["source_frame"]):
            raise RuntimeError(f"target {target}: source identity or causal time mismatch")
        track, cue = track_order(frames, source_mask, config, target)
        formal = objects["frames"][target]["physical_instances"]["0"]
        proposal = strict_rows[target]
        pixel_map = Path(proposal["pixel_map"]["path"])
        if ref(pixel_map)["sha256"] != proposal["pixel_map"]["sha256"]:
            raise RuntimeError(f"target {target}: proposal map SHA mismatch")
        with np.load(pixel_map, allow_pickle=False) as mapped:
            proposed = np.asarray(mapped["proposal_source"] > 0)
            donor_frames = np.asarray(mapped["donor_frame_id"])
            donor_xy = np.asarray(mapped["donor_xy"])
            write_mask = np.asarray(mapped["m_write"] > 0)
        if proposed.shape != frames[target].shape[:2] or int(proposed.sum()) != proposal["proposal_pixels"]:
            raise RuntimeError(f"target {target}: proposal pixel count/shape mismatch")
        if np.any(proposed & ~write_mask) or np.any(donor_frames[proposed] != source):
            raise RuntimeError(f"target {target}: source map leaves write domain or source frame")
        ys, xs = np.nonzero(proposed)
        coords = donor_xy[ys, xs]
        if np.any(coords < 0) or np.any(coords[:, 0] >= frames[source].shape[1]) or np.any(coords[:, 1] >= frames[source].shape[0]):
            raise RuntimeError(f"target {target}: donor coordinates out of bounds")
        development_preview_allowed = bool(
            cue["all_prefix_cue_pass"] and v4_rows[target]["runtime_pass"]
            and proposal["proposal_pixels"] > 0
            and objects["frames"][source]["physical_instances"]["0"]["valid"]
        )
        preview = frames[target].copy()
        if development_preview_allowed:
            preview[ys, xs] = frames[source][coords[:, 1], coords[:, 0]]
        # Keep the formal strict result unchanged: target Object Mask remains invalid.
        formal_write_allowed = bool(formal["valid"] and proposal["independent_physical_face_proof"])
        if formal_write_allowed:
            raise RuntimeError("unexpected formal object/face proof: investigate before writing")
        event_row = {
            "target_frame": target, "donor_frame": source,
            "source_physical_instance_observed": bool(objects["frames"][source]["physical_instances"]["0"]["valid"]),
            "source_face_appearance": proposal["source_face_appearance"],
            "target_formal_object_valid": bool(formal["valid"]),
            "target_formal_face_id": proposal["target_face_id"],
            "target_visible_back_component_px": cue["target_rightmost_visible_px"],
            "three_card_order_prefix_pass": cue["all_prefix_cue_pass"],
            "v4_homography_and_flow_runtime_pass": bool(v4_rows[target]["runtime_pass"]),
            "source_map_closed_pixels": int(proposed.sum()),
            "development_preview_allowed": development_preview_allowed,
            "formal_strict_write_allowed": False,
            "formal_rejection_reason": "TARGET_OBJECT_MASK_INVALID_AND_FACE_ID_UNKNOWN",
            "claim_limit": "Card-order and visible back are causal development cues, not independent physical ID ground truth or hidden-face proof.",
        }
        rows.append(event_row)
        write_json(output / f"TRACK_PREFIX_{target:06d}.json", {"session_id": SESSION,
                   "source_anchor_frame": SOURCE_ANCHOR, "target_frame": target,
                   "cue": cue, "rows": track})
        if development_preview_allowed:
            # Name this PROPOSAL, never a Clean source map or final training input.
            np.savez_compressed(output / f"DEVELOPMENT_PROPOSAL_MAP_{target:06d}.npz",
                                proposal_mask=proposed.astype(np.uint8),
                                donor_frame_id=donor_frames, donor_xy=donor_xy,
                                m_write=write_mask.astype(np.uint8))
        for _ in range(18):
            panels = [
                draw_panel(frames[source], f"PAST RAW {source} / SAME BACK CUE"),
                draw_panel(frames[target], f"TARGET RAW {target} / ORDER CUE", track[-1]),
                draw_panel(preview, f"DEV PROPOSAL {int(proposed.sum()) if development_preview_allowed else 0}px / FORMAL 0"),
            ]
            all_video_frames.append(np.concatenate(panels, axis=1))
    video_path = output / "Poker245_同牌同面可观测性_两事件开发对照.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1920, 480))
    if not writer.isOpened():
        raise RuntimeError("video encoder unavailable")
    for frame in all_video_frames:
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(video_path))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != len(all_video_frames):
        raise RuntimeError("diagnostic video full decode count mismatch")
    write_json(output / "EVENT_DECISIONS.json", {"schema_version": "poker245-observability-two-events-v1",
               "session_id": SESSION, "rows": rows,
               "decision": "DEVELOPMENT_CUE_SUPPORT_ONLY_FORMAL_STRICT_ZERO_UNCHANGED"})
    result = {"schema_version": "poker245-observability-two-events-v1", "session_id": SESSION,
              "status": "PASSED_DEVELOPMENT_OBSERVABILITY" if all(r["development_preview_allowed"] for r in rows) else "FAILED_QUALITY_C",
              "created_at": datetime.now().astimezone().isoformat(),
              "run_signature": run_signature,
              "inputs": {key: ref(path) for key, path in paths.items()},
              "packet": ref(packet_path), "event_decisions": ref(output / "EVENT_DECISIONS.json"),
              "video": ref(video_path) | {"frames": decoded, "fps": 10},
              "proposal_maps": [ref(output / f"DEVELOPMENT_PROPOSAL_MAP_{t:06d}.npz") for t in TARGETS if (output / f"DEVELOPMENT_PROPOSAL_MAP_{t:06d}.npz").is_file()],
              "formal_written_pixels": 0, "input_mode": "OFFLINE_VISUAL",
              "training_eligible": False, "authority_promoted": False,
              "claim_limit": "Two-frame visible-back/order development evidence only. Strict Object Mask face/identity remains UNKNOWN; no Clean improvement, hidden truth, causal training or authority.",
              "next_action": "Freeze card-order/face cue on independent B regressions before proposing any formal same-card surface write gate."}
    write_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": str(output / "RESULT.json"),
                      "video": str(video_path), "formal_written_pixels": 0}, ensure_ascii=False))
    return 0 if result["status"] == "PASSED_DEVELOPMENT_OBSERVABILITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
