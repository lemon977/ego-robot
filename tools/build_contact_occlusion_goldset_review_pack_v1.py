from __future__ import annotations

"""Freeze the CONTACT_OCCLUSION_GOLDSET_V1 review frame/evidence pack.

This tool deliberately does **not** create ground truth.  It deterministically
selects uniform, hand/object-near, and visibility-transition frames for
Chips107 and Poker243, freezes their evidence SHA closure, writes UNKNOWN-only
pixel-label templates, and makes a review video.  Two independent humans must
fill the templates before a later scorer may publish a goldset result.
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WAVE = ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
DEFAULT_OUT = ROOT / "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/goldset_v1_review_pack"
TARGETS = ("get_potato_chips_0902_107", "play_cards_0903_243")
UNKNOWN = np.uint8(255)
NARROWBAND_RADIUS_PX = 7
NEAR_RADIUS_PX = 15


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular file required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def checked(reference: dict[str, Any]) -> Path:
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if ref(path) != reference:
        raise ValueError(f"artifact mismatch: {path}")
    return path


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_mask(reference: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    path = checked(reference)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise ValueError(f"mask decode/shape mismatch: {path}")
    return mask > 0


def choose_spaced(scores: list[float], count: int, spacing: int = 2) -> list[int]:
    chosen: list[int] = []
    for frame in sorted(range(len(scores)), key=lambda item: (-scores[item], item)):
        if scores[frame] <= 0:
            break
        if all(abs(frame - prior) > spacing for prior in chosen):
            chosen.append(frame)
        if len(chosen) == count:
            break
    return sorted(chosen)


def session_inputs(row: dict[str, Any]) -> dict[str, Any]:
    role_result_path = checked(row["upstream"]["role_mask"])
    object_result_path = checked(row["upstream"]["task_object_mask"])
    role_result, object_result = load(role_result_path), load(object_result_path)
    role_manifest_path = checked(role_result["artifacts"]["frame_manifest"])
    object_manifest_path = checked(object_result["artifacts"]["manifest"])
    role_manifest, object_manifest = load(role_manifest_path), load(object_manifest_path)
    selected_rgb = object_manifest["input"]["selected_rgb"]
    video_path = checked(selected_rgb)
    clean_result_ref = row.get("existing_clean")
    if clean_result_ref is None:
        clean_result_path = WAVE / "propainter_v1" / row["session_id"] / "RESULT.json"
        clean_result_ref = ref(clean_result_path)
    clean_result_path = checked(clean_result_ref)
    clean_result = load(clean_result_path)
    clean_video_ref = clean_result.get("artifacts", {}).get("clean_synthetic_master")
    if not isinstance(clean_video_ref, dict):
        # Older existing Clean results use a compatible artifact key.
        for key, value in clean_result.get("artifacts", {}).items():
            if isinstance(value, dict) and key in {"clean_master", "master_video", "clean_video"}:
                clean_video_ref = value
                break
    clean_video_path = checked(clean_video_ref) if isinstance(clean_video_ref, dict) else None
    frame_count = int(row["frame_count"])
    if len(role_manifest["frames"]) != frame_count or len(object_manifest["frames"]) != frame_count:
        raise ValueError(f"{row['session_id']}: manifest frame closure")
    return {
        "role_result": ref(role_result_path), "object_result": ref(object_result_path),
        "role_manifest": ref(role_manifest_path), "object_manifest": ref(object_manifest_path),
        "role_frames": role_manifest["frames"], "object_frames": object_manifest["frames"],
        "selected_rgb": selected_rgb, "video_path": video_path,
        "clean_result": ref(clean_result_path),
        "clean_video": ref(clean_video_path) if clean_video_path else None,
        "clean_video_path": clean_video_path,
    }


def analyze(row: dict[str, Any], inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    raw_cap = cv2.VideoCapture(str(inputs["video_path"]))
    clean_cap = cv2.VideoCapture(str(inputs["clean_video_path"])) if inputs["clean_video_path"] else None
    if not raw_cap.isOpened() or (clean_cap is not None and not clean_cap.isOpened()):
        raise ValueError(f"{row['session_id']}: video open failure")
    metrics, raws, cleans = [], [], []
    try:
        for frame_id, (role_frame, object_frame) in enumerate(zip(inputs["role_frames"], inputs["object_frames"], strict=True)):
            ok, raw = raw_cap.read()
            if not ok:
                raise ValueError(f"{row['session_id']}: raw decode failed at {frame_id}")
            if clean_cap is not None:
                clean_ok, clean = clean_cap.read()
                if not clean_ok:
                    raise ValueError(f"{row['session_id']}: Clean decode failed at {frame_id}")
            else:
                clean = raw.copy()
            shape = raw.shape[:2]
            human = np.zeros(shape, dtype=bool)
            for side in ("left_human", "right_human"):
                human |= read_mask(role_frame["role_masks"][side], shape)
            objects: dict[str, np.ndarray] = {}
            observed = []
            for instance_id, item in sorted(object_frame["physical_instances"].items()):
                mask = read_mask(item["mask"], shape)
                objects[instance_id] = mask
                observed.append(bool(item.get("observed") and item.get("valid") and mask.any()))
            object_union = np.logical_or.reduce(list(objects.values())) if objects else np.zeros(shape, bool)
            kernel_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR_RADIUS_PX + 1,) * 2)
            kernel_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NARROWBAND_RADIUS_PX + 1,) * 2)
            near = cv2.dilate(object_union.astype(np.uint8), kernel_near) > 0
            dilated = cv2.dilate(object_union.astype(np.uint8), kernel_band) > 0
            eroded = cv2.erode(object_union.astype(np.uint8), kernel_band) > 0
            band = dilated ^ eroded
            direct = human & object_union
            near_hand = human & near
            band_hand = human & band
            metrics.append({
                "source_frame": frame_id, "observed_instances": observed,
                "direct_overlap_px": int(direct.sum()), "near_hand_px": int(near_hand.sum()),
                "narrowband_hand_px": int(band_hand.sum()), "object_area_px": int(object_union.sum()),
                "human_area_px": int(human.sum()), "human_mask": human, "object_masks": objects,
                "narrowband": band,
            })
            raws.append(raw); cleans.append(clean)
        if raw_cap.read()[0] or (clean_cap is not None and clean_cap.read()[0]):
            raise ValueError(f"{row['session_id']}: extra video frames")
    finally:
        raw_cap.release()
        if clean_cap is not None:
            clean_cap.release()
    return metrics, raws, cleans


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_symlink():
        raise SystemExit(f"fresh output required: {out}")
    out.mkdir(parents=True)
    selection_path = WAVE / "EXACT78_WAVE0_SELECTION.json"
    selection = load(selection_path)
    rows = {row["session_id"]: row for row in selection["sessions"]}
    if set(TARGETS) - set(rows):
        raise ValueError("goldset sessions not in frozen Wave0")
    colors = [(255, 80, 40), (40, 220, 80), (60, 80, 255)]
    all_entries, input_manifest = [], {}
    video_frames = []
    for session in TARGETS:
        row = rows[session]
        inputs = session_inputs(row); input_manifest[session] = {k: v for k, v in inputs.items() if k not in {"role_frames", "object_frames", "video_path", "clean_video_path"}}
        metrics, raws, cleans = analyze(row, inputs)
        count = len(metrics)
        uniform = sorted(set(int(round(value)) for value in np.linspace(0, count - 1, 12)))
        transitions = []
        for frame in range(1, count):
            if metrics[frame]["observed_instances"] != metrics[frame - 1]["observed_instances"]:
                transitions.extend(index for index in (frame - 1, frame, frame + 1) if 0 <= index < count)
        difficult_score = [m["near_hand_px"] + 4 * m["direct_overlap_px"] + 2 * m["narrowband_hand_px"] for m in metrics]
        difficult = choose_spaced(difficult_score, 12)
        selected = sorted(set(uniform + transitions + difficult))
        reasons = {frame: [] for frame in selected}
        for frame in uniform: reasons.setdefault(frame, []).append("UNIFORM")
        for frame in transitions: reasons.setdefault(frame, []).append("VISIBILITY_TRANSITION")
        for frame in difficult: reasons.setdefault(frame, []).append("HAND_OBJECT_DIFFICULT")
        session_panel_dir = out / "review_panels" / session
        session_band_dir = out / "contact_narrowband" / session
        session_label_dir = out / "label_templates_unknown" / session
        session_panel_dir.mkdir(parents=True); session_band_dir.mkdir(parents=True); session_label_dir.mkdir(parents=True)
        for frame in selected:
            metric=metrics[frame];raw=raws[frame];clean=cleans[frame]
            overlay=raw.copy(); overlay[metric["human_mask"]]=(0.45*overlay[metric["human_mask"]]+0.55*np.array([255,180,40])).astype(np.uint8)
            for instance_index,(instance_id,mask) in enumerate(metric["object_masks"].items()):
                color=np.array(colors[instance_index % len(colors)])
                overlay[mask]=(0.35*overlay[mask]+0.65*color).astype(np.uint8)
            overlay[metric["narrowband"]]=np.array([255,255,255],dtype=np.uint8)
            label=np.zeros_like(raw);label[:]=30
            lines=[f"CONTACT_OCCLUSION_GOLDSET_V1 / {session}",f"frame={frame} task={row['task']}",
                   f"reasons={'+'.join(reasons[frame])}",f"direct_overlap={metric['direct_overlap_px']} px",
                   f"near_hand={metric['near_hand_px']} px  narrowband_hand={metric['narrowband_hand_px']} px",
                   "LABEL STATUS: UNKNOWN / TWO HUMAN REVIEWS REQUIRED",
                   "No Robot/contact truth is inferred by this pack."]
            for line_index,line in enumerate(lines):cv2.putText(label,line,(30,70+55*line_index),cv2.FONT_HERSHEY_SIMPLEX,0.9,(230,230,230),2,cv2.LINE_AA)
            top=np.concatenate([raw,clean],axis=1);bottom=np.concatenate([overlay,label],axis=1);panel=np.concatenate([top,bottom],axis=0)
            panel_path=session_panel_dir/f"{frame:05d}.png";band_path=session_band_dir/f"{frame:05d}.png";template_path=session_label_dir/f"{frame:05d}.png"
            if not cv2.imwrite(str(panel_path),panel) or not cv2.imwrite(str(band_path),metric["narrowband"].astype(np.uint8)*255) or not cv2.imwrite(str(template_path),np.full(raw.shape[:2],UNKNOWN,np.uint8)):
                raise IOError(f"write failure {session} {frame}")
            video_frames.append(panel)
            all_entries.append({
                "task":row["task"],"session":session,"source_frame":frame,"selection_reasons":sorted(set(reasons[frame])),
                "physical_instance_ids":sorted(metric["object_masks"]),"observed_instances":metric["observed_instances"],
                "metrics":{k:metric[k] for k in ("direct_overlap_px","near_hand_px","narrowband_hand_px","object_area_px","human_area_px")},
                "panel":ref(panel_path),"contact_narrowband":ref(band_path),"unknown_label_template":ref(template_path),
                "reviewer_1":{"status":"PENDING","label_path":None,"reviewer_id":None},
                "reviewer_2":{"status":"PENDING","label_path":None,"reviewer_id":None},
                "consensus":"UNKNOWN",
            })
    height,width=video_frames[0].shape[:2]
    video_path=out/"CONTACT_OCCLUSION_GOLDSET_V1_REVIEW_SEQUENCE.mp4"
    writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),2.0,(width,height))
    if not writer.isOpened():raise IOError("review video writer")
    for frame in video_frames:writer.write(frame)
    writer.release()
    # Full decode is a hard review-media gate.
    cap=cv2.VideoCapture(str(video_path));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=len(video_frames):raise ValueError("review video frame mismatch")
    selection_result={
        "schema_version":"contact-occlusion-goldset-selection-v1","created_at":now(),
        "status":"FROZEN_REVIEW_SELECTION_PENDING_LABELS","goldset_name":"CONTACT_OCCLUSION_GOLDSET_V1",
        "selection_policy":{"uniform_frames_per_session":12,"visibility_transition_radius_frames":1,
                            "difficult_frames_per_session":12,"difficult_min_spacing_frames":3,
                            "contact_narrowband_radius_px":NARROWBAND_RADIUS_PX,"hand_object_near_radius_px":NEAR_RADIUS_PX},
        "label_encoding":{"ROBOT_FRONT":1,"OBJECT_FRONT":2,"NO_CONTEST_KNOWN":3,"UNKNOWN":255},
        "double_review_required":True,"disagreement_policy":"UNKNOWN","entries":all_entries,
        "inputs":{"wave0_selection":ref(selection_path),"sessions":input_manifest},
        "claim_limit":"Frozen evidence/review selection only; no pixel or contact ground truth and no goldset pass claim.",
    }
    write_json(out/"GOLDSET_SELECTION.json",selection_result)
    with (out/"REVIEW_INDEX.csv").open("x",encoding="utf-8",newline="") as handle:
        writer_csv=csv.DictWriter(handle,fieldnames=["task","session","source_frame","selection_reasons","direct_overlap_px","near_hand_px","narrowband_hand_px","panel_path"]);writer_csv.writeheader()
        for item in all_entries:
            writer_csv.writerow({"task":item["task"],"session":item["session"],"source_frame":item["source_frame"],
             "selection_reasons":"+".join(item["selection_reasons"]),**{k:item["metrics"][k] for k in ("direct_overlap_px","near_hand_px","narrowband_hand_px")},"panel_path":item["panel"]["path"]})
    write_json(out/"RESULT.json",{
        "schema_version":"contact-occlusion-goldset-review-pack-result-v1","created_at":now(),
        "status":"BLOCKED_EXTERNAL_PENDING_TWO_INDEPENDENT_HUMAN_LABELS_AND_ROBOT_CANDIDATE",
        "goldset_selection":ref(out/"GOLDSET_SELECTION.json"),"review_index":ref(out/"REVIEW_INDEX.csv"),
        "review_video":ref(video_path),"selected_frame_count":len(all_entries),"decoded_review_frame_count":decoded,
        "goldset_passed":False,"authority_promotable":False,
        "next_required":"Two independent pixel-label sets, consensus generation, then fixed-threshold scoring against a Robot compositor candidate.",
        "claim_limit":"Review pack only; UNKNOWN templates must not be interpreted as contact/occlusion truth.",
    })
    print(json.dumps({"status":"PASS_REVIEW_PACK","frames":len(all_entries),"result":ref(out/"RESULT.json")},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
