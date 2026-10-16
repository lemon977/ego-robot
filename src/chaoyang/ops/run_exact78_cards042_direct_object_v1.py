"""Fresh, frame-independent SAM3.1 identity mask for the Cards042 A-diamond."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.exact78_task_runtime_v1 import authorize, checked, verify_refs
from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease


ONSET = 96
KEYFRAMES = {
    96: ((1014.0, 627.0), (984.0, 634.0)),
    110: ((980.0, 639.0), (1065.0, 580.0)),
    125: ((555.0, 564.0), (640.0, 485.0)),
    145: ((626.0, 617.0), (553.0, 649.0)),
    165: ((631.0, 628.0), (562.0, 658.0)),
    170: ((631.0, 628.0), (562.0, 658.0)),
}


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def hint_points(frame: int) -> list[tuple[float, float]]:
    keys = sorted(KEYFRAMES)
    if frame <= keys[0]:
        return list(KEYFRAMES[keys[0]])
    if frame >= keys[-1]:
        return list(KEYFRAMES[keys[-1]])
    high = next(value for value in keys if value >= frame)
    low = max(value for value in keys if value <= frame)
    if low == high:
        return list(KEYFRAMES[low])
    alpha = (frame - low) / (high - low)
    return [((1-alpha)*a[0] + alpha*b[0], (1-alpha)*a[1] + alpha*b[1])
            for a, b in zip(KEYFRAMES[low], KEYFRAMES[high])]


def rgb_components(image: np.ndarray, hints: list[tuple[float, float]]) -> list[dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = ((hsv[:, :, 1] <= 75) & (hsv[:, :, 2] >= 135)).astype(np.uint8)
    binary[:390] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    rows = []
    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        if area < 300:
            continue
        mask = labels == label
        hits = sum(bool(mask[min(max(int(round(py)), 0), mask.shape[0]-1),
                             min(max(int(round(px)), 0), mask.shape[1]-1)]) for px, py in hints)
        center = (float(centers[label, 0]), float(centers[label, 1]))
        distance = float(np.linalg.norm(np.asarray(center) - np.mean(np.asarray(hints), axis=0)))
        rows.append({"mask": mask, "bbox": (x, y, width, height), "area": area,
                     "centroid": center, "positive_hits": hits, "hint_distance": distance})
    return rows


def select_rgb_component(rows: list[dict]) -> dict | None:
    eligible = [row for row in rows if row["positive_hits"] > 0]
    if eligible:
        return min(eligible, key=lambda row: (-row["positive_hits"], row["hint_distance"], -row["area"]))
    eligible = [row for row in rows if row["hint_distance"] <= 120.0]
    return min(eligible, key=lambda row: (row["hint_distance"], -row["area"])) if eligible else None


def component_prompt(component: dict, hints: list[tuple[float, float]], shape: tuple[int, int]):
    height, width = shape
    positive = []
    for px, py in hints:
        x, y = min(max(int(round(px)), 0), width-1), min(max(int(round(py)), 0), height-1)
        if component["mask"][y, x]:
            positive.append((float(px), float(py)))
    if not positive:
        positive = [component["centroid"]]
    x, y, w, h = component["bbox"]
    margin = 55
    candidates = [(x-margin,y-margin),(x+w+margin,y-margin),(x-margin,y+h+margin),(x+w+margin,y+h+margin),
                  (x+w/2,y-margin),(x+w/2,y+h+margin),(x-margin,y+h/2),(x+w+margin,y+h/2)]
    negative = []
    for px, py in candidates:
        px, py = float(min(max(px,0),width-1)), float(min(max(py,0),height-1))
        if min(np.hypot(px-qx, py-qy) for qx, qy in positive) >= 42.0:
            negative.append((px, py))
    return positive, negative


def normalize(outputs, height: int, width: int, object_id: int) -> np.ndarray:
    import torch
    def numpy(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    masks = numpy(outputs.get("out_binary_masks", np.zeros((0,height,width),bool)))
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = np.squeeze(masks, axis=1)
    if masks.ndim == 2:
        masks = masks[None]
    ids = numpy(outputs.get("out_obj_ids", [])).reshape(-1)
    scores = numpy(outputs.get("out_probs", np.ones(len(ids)))).reshape(-1)
    if masks.shape[1:] != (height,width) or len(masks) != len(ids) or len(scores) != len(ids):
        raise RuntimeError("SAM_OUTPUT_CONTRACT")
    matches = np.flatnonzero((ids == object_id) & (scores >= .5))
    return masks[int(matches[0])].astype(bool) if len(matches) == 1 else np.zeros((height,width),bool)


def associated_component(mask: np.ndarray, positive: list[tuple[float,float]]) -> tuple[np.ndarray, int, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    choices = []
    for label in range(1,count):
        hits = sum(labels[min(max(int(round(py)),0),mask.shape[0]-1),
                          min(max(int(round(px)),0),mask.shape[1]-1)] == label for px,py in positive)
        choices.append((hits, int(stats[label,cv2.CC_STAT_AREA]), label))
    if not choices:
        return np.zeros_like(mask), 0, 0
    hits, area, label = max(choices)
    if hits == 0:
        return np.zeros_like(mask), int(mask.sum()), 0
    selected = labels == label
    return selected, int(mask.sum()-selected.sum()), int(hits)


def save(path: Path, value: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not cv2.imwrite(str(path), value):
        raise RuntimeError("NO_CLOBBER_IMAGE:" + str(path))
    return artifact_ref(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    supplied = load_json(args.packet)
    packet, root = authorize(supplied["task_id"])
    if packet != supplied:
        raise RuntimeError("PACKET_MISMATCH")
    verify_refs(packet["inputs"])
    verify_refs(packet["runtime_code"])
    source = checked(packet["inputs"]["raw_video"])
    role_manifest = load_json(checked(packet["inputs"]["role_manifest"]))
    role_by_frame = {row["source_frame"]: row for row in role_manifest["frames"]}
    if set(role_by_frame) != set(range(171)):
        raise RuntimeError("ROLE_FRAME_SET")
    output = root / "object_identity"
    output.mkdir()
    review = cv2.VideoWriter(str(output/"review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280,960))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened() or not review.isOpened():
        raise RuntimeError("VIDEO_OPEN")
    rows = []
    gates_by_frame = []
    started = time.monotonic()
    import torch
    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter
    lease = FileGpuLease(
        lease_path=REPO_ROOT / "_run/current/GPU_LEASE.json",
        lock_path=REPO_ROOT / "_run/current/GPU_LEASE.lock",
        task_id=packet["task_id"], attempt_id="attempt_0001", gpu_id=0,
        executor_epoch=packet["writer"]["executor_epoch"],
        fencing_token=packet["writer"]["fencing_token"],
    )
    lease.__enter__()
    try:
        adapter, build_evidence = build_pinned_adapter(
            official_code_root=REPO_ROOT / "vendor/SAM3",
            checkpoint_path=checked(packet["inputs"]["sam_checkpoint"]),
        )
    except BaseException:
        lease.__exit__(*sys.exc_info())
        raise
    try:
        for frame in range(171):
            ok, image = capture.read()
            if not ok or image.shape[:2] != (960,1280):
                raise RuntimeError(f"FRAME_DECODE:{frame}")
            mask = np.zeros(image.shape[:2],bool)
            metadata = {"direct_sam31_invoked": False, "single_frame_state": False,
                        "positive_points": [], "negative_points": [], "discarded_pixels": 0}
            frame_gates = {}
            if frame >= ONSET:
                hints = hint_points(frame)
                component = select_rgb_component(rgb_components(image,hints))
                if component is None:
                    frame_gates = {"rgb_component": False}
                else:
                    positive, negative = component_prompt(component,hints,image.shape[:2])
                    frame_dir = output/"sam_inputs"/f"{frame:05d}"
                    save(frame_dir/"000000.png",image)
                    state = adapter.model.init_state(resource_path=str(frame_dir),offload_video_to_cpu=True,async_loading_frames=False)
                    try:
                        pos = torch.tensor([[x/1280,y/960] for x,y in positive],dtype=torch.float32)
                        adapter.model.add_prompt(inference_state=state,frame_idx=0,points=pos,
                            point_labels=torch.ones(len(positive),dtype=torch.int32),obj_id=301,
                            rel_coordinates=True,clear_old_points=True,output_prob_thresh=.5)
                        points = positive+negative
                        labels = [1]*len(positive)+[0]*len(negative)
                        _, outputs = adapter.model.add_prompt(inference_state=state,frame_idx=0,
                            points=torch.tensor([[x/1280,y/960] for x,y in points],dtype=torch.float32),
                            point_labels=torch.tensor(labels,dtype=torch.int32),obj_id=301,
                            rel_coordinates=True,clear_old_points=True,output_prob_thresh=.5)
                        raw = normalize(outputs,960,1280,301)
                    finally:
                        state.clear()
                    mask, discarded, positive_hits = associated_component(raw,positive)
                    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                    low_sat_bright = float(np.mean((hsv[:,:,1][mask] <= 75) & (hsv[:,:,2][mask] >= 135))) if mask.any() else 0.0
                    role_union = np.zeros_like(mask)
                    for ref in role_by_frame[frame]["role_masks"].values():
                        role_union |= cv2.imread(str(checked(ref)),cv2.IMREAD_GRAYSCALE)>0
                    overlap = float(np.logical_and(mask,role_union).sum()/max(int(mask.sum()),1))
                    ys,xs = np.nonzero(mask)
                    center = np.asarray([xs.mean(),ys.mean()]) if len(xs) else np.asarray([np.inf,np.inf])
                    hint_distance = float(np.linalg.norm(center-np.mean(np.asarray(hints),axis=0)))
                    area = int(mask.sum())
                    negative_clear = all(not mask[min(max(int(round(y)),0),959),min(max(int(round(x)),0),1279)] for x,y in negative)
                    frame_gates = {"area": bool(1200<=area<=50000),
                        "positive_associated_component": bool(positive_hits>=1),
                        "low_saturation_bright_fraction": bool(low_sat_bright>=.62),
                        "role_overlap": bool(overlap<=.10), "hint_distance": bool(hint_distance<=150.0),
                        "negative_exclusion": bool(negative_clear)}
                    metadata.update(direct_sam31_invoked=True,single_frame_state=True,positive_points=positive,
                        negative_points=negative,discarded_pixels=discarded,area_pixels=area,
                        low_saturation_bright_fraction=low_sat_bright,role_overlap_fraction=overlap,
                        hint_distance_pixels=hint_distance,positive_hits=positive_hits)
            mask_ref = save(output/"physical_object_0_masks"/f"{frame:05d}.png",mask.astype(np.uint8)*255)
            overlay=image.copy(); overlay[mask]=(0.35*overlay[mask]+0.65*np.asarray([0,255,255])).astype(np.uint8)
            cv2.putText(overlay,f"frame {frame:03d} {'OBSERVED' if mask.any() else 'UNKNOWN'}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if mask.any() else (0,0,255),2)
            review.write(overlay)
            rows.append({"source_frame":frame,"selected_rgb_decoded_sha256":array_sha(image),
                         "observed":bool(mask.any()),"mask":mask_ref,"metadata":metadata,"gates":frame_gates})
            if frame >= ONSET:
                gates_by_frame.append(all(frame_gates.values()))
            if frame%10==0:
                print(json.dumps({"frame":frame,"observed":bool(mask.any()),"seconds":round(time.monotonic()-started,1)}),flush=True)
    finally:
        adapter.predictor.shutdown(); capture.release(); review.release()
        lease.__exit__(*sys.exc_info())
    canary = cv2.imread(str(checked(packet["inputs"]["accepted_canary_125"])),cv2.IMREAD_GRAYSCALE)>0
    produced = cv2.imread(str(Path(rows[125]["mask"]["path"])),cv2.IMREAD_GRAYSCALE)>0
    iou = float(np.logical_and(canary,produced).sum()/max(int(np.logical_or(canary,produced).sum()),1))
    hard = {"frame_count_exact":len(rows)==171,"pre_onset_empty":not any(r["observed"] for r in rows[:ONSET]),
            "all_visible_direct_sam31":all(r["metadata"]["direct_sam31_invoked"] for r in rows[ONSET:]),
            "all_visible_single_frame_state":all(r["metadata"]["single_frame_state"] for r in rows[ONSET:]),
            "all_visible_observed":all(r["observed"] for r in rows[ONSET:]),
            "all_frame_instance_gates":all(gates_by_frame),"accepted_canary_125_iou":iou>=.50,
            "video_propagation_absent":True}
    manifest = {"schema_version":"exact78-cards042-direct-object-manifest-v1","session":"play_cards_0902_042",
                "identity":"single A-diamond physical instance 0","identity_onset":ONSET,"frames":rows}
    atomic_json(output/"FRAME_MANIFEST.json",manifest)
    result = {"schema_version":"exact78-cards042-direct-object-result-v1","status":"PASS_GRADE_B_CANDIDATE" if all(hard.values()) else "FAILED_QUALITY_C",
              "grade":"B" if all(hard.values()) else "C","downstream_authorized":all(hard.values()),
              "hard_gates":hard,"metrics":{"observed_count":sum(r["observed"] for r in rows),"canary_125_iou":iou},
              "artifacts":{"frame_manifest":artifact_ref(output/"FRAME_MANIFEST.json"),"review_video":artifact_ref(output/"review.mp4")},
              "build_evidence":build_evidence,"claim_limit":"Single Cards042 object identity Mask only; no Clean, Object6D, contact, baseline or deletion authority."}
    atomic_json(output/"RESULT.json",result)
    atomic_json(root/"RESULT.json",{"status":result["status"],"object_result":artifact_ref(output/"RESULT.json"),"deletion_authorized":False,"baseline_adopted":False})
    return 0 if all(hard.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
