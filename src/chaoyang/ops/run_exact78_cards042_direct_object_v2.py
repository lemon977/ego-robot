"""Fresh Cards042 object mask: points first, bounded role-aware negative refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops.run_exact78_cards042_direct_object_v1 import (
    ONSET, associated_component, component_prompt, hint_points, normalize,
    rgb_components, save, select_rgb_component,
)
from chaoyang.pipeline.exact78_task_runtime_v1 import authorize, checked, verify_refs
from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def quality(mask: np.ndarray, image: np.ndarray, role_union: np.ndarray,
            hints: list[tuple[float, float]], negative: list[tuple[float, float]]) -> tuple[dict, dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    area = int(mask.sum())
    object_fraction = float(np.mean((hsv[:, :, 1][mask] <= 90) & (hsv[:, :, 2][mask] >= 105))) if area else 0.0
    overlap = float(np.logical_and(mask, role_union).sum()/max(area, 1))
    ys, xs = np.nonzero(mask)
    center = np.asarray([xs.mean(), ys.mean()]) if len(xs) else np.asarray([np.inf, np.inf])
    distance = float(np.linalg.norm(center-np.mean(np.asarray(hints), axis=0)))
    negative_clear = all(not mask[min(max(int(round(y)), 0), 959), min(max(int(round(x)), 0), 1279)]
                         for x, y in negative)
    gates = {
        "area": bool(1200 <= area <= 50000),
        "object_appearance": bool(object_fraction >= .40),
        "role_overlap": bool(overlap <= .10),
        "hint_distance": bool(distance <= 150.0),
        "negative_exclusion": bool(negative_clear),
    }
    return gates, {"area_pixels": area, "object_appearance_fraction": object_fraction,
                   "role_overlap_fraction": overlap, "hint_distance_pixels": distance}


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
    output = root/"object_identity"
    output.mkdir()
    review = cv2.VideoWriter(str(output/"review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 960))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened() or not review.isOpened():
        raise RuntimeError("VIDEO_OPEN")

    import torch
    sys.path.insert(0, str(REPO_ROOT/"vendor/SAM3"))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter
    lease = FileGpuLease(
        lease_path=REPO_ROOT/"_run/current/GPU_LEASE.json", lock_path=REPO_ROOT/"_run/current/GPU_LEASE.lock",
        task_id=packet["task_id"], attempt_id="attempt_0001", gpu_id=0,
        executor_epoch=packet["writer"]["executor_epoch"], fencing_token=packet["writer"]["fencing_token"],
    )
    lease.__enter__()
    try:
        adapter, build_evidence = build_pinned_adapter(
            official_code_root=REPO_ROOT/"vendor/SAM3", checkpoint_path=checked(packet["inputs"]["sam_checkpoint"]))
    except BaseException:
        lease.__exit__(*sys.exc_info())
        raise

    rows = []
    visible_gate_rows = []
    started = time.monotonic()
    try:
        for frame in range(171):
            ok, image = capture.read()
            if not ok or image.shape[:2] != (960, 1280):
                raise RuntimeError(f"FRAME_DECODE:{frame}")
            mask = np.zeros(image.shape[:2], dtype=bool)
            metadata = {"direct_sam31_invoked": False, "single_frame_state": False,
                        "video_propagation": False, "strategy": "PRE_ONSET_EMPTY"}
            frame_gates = {}
            if frame >= ONSET:
                hints = hint_points(frame)
                role_union = np.zeros_like(mask)
                for ref in role_by_frame[frame]["role_masks"].values():
                    role_union |= cv2.imread(str(checked(ref)), cv2.IMREAD_GRAYSCALE) > 0
                frame_dir = output/"sam_inputs"/f"{frame:05d}"
                save(frame_dir/"000000.png", image)

                def infer(positive, negative):
                    state = adapter.model.init_state(resource_path=str(frame_dir), offload_video_to_cpu=True,
                                                     async_loading_frames=False)
                    try:
                        positive_tensor = torch.tensor([[x/1280, y/960] for x, y in positive], dtype=torch.float32)
                        adapter.model.add_prompt(inference_state=state, frame_idx=0, points=positive_tensor,
                            point_labels=torch.ones(len(positive), dtype=torch.int32), obj_id=301,
                            rel_coordinates=True, clear_old_points=True, output_prob_thresh=.5)
                        points = positive+negative
                        _, outputs = adapter.model.add_prompt(inference_state=state, frame_idx=0,
                            points=torch.tensor([[x/1280, y/960] for x, y in points], dtype=torch.float32),
                            point_labels=torch.tensor([1]*len(positive)+[0]*len(negative), dtype=torch.int32),
                            obj_id=301, rel_coordinates=True, clear_old_points=True, output_prob_thresh=.5)
                        return normalize(outputs, 960, 1280, 301)
                    finally:
                        state.clear()

                point_positive = [(float(x), float(y)) for x, y in hints]
                candidate = infer(point_positive, [])
                candidate_gates, candidate_metrics = quality(candidate, image, role_union, hints, [])
                selected_positive, selected_negative = point_positive, []
                strategy = "POINTS_ONLY"
                mask = candidate
                frame_gates, selected_metrics = candidate_gates, candidate_metrics
                refinement = None
                if not all(candidate_gates.values()):
                    component = select_rgb_component(rgb_components(image, hints))
                    if component is not None:
                        refine_positive, refine_negative = component_prompt(component, hints, image.shape[:2])
                        raw_refined = infer(refine_positive, refine_negative)
                        refined, discarded, positive_hits = associated_component(raw_refined, refine_positive)
                        refined_gates, refined_metrics = quality(refined, image, role_union, hints, refine_negative)
                        refinement = {"gates": refined_gates, "metrics": refined_metrics,
                                      "discarded_pixels": discarded, "positive_hits": positive_hits}
                        mask, frame_gates, selected_metrics = refined, refined_gates, refined_metrics
                        selected_positive, selected_negative = refine_positive, refine_negative
                        strategy = "ROLE_AWARE_NEGATIVE_REFINEMENT"
                metadata.update(direct_sam31_invoked=True, single_frame_state=True, strategy=strategy,
                                positive_points=selected_positive, negative_points=selected_negative,
                                selected_metrics=selected_metrics, points_only_gates=candidate_gates,
                                points_only_metrics=candidate_metrics, refinement=refinement)
                visible_gate_rows.append(all(frame_gates.values()))
            mask_ref = save(output/"physical_object_0_masks"/f"{frame:05d}.png", mask.astype(np.uint8)*255)
            overlay = image.copy()
            overlay[mask] = (.35*overlay[mask]+.65*np.asarray([0,255,255])).astype(np.uint8)
            color = (0,255,0) if (frame < ONSET or all(frame_gates.values())) else (0,0,255)
            cv2.putText(overlay, f"frame {frame:03d} {metadata['strategy']}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2)
            review.write(overlay)
            rows.append({"source_frame": frame, "selected_rgb_decoded_sha256": array_sha(image),
                         "observed": bool(mask.any()), "mask": mask_ref, "metadata": metadata, "gates": frame_gates})
            if frame % 10 == 0:
                print(json.dumps({"frame": frame, "strategy": metadata["strategy"],
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
    finally:
        adapter.predictor.shutdown()
        capture.release()
        review.release()
        lease.__exit__(*sys.exc_info())

    canary = cv2.imread(str(checked(packet["inputs"]["accepted_canary_125"])), cv2.IMREAD_GRAYSCALE) > 0
    produced = cv2.imread(str(Path(rows[125]["mask"]["path"])), cv2.IMREAD_GRAYSCALE) > 0
    iou = float(np.logical_and(canary, produced).sum()/max(int(np.logical_or(canary, produced).sum()), 1))
    hard = {
        "frame_count_exact": len(rows) == 171,
        "pre_onset_empty": not any(row["observed"] for row in rows[:ONSET]),
        "all_visible_direct_sam31": all(row["metadata"]["direct_sam31_invoked"] for row in rows[ONSET:]),
        "all_visible_single_frame_state": all(row["metadata"]["single_frame_state"] for row in rows[ONSET:]),
        "all_visible_observed": all(row["observed"] for row in rows[ONSET:]),
        "all_frame_instance_gates": all(visible_gate_rows),
        "accepted_canary_125_iou": iou >= .50,
        "video_propagation_absent": True,
    }
    manifest = {"schema_version": "exact78-cards042-direct-object-manifest-v2", "session": "play_cards_0902_042",
                "identity": "single A-diamond physical instance 0", "identity_onset": ONSET, "frames": rows}
    atomic_json(output/"FRAME_MANIFEST.json", manifest)
    passed = all(hard.values())
    result = {"schema_version": "exact78-cards042-direct-object-result-v2",
              "status": "PASS_GRADE_B_CANDIDATE" if passed else "FAILED_QUALITY_C",
              "grade": "B" if passed else "C", "downstream_authorized": passed,
              "hard_gates": hard,
              "metrics": {"observed_count": sum(row["observed"] for row in rows),
                          "canary_125_iou": iou,
                          "points_only_count": sum(row["metadata"].get("strategy") == "POINTS_ONLY" for row in rows),
                          "refinement_count": sum(row["metadata"].get("strategy") == "ROLE_AWARE_NEGATIVE_REFINEMENT" for row in rows)},
              "artifacts": {"frame_manifest": artifact_ref(output/"FRAME_MANIFEST.json"),
                            "review_video": artifact_ref(output/"review.mp4")},
              "build_evidence": build_evidence,
              "claim_limit": "Single Cards042 object identity Mask only; no Clean, Object6D, contact, baseline or deletion authority."}
    atomic_json(output/"RESULT.json", result)
    atomic_json(root/"RESULT.json", {"status": result["status"], "object_result": artifact_ref(output/"RESULT.json"),
                                      "deletion_authorized": False, "baseline_adopted": False})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
