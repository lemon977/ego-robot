"""Generic Raw -> fresh SAM3.1 -> provenance-labelled ProPainter frontend.

No historical masks, Clean images, donor maps, depth or Object6D are inputs.
When no independently qualified donor is available, removal pixels use the
already-selected synthetic fallback; this is offline visualization, not truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


def checked(ref):
    path = Path(ref["path"])
    if artifact_ref(path) != ref:
        raise RuntimeError(f"SOURCE_SHA_DRIFT:{path}")
    return path


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"NO_CLOBBER_IMAGE_WRITE:{path}")
    return artifact_ref(path)


def raw(spec, output):
    capture = cv2.VideoCapture(str(checked(spec["stereo"])))
    with np.load(checked(spec["camera_maps"]), allow_pickle=False) as maps:
        mx = maps["rectify_to_native_x"].astype(np.float32)
        my = maps["rectify_to_native_y"].astype(np.float32)
    rows = []
    try:
        for frame in range(spec["frame_count"]):
            ok, image = capture.read()
            if not ok or image.shape[:2] != (1536, 4096):
                raise RuntimeError(f"STEREO_DECODE_OR_DOMAIN:{frame}")
            eye = image[:, :2048] if spec["source_index"] == 0 else image[:, 2048:]
            rgb = cv2.remap(eye, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rows.append({"frame_id": frame, "raw": save(output / "raw" / f"{frame:06d}.png", rgb),
                         "array_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        if capture.read()[0]:
            raise RuntimeError("STEREO_EXTRA_FRAME")
    finally:
        capture.release()
    atomic_json(output / "RAW.json", {"session_id": spec["session_id"], "rows": rows,
                                      "image_domain": spec["image_domain"], "stereo": spec["stereo"],
                                      "camera_maps": spec["camera_maps"], "metric_authority": False})


def normalize(outputs, shape):
    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    value = outputs.get("out_binary_masks")
    masks = np.zeros((0, *shape), bool) if value is None else numpy(value)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = np.squeeze(masks, axis=1)
    if masks.ndim == 2:
        masks = masks[None]
    scores = numpy(outputs.get("out_probs", [])).reshape(-1)
    if masks.shape[1:] != shape or len(masks) != len(scores) or not np.isfinite(scores).all():
        raise RuntimeError("SAM_OUTPUT_DOMAIN")
    return masks.astype(bool), scores


def masks(spec, output):
    import torch
    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter
    torch.set_num_threads(2)
    adapter, evidence = build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3",
        checkpoint_path=checked(spec["sam_checkpoint"]),
    )
    raw_rows = load_json(output / "RAW.json")["rows"]
    rows = []
    started = time.time()
    prompts = {
        "human": "a person's hand and forearm",
        "equipment": "a wrist worn tracking device",
        "object": "a playing card" if spec["task"] == "poker" else "a potato chip",
    }
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in raw_rows:
                frame = row["frame_id"]
                image = cv2.imread(str(checked(row["raw"])))
                shape = image.shape[:2]
                frame_dir = output / "sam_inputs" / f"{frame:06d}"
                save(frame_dir / "000000.jpg", image)
                selected = {}
                counts = {}
                for role, prompt in prompts.items():
                    state = adapter.model.init_state(resource_path=str(frame_dir), offload_video_to_cpu=True,
                                                     async_loading_frames=False)
                    try:
                        _, prediction = adapter.model.add_prompt(inference_state=state, frame_idx=0,
                                                                 text_str=prompt, output_prob_thresh=0.5)
                        instances, scores = normalize(prediction, shape)
                        selected[role] = np.any(instances[scores >= 0.5], axis=0) if len(instances) else np.zeros(shape, bool)
                        counts[role] = int((scores >= 0.5).sum())
                    finally:
                        state.clear()
                human = selected["human"] | selected["equipment"]
                overlap_unknown = human & selected["object"]
                protected = selected["object"] & ~human
                removal = human & ~selected["object"]
                entries = {}
                for key, array in {**selected, "unknown": overlap_unknown, "protected": protected, "removal": removal}.items():
                    entries[key] = save(output / "masks" / key / f"{frame:06d}.png", array.astype(np.uint8)*255)
                rows.append({"frame_id": frame, "raw": row["raw"], "masks": entries,
                             "instance_counts": counts, "removal_pixels": int(removal.sum()),
                             "unknown_overlap_pixels": int(overlap_unknown.sum()),
                             "mask_quality": "MODEL_OUTPUT_NOT_GROUND_TRUTH"})
                if frame % 10 == 0:
                    print(json.dumps({"stage": "SAM", "frame": frame, "seconds": round(time.time()-started, 1)}), flush=True)
    finally:
        adapter.predictor.shutdown()
    if not rows or not any(r["removal_pixels"] for r in rows):
        raise RuntimeError("EMPTY_HUMAN_MASK_ALL_FRAMES")
    atomic_json(output / "MASK.json", {
        "session_id": spec["session_id"], "rows": rows, "build_evidence": evidence,
        "prompts": prompts, "probability_threshold": 0.5, "fresh_model_inference": True,
        "frame_independent_segmentation": True, "physical_instance_tracking": "NOT_CLAIMED",
        "unknown_policy": "human/object disagreement not repaired or passed as background",
    })


def prepare(spec, output):
    rows = load_json(output / "MASK.json")["rows"]
    prepared = []
    for row in rows:
        frame = row["frame_id"]
        image = cv2.imread(str(checked(row["raw"])))
        removal = cv2.imread(str(checked(row["masks"]["removal"])), cv2.IMREAD_UNCHANGED) > 0
        save(output / "propainter_input/frames" / f"{frame:06d}.png", cv2.resize(image, (960,720), interpolation=cv2.INTER_AREA))
        save(output / "propainter_input/masks" / f"{frame:06d}.png", cv2.resize(removal.astype(np.uint8)*255, (960,720), interpolation=cv2.INTER_NEAREST))
        prepared.append({"frame_id": frame, "raw": row["raw"], "masks": row["masks"]})
    atomic_json(output / "PREPARE.json", {"rows": prepared, "qualified_real_donors": 0,
        "donor_status": "NONE_INDEPENDENTLY_VALIDATED_IN_CURRENT_DOMAIN",
        "fallback": "SYNTHETIC_PROPAINTER_NOT_OBSERVED_BACKGROUND",
        "old_depth_or_donor_maps_consumed": False})


def clean(spec, output):
    # The unmodified pinned vendor performs the same fixed model computation.
    command = [sys.executable, "-B", str(checked(spec["propainter_entry"])),
               "--video", str(output / "propainter_input/frames"), "--mask", str(output / "propainter_input/masks"),
               "--output", str(output / "propainter_model"), "--width", "960", "--height", "720",
               "--mask_dilation", "4", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30", "--save_frames", "--fp16"]
    with (output / "propainter.log").open("x") as log:
        result = subprocess.run(command, cwd=REPO_ROOT / "vendor/ProPainter", stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"PROPAINTER_RETURN_CODE:{result.returncode}")
    prepared = load_json(output / "PREPARE.json")["rows"]
    model_root = output / "propainter_model/frames/frames"
    if sorted(p.name for p in model_root.glob("*.png")) != [f"{i:04d}.png" for i in range(len(prepared))]:
        raise RuntimeError("PROPAINTER_FRAME_SET")
    rows = []
    for row in prepared:
        frame = row["frame_id"]
        image = cv2.imread(str(checked(row["raw"])))
        removal = cv2.imread(str(checked(row["masks"]["removal"])), cv2.IMREAD_UNCHANGED) > 0
        protected = cv2.imread(str(checked(row["masks"]["protected"])), cv2.IMREAD_UNCHANGED) > 0
        unknown = cv2.imread(str(checked(row["masks"]["unknown"])), cv2.IMREAD_UNCHANGED) > 0
        generated = cv2.imread(str(model_root / f"{frame:04d}.png"))
        if generated is None:
            raise RuntimeError("PROPAINTER_FRAME_UNREADABLE")
        generated = cv2.resize(generated, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        final = image.copy()
        final[removal] = generated[removal]
        if not np.array_equal(final[~removal], image[~removal]) or not np.array_equal(final[protected], image[protected]):
            raise RuntimeError("CLEAN_PROTECTED_OR_OUTSIDE_CHANGED")
        provenance = np.zeros(image.shape[:2], np.uint8)
        provenance[removal] = 3  # SYNTHETIC_PROPAINTER
        provenance[protected] = 2  # VISIBLE_OBJECT_TARGET_RAW
        provenance[unknown] = 4  # UNKNOWN_TARGET_RAW_RETAINED
        rows.append({"frame_id": frame, "clean": save(output / "clean" / f"{frame:06d}.png", final),
                     "pixel_source": save(output / "pixel_source" / f"{frame:06d}.png", provenance),
                     "raw": row["raw"], "masks": row["masks"], "outside_changed": 0, "protected_changed": 0,
                     "synthetic_pixels": int(removal.sum())})
    atomic_json(output / "CLEAN.json", {"session_id": spec["session_id"], "rows": rows,
        "fresh_mask_and_clean_inference": True, "real_donor_pixels": 0,
        "source_enum": {"0": "TARGET_RAW", "2": "PROTECTED_VISIBLE_RAW", "3": "SYNTHETIC_PROPAINTER", "4": "UNKNOWN_RAW_RETAINED"},
        "training_eligible": False, "human_review": "PENDING", "quality": "NOT_YET_ADOPTED"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=("raw", "masks", "prepare", "clean"), required=True)
    args = parser.parse_args()
    spec = load_json(args.spec)
    from chaoyang.ops.run_exact78_full_pipeline_v1 import authorize
    _, task_root = authorize(spec["task_id"])
    output = Path(spec["output_root"]).resolve()
    output.relative_to(task_root)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    globals()[args.phase](spec, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
