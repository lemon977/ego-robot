"""CPU-only V5 Scene write-mask diagnostic; never a product Clean input."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.run_v5_scene import (
    _mask_files, load, merge_roles, output_root, ref, save,
)
from chaoyang.pipeline.v5_scene import build_scene_mask_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load(args.spec)
    project = Path(spec["project_root"]).resolve(strict=True)
    domain_path = Path(spec["domain_manifest"])
    domain = load(domain_path)
    sid = spec["session_id"]
    mask_paths = [Path(value) for value in spec["mask_manifests"]]
    mask_docs = [load(value) for value in mask_paths]
    if domain["session_id"] != sid or any(mask["session_id"] != sid for mask in mask_docs):
        raise ValueError("SESSION_MISMATCH")
    if any(mask["domain"] != domain["image_domain"] or
           mask["frame_count"] != domain["frame_count"] for mask in mask_docs):
        raise ValueError("PIXEL_DOMAIN_MISMATCH")
    frame_count = domain["frame_count"]
    if frame_count != domain.get("full_source_frame_count", frame_count) or \
       len(domain["frames"]) != frame_count:
        raise ValueError("FULL_SESSION_REQUIRED")
    shape = (int(domain["height"]), int(domain["width"]))
    destination = output_root(project, args.output)
    for name in ("write", "protect", "context_exclude", "unknown"):
        (destination / name).mkdir()
    cache = {}
    rows = []
    for i in range(frame_count):
        start, stop = max(0, i - 2), min(frame_count, i + 3)
        for j in range(start, stop):
            if j not in cache:
                cache[j] = merge_roles(_mask_files(mask_docs, j, shape), shape)
        for j in list(cache):
            if j < start:
                del cache[j]
        scene = build_scene_mask_window([cache[j] for j in range(start, stop)],
                                        i - start, i, False)
        current = cache[i]
        write = scene["write"]
        protect = scene["protect"]
        context = scene["context_exclude"]
        unknown = scene["unknown"]
        if np.any(protect):
            raise AssertionError("DIAGNOSTIC_PROTECTION_MUST_BE_EMPTY")
        paths = {}
        for name, value in (("write", write), ("protect", protect),
                            ("context_exclude", context), ("unknown", unknown)):
            path = destination / name / f"{i:06d}.png"
            if not cv2.imwrite(str(path), value.astype(np.uint8) * 255):
                raise RuntimeError(f"IMAGE_WRITE:{path}")
            paths[name] = str(path)
        rows.append({"frame_id": i, **paths, **scene["stats"],
                     "write_object_candidate_px": int((write & current.object_visible).sum()),
                     "write_device_observed_px": int((write & current.device).sum()),
                     "device_observation_empty": not bool(current.device.any())})
    manifest = {
        "schema_version": "v5-scene-diagnostic-mask-v1", "session_id": sid,
        "frame_count": frame_count, "image_domain": domain["image_domain"],
        "source_domain": ref(domain_path), "source_masks": [ref(path) for path in mask_paths],
        "config": ref(args.spec), "code": ref(Path(__file__)),
        "rows": rows, "trusted_object_frames": [],
        "semantic_ready_for_product": False, "visual_quality_adopted": False,
        "product_clean_allowed": False, "geometry_input_allowed": False,
        "control_ground_truth": False, "training_eligible": False,
    }
    path = destination / "DIAGNOSTIC_MASK_MANIFEST.json"
    save(path, manifest)
    summary = {
        "status": "DIAGNOSTIC_ONLY", "session_id": sid, "manifest": ref(path),
        "device_observation_empty_frames": sum(row["device_observation_empty"] for row in rows),
        "frames_writing_object_candidate": sum(row["write_object_candidate_px"] > 0 for row in rows),
        "write_object_candidate_px_total": sum(row["write_object_candidate_px"] for row in rows),
        "product_clean_allowed": False,
        "reason": "NO_TRUSTED_VISIBLE_OBJECT_PROTECTION_AND_DEVICE_ROLE_QUALITY_UNVERIFIED",
    }
    save(destination / "RESULT.json", summary)
    print(destination / "RESULT.json")


if __name__ == "__main__":
    main()
