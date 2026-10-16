"""Candidate-only tracker equipment synonym union over pinned frontend v3.

Only the tracker MASK phase differs. No session-specific prompt, ROI, model,
threshold, object-protection rule, or ProPainter change is admitted. Register
with a fresh finite Task Packet before any model invocation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_exact78_fresh_frontend_v2 as frontend_v2
from chaoyang.ops import run_exact78_fresh_frontend_v3 as frontend_v3


PINNED_V3_SHA256 = "64f32975f85ffcec7b05f8ea011f6efae7a957094163ace8113c133a690bd841"
ORIGINAL_MASKS = frontend_v2.masks
TRACKER_EQUIPMENT_PROMPTS = {
    "equipment_tracker": "a wrist worn tracking device",
    "equipment_wristband": "a wristband",
}
EMPTY_DEVICE_STATUS = "MASK_DEVICE_UNVERIFIED"


def prompt_plan(spec: dict[str, Any]) -> dict[str, str]:
    """The only new prompts are a frozen, source-family-wide tracker pair."""
    if spec["source_family"] not in frontend_v2.TRACKER_SOURCES:
        return frontend_v2._prompts(spec)
    existing = frontend_v2._prompts(spec)
    if existing["equipment"] != TRACKER_EQUIPMENT_PROMPTS["equipment_tracker"]:
        raise RuntimeError("PINNED_TRACKER_PROMPT_DRIFT")
    return {
        "human": existing["human"],
        "object": existing["object"],
        **TRACKER_EQUIPMENT_PROMPTS,
    }


def compose_tracker_roles(selected: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Use v2's exact human/object Boolean rules after equipment prompt union."""
    if set(selected) != {"human", "object", *TRACKER_EQUIPMENT_PROMPTS}:
        raise ValueError("TRACKER_ROLE_SET_DRIFT")
    shape = selected["human"].shape
    if any(value.shape != shape or value.dtype != np.bool_ for value in selected.values()):
        raise ValueError("TRACKER_ROLE_MASK_DOMAIN")
    tracker = selected["equipment_tracker"]
    wristband = selected["equipment_wristband"]
    equipment = tracker | wristband
    human_equipment = selected["human"] | equipment
    object_mask = selected["object"]
    unknown = human_equipment & object_mask
    protected = object_mask & ~human_equipment
    removal = human_equipment & ~object_mask
    if np.any(removal & object_mask) or np.any(protected & human_equipment):
        raise RuntimeError("OBJECT_PROTECTION_RULE_DRIFT")
    images = {
        **selected,
        "equipment": equipment,
        "human_equipment": human_equipment,
        "unknown": unknown,
        "protected": protected,
        "removal": removal,
    }
    pixels = {
        "equipment_tracker": int(tracker.sum()),
        "equipment_wristband": int(wristband.sum()),
        "equipment_union": int(equipment.sum()),
        "equipment_wristband_only": int((wristband & ~tracker).sum()),
    }
    return images, pixels


def equipment_provenance(candidate_pixels: dict[str, int]) -> tuple[list[str], str]:
    sources = [role for role in TRACKER_EQUIPMENT_PROMPTS if candidate_pixels[role] > 0]
    return sources, ("MODEL_OUTPUT_UNREVIEWED" if sources else EMPTY_DEVICE_STATUS)


def masks_tracker(spec: dict[str, Any], output: Path) -> None:
    """Reproduce v2 MASK inference, adding only a second tracker device prompt."""
    if spec["source_family"] not in frontend_v2.TRACKER_SOURCES:
        raise ValueError("TRACKER_MASK_SOURCE_REQUIRED")
    import torch

    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    raw_record = load_json(output / "RAW.json")
    if raw_record["session_id"] != spec["session_id"] or len(raw_record["rows"]) != spec["frame_count"]:
        raise RuntimeError("RAW_STAGE_NOT_SEALED")
    torch.set_num_threads(2)
    adapter, evidence = build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3",
        checkpoint_path=frontend_v2.checked(spec["sam_checkpoint"]),
    )
    prompts = prompt_plan(spec)
    rows = []
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for raw_row in raw_record["rows"]:
                index = raw_row["decode_display_index"]
                image = cv2.imread(str(frontend_v2.checked(raw_row["model_input"])))
                if image is None or image.shape != (960, 1280, 3):
                    raise RuntimeError(f"MODEL_INPUT_DOMAIN:{index}")
                frame_dir = output / "sam_inputs" / f"{index:06d}"
                sam_input = frontend_v2.save(frame_dir / "000000.jpg", image)
                selected: dict[str, np.ndarray] = {}
                counts: dict[str, int] = {}
                for role, prompt in prompts.items():
                    state = adapter.model.init_state(
                        resource_path=str(frame_dir), offload_video_to_cpu=True,
                        async_loading_frames=False,
                    )
                    try:
                        _, prediction = adapter.model.add_prompt(
                            inference_state=state, frame_idx=0, text_str=prompt,
                            output_prob_thresh=0.5,
                        )
                        instances, scores = frontend_v2.normalize(prediction, image.shape[:2])
                        chosen = instances[scores >= 0.5]
                        selected[role] = (np.any(chosen, axis=0) if len(chosen)
                                          else np.zeros(image.shape[:2], bool))
                        counts[role] = int(len(chosen))
                    finally:
                        state.clear()
                images, candidate_pixels = compose_tracker_roles(selected)
                counts["equipment"] = counts["equipment_tracker"] + counts["equipment_wristband"]
                refs = {
                    key: frontend_v2.save(output / "masks" / key / f"{index:06d}.png",
                                          value.astype(np.uint8) * 255)
                    for key, value in images.items()
                }
                sources, status = equipment_provenance(candidate_pixels)
                rows.append({
                    "decode_display_index": index,
                    "model_input": raw_row["model_input"], "sam_input": sam_input,
                    "sam_output_index": 0, "masks": refs, "instance_counts": counts,
                    "equipment_prompt_sources": sources,
                    "equipment_candidate_pixels": candidate_pixels,
                    "removal_pixels": int(images["removal"].sum()),
                    "unknown_overlap_pixels": int(images["unknown"].sum()),
                    "protected_pixels": int(images["protected"].sum()),
                    "removal_object_overlap_pixels": 0,
                    "role_status": status,
                    "mask_quality": "MODEL_OUTPUT_NOT_GROUND_TRUTH",
                })
                if index % 10 == 0:
                    print(json.dumps({"stage": "SAM", "frame": index,
                                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        adapter.predictor.shutdown()
    if not rows or not any(row["removal_pixels"] for row in rows):
        raise RuntimeError("EMPTY_HUMAN_EQUIPMENT_MASK_ALL_FRAMES")
    atomic_json(output / "MASK.json", {
        "schema_version": "exact78-fresh-frontend-mask-v4-tracker-equipment",
        "session_id": spec["session_id"], "source_family": spec["source_family"],
        "raw_result": artifact_ref(output / "RAW.json"),
        "rows": rows, "prompts": prompts, "probability_threshold": 0.5,
        "build_evidence": evidence, "fresh_model_inference": True,
        "physical_instance_tracking": "NOT_CLAIMED",
        "visible_object_identity": "UNVERIFIED_TEXT_UNION",
        "object_protection_rule": "V2_UNCHANGED_OBJECT_MINUS_HUMAN_EQUIPMENT",
        "sleeve_cable_status": "NOT_APPLICABLE",
        "training_eligible": False,
    })


def masks(spec: dict[str, Any], output: Path) -> None:
    if spec["source_family"] == frontend_v2.VST_SOURCE:
        return ORIGINAL_MASKS(spec, output)
    return masks_tracker(spec, output)


def verify_pinned_closure() -> None:
    if not PINNED_V3_SHA256:
        raise RuntimeError("V4_V3_SHA_NOT_FROZEN")
    if artifact_ref(Path(frontend_v3.__file__).resolve(strict=True))["sha256"] != PINNED_V3_SHA256:
        raise RuntimeError("V4_PINNED_V3_SHA_DRIFT")
    frontend_v3.verify_pinned_closure()


def main() -> int:
    """Patch only this process's MASK function; v3 manages CLEAN verifier."""
    verify_pinned_closure()
    original = frontend_v2.masks
    if original is not ORIGINAL_MASKS or original.__module__ != frontend_v2.__name__:
        raise RuntimeError("V4_V2_MASK_ALREADY_PATCHED")
    try:
        frontend_v2.masks = masks
        return frontend_v3.main()
    finally:
        frontend_v2.masks = original


if __name__ == "__main__":
    raise SystemExit(main())
