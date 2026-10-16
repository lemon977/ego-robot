"""Unregistered, three-source white-device prompt candidate over pinned v3.

Only MASK inference adds one identical text prompt.  Existing source-specific
prompts, models, score threshold, object algebra, and Clean remain unchanged.
No model runs on import; a new fenced Task Packet and candidate CLI route are
required before ``main`` can execute.
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
from chaoyang.ops import run_exact78_fresh_frontend_v2 as v2
from chaoyang.ops import run_exact78_fresh_frontend_v3 as v3


PINNED_V3_SHA256 = "64f32975f85ffcec7b05f8ea011f6efae7a957094163ace8113c133a690bd841"
WHITE_DEVICE_PROMPT = "a white device attached to a hand or wrist"
ORIGINAL_MASKS = v2.masks


def prompt_plan(spec: dict[str, Any]) -> dict[str, str]:
    """Retain every v2 prompt; rename its equipment key only for provenance."""
    old = v2._prompts(spec)
    if spec["source_family"] not in v2.SOURCE_FAMILIES or "equipment" not in old:
        raise ValueError("SOURCE_OR_BASELINE_PROMPTS_DRIFT")
    if set(old) != ({"human", "object", "equipment"} |
                   ({"finger_sleeve", "yellow_cable"} if spec["source_family"] == v2.VST_SOURCE else set())):
        raise ValueError("BASELINE_PROMPT_ROLE_DRIFT")
    return {"human": old["human"], "object": old["object"],
            "equipment_original": old["equipment"],
            **({key: old[key] for key in ("finger_sleeve", "yellow_cable")}
               if spec["source_family"] == v2.VST_SOURCE else {}),
            "equipment_white_device": WHITE_DEVICE_PROMPT}


def compose_roles(selected: dict[str, np.ndarray], source_family: str
                  ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    expected = {"human", "object", "equipment_original", "equipment_white_device"}
    if source_family == v2.VST_SOURCE:
        expected |= {"finger_sleeve", "yellow_cable"}
    elif source_family not in v2.TRACKER_SOURCES:
        raise ValueError("UNSUPPORTED_SOURCE_FAMILY")
    if set(selected) != expected:
        raise ValueError("WHITE_DEVICE_ROLE_SET_DRIFT")
    shape = selected["human"].shape
    if any(value.shape != shape or value.dtype != np.bool_ for value in selected.values()):
        raise ValueError("WHITE_DEVICE_ROLE_MASK_DOMAIN")
    baseline_equipment = selected["equipment_original"].copy()
    if source_family == v2.VST_SOURCE:
        baseline_equipment |= selected["finger_sleeve"] | selected["yellow_cable"]
    baseline_human_equipment = selected["human"] | baseline_equipment
    white = selected["equipment_white_device"]
    white_only = white & ~baseline_human_equipment
    object_mask = selected["object"]
    protected_before = object_mask & ~baseline_human_equipment
    lost_protected = white_only & protected_before
    # A no-threshold fail-closed pre-Clean gate: a new device prompt cannot
    # revoke even one previously protected visible-object candidate pixel.
    if np.any(lost_protected):
        raise RuntimeError("VISIBLE_OBJECT_PROTECTION_REGRESSION")
    equipment = selected["equipment_original"] | white
    human_equipment = selected["human"] | equipment
    if source_family == v2.VST_SOURCE:
        human_equipment |= selected["finger_sleeve"] | selected["yellow_cable"]
    unknown = human_equipment & object_mask
    protected = object_mask & ~human_equipment
    removal = human_equipment & ~object_mask
    if np.any(removal & object_mask) or np.any(protected & human_equipment):
        raise RuntimeError("OBJECT_PROTECTION_RULE_DRIFT")
    images = {**selected, "equipment": equipment, "human_equipment": human_equipment,
              "unknown": unknown, "protected": protected, "removal": removal}
    counts = {"equipment_original_pixels": int(selected["equipment_original"].sum()),
              "equipment_white_device_pixels": int(white.sum()),
              "equipment_white_device_new_pixels": int(white_only.sum()),
              "baseline_protected_pixels": int(protected_before.sum()),
              "protected_pixels": int(protected.sum()),
              "protected_lost_pixels": int(lost_protected.sum()),
              "baseline_human_equipment_pixels": int(baseline_human_equipment.sum()),
              "equipment_union_pixels": int(equipment.sum())}
    return images, counts


def masks(spec: dict[str, Any], output: Path) -> None:
    """One fresh SAM pass per baseline role plus the same new prompt per source."""
    import torch

    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    raw_record = load_json(output / "RAW.json")
    if raw_record["session_id"] != spec["session_id"] or len(raw_record["rows"]) != spec["frame_count"]:
        raise RuntimeError("RAW_STAGE_NOT_SEALED")
    torch.set_num_threads(2)
    adapter, evidence = build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3",
        checkpoint_path=v2.checked(spec["sam_checkpoint"]),
    )
    prompts = prompt_plan(spec)
    rows = []
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for raw_row in raw_record["rows"]:
                index = raw_row["decode_display_index"]
                image = cv2.imread(str(v2.checked(raw_row["model_input"])))
                if image is None or image.shape != (960, 1280, 3):
                    raise RuntimeError(f"MODEL_INPUT_DOMAIN:{index}")
                frame_dir = output / "sam_inputs" / f"{index:06d}"
                sam_input = v2.save(frame_dir / "000000.jpg", image)
                selected: dict[str, np.ndarray] = {}
                instances_by_role: dict[str, int] = {}
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
                        instances, scores = v2.normalize(prediction, image.shape[:2])
                        chosen = instances[scores >= 0.5]
                        selected[role] = (np.any(chosen, axis=0) if len(chosen)
                                          else np.zeros(image.shape[:2], bool))
                        instances_by_role[role] = int(len(chosen))
                    finally:
                        state.clear()
                images, pixels = compose_roles(selected, spec["source_family"])
                equipment_status = ("MODEL_OUTPUT_UNREVIEWED" if pixels["equipment_union_pixels"]
                                    else "MASK_DEVICE_UNVERIFIED")
                refs = {key: v2.save(output / "masks" / key / f"{index:06d}.png",
                                     value.astype(np.uint8) * 255)
                        for key, value in images.items()}
                rows.append({
                    "decode_display_index": index, "model_input": raw_row["model_input"],
                    "sam_input": sam_input, "sam_output_index": 0, "masks": refs,
                    "instance_counts": instances_by_role,
                    "equipment_prompt_sources": (["equipment_original"] if pixels["equipment_original_pixels"] else [])
                        + (["equipment_white_device"] if pixels["equipment_white_device_pixels"] else []),
                    "equipment_candidate_pixels": pixels,
                    "removal_pixels": int(images["removal"].sum()),
                    "unknown_overlap_pixels": int(images["unknown"].sum()),
                    "protected_pixels": int(images["protected"].sum()),
                    "role_status": (v2.MASK_MISSING_SLEEVE_CABLE
                                    if spec["source_family"] == v2.VST_SOURCE else equipment_status),
                    "equipment_status": equipment_status,
                    "sleeve_cable_status": (v2.MASK_MISSING_SLEEVE_CABLE
                                            if spec["source_family"] == v2.VST_SOURCE else "NOT_APPLICABLE"),
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
        "schema_version": "exact78-fresh-frontend-mask-v5-white-device",
        "session_id": spec["session_id"], "source_family": spec["source_family"],
        "raw_result": artifact_ref(output / "RAW.json"),
        "rows": rows, "prompts": prompts, "probability_threshold": 0.5,
        "build_evidence": evidence, "fresh_model_inference": True,
        "physical_instance_tracking": "NOT_CLAIMED",
        "visible_object_identity": "UNVERIFIED_TEXT_UNION",
        "object_protection_rule": "V2_UNCHANGED_OBJECT_MINUS_HUMAN_EQUIPMENT",
        "preclean_stop_gate": "ZERO_NEW_DEVICE_OVERLAP_WITH_BASELINE_PROTECTED",
        "sleeve_cable_status": (v2.MASK_MISSING_SLEEVE_CABLE
                                if spec["source_family"] == v2.VST_SOURCE else "NOT_APPLICABLE"),
        "training_eligible": False,
    })


def verify_pinned_closure() -> None:
    if artifact_ref(Path(v3.__file__).resolve(strict=True))["sha256"] != PINNED_V3_SHA256:
        raise RuntimeError("V5_PINNED_V3_SHA_DRIFT")
    v3.verify_pinned_closure()


def main() -> int:
    """Process-local patch; v3/v2 retain the existing packet and writer gate."""
    verify_pinned_closure()
    original = v2.masks
    if original is not ORIGINAL_MASKS or original.__module__ != v2.__name__:
        raise RuntimeError("V5_V2_MASK_ALREADY_PATCHED")
    try:
        v2.masks = masks
        return v3.main()
    finally:
        v2.masks = original


if __name__ == "__main__":
    raise SystemExit(main())
