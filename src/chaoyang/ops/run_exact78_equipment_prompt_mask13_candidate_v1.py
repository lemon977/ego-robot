"""Gated, unregistered equipment-only Mask13 candidate against sealed Mask12.

Only an additional equipment text query changes.  The sealed Mask12 human,
object, original equipment and VST attachment queries are unchanged.  This
operation needs a later route and packet; it never calls ProPainter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_equipment_prompt_mask13_candidate_v1"


def build_prompts(original: dict[str, str], config: dict[str, Any], source_family: str) -> dict[str, str]:
    """Add one independent equipment query to the sealed Mask12 role texts."""
    if source_family not in {"TRACKER_0901", "TRACKER_0902", "VST_0915"}:
        raise RuntimeError("MASK13_UNSUPPORTED_SOURCE")
    other = config["other_role_prompts"]
    expected = {"human": config["old_human_prompt"],
                "object": other["poker_object"] if original.get("object") == other["poker_object"]
                          else other["chips_object"],
                "equipment": other["vst_equipment"] if source_family == "VST_0915"
                             else other["tracker_equipment"]}
    if source_family == "VST_0915":
        expected.update(finger_sleeve=other["vst_finger_sleeve"],
                        yellow_cable=other["vst_yellow_cable"])
    if (original != expected
            or config["new_human_prompt"] != "a person's complete hand and forearm, including anything worn on or attached to them"
            or config["equipment_prompt_added"] != "a white device attached to a hand or wrist"):
        raise RuntimeError("MASK13_BASE_OR_ADDED_PROMPT_DRIFT")
    result = {"human": config["new_human_prompt"], "object": original["object"],
              "equipment_original": original["equipment"],
              "equipment_added": config["equipment_prompt_added"]}
    if source_family == "VST_0915":
        result.update(finger_sleeve=original["finger_sleeve"],
                      yellow_cable=original["yellow_cable"])
    return result


def _active_packet() -> tuple[dict[str, Any], Path]:
    from chaoyang.governance.common import REPO_ROOT, artifact_ref
    from chaoyang.governance.exact78_equipment_prompt_mask13_candidate_v1 import TASK_ID

    index = json.loads((REPO_ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [item for item in index.get("task_packets", [])
              if item.get("execution_allowed") is True]
    if len(active) != 1 or active[0].get("task_id") != TASK_ID:
        raise RuntimeError("EXACT_MASK13_PACKET_REQUIRED")
    packet_path = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
    if (active[0].get("packet_path") != str(packet_path.relative_to(REPO_ROOT))
            or active[0].get("packet_sha256") != artifact_ref(packet_path)["sha256"]):
        raise RuntimeError("MASK13_PACKET_SHA_OR_PATH_DRIFT")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != TASK_ID
            or packet.get("phase") != "EXACT78_CLEAN_EQUIPMENT_PROMPT_MASK13_DEV_CANDIDATE"
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise RuntimeError("MASK13_PACKET_IDENTITY_DRIFT")
    attempt = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
    if str(attempt.relative_to(REPO_ROOT)) not in packet.get("write_set", []):
        raise RuntimeError("MASK13_WRITE_SET_DRIFT")
    return packet, attempt


def run_candidate(*, output_root: Path) -> dict[str, Any]:
    """Call SAM only after exact CLI/packet/writer and frozen-input checks."""
    authorize_exact78_operation(OPERATION, ["--output-root", str(output_root)])
    from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
    from chaoyang.governance.exact78_equipment_prompt_mask13_candidate_v1 import packet_value, route_status

    packet, attempt = _active_packet()
    if output_root != attempt / "candidate" or output_root.exists():
        raise RuntimeError("FRESH_MASK13_OUTPUT_IN_ATTEMPT_REQUIRED")
    fresh = packet_value()
    if any(packet.get(key) != fresh.get(key) for key in
           ("inputs", "runtime_code", "config", "run_signature", "commands")):
        raise RuntimeError("MASK13_FROZEN_PACKET_OR_SOURCE_DRIFT")
    if route_status(packet) != "PUBLISHED_ROUTE_PACKET_STILL_REQUIRED":
        raise RuntimeError("MASK13_ALGORITHM_ROUTE_NOT_PUBLISHED")
    # No algorithm modules or CUDA are imported before the exact authority and
    # same-session raw-image SHA gates above.
    import cv2
    import numpy as np
    import torch
    from chaoyang.ops import run_exact78_fresh_frontend_v2 as frontend
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    torch.set_num_threads(2)
    adapter, model_evidence = build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3",
        checkpoint_path=Path(packet["inputs"]["sam_checkpoint"]["path"]),
    )
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    calls = 0
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for scene in packet["inputs"]["sessions"]:
                original = frontend._prompts({"task": scene["task"],
                                              "source_family": scene["source_family"]})
                prompts = build_prompts(original, packet["config"], scene["source_family"])
                for anchor in scene["anchors"]:
                    index = anchor["frame"]
                    raw_image = Path(anchor["raw_image"]["path"])
                    if artifact_ref(raw_image) != anchor["raw_image"]:
                        raise RuntimeError(f"MASK13_RAW_IMAGE_SHA_DRIFT:{scene['session_id']}:{index}")
                    image = cv2.imread(str(raw_image))
                    if image is None or image.shape != (960, 1280, 3):
                        raise RuntimeError(f"MASK13_IMAGE_DOMAIN:{scene['session_id']}:{index}")
                    frame_dir = output_root / "sam_inputs" / scene["session_id"] / f"{index:06d}"
                    sam_input = frontend.save(frame_dir / "000000.jpg", image)
                    selected: dict[str, np.ndarray] = {}
                    instance_counts: dict[str, int] = {}
                    selected_scores: dict[str, list[float]] = {}
                    for role, prompt in prompts.items():
                        state = adapter.model.init_state(
                            resource_path=str(frame_dir), offload_video_to_cpu=True,
                            async_loading_frames=False,
                        )
                        try:
                            _, prediction = adapter.model.add_prompt(
                                inference_state=state, frame_idx=0, text_str=prompt,
                                output_prob_thresh=packet["config"]["output_prob_thresh"],
                            )
                            instances, scores = frontend.normalize(prediction, image.shape[:2])
                            chosen = instances[scores >= packet["config"]["instance_score_gate"]]
                            selected[role] = (np.any(chosen, axis=0) if len(chosen)
                                              else np.zeros(image.shape[:2], bool))
                            instance_counts[role] = int(len(chosen))
                            selected_scores[role] = [float(value) for value in
                                                     scores[scores >= packet["config"]["instance_score_gate"]]]
                            calls += 1
                            if calls > packet["config"]["model_calls_max"]:
                                raise RuntimeError("MASK13_MODEL_CALL_BUDGET_EXCEEDED")
                        finally:
                            state.clear()
                    selected["equipment"] = selected["equipment_original"] | selected["equipment_added"]
                    equipment = selected["equipment"].copy()
                    if scene["source_family"] == frontend.VST_SOURCE:
                        equipment |= selected["finger_sleeve"] | selected["yellow_cable"]
                    human_equipment = selected["human"] | equipment
                    unknown = human_equipment & selected["object"]
                    protected = selected["object"] & ~human_equipment
                    removal = human_equipment & ~selected["object"]
                    masks = {**selected, "human_equipment": human_equipment,
                             "unknown": unknown, "protected": protected,
                             "removal": removal}
                    refs = {role: frontend.save(
                        output_root / "masks" / scene["session_id"] / role / f"{index:06d}.png",
                        value.astype(np.uint8) * 255)
                            for role, value in masks.items()}
                    rows.append({
                        "session_id": scene["session_id"], "source_family": scene["source_family"],
                        "frame": index, "raw_image": anchor["raw_image"],
                        "sam_input": sam_input, "prompts": prompts,
                        "masks": refs, "instance_counts": instance_counts,
                        "selected_scores": selected_scores,
                        "equipment_source_pixels": {
                            "original_only": int(np.count_nonzero(selected["equipment_original"] &
                                                                   ~selected["equipment_added"])),
                            "added_only": int(np.count_nonzero(selected["equipment_added"] &
                                                                ~selected["equipment_original"])),
                            "both": int(np.count_nonzero(selected["equipment_original"] &
                                                         selected["equipment_added"])),
                            "union": int(np.count_nonzero(selected["equipment"])),
                        },
                        "removal_pixels": int(removal.sum()),
                        "unknown_overlap_pixels": int(unknown.sum()),
                        "protected_candidate_pixels": int(protected.sum()),
                        "review": "PENDING_HUMAN_VISIBLE_OBJECT_AND_DEVICE",
                    })
    finally:
        adapter.predictor.shutdown()
    if len(rows) != 12 or calls != 54:
        raise RuntimeError("MASK13_ANCHOR_OR_MODEL_CALL_COUNT_DRIFT")
    record = {
        "schema_version": "exact78-equipment-prompt-mask13-v1",
        "status": "MASK_CANDIDATE_UNREVIEWED", "rows": rows,
        "source_sessions": [item["session_id"] for item in packet["inputs"]["sessions"]],
        "model_calls": calls, "model_evidence": model_evidence,
        "human_prompt": packet["config"]["new_human_prompt"],
        "human_object_original_equipment_unchanged_from_mask12": True,
        "equipment_prompt_added": packet["config"]["equipment_prompt_added"],
        "old_mask_clean_algorithm_inputs": False,
        "clean_called": False, "visual_qualified": False,
        "training_eligible": False, "review": "PENDING_HUMAN",
    }
    atomic_json(output_root / "MASK_ANCHOR.json", record)
    result = {
        "schema_version": "exact78-equipment-prompt-mask13-result-v1",
        "status": "MASK_CANDIDATE_UNREVIEWED",
        "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "mask_anchor": artifact_ref(output_root / "MASK_ANCHOR.json"),
        "anchors_total": 12, "model_calls": calls,
        "clean_called": False, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "claim_limit": packet["claim_limit"],
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    authorize_exact78_operation(OPERATION, args_in)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(args_in)
    result = run_candidate(output_root=args.output_root)
    print(json.dumps({"status": result["status"], "mask_anchor": result["mask_anchor"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
