"""Gated six-call Mask14 chip-object prompt candidate; unregistered by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


def _active_packet(output_root: Path) -> dict[str, Any]:
    from chaoyang.governance.common import REPO_ROOT, artifact_ref
    from chaoyang.governance import exact78_chips_object_prompt_mask14_candidate_v1 as design

    index = json.loads((REPO_ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    path = REPO_ROOT / "tasks/current" / design.TASK_ID / "TASK_PACKET.json"
    expected = REPO_ROOT / "_run/current" / design.TASK_ID / "attempts/attempt_0001/candidate"
    if (len(active) != 1 or active[0].get("task_id") != design.TASK_ID
            or active[0].get("packet_path") != str(path.relative_to(REPO_ROOT))
            or output_root != expected or output_root.exists()):
        raise RuntimeError("MASK14_EXACT_ACTIVE_PACKET_AND_FRESH_OUTPUT_REQUIRED")
    if active[0].get("packet_sha256") != artifact_ref(path)["sha256"]:
        raise RuntimeError("MASK14_PACKET_SHA_DRIFT")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != design.TASK_ID
            or packet.get("phase") != "EXACT78_CHIPS_OBJECT_PROMPT_MASK14_DEV_CANDIDATE"
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise RuntimeError("MASK14_PACKET_IDENTITY_DRIFT")
    fresh = design.packet_value()
    if any(packet.get(key) != fresh.get(key) for key in
           ("inputs", "runtime_code", "config", "run_signature", "commands")):
        raise RuntimeError("MASK14_FROZEN_PACKET_OR_SOURCE_DRIFT")
    if design.route_status(packet) != "PUBLISHED_ROUTE_PACKET_STILL_REQUIRED":
        raise RuntimeError("MASK14_ROUTE_NOT_PUBLISHED")
    return packet


def run_candidate(*, output_root: Path) -> dict[str, Any]:
    from chaoyang.governance import exact78_chips_object_prompt_mask14_candidate_v1 as design
    from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json

    authorize_exact78_operation(design.OPERATION, ["--output-root", str(output_root)])
    packet = _active_packet(output_root)
    # Model and CUDA are imported only after the operation, packet, route,
    # frozen code, checkpoint and six source-image gates have passed.
    import cv2
    import numpy as np
    import torch
    from chaoyang.ops import run_exact78_fresh_frontend_v2 as frontend
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    torch.set_num_threads(2)
    adapter, evidence = build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3",
        checkpoint_path=Path(packet["inputs"]["sam_checkpoint"]["path"]),
    )
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    calls = 0
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for anchor, old in zip(packet["inputs"]["anchors"], design.frozen_six()):
                session, frame = anchor["session_id"], anchor["frame"]
                if (session, frame) != (old["session_id"], old["frame"]):
                    raise RuntimeError("MASK14_ROW_IDENTITY_DRIFT")
                if artifact_ref(Path(anchor["sam_input"]["path"])) != anchor["sam_input"]:
                    raise RuntimeError("MASK14_SAM_INPUT_SHA_DRIFT")
                jpg = Path(anchor["sam_input"]["path"])
                frame_dir = jpg.parent
                state = adapter.model.init_state(
                    resource_path=str(frame_dir), offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                try:
                    _, prediction = adapter.model.add_prompt(
                        inference_state=state, frame_idx=0,
                        text_str=packet["config"]["new_object_prompt"],
                        output_prob_thresh=packet["config"]["output_prob_thresh"],
                    )
                    instances, scores = frontend.normalize(prediction, (960, 1280))
                    chosen_scores = scores[scores >= packet["config"]["instance_score_gate"]]
                    chosen = instances[scores >= packet["config"]["instance_score_gate"]]
                    obj = np.any(chosen, axis=0) if len(chosen) else np.zeros((960, 1280), bool)
                    calls += 1
                    if calls > packet["config"]["model_calls_max"]:
                        raise RuntimeError("MASK14_MODEL_CALL_BUDGET_EXCEEDED")
                finally:
                    state.clear()
                fixed = {}
                for role, ref in anchor["fixed_masks"].items():
                    if artifact_ref(Path(ref["path"])) != ref:
                        raise RuntimeError(f"MASK14_FIXED_MASK_SHA_DRIFT:{session}:{frame}:{role}")
                    image = cv2.imread(ref["path"], cv2.IMREAD_GRAYSCALE)
                    if image is None or image.shape != (960, 1280) or not np.isin(image, (0, 255)).all():
                        raise RuntimeError(f"MASK14_FIXED_MASK_DOMAIN:{session}:{frame}:{role}")
                    fixed[role] = image > 0
                derived = design.derive_masks(fixed["human"], fixed["equipment_original"],
                                              fixed["equipment_added"], obj)
                if not np.array_equal(derived["equipment"], fixed["equipment"]):
                    raise RuntimeError("MASK14_EQUIPMENT_SOURCE_DRIFT")
                refs = {role: frontend.save(
                    output_root / "masks" / session / role / f"{frame:06d}.png",
                    value.astype(np.uint8) * 255)
                    for role, value in derived.items() if role != "equipment"}
                rows.append({
                    "session_id": session, "frame": frame,
                    "raw_image": anchor["raw_image"], "sam_input": anchor["sam_input"],
                    "prompts": design.object_only_prompts(old["prompts"]),
                    "fixed_masks": anchor["fixed_masks"],
                    "old_object": anchor["old_object"], "new_masks": refs,
                    "instance_count": int(len(chosen)),
                    "selected_scores": [float(score) for score in chosen_scores],
                    "object_pixels": int(obj.sum()),
                    "protected_pixels": int(derived["protected"].sum()),
                    "removal_pixels": int(derived["removal"].sum()),
                    "unknown_overlap_pixels": int(derived["unknown"].sum()),
                    "review": "PENDING_HUMAN_VISIBLE_CHIP_AND_FALSE_POSITIVE",
                })
    finally:
        adapter.predictor.shutdown()
    if len(rows) != 6 or calls != 6:
        raise RuntimeError("MASK14_ANCHOR_OR_CALL_COUNT_DRIFT")
    record = {
        "schema_version": "exact78-chips-object-prompt-mask14-v1",
        "status": "MASK_CANDIDATE_UNREVIEWED", "rows": rows,
        "model_calls": calls, "model_evidence": evidence,
        "old_object_prompt": design.OLD_OBJECT,
        "new_object_prompt": design.NEW_OBJECT,
        "mask13_human_equipment_reused": True,
        "clean_called": False, "visual_qualified": False,
        "training_eligible": False, "review": "PENDING_HUMAN",
    }
    atomic_json(output_root / "MASK_ANCHOR.json", record)
    result = {
        "schema_version": "exact78-chips-object-prompt-mask14-result-v1",
        "status": "MASK_CANDIDATE_UNREVIEWED", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"],
        "mask_anchor": artifact_ref(output_root / "MASK_ANCHOR.json"),
        "anchors_total": 6, "model_calls": calls,
        "clean_called": False, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "claim_limit": packet["claim_limit"],
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    from chaoyang.governance import exact78_chips_object_prompt_mask14_candidate_v1 as design

    authorize_exact78_operation(design.OPERATION, args_in)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(args_in)
    result = run_candidate(output_root=args.output_root)
    print(json.dumps({"status": result["status"], "mask_anchor": result["mask_anchor"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
