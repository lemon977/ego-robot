"""One-role, three-frame SAM candidate. Requires a separately published Exact78 packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_vst_finger_sleeve_prompt14_candidate_v1"


def _packet() -> tuple[dict[str, Any], Path]:
    from chaoyang.governance.common import REPO_ROOT, artifact_ref
    from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep

    index = json.loads((REPO_ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    path = REPO_ROOT / "tasks/current" / prep.TASK_ID / "TASK_PACKET.json"
    if (len(active) != 1 or active[0].get("task_id") != prep.TASK_ID
            or active[0].get("packet_path") != str(path.relative_to(REPO_ROOT))
            or active[0].get("packet_sha256") != artifact_ref(path)["sha256"]):
        raise RuntimeError("PROMPT14_EXACT_ACTIVE_PACKET_REQUIRED")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != prep.TASK_ID or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("candidate_operation") != OPERATION):
        raise RuntimeError("PROMPT14_PACKET_IDENTITY_DRIFT")
    return packet, prep.ATTEMPT


def validate_one_change(packet: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Fail before importing CUDA if anything besides the sleeve text moves."""
    from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep

    if (packet.get("inputs") != frozen
            or packet.get("source_session") != prep.SOURCE_SESSION
            or packet.get("anchor_frames") != list(prep.FRAMES)
            or packet.get("single_change") != {"role": prep.ROLE, "from": prep.OLD_PROMPT,
                                                "to": prep.NEW_PROMPT}
            or packet.get("budgets", {}).get("model_calls_max") != 3
            or packet.get("budgets", {}).get("session_attempts_max") != 1
            or packet.get("budgets", {}).get("clean_calls") != 0
            or packet.get("budgets", {}).get("render_calls") != 0
            or packet.get("config") != {"output_prob_thresh": 0.5, "instance_score_gate": 0.5}):
        raise RuntimeError("PROMPT14_ONE_CHANGE_OR_BUDGET_DRIFT")
    for row in frozen["anchors"]:
        prompts = row["candidate_prompts"]
        old = row["old_prompts"]
        if ({key for key in old if prompts.get(key) != old[key]} != {prep.ROLE}
                or prompts[prep.ROLE] != prep.NEW_PROMPT
                or prompts["yellow_cable"] != old["yellow_cable"]):
            raise RuntimeError("PROMPT14_ROLE_DRIFT")


def run_candidate(*, output_root: Path) -> dict[str, Any]:
    authorize_exact78_operation(OPERATION, ["--output-root", str(output_root)])
    from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
    from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep
    from chaoyang.governance import execute_exact78_vst_finger_sleeve_prompt14_candidate_v1 as executor

    packet, attempt = _packet()
    if output_root != attempt / "candidate" or output_root.exists() or output_root.is_symlink():
        raise RuntimeError("PROMPT14_FRESH_CANDIDATE_DIR_REQUIRED")
    frozen = prep.frozen_inputs()
    validate_one_change(packet, frozen)
    executor.verify_packet(packet)
    # All exact refs, run signature, CLI authority and writer are checked before CUDA import.
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
        checkpoint_path=Path(frozen["sam_checkpoint"]["path"]),
    )
    output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    calls = 0
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in frozen["anchors"]:
                frame, jpg_ref = row["frame"], row["refs"]["sam_input"]
                if artifact_ref(Path(jpg_ref["path"])) != jpg_ref:
                    raise RuntimeError(f"PROMPT14_JPG_DRIFT:{frame}")
                frame_dir = Path(jpg_ref["path"]).parent
                state = adapter.model.init_state(
                    resource_path=str(frame_dir), offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                try:
                    if calls >= 3:
                        raise RuntimeError("PROMPT14_THREE_MODEL_CALL_LIMIT")
                    calls += 1
                    _, prediction = adapter.model.add_prompt(
                        inference_state=state, frame_idx=0, text_str=prep.NEW_PROMPT,
                        output_prob_thresh=0.5,
                    )
                    image = cv2.imread(str(frame_dir / "000000.jpg"))
                    if image is None or image.shape != (960, 1280, 3):
                        raise RuntimeError(f"PROMPT14_JPG_DOMAIN:{frame}")
                    instances, scores = frontend.normalize(prediction, image.shape[:2])
                    chosen = instances[scores >= 0.5]
                    mask = (np.any(chosen, axis=0) if len(chosen)
                            else np.zeros(image.shape[:2], dtype=bool))
                    scores_out = [float(value) for value in scores[scores >= 0.5]]
                finally:
                    state.clear()
                mask_ref = frontend.save(output_root / "finger_sleeve" / f"{frame:06d}.png",
                                         mask.astype(np.uint8) * 255)
                rows.append({"frame": frame, "sam_input": jpg_ref,
                             "mask": mask_ref, "prompt": prep.NEW_PROMPT,
                             "instance_count": len(scores_out), "selected_scores": scores_out,
                             "pixels": int(mask.sum()), "review": "PENDING_HUMAN"})
    finally:
        adapter.predictor.shutdown()
    if calls != 3 or [row["frame"] for row in rows] != list(prep.FRAMES):
        raise RuntimeError("PROMPT14_CALL_OR_FRAME_COUNT_DRIFT")
    anchors = {"schema_version": "exact78-vst-finger-sleeve-prompt14-anchors-v1",
               "status": "MASK_CANDIDATE_UNREVIEWED", "task_id": prep.TASK_ID,
               "source_session": prep.SOURCE_SESSION, "rows": rows, "model_calls": calls,
               "model_evidence": model_evidence, "yellow_cable": "SEALED_ZERO_DETECTION_HOLD",
               "clean_called": False, "training_eligible": False}
    atomic_json(output_root / "FINGER_SLEEVE_ANCHORS.json", anchors)
    result = {"schema_version": "exact78-vst-finger-sleeve-prompt14-result-v1",
              "status": "MASK_CANDIDATE_UNREVIEWED", "task_id": prep.TASK_ID,
              "run_signature": packet["run_signature"],
              "anchors": artifact_ref(output_root / "FINGER_SLEEVE_ANCHORS.json"),
              "model_calls": 3, "clean_called": False, "visual_qualified": False,
              "training_eligible": False, "user_review": "PENDING",
              "yellow_cable": "UNRESOLVED_QUALITY_HOLD"}
    atomic_json(output_root / "RESULT.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    authorize_exact78_operation(OPERATION, args_in)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(args_in)
    result = run_candidate(output_root=args.output_root)
    print(json.dumps({"status": result["status"], "anchors": result["anchors"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
