"""Single controller replay with read-only telemetry. No POINT or Clean."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, now_iso
from chaoyang.inpainting_candidate.contracts import verify_ref, verify_sealed_stage


def execute_controller_replay(packet):
    started = time.monotonic()
    root = Path(packet["output_dir"])
    row = packet["sessions"][0]
    source_path = Path(row["source_receipt"]["path"])
    verify_ref(row["source_receipt"])
    source = json.loads(source_path.read_text())
    verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
    mapping = json.loads((source_path.parent / "FRAME_MAP.json").read_text())
    with np.load(source_path.parent / "RAW_RGB.npz", allow_pickle=False) as a:
        rgb = a["rgb"].copy()
    if rgb.shape != (238, 480, 640, 3) or rgb.dtype != np.uint8 or len(mapping) != 238:
        raise ValueError("OBSERVER_FULL_NATIVE_RGB_REQUIRED")
    for i, m in enumerate(mapping):
        if m["native_frame_index"] != i or m["pixel_sha256"] != hashlib.sha256(rgb[i].tobytes()).hexdigest():
            raise ValueError("OBSERVER_NATIVE_PIXEL_MAPPING_DRIFT")
    out = root / "sessions/001/Observer"
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "MODEL_PROGRESS.json", {"stage": "LOADING_SAM", "actual_input": row["source_receipt"], "at": now_iso()})
    from chaoyang.exact78_baseline.mask import Sam3Video
    from chaoyang.inpainting_candidate.controller_observer import Sam31ControllerObserver
    loaded_at = time.monotonic()
    sam = Sam3Video(REPO_ROOT)
    atomic_json(root / "MODEL_LOAD.json", {"model_class": type(sam.model).__name__,
        "adapter_evidence": sam.evidence, "model_weights": packet["model_weights"],
        "load_wall_seconds": time.monotonic() - loaded_at})
    call_started = time.monotonic()
    calls = 0
    bindings = {"source_receipt_sha256": row["source_receipt"]["sha256"],
                "run_signature": packet["run_signature"], "sample_masks_are_evaluation_only": True}
    try:
        with Sam31ControllerObserver(sam.model, native_frames=238,
                sample_native_frames=(0, 103, 104, 119, 132), bindings=bindings) as obs:
            atomic_json(root / "MODEL_PROGRESS.json", {"stage": "ONE_CONTROLLER_TRACK_TEXT", "calls_attempted": 1, "at": now_iso()})
            calls += 1
            tracks = sam.track_text(list(rgb), "VR controller", 0)
        aborted = list(sam.aborted)
        summary = obs.summary()
    finally:
        sam.close()
    if datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
        raise RuntimeError("OBSERVER_FINITE_DEADLINE")
    atomic_json(out / "EVENTS.json", obs.records)
    np.savez_compressed(out / "QUERY_MASKS.npz", **obs.masks)
    # No old generic masks/probes/manual points are read by this producer.
    packed = {f"obj_{obj}_frame_{frame}": mask for obj, frames in tracks.items() for frame, mask in frames.items()}
    np.savez_compressed(out / "FINAL_TRACKS.npz", **packed)
    equivalence = "MATCH_PRIOR_EMPTY_CONTROLLER_OUTPUT" if not tracks and not aborted else "FINAL_OUTPUT_DIFFERS_OR_ABORTED"
    summary.update({"native_frames": 238, "controller_tracks": len(tracks), "aborted": aborted,
        "track_text_api_calls": calls, "controller_call_wall_seconds": time.monotonic() - call_started,
        "all_wall_seconds": time.monotonic() - started, "output_equivalence": equivalence,
        "whole_video_visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
        "physical_late_entry_tested": False, "threshold_changes": False})
    atomic_json(out / "SUMMARY.json", summary)
    files = [out / name for name in ("EVENTS.json", "QUERY_MASKS.npz", "FINAL_TRACKS.npz", "SUMMARY.json")]
    receipt = {"schema": "inpainting-candidate-component-v1", "stage": "ControllerObserver",
        "status": "SEALED_COMPLETE", "component_signature": hashlib.sha256((packet["run_signature"] + "controller-observer").encode()).hexdigest(),
        "session_id": "pick_place_1008_001", "native_input_frames": 238, "native_output_frames": 238,
        "native_frames": 238, "actual_inputs": [row["source_receipt"], *source["outputs"]],
        "outputs": [artifact_ref(p) for p in files], "threshold_changes": False,
        "manual_seed_reads": 0, "clean_calls": 0, "robot_calls": 0, "point_calls": 0,
        "hidden_regions": "UNKNOWN", "human_review": "PENDING", "control_ground_truth": False,
        "timebase_status": source["timebase_status"], "completed_at": now_iso()}
    atomic_json(out / "RECEIPT.json", receipt)
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > packet["budgets"]["output_bytes_estimate"]:
        raise RuntimeError("OBSERVER_OUTPUT_CAP_EXCEEDED")
    result = {"schema": "inpainting-controller-observer-model-result-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "experiment_scope": packet["experiment_scope"],
        "planned_sessions": 1, "completed_replay_sessions": 1, "candidate_adopted": False,
        "visual_quality": "NOT_REVIEWED", "model_call_accounting": {
            "unit": "SAM_ADAPTER_INFERENCE_API_INVOCATION", "attempted_api_calls": calls, "maximum": 1,
            "counts_constructor_or_init_state": False, "counts_neural_forward_passes": False},
        "sessions": [{"session_key": "001", "native_frames": 238, "source": row["source_receipt"],
            "observer": artifact_ref(out / "RECEIPT.json"), "manual_seed_reads": 0,
            "execution": "COMPLETE_CONTROLLER_REPLAY_ONLY", "output_equivalence": equivalence}],
        "producer_historical_mask_reads": 0, "producer_manual_seed_reads": 0, "files_deleted": 0,
        "point_calls": 0, "clean_calls": 0, "robot_calls": 0, "training_steps": 0, "completed_at": now_iso()}
    atomic_json(root / "MODEL_RESULT.json", result)
    print(json.dumps({"status": "CONTROLLER_REPLAY_COMPLETE", "tracks": len(tracks), "output_equivalence": equivalence,
                      "events": len(obs.records), "summary": summary}, ensure_ascii=False), flush=True)
    return 0
