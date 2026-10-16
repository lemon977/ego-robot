"""Single-change V1 automatic-seed candidate: generic object negatives are UNKNOWN.

Retains frozen Source/TEXT, first-nonempty anchors, numerical gates, and V1
POINT/CUP tracking. Only the pure prompt policy changes; no baseline is adopted.
"""
from __future__ import annotations

import numpy as np

from chaoyang.exact78_baseline import mask as parent_mask
from .automatic_masks import RGBProposal, frames_sha256
from .automatic_v1_seeds import (
    SEED_AUTHORITY, build_v1_masks_from_automatic_seeds, generate_v1_seed_bundle,
)
from .neutral_object_prompts import neutralize_generic_object_negatives

MODEL_RESULT_SCHEMA = "inpainting-neutral-object-mask-model-result-v1"
EXPERIMENT_SCOPE = "UNKNOWN_GENERIC_OBJECT_NEGATIVE_ABLATION_V1"


def execute_neutral_object_experiment(packet: dict) -> int:
    """One registered generic-object-negative ablation; no historical prompts.

    This orchestration is candidate-only. Source-bound sealed TEXT proposals
    are reused; raw hardware poses are freshly parsed with the frozen V1
    visual installation prior for identity hints only, not segmentation truth.
    """
    import importlib.util
    import json
    import os
    from pathlib import Path
    import time
    from datetime import datetime, timezone
    from PIL import Image, ImageDraw
    from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, now_iso
    from .contracts import canonical_sha, verify_ref, verify_sealed_stage
    from .resources import CANDIDATE_CARD_LOCK_ROOT
    from .model_stages import configure_project_caches, load_frozen_models
    from .model_worker import CountedSam, ModelCallBudget, mask_review
    from .source_frames import write_review_video
    from chaoyang.pick_place_1008.source import prepare_session, load_rgb_frames
    from chaoyang.pick_place_1008.visual import effective_clean_masks, SETTINGS, _gate_diagnostic_geometry

    if packet.get("experiment_scope") != EXPERIMENT_SCOPE:
        raise ValueError("NEUTRAL_OBJECT_EXPLICIT_SCOPE_REQUIRED")
    root = Path(packet["output_dir"])
    definition = importlib.util.spec_from_file_location("_neutral_object_worker_gate",
        REPO_ROOT / "scripts/maintenance/inpainting_neutral_object_v1.py")
    gate = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(gate)
    helpers, infra = gate._model_helpers(), gate._infra()

    def authority():
        _, state, index = infra._snapshot()
        row, _ = helpers.own_row(packet, state, index)
        if row["status"] != "RUNNING" or datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
            raise ValueError("NEUTRAL_OBJECT_RUNNING_FENCE_OR_DEADLINE")
        lease = json.loads((CANDIDATE_CARD_LOCK_ROOT / (packet["gpu_mapping"]["uuid"] + ".json")).read_text())
        required = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "fencing_token": packet["writer"]["fencing_token"], "pid": packet["writer"]["pid"],
            "proc_start_ticks": packet["writer"]["proc_start_ticks"],
            "uuid": packet["gpu_mapping"]["uuid"], "physical_index": packet["gpu_mapping"]["physical_index"],
            "worker_logical_index": 0}
        if any(lease.get(k) != v for k, v in required.items()) or os.environ.get("CUDA_VISIBLE_DEVICES") != required["uuid"]:
            raise ValueError("NEUTRAL_OBJECT_CARD_LEASE_DRIFT")
        return {**required, "execution_allowed": True, "algorithm_execution_allowed": True,
            "gpu_uuid": required["uuid"]}

    configure_project_caches(root)
    authority()
    budget = ModelCallBudget(packet["budgets"]["model_calls_max"], packet["budgets"]["model_calls_unit"])
    models = load_frozen_models(REPO_ROOT, root, authorization=authority,
        frozen_inputs=[*packet["runtime_code"], *packet["model_weights"]], need_sam=True, need_clean=False)
    atomic_json(root / "MODEL_LOAD.json", models["load_receipt"])
    sam = CountedSam(models["sam"], budget)
    weight_ref = packet["model_weights"][0]
    results = []
    for row in packet["sessions"]:
        authority()
        start = time.monotonic()
        out = root / "sessions" / row["session_key"]
        out.mkdir(parents=True, exist_ok=False)
        source_path = Path(row["source_receipt"]["path"])
        verify_ref(row["source_receipt"])
        source = json.loads(source_path.read_text())
        verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
        source_outputs = {Path(pin["path"]).name: pin for pin in source["outputs"]}
        report = json.loads(Path(source_outputs["SOURCE_REPORT.json"]["path"]).read_text())
        mapping = json.loads(Path(source_outputs["FRAME_MAP.json"]["path"]).read_text())
        with np.load(source_outputs["RAW_RGB.npz"]["path"], allow_pickle=False) as archive:
            rgb = archive["rgb"]
        if len(rgb) != row["native_frames"] or len(mapping) != len(rgb):
            raise ValueError("NEUTRAL_OBJECT_FULL_NATIVE_SOURCE_REQUIRED")
        for pin in report["actual_inputs"].values():
            verify_ref(pin)
        verify_ref(row["camera_config"])
        config = json.loads(Path(row["camera_config"]["path"]).read_text())
        session = prepare_session(Path(row["raw_session_dir"]), out / "FreshRawPose",
            config, expected_frames=len(rgb))
        fresh_rgb = load_rgb_frames(session)
        if not np.array_equal(fresh_rgb, rgb) or not np.array_equal(session.qpc_ns, [r["qpc_declared_ns"] for r in mapping]):
            raise ValueError("NEUTRAL_OBJECT_FRESH_RAW_RGB_OR_QPC_DRIFT")
        del fresh_rgb
        geometry = parent_mask.pico_geometry(session, rgb.shape[2] / session.width)
        geometry, invalid = _gate_diagnostic_geometry(geometry, session)
        # These are search/identity hints only; no hull or projected pixels
        # enter a foreground mask or any positive/negative prompt.
        hints = np.concatenate([geometry["hands"], geometry["forearm"], geometry["tracker"]], axis=2)
        mask_dir = out / "Mask"
        mask_dir.mkdir()
        np.savez_compressed(mask_dir / "IDENTITY_HINTS.npz", xy=hints)
        verify_ref(row["text_receipt"])
        text = json.loads(Path(row["text_receipt"]["path"]).read_text())
        for name in ("text_proposals", "text_arrays"):
            verify_ref(row[name])
        proposal_meta = json.loads(Path(row["text_proposals"]["path"]).read_text())
        if proposal_meta.get("stage") != "TEXT" or proposal_meta.get("arrays") != row["text_arrays"]:
            raise ValueError("NEUTRAL_OBJECT_SIGNED_TEXT_ARRAY_BINDING_REQUIRED")
        proposals = []
        with np.load(row["text_arrays"]["path"], allow_pickle=False) as archive:
            for entry in proposal_meta["proposals"]:
                shape = tuple(entry["native_shape"])
                if shape != rgb.shape[:3] or entry.get("source") != "SAM31_RGB_TEXT":
                    raise ValueError("NEUTRAL_OBJECT_ONLY_SOURCE_BOUND_TEXT_REQUIRED")
                masks = np.unpackbits(archive[entry["array_key"]], axis=1,
                    count=shape[1]*shape[2]).reshape(shape).astype(bool)
                unsupported_key = entry.get("unsupported_frames_array_key")
                unsupported = archive[unsupported_key].astype(bool) if unsupported_key else np.zeros(len(rgb), bool)
                proposals.append(RGBProposal(role=entry["role"], track_id=entry["track_id"], masks=masks,
                    source=entry["source"], unsupported_frames=unsupported,
                    frames_sha256=entry["frames_sha256"], raw_source_sha256=entry["raw_source_sha256"],
                    frame_map_sha256=entry["frame_map_sha256"], model_sha256=entry["model_sha256"]))
        if (text["frames_sha256"] != frames_sha256(rgb) or text["frame_map_sha256"] != canonical_sha(mapping)
                or text["model_sha256"] != weight_ref["sha256"]
                or text["raw_source_sha256"] != report["actual_inputs"]["video"]["sha256"]):
            raise ValueError("NEUTRAL_OBJECT_TEXT_CURRENT_RGB_SOURCE_OR_MODEL_DRIFT")
        bundle = generate_v1_seed_bundle(rgb, proposals, hints,
            raw_source_sha256=text["raw_source_sha256"], frame_map_sha256=text["frame_map_sha256"],
            model_sha256=text["model_sha256"])
        atomic_json(mask_dir / "AUTOMATIC_SEEDS_BEFORE_NEUTRALIZATION.json", bundle)
        bundle = neutralize_generic_object_negatives(bundle)
        policy = bundle["generic_object_negative_policy"]
        atomic_json(mask_dir / "AUTOMATIC_SEEDS.json", bundle)
        atomic_json(mask_dir / "GENERIC_OBJECT_NEGATIVE_POLICY.json", policy)
        del proposals
        before_calls = budget.total
        authority()
        masks = build_v1_masks_from_automatic_seeds(sam, rgb, session, bundle)
        authority()
        masks["metadata"]["generic_object_negative_policy"] = policy
        arrays = {name: value for name, value in masks.items()
            if isinstance(value, np.ndarray) and value.dtype == np.bool_ and value.shape == rgb.shape[:3]}
        arrays["effective_removal"], preservation = effective_clean_masks(
            arrays["removal"], arrays["protected_visible"], SETTINGS)
        np.savez_compressed(mask_dir / "MASK_ARRAYS.npz", **arrays)
        atomic_json(mask_dir / "MASK_METADATA.json", masks["metadata"])
        states = [{"native_frame_index": i, "output_frame": i,
            "qpc_declared_ns": mapping[i]["qpc_declared_ns"], "mask_status": "MODEL_PREDICTION",
            "seed_source": SEED_AUTHORITY, "manual_seed_reads": 0,
            "timebase_status": source["timebase_status"], "projection_qualified": False,
            "human_review": "PENDING", "hidden_regions": "UNKNOWN", "training_eligible": False,
            "control_ground_truth": False} for i in range(len(rgb))]
        atomic_json(mask_dir / "FRAME_STATUS.json", states)
        atomic_json(mask_dir / "FRAME_MAP.json", mapping)
        overlay = mask_review(rgb, arrays)
        review = write_review_video(out / "raw_mask_review.mp4", overlay, mapping)
        anchors = row["review_anchors"]
        for index in anchors:
            Image.fromarray(overlay[index]).save(mask_dir / f"anchor_{index:04d}.png")
        # Show the generated prompts separately, without claiming they were
        # manually annotated. Points remain attributable to RGB TEXT tracks.
        for index in sorted(set(obj["frame_idx"] for obj in bundle["objects"])):
            canvas = Image.fromarray(rgb[index]).copy()
            draw = ImageDraw.Draw(canvas)
            for obj in bundle["objects"]:
                if obj["frame_idx"] != index:
                    continue
                for (x,y), label in zip(obj["points_px"], obj["point_labels"]):
                    draw.ellipse((x-3,y-3,x+3,y+3), outline=(20,255,20) if label else (255,20,20), width=2)
            canvas.save(mask_dir / f"automatic_prompts_{index:04d}.png")
        outputs = [artifact_ref(path) for path in sorted(mask_dir.iterdir()) if path.is_file()]
        provenance_refs = [row[name] for name in
            ("source_receipt", "text_receipt", "text_proposals", "text_arrays", "camera_config")]
        provenance_refs.extend(report["actual_inputs"].values())
        provenance_refs.append(artifact_ref(out / "FreshRawPose/SESSION_INPUT.json"))
        actual_inputs = {"raw_source": dict(report["actual_inputs"]["video"]),
            "frames_sha256": frames_sha256(rgb), "frame_map_sha256": canonical_sha(mapping)}
        receipt = {"schema": "inpainting-candidate-component-v1", "stage": "Mask",
            "status": "SEALED_COMPLETE", "component_signature": row["mask_signature"],
            "native_input_frames": len(rgb), "native_output_frames": len(rgb),
            "producer_manual_seed_reads": 0, "producer_historical_mask_reads": 0,
            "actual_inputs": actual_inputs, "provenance_refs": provenance_refs,
            "outputs": outputs, "authority": authority(),
            "experiment_scope": EXPERIMENT_SCOPE, "generic_object_negative_policy": policy,
            "automatic_seed_authority": SEED_AUTHORITY,
            "model_api_calls_attempted": budget.total-before_calls,
            "projection_qualified": False, "metric_calibration_claim": False,
            "invalid_hint_frame_sides": invalid, "visual_quality": "NOT_REVIEWED",
            "human_review": "PENDING", "hidden_regions": "UNKNOWN", "training_eligible": False,
            "control_ground_truth": False, "timings": {"whole_stage_wall_seconds": time.monotonic()-start},
            "completed_at": now_iso()}
        atomic_json(mask_dir / "RECEIPT.json", receipt)
        status = {"session_key": row["session_key"], "session_id": session.session_id,
            "source": row["source_receipt"], "native_frames": len(rgb), "execution": "COMPLETE_MASK_ONLY",
            "manual_seed_reads": 0, "mask": artifact_ref(mask_dir / "RECEIPT.json"), "review": review,
            "positive_points": sum(obj["point_labels"].count(1) for obj in bundle["objects"]),
            "negative_points": sum(obj["point_labels"].count(0) for obj in bundle["objects"]),
            "automatic_instances": len(bundle["objects"]), "visual_quality": "NOT_REVIEWED",
            "generic_object_negative_policy": policy}
        atomic_json(out / "SESSION_RESULT.json", status)
        results.append(status)
        atomic_json(root / "MODEL_PROGRESS.json", {"sessions": results, "model_call_accounting": budget.receipt()})
        print(json.dumps({"session_key": row["session_key"], "completed": True,
            "instances": len(bundle["objects"]), "api_calls": budget.total-before_calls}), flush=True)
        del rgb, masks, arrays, overlay, hints, geometry, session
    result = {"schema": MODEL_RESULT_SCHEMA, "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "experiment_scope": EXPERIMENT_SCOPE,
        "planned_sessions": 2, "completed_mask_sessions": len(results), "sessions": results,
        "model_call_accounting": budget.receipt(), "candidate_adopted": False,
        "visual_quality": "NOT_REVIEWED", "training_steps": 0, "files_deleted": 0,
        "producer_manual_seed_reads": 0, "producer_historical_mask_reads": 0,
        "completed_at": now_iso()}
    atomic_json(root / "MODEL_RESULT.json", result)
    return 0 if len(results) == 2 else 2
