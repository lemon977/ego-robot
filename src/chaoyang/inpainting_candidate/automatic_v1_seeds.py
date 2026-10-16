"""One candidate: current RGB proposals -> V1 multi-instance point prompts.

This module has no path, historical seed, session-ID, model-loading or writer
API. Projection associates existing RGB instances; it cannot supply pixels.
Automatic points and tracked masks are model proposals, never human annotations
or physical segmentation truth. V1 tracking/protection/Clean remain unchanged.
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from chaoyang.exact78_baseline import mask as parent_mask
from .automatic_masks import (
    AutomaticMaskContractError, RGBProposal, REFERENCE_SIZE, _owned_components,
    _projection, _rgb, _signature_values, frames_sha256,
)

SEED_AUTHORITY = "AUTO_CURRENT_RGB_MODEL_PROPOSAL_NOT_MANUAL_OBSERVATION"
MAX_POSITIVE_POINTS = 3
MAX_NEGATIVE_POINTS = 8
MAX_INSTANCES = 16
WEARER_ROLES = frozenset({"human", "glove"})
EQUIPMENT_ROLES = frozenset({"controller", "attachment", "cable"})


def _interior_points(mask: np.ndarray, maximum: int) -> list[list[float]]:
    """Deterministic, dispersed supported points, without a score threshold."""
    if not mask.any():
        return []
    depth = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1),
                                  cv2.DIST_L2, 5)[1:-1, 1:-1]
    yy, xx = np.indices(mask.shape)
    score = depth.copy()
    points = []
    nearest = np.full(mask.shape, np.inf, dtype=np.float64)
    for _ in range(min(maximum, int(mask.sum()))):
        row, column = np.unravel_index(int(np.argmax(score)), mask.shape)
        if not mask[row, column] or score[row, column] <= 0:
            break
        points.append([float(column), float(row)])
        nearest = np.minimum(nearest, (xx - column) ** 2 + (yy - row) ** 2)
        score = depth * np.sqrt(nearest)
        score[~mask] = 0
    return points


def _hits(masks: np.ndarray, hints: np.ndarray, radius: int) -> np.ndarray:
    count, height, width = masks.shape
    hits = np.zeros((2, count), bool)
    for frame in np.flatnonzero(masks.any(axis=(1, 2))):
        near = parent_mask.dilate(masks[frame], radius)
        for side in range(2):
            points = hints[side, frame]
            points = points[np.isfinite(points).all(axis=1)]
            points = points[(points[:, 0] >= 0) & (points[:, 0] < width)
                            & (points[:, 1] >= 0) & (points[:, 1] < height)]
            if len(points):
                pixels = np.rint(points).astype(int)
                pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
                pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
                hits[side, frame] = near[pixels[:, 1], pixels[:, 0]].any()
    return hits


def _unique_side(rates: Sequence[float]) -> int | None:
    best = max(rates)
    if best < parent_mask.TRACK_HIT_RATE_MIN or rates[0] == rates[1]:
        return None
    return int(rates[1] > rates[0])


def generate_v1_seed_bundle(
    rgb: np.ndarray, text_proposals: Sequence[RGBProposal],
    projection_hints: np.ndarray | None, *, raw_source_sha256: str,
    frame_map_sha256: str, model_sha256: str,
) -> dict:
    """Create V1-compatible positive/negative seeds with explicit UNKNOWNs.

    Caller supplies SAME_CAPTURE frozen visual-installation projection hints;
    a non-metric adopted visual prior is permitted for identity only. Missing
    pose support must be NaN. Association does not promote geometry. V1 hit,
    proximity and model gates are reused. Point counts are prompt budgets, not
    new quality thresholds. This API accepts only source-bound RGB TEXT tracks.
    """
    frames = _rgb(rgb)
    raw_sha, map_sha, model_sha = _signature_values(
        raw_source_sha256, frame_map_sha256, model_sha256)
    frame_sha = frames_sha256(frames)
    count, height, width = frames.shape[:3]
    if not np.isclose(width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]):
        raise AutomaticMaskContractError("V1_SEEDS_ISOTROPIC_640_480_DOMAIN_REQUIRED")
    if not isinstance(text_proposals, (list, tuple)):
        raise AutomaticMaskContractError("V1_SEEDS_RGB_TEXT_SEQUENCE_REQUIRED")
    hints = _projection(projection_hints, count)
    prepared, seen = [], set()
    for proposal in text_proposals:
        if (not isinstance(proposal, RGBProposal)
                or proposal.source != "SAM31_RGB_TEXT"
                or proposal.role not in WEARER_ROLES | EQUIPMENT_ROLES | {"object"}
                or isinstance(proposal.track_id, bool)
                or not isinstance(proposal.track_id, (int, np.integer))
                or proposal.track_id < 0):
            raise AutomaticMaskContractError("V1_SEEDS_ONLY_RGB_TEXT_PROPOSALS_ALLOWED")
        key = (proposal.role, int(proposal.track_id))
        if key in seen:
            raise AutomaticMaskContractError("V1_SEEDS_DUPLICATE_ROLE_TRACK_ID")
        seen.add(key)
        if (proposal.frames_sha256 != frame_sha
                or proposal.raw_source_sha256 != raw_sha
                or proposal.frame_map_sha256 != map_sha
                or proposal.model_sha256 != model_sha):
            raise AutomaticMaskContractError("V1_SEEDS_CURRENT_RGB_SOURCE_SIGNATURE_MISMATCH")
        mask = np.asarray(proposal.masks)
        if mask.dtype != np.bool_ or mask.shape != (count, height, width):
            raise AutomaticMaskContractError("V1_SEEDS_NATIVE_RGB_MASK_DOMAIN_REQUIRED")
        unsupported = proposal.unsupported_frames
        if unsupported is None:
            unsupported = np.zeros(count, bool)
        unsupported = np.asarray(unsupported)
        if unsupported.dtype != np.bool_ or unsupported.shape != (count,):
            raise AutomaticMaskContractError("V1_SEEDS_UNSUPPORTED_FRAME_DOMAIN")
        mask = mask.copy()
        mask[unsupported] = False
        prepared.append({"proposal": proposal, "mask": mask, "side": None,
                         "selected": np.zeros_like(mask), "rates": [0.0, 0.0],
                         "reason": "OBJECT_NEGATIVE_PROPOSAL_ONLY" if proposal.role == "object"
                         else "IDENTITY_UNKNOWN"})
    prepared.sort(key=lambda row: (row["proposal"].role, int(row["proposal"].track_id)))
    side_support = np.zeros((2, count, height, width), bool)
    if hints is not None:
        for row in prepared:
            if row["proposal"].role not in WEARER_ROLES:
                continue
            hits = _hits(row["mask"], hints, parent_mask.HIT_RADIUS_PX)
            denominator = max(1, int(row["mask"].any(axis=(1, 2)).sum()))
            row["rates"] = [float(value / denominator) for value in hits.sum(axis=1)]
            side = _unique_side(row["rates"])
            if side is not None:
                row.update(side=side, selected=row["mask"],
                           reason="UNIQUE_MAX_SAME_CAPTURE_PROJECTION_HIT_RATE")
                side_support[side] |= row["selected"]
        radius = parent_mask.CONTACT_PX * width / REFERENCE_SIZE[0]
        for row in prepared:
            if row["proposal"].role not in EQUIPMENT_ROLES:
                continue
            alternatives, details = [], []
            for side in range(2):
                selected, detail = _owned_components(row["mask"], side_support[side], radius)
                alternatives.append(selected)
                details.append(detail)
            row["rates"] = [float(detail["wearer_hit_rate"]) for detail in details]
            side = _unique_side(row["rates"])
            if side is not None and alternatives[side].any():
                row.update(side=side, selected=alternatives[side],
                           reason="UNIQUE_MAX_CURRENT_RGB_WEARER_COMPONENT_ASSOCIATION")
                side_support[side] |= row["selected"]
    selected_rows = [row for row in prepared if row["side"] is not None
                     and row["selected"].any()]
    if len(selected_rows) > MAX_INSTANCES:
        raise AutomaticMaskContractError("V1_SEEDS_INSTANCE_CAPACITY_EXCEEDED")
    objects, diagnostics = [], []
    for row in prepared:
        proposal = row["proposal"]
        identity = {"role": proposal.role, "track_id": int(proposal.track_id),
                    "side": "UNKNOWN" if row["side"] is None else ("left", "right")[row["side"]],
                    "association": row["reason"], "side_hit_rates": row["rates"],
                    "present_native_frames": np.flatnonzero(row["mask"].any(axis=(1, 2))).tolist(),
                    "selected_pixels_per_native_frame": row["selected"].sum(axis=(1, 2)).tolist()}
        diagnostics.append(identity)
        if row["side"] is None or not row["selected"].any():
            continue
        anchor = int(np.flatnonzero(row["selected"].any(axis=(1, 2)))[0])
        target = row["selected"][anchor]
        positive = _interior_points(target, MAX_POSITIVE_POINTS)
        negative, negative_sources = [], []
        # An already RGB-supported same-side wearable part is not a negative.
        # Different instances/objects can contribute only outside that union.
        own = side_support[row["side"], anchor]
        for other in prepared:
            if other is row or other["side"] == row["side"]:
                continue
            region = other["mask"][anchor] & ~target & ~own
            for point in _interior_points(region, 1):
                if point in negative:
                    continue
                negative.append(point)
                negative_sources.append({"role": other["proposal"].role,
                                         "track_id": int(other["proposal"].track_id),
                                         "source": "OTHER_CURRENT_RGB_PROPOSAL_OUTSIDE_OWN_SUPPORT"})
                if len(negative) == MAX_NEGATIVE_POINTS:
                    break
            if len(negative) == MAX_NEGATIVE_POINTS:
                break
        objects.append({"obj_id": len(objects) + 1, "role": proposal.role,
                        "side": identity["side"], "frame_idx": anchor,
                        "points_px": positive + negative,
                        "point_labels": [1] * len(positive) + [0] * len(negative),
                        "seed_authority": SEED_AUTHORITY,
                        "proposal_role": proposal.role, "proposal_track_id": int(proposal.track_id),
                        "negative_proposal_sources": negative_sources,
                        "physical_segmentation_ground_truth": False,
                        "human_visible_confirmed": False})
    return {
        "schema": "inpainting-automatic-v1-rgb-seed-bundle-v1",
        "status": "READY_MODEL_CANDIDATE" if objects else "IDENTITY_UNKNOWN_NO_SEEDS",
        "objects": objects, "anchor_frame": min((obj["frame_idx"] for obj in objects), default=0),
        "seed_authority": SEED_AUTHORITY,
        "input_bindings": {"raw_source_sha256": raw_sha, "frame_map_sha256": map_sha,
                           "model_sha256": model_sha, "frames_sha256": frame_sha},
        "instance_association": diagnostics,
        "projection_is_pixel_evidence": False, "manual_seed_reads": 0,
        "projection_support_basis": "SAME_CAPTURE_FROZEN_VISUAL_PRIOR_IDENTITY_ONLY",
        "projection_qualified": False, "metric_calibration_claim": False,
        "historical_mask_reads": 0, "producer_session_id_branches": 0,
        "native_frame_count": count, "rgb_domain": [width, height],
        "prompt_budget": {"positive_per_instance": MAX_POSITIVE_POINTS,
                          "negative_per_instance": MAX_NEGATIVE_POINTS},
        "identity_hit_rate_min_unchanged": parent_mask.TRACK_HIT_RATE_MIN,
        "human_review": "PENDING", "physical_segmentation_ground_truth": False,
        "training_eligible": False, "control_ground_truth": False,
    }


def build_v1_masks_from_automatic_seeds(sam: Any, rgb: np.ndarray,
                                        session: Any, seed_bundle: dict) -> dict:
    """Call unchanged V1, then correct the seed-authority wording only."""
    if (not isinstance(seed_bundle, dict)
            or seed_bundle.get("seed_authority") != SEED_AUTHORITY
            or seed_bundle.get("status") != "READY_MODEL_CANDIDATE"
            or seed_bundle.get("input_bindings", {}).get("frames_sha256") != frames_sha256(_rgb(rgb))):
        raise AutomaticMaskContractError("V1_SEEDS_READY_BOUND_AUTOMATIC_BUNDLE_REQUIRED")
    from chaoyang.pick_place_1008.visual import build_visual_masks
    masks = build_visual_masks(sam, rgb, session, visible_seeds=seed_bundle)
    masks["metadata"]["recipe"] = "AUTO_RGB_INSTANCE_SEEDS_UNCHANGED_V1_POINT_TRACKING"
    masks["metadata"]["automatic_seed_authority"] = SEED_AUTHORITY
    masks["metadata"]["projection_support_basis"] = seed_bundle["projection_support_basis"]
    masks["metadata"]["projection_qualified"] = False
    masks["metadata"]["automatic_seed_evidence"] = seed_bundle
    masks["metadata"]["point_tracking"]["seed_authority"] = SEED_AUTHORITY
    for role in ("human", "equipment"):
        masks["metadata"][role]["provider"] = "AUTOMATIC_RGB_MODEL_POINT_SEGMENTATION_PROPOSAL"
    return masks


def execute_mask_experiment(packet: dict) -> int:
    """One registered real-SAM trial; no history seed/POINT/Mask producer read.

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
    from PIL import Image, ImageDraw, ImageFont
    from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, now_iso
    from .contracts import canonical_sha, verify_ref, verify_sealed_stage
    from .resources import CANDIDATE_CARD_LOCK_ROOT
    from .model_stages import configure_project_caches, load_frozen_models
    from .model_worker import CountedSam, ModelCallBudget, mask_review
    from .source_frames import write_review_video
    from chaoyang.pick_place_1008.source import prepare_session, load_rgb_frames
    from chaoyang.pick_place_1008.visual import effective_clean_masks, SETTINGS, _gate_diagnostic_geometry

    root = Path(packet["output_dir"])
    definition = importlib.util.spec_from_file_location("_auto_seed_worker_gate",
        REPO_ROOT / "scripts/maintenance/inpainting_auto_v1_seed_one_hour_v1.py")
    gate = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(gate)
    helpers, infra = gate._model_helpers(), gate._infra()

    def authority():
        _, state, index = infra._snapshot()
        row, _ = helpers.own_row(packet, state, index)
        if row["status"] != "RUNNING" or datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
            raise ValueError("AUTO_SEED_RUNNING_FENCE_OR_DEADLINE")
        lease = json.loads((CANDIDATE_CARD_LOCK_ROOT / (packet["gpu_mapping"]["uuid"] + ".json")).read_text())
        required = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "fencing_token": packet["writer"]["fencing_token"], "pid": packet["writer"]["pid"],
            "proc_start_ticks": packet["writer"]["proc_start_ticks"],
            "uuid": packet["gpu_mapping"]["uuid"], "physical_index": packet["gpu_mapping"]["physical_index"],
            "worker_logical_index": 0}
        if any(lease.get(k) != v for k, v in required.items()) or os.environ.get("CUDA_VISIBLE_DEVICES") != required["uuid"]:
            raise ValueError("AUTO_SEED_CARD_LEASE_DRIFT")
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
            raise ValueError("AUTO_SEED_FULL_NATIVE_SOURCE_REQUIRED")
        for pin in report["actual_inputs"].values():
            verify_ref(pin)
        verify_ref(row["camera_config"])
        config = json.loads(Path(row["camera_config"]["path"]).read_text())
        session = prepare_session(Path(row["raw_session_dir"]), out / "FreshRawPose",
            config, expected_frames=len(rgb))
        fresh_rgb = load_rgb_frames(session)
        if not np.array_equal(fresh_rgb, rgb) or not np.array_equal(session.qpc_ns, [r["qpc_declared_ns"] for r in mapping]):
            raise ValueError("AUTO_SEED_FRESH_RAW_RGB_OR_QPC_DRIFT")
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
            raise ValueError("AUTO_SEED_SIGNED_TEXT_ARRAY_BINDING_REQUIRED")
        proposals = []
        with np.load(row["text_arrays"]["path"], allow_pickle=False) as archive:
            for entry in proposal_meta["proposals"]:
                shape = tuple(entry["native_shape"])
                if shape != rgb.shape[:3] or entry.get("source") != "SAM31_RGB_TEXT":
                    raise ValueError("AUTO_SEED_ONLY_SOURCE_BOUND_TEXT_REQUIRED")
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
            raise ValueError("AUTO_SEED_TEXT_CURRENT_RGB_SOURCE_OR_MODEL_DRIFT")
        bundle = generate_v1_seed_bundle(rgb, proposals, hints,
            raw_source_sha256=text["raw_source_sha256"], frame_map_sha256=text["frame_map_sha256"],
            model_sha256=text["model_sha256"])
        atomic_json(mask_dir / "AUTOMATIC_SEEDS.json", bundle)
        del proposals
        before_calls = budget.total
        authority()
        masks = build_v1_masks_from_automatic_seeds(sam, rgb, session, bundle)
        authority()
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
        actual_inputs = [row[name] for name in ("source_receipt", "text_receipt", "text_proposals", "text_arrays", "camera_config")]
        actual_inputs.extend(report["actual_inputs"].values())
        actual_inputs.append(artifact_ref(out / "FreshRawPose/SESSION_INPUT.json"))
        receipt = {"schema": "inpainting-candidate-component-v1", "stage": "Mask",
            "status": "SEALED_COMPLETE", "component_signature": row["mask_signature"],
            "native_input_frames": len(rgb), "native_output_frames": len(rgb),
            "producer_manual_seed_reads": 0, "producer_historical_mask_reads": 0,
            "actual_inputs": actual_inputs, "outputs": outputs, "authority": authority(),
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
            "automatic_instances": len(bundle["objects"]), "visual_quality": "NOT_REVIEWED"}
        atomic_json(out / "SESSION_RESULT.json", status)
        results.append(status)
        atomic_json(root / "MODEL_PROGRESS.json", {"sessions": results, "model_call_accounting": budget.receipt()})
        print(json.dumps({"session_key": row["session_key"], "completed": True,
            "instances": len(bundle["objects"]), "api_calls": budget.total-before_calls}), flush=True)
        del rgb, masks, arrays, overlay, hints, geometry, session
    result = {"schema": "inpainting-auto-v1-seed-mask-model-result-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "experiment_scope": packet["experiment_scope"],
        "planned_sessions": 2, "completed_mask_sessions": len(results), "sessions": results,
        "model_call_accounting": budget.receipt(), "candidate_adopted": False,
        "visual_quality": "NOT_REVIEWED", "training_steps": 0, "files_deleted": 0,
        "producer_manual_seed_reads": 0, "producer_historical_mask_reads": 0,
        "completed_at": now_iso()}
    atomic_json(root / "MODEL_RESULT.json", result)
    return 0 if len(results) == 2 else 2
