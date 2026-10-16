"""Bounded fresh RGB SPREAD_MAX_GAP_THREE repair over original V1.

Reuses the existing proposal association, component identity veto, point tracker
and POINT intersection merge. Historical Mask/Clean/action arrays are not inputs.
"""
from __future__ import annotations


def associate_original_objects(bundle, original_objects):
    """Retain only one original glove identity on an accepted same-capture side."""
    assignments, unknown = [], []
    for value in bundle["instance_association"]:
        if value["role"] not in {"human", "glove"}:
            continue
        originals = [o for o in original_objects if o["role"] == "glove" and o["side"] == value["side"]]
        if value["side"] == "UNKNOWN" or len(originals) != 1:
            unknown.append({**value, "decision": "ORIGINAL_IDENTITY_UNKNOWN_NO_REPAIR"})
            continue
        assignments.append({"role": value["role"], "track_id": value["track_id"], "side": value["side"],
            "obj_id": originals[0]["obj_id"], "original_role": originals[0]["role"],
            "association": value["association"], "identity_unique": True,
            "original_identity_basis": "ONE_ORIGINAL_GLOVE_OBJECT_PER_ACCEPTED_SIDE",
            "side_hit_rates": value["side_hit_rates"]})
    return assignments, unknown


def repair_current_masks(*, sam, rgb, session, original_masks, original_objects,
                         raw_source_sha256, frame_map_sha256, model_sha256,
                         glove_text_prompt="a gloved hand", tracker=None,
                         repair_rounds=1, return_round_masks=False, round_snapshots=None):
    from copy import deepcopy
    import numpy as np
    from chaoyang.exact78_baseline import mask as parent_mask
    from chaoyang.pick_place_1008 import visual
    from .automatic_masks import collect_sam_rgb_proposals, frames_sha256
    from .automatic_v1_seeds import generate_v1_seed_bundle
    from .v1_mask_repair import (SPREAD_MAX_GAP_THREE, REPAIR_AUTHORITY, make_repair_prompts,
                                merge_repair_support, qualify_identity_support)
    if type(repair_rounds) is not int or repair_rounds not in (1, 2, 3):
        raise ValueError("AUTOMATION_REPAIR_ROUNDS_MUST_BE_ONE_TO_THREE")
    if round_snapshots is not None and (not isinstance(round_snapshots, dict) or round_snapshots):
        raise ValueError("AUTOMATION_EMPTY_ROUND_SNAPSHOT_MAPPING_REQUIRED")
    if type(return_round_masks) is not bool:
        raise ValueError("AUTOMATION_RETURN_ROUND_MASKS_BOOL_REQUIRED")
    if tracker is None:
        from chaoyang.pick_place_1008.sam_point_adapter import track_visible_points
        tracker = track_visible_points
    binding = {"frames_sha256": frames_sha256(rgb), "raw_source_sha256": raw_source_sha256,
               "frame_map_sha256": frame_map_sha256, "model_sha256": model_sha256}
    proposals, text_report = collect_sam_rgb_proposals(sam, rgb, raw_source_sha256=raw_source_sha256,
        frame_map_sha256=frame_map_sha256, model_sha256=model_sha256, glove_text_prompt=glove_text_prompt)
    proposals = [p for p in proposals if p.role in {"human", "glove"}]
    geometry = parent_mask.pico_geometry(session, rgb.shape[2] / session.width)
    geometry, invalid = visual._gate_diagnostic_geometry(geometry, session)
    hints = np.concatenate((geometry["hands"], geometry["forearm"], geometry["tracker"]), axis=2)
    bundle = generate_v1_seed_bundle(rgb, proposals, hints, raw_source_sha256=raw_source_sha256,
        frame_map_sha256=frame_map_sha256, model_sha256=model_sha256)
    assignments, unknown = associate_original_objects(bundle, original_objects)
    qualified, identity = qualify_identity_support(proposals, assignments, hints, binding)

    def preservation(before, after):
        lost = {key: int(np.count_nonzero(before[key] & ~after[key]))
                for key in ("core", "sam_core", "equipment", "removal", "effective_removal")
                if key in before}
        protected = int(np.count_nonzero(before["protected_visible"] != after["protected_visible"]))
        equipment = int(np.count_nonzero(before["equipment"] != after["equipment"]))
        if any(lost.values()) or protected or equipment:
            raise ValueError("AUTOMATION_REPAIR_PREVIOUS_ROUND_SUPPORT_NOT_PRESERVED")
        return {"support_lost_pixels": lost, "protected_mask_changed_pixels": protected,
                "equipment_changed_pixels": equipment}

    rounds, candidate, round_1_masks = [], original_masks, None
    for round_number in range(1, repair_rounds + 1):
        previous = candidate
        prompts, prompt_report = make_repair_prompts(previous, qualified, assignments,
            input_bindings=binding, prompt_policy=SPREAD_MAX_GAP_THREE)
        if prompts:
            # The existing adapter creates and finally clears a fresh video state
            # on each invocation. Never append round-two points to round one.
            tracks, point_report = tracker(sam, rgb, prompts, min(p["frame_idx"] for p in prompts))
            if point_report.get("errors") or point_report.get("aborted") or point_report.get("status") != "COMPLETE":
                raise ValueError("AUTOMATION_REPAIR_POINT_TRACKING_INCOMPLETE")
        else:
            tracks, point_report = {}, {"status": "NO_RGB_GAP", "errors": [], "aborted": []}
        point_report = deepcopy(point_report)
        point_report.update(seed_authority=REPAIR_AUTHORITY, human_visible_confirmed=False, manual_points_added=0,
            automatic_points=sum(len(p["points_px"]) for p in prompts), hardware_pixels_used_as_evidence=False)
        candidate, merge = merge_repair_support(previous, tracks, qualified, prompts, assignments,
            input_bindings=binding, prompt_policy=SPREAD_MAX_GAP_THREE)
        retained = preservation(previous, candidate)
        remaining_prompts, remaining_report = make_repair_prompts(candidate, qualified, assignments,
            input_bindings=binding, prompt_policy=SPREAD_MAX_GAP_THREE)
        rounds.append({"round_number": round_number,
            "input_mask_basis": "FRESH_ORIGINAL_V1" if round_number == 1 else "PREVIOUS_ROUND_CANDIDATE",
            "prompts": prompts, "prompt_metrics": prompt_report, "point_tracking": point_report, "merge": merge,
            "point_tracking_called": bool(prompts),
            "remaining_prompts": remaining_prompts, "remaining_prompt_metrics": remaining_report,
            "new_core_pixels_per_native_frame": np.count_nonzero(candidate["core"] & ~previous["core"], axis=(1, 2)).astype(int).tolist(),
            "new_removal_pixels_per_native_frame": np.count_nonzero(candidate["removal"] & ~previous["removal"], axis=(1, 2)).astype(int).tolist(),
            "previous_round_preservation": retained})
        if round_number == 1:
            # Always independent, including the one-round path where final and
            # round one have identical values and still must not share arrays.
            round_1_masks = deepcopy(candidate)
        if round_snapshots is not None and round_number == 2:
            round_snapshots[2] = deepcopy(candidate)
    first = rounds[0]
    final_preservation = preservation(original_masks, candidate)
    round_1_preservation = preservation(round_1_masks, candidate)
    audit = {"schema": "inpainting-v1-fresh-mask-repair-audit-v1", "input_bindings": binding,
        "repair_prompt_policy": SPREAD_MAX_GAP_THREE, "assignments": assignments, "unknown_identities": unknown,
        "text": text_report, "identity_qualification": identity, "invalid_geometry_frame_sides": invalid,
        "prompts": first["prompts"], "prompt_metrics": first["prompt_metrics"],
        "point_tracking": first["point_tracking"], "merge": first["merge"],
        "repair_rounds": repair_rounds, "rounds": rounds,
        "text_collection_passes": 1, "identity_qualification_passes": 1,
        "point_tracking_calls": sum(r["point_tracking_called"] for r in rounds),
        "point_state_policy": "FRESH_ADAPTER_STATE_PER_NONEMPTY_ROUND",
        "original_support_preservation": final_preservation,
        "round_1_support_preservation": round_1_preservation,
        "remaining_prompts": rounds[-1]["remaining_prompts"],
        "remaining_prompt_metrics": rounds[-1]["remaining_prompt_metrics"],
        "manual_points_added": 0, "historical_masks_consumed": False,
        "original_removal_lost_pixels": int(np.count_nonzero(original_masks["removal"] & ~candidate["removal"])),
        "protected_mask_changed_pixels": int(np.count_nonzero(original_masks["protected_visible"] != candidate["protected_visible"])),
        "training_eligible": False, "control_ground_truth": False}
    result = (candidate, audit, qualified, hints)
    return (*result, round_1_masks) if return_round_masks else result
