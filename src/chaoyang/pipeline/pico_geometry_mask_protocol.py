#!/usr/bin/env python3
"""Fail-closed validation for the PICO/raw-point Mask successor."""

from __future__ import annotations

from typing import Any, Mapping


class PicoGeometryProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PicoGeometryProtocolError(message)


def validate_shared(config: Mapping[str, Any]) -> None:
    require(
        config.get("schema_version") == "pico-raw-point-geometry-mask-successor-v1",
        "unexpected shared successor schema",
    )
    require(config.get("status") == "FROZEN_PENDING_GPU_FIT_EVAL", "successor is not frozen")
    model = config.get("shared_model", {})
    require(model.get("checkpoint_write_forbidden") is True, "checkpoint writes are not forbidden")
    spatial = config.get("spatial_execution", {})
    temporal = config.get("temporal_execution", {})
    setup_world = config.get("setup_world_anchor_generation", {})
    contact = config.get("task_object_contact_binding", {})
    providers = config.get("hand_anchor_provider_contract", {})
    object_authority = config.get("independent_2d_object_authority", {})
    require(spatial.get("sample_count") == 12, "spatial sample count must be 12")
    require(spatial.get("one_independent_model_state_per_frame") is True, "spatial state independence missing")
    require(spatial.get("cross_sample_flow_forbidden") is True, "cross-sample flow not forbidden")
    require(temporal.get("window_count") == 3, "temporal window count must be three")
    require(temporal.get("frames_per_window") == 12, "temporal windows must contain 12 frames")
    require(temporal.get("unit_stride_required") is True, "temporal unit stride missing")
    require(setup_world.get("full_session_pico_scan") is True, "setup anchor PICO scan missing")
    require(
        setup_world.get("exact_task_setup_reference_ray_preserved") is True,
        "setup reference ray is not preserved",
    )
    require(
        setup_world.get("manual_per_frame_points_forbidden") is True,
        "setup world anchors permit manual points",
    )
    require(
        setup_world.get("image_colour_or_intensity_used") is False,
        "setup world anchors use image colour",
    )
    require(
        float(setup_world.get("reference_match_distance_px_max", 0)) > 0,
        "setup world anchor match gate missing",
    )
    require(
        contact.get("static_setup_anchor_scope") == "INITIAL_SLOT_ONLY_THROUGH_BOUND_CONTACT_FRAME",
        "movable setup anchor scope is not contact bounded",
    )
    require(contact.get("post_contact_static_anchor_forbidden") is True, "post-contact static anchors permitted")
    require(
        contact.get("post_contact_requires")
        == "INDEPENDENT_OBJECT_INSTANCE_PROPAGATION_OR_OBJECT6D",
        "post-contact moving-object authority is not independent",
    )
    require(contact.get("contact_pre_post_same_raw_id_gate_required") is True, "contact identity gate missing")
    require(float(contact.get("thumb_index_distance_m_max", 0)) > 0, "contact pinch threshold missing")
    require(float(contact.get("reference_distance_px_max", 0)) > 0, "contact setup-distance threshold missing")
    require(
        providers.get("selection_scope") == "PER_SESSION_PER_PHYSICAL_SIDE",
        "anchor provider scope is not session/side fixed",
    )
    require(providers.get("session_side_constant_required") is True, "provider may change within a side")
    require(providers.get("silent_provider_mixing_forbidden") is True, "silent provider mixing permitted")
    provider_specs = providers.get("providers", {})
    require(set(provider_specs) == {"PICO21", "HAWOR_MANO21"}, "provider inventory drift")
    require(provider_specs["PICO21"].get("wrist_index") == 5, "PICO21 wrist index drift")
    require(
        provider_specs["PICO21"].get("required_applicability_gate") == "G2_PICO21_PRELIMINARY",
        "PICO21 G2 prerequisite missing",
    )
    require(provider_specs["HAWOR_MANO21"].get("wrist_index") == 0, "HaWoR wrist index drift")
    require(provider_specs["HAWOR_MANO21"].get("numeric_gate_required") is True, "HaWoR numeric gate missing")
    require(
        provider_specs["HAWOR_MANO21"].get("independent_identity_gate_required") is True,
        "HaWoR identity gate missing",
    )
    require(
        object_authority.get("one_independent_sam_state_per_instance") is True,
        "object instances may share a SAM state",
    )
    require(object_authority.get("one_fixed_raw_id_per_instance") is True, "object raw IDs are not fixed")
    require(
        object_authority.get("contact_reseed_source") == "PREREGISTERED_PICO_CONTACT_BINDING_ONLY",
        "object reseed is not bound to preregistered PICO contact",
    )
    require(
        object_authority.get("contact_reseed_label") == "ASSISTED_HAND_CARRIED",
        "assisted moving-object label drift",
    )
    require(object_authority.get("manual_per_frame_reseed_forbidden") is True, "manual object reseed permitted")
    require(
        object_authority.get("cross_instance_mutual_exclusion_forbidden") is True,
        "cross-instance exclusion could drop an object",
    )
    require(object_authority.get("full_source_frame_coverage_required") is True, "object authority is not full-frame")
    require(len(object_authority.get("frame_gate", {})) >= 7, "object frame gate incomplete")
    require(len(object_authority.get("identity_gate", {})) >= 6, "object identity gate incomplete")
    for section in ("tracker_anchor_generation", "task_object_anchor_generation"):
        generation = config.get(section, {})
        require(generation.get("manual_per_frame_points_forbidden") is True, f"{section} permits manual points")
        require(generation.get("image_colour_or_intensity_used") is False, f"{section} uses image colour")
        require(generation.get("discarded_priming_output_consumed") is False, f"{section} consumes priming output")
    forbidden = " ".join(config.get("forbidden", [])).lower()
    require("colour" in forbidden or "color" in forbidden, "colour-specific logic not forbidden")
    require("manual per-frame" in forbidden, "manual per-frame prompts not forbidden")
    tracker_gate = config.get("tracker_mask_gate", {})
    object_gate = config.get("protected_object_mask_gate", {})
    human_gate = config.get("human_mask_gate", {})
    require(len(tracker_gate) >= 10, "tracker hard gate incomplete")
    require(len(object_gate) >= 7, "protected-object hard gate incomplete")
    require(float(human_gate.get("side_identity_cost_margin_px_min", 0)) > 0, "human side-swap margin missing")
    require(len(config.get("fit_eval_order", [])) == 3, "FIT plus two EVAL identities required")


def validate_task(config: Mapping[str, Any], shared_path: str) -> None:
    require(config.get("schema_version") == "task-mask-pico-geometry-roles-v1", "unexpected task config schema")
    require(config.get("task_id") in {"chips", "poker"}, "unsupported task id")
    require(config.get("shared_successor") == shared_path, "task points at a different shared successor")
    require(config.get("threshold_override") is None, "task threshold override forbidden")
    require(bool(config.get("setup_geometry_normalized_xy")), "task setup geometry missing")
    require(bool(config.get("raw_point_role_limits")), "task role limits missing")
    groups = config.get("role_groups", {})
    require(set(groups) == {"task_object", "fixture"}, "task and fixture role groups required")
    covered: set[str] = set()
    for group_name, group in groups.items():
        role = group.get("published_role")
        require(role in config["raw_point_role_limits"], f"{group_name} published role has no limits")
        setup_roles = set(group.get("setup_roles", []))
        require(bool(setup_roles), f"{group_name} setup roles missing")
        require(not covered.intersection(setup_roles), "setup role assigned to multiple groups")
        covered.update(setup_roles)
    require(covered == set(config["setup_geometry_normalized_xy"]), "role groups do not cover setup geometry")
    task_group = groups["task_object"]
    require(
        task_group.get("setup_anchor_scope") == "INITIAL_SLOT_ONLY_THROUGH_BOUND_CONTACT_FRAME",
        "task movable setup anchors are not contact bounded",
    )
    require(
        task_group.get("post_contact_static_anchor_forbidden") is True,
        "task permits post-contact static anchors",
    )
    fixture_group = groups["fixture"]
    require(fixture_group.get("session_static") is True, "fixture is not declared session-static")
    for name, point in config["setup_geometry_normalized_xy"].items():
        require(len(point) == 2, f"{name} is not 2D")
        require(all(0.0 <= float(value) <= 1.0 for value in point), f"{name} is outside normalized image")
    for name, limits in config["raw_point_role_limits"].items():
        require(float(limits["frame_area_fraction_min"]) > 0, f"{name} minimum area invalid")
        require(
            float(limits["frame_area_fraction_min"]) < float(limits["frame_area_fraction_max"]) < 1,
            f"{name} area bounds invalid",
        )
        require(int(limits["maximum_instances"]) >= 1, f"{name} maximum instances invalid")


def validate_evaluation_plan(plan: Mapping[str, Any], shared: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    require(plan.get("schema_version") == "task-mask-successor-evaluation-plan-v1", "unexpected evaluation plan schema")
    require(plan.get("task_id") == task.get("task_id"), "plan/task identity mismatch")
    require(plan.get("threshold_override") is None, "evaluation threshold override forbidden")
    frames = list(map(int, plan.get("spatial_source_frames", [])))
    require(len(frames) == int(shared["spatial_execution"]["sample_count"]), "wrong spatial sample count")
    require(len(frames) == len(set(frames)), "duplicate spatial samples")
    require(frames == sorted(frames), "spatial samples must be sorted")
    frame_count = int(plan.get("frame_count", 0))
    require(all(0 <= frame < frame_count for frame in frames), "spatial frame outside session")
    windows = plan.get("temporal_windows", [])
    require(len(windows) == int(shared["temporal_execution"]["window_count"]), "wrong window count")
    used: set[int] = set()
    for window in windows:
        source = list(map(int, window.get("source_frames", [])))
        require(len(source) == int(shared["temporal_execution"]["frames_per_window"]), "wrong window length")
        require(source == list(range(source[0], source[0] + len(source))), "window is not unit stride")
        require(not used.intersection(source), "temporal windows overlap")
        require(all(0 <= frame < frame_count for frame in source), "temporal frame outside session")
        used.update(source)
    identity = f"{plan['task_id']}:{plan['session_id']}:{plan['split']}"
    require(identity in shared["fit_eval_order"], f"identity not frozen: {identity}")
    if plan.get("split") == "EVAL":
        require(plan.get("fit_result_access_for_prompt_design") is False, "EVAL may inspect FIT prompt results")


def validate_bundle(
    shared: Mapping[str, Any], task: Mapping[str, Any], plan: Mapping[str, Any], shared_path: str
) -> dict[str, Any]:
    validate_shared(shared)
    validate_task(task, shared_path)
    validate_evaluation_plan(plan, shared, task)
    return {
        "status": "PASS_FROZEN_CPU_CONTRACT",
        "task_id": task["task_id"],
        "session_id": plan["session_id"],
        "split": plan["split"],
        "spatial_independent": True,
        "temporal_unit_stride": True,
        "manual_points_forbidden": True,
        "colour_logic_forbidden": True,
        "threshold_override_absent": True,
    }


def validate_anchor_provider_authority(
    authority: Mapping[str, Any],
    shared: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a frozen per-session/per-side provider choice.

    This intentionally validates eligibility separately from model quality.  A
    missing provider means NOT_RUN/HOLD_PROVIDER_AUTHORITY, never a failed Mask
    model evaluation.
    """
    require(
        authority.get("schema_version") == "mask-hand-anchor-provider-authority-v1",
        "unexpected provider authority schema",
    )
    require(authority.get("status") == "PASS_FIXED_PROVIDER_AUTHORITY", "provider authority is not PASS")
    require(authority.get("task_id") == plan.get("task_id"), "provider task mismatch")
    require(authority.get("session_id") == plan.get("session_id"), "provider session mismatch")
    require(authority.get("split") == plan.get("split"), "provider split mismatch")
    require(authority.get("selection_scope") == "PER_SESSION_PER_PHYSICAL_SIDE", "provider scope drift")
    require(authority.get("silent_provider_mixing_forbidden") is True, "provider mixing not forbidden")
    rows = authority.get("providers_by_side", {})
    require(set(rows) == {"left", "right"}, "provider side inventory drift")
    specs = shared["hand_anchor_provider_contract"]["providers"]
    for side, row in rows.items():
        provider_id = row.get("provider_id")
        require(provider_id in specs, f"unknown {side} provider")
        spec = specs[provider_id]
        require(row.get("wrist_index") == spec["wrist_index"], f"{side} provider wrist index drift")
        require(row.get("eligibility_pass") is True, f"{side} provider is ineligible")
        if provider_id == "PICO21":
            require(row.get("applicability_gate") == "G2_PICO21_PRELIMINARY", f"{side} lacks G2 evidence")
            require(
                row.get("applicability_value") in spec["accepted_applicability_values"],
                f"{side} PICO21 G2 is not applicable",
            )
        elif provider_id == "HAWOR_MANO21":
            require(row.get("numeric_gate_pass") is True, f"{side} HaWoR numeric gate failed")
            require(row.get("independent_identity_gate_pass") is True, f"{side} HaWoR identity gate failed")
    return {
        "status": "PASS_FIXED_PROVIDER_AUTHORITY",
        "session_id": plan["session_id"],
        "providers_by_side": {side: rows[side]["provider_id"] for side in ("left", "right")},
        "silent_provider_mixing_forbidden": True,
    }
