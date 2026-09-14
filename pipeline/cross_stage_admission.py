"""Task-agnostic, fail-closed admission and blame routing across the pipeline.

The input is a normalized evidence envelope.  Stage producers keep ownership of
their numeric thresholds; this module owns only cross-stage scope, authority,
dependency, and consumption rules.  In particular, it never turns a producer
HOLD/NOT_EVALUATED into PASS and it never treats a canary or synthetic PASS as
a full-session PASS.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "chaoyang-cross-stage-evidence-v1"
OUTPUT_SCHEMA_VERSION = "chaoyang-cross-stage-admission-v1"

STATUSES = frozenset({"PASS", "FAIL", "HOLD", "NOT_EVALUATED"})
AUTHORITIES = frozenset({"CAPTURE", "PRODUCER", "INDEPENDENT_QA", "HUMAN_REVIEW", "NONE"})
SCOPES = frozenset({"FULL_SESSION", "CANARY", "WINDOW", "SYNTHETIC", "READINESS_ONLY", "UNKNOWN"})
PROFILES = frozenset({"RAW_RGB_KAI22", "CLEAN_RGB_KAI22", "ROBOT_RGB_COMPOSITE_KAI22"})
IMAGE_VARIANTS = frozenset({"RAW", "CLEAN", "ROBOT_COMPOSITE"})
MASK_PROVIDER_SLOTS = {
    # Hand and worn-tracker roles are side-specific.  A global PICO result may
    # not wash a failed left side into an apparently usable right-side route.
    "human_core": ("left", "right"),
    "wearable_tracker": ("left", "right"),
    # Task objects and the setup/background anchor are deliberately distinct.
    # A static setup observation is never authority for a subsequently moving
    # object instance.
    "moving_task_object": ("global",),
    "static_setup_anchor": ("global",),
}
MASK_PROVIDER_ROUTES = {
    "human_core": frozenset(
        {"HAWOR_MANO21", "PICO_OPENXR21", "MANUAL", "NOT_USED"}
    ),
    "wearable_tracker": frozenset(
        {"PICO_OPENXR21", "VISUAL_INSTANCE", "MANUAL", "NOT_USED"}
    ),
    "moving_task_object": frozenset(
        {"OBJECT6D_PROJECTED", "VISUAL_INSTANCE", "MANUAL", "NOT_USED"}
    ),
    "static_setup_anchor": frozenset(
        {"CAPTURE_STATIC_SETUP", "PICO_REPROJECTED_SETUP", "MANUAL", "NOT_USED"}
    ),
}

PROFILE_VARIANT = {
    "RAW_RGB_KAI22": "RAW",
    "CLEAN_RGB_KAI22": "CLEAN",
    "ROBOT_RGB_COMPOSITE_KAI22": "ROBOT_COMPOSITE",
}

STAGE_GATES = {
    "capture": frozenset(
        {
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "camera_world",
            "pico21_diagnostic",
            "pico21_left_provider",
            "pico21_right_provider",
            "task_interval_defined",
            "segment_contract",
            "clean_donor_authority",
            "static_setup_anchor_authority",
            "object6d_authority",
            "session_static_base",
        }
    ),
    "hawor": frozenset(
        {
            "execution",
            "mano21_structure",
            "numeric_mask_seed",
            "numeric_robot_seed",
            "anatomical_identity",
            "independent_contour",
            "direct_provenance",
            "contact_direct_observation",
        }
    ),
    "mask": frozenset(
        {
            "input_authority",
            "semantic_roles",
            "moving_task_object_identity",
            "static_setup_anchor_scope",
            "prompt_provider_lineage",
            "temporal_identity",
            "full_session_coverage",
            "object_protection",
            "lineage_integrity",
            "manual_review",
        }
    ),
    "clean": frozenset(
        {
            "input_mask_authority",
            "donor_authority",
            "donor_purity",
            "source_map",
            "spatial_residual",
            "illumination",
            "shadow",
            "seam",
            "temporal",
            "object_protection",
            "codec",
            "manual_review",
        }
    ),
    "robot": frozenset(
        {
            "input_hawor_authority",
            "input_clean_authority",
            "mount_calibration",
            "session_static_base",
            "object6d",
            "retarget_schema",
            "mano_reorder",
            "pose",
            "joint_limits",
            "temporal",
            "hand_morphology",
            "retarget_independent_qa",
            "functional_retarget_authority",
            "contact",
            "nonpenetration",
            "self_collision",
            "depth_occlusion",
            "manual_review",
        }
    ),
    "training": frozenset(
        {
            "split_frozen",
            "eval_disjoint",
            "task_isolated",
            "image_variant_frozen",
            "upstream_lineage_frozen",
            "selector_reproducible",
        }
    ),
}

INDEPENDENT_GATES = frozenset(
    {
        ("hawor", "anatomical_identity"),
        ("hawor", "independent_contour"),
        ("mask", "manual_review"),
        ("clean", "manual_review"),
        ("robot", "retarget_independent_qa"),
        ("robot", "functional_retarget_authority"),
        ("robot", "manual_review"),
        ("training", "eval_disjoint"),
    }
)

_STATUS_PRIORITY = {"PASS": 0, "NOT_EVALUATED": 1, "HOLD": 2, "FAIL": 3}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence_files(
    bundle: Mapping[str, Any], *, evidence_root: Path | None = None
) -> list[str]:
    """Re-hash every referenced evidence file without touching stage payloads."""

    errors: list[str] = []
    stages = bundle.get("stages")
    if not isinstance(stages, Mapping):
        return ["stages must be an object before evidence verification"]
    root = evidence_root or Path.cwd()
    for stage_name, stage in stages.items():
        if not isinstance(stage, Mapping):
            continue
        evidence = stage.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        contract = stage.get("gate_contract")
        items = list(evidence)
        if isinstance(contract, Mapping):
            items.append(contract)
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            raw_path = item.get("path")
            expected = item.get("sha256")
            if not isinstance(raw_path, str):
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            label = (
                f"stages.{stage_name}.evidence[{index}]"
                if index < len(evidence)
                else f"stages.{stage_name}.gate_contract"
            )
            if not path.is_file():
                errors.append(f"{label}: missing file {path}")
                continue
            actual = _sha256_file(path)
            if actual != expected:
                errors.append(f"{label}: sha256 mismatch expected={expected!r} actual={actual}")
    return errors


def validate_cross_stage_input(bundle: Mapping[str, Any]) -> list[str]:
    """Validate the normalized envelope while allowing absent/pending stages."""

    errors: list[str] = []
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("session_id", "task_id"):
        if not isinstance(bundle.get(key), str) or not bundle[key]:
            errors.append(f"{key} must be a non-empty string")
    raw_profile = bundle.get("training_profile")
    profile = raw_profile if isinstance(raw_profile, str) else None
    if not isinstance(profile, str) or profile not in PROFILES:
        errors.append(f"training_profile must be one of {sorted(PROFILES)}")
    variant = bundle.get("image_variant")
    if not isinstance(variant, str) or variant not in IMAGE_VARIANTS:
        errors.append(f"image_variant must be one of {sorted(IMAGE_VARIANTS)}")
    stages = bundle.get("stages")
    if not isinstance(stages, Mapping):
        return errors + ["stages must be an object"]
    unknown_stages = sorted(set(stages) - set(STAGE_GATES))
    if unknown_stages:
        errors.append(f"unknown stages: {unknown_stages}")
    for stage_name, stage in stages.items():
        if stage_name not in STAGE_GATES:
            continue
        prefix = f"stages.{stage_name}"
        if not isinstance(stage, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        scope = stage.get("scope")
        if not isinstance(scope, str) or scope not in SCOPES:
            errors.append(f"{prefix}.scope must be one of {sorted(SCOPES)}")
        for metadata_key in ("reported_status", "claim_limit"):
            value = stage.get(metadata_key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{metadata_key} must be a non-empty string")
        evidence = stage.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{prefix}.evidence must be a non-empty array")
        else:
            for index, item in enumerate(evidence):
                label = f"{prefix}.evidence[{index}]"
                if not isinstance(item, Mapping):
                    errors.append(f"{label} must be an object")
                    continue
                if not isinstance(item.get("path"), str) or not item["path"]:
                    errors.append(f"{label}.path must be a non-empty string")
                digest = item.get("sha256")
                if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                    errors.append(f"{label}.sha256 must be lowercase 64-hex")
        contract = stage.get("gate_contract")
        if not isinstance(contract, Mapping):
            errors.append(f"{prefix}.gate_contract must be an object")
        else:
            if not isinstance(contract.get("id"), str) or not contract["id"]:
                errors.append(f"{prefix}.gate_contract.id must be a non-empty string")
            if not isinstance(contract.get("path"), str) or not contract["path"]:
                errors.append(f"{prefix}.gate_contract.path must be a non-empty string")
            digest = contract.get("sha256")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                errors.append(f"{prefix}.gate_contract.sha256 must be lowercase 64-hex")
            if not isinstance(contract.get("frozen_before_evaluation"), bool):
                errors.append(f"{prefix}.gate_contract.frozen_before_evaluation must be boolean")
        gates = stage.get("gates")
        if not isinstance(gates, Mapping):
            errors.append(f"{prefix}.gates must be an object")
            continue
        unknown_gates = sorted(set(gates) - set(STAGE_GATES[stage_name]))
        if unknown_gates:
            errors.append(f"{prefix}.gates has unknown keys: {unknown_gates}")
        for gate_name, gate in gates.items():
            label = f"{prefix}.gates.{gate_name}"
            if not isinstance(gate, Mapping):
                errors.append(f"{label} must be an object")
                continue
            status = gate.get("status")
            if not isinstance(status, str) or status not in STATUSES:
                errors.append(f"{label}.status must be one of {sorted(STATUSES)}")
            authority = gate.get("authority")
            if not isinstance(authority, str) or authority not in AUTHORITIES:
                errors.append(f"{label}.authority must be one of {sorted(AUTHORITIES)}")
            reasons = gate.get("reason_codes")
            if not isinstance(reasons, list) or any(not isinstance(value, str) or not value for value in reasons):
                errors.append(f"{label}.reason_codes must be an array of non-empty strings")
            elif status != "PASS" and not reasons:
                errors.append(f"{label}.reason_codes cannot be empty for {status}")
        if stage_name in {"mask", "clean", "robot"} and not isinstance(
            stage.get("consumption_authorized"), bool
        ):
            errors.append(f"{prefix}.consumption_authorized must be boolean")
        if stage_name == "robot" and not isinstance(
            stage.get("sidecar_consumption_authorized"), bool
        ):
            errors.append(f"{prefix}.sidecar_consumption_authorized must be boolean")
        seed_route = stage.get("seed_route")
        if stage_name == "mask" and (
            not isinstance(seed_route, str)
            or seed_route not in {"HAWOR", "PICO_RAW_POINT", "MANUAL", "MIXED"}
        ):
            errors.append(
                f"{prefix}.seed_route must be HAWOR, PICO_RAW_POINT, MANUAL, or MIXED"
            )
        if stage_name == "mask":
            providers = stage.get("provider_routes")
            if not isinstance(providers, Mapping):
                errors.append(f"{prefix}.provider_routes must be an object")
            else:
                missing = sorted(set(MASK_PROVIDER_SLOTS) - set(providers))
                unknown = sorted(set(providers) - set(MASK_PROVIDER_SLOTS))
                if missing:
                    errors.append(f"{prefix}.provider_routes missing roles: {missing}")
                if unknown:
                    errors.append(f"{prefix}.provider_routes has unknown roles: {unknown}")
                for role, slots in MASK_PROVIDER_SLOTS.items():
                    if role not in providers:
                        continue
                    role_routes = providers[role]
                    role_prefix = f"{prefix}.provider_routes.{role}"
                    if not isinstance(role_routes, Mapping):
                        errors.append(f"{role_prefix} must be an object")
                        continue
                    missing_slots = sorted(set(slots) - set(role_routes))
                    unknown_slots = sorted(set(role_routes) - set(slots))
                    if missing_slots:
                        errors.append(f"{role_prefix} missing slots: {missing_slots}")
                    if unknown_slots:
                        errors.append(f"{role_prefix} has unknown slots: {unknown_slots}")
                    for side in slots:
                        entry = role_routes.get(side)
                        entry_prefix = f"{role_prefix}.{side}"
                        if not isinstance(entry, Mapping):
                            errors.append(f"{entry_prefix} must be an object")
                            continue
                        unknown_entry = sorted(set(entry) - {"provider", "eligibility"})
                        if unknown_entry:
                            errors.append(f"{entry_prefix} has unknown keys: {unknown_entry}")
                        provider = entry.get("provider")
                        if (
                            not isinstance(provider, str)
                            or provider not in MASK_PROVIDER_ROUTES[role]
                        ):
                            errors.append(
                                f"{entry_prefix}.provider must be one of "
                                f"{sorted(MASK_PROVIDER_ROUTES[role])}"
                            )
                        eligibility = entry.get("eligibility")
                        if not isinstance(eligibility, Mapping):
                            errors.append(f"{entry_prefix}.eligibility must be an object")
                            continue
                        unknown_eligibility = sorted(
                            set(eligibility) - {"status", "authority", "reason_codes"}
                        )
                        if unknown_eligibility:
                            errors.append(
                                f"{entry_prefix}.eligibility has unknown keys: "
                                f"{unknown_eligibility}"
                            )
                        status = eligibility.get("status")
                        authority = eligibility.get("authority")
                        reasons = eligibility.get("reason_codes")
                        if not isinstance(status, str) or status not in STATUSES:
                            errors.append(
                                f"{entry_prefix}.eligibility.status must be one of "
                                f"{sorted(STATUSES)}"
                            )
                        if not isinstance(authority, str) or authority not in AUTHORITIES:
                            errors.append(
                                f"{entry_prefix}.eligibility.authority must be one of "
                                f"{sorted(AUTHORITIES)}"
                            )
                        if not isinstance(reasons, list) or any(
                            not isinstance(value, str) or not value for value in reasons
                        ):
                            errors.append(
                                f"{entry_prefix}.eligibility.reason_codes must be an array "
                                "of non-empty strings"
                            )
                        elif status != "PASS" and not reasons:
                            errors.append(
                                f"{entry_prefix}.eligibility.reason_codes cannot be empty "
                                f"for {status}"
                            )
                expected_seed = None
                if isinstance(providers.get("human_core"), Mapping):
                    human_providers = {
                        str(entry.get("provider"))
                        for entry in providers["human_core"].values()
                        if isinstance(entry, Mapping)
                        and isinstance(entry.get("provider"), str)
                    }
                    expected_seed = {
                        frozenset({"HAWOR_MANO21"}): "HAWOR",
                        frozenset({"PICO_OPENXR21"}): "PICO_RAW_POINT",
                        frozenset({"MANUAL"}): "MANUAL",
                    }.get(frozenset(human_providers), "MIXED")
                if expected_seed is not None and stage.get("seed_route") != expected_seed:
                    errors.append(f"{prefix}.seed_route conflicts with human-core side providers")
    return errors


def _pseudo(gate: str, status: str, reason_code: str | None = None) -> dict[str, Any]:
    reasons = [] if reason_code is None else [reason_code]
    return {
        "gate": gate,
        "observed_status": status,
        "status": status,
        "authority": "NONE",
        "reason_codes": reasons,
    }


def _stage(bundle: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    stages = bundle.get("stages", {})
    value = stages.get(name) if isinstance(stages, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _gate(bundle: Mapping[str, Any], stage_name: str, gate_name: str) -> dict[str, Any]:
    stage = _stage(bundle, stage_name)
    if stage is None:
        return _pseudo(f"{stage_name}.{gate_name}", "NOT_EVALUATED", "STAGE_RESULT_MISSING")
    gates = stage.get("gates")
    raw = gates.get(gate_name) if isinstance(gates, Mapping) else None
    if not isinstance(raw, Mapping):
        return _pseudo(f"{stage_name}.{gate_name}", "NOT_EVALUATED", "REQUIRED_GATE_MISSING")
    observed = raw.get("status")
    if not isinstance(observed, str) or observed not in STATUSES:
        return _pseudo(f"{stage_name}.{gate_name}", "NOT_EVALUATED", "INVALID_GATE_STATUS")
    raw_authority = raw.get("authority", "NONE")
    authority = (
        raw_authority
        if isinstance(raw_authority, str) and raw_authority in AUTHORITIES
        else "NONE"
    )
    raw_reasons = raw.get("reason_codes")
    reasons = (
        [value for value in raw_reasons if isinstance(value, str) and value]
        if isinstance(raw_reasons, list)
        else []
    )
    effective = observed
    if not isinstance(raw_authority, str) or raw_authority not in AUTHORITIES:
        effective = "HOLD" if observed == "PASS" else observed
        reasons.append("INVALID_GATE_AUTHORITY")
    if not isinstance(raw_reasons, list) or len(reasons) != len(raw_reasons):
        effective = "HOLD" if observed == "PASS" else observed
        reasons.append("INVALID_GATE_REASON_CODES")
    if observed == "PASS":
        evidence = stage.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            effective = "HOLD"
            reasons.append("PASS_WITHOUT_EVIDENCE_REFERENCE")
        if authority == "NONE":
            effective = "HOLD"
            reasons.append("PASS_WITHOUT_AUTHORITY")
        if (stage_name, gate_name) in INDEPENDENT_GATES and authority not in {
            "INDEPENDENT_QA",
            "HUMAN_REVIEW",
        }:
            effective = "HOLD"
            reasons.append("PASS_WITHOUT_REQUIRED_INDEPENDENT_REVIEW")
    if observed != "PASS" and not reasons:
        reasons.append("NON_PASS_WITHOUT_REASON_CODE")
    return {
        "gate": f"{stage_name}.{gate_name}",
        "observed_status": observed,
        "status": effective,
        "authority": authority,
        "reason_codes": sorted(set(reasons)),
    }


def _scope_gate(bundle: Mapping[str, Any], stage_name: str) -> dict[str, Any]:
    stage = _stage(bundle, stage_name)
    if stage is None:
        return _pseudo(f"{stage_name}.__full_session_scope", "NOT_EVALUATED", "STAGE_RESULT_MISSING")
    scope = stage.get("scope")
    if scope == "FULL_SESSION":
        return _pseudo(f"{stage_name}.__full_session_scope", "PASS")
    return _pseudo(
        f"{stage_name}.__full_session_scope",
        "HOLD",
        f"NON_FULL_SESSION_SCOPE_{str(scope).upper()}",
    )


def _contract_gate(bundle: Mapping[str, Any], stage_name: str) -> dict[str, Any]:
    stage = _stage(bundle, stage_name)
    if stage is None:
        return _pseudo(f"{stage_name}.__gate_contract", "NOT_EVALUATED", "STAGE_RESULT_MISSING")
    contract = stage.get("gate_contract")
    if not isinstance(contract, Mapping):
        return _pseudo(f"{stage_name}.__gate_contract", "HOLD", "GATE_CONTRACT_MISSING")
    digest = contract.get("sha256")
    metadata_ok = (
        isinstance(contract.get("id"), str)
        and bool(contract["id"])
        and isinstance(contract.get("path"), str)
        and bool(contract["path"])
        and isinstance(digest, str)
        and _SHA256_RE.fullmatch(digest) is not None
    )
    if not metadata_ok:
        return _pseudo(f"{stage_name}.__gate_contract", "HOLD", "GATE_CONTRACT_IDENTITY_INVALID")
    if contract.get("frozen_before_evaluation") is not True:
        return _pseudo(
            f"{stage_name}.__gate_contract",
            "HOLD",
            "GATE_CONTRACT_NOT_FROZEN_BEFORE_EVALUATION",
        )
    return _pseudo(f"{stage_name}.__gate_contract", "PASS")


def _authorization_gate(bundle: Mapping[str, Any], stage_name: str) -> dict[str, Any]:
    stage = _stage(bundle, stage_name)
    if stage is None:
        return _pseudo(f"{stage_name}.__consumption_authorized", "NOT_EVALUATED", "STAGE_RESULT_MISSING")
    if stage.get("consumption_authorized") is True:
        return _pseudo(f"{stage_name}.__consumption_authorized", "PASS")
    return _pseudo(
        f"{stage_name}.__consumption_authorized",
        "HOLD",
        "CONSUMPTION_NOT_AUTHORIZED",
    )


def _robot_sidecar_authorization_gate(bundle: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(bundle, "robot")
    if stage is None:
        return _pseudo(
            "robot.__sidecar_consumption_authorized",
            "NOT_EVALUATED",
            "STAGE_RESULT_MISSING",
        )
    if stage.get("sidecar_consumption_authorized") is True:
        return _pseudo("robot.__sidecar_consumption_authorized", "PASS")
    return _pseudo(
        "robot.__sidecar_consumption_authorized",
        "HOLD",
        "ROBOT_SIDECAR_CONSUMPTION_NOT_AUTHORIZED",
    )


def _mask_provider_eligibility_gate(
    bundle: Mapping[str, Any], role: str, side: str
) -> dict[str, Any]:
    """Return independently authorized eligibility for one semantic provider slot."""

    gate_name = f"mask.__provider_eligibility_{role}_{side}"
    stage = _stage(bundle, "mask")
    providers = stage.get("provider_routes") if stage else None
    role_routes = providers.get(role) if isinstance(providers, Mapping) else None
    entry = role_routes.get(side) if isinstance(role_routes, Mapping) else None
    raw = entry.get("eligibility") if isinstance(entry, Mapping) else None
    if not isinstance(raw, Mapping):
        return _pseudo(gate_name, "NOT_EVALUATED", "PROVIDER_ELIGIBILITY_MISSING")
    observed = raw.get("status")
    if not isinstance(observed, str) or observed not in STATUSES:
        return _pseudo(gate_name, "NOT_EVALUATED", "INVALID_PROVIDER_ELIGIBILITY_STATUS")
    raw_authority = raw.get("authority", "NONE")
    authority = (
        raw_authority
        if isinstance(raw_authority, str) and raw_authority in AUTHORITIES
        else "NONE"
    )
    raw_reasons = raw.get("reason_codes")
    reasons = (
        [value for value in raw_reasons if isinstance(value, str) and value]
        if isinstance(raw_reasons, list)
        else []
    )
    effective = observed
    if not isinstance(raw_authority, str) or raw_authority not in AUTHORITIES:
        effective = "HOLD" if observed == "PASS" else observed
        reasons.append("INVALID_PROVIDER_ELIGIBILITY_AUTHORITY")
    if not isinstance(raw_reasons, list) or len(reasons) != len(raw_reasons):
        effective = "HOLD" if observed == "PASS" else observed
        reasons.append("INVALID_PROVIDER_ELIGIBILITY_REASON_CODES")
    if observed == "PASS" and authority not in {"INDEPENDENT_QA", "HUMAN_REVIEW"}:
        effective = "HOLD"
        reasons.append("PROVIDER_ELIGIBILITY_REQUIRES_INDEPENDENT_REVIEW")
    if observed != "PASS" and not reasons:
        reasons.append("NON_PASS_WITHOUT_REASON_CODE")
    return {
        "gate": gate_name,
        "observed_status": observed,
        "status": effective,
        "authority": authority,
        "reason_codes": sorted(set(reasons)),
    }


def _aggregate(name: str, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    if not rows:
        rows = [_pseudo(f"{name}.__empty", "NOT_EVALUATED", "NO_GATE_RECORDS")]
    status = max((str(row["status"]) for row in rows), key=_STATUS_PRIORITY.__getitem__)
    blockers = [
        {
            "gate": row["gate"],
            "status": row["status"],
            "observed_status": row["observed_status"],
            "reason_codes": row["reason_codes"],
        }
        for row in rows
        if row["status"] != "PASS"
    ]
    return {"name": name, "status": status, "gates": rows, "blockers": blockers}


def _capability(name: str, bundle: Mapping[str, Any], gate_names: Iterable[str]) -> dict[str, Any]:
    result = _aggregate(
        name,
        [
            *(_gate(bundle, "capture", gate) for gate in gate_names),
            _contract_gate(bundle, "capture"),
        ],
    )
    result["allowed"] = result["status"] == "PASS"
    return result


def _formal_capability(
    name: str, capability: Mapping[str, Any], capture_scope_gate: Mapping[str, Any]
) -> dict[str, Any]:
    result = _aggregate(name, [*capability["gates"], capture_scope_gate])
    result["allowed"] = result["status"] == "PASS"
    return result


def _cross_requirement(gate: str, passed: bool, reason: str) -> dict[str, Any]:
    return _pseudo(gate, "PASS" if passed else "HOLD", None if passed else reason)


def _causal_status(observed: str, dependencies: Iterable[str]) -> str:
    return observed if all(value == "PASS" for value in dependencies) else "BLOCKED_BY_UPSTREAM"


def _category_view(
    category: Mapping[str, Any], dependencies: Iterable[tuple[str, str]]
) -> dict[str, Any]:
    result = dict(category)
    dependency_map: dict[str, str] = {}
    for name, status in dependencies:
        dependency_map[name] = status
    deps = [{"route": name, "status": status} for name, status in dependency_map.items()]
    result["dependencies"] = deps
    result["dependency_statuses"] = [item["status"] for item in deps]
    result["causal_status"] = _causal_status(
        str(category["status"]), (item["status"] for item in deps)
    )
    return result


def evaluate_cross_stage(
    bundle: Mapping[str, Any], *, evidence_verification_errors: Iterable[str] = ()
) -> dict[str, Any]:
    """Evaluate route-specific readiness and accountable failure ownership."""

    validation_errors = validate_cross_stage_input(bundle)
    verification_errors = list(evidence_verification_errors)
    input_errors = validation_errors + verification_errors

    hawor_input = _capability(
        "hawor_rgb_input",
        bundle,
        ("media_integrity", "timeline_integrity", "camera_intrinsics"),
    )
    mask_input = _capability(
        "mask_image_input",
        bundle,
        ("media_integrity", "timeline_integrity", "camera_intrinsics"),
    )
    pico_raw_point_input = _capability(
        "pico_raw_point_input",
        bundle,
        (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "pico21_diagnostic",
        ),
    )
    pico_side_inputs = {
        side: _capability(
            f"pico_{side}_provider_input",
            bundle,
            (
                "media_integrity",
                "timeline_integrity",
                "camera_intrinsics",
                f"pico21_{side}_provider",
            ),
        )
        for side in ("left", "right")
    }
    static_setup_anchor_input = _capability(
        "static_setup_anchor_input",
        bundle,
        (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "static_setup_anchor_authority",
        ),
    )
    object6d_projection_input = _capability(
        "object6d_projection_input",
        bundle,
        (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "camera_world",
            "object6d_authority",
        ),
    )
    clean_input = _capability(
        "clean_capture_input",
        bundle,
        (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "task_interval_defined",
            "segment_contract",
            "clean_donor_authority",
        ),
    )
    robot_motion_input = _capability(
        "robot_motion_capture_input",
        bundle,
        (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "camera_world",
            "task_interval_defined",
            "session_static_base",
        ),
    )
    robot_composite_input = _capability(
        "robot_composite_capture_input",
        bundle,
        (
            "media_integrity",
            "timeline_integrity",
            "camera_intrinsics",
            "camera_world",
            "task_interval_defined",
            "segment_contract",
            "session_static_base",
            "object6d_authority",
        ),
    )
    capture_scope = _scope_gate(bundle, "capture")
    formal_hawor_input = _formal_capability(
        "formal_hawor_rgb_input", hawor_input, capture_scope
    )
    formal_mask_input = _formal_capability(
        "formal_mask_image_input", mask_input, capture_scope
    )
    formal_pico_raw_point_input = _formal_capability(
        "formal_pico_raw_point_input", pico_raw_point_input, capture_scope
    )
    formal_pico_side_inputs = {
        side: _formal_capability(
            f"formal_pico_{side}_provider_input", capability, capture_scope
        )
        for side, capability in pico_side_inputs.items()
    }
    formal_static_setup_anchor_input = _formal_capability(
        "formal_static_setup_anchor_input", static_setup_anchor_input, capture_scope
    )
    formal_object6d_projection_input = _formal_capability(
        "formal_object6d_projection_input", object6d_projection_input, capture_scope
    )
    formal_clean_input = _formal_capability(
        "formal_clean_capture_input", clean_input, capture_scope
    )
    formal_robot_motion_input = _formal_capability(
        "formal_robot_motion_capture_input", robot_motion_input, capture_scope
    )
    formal_robot_composite_input = _formal_capability(
        "formal_robot_composite_capture_input", robot_composite_input, capture_scope
    )

    raw_profile = bundle.get("training_profile")
    profile = raw_profile if isinstance(raw_profile, str) else None
    capture_for_profile = {
        "RAW_RGB_KAI22": formal_robot_motion_input,
        "CLEAN_RGB_KAI22": formal_clean_input,
        "ROBOT_RGB_COMPOSITE_KAI22": formal_robot_composite_input,
    }.get(profile, _aggregate("invalid_profile_capture", []))
    upstream_capture = dict(capture_for_profile)
    upstream_capture["name"] = "upstream_capture"
    upstream_capture["capabilities"] = {
        "hawor_rgb_input": hawor_input,
        "mask_image_input": mask_input,
        "pico_raw_point_input": pico_raw_point_input,
        "pico_left_provider_input": pico_side_inputs["left"],
        "pico_right_provider_input": pico_side_inputs["right"],
        "static_setup_anchor_input": static_setup_anchor_input,
        "object6d_projection_input": object6d_projection_input,
        "clean_capture_input": clean_input,
        "robot_motion_capture_input": robot_motion_input,
        "robot_composite_capture_input": robot_composite_input,
    }
    upstream_capture["formal_capabilities"] = {
        "hawor_rgb_input": formal_hawor_input,
        "mask_image_input": formal_mask_input,
        "pico_raw_point_input": formal_pico_raw_point_input,
        "pico_left_provider_input": formal_pico_side_inputs["left"],
        "pico_right_provider_input": formal_pico_side_inputs["right"],
        "static_setup_anchor_input": formal_static_setup_anchor_input,
        "object6d_projection_input": formal_object6d_projection_input,
        "clean_capture_input": formal_clean_input,
        "robot_motion_capture_input": formal_robot_motion_input,
        "robot_composite_capture_input": formal_robot_composite_input,
    }
    upstream_capture["advisories"] = {
        "pico21_diagnostic": _gate(bundle, "capture", "pico21_diagnostic"),
        "rule": "PICO21 is blocking only for a consumer that explicitly declares PICO_RAW_POINT.",
    }

    hawor_scope = _scope_gate(bundle, "hawor")
    hawor_contract = _contract_gate(bundle, "hawor")
    hawor_mask_seed = _aggregate(
        "hawor_mask_seed",
        [
            _gate(bundle, "hawor", "execution"),
            _gate(bundle, "hawor", "mano21_structure"),
            _gate(bundle, "hawor", "numeric_mask_seed"),
            _gate(bundle, "hawor", "anatomical_identity"),
            _gate(bundle, "hawor", "independent_contour"),
            _gate(bundle, "hawor", "direct_provenance"),
            hawor_scope,
            hawor_contract,
        ],
    )
    hawor_motion = _aggregate(
        "hawor_motion_sidecar",
        [
            _gate(bundle, "hawor", "execution"),
            _gate(bundle, "hawor", "mano21_structure"),
            _gate(bundle, "hawor", "numeric_robot_seed"),
            _gate(bundle, "hawor", "anatomical_identity"),
            _gate(bundle, "hawor", "independent_contour"),
            _gate(bundle, "hawor", "direct_provenance"),
            hawor_scope,
            hawor_contract,
        ],
    )
    hawor_contact = _aggregate(
        "hawor_contact_seed",
        [
            *hawor_motion["gates"],
            _gate(bundle, "hawor", "contact_direct_observation"),
        ],
    )
    hawor_algorithm = _aggregate(
        "hawor_algorithm",
        [*hawor_mask_seed["gates"], *hawor_motion["gates"]],
    )
    hawor_algorithm["routes"] = {
        "mask_seed": _category_view(
            hawor_mask_seed,
            [("upstream_capture.hawor_rgb_input", hawor_input["status"])],
        ),
        "motion_sidecar": _category_view(
            hawor_motion,
            [("upstream_capture.hawor_rgb_input", hawor_input["status"])],
        ),
        "contact_seed": _category_view(
            hawor_contact,
            [
                ("upstream_capture.hawor_rgb_input", hawor_input["status"]),
                (
                    "upstream_capture.robot_composite_capture_input",
                    robot_composite_input["status"],
                ),
            ],
        ),
    }
    hawor_algorithm = _category_view(
        hawor_algorithm,
        [("upstream_capture.hawor_rgb_input", hawor_input["status"])],
    )

    mask_stage = _stage(bundle, "mask")
    provider_routes = mask_stage.get("provider_routes") if mask_stage else None
    provider_admissions: dict[str, dict[str, Any]] = {}
    provider_dependency_statuses: list[tuple[str, str]] = []
    for role, slots in MASK_PROVIDER_SLOTS.items():
        role_routes = provider_routes.get(role) if isinstance(provider_routes, Mapping) else None
        for side in slots:
            entry = role_routes.get(side) if isinstance(role_routes, Mapping) else None
            provider = entry.get("provider") if isinstance(entry, Mapping) else None
            slot_id = f"{role}.{side}"
            eligibility = _mask_provider_eligibility_gate(bundle, role, side)
            cross_records: list[dict[str, Any]] = []
            graph_dependencies: list[str] = []
            dependency_rows: list[tuple[str, str]] = []
            if provider == "HAWOR_MANO21":
                # Numeric MANO21 alone is intentionally insufficient.  The
                # hawor_mask_seed aggregate also requires anatomical identity
                # and independent contour review before substitution is legal.
                cross_records.extend(
                    [
                        _cross_requirement(
                            f"mask.__provider_{role}_{side}_hawor_capture",
                            formal_hawor_input["allowed"],
                            "HAWOR_PROVIDER_CAPTURE_NOT_ADMITTED",
                        ),
                        _cross_requirement(
                            f"mask.__provider_{role}_{side}_hawor_identity_contour",
                            hawor_mask_seed["status"] == "PASS",
                            "HAWOR_PROVIDER_REQUIRES_IDENTITY_AND_CONTOUR_PASS",
                        ),
                    ]
                )
                graph_dependencies.append("hawor.mask_seed")
                dependency_rows.append(
                    ("hawor_algorithm.mask_seed", hawor_mask_seed["status"])
                )
            elif provider == "PICO_OPENXR21":
                side_capability = formal_pico_side_inputs.get(side)
                side_allowed = bool(side_capability and side_capability["allowed"])
                cross_records.append(
                    _cross_requirement(
                        f"mask.__provider_{role}_{side}_pico",
                        side_allowed,
                        f"PICO_{side.upper()}_PROVIDER_NOT_ADMITTED",
                    )
                )
                graph_dependencies.append(f"capture.pico_{side}")
                dependency_rows.append(
                    (
                        f"upstream_capture.pico_{side}_provider_input",
                        side_capability["status"] if side_capability else "NOT_EVALUATED",
                    )
                )
            elif provider == "OBJECT6D_PROJECTED":
                cross_records.append(
                    _cross_requirement(
                        f"mask.__provider_{role}_{side}_object6d",
                        formal_object6d_projection_input["allowed"],
                        "OBJECT6D_PROVIDER_NOT_ADMITTED",
                    )
                )
                graph_dependencies.append("capture.object6d_projection")
                dependency_rows.append(
                    (
                        "upstream_capture.object6d_projection_input",
                        formal_object6d_projection_input["status"],
                    )
                )
            elif provider == "CAPTURE_STATIC_SETUP":
                cross_records.append(
                    _cross_requirement(
                        f"mask.__provider_{role}_{side}_static_setup",
                        formal_static_setup_anchor_input["allowed"],
                        "STATIC_SETUP_ANCHOR_PROVIDER_NOT_ADMITTED",
                    )
                )
                graph_dependencies.append("capture.static_setup_anchor")
                dependency_rows.append(
                    (
                        "upstream_capture.static_setup_anchor_input",
                        formal_static_setup_anchor_input["status"],
                    )
                )
            elif provider == "PICO_REPROJECTED_SETUP":
                cross_records.extend(
                    [
                        _cross_requirement(
                            f"mask.__provider_{role}_{side}_pico_setup",
                            formal_pico_raw_point_input["allowed"],
                            "PICO_REPROJECTED_SETUP_INPUT_NOT_ADMITTED",
                        ),
                        _cross_requirement(
                            f"mask.__provider_{role}_{side}_static_setup",
                            formal_static_setup_anchor_input["allowed"],
                            "STATIC_SETUP_ANCHOR_PROVIDER_NOT_ADMITTED",
                        ),
                    ]
                )
                graph_dependencies.extend(
                    ["capture.pico_raw_point", "capture.static_setup_anchor"]
                )
                dependency_rows.extend(
                    [
                        (
                            "upstream_capture.pico_raw_point_input",
                            formal_pico_raw_point_input["status"],
                        ),
                        (
                            "upstream_capture.static_setup_anchor_input",
                            formal_static_setup_anchor_input["status"],
                        ),
                    ]
                )
            elif provider in ("VISUAL_INSTANCE", "MANUAL", "NOT_USED"):
                cross_records.append(
                    _pseudo(f"mask.__provider_{role}_{side}_{provider.lower()}", "PASS")
                )
            else:
                cross_records.append(
                    _pseudo(
                        f"mask.__provider_{role}_{side}",
                        "NOT_EVALUATED",
                        "MASK_PROVIDER_ROUTE_MISSING_OR_INVALID",
                    )
                )
            admission = _aggregate(
                f"mask_provider_{role}_{side}", [eligibility, *cross_records]
            )
            provider_admissions[slot_id] = {
                "provider": provider,
                "admission": admission,
                "graph_dependencies": graph_dependencies,
                "dependency_rows": dependency_rows,
            }
            provider_dependency_statuses.extend(dependency_rows)
    provider_gate_records = [
        record
        for item in provider_admissions.values()
        for record in item["admission"]["gates"]
    ]
    mask_scope = _scope_gate(bundle, "mask")
    mask_auth = _authorization_gate(bundle, "mask")
    mask_contract = _contract_gate(bundle, "mask")
    mask_semantic = _aggregate(
        "mask_semantic",
        [
            _gate(bundle, "mask", "input_authority"),
            *provider_gate_records,
            _gate(bundle, "mask", "prompt_provider_lineage"),
            _gate(bundle, "mask", "semantic_roles"),
            _gate(bundle, "mask", "moving_task_object_identity"),
            _gate(bundle, "mask", "static_setup_anchor_scope"),
            _gate(bundle, "mask", "object_protection"),
            _gate(bundle, "mask", "lineage_integrity"),
            _gate(bundle, "mask", "manual_review"),
            mask_scope,
            mask_auth,
            mask_contract,
        ],
    )
    mask_temporal = _aggregate(
        "mask_temporal",
        [
            _gate(bundle, "mask", "input_authority"),
            *provider_gate_records,
            _gate(bundle, "mask", "prompt_provider_lineage"),
            _gate(bundle, "mask", "temporal_identity"),
            _gate(bundle, "mask", "moving_task_object_identity"),
            _gate(bundle, "mask", "static_setup_anchor_scope"),
            _gate(bundle, "mask", "full_session_coverage"),
            _gate(bundle, "mask", "lineage_integrity"),
            _gate(bundle, "mask", "manual_review"),
            mask_scope,
            mask_auth,
            mask_contract,
        ],
    )
    mask_formal = _aggregate("mask_formal", [*mask_semantic["gates"], *mask_temporal["gates"]])
    mask_dependencies = [
        ("upstream_capture.mask_image_input", formal_mask_input["status"]),
        *provider_dependency_statuses,
    ]
    mask_semantic = _category_view(mask_semantic, mask_dependencies)
    mask_temporal = _category_view(mask_temporal, mask_dependencies)
    mask_semantic["provider_admission"] = {
        slot: {
            "provider": item["provider"],
            "status": item["admission"]["status"],
            "blockers": item["admission"]["blockers"],
        }
        for slot, item in provider_admissions.items()
    }

    clean_scope = _scope_gate(bundle, "clean")
    clean_auth = _authorization_gate(bundle, "clean")
    clean_contract = _contract_gate(bundle, "clean")
    mask_cross = _cross_requirement(
        "clean.__formal_mask_upstream",
        mask_formal["status"] == "PASS",
        "FORMAL_MASK_NOT_ADMITTED",
    )
    clean_donor = _aggregate(
        "clean_donor",
        [
            _gate(bundle, "clean", "input_mask_authority"),
            mask_cross,
            _gate(bundle, "clean", "donor_authority"),
            _gate(bundle, "clean", "donor_purity"),
            _gate(bundle, "clean", "source_map"),
            clean_scope,
            clean_auth,
            clean_contract,
        ],
    )
    clean_photometric = _aggregate(
        "clean_photometric",
        [
            _gate(bundle, "clean", "input_mask_authority"),
            mask_cross,
            _gate(bundle, "clean", "spatial_residual"),
            _gate(bundle, "clean", "illumination"),
            _gate(bundle, "clean", "shadow"),
            _gate(bundle, "clean", "seam"),
            _gate(bundle, "clean", "temporal"),
            _gate(bundle, "clean", "object_protection"),
            _gate(bundle, "clean", "codec"),
            _gate(bundle, "clean", "manual_review"),
            clean_scope,
            clean_auth,
            clean_contract,
        ],
    )
    clean_formal = _aggregate("clean_formal", [*clean_donor["gates"], *clean_photometric["gates"]])
    clean_donor = _category_view(
        clean_donor,
        [
            ("upstream_capture.clean_capture_input", clean_input["status"]),
            ("mask.formal", mask_formal["status"]),
        ],
    )
    clean_photometric = _category_view(
        clean_photometric,
        [
            ("upstream_capture.clean_capture_input", clean_input["status"]),
            ("mask.formal", mask_formal["status"]),
            ("clean_donor", clean_donor["status"]),
        ],
    )

    robot_scope = _scope_gate(bundle, "robot")
    robot_auth = _authorization_gate(bundle, "robot")
    robot_sidecar_auth = _robot_sidecar_authorization_gate(bundle)
    robot_contract = _contract_gate(bundle, "robot")
    hawor_cross = _cross_requirement(
        "robot.__formal_hawor_motion_upstream",
        hawor_motion["status"] == "PASS",
        "FORMAL_HAWOR_MOTION_NOT_ADMITTED",
    )
    clean_cross = _cross_requirement(
        "robot.__formal_clean_upstream",
        clean_formal["status"] == "PASS",
        "FORMAL_CLEAN_NOT_ADMITTED",
    )
    robot_sidecar = _aggregate(
        "robot_sidecar",
        [
            _gate(bundle, "robot", "input_hawor_authority"),
            hawor_cross,
            _gate(bundle, "robot", "retarget_schema"),
            _gate(bundle, "robot", "mano_reorder"),
            _gate(bundle, "robot", "pose"),
            _gate(bundle, "robot", "joint_limits"),
            _gate(bundle, "robot", "temporal"),
            _gate(bundle, "robot", "hand_morphology"),
            _gate(bundle, "robot", "retarget_independent_qa"),
            _gate(bundle, "robot", "functional_retarget_authority"),
            robot_scope,
            robot_sidecar_auth,
            robot_contract,
        ],
    )
    robot_authority = _aggregate(
        "robot_authority",
        [
            _gate(bundle, "robot", "input_hawor_authority"),
            hawor_cross,
            _gate(bundle, "robot", "mount_calibration"),
            _gate(bundle, "robot", "session_static_base"),
            _gate(bundle, "robot", "object6d"),
            _gate(bundle, "robot", "functional_retarget_authority"),
            robot_scope,
            robot_contract,
        ],
    )
    robot_geometry = _aggregate(
        "robot_geometry",
        [
            _gate(bundle, "robot", "pose"),
            _gate(bundle, "robot", "joint_limits"),
            _gate(bundle, "robot", "temporal"),
            _gate(bundle, "robot", "hand_morphology"),
            _gate(bundle, "robot", "contact"),
            _gate(bundle, "robot", "nonpenetration"),
            _gate(bundle, "robot", "self_collision"),
            _gate(bundle, "robot", "depth_occlusion"),
            _gate(bundle, "robot", "manual_review"),
            robot_scope,
            robot_contract,
        ],
    )
    robot_formal = _aggregate(
        "robot_formal",
        [
            *robot_authority["gates"],
            *robot_geometry["gates"],
            _gate(bundle, "robot", "input_clean_authority"),
            clean_cross,
            robot_auth,
        ],
    )
    robot_sidecar = _category_view(
        robot_sidecar,
        [
            (
                "upstream_capture.robot_motion_capture_input",
                robot_motion_input["status"],
            ),
            ("hawor_algorithm.motion_sidecar", hawor_motion["status"]),
        ],
    )
    robot_authority = _category_view(
        robot_authority,
        [
            (
                "upstream_capture.robot_composite_capture_input",
                robot_composite_input["status"],
            ),
            ("hawor_algorithm.motion_sidecar", hawor_motion["status"]),
        ],
    )
    robot_geometry = _category_view(
        robot_geometry,
        [
            ("robot_geometry.motion_sidecar", robot_sidecar["status"]),
            ("robot_authority", robot_authority["status"]),
            ("clean.formal", clean_formal["status"]),
        ],
    )
    robot_geometry["routes"] = {
        "motion_sidecar": robot_sidecar,
        "composite_geometry": {
            key: value for key, value in robot_geometry.items() if key != "routes"
        },
    }

    training_contract = _aggregate(
        "training_contract",
        [
            *(_gate(bundle, "training", gate) for gate in sorted(STAGE_GATES["training"])),
            _contract_gate(bundle, "training"),
        ],
    )
    expected_variant = PROFILE_VARIANT.get(profile)
    variant_gate = _cross_requirement(
        "training.__profile_image_variant_match",
        bundle.get("image_variant") == expected_variant and expected_variant is not None,
        "TRAINING_PROFILE_IMAGE_VARIANT_MISMATCH",
    )
    training_records = [*training_contract["gates"], variant_gate]
    training_base = _aggregate("training_contract_and_variant", training_records)
    training_dependencies: list[tuple[str, Mapping[str, Any]]] = [
        ("upstream_capture", capture_for_profile),
        ("robot_motion_capture", formal_robot_motion_input),
        ("hawor_algorithm", hawor_motion),
        ("robot_motion_sidecar", robot_sidecar),
    ]
    if profile in {"CLEAN_RGB_KAI22", "ROBOT_RGB_COMPOSITE_KAI22"}:
        training_dependencies.extend(
            [
                ("mask_semantic", mask_semantic),
                ("mask_temporal", mask_temporal),
                ("clean_donor", clean_donor),
                ("clean_photometric", clean_photometric),
            ]
        )
    if profile == "ROBOT_RGB_COMPOSITE_KAI22":
        training_dependencies.extend(
            [
                ("robot_authority", robot_authority),
                ("robot_composite_geometry", robot_geometry),
                ("robot_formal_admission", robot_formal),
            ]
        )
    dependency_gates = [
        _cross_requirement(
            f"training.__dependency_{name}",
            dependency["status"] == "PASS",
            f"DEPENDENCY_NOT_PASS_{name.upper()}",
        )
        for name, dependency in training_dependencies
    ]
    training_raw = _aggregate("training_eligibility", [*training_records, *dependency_gates])
    training_status = training_raw["status"]
    training_eligibility = dict(training_raw)
    training_eligibility["status"] = training_status
    training_eligibility["eligible"] = training_status == "PASS" and not input_errors
    training_eligibility["profile"] = profile
    training_eligibility["image_variant"] = bundle.get("image_variant")
    training_eligibility["dependencies"] = [
        {"route": name, "status": dependency["status"]}
        for name, dependency in training_dependencies
    ]

    categories = {
        "upstream_capture": upstream_capture,
        "hawor_algorithm": hawor_algorithm,
        "mask_semantic": mask_semantic,
        "mask_temporal": mask_temporal,
        "clean_donor": clean_donor,
        "clean_photometric": clean_photometric,
        "robot_geometry": robot_geometry,
        "robot_authority": robot_authority,
        "training_eligibility": training_eligibility,
    }

    mask_provider_nodes = {
        f"mask.provider.{slot}": {
            "category": "mask_semantic",
            "route": f"provider:{slot}",
            "value": item["admission"],
            "dependencies": list(item["graph_dependencies"]),
        }
        for slot, item in provider_admissions.items()
    }
    mask_node_dependencies = ["capture.mask_image", *mask_provider_nodes]

    # The product dependency graph is deliberately not a single stage order.
    # Capture/Object6D and Mask/Clean are parallel prerequisites for a Robot
    # composite, so an Object6D HOLD must not hide an independently observable
    # Mask or kinematic failure.
    blame_nodes: dict[str, dict[str, Any]] = {
        "capture.hawor_rgb": {
            "category": "upstream_capture",
            "route": "hawor_rgb_input",
            "value": formal_hawor_input,
            "dependencies": [],
        },
        "capture.mask_image": {
            "category": "upstream_capture",
            "route": "mask_image_input",
            "value": formal_mask_input,
            "dependencies": [],
        },
        "capture.pico_raw_point": {
            "category": "upstream_capture",
            "route": "pico_raw_point_input",
            "value": formal_pico_raw_point_input,
            "dependencies": [],
        },
        "capture.pico_left": {
            "category": "upstream_capture",
            "route": "pico_left_provider_input",
            "value": formal_pico_side_inputs["left"],
            "dependencies": [],
        },
        "capture.pico_right": {
            "category": "upstream_capture",
            "route": "pico_right_provider_input",
            "value": formal_pico_side_inputs["right"],
            "dependencies": [],
        },
        "capture.static_setup_anchor": {
            "category": "upstream_capture",
            "route": "static_setup_anchor_input",
            "value": formal_static_setup_anchor_input,
            "dependencies": [],
        },
        "capture.object6d_projection": {
            "category": "upstream_capture",
            "route": "object6d_projection_input",
            "value": formal_object6d_projection_input,
            "dependencies": [],
        },
        "capture.clean": {
            "category": "upstream_capture",
            "route": "clean_capture_input",
            "value": formal_clean_input,
            "dependencies": [],
        },
        "capture.robot_motion": {
            "category": "upstream_capture",
            "route": "robot_motion_capture_input",
            "value": formal_robot_motion_input,
            "dependencies": [],
        },
        "capture.robot_composite": {
            "category": "upstream_capture",
            "route": "robot_composite_capture_input",
            "value": formal_robot_composite_input,
            "dependencies": [],
        },
        "hawor.mask_seed": {
            "category": "hawor_algorithm",
            "route": "mask_seed",
            "value": hawor_mask_seed,
            "dependencies": ["capture.hawor_rgb"],
        },
        "hawor.motion": {
            "category": "hawor_algorithm",
            "route": "motion_sidecar",
            "value": hawor_motion,
            "dependencies": ["capture.hawor_rgb"],
        },
        **mask_provider_nodes,
        "mask.semantic": {
            "category": "mask_semantic",
            "route": "formal_semantic",
            "value": mask_semantic,
            "dependencies": list(mask_node_dependencies),
        },
        "mask.temporal": {
            "category": "mask_temporal",
            "route": "formal_temporal",
            "value": mask_temporal,
            "dependencies": list(mask_node_dependencies),
        },
        "mask.formal": {
            "category": "mask_semantic",
            "route": "formal_join",
            "value": mask_formal,
            "dependencies": ["mask.semantic", "mask.temporal"],
        },
        "clean.donor": {
            "category": "clean_donor",
            "route": "formal_donor",
            "value": clean_donor,
            "dependencies": ["capture.clean", "mask.formal"],
        },
        "clean.photometric": {
            "category": "clean_photometric",
            "route": "formal_photometric",
            "value": clean_photometric,
            "dependencies": ["capture.clean", "mask.formal", "clean.donor"],
        },
        "clean.formal": {
            "category": "clean_photometric",
            "route": "formal_join",
            "value": clean_formal,
            "dependencies": ["clean.donor", "clean.photometric"],
        },
        "robot.sidecar": {
            "category": "robot_geometry",
            "route": "motion_sidecar",
            "value": robot_sidecar,
            "dependencies": ["capture.robot_motion", "hawor.motion"],
        },
        "robot.authority": {
            "category": "robot_authority",
            "route": "composite_authority",
            "value": robot_authority,
            "dependencies": ["capture.robot_composite", "hawor.motion"],
        },
        "robot.geometry": {
            "category": "robot_geometry",
            "route": "composite_geometry",
            "value": robot_geometry,
            "dependencies": ["robot.sidecar", "robot.authority", "clean.formal"],
        },
        "robot.formal": {
            "category": "robot_authority",
            "route": "formal_admission",
            "value": robot_formal,
            "dependencies": ["robot.authority", "robot.geometry", "clean.formal"],
        },
        "training.base": {
            "category": "training_eligibility",
            "route": "contract_and_variant",
            "value": training_base,
            "dependencies": [],
        },
    }
    profile_targets = {
        "RAW_RGB_KAI22": ["robot.sidecar", "training.base"],
        "CLEAN_RGB_KAI22": ["clean.formal", "robot.sidecar", "training.base"],
        "ROBOT_RGB_COMPOSITE_KAI22": ["robot.formal", "training.base"],
    }.get(profile, ["training.base"])

    primary: list[dict[str, Any]] = []
    if input_errors:
        primary.append(
            {
                "category": "input_contract",
                "route": "schema_and_evidence",
                "status": "HOLD",
                "reason_codes": ["INVALID_OR_UNVERIFIED_INPUT_EVIDENCE"],
            }
        )
    else:
        visited: set[str] = set()
        frontier: set[str] = set()

        readiness_cache: dict[str, bool] = {}

        def node_ready(node_id: str) -> bool:
            if node_id in readiness_cache:
                return readiness_cache[node_id]
            node = blame_nodes[node_id]
            ready = node["value"]["status"] == "PASS" and all(
                node_ready(dependency) for dependency in node["dependencies"]
            )
            readiness_cache[node_id] = ready
            return ready

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            node = blame_nodes[node_id]
            blocked_dependencies = [
                dependency
                for dependency in node["dependencies"]
                if not node_ready(dependency)
            ]
            if blocked_dependencies:
                for dependency in blocked_dependencies:
                    visit(dependency)
                return
            if node["value"]["status"] != "PASS":
                frontier.add(node_id)

        for target in profile_targets:
            visit(target)
        for node_id, node in blame_nodes.items():
            if node_id not in frontier:
                continue
            value = node["value"]
            primary.append(
                {
                    "category": node["category"],
                    "route": node["route"],
                    "status": value["status"],
                    "reason_codes": sorted(
                        {
                            reason
                            for blocker in value["blockers"]
                            for reason in blocker["reason_codes"]
                        }
                    ),
                }
            )

    observed_failures = [name for name, category in categories.items() if category["status"] == "FAIL"]
    unresolved = [
        name
        for name, category in categories.items()
        if category["status"] in {"HOLD", "NOT_EVALUATED"}
    ]
    suppressed: list[dict[str, Any]] = []
    for name, category in categories.items():
        if category["status"] != "FAIL" or category.get("causal_status") != "BLOCKED_BY_UPSTREAM":
            continue
        suppressed.append(
            {
                "category": name,
                "observed_status": "FAIL",
                "causal_status": "BLOCKED_BY_UPSTREAM",
                "not_primary_because": [
                    dependency["route"]
                    for dependency in category.get("dependencies", [])
                    if dependency["status"] != "PASS"
                ],
            }
        )

    if input_errors:
        training_eligibility["status"] = "HOLD"
        training_eligibility["eligible"] = False
        training_eligibility["blockers"].append(
            {
                "gate": "input.__contract_and_evidence",
                "status": "HOLD",
                "observed_status": "HOLD",
                "reason_codes": ["INVALID_OR_UNVERIFIED_INPUT_EVIDENCE"],
            }
        )

    hawor_motion_ready = formal_hawor_input["allowed"] and hawor_motion["status"] == "PASS"
    mask_provider_ready = bool(provider_admissions) and all(
        item["admission"]["status"] == "PASS" for item in provider_admissions.values()
    )
    mask_formal_ready = (
        formal_mask_input["allowed"]
        and mask_provider_ready
        and mask_formal["status"] == "PASS"
    )
    clean_formal_ready = (
        formal_clean_input["allowed"] and mask_formal_ready and clean_formal["status"] == "PASS"
    )
    robot_sidecar_ready = (
        formal_robot_motion_input["allowed"]
        and hawor_motion_ready
        and robot_sidecar["status"] == "PASS"
    )
    robot_formal_ready = (
        formal_robot_composite_input["allowed"]
        and clean_formal_ready
        and robot_sidecar_ready
        and robot_authority["status"] == "PASS"
        and robot_geometry["status"] == "PASS"
        and robot_formal["status"] == "PASS"
    )

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "session_id": bundle.get("session_id"),
        "task_id": bundle.get("task_id"),
        "input_contract": {
            "valid": not input_errors,
            "schema_errors": validation_errors,
            "evidence_verification_errors": verification_errors,
        },
        "categories": categories,
        "routes": {
            "hawor_rgb_execution_allowed": hawor_input["allowed"] and not input_errors,
            "mask_execution_allowed": mask_input["allowed"] and not input_errors,
            "mask_formal_ready": mask_formal_ready and not input_errors,
            "clean_execution_allowed": (
                formal_clean_input["allowed"] and mask_formal_ready and not input_errors
            ),
            "clean_formal_ready": clean_formal_ready and not input_errors,
            "robot_motion_execution_allowed": (
                formal_robot_motion_input["allowed"] and hawor_motion_ready and not input_errors
            ),
            "robot_sidecar_ready": robot_sidecar_ready and not input_errors,
            "robot_composite_execution_allowed": (
                formal_robot_composite_input["allowed"]
                and hawor_motion_ready
                and clean_formal_ready
                and not input_errors
            ),
            "robot_formal_ready": robot_formal_ready and not input_errors,
            "humanego_train_eligible": training_eligibility["eligible"],
        },
        "blame": {
            "primary": primary,
            "observed_failures": observed_failures,
            "holds_or_not_evaluated": unresolved,
            "suppressed_downstream_failures": suppressed,
            "required_product_graph_targets": profile_targets,
            "policy": (
                "Primary blame is the non-PASS frontier of the selected product dependency DAG; "
                "parallel prerequisite failures can coexist. Explicit downstream FAIL remains visible, "
                "but is not causal until its declared upstream dependencies PASS. HOLD and "
                "NOT_EVALUATED never promote."
            ),
        },
    }
