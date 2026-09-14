"""Fail-closed selector and paired-cohort contracts for HumanEgo."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils.frozen_contract import (
    SHA256_PATTERN,
    load_frozen_split,
    read_file_reference,
    read_ordinary_file_bytes,
    sha256_file,
    validate_directory_path,
    validate_session_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SELECTOR_HOLD = "HOLD_MANIFEST_SELECTOR_REQUIRED"
SELECTOR_SCHEMA = "humanego-selector-manifest-v1"
PAIRED_SCHEMA = "humanego-paired-kept-manifest-v1"
PAIRED_TERMINAL_NAME = "PAIRED_KEPT_MANIFEST.json"
PAIRED_COMPLETE_PAYLOAD_ALIAS_NAME = ".PAIRED_KEPT_MANIFEST.complete_payload"
PAIRED_RAW_SELECTOR_NAME = "RAW_SELECTOR_MANIFEST.json"
PAIRED_ROBOT_SELECTOR_NAME = "ROBOT_RGB_SELECTOR_MANIFEST.json"
RAW_SELECTOR_STANDALONE_TERMINAL_NAME = "RAW_SELECTOR_MANIFEST.json"
RAW_SELECTOR_STANDALONE_ALIAS_NAME = ".RAW_SELECTOR_MANIFEST.complete_payload"
RAW_SELECTOR_STANDALONE_PUBLICATION_PROFILE = "STANDALONE_PERMANENT_HARDLINK_ALIAS_V1"
RAW_SELECTOR_STANDALONE_ROOT_PATTERN = re.compile(
    r"eligible68_raw_selector_[0-9]{8}_v[1-9][0-9]*"
)
PAIRED_TERMINAL_PUBLICATION_PROFILE = "PERMANENT_HARDLINK_ALIAS_V1"
ROBOT_RGB_LINEAGE_SCHEMA = "humanego-robotrgb-lineage-authority-v1"
ROBOT_RGB_STAGE_SCHEMA = "humanego-robotrgb-formal-stage-authority-v1"
ROBOT_RGB_LINEAGE_HOLD = "HOLD_ROBOT_RGB_LINEAGE_AUTHORITY_REQUIRED"
# Deliberately absent until an owner reviews and commits the exact aggregate
# publisher reference.  A selector-provided reference is not a trust anchor.
REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF: Mapping[str, Any] | None = None
ROBOT_RGB_STAGE_REFERENCE_FIELDS = {
    "SCENE_STATE": "scene_state_ref",
    "MOUNT_AUTHORITY": "mount_authority_ref",
    "EEVEE_RENDER": "eevee_render_ref",
    "CLEAN": "clean_ref",
    "S8": "s8_ref",
}
ROBOT_RGB_STAGE_UPSTREAM_STAGES = {
    "SCENE_STATE": (),
    "MOUNT_AUTHORITY": (),
    "EEVEE_RENDER": ("SCENE_STATE", "MOUNT_AUTHORITY"),
    "CLEAN": ("EEVEE_RENDER",),
    "S8": ("CLEAN",),
}
ELIGIBLE68_SPLIT_SCHEMA = "humanego-eligible68-split-v2"
ELIGIBLE68_SPLIT_PROTOCOL = "eligible68-52-8-8-v1"
DEV_SELECTION_NAMESPACE = "eligible68-dev-v1"
DEV_SELECTION_SEED = 7
DEV_SELECTION_ALGORITHM = "SHA256_UTF8_HEX_ASCENDING"
FINAL_TEST_POLICY = "ONE_SHOT_AFTER_RAW_AND_ROBOT_RGB_BEST_FROZEN"
FINAL_TEST_GATE_SCHEMA = "humanego-final-test-gate-v1"
FROZEN_BEST_SCHEMA = "humanego-frozen-best-v2"
PRODUCT_LINES = frozenset({"RAW", "ROBOT_RGB"})
SESSION_PATTERN = re.compile(r"grap_a_cap_[0-9]{3}")
FORBIDDEN_SOURCE_COMPONENTS = frozenset(
    {"labels", "blind", "盲测", "grap_a_cap_025", "processed"}
)

# Literal reviewed eligible68 population and frame denominator. Never discover
# either value by walking a directory: doing so could silently admit forbidden
# identities or let an incomplete producer redefine the cohort.
TRAIN60 = (
    "grap_a_cap_004",
    "grap_a_cap_005",
    "grap_a_cap_010",
    "grap_a_cap_021",
    "grap_a_cap_023",
    "grap_a_cap_026",
    "grap_a_cap_032",
    "grap_a_cap_034",
    "grap_a_cap_035",
    "grap_a_cap_039",
    "grap_a_cap_040",
    "grap_a_cap_043",
    "grap_a_cap_044",
    "grap_a_cap_045",
    "grap_a_cap_048",
    "grap_a_cap_051",
    "grap_a_cap_053",
    "grap_a_cap_056",
    "grap_a_cap_057",
    "grap_a_cap_058",
    "grap_a_cap_059",
    "grap_a_cap_060",
    "grap_a_cap_065",
    "grap_a_cap_068",
    "grap_a_cap_069",
    "grap_a_cap_071",
    "grap_a_cap_072",
    "grap_a_cap_073",
    "grap_a_cap_076",
    "grap_a_cap_079",
    "grap_a_cap_081",
    "grap_a_cap_085",
    "grap_a_cap_090",
    "grap_a_cap_091",
    "grap_a_cap_092",
    "grap_a_cap_094",
    "grap_a_cap_095",
    "grap_a_cap_102",
    "grap_a_cap_103",
    "grap_a_cap_106",
    "grap_a_cap_109",
    "grap_a_cap_114",
    "grap_a_cap_116",
    "grap_a_cap_118",
    "grap_a_cap_121",
    "grap_a_cap_122",
    "grap_a_cap_123",
    "grap_a_cap_124",
    "grap_a_cap_125",
    "grap_a_cap_132",
    "grap_a_cap_134",
    "grap_a_cap_137",
    "grap_a_cap_138",
    "grap_a_cap_139",
    "grap_a_cap_140",
    "grap_a_cap_141",
    "grap_a_cap_142",
    "grap_a_cap_144",
    "grap_a_cap_148",
    "grap_a_cap_157",
)
VALIDATION8 = (
    "grap_a_cap_002",
    "grap_a_cap_019",
    "grap_a_cap_024",
    "grap_a_cap_041",
    "grap_a_cap_050",
    "grap_a_cap_062",
    "grap_a_cap_101",
    "grap_a_cap_155",
)

# The upstream eligible68 authority called these roles train60/validation8.
# Training protocol v2 preserves that complete 68-session population but
# derives an eight-session development role exclusively from the original
# train60.  The original validation8 is never used for fitting, early stop,
# checkpoint selection or ordinary evaluation.
DEV8 = (
    "grap_a_cap_095",
    "grap_a_cap_092",
    "grap_a_cap_085",
    "grap_a_cap_056",
    "grap_a_cap_079",
    "grap_a_cap_114",
    "grap_a_cap_072",
    "grap_a_cap_118",
)
TRAIN52 = tuple(session_id for session_id in TRAIN60 if session_id not in DEV8)
FINAL_TEST8 = VALIDATION8
ROLE_SEMANTICS = {
    "train": "TRAIN_ONLY",
    "validation": "DEV_ONLY",
    "test": "FINAL_TEST_ONE_SHOT_ONLY",
}
DEV_SELECTION_CONTRACT = {
    "namespace": DEV_SELECTION_NAMESPACE,
    "seed": DEV_SELECTION_SEED,
    "algorithm": DEV_SELECTION_ALGORITHM,
    "expression": "sha256('eligible68-dev-v1|7|'+session_id)",
    "source_role": "original_train60",
    "count": 8,
}
ELIGIBLE68_ORDER = TRAIN60 + VALIDATION8
ELIGIBLE68 = frozenset(ELIGIBLE68_ORDER)
FORBIDDEN10 = frozenset(
    {
        "grap_a_cap_012",
        "grap_a_cap_025",
        "grap_a_cap_052",
        "grap_a_cap_055",
        "grap_a_cap_070",
        "grap_a_cap_087",
        "grap_a_cap_113",
        "grap_a_cap_119",
        "grap_a_cap_143",
        "grap_a_cap_149",
    }
)
ELIGIBLE68_FRAME_COUNTS = {
    "grap_a_cap_004": 460,
    "grap_a_cap_005": 391,
    "grap_a_cap_010": 386,
    "grap_a_cap_021": 431,
    "grap_a_cap_023": 407,
    "grap_a_cap_026": 342,
    "grap_a_cap_032": 436,
    "grap_a_cap_034": 402,
    "grap_a_cap_035": 369,
    "grap_a_cap_039": 373,
    "grap_a_cap_040": 359,
    "grap_a_cap_043": 362,
    "grap_a_cap_044": 359,
    "grap_a_cap_045": 354,
    "grap_a_cap_048": 594,
    "grap_a_cap_051": 376,
    "grap_a_cap_053": 429,
    "grap_a_cap_056": 440,
    "grap_a_cap_057": 419,
    "grap_a_cap_058": 432,
    "grap_a_cap_059": 391,
    "grap_a_cap_060": 378,
    "grap_a_cap_065": 406,
    "grap_a_cap_068": 455,
    "grap_a_cap_069": 399,
    "grap_a_cap_071": 434,
    "grap_a_cap_072": 390,
    "grap_a_cap_073": 414,
    "grap_a_cap_076": 451,
    "grap_a_cap_079": 444,
    "grap_a_cap_081": 451,
    "grap_a_cap_085": 464,
    "grap_a_cap_090": 415,
    "grap_a_cap_091": 425,
    "grap_a_cap_092": 415,
    "grap_a_cap_094": 402,
    "grap_a_cap_095": 386,
    "grap_a_cap_102": 478,
    "grap_a_cap_103": 429,
    "grap_a_cap_106": 438,
    "grap_a_cap_109": 421,
    "grap_a_cap_114": 441,
    "grap_a_cap_116": 433,
    "grap_a_cap_118": 455,
    "grap_a_cap_121": 455,
    "grap_a_cap_122": 462,
    "grap_a_cap_123": 413,
    "grap_a_cap_124": 459,
    "grap_a_cap_125": 449,
    "grap_a_cap_132": 434,
    "grap_a_cap_134": 471,
    "grap_a_cap_137": 384,
    "grap_a_cap_138": 418,
    "grap_a_cap_139": 429,
    "grap_a_cap_140": 424,
    "grap_a_cap_141": 423,
    "grap_a_cap_142": 406,
    "grap_a_cap_144": 416,
    "grap_a_cap_148": 377,
    "grap_a_cap_157": 419,
    "grap_a_cap_002": 393,
    "grap_a_cap_019": 375,
    "grap_a_cap_024": 305,
    "grap_a_cap_041": 406,
    "grap_a_cap_050": 298,
    "grap_a_cap_062": 407,
    "grap_a_cap_101": 517,
    "grap_a_cap_155": 389,
}


def _dev_identity_digest(session_id: str) -> str:
    payload = f"{DEV_SELECTION_NAMESPACE}|{DEV_SELECTION_SEED}|{session_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if (
    tuple(sorted(TRAIN60, key=lambda value: (_dev_identity_digest(value), value))[:8])
    != DEV8
):
    raise RuntimeError("literal DEV8 differs from the predeclared identity-hash rule")
if len(TRAIN52) != 52 or len(DEV8) != 8 or len(FINAL_TEST8) != 8:
    raise RuntimeError("eligible68 role cardinality drift")
if set(TRAIN52) | set(DEV8) | set(FINAL_TEST8) != ELIGIBLE68:
    raise RuntimeError("eligible68 52/8/8 partition drift")


def eligible68_split_protocol_fields() -> dict[str, Any]:
    """Return the immutable metadata required on a v2 split manifest."""
    return {
        "cohort": "eligible68",
        "cohort_frame_count": 28_265,
        "split_protocol": ELIGIBLE68_SPLIT_PROTOCOL,
        "role_semantics": dict(ROLE_SEMANTICS),
        "dev_selection": dict(DEV_SELECTION_CONTRACT),
        "final_test_policy": FINAL_TEST_POLICY,
    }


def training_evaluation_protocol_fields() -> dict[str, Any]:
    """Return the run-manifest fields that keep final-test data withheld."""
    return {
        "schema_version": "humanego-training-evaluation-protocol-v1",
        "split_protocol": ELIGIBLE68_SPLIT_PROTOCOL,
        "train_role": "train52",
        "selection_role": "dev8",
        "final_test_role": "final_test8",
        "final_test_access": "WITHHELD",
        "final_test_policy": FINAL_TEST_POLICY,
    }


def validate_eligible68_role_contract(split: Mapping[str, Any]) -> None:
    """Validate the exact 52 train / 8 dev / 8 final-test partition."""
    train = split.get("train")
    validation = split.get("validation")
    test = split.get("test")
    if train != list(TRAIN52) or validation != list(DEV8) or test != list(FINAL_TEST8):
        observed = {
            value
            for role in (train, validation, test)
            if isinstance(role, list)
            for value in role
            if isinstance(value, str)
        }
        forbidden = sorted(observed & FORBIDDEN10)
        raise ValueError(
            "split session cohort is not exact eligible68 train52/dev8/final-test8; "
            f"forbidden={forbidden}"
        )
    for key, expected in eligible68_split_protocol_fields().items():
        if split.get(key) != expected:
            raise ValueError(f"eligible68 split {key} drift")


def validate_training_run_role_contract(
    run_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    allow_overfit: bool = False,
) -> None:
    """Reject runs that train/select on anything outside train52/dev8."""
    validate_eligible68_role_contract(split)
    if run_manifest.get("train_sessions") != list(TRAIN52):
        raise ValueError("formal run train sessions are not exact train52")
    if run_manifest.get("validation_sessions") != list(DEV8):
        raise ValueError("formal run selection sessions are not exact dev8")
    if run_manifest.get("evaluation_protocol") != training_evaluation_protocol_fields():
        raise ValueError("formal run training/evaluation protocol drift")
    adapter_paths = run_manifest.get("adapter_paths")
    if not isinstance(adapter_paths, Mapping) or set(adapter_paths) != set(
        TRAIN52 + DEV8
    ):
        raise ValueError("formal run adapter path roles are not exact train52/dev8")
    if any(
        not isinstance(adapter_paths[session_id], str) for session_id in adapter_paths
    ):
        raise ValueError("formal run adapter paths must be strings")
    resolved_config = run_manifest.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("formal run has no resolved config for role binding")
    overfit_session = run_manifest.get("overfit_session")
    if overfit_session is None:
        expected_train_paths = [adapter_paths[session_id] for session_id in TRAIN52]
        expected_dev_paths = [adapter_paths[session_id] for session_id in DEV8]
    else:
        if not allow_overfit:
            raise ValueError(
                "overfit runs cannot enter snapshot/freeze/formal evaluation"
            )
        if not isinstance(overfit_session, str) or overfit_session not in TRAIN52:
            raise ValueError("overfit session must be exactly one train52 session")
        expected_train_paths = [adapter_paths[overfit_session]]
        expected_dev_paths = [adapter_paths[overfit_session]]
    if resolved_config.get("MPS_PATHS_TRAIN") != expected_train_paths:
        raise ValueError("resolved training paths are not the exact ordered role paths")
    if resolved_config.get("MPS_PATHS_EVAL") != expected_dev_paths:
        raise ValueError(
            "resolved evaluation paths are not the exact ordered role paths"
        )
    forbidden_result_fields = {
        "final_test_metrics",
        "final_test_results",
        "final_test_score",
    }
    leaked = sorted(forbidden_result_fields & set(run_manifest))
    if leaked:
        raise ValueError(f"training run contains final-test results: {leaked}")


def validate_eligible68_split_phase1(
    split_path: str | Path,
    embodiment: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate only the split manifest before any session source is opened.

    The current legacy 62/8/8 manifests contain forbidden identities. This
    phase therefore precedes ``load_frozen_split(..., verify_sidecars=True)``
    in every formal consumer.
    """
    project_root = Path(project_root).resolve(strict=True)
    split_root = project_root / "HumanEgo" / "data_manifests"
    lexical_path = Path(split_path).absolute()
    if not lexical_path.is_relative_to(split_root):
        raise ValueError(
            "eligible68 split must be versioned under HumanEgo/data_manifests"
        )
    path, encoded = read_ordinary_file_bytes(
        lexical_path,
        label="eligible68 split phase-1 manifest",
    )
    if not path.is_relative_to(split_root):
        raise ValueError(
            "eligible68 split must be versioned under HumanEgo/data_manifests"
        )
    try:
        split = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("eligible68 split is not valid JSON") from error
    if not isinstance(split, dict):
        raise ValueError("eligible68 split must be a JSON object")

    validate_eligible68_role_contract(split)
    if split.get("schema_version") != ELIGIBLE68_SPLIT_SCHEMA:
        raise ValueError("eligible68 split schema drift")
    if split.get("immutable") is not True or split.get("no_fallback") is not True:
        raise ValueError("eligible68 split lacks immutable/no-fallback guarantees")

    sessions = split.get("sessions")
    if not isinstance(sessions, dict) or set(sessions) != ELIGIBLE68:
        raise ValueError("eligible68 split sessions metadata is not exact")
    production_value = split.get("production_root")
    sidecar_value = split.get("sidecar_root")
    if (
        not isinstance(production_value, str)
        or not Path(production_value).is_absolute()
    ):
        raise ValueError(
            "eligible68 split production_root must be canonical and absolute"
        )
    if not isinstance(sidecar_value, str) or not Path(sidecar_value).is_absolute():
        raise ValueError("eligible68 split sidecar_root must be canonical and absolute")
    _reject_forbidden_source_path(
        production_value,
        label="eligible68 production root",
    )
    _reject_forbidden_source_path(sidecar_value, label="eligible68 sidecar root")
    production_root = validate_directory_path(
        production_value,
        allowed_roots=[project_root],
        label="eligible68 production root",
    )
    sidecar_root = validate_directory_path(
        sidecar_value, allowed_roots=[project_root], label="eligible68 sidecar root"
    )
    if production_root.is_relative_to(
        project_root / "_run"
    ) or sidecar_root.is_relative_to(project_root / "_run"):
        raise ValueError("eligible68 split roots cannot come from _run staging")

    for session_id in ELIGIBLE68_ORDER:
        row = sessions[session_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"eligible68 split session row is invalid: {session_id}")
        adapter = row.get("adapter")
        expected_adapter = production_root / session_id / "09_humanego_adapter"
        if not isinstance(adapter, str) or Path(adapter) != expected_adapter:
            raise ValueError(f"eligible68 adapter declaration drift: {session_id}")
        status_digest = row.get("production_status_sha256")
        if not isinstance(status_digest, str) or not SHA256_PATTERN.fullmatch(
            status_digest
        ):
            raise ValueError(
                f"eligible68 production status declaration drift: {session_id}"
            )
        try:
            embodiment_row = row["embodiments"][embodiment]
            frame_count = embodiment_row["frames"]
            sidecar_digest = embodiment_row["sidecar_sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"eligible68 split lacks {embodiment} declaration: {session_id}"
            ) from error
        if frame_count != ELIGIBLE68_FRAME_COUNTS[session_id]:
            raise ValueError(f"eligible68 frame denominator drift: {session_id}")
        if not isinstance(sidecar_digest, str) or not SHA256_PATTERN.fullmatch(
            sidecar_digest
        ):
            raise ValueError(f"eligible68 sidecar declaration drift: {session_id}")
    return split


def load_eligible68_frozen_split(
    split_path: str | Path,
    embodiment: str,
    *,
    sidecar_root: str | Path | None = None,
    verify_sidecars: bool = False,
    verify_roles: tuple[str, ...] | None = None,
    allow_final_test: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Two-phase split load with final-test sidecars withheld by default."""
    split = validate_eligible68_split_phase1(
        split_path, embodiment, project_root=project_root
    )
    return verify_eligible68_split_phase2(
        split_path,
        split,
        embodiment,
        sidecar_root=sidecar_root,
        verify_sidecars=verify_sidecars,
        verify_roles=verify_roles,
        allow_final_test=allow_final_test,
    )


def verify_eligible68_split_phase2(
    split_path: str | Path,
    split: Mapping[str, Any],
    embodiment: str,
    *,
    sidecar_root: str | Path | None = None,
    verify_sidecars: bool = False,
    verify_roles: tuple[str, ...] | None = None,
    allow_final_test: bool = False,
) -> dict[str, Any]:
    """Verify selected sidecar roles from an already accepted phase-1 payload."""
    validate_eligible68_role_contract(split)
    if verify_roles is None:
        verify_roles = ("train", "validation") if verify_sidecars else ()
    if not verify_sidecars and verify_roles:
        raise ValueError("verify_roles requires verify_sidecars=True")
    unknown_roles = sorted(set(verify_roles) - {"train", "validation", "test"})
    if unknown_roles:
        raise ValueError(f"unknown eligible68 verification roles: {unknown_roles}")
    if "test" in verify_roles and not allow_final_test:
        raise RuntimeError(
            "HOLD_FINAL_TEST_GATE_REQUIRED: final-test sidecars are withheld"
        )
    return load_frozen_split(
        split_path,
        embodiment,
        sidecar_root=sidecar_root,
        verify_sidecars=verify_sidecars,
        verify_roles=verify_roles,
        split_payload=split,
    )


def _load_reference(
    reference: Mapping[str, Any] | None,
    *,
    project_root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if reference is None:
        raise RuntimeError(f"{FORMAL_SELECTOR_HOLD}: missing {label}")
    raw_path = reference.get("path") if isinstance(reference, Mapping) else None
    if isinstance(raw_path, str):
        _reject_forbidden_source_path(raw_path, label=label)
    path, encoded = read_file_reference(
        reference, allowed_roots=[project_root], label=label
    )
    if path.is_relative_to(project_root / "_run"):
        raise ValueError(f"{label} cannot be consumed from _run staging")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return path, payload


def _validate_common(payload: Mapping[str, Any], schema: str, label: str) -> None:
    if payload.get("schema_version") != schema:
        raise ValueError(f"{label} schema drift")
    if payload.get("immutable") is not True or payload.get("no_fallback") is not True:
        raise ValueError(f"{label} lacks immutable/no-fallback guarantees")


def _reject_forbidden_source_path(path: str | Path, *, label: str) -> None:
    components = {part.casefold() for part in Path(path).parts}
    forbidden = sorted(components & FORBIDDEN_SOURCE_COMPONENTS)
    if forbidden:
        raise ValueError(f"{label} enters a forbidden source namespace: {forbidden}")


def _validate_frame_key(value: Any, *, label: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{5}", value):
        raise ValueError(f"{label} frame key must be five decimal digits")
    return int(value)


def _validate_declared_file_reference(value: Any, *, label: str) -> None:
    """Validate nested reference syntax without reopening every image here."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} reference must be an object")
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"{label} reference path must be canonical and absolute")
    _reject_forbidden_source_path(path, label=label)
    if Path(path) != Path(path).resolve(strict=False):
        raise ValueError(f"{label} reference path is not canonical")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{label} reference bytes must be positive")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} reference SHA256 is invalid")


def _validate_exact_lineage_reference(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} must be an exact path/bytes/SHA256 reference")
    _validate_declared_file_reference(value, label=label)


def _reject_robot_lineage_path(
    path: str | Path,
    *,
    project_root: Path,
    label: str,
) -> None:
    """Keep every external lineage edge out of staging and forbidden domains."""
    lexical = Path(path).absolute()
    components = {part.casefold() for part in lexical.parts}
    forbidden = components & (
        FORBIDDEN_SOURCE_COMPONENTS
        | {session_id.casefold() for session_id in FORBIDDEN10}
    )
    if forbidden:
        raise ValueError(
            f"{label} enters a forbidden RobotRGB lineage namespace: "
            f"{sorted(forbidden)}"
        )
    if "_run" in components or lexical.is_relative_to(project_root / "_run"):
        raise ValueError(f"{label} cannot come from _run staging")


def _register_lineage_path(
    path: Path,
    *,
    owner: str,
    owners: dict[tuple[Any, ...], str],
) -> None:
    key = ("path", str(path))
    previous = owners.get(key)
    if previous is not None and previous != owner:
        raise ValueError(
            f"RobotRGB lineage external path aliases logical records: "
            f"{previous}, {owner}"
        )
    owners[key] = owner


def _register_lineage_identity(
    binding: Mapping[str, Any],
    *,
    owner: str,
    owners: dict[tuple[Any, ...], str],
) -> None:
    key = ("inode", binding.get("st_dev"), binding.get("st_ino"))
    previous = owners.get(key)
    if previous is not None and previous != owner:
        raise ValueError(
            f"RobotRGB lineage external inode aliases logical records: "
            f"{previous}, {owner}"
        )
    owners[key] = owner


def _read_robot_lineage_reference(
    reference: Any,
    *,
    project_root: Path,
    label: str,
    owner: str,
    owners: dict[tuple[Any, ...], str],
    snapshots: list[tuple[dict[str, Any], str, dict[str, Any]]],
) -> tuple[Path, bytes]:
    """Read and bind one external lineage edge around the same-FD reader."""
    _validate_exact_lineage_reference(reference, label=label)
    normalized_reference = dict(reference)
    lexical = Path(str(normalized_reference["path"])).absolute()
    _reject_robot_lineage_path(lexical, project_root=project_root, label=label)
    _register_lineage_path(lexical, owner=owner, owners=owners)
    before = _ordinary_file_binding(lexical, label=f"{label} pre-read binding")
    parent = validate_directory_path(
        lexical.parent,
        allowed_roots=[lexical.anchor],
        label=f"{label} parent",
    )
    path, encoded = read_file_reference(
        normalized_reference,
        allowed_roots=[parent],
        label=label,
    )
    after = _ordinary_file_binding(path, label=f"{label} post-read binding")
    if before != after:
        raise ValueError(f"{label} path changed around its verified read")
    _register_lineage_identity(after, owner=owner, owners=owners)
    snapshots.append((normalized_reference, label, after))
    return path, encoded


def _read_robot_lineage_json_reference(
    reference: Any,
    *,
    project_root: Path,
    label: str,
    owner: str,
    owners: dict[tuple[Any, ...], str],
    snapshots: list[tuple[dict[str, Any], str, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    path, encoded = _read_robot_lineage_reference(
        reference,
        project_root=project_root,
        label=label,
        owner=owner,
        owners=owners,
        snapshots=snapshots,
    )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return path, payload


_FORMAL_ROBOT_RGB_STATE = {
    "immutable": True,
    "no_fallback": True,
    "status": "ARTIFACT_EXISTS",
    "completion_mode": "ARTIFACT_EXISTS",
    "formal_consumer_allowed": True,
    "development_only": False,
    "provisional": False,
    "visual_only": False,
    "next_bucket_blocked": False,
    "fallback_used": False,
    "candidate_requires_human_review": False,
    "advancement_authorized": True,
    "baseline_frozen": True,
    "synthetic_fixture": False,
}


def _validate_formal_robot_rgb_state(value: Mapping[str, Any], *, label: str) -> None:
    for field, expected in _FORMAL_ROBOT_RGB_STATE.items():
        observed = value.get(field)
        if (
            isinstance(expected, bool)
            and observed is not expected
            or not isinstance(expected, bool)
            and observed != expected
        ):
            raise ValueError(f"{label} {field} is not formal-consumable")


def _validate_robot_rgb_stage(
    reference: Any,
    *,
    stage: str,
    session_id: str,
    expected_frame_keys: list[str],
    selector_frames: Mapping[str, Mapping[str, Any]],
    expected_upstream_references: Mapping[str, Mapping[str, Any]],
    project_root: Path,
    owners: dict[tuple[Any, ...], str],
    snapshots: list[tuple[dict[str, Any], str, dict[str, Any]]],
) -> None:
    label = f"RobotRGB {session_id} {stage} authority"
    _, payload = _read_robot_lineage_json_reference(
        reference,
        project_root=project_root,
        label=label,
        owner=f"{session_id}/{stage}/authority",
        owners=owners,
        snapshots=snapshots,
    )
    common_fields = {
        "schema_version",
        "lineage_stage",
        *_FORMAL_ROBOT_RGB_STATE,
        "session_id",
        "frame_count",
        "success_evidence",
    }
    stage_fields = {
        "SCENE_STATE": {
            "scene_state_mode",
            "q_arm_and_camera_base_authoritative",
        },
        "MOUNT_AUTHORITY": {
            "mount_provenance",
            "independent_measurement",
            "selected_by_ik_residual",
        },
        "EEVEE_RENDER": {"engine"},
        "CLEAN": {"clean_authority"},
        "S8": {"s8_authority", "robot_rgb_outputs"},
    }[stage]
    upstream_fields = {
        ROBOT_RGB_STAGE_REFERENCE_FIELDS[upstream_stage]
        for upstream_stage in ROBOT_RGB_STAGE_UPSTREAM_STAGES[stage]
    }
    if set(expected_upstream_references) != upstream_fields:
        raise ValueError(f"{label} validator upstream schema drift")
    if set(payload) != common_fields | stage_fields | upstream_fields:
        raise ValueError(f"{label} schema fields drift")
    if payload.get("schema_version") != ROBOT_RGB_STAGE_SCHEMA:
        raise ValueError(f"{label} schema drift")
    if payload.get("lineage_stage") != stage:
        raise ValueError(f"{label} lineage stage mismatch")
    _validate_formal_robot_rgb_state(payload, label=label)
    if payload.get("session_id") != session_id:
        raise ValueError(f"{label} session mismatch")
    frame_count = payload.get("frame_count")
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count != len(expected_frame_keys)
    ):
        raise ValueError(f"{label} frame count mismatch")

    if stage == "SCENE_STATE" and (
        payload.get("scene_state_mode") != "EXTERNAL_FORMAL_AUTHORITY"
        or payload.get("q_arm_and_camera_base_authoritative") is not True
    ):
        raise ValueError(f"{label} lacks an authoritative external scene state")
    if stage == "MOUNT_AUTHORITY" and (
        payload.get("mount_provenance") != "INDEPENDENT_BILATERAL_FORMAL_MOUNT"
        or payload.get("independent_measurement") is not True
        or payload.get("selected_by_ik_residual") is not False
    ):
        raise ValueError(f"{label} lacks an independent formal mount")
    if stage == "EEVEE_RENDER" and payload.get("engine") != "BLENDER_EEVEE_NEXT":
        raise ValueError(f"{label} is not the frozen EEVEE renderer")
    if stage == "CLEAN" and payload.get("clean_authority") != "FORMAL_CLEAN":
        raise ValueError(f"{label} is not formal CLEAN")
    if stage == "S8" and payload.get("s8_authority") != "FORMAL_S8_ROBOT_RGB":
        raise ValueError(f"{label} is not formal S8 RobotRGB")

    for field, expected_reference in expected_upstream_references.items():
        observed_reference = payload.get(field)
        _validate_exact_lineage_reference(
            observed_reference,
            label=f"{label} {field}",
        )
        if observed_reference != expected_reference:
            raise ValueError(
                f"{label} upstream {field} differs from aggregate authority"
            )

    evidence = payload.get("success_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{label} success evidence must be non-empty")
    for index, item in enumerate(evidence):
        _validate_exact_lineage_reference(
            item,
            label=f"{label} success_evidence[{index}]",
        )

    if stage == "S8":
        outputs = payload.get("robot_rgb_outputs")
        if not isinstance(outputs, Mapping) or list(outputs) != expected_frame_keys:
            raise ValueError(f"{label} RobotRGB output frame ledger mismatch")
        ordered_outputs = [outputs[frame_key] for frame_key in expected_frame_keys]
        if evidence != ordered_outputs:
            raise ValueError(
                f"{label} success evidence does not exactly cover RobotRGB outputs"
            )
        for frame_key, output_ref in zip(
            expected_frame_keys, ordered_outputs, strict=True
        ):
            selector_ref = selector_frames[frame_key]["image"]
            if _reference_identity(selector_ref) != _reference_identity(output_ref):
                raise ValueError(
                    f"RobotRGB selector/S8 output differs: {session_id}/{frame_key}"
                )
            if selector_frames[frame_key]["unresolved"] is not False:
                raise ValueError(
                    f"formal RobotRGB output remains unresolved: {session_id}/{frame_key}"
                )
            _read_robot_lineage_reference(
                output_ref,
                project_root=project_root,
                label=f"{label} output {frame_key}",
                owner=f"{session_id}/{frame_key}/ROBOT_RGB",
                owners=owners,
                snapshots=snapshots,
            )
    else:
        for index, evidence_ref in enumerate(evidence):
            _read_robot_lineage_reference(
                evidence_ref,
                project_root=project_root,
                label=f"{label} success evidence {index}",
                owner=f"{session_id}/{stage}/evidence/{index}",
                owners=owners,
                snapshots=snapshots,
            )


def _revalidate_robot_lineage_snapshots(
    snapshots: list[tuple[dict[str, Any], str, dict[str, Any]]],
) -> None:
    """Reject leaf or ancestor replacement before returning a formal binding."""
    for reference, label, accepted_binding in snapshots:
        lexical = Path(str(reference["path"]))
        before = _ordinary_file_binding(
            lexical,
            label=f"{label} terminal binding",
        )
        if before != accepted_binding:
            raise ValueError(f"{label} path changed before lineage completion")
        parent = validate_directory_path(
            lexical.parent,
            allowed_roots=[lexical.anchor],
            label=f"{label} terminal parent",
        )
        path, _ = read_file_reference(
            reference,
            allowed_roots=[parent],
            label=f"{label} terminal recheck",
        )
        after = _ordinary_file_binding(
            path,
            label=f"{label} terminal post-read binding",
        )
        if after != accepted_binding:
            raise ValueError(f"{label} path changed during lineage completion")


def _robot_lineage_session_order() -> tuple[str, ...]:
    frozen = tuple(
        session_id for session_id in ELIGIBLE68_ORDER if session_id in ELIGIBLE68
    )
    if len(frozen) == len(ELIGIBLE68) and set(frozen) == ELIGIBLE68:
        return frozen
    # Unit tests may replace the cohort with a deliberately tiny closed world.
    return tuple(sorted(ELIGIBLE68))


def _validate_robot_rgb_lineage_authority(
    reference: Any,
    *,
    selector_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    project_root: Path,
) -> dict[str, Any]:
    _validate_exact_lineage_reference(
        reference,
        label="RobotRGB selector aggregate lineage authority",
    )
    reviewed_reference = REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF
    if reviewed_reference is None:
        raise RuntimeError(
            f"{ROBOT_RGB_LINEAGE_HOLD}: no owner-reviewed publisher authority "
            "is configured"
        )
    _validate_exact_lineage_reference(
        reviewed_reference,
        label="owner-reviewed RobotRGB aggregate lineage authority",
    )
    if dict(reference) != dict(reviewed_reference):
        raise RuntimeError(
            f"{ROBOT_RGB_LINEAGE_HOLD}: selector aggregate lineage authority "
            "differs from the owner-reviewed publisher authority"
        )
    owners: dict[tuple[Any, ...], str] = {}
    snapshots: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    _, authority = _read_robot_lineage_json_reference(
        reference,
        project_root=project_root,
        label="RobotRGB aggregate lineage authority",
        owner="aggregate-lineage-authority",
        owners=owners,
        snapshots=snapshots,
    )
    expected_fields = {
        "schema_version",
        *_FORMAL_ROBOT_RGB_STATE,
        "cohort",
        "session_count",
        "frame_count",
        "success_evidence",
        "sessions",
    }
    if set(authority) != expected_fields:
        raise ValueError("RobotRGB aggregate lineage authority schema fields drift")
    if authority.get("schema_version") != ROBOT_RGB_LINEAGE_SCHEMA:
        raise ValueError("RobotRGB aggregate lineage authority schema drift")
    _validate_formal_robot_rgb_state(
        authority,
        label="RobotRGB aggregate lineage authority",
    )
    if authority.get("cohort") != "eligible68":
        raise ValueError("RobotRGB lineage cohort must be eligible68")
    session_count = authority.get("session_count")
    if (
        not isinstance(session_count, int)
        or isinstance(session_count, bool)
        or session_count != len(ELIGIBLE68)
    ):
        raise ValueError("RobotRGB lineage session count must be exact eligible68")
    expected_total_frames = sum(ELIGIBLE68_FRAME_COUNTS.values())
    total_frame_count = authority.get("frame_count")
    if (
        not isinstance(total_frame_count, int)
        or isinstance(total_frame_count, bool)
        or total_frame_count != expected_total_frames
    ):
        raise ValueError("RobotRGB lineage frame count must be exact 28265")
    sessions = authority.get("sessions")
    if not isinstance(sessions, Mapping) or set(sessions) != ELIGIBLE68:
        raise ValueError("RobotRGB lineage sessions must be exact eligible68")

    expected_authority_evidence: list[Any] = []
    session_order = _robot_lineage_session_order()
    for session_id in session_order:
        row = sessions[session_id]
        expected_row_fields = {
            "frame_count",
            *ROBOT_RGB_STAGE_REFERENCE_FIELDS.values(),
        }
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError(f"RobotRGB lineage session schema drift: {session_id}")
        expected_count = ELIGIBLE68_FRAME_COUNTS[session_id]
        row_frame_count = row.get("frame_count")
        if (
            not isinstance(row_frame_count, int)
            or isinstance(row_frame_count, bool)
            or row_frame_count != expected_count
        ):
            raise ValueError(
                f"RobotRGB lineage session frame count mismatch: {session_id}"
            )
        for field in ROBOT_RGB_STAGE_REFERENCE_FIELDS.values():
            _validate_exact_lineage_reference(
                row[field],
                label=f"RobotRGB {session_id} {field}",
            )
            expected_authority_evidence.append(row[field])
    authority_evidence = authority.get("success_evidence")
    if (
        not isinstance(authority_evidence, list)
        or not authority_evidence
        or authority_evidence != expected_authority_evidence
    ):
        raise ValueError(
            "RobotRGB aggregate success evidence must exactly bind all stage authorities"
        )

    for session_id in session_order:
        frame_keys = [
            f"{frame_index:05d}"
            for frame_index in range(ELIGIBLE68_FRAME_COUNTS[session_id])
        ]
        selector_frames = selector_records[session_id]
        for stage, field in ROBOT_RGB_STAGE_REFERENCE_FIELDS.items():
            expected_upstream_references = {
                ROBOT_RGB_STAGE_REFERENCE_FIELDS[upstream_stage]: sessions[session_id][
                    ROBOT_RGB_STAGE_REFERENCE_FIELDS[upstream_stage]
                ]
                for upstream_stage in ROBOT_RGB_STAGE_UPSTREAM_STAGES[stage]
            }
            _validate_robot_rgb_stage(
                sessions[session_id][field],
                stage=stage,
                session_id=session_id,
                expected_frame_keys=frame_keys,
                selector_frames=selector_frames,
                expected_upstream_references=expected_upstream_references,
                project_root=project_root,
                owners=owners,
                snapshots=snapshots,
            )
    _revalidate_robot_lineage_snapshots(snapshots)
    return dict(authority)


def _is_standalone_raw_selector_wrapper_member(path: Path, project_root: Path) -> bool:
    """Recognize any member of the reserved standalone RAW namespace."""
    return (
        path.parent.parent == project_root / "HumanEgo" / "outputs"
        and RAW_SELECTOR_STANDALONE_ROOT_PATTERN.fullmatch(path.parent.name) is not None
    )


def _is_standalone_raw_selector_terminal(path: Path, project_root: Path) -> bool:
    """Recognize the sole consumer-facing member of the reserved wrapper."""
    return (
        path.name == RAW_SELECTOR_STANDALONE_TERMINAL_NAME
        and _is_standalone_raw_selector_wrapper_member(path, project_root)
    )


def _standalone_stat_binding(
    state: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
        int(state.st_nlink),
        int(state.st_mode),
    )


def _load_standalone_raw_selector_terminal(
    reference: Mapping[str, Any], *, project_root: Path
) -> tuple[
    Path,
    dict[str, Any],
    tuple[
        tuple[int, int, int, int, int, int, int],
        tuple[int, int, int, int, int, int, int],
    ],
]:
    """Load the terminal-last RAW hardlink pair through its held root FD."""
    _validate_exact_lineage_reference(
        reference, label="standalone RAW selector terminal"
    )
    path = Path(str(reference["path"])).absolute()
    if (
        Path(str(reference["path"])) != path
        or path != Path(os.path.normpath(path))
        or not (_is_standalone_raw_selector_terminal(path, project_root))
    ):
        raise ValueError("standalone RAW selector terminal path/profile drift")
    _reject_forbidden_source_path(path, label="standalone RAW selector terminal")
    parent = path.parent
    parent_descriptor = _open_directory_chain_nofollow(
        parent, label="standalone RAW selector output root"
    )
    terminal_descriptor: int | None = None
    alias_descriptor: int | None = None
    try:
        parent_before = os.fstat(parent_descriptor)
        parent_named = os.lstat(parent)
        parent_identity = (parent_before.st_dev, parent_before.st_ino)
        if (
            not stat.S_ISDIR(parent_before.st_mode)
            or not stat.S_ISDIR(parent_named.st_mode)
            or stat.S_IMODE(parent_before.st_mode) != 0o555
            or stat.S_IMODE(parent_named.st_mode) != 0o555
            or (parent_named.st_dev, parent_named.st_ino) != parent_identity
        ):
            raise ValueError("standalone RAW selector root identity/mode drift")
        expected_inventory = {
            RAW_SELECTOR_STANDALONE_ALIAS_NAME,
            RAW_SELECTOR_STANDALONE_TERMINAL_NAME,
        }
        if (
            _fresh_directory_inventory(
                parent_descriptor, label="standalone RAW selector output root"
            )
            != expected_inventory
        ):
            raise ValueError("standalone RAW selector root inventory drift")
        terminal_descriptor = os.open(
            RAW_SELECTOR_STANDALONE_TERMINAL_NAME,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        alias_descriptor = os.open(
            RAW_SELECTOR_STANDALONE_ALIAS_NAME,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        terminal_before = os.fstat(terminal_descriptor)
        alias_before = os.fstat(alias_descriptor)
        terminal_named = os.stat(
            RAW_SELECTOR_STANDALONE_TERMINAL_NAME,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        terminal_live = os.lstat(path)
        alias_named = os.stat(
            RAW_SELECTOR_STANDALONE_ALIAS_NAME,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        alias_live = os.lstat(parent / RAW_SELECTOR_STANDALONE_ALIAS_NAME)
        terminal_identity = (terminal_before.st_dev, terminal_before.st_ino)
        if (
            not stat.S_ISREG(terminal_before.st_mode)
            or not stat.S_ISREG(terminal_named.st_mode)
            or not stat.S_ISREG(terminal_live.st_mode)
            or stat.S_IMODE(terminal_before.st_mode) != 0o444
            or stat.S_IMODE(terminal_named.st_mode) != 0o444
            or stat.S_IMODE(terminal_live.st_mode) != 0o444
            or not stat.S_ISREG(alias_before.st_mode)
            or not stat.S_ISREG(alias_named.st_mode)
            or not stat.S_ISREG(alias_live.st_mode)
            or stat.S_IMODE(alias_before.st_mode) != 0o444
            or stat.S_IMODE(alias_named.st_mode) != 0o444
            or stat.S_IMODE(alias_live.st_mode) != 0o444
            or terminal_before.st_nlink != 2
            or terminal_named.st_nlink != 2
            or terminal_live.st_nlink != 2
            or alias_before.st_nlink != 2
            or alias_named.st_nlink != 2
            or alias_live.st_nlink != 2
            or (terminal_named.st_dev, terminal_named.st_ino) != terminal_identity
            or (terminal_live.st_dev, terminal_live.st_ino) != terminal_identity
            or (alias_before.st_dev, alias_before.st_ino) != terminal_identity
            or (alias_named.st_dev, alias_named.st_ino) != terminal_identity
            or (alias_live.st_dev, alias_live.st_ino) != terminal_identity
        ):
            raise ValueError("standalone RAW selector terminal/alias identity drift")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while True:
            block = os.read(terminal_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
        encoded = b"".join(blocks)
        terminal_after = os.fstat(terminal_descriptor)
        alias_digest = hashlib.sha256()
        alias_blocks: list[bytes] = []
        while True:
            block = os.read(alias_descriptor, 1024 * 1024)
            if not block:
                break
            alias_digest.update(block)
            alias_blocks.append(block)
        alias_encoded = b"".join(alias_blocks)
        alias_after = os.fstat(alias_descriptor)
        terminal_named_after = os.stat(
            RAW_SELECTOR_STANDALONE_TERMINAL_NAME,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        terminal_live_after = os.lstat(path)
        alias_named_after = os.stat(
            RAW_SELECTOR_STANDALONE_ALIAS_NAME,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        alias_live_after = os.lstat(parent / RAW_SELECTOR_STANDALONE_ALIAS_NAME)
        if (
            _standalone_stat_binding(terminal_before)
            != _standalone_stat_binding(terminal_after)
            or _standalone_stat_binding(alias_before)
            != _standalone_stat_binding(alias_after)
            or _standalone_stat_binding(alias_after)
            != _standalone_stat_binding(terminal_after)
            or len(encoded) != terminal_after.st_size
            or len(encoded) != reference["bytes"]
            or digest.hexdigest() != reference["sha256"]
            or alias_encoded != encoded
            or alias_digest.hexdigest() != reference["sha256"]
            or (terminal_named_after.st_dev, terminal_named_after.st_ino)
            != terminal_identity
            or (terminal_live_after.st_dev, terminal_live_after.st_ino)
            != terminal_identity
            or (alias_named_after.st_dev, alias_named_after.st_ino) != terminal_identity
            or (alias_live_after.st_dev, alias_live_after.st_ino) != terminal_identity
        ):
            raise ValueError(
                "standalone RAW selector terminal pair changed during read"
            )
        if (
            _fresh_directory_inventory(
                parent_descriptor, label="standalone RAW selector output root"
            )
            != expected_inventory
        ):
            raise ValueError("standalone RAW selector root inventory changed")
        parent_after = os.fstat(parent_descriptor)
        parent_named_after = os.lstat(parent)
        if (
            (parent_after.st_dev, parent_after.st_ino) != parent_identity
            or (parent_named_after.st_dev, parent_named_after.st_ino) != parent_identity
            or stat.S_IMODE(parent_after.st_mode) != 0o555
            or stat.S_IMODE(parent_named_after.st_mode) != 0o555
        ):
            raise ValueError("standalone RAW selector root changed during read")
    finally:
        if alias_descriptor is not None:
            os.close(alias_descriptor)
        if terminal_descriptor is not None:
            os.close(terminal_descriptor)
        os.close(parent_descriptor)
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid standalone RAW selector terminal JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("standalone RAW selector terminal must be a JSON object")
    return (
        path,
        payload,
        (
            _standalone_stat_binding(parent_after),
            _standalone_stat_binding(terminal_after),
        ),
    )


def _validate_selector_manifest_reference_impl(
    reference: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    payload_override: Mapping[str, Any] | None = None,
    standalone_payload: bool = False,
) -> dict[str, Any]:
    """Shared semantic core for held references and unpublished payloads."""
    project_root = Path(project_root).resolve(strict=True)
    raw_path: Any = None
    normalized_raw_path: Path | None = None
    if payload_override is None:
        raw_path = reference.get("path") if isinstance(reference, Mapping) else None
        normalized_raw_path = (
            Path(os.path.normpath(Path(raw_path).absolute()))
            if isinstance(raw_path, str)
            else None
        )
    standalone_terminal = standalone_payload or (
        normalized_raw_path is not None
        and _is_standalone_raw_selector_terminal(normalized_raw_path, project_root)
    )
    standalone_snapshot: (
        tuple[
            tuple[int, int, int, int, int, int, int],
            tuple[int, int, int, int, int, int, int],
        ]
        | None
    ) = None
    if payload_override is not None:
        payload = dict(payload_override)
    elif standalone_terminal:
        _, payload, standalone_snapshot = _load_standalone_raw_selector_terminal(
            reference, project_root=project_root
        )
    else:
        if normalized_raw_path is not None and (
            _is_standalone_raw_selector_wrapper_member(
                normalized_raw_path, project_root
            )
        ):
            raise ValueError(
                "only the fixed terminal basename is consumable from the "
                "standalone RAW selector wrapper"
            )
        _, payload = _load_reference(
            reference, project_root=project_root, label="formal selector manifest"
        )
    _validate_common(payload, SELECTOR_SCHEMA, "selector manifest")
    product_line = payload.get("product_line")
    if product_line not in PRODUCT_LINES:
        raise ValueError("selector product_line must be RAW or ROBOT_RGB")
    if standalone_terminal and product_line != "RAW":
        raise ValueError("standalone RAW terminal has a non-RAW product line")
    image_name = payload.get("image_name")
    if (
        not isinstance(image_name, str)
        or image_name != Path(image_name).name
        or image_name in {"", ".", ".."}
    ):
        raise ValueError("selector image_name must be a frame-local basename")
    if product_line == "RAW" and image_name != "rgb.png":
        raise ValueError("RAW selector must bind rgb.png")
    if product_line == "ROBOT_RGB" and image_name == "rgb.png":
        raise ValueError("RobotRGB selector cannot bind the RAW rgb.png domain")
    artifact_root_value = payload.get("artifact_root")
    if (
        not isinstance(artifact_root_value, str)
        or not Path(artifact_root_value).is_absolute()
    ):
        raise ValueError("selector artifact_root must be canonical and absolute")
    _reject_forbidden_source_path(
        artifact_root_value,
        label="selector artifact root",
    )
    declared_artifact_root = validate_directory_path(
        artifact_root_value,
        allowed_roots=[Path(artifact_root_value).anchor],
        label="selector artifact root",
    )
    if declared_artifact_root.parent == Path(declared_artifact_root.anchor):
        raise ValueError("selector artifact_root must name a project namespace")
    if declared_artifact_root.is_relative_to(project_root / "_run"):
        raise ValueError("selector artifact root cannot come from _run staging")
    if artifact_root is not None:
        approved_artifact_root = validate_directory_path(
            artifact_root,
            allowed_roots=[Path(artifact_root).absolute().anchor],
            label="caller-approved selector artifact root",
        )
        if declared_artifact_root != approved_artifact_root:
            raise ValueError(
                "selector artifact_root differs from the caller-approved root"
            )
    selector_root_value = payload.get("selector_root")
    if (
        not isinstance(selector_root_value, str)
        or not Path(selector_root_value).is_absolute()
    ):
        raise ValueError("selector_root must be canonical and absolute")
    _reject_forbidden_source_path(selector_root_value, label="selector root")
    selector_root = validate_directory_path(
        selector_root_value,
        allowed_roots=[declared_artifact_root],
        label="selector root",
    )
    if selector_root.is_relative_to(project_root / "_run"):
        raise ValueError("selector root cannot be consumed from _run staging")
    sessions = payload.get("sessions")
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError("selector sessions must be a non-empty object")
    if set(sessions) != ELIGIBLE68:
        raise ValueError("selector session cohort must be exact eligible68")
    normalized: dict[str, dict[str, Mapping[str, Any]]] = {}
    metadata_owners: dict[Path, str] = {}
    image_owners: dict[Path, str] = {}
    for session_id, session in sessions.items():
        if not SESSION_PATTERN.fullmatch(session_id) or not isinstance(session, dict):
            raise ValueError(f"invalid selector session row: {session_id!r}")
        if set(session) != {"frames"}:
            raise ValueError(f"selector session schema drift: {session_id}")
        frames = session.get("frames")
        if not isinstance(frames, dict) or not frames:
            raise ValueError(f"selector session has no frame records: {session_id}")
        ordered = [_validate_frame_key(key, label=session_id) for key in frames]
        expected_frames = list(range(ELIGIBLE68_FRAME_COUNTS[session_id]))
        if ordered != expected_frames:
            raise ValueError(
                f"selector frame ledger differs from eligible68: {session_id}"
            )
        for frame_key, record in frames.items():
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"selector frame record is not an object: {session_id}/{frame_key}"
                )
            if set(record) != {"metadata", "image", "unresolved"}:
                raise ValueError(
                    f"selector frame schema drift: {session_id}/{frame_key}"
                )
            _validate_declared_file_reference(
                record["metadata"], label=f"{session_id}/{frame_key} metadata"
            )
            _validate_declared_file_reference(
                record["image"], label=f"{session_id}/{frame_key} image"
            )
            for role in ("metadata", "image"):
                if Path(record[role]["path"]).is_relative_to(project_root / "_run"):
                    raise ValueError(
                        f"{session_id}/{frame_key} {role} cannot come from _run staging"
                    )
            metadata_path = Path(record["metadata"]["path"])
            image_path = Path(record["image"]["path"])
            logical_key = f"{session_id}/{frame_key}"
            for declared_path, owners, role in (
                (metadata_path, metadata_owners, "metadata"),
                (image_path, image_owners, "image"),
            ):
                previous = owners.get(declared_path)
                if previous is not None and previous != logical_key:
                    raise ValueError(
                        f"selector {role} path aliases frames: {previous}, {logical_key}"
                    )
                owners[declared_path] = logical_key
            if not metadata_path.is_relative_to(project_root):
                raise ValueError(
                    f"{session_id}/{frame_key} metadata escapes the project root"
                )
            if not image_path.is_relative_to(selector_root):
                raise ValueError(
                    f"{session_id}/{frame_key} image escapes selector_root"
                )
            if image_path.name != image_name:
                raise ValueError(
                    f"{session_id}/{frame_key} image basename differs from image_name"
                )
            if not isinstance(record["unresolved"], bool):
                raise ValueError(
                    f"selector unresolved flag is not boolean: {session_id}/{frame_key}"
                )
        normalized[session_id] = dict(frames)
    result = {
        **payload,
        "artifact_root": str(declared_artifact_root),
        "selector_root": str(selector_root),
        "normalized_records": normalized,
    }
    if product_line == "ROBOT_RGB":
        lineage_reference = payload.get("robot_rgb_lineage_authority_ref")
        if lineage_reference is None:
            raise RuntimeError(
                f"{ROBOT_RGB_LINEAGE_HOLD}: missing aggregate lineage authority"
            )
        ambiguous_lineage_fields = sorted(
            field
            for field in payload
            if "lineage" in field.casefold()
            and field != "robot_rgb_lineage_authority_ref"
        )
        if ambiguous_lineage_fields:
            raise ValueError(
                "RobotRGB selector contains ambiguous self-declared lineage fields: "
                f"{ambiguous_lineage_fields}"
            )
        result["robot_rgb_lineage_authority"] = _validate_robot_rgb_lineage_authority(
            lineage_reference,
            selector_records=normalized,
            project_root=project_root,
        )
    if standalone_snapshot is not None:
        _, terminal_payload, terminal_snapshot = _load_standalone_raw_selector_terminal(
            reference,
            project_root=project_root,
        )
        if terminal_snapshot != standalone_snapshot or terminal_payload != payload:
            raise ValueError("standalone RAW selector changed during validation")
    return result


def _validate_standalone_raw_selector_payload(
    payload: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate an unpublished standalone payload through the public semantic core."""
    return _validate_selector_manifest_reference_impl(
        {},
        artifact_root=artifact_root,
        project_root=project_root,
        payload_override=payload,
        standalone_payload=True,
    )


def validate_selector_manifest_reference(
    reference: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate one immutable RAW or RobotRGB selector manifest."""
    return _validate_selector_manifest_reference_impl(
        reference,
        artifact_root=artifact_root,
        project_root=project_root,
    )


def _reference_identity(reference: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return reference.get("path"), reference.get("bytes"), reference.get("sha256")


def _open_directory_chain_nofollow(path: Path, *, label: str) -> int:
    """Open an absolute directory one component at a time without symlinks."""
    path = path.absolute()
    if path != Path(os.path.normpath(path)):
        raise ValueError(f"{label} path is not canonical: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"{label} path chain is no longer an ordinary "
                    f"non-symlink directory: {path}"
                ) from error
            os.close(descriptor)
            descriptor = next_descriptor
        state = os.fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"{label} must be an ordinary directory: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _fresh_directory_inventory(descriptor: int, *, label: str) -> set[str]:
    """List a held directory through a fresh getdents offset on every pass."""
    held_before = os.fstat(descriptor)
    if not stat.S_ISDIR(held_before.st_mode):
        raise ValueError(f"{label} held FD is not an ordinary directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fresh_descriptor = os.open(".", flags, dir_fd=descriptor)
    try:
        fresh_before = os.fstat(fresh_descriptor)
        held_identity = (held_before.st_dev, held_before.st_ino)
        if (
            not stat.S_ISDIR(fresh_before.st_mode)
            or (fresh_before.st_dev, fresh_before.st_ino) != held_identity
        ):
            raise ValueError(f"{label} fresh inventory FD identity drift")
        names = set(os.listdir(fresh_descriptor))
        fresh_after = os.fstat(fresh_descriptor)
        held_after = os.fstat(descriptor)
        if (fresh_after.st_dev, fresh_after.st_ino) != held_identity or (
            held_after.st_dev,
            held_after.st_ino,
        ) != held_identity:
            raise ValueError(f"{label} identity changed during inventory")
        return names
    finally:
        os.close(fresh_descriptor)


def _directory_binding(path: Path, *, label: str) -> dict[str, Any]:
    descriptor = _open_directory_chain_nofollow(path, label=label)
    try:
        state = os.fstat(descriptor)
        return {
            "path": str(path),
            "st_dev": int(state.st_dev),
            "st_ino": int(state.st_ino),
        }
    finally:
        os.close(descriptor)


def _ordinary_file_binding(path: Path, *, label: str) -> dict[str, Any]:
    state = os.lstat(path)
    if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
        raise ValueError(f"{label} must be a singly-linked ordinary file: {path}")
    return {
        "path": str(path),
        "st_dev": int(state.st_dev),
        "st_ino": int(state.st_ino),
    }


def _load_explicit_json_reference(
    reference: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    raw_path = reference.get("path") if isinstance(reference, Mapping) else None
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} reference path must be canonical and absolute")
    _reject_forbidden_source_path(raw_path, label=label)
    parent = validate_directory_path(
        Path(raw_path).parent,
        allowed_roots=[Path(raw_path).anchor],
        label=f"{label} parent",
    )
    path, encoded = read_file_reference(reference, allowed_roots=[parent], label=label)
    if path.is_relative_to(PROJECT_ROOT / "_run"):
        raise ValueError(f"{label} cannot come from _run staging")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return path, payload


def _validate_frozen_twin(
    product_line: str,
    row: Mapping[str, Any],
    *,
    split_reference: Mapping[str, Any],
    embodiment: str,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"final-test twin row is invalid: {product_line}")
    freeze_reference = row.get("freeze_manifest_ref")
    if not isinstance(freeze_reference, Mapping):
        raise ValueError(f"final-test twin lacks freeze reference: {product_line}")
    freeze_path, freeze = _load_explicit_json_reference(
        freeze_reference, label=f"{product_line} frozen-best manifest"
    )
    if freeze_path != freeze_path.parent / "freeze_manifest.json":
        raise ValueError(f"frozen-best manifest path drift: {product_line}")
    _validate_common(freeze, FROZEN_BEST_SCHEMA, "frozen-best manifest")
    if freeze.get("status") != "frozen" or freeze.get("product_line") != product_line:
        raise ValueError(f"frozen-best status/product mismatch: {product_line}")
    if freeze.get("embodiment") != embodiment:
        raise ValueError(f"frozen-best embodiment mismatch: {product_line}")
    if freeze.get("selection_role") != "dev" or freeze.get(
        "selection_sessions"
    ) != list(DEV8):
        raise ValueError(f"frozen-best was not selected on exact dev8: {product_line}")
    if freeze.get("final_test_status") != "NOT_EVALUATED":
        raise ValueError(f"frozen-best has already entered final test: {product_line}")
    if _reference_identity(freeze.get("split_ref", {})) != _reference_identity(
        split_reference
    ):
        raise ValueError(f"frozen-best split differs: {product_line}")
    checkpoint_reference = freeze.get("checkpoint_ref")
    if not isinstance(checkpoint_reference, Mapping):
        raise ValueError(f"frozen-best lacks checkpoint reference: {product_line}")
    checkpoint_path, _ = read_file_reference(
        checkpoint_reference,
        allowed_roots=[freeze_path.parent],
        label=f"{product_line} frozen-best checkpoint",
    )
    if checkpoint_path != freeze_path.parent / "best.pt":
        raise ValueError(f"frozen-best checkpoint path drift: {product_line}")
    if _reference_identity(row.get("checkpoint_ref", {})) != _reference_identity(
        checkpoint_reference
    ):
        raise ValueError(f"final-test checkpoint binding differs: {product_line}")

    run_manifest_reference = freeze.get("frozen_run_manifest_ref")
    if not isinstance(run_manifest_reference, Mapping):
        raise ValueError(f"frozen-best lacks run-manifest reference: {product_line}")
    run_manifest_path, run_manifest = _load_explicit_json_reference(
        run_manifest_reference,
        label=f"{product_line} frozen run manifest",
    )
    if run_manifest_path != freeze_path.parent / "run_manifest.json":
        raise ValueError(f"frozen run-manifest path drift: {product_line}")
    if run_manifest.get("embodiment") != embodiment:
        raise ValueError(f"frozen run-manifest embodiment mismatch: {product_line}")
    if _reference_identity(run_manifest.get("split_ref", {})) != _reference_identity(
        split_reference
    ):
        raise ValueError(f"frozen run-manifest split differs: {product_line}")

    selector_reference = freeze.get("selector_manifest_ref")
    paired_reference = freeze.get("paired_kept_manifest_ref")
    artifact_root = freeze.get("artifact_root")
    if not isinstance(selector_reference, Mapping) or not isinstance(
        paired_reference, Mapping
    ):
        raise ValueError(f"frozen-best lacks selector lineage: {product_line}")
    if (
        _reference_identity(run_manifest.get("selector_manifest_ref", {}))
        != _reference_identity(selector_reference)
        or _reference_identity(run_manifest.get("paired_kept_manifest_ref", {}))
        != _reference_identity(paired_reference)
        or run_manifest.get("artifact_root") != artifact_root
    ):
        raise ValueError(f"frozen-best/run selector lineage differs: {product_line}")
    selector_binding = require_manifest_selector_ready(
        selector_reference,
        paired_reference,
        artifact_root=artifact_root,
    )
    if (
        selector_binding["product_line"] != product_line
        or freeze.get("image_name") != selector_binding["image_name"]
        or freeze.get("artifact_roots") != selector_binding["artifact_roots"]
    ):
        raise ValueError(f"frozen-best selector domain mismatch: {product_line}")

    run_root_value = freeze.get("run_root")
    run_directory_value = freeze.get("run_directory")
    if (
        not isinstance(run_root_value, str)
        or not Path(run_root_value).is_absolute()
        or not isinstance(run_directory_value, str)
        or not Path(run_directory_value).is_absolute()
        or run_manifest.get("run_root") != run_root_value
        or run_manifest.get("run_directory") != run_directory_value
    ):
        raise ValueError(f"frozen-best run identity drift: {product_line}")
    run_root = validate_directory_path(
        run_root_value,
        allowed_roots=[Path(run_root_value).anchor],
        label=f"{product_line} frozen run root",
    )
    run_directory = validate_directory_path(
        run_directory_value,
        allowed_roots=[run_root],
        label=f"{product_line} frozen run directory",
    )
    if run_directory.parent != run_root:
        raise ValueError(f"frozen-best run directory is not isolated: {product_line}")
    return {
        "freeze": freeze,
        "freeze_binding": _ordinary_file_binding(
            freeze_path, label=f"{product_line} frozen-best manifest"
        ),
        "checkpoint_binding": _ordinary_file_binding(
            checkpoint_path, label=f"{product_line} frozen-best checkpoint"
        ),
        "run_manifest_binding": _ordinary_file_binding(
            run_manifest_path, label=f"{product_line} frozen run manifest"
        ),
        "run_directory_binding": _directory_binding(
            run_directory, label=f"{product_line} frozen run directory"
        ),
        "selector_reference": dict(selector_reference),
        "paired_reference": dict(paired_reference),
    }


def authorize_evaluation_role(
    split: Mapping[str, Any],
    evaluation_role: str,
    *,
    final_test_gate_reference: Mapping[str, Any] | None = None,
    split_reference: Mapping[str, Any] | None = None,
    checkpoint_reference: Mapping[str, Any] | None = None,
    product_line: str | None = None,
    embodiment: str | None = None,
    evaluator: str | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Map public dev/final-test roles and gate every final-test access."""
    validate_eligible68_role_contract(split)
    if evaluation_role == "dev":
        if final_test_gate_reference is not None:
            raise ValueError("a final-test gate may not be supplied for dev evaluation")
        return {
            "evaluation_role": "dev",
            "storage_role": "validation",
            "sessions": list(DEV8),
            "final_test_authorized": False,
        }
    if evaluation_role != "final_test":
        raise ValueError(f"unknown evaluation role: {evaluation_role!r}")
    required = {
        "gate": final_test_gate_reference,
        "split": split_reference,
        "checkpoint": checkpoint_reference,
        "product_line": product_line,
        "embodiment": embodiment,
        "evaluator": evaluator,
        "output": output_directory,
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise RuntimeError(
            f"HOLD_FINAL_TEST_GATE_REQUIRED: missing explicit fields {missing}"
        )
    assert final_test_gate_reference is not None
    assert split_reference is not None
    assert checkpoint_reference is not None
    assert isinstance(product_line, str)
    assert isinstance(embodiment, str)
    assert isinstance(evaluator, str)
    _, gate = _load_explicit_json_reference(
        final_test_gate_reference, label="final-test gate"
    )
    _validate_common(gate, FINAL_TEST_GATE_SCHEMA, "final-test gate")
    if gate.get("status") != "READY_FOR_ONE_SHOT_FINAL_TEST":
        raise ValueError("final-test gate is not ready")
    if (
        gate.get("cohort") != "eligible68"
        or gate.get("split_protocol") != ELIGIBLE68_SPLIT_PROTOCOL
    ):
        raise ValueError("final-test gate cohort/protocol drift")
    if gate.get("final_test_sessions") != list(FINAL_TEST8):
        raise ValueError("final-test gate session cohort drift")
    if gate.get("one_shot") is not True or gate.get("selection_role") != "dev":
        raise ValueError("final-test gate does not preserve one-shot dev selection")
    if gate.get("embodiment") != embodiment or gate.get("evaluator") != evaluator:
        raise ValueError("final-test gate embodiment/evaluator drift")
    if _reference_identity(gate.get("split_ref", {})) != _reference_identity(
        split_reference
    ):
        raise ValueError("final-test gate split differs from checkpoint")
    evaluation_id = gate.get("evaluation_id")
    if not isinstance(evaluation_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", evaluation_id
    ):
        raise ValueError("final-test evaluation_id is invalid")
    twins = gate.get("frozen_twins")
    if not isinstance(twins, Mapping) or set(twins) != PRODUCT_LINES:
        raise ValueError("final-test gate must bind RAW and ROBOT_RGB frozen twins")
    outputs: dict[str, Path] = {}
    output_parent_bindings: dict[str, dict[str, Any]] = {}
    validated_twins: dict[str, dict[str, Any]] = {}
    for candidate_product in sorted(PRODUCT_LINES):
        row = twins[candidate_product]
        validated_twins[candidate_product] = _validate_frozen_twin(
            candidate_product,
            row,
            split_reference=split_reference,
            embodiment=embodiment,
        )
        raw_output = row.get("output_directory") if isinstance(row, Mapping) else None
        if not isinstance(raw_output, str) or not Path(raw_output).is_absolute():
            raise ValueError(f"final-test output is invalid: {candidate_product}")
        _reject_forbidden_source_path(raw_output, label="final-test output")
        output_path = Path(raw_output)
        parent = validate_directory_path(
            output_path.parent,
            allowed_roots=[output_path.anchor],
            label=f"{candidate_product} final-test output parent",
        )
        if output_path != parent / output_path.name:
            raise ValueError(f"final-test output is not canonical: {candidate_product}")
        outputs[candidate_product] = output_path
        output_parent_bindings[candidate_product] = _directory_binding(
            parent,
            label=f"{candidate_product} final-test output parent",
        )
    for binding_name in (
        "freeze_binding",
        "checkpoint_binding",
        "run_manifest_binding",
        "run_directory_binding",
    ):
        raw_binding = validated_twins["RAW"][binding_name]
        robot_binding = validated_twins["ROBOT_RGB"][binding_name]
        if raw_binding["path"] == robot_binding["path"] or (
            raw_binding["st_dev"],
            raw_binding["st_ino"],
        ) == (robot_binding["st_dev"], robot_binding["st_ino"]):
            raise ValueError(f"final-test frozen twins alias {binding_name}")
    if _reference_identity(validated_twins["RAW"]["selector_reference"]) == (
        _reference_identity(validated_twins["ROBOT_RGB"]["selector_reference"])
    ):
        raise ValueError("final-test frozen twins alias selector_reference")
    if _reference_identity(validated_twins["RAW"]["paired_reference"]) != (
        _reference_identity(validated_twins["ROBOT_RGB"]["paired_reference"])
    ):
        raise ValueError("final-test frozen twins do not share one paired ledger")
    if outputs["RAW"] == outputs["ROBOT_RGB"]:
        raise ValueError("final-test twin outputs must be distinct")
    if product_line not in PRODUCT_LINES:
        raise ValueError("final-test product line must be RAW or ROBOT_RGB")
    current_row = twins[product_line]
    if _reference_identity(
        current_row.get("checkpoint_ref", {})
    ) != _reference_identity(checkpoint_reference):
        raise ValueError("current checkpoint is not the gated frozen twin")
    requested_output = Path(output_directory).absolute()
    if requested_output != outputs[product_line]:
        raise ValueError("final-test output differs from the one-shot gate")
    return {
        "evaluation_role": "final_test",
        "storage_role": "test",
        "sessions": list(FINAL_TEST8),
        "final_test_authorized": True,
        "evaluation_id": evaluation_id,
        "output_directory": str(requested_output),
        "output_parent_binding": output_parent_bindings[product_line],
        "gate_reference": dict(final_test_gate_reference),
    }


class FinalTestOutputLease:
    """Hold the exact claimed output directory until evaluation is published."""

    def __init__(
        self,
        output: Path,
        descriptor: int,
        identity: tuple[int, int],
    ) -> None:
        self.output = output
        self._descriptor = descriptor
        self._identity = identity
        self._closed = False

    @property
    def bound_directory(self) -> Path:
        if self._closed:
            raise RuntimeError("final-test output lease is closed")
        # Every subsequent pathname write traverses this held descriptor, not
        # a mutable lexical parent chain.
        return Path(f"/proc/self/fd/{self._descriptor}")

    def close(self) -> None:
        if self._closed:
            return
        try:
            held = os.fstat(self._descriptor)
            if (held.st_dev, held.st_ino) != self._identity:
                raise ValueError("held final-test output directory changed")
            current_descriptor = _open_directory_chain_nofollow(
                self.output,
                label="final-test output directory publication recheck",
            )
            try:
                current = os.fstat(current_descriptor)
                if (current.st_dev, current.st_ino) != self._identity:
                    raise ValueError(
                        "final-test output directory changed before publication completion"
                    )
            finally:
                os.close(current_descriptor)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                os.close(self._descriptor)
            except OSError:
                pass
            self._closed = True


def claim_final_test_output(
    authorization: Mapping[str, Any],
    *,
    hold: bool = False,
) -> Path | FinalTestOutputLease:
    """Atomically consume one gated output and optionally hold its directory."""
    if authorization.get("final_test_authorized") is not True:
        raise ValueError("only final-test authorization can be claimed")
    output = Path(str(authorization["output_directory"])).absolute()
    binding = authorization.get("output_parent_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("final-test authorization lacks an output-parent binding")
    if binding.get("path") != str(output.parent) or any(
        not isinstance(binding.get(field), int) or isinstance(binding.get(field), bool)
        for field in ("st_dev", "st_ino")
    ):
        raise ValueError("final-test output-parent binding is invalid")

    parent_descriptor = _open_directory_chain_nofollow(
        output.parent,
        label="final-test output parent",
    )
    expected_parent_identity = (binding["st_dev"], binding["st_ino"])
    parent_before = os.fstat(parent_descriptor)
    if (parent_before.st_dev, parent_before.st_ino) != expected_parent_identity:
        os.close(parent_descriptor)
        raise ValueError("final-test output parent changed after authorization")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    output_descriptor: int | None = None
    claim_descriptor: int | None = None
    claim = output / "FINAL_TEST_CLAIM.json"
    encoded = json.dumps(
        {
            "schema_version": "humanego-final-test-claim-v1",
            "evaluation_id": authorization["evaluation_id"],
            "gate_ref": authorization["gate_reference"],
            "status": "CLAIMED_BEFORE_FINAL_TEST_PHASE2",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    lease: FinalTestOutputLease | None = None
    try:
        try:
            os.mkdir(output.name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise FileExistsError(
                f"final-test output was already claimed: {output}"
            ) from error
        try:
            output_descriptor = os.open(
                output.name,
                directory_flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise ValueError(
                "final-test output changed after its atomic creation"
            ) from error
        output_before = os.fstat(output_descriptor)
        output_at_parent = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        output_identity = (output_before.st_dev, output_before.st_ino)
        if (
            not stat.S_ISDIR(output_before.st_mode)
            or output_identity != (output_at_parent.st_dev, output_at_parent.st_ino)
            or _fresh_directory_inventory(
                output_descriptor, label="new final-test output directory"
            )
        ):
            raise ValueError("new final-test output directory is not empty and stable")

        # Reopen the lexical parent chain immediately before consuming the
        # one-shot claim.  Both this descriptor and the held descriptor must
        # still identify the parent accepted during authorization.
        recheck_descriptor = _open_directory_chain_nofollow(
            output.parent,
            label="final-test output parent recheck",
        )
        try:
            recheck = os.fstat(recheck_descriptor)
            if (recheck.st_dev, recheck.st_ino) != expected_parent_identity:
                raise ValueError("final-test output parent changed before claim write")
        finally:
            os.close(recheck_descriptor)

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        claim_descriptor = os.open(
            claim.name,
            flags,
            0o444,
            dir_fd=output_descriptor,
        )
        claim_before = os.fstat(claim_descriptor)
        offset = 0
        while offset < len(encoded):
            written = os.write(claim_descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short write while creating final-test claim")
            offset += written
        os.fsync(claim_descriptor)
        claim_after = os.fstat(claim_descriptor)
        claim_at_output = os.stat(
            claim.name,
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(claim_after.st_mode)
            or claim_after.st_nlink != 1
            or claim_after.st_size != len(encoded)
            or (claim_before.st_dev, claim_before.st_ino)
            != (claim_after.st_dev, claim_after.st_ino)
            or (claim_after.st_dev, claim_after.st_ino)
            != (claim_at_output.st_dev, claim_at_output.st_ino)
        ):
            raise ValueError("final-test claim file changed while being written")
        output_after = os.fstat(output_descriptor)
        output_at_parent = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if output_identity != (
            output_after.st_dev,
            output_after.st_ino,
        ) or output_identity != (output_at_parent.st_dev, output_at_parent.st_ino):
            raise ValueError("final-test output changed while claim was written")
        parent_after = os.fstat(parent_descriptor)
        if (parent_after.st_dev, parent_after.st_ino) != expected_parent_identity:
            raise ValueError("final-test output parent changed while claim was written")
        os.fchmod(output_descriptor, 0o755)
        os.fsync(output_descriptor)
        os.fsync(parent_descriptor)
        if hold:
            lease = FinalTestOutputLease(
                output,
                output_descriptor,
                output_identity,
            )
            output_descriptor = None
    finally:
        if claim_descriptor is not None:
            os.close(claim_descriptor)
        if output_descriptor is not None:
            os.close(output_descriptor)
        os.close(parent_descriptor)
    return lease if lease is not None else claim


def _validate_paired_terminal_publication_declaration(
    paired: Mapping[str, Any],
) -> bool:
    declaration = paired.get("terminal_publication")
    if declaration is None:
        return False
    expected = {
        "profile": PAIRED_TERMINAL_PUBLICATION_PROFILE,
        "terminal_name": PAIRED_TERMINAL_NAME,
        "permanent_alias_name": PAIRED_COMPLETE_PAYLOAD_ALIAS_NAME,
        "required_nlink": 2,
    }
    if declaration != expected:
        raise ValueError("paired-kept terminal publication declaration drift")
    return True


def _read_paired_hardlink_leaf(
    parent_descriptor: int,
    parent: Path,
    name: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        held_entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        live_entry = os.lstat(parent / name)
        expected_identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 2
            or (held_entry.st_dev, held_entry.st_ino) != expected_identity
            or (live_entry.st_dev, live_entry.st_ino) != expected_identity
        ):
            raise ValueError(f"{label} hardlink identity/mode drift")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        held_after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        live_after = os.lstat(parent / name)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        encoded = b"".join(blocks)
        if (
            len(encoded) != after.st_size
            or any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            )
            or (held_after.st_dev, held_after.st_ino) != expected_identity
            or (live_after.st_dev, live_after.st_ino) != expected_identity
        ):
            raise ValueError(f"{label} changed during same-FD read")
        return encoded, after
    finally:
        os.close(descriptor)


def _load_paired_hardlink_terminal(
    reference: Mapping[str, Any], *, project_root: Path
) -> tuple[Path, dict[str, Any]]:
    _validate_exact_lineage_reference(reference, label="paired-kept hardlink terminal")
    path = Path(str(reference["path"])).absolute()
    if path.name != PAIRED_TERMINAL_NAME:
        raise ValueError("marked paired-kept reference must use the terminal name")
    _reject_forbidden_source_path(path, label="paired-kept hardlink terminal")
    if not path.is_relative_to(project_root) or path.is_relative_to(
        project_root / "_run"
    ):
        raise ValueError(
            "paired-kept hardlink terminal must be formal project-root output"
        )
    parent = validate_directory_path(
        path.parent,
        allowed_roots=[project_root],
        label="paired-kept hardlink output root",
    )
    parent_descriptor = _open_directory_chain_nofollow(
        parent, label="paired-kept hardlink output root"
    )
    expected_inventory = {
        PAIRED_RAW_SELECTOR_NAME,
        PAIRED_ROBOT_SELECTOR_NAME,
        PAIRED_COMPLETE_PAYLOAD_ALIAS_NAME,
        PAIRED_TERMINAL_NAME,
    }
    try:
        parent_state = os.fstat(parent_descriptor)
        named_parent = os.lstat(parent)
        if (
            not stat.S_ISDIR(parent_state.st_mode)
            or stat.S_IMODE(parent_state.st_mode) != 0o555
            or (parent_state.st_dev, parent_state.st_ino)
            != (named_parent.st_dev, named_parent.st_ino)
        ):
            raise ValueError("paired-kept hardlink output root identity/mode drift")
        if (
            _fresh_directory_inventory(
                parent_descriptor, label="paired-kept hardlink output root"
            )
            != expected_inventory
        ):
            raise ValueError("paired-kept hardlink output inventory drift")
        terminal_encoded, terminal_state = _read_paired_hardlink_leaf(
            parent_descriptor,
            parent,
            PAIRED_TERMINAL_NAME,
            label="paired-kept terminal",
        )
        alias_encoded, alias_state = _read_paired_hardlink_leaf(
            parent_descriptor,
            parent,
            PAIRED_COMPLETE_PAYLOAD_ALIAS_NAME,
            label="paired-kept permanent payload alias",
        )
        if (
            (terminal_state.st_dev, terminal_state.st_ino)
            != (alias_state.st_dev, alias_state.st_ino)
            or terminal_encoded != alias_encoded
            or len(terminal_encoded) != reference["bytes"]
            or hashlib.sha256(terminal_encoded).hexdigest() != reference["sha256"]
        ):
            raise ValueError("paired-kept terminal/alias content identity drift")
        if (
            _fresh_directory_inventory(
                parent_descriptor, label="paired-kept hardlink output root"
            )
            != expected_inventory
        ):
            raise ValueError(
                "paired-kept hardlink output inventory changed during read"
            )
        named_parent_after = os.lstat(parent)
        if (parent_state.st_dev, parent_state.st_ino) != (
            named_parent_after.st_dev,
            named_parent_after.st_ino,
        ):
            raise ValueError("paired-kept hardlink output root changed during read")
    finally:
        os.close(parent_descriptor)
    try:
        payload = json.loads(terminal_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid paired-kept hardlink terminal JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("paired-kept hardlink terminal must be a JSON object")
    if not _validate_paired_terminal_publication_declaration(payload):
        raise ValueError("hardlinked paired-kept terminal lacks publication marker")
    selector_refs = payload.get("selector_refs")
    expected_selector_paths = {
        "RAW": str(parent / PAIRED_RAW_SELECTOR_NAME),
        "ROBOT_RGB": str(parent / PAIRED_ROBOT_SELECTOR_NAME),
    }
    if not isinstance(selector_refs, Mapping) or any(
        not isinstance(selector_refs.get(product_line), Mapping)
        or selector_refs[product_line].get("path") != expected_path
        for product_line, expected_path in expected_selector_paths.items()
    ):
        raise ValueError("paired-kept hardlink selectors are not fixed output leaves")
    return path, payload


def _load_paired_kept_reference(
    reference: Mapping[str, Any] | None, *, project_root: Path
) -> tuple[Path, dict[str, Any]]:
    raw_path = reference.get("path") if isinstance(reference, Mapping) else None
    if isinstance(raw_path, str) and Path(raw_path).name == (
        PAIRED_COMPLETE_PAYLOAD_ALIAS_NAME
    ):
        raise ValueError("paired-kept permanent payload alias is not a terminal input")
    if isinstance(raw_path, str):
        try:
            state = os.lstat(raw_path)
        except OSError:
            state = None
        if state is not None and state.st_nlink == 2:
            return _load_paired_hardlink_terminal(
                reference or {}, project_root=project_root
            )
    path, paired = _load_reference(
        reference, project_root=project_root, label="paired-kept manifest"
    )
    marked = _validate_paired_terminal_publication_declaration(paired)
    if marked:
        if path.name != PAIRED_TERMINAL_NAME:
            raise ValueError("marked paired-kept payload is not the terminal input")
        # A marked nlink1 terminal is either incomplete or had its permanent
        # alias removed/replaced.  Route it through the strict wrapper so it
        # fails rather than becoming indistinguishable from a legacy manifest.
        return _load_paired_hardlink_terminal(
            reference or {}, project_root=project_root
        )
    return path, paired


def require_manifest_selector_ready(
    selector_reference: Mapping[str, Any] | None = None,
    paired_kept_reference: Mapping[str, Any] | None = None,
    *,
    artifact_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return a verified selector binding or reject before any consumer runs."""
    project_root = Path(project_root).resolve(strict=True)
    if selector_reference is None:
        raise RuntimeError(f"{FORMAL_SELECTOR_HOLD}: missing selector manifest")
    if artifact_root is None:
        raise RuntimeError(f"{FORMAL_SELECTOR_HOLD}: missing explicit artifact root")
    if not Path(artifact_root).is_absolute():
        raise ValueError("caller-approved selector artifact root must be absolute")
    selector = validate_selector_manifest_reference(
        selector_reference,
        artifact_root=artifact_root,
        project_root=project_root,
    )
    _, paired = _load_paired_kept_reference(
        paired_kept_reference, project_root=project_root
    )
    _validate_common(paired, PAIRED_SCHEMA, "paired-kept manifest")
    if paired.get("cohort") != "eligible68":
        raise ValueError("paired-kept cohort must be eligible68")
    if paired.get("selection_policy") != "SYMMETRIC_FRAME_AND_WINDOW_INTERSECTION":
        raise ValueError("paired-kept selection policy drift")
    if paired.get("unresolved_policy") != (
        "RETAIN_PAIRED_FRAME_REGARDLESS_OF_DOMAIN_UNRESOLVED"
    ):
        raise ValueError("paired-kept unresolved policy permits asymmetric deletion")
    selector_refs = paired.get("selector_refs")
    if not isinstance(selector_refs, dict) or set(selector_refs) != PRODUCT_LINES:
        raise ValueError("paired-kept must bind exactly RAW and ROBOT_RGB selectors")
    artifact_roots = paired.get("artifact_roots")
    if not isinstance(artifact_roots, dict) or set(artifact_roots) != PRODUCT_LINES:
        raise ValueError("paired-kept must bind RAW and ROBOT_RGB artifact roots")
    selectors: dict[str, dict[str, Any]] = {}
    for product_line in sorted(PRODUCT_LINES):
        ref = selector_refs[product_line]
        if not isinstance(ref, Mapping):
            raise ValueError(
                f"paired-kept selector reference is invalid: {product_line}"
            )
        candidate_root = artifact_roots[product_line]
        if (
            not isinstance(candidate_root, str)
            or not Path(candidate_root).is_absolute()
        ):
            raise ValueError(f"paired-kept artifact root is invalid: {product_line}")
        candidate = validate_selector_manifest_reference(
            ref,
            artifact_root=candidate_root,
            project_root=project_root,
        )
        if candidate["product_line"] != product_line:
            raise ValueError(f"paired-kept selector product mismatch: {product_line}")
        if candidate["artifact_root"] != candidate_root:
            raise ValueError(f"paired-kept artifact root drift: {product_line}")
        selectors[product_line] = candidate
    selected_ref = selector_refs[selector["product_line"]]
    if _reference_identity(selected_ref) != _reference_identity(selector_reference):
        raise ValueError("CLI selector differs from paired-kept bound selector")
    approved_root = validate_directory_path(
        artifact_root,
        allowed_roots=[Path(artifact_root).absolute().anchor],
        label="caller-approved selector artifact root",
    )
    if Path(artifact_roots[selector["product_line"]]) != approved_root:
        raise ValueError("CLI artifact root differs from paired-kept bound root")

    sessions = paired.get("sessions")
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError("paired-kept sessions must be a non-empty object")
    if set(sessions) != ELIGIBLE68:
        raise ValueError("paired-kept session cohort must be exact eligible68")
    if set(sessions) != set(selectors["RAW"]["sessions"]) or set(sessions) != set(
        selectors["ROBOT_RGB"]["sessions"]
    ):
        raise ValueError("RAW/Robot selector session cohort differs from paired ledger")

    frame_ledger: dict[str, frozenset[int]] = {}
    window_starts: dict[str, frozenset[int]] = {}
    for session_id, session in sessions.items():
        if not isinstance(session, dict) or set(session) != {"frames", "window_starts"}:
            raise ValueError(f"paired-kept session schema drift: {session_id}")
        frames = session["frames"]
        starts = session["window_starts"]
        if (
            not isinstance(frames, list)
            or not frames
            or any(not isinstance(value, int) or value < 0 for value in frames)
            or frames != sorted(set(frames))
        ):
            raise ValueError(f"paired-kept frame ledger drift: {session_id}")
        if (
            not isinstance(starts, list)
            or not starts
            or any(not isinstance(value, int) or value < 0 for value in starts)
            or starts != sorted(set(starts))
        ):
            raise ValueError(f"paired-kept window ledger drift: {session_id}")
        frame_set = frozenset(frames)
        for start in starts:
            if not set(range(start, start + 51)).issubset(frame_set):
                raise ValueError(
                    f"paired H50 window escapes frame ledger: {session_id}/{start}"
                )
        expected_keys = {f"{value:05d}" for value in frames}
        for product_line in PRODUCT_LINES:
            observed = set(selectors[product_line]["normalized_records"][session_id])
            if observed != expected_keys:
                raise ValueError(
                    f"{product_line} frame set differs from paired ledger: {session_id}"
                )
        raw_records = selectors["RAW"]["normalized_records"][session_id]
        robot_records = selectors["ROBOT_RGB"]["normalized_records"][session_id]
        for frame_key in expected_keys:
            raw = raw_records[frame_key]
            robot = robot_records[frame_key]
            if _reference_identity(raw["metadata"]) != _reference_identity(
                robot["metadata"]
            ):
                raise ValueError(
                    f"RAW/Robot metadata ledger differs: {session_id}/{frame_key}"
                )
            if _reference_identity(raw["image"]) == _reference_identity(
                robot["image"]
            ) or raw["image"].get("sha256") == robot["image"].get("sha256"):
                raise ValueError(
                    f"RobotRGB image domain equals RAW: {session_id}/{frame_key}"
                )
        frame_ledger[session_id] = frame_set
        window_starts[session_id] = frozenset(starts)
    return {
        "selector_manifest": selector,
        "paired_kept_manifest": paired,
        "selector_reference": dict(selector_reference),
        "paired_kept_reference": dict(paired_kept_reference or {}),
        "selector_records": selector["normalized_records"],
        "selector_root": selector["selector_root"],
        "artifact_root": selector["artifact_root"],
        "artifact_roots": dict(artifact_roots),
        "frame_ledger": frame_ledger,
        "window_starts": window_starts,
        "product_line": selector["product_line"],
        "image_name": selector["image_name"],
    }


def validate_frame_images(
    adapter: str | Path,
    img_name: str,
    *,
    selector_records: Mapping[str, Mapping[str, Any]] | None = None,
    selector_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify every selected adapter metadata/image pair by exact reference."""
    if img_name != Path(img_name).name or img_name in {"", ".", ".."}:
        raise ValueError("image selector must be a frame-local basename")
    if selector_records is None or selector_root is None:
        raise RuntimeError(
            f"{FORMAL_SELECTOR_HOLD}: frame images require manifest records"
        )
    adapter_value = Path(adapter).absolute()
    adapter = validate_directory_path(
        adapter_value, allowed_roots=[adapter_value.anchor], label="HumanEgo adapter"
    )
    selector_root_value = Path(selector_root).absolute()
    selector_root = validate_directory_path(
        selector_root_value,
        allowed_roots=[selector_root_value.anchor],
        label="selector root",
    )
    paths = sorted(
        (adapter / "preprocess" / "all_data").glob("[0-9]*/training_data.json")
    )
    if not paths:
        raise ValueError(f"adapter contains no frame metadata: {adapter}")
    observed = {path.parent.name: path for path in paths}
    if not set(selector_records).issubset(observed):
        raise ValueError(
            "selector manifest contains frames absent from adapter metadata"
        )
    failures: list[str] = []
    for frame_key, record in selector_records.items():
        path = observed[frame_key]
        metadata_path, metadata_encoded = read_file_reference(
            record.get("metadata", {}),
            allowed_roots=[adapter],
            label=f"{frame_key} metadata",
        )
        if metadata_path != path.resolve(strict=True):
            raise ValueError(f"selector metadata path mismatch: {frame_key}")
        try:
            data = json.loads(metadata_encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid frame metadata: {path}") from error
        raw = data.get("obs", {}).get("rgb_path") if isinstance(data, dict) else None
        if (
            not raw
            or "clean" in Path(raw).name.lower()
            or "inpaint" in Path(raw).name.lower()
        ):
            failures.append(str(path))
            continue
        selected_path, selected_encoded = read_file_reference(
            record.get("image", {}),
            allowed_roots=[selector_root],
            label=f"{frame_key} image",
        )
        if selected_path.name != img_name:
            failures.append(
                f"{path}: selector target basename differs from manifest image_name"
            )
            continue
        if (
            cv2.imdecode(
                np.frombuffer(selected_encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            is None
        ):
            failures.append(f"{path}: undecodable selector target {selected_path}")
    if failures:
        preview = "\n  ".join(failures[:10])
        raise ValueError(
            f"image selector contract failed for {len(failures)} frames:\n  {preview}"
        )
    return {
        "frames": len(selector_records),
        "model_rgb_path": "exact selector frame image.path",
        "ordinary_files_only": True,
        "manifest_bound": True,
        "selector_root": str(selector_root),
        "clean_rgb_used": False,
    }


def validate_frozen_adapters(
    split: Mapping[str, Any],
    production_root: str | Path,
    session_ids: list[str],
    binding: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Bind adapters/status/images to one verified paired selector binding."""
    production_root = Path(production_root).resolve()
    declared_root = split.get("production_root")
    if (
        not isinstance(declared_root, str)
        or Path(declared_root).resolve() != production_root
    ):
        raise ValueError("production root differs from the frozen split")
    if not set(session_ids).issubset(binding["selector_records"]):
        raise ValueError("requested split sessions are absent from paired selector")
    adapters = [
        production_root / session_id / "09_humanego_adapter"
        for session_id in session_ids
    ]
    validate_session_paths(
        adapters,
        session_ids,
        expected_paths={
            session_id: split["sessions"][session_id]["adapter"]
            for session_id in session_ids
        },
        allowed_root=production_root,
    )
    reports = {}
    for session_id in session_ids:
        metadata = split["sessions"][session_id]
        adapter = production_root / session_id / "09_humanego_adapter"
        expected_status = metadata.get("production_status_sha256")
        if not isinstance(expected_status, str) or not SHA256_PATTERN.fullmatch(
            expected_status
        ):
            raise ValueError(f"invalid frozen production status hash: {session_id}")
        status_path = production_root / session_id / "status.json"
        if status_path.is_symlink() or sha256_file(status_path) != expected_status:
            raise ValueError(f"production status hash mismatch: {session_id}")
        reports[session_id] = validate_frame_images(
            adapter,
            str(binding["image_name"]),
            selector_records=binding["selector_records"][session_id],
            selector_root=binding["selector_root"],
        )
    return reports
