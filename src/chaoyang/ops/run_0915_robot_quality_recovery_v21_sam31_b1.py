#!/usr/bin/env python3
"""Bounded SAM3.1 B1 child runner for the 0915 Robot-quality V2.1 campaign.

This file is intentionally a new child runner.  It never edits or adopts an
old receipt.  It freezes exactly four W0 cache/contract regression sessions
plus W1-DIAG Poker044/Chips097.  W0 is never rerun through SAM; only the two
W1-DIAG rows enter the single persistent GPU model.  Inputs are physical-left
sourceIndex1 crop+resize-only RGB, Hand prompts are admitted per side from raw
direct-observed 2D evidence, and Object runs independently of Hand terminal.

The module keeps its contract helpers stdlib-only so CPU fixtures can import
and test them without importing Torch/SAM.  Heavy imports occur only inside
the future GPU worker or resize-only preparation path.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import uuid


TASK_ID = "0915_robot_quality_recovery_v21_sam31_b1"
SCHEMA = "0915-robot-recovery-v21-sam31-b1"
EXPECTED_IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
ADMITTED_SIDE = "ADMITTED_DIRECT_OBSERVED"
FIXED_REGRESSION = (
    "play_cards_0915_031",
    "play_cards_0915_119",
    "get_potato_chips_0915_007",
    "get_potato_chips_0915_042",
)
FIXED_W1_DIAG = (
    "play_cards_0915_044",
    "get_potato_chips_0915_097",
)
FIXED_ALL = FIXED_REGRESSION + FIXED_W1_DIAG
RUNTIME_REF_NAMES = {
    "sam31_checkpoint",
    "sam31_adapter",
    "legacy_hand_runner",
    "legacy_object_runner",
    "resize_only_preparer",
    "gpu_lease_wrapper",
}
TERMINAL_STATUSES = (
    "PASSED_DIAGNOSTIC",
    "COMPLETED_WITH_QUALITY_REJECTION",
    "BLOCKED_INPUT",
    "BLOCKED_RESOURCE",
    "FAILED_RUNTIME_FINAL",
)


class ContractError(RuntimeError):
    """A fail-closed input, provenance, cohort, or identity-contract error."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def verify_ref(value: Mapping[str, Any], label: str) -> Path:
    required = {"path", "bytes", "sha256"}
    if not required.issubset(value):
        raise ContractError(f"{label} lacks path/bytes/sha256")
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]):
        raise ContractError(f"{label} byte-size drift: {path}")
    if sha256_file(path) != str(value["sha256"]):
        raise ContractError(f"{label} SHA drift: {path}")
    return path


def verify_ref_under(value: Mapping[str, Any], label: str, root: Path) -> Path:
    path = verify_ref(value, label)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ContractError(f"{label} escapes its frozen write root: {path}") from error
    return path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def process_has_ancestor(pid: int, ancestor: int) -> bool:
    current = pid
    for _ in range(128):
        if current == ancestor:
            return True
        if current <= 1:
            return False
        fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
        current = int(fields[3])
    return False


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != "0915-robot-recovery-v21-sam31-b1-task-policy-v1":
        raise ContractError("B1 policy schema drift")
    cohort = policy.get("fixed_cohort", {})
    if tuple(cohort.get("regression", ())) != FIXED_REGRESSION:
        raise ContractError("W0 regression cohort drift")
    if tuple(cohort.get("w1_diag", ())) != FIXED_W1_DIAG:
        raise ContractError("W1-DIAG cohort drift")
    image = policy.get("image_domain", {})
    expected = {
        "name": EXPECTED_IMAGE_DOMAIN,
        "physical_left_source_index": 1,
        "physical_right_source_index": 0,
        "output_size": [1280, 960],
        "lens_undistortion": False,
        "remap_applied": False,
        "processed_mono_allowed": False,
        "old_rectified_fov90_cache_allowed": False,
    }
    if image != expected:
        raise ContractError("B1 image-domain policy drift")
    hand = policy.get("hand", {})
    if hand.get("whole_session_hawor_quality_is_execution_gate") is not False:
        raise ContractError("whole-session HaWoR quality must not gate B1 Hand")
    if hand.get("hawor_3d_consumed") is not False:
        raise ContractError("B1 Hand must not consume HaWoR 3D")
    obj = policy.get("object", {})
    if obj.get("depends_on_hand_terminal") is not False or obj.get("depends_on_hawor") is not False:
        raise ContractError("B1 Object must be independent of Hand/HaWoR")
    if obj.get("playing_cards", {}).get("union_mask_allowed") is not False:
        raise ContractError("Poker union mask is forbidden")
    chips = obj.get("potato_chips", {})
    if chips.get("physical_instance_slots") != 3 or chips.get("union_mask_allowed") is not False:
        raise ContractError("Chips requires three separate slots and no union")
    scope = policy.get("execution_scope", {})
    if scope.get("w0_regression_mode") != "CACHE_SHA_AND_CONTRACT_ONLY_NO_GPU_RERUN":
        raise ContractError("W0 must be cache/SHA regression only")
    if tuple(scope.get("gpu_execution_cohort", ())) != FIXED_W1_DIAG:
        raise ContractError("only W1-DIAG 044/097 may enter the GPU execution cohort")
    if scope.get("gpu_execution_session_count") != 2 or scope.get("cache_regression_session_count") != 4:
        raise ContractError("B1 execution counts must remain GPU=2/cache-regression=4")
    lease = policy.get("gpu_lease", {})
    if lease.get("single_lease") is not True or lease.get("persistent_model_load_count") != 1:
        raise ContractError("B1 must use one persistent model under one GPU lease")
    if lease.get("fresh_state_per_role_prompt") is not True:
        raise ContractError("B1 requires fresh inference state per role prompt")
    if set(policy.get("runtime_refs", {})) != RUNTIME_REF_NAMES:
        raise ContractError("B1 runtime-ref closure drift")
    if tuple(policy.get("terminal_statuses", ())) != TERMINAL_STATUSES:
        raise ContractError("B1 terminal-status vocabulary drift")


def validate_cohort(policy: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_policy(policy)
    rows = manifest.get("sessions")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ContractError("B1 input manifest must contain exactly six sessions")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ContractError("session row must be a JSON object")
        session_id = str(raw.get("session_id", ""))
        if session_id in by_id:
            raise ContractError(f"duplicate session: {session_id}")
        by_id[session_id] = dict(raw)
    if tuple(by_id) != FIXED_ALL:
        raise ContractError("session order/identity differs from frozen B1 cohort")
    for index, session_id in enumerate(FIXED_ALL):
        row = by_id[session_id]
        expected_role = "regression" if index < 4 else "w1_diag"
        if row.get("cohort_role") != expected_role:
            raise ContractError(f"cohort role drift: {session_id}")
        expected_task = "playing_cards" if session_id.startswith("play_cards_") else "potato_chips"
        if row.get("task") != expected_task:
            raise ContractError(f"task drift: {session_id}")
        if not isinstance(row.get("frame_count"), int) or int(row["frame_count"]) <= 0:
            raise ContractError(f"invalid frame denominator: {session_id}")
        if not isinstance(row.get("source_group"), str) or not row["source_group"]:
            raise ContractError(f"missing source group: {session_id}")
        if expected_role == "regression":
            cache = row.get("w0_cache_regression")
            if not isinstance(cache, dict) or set(cache) != {"hand_result", "object_result"}:
                raise ContractError(f"W0 cache refs missing: {session_id}")
    if len({row["source_group"] for row in rows}) != 6:
        raise ContractError("B1 sessions must bind six distinct source groups")
    return [by_id[session_id] for session_id in FIXED_ALL]


def validate_image_descriptor(value: Mapping[str, Any]) -> None:
    if value.get("image_domain") != EXPECTED_IMAGE_DOMAIN:
        raise ContractError("prepared RGB is not in the frozen resize-only domain")
    if value.get("remap_applied") is not False:
        raise ContractError("prepared RGB used a forbidden lens remap")


def side_plan(prompt: Mapping[str, Any] | None, cohort_role: str) -> dict[str, str]:
    if prompt is None:
        return {"left": "UNKNOWN_BLOCKED_NO_DIRECT_ANCHOR", "right": "UNKNOWN_BLOCKED_NO_DIRECT_ANCHOR"}
    expected_provider = (
        "A1_SAME_DOMAIN_RAW_DIRECT_OBSERVED_2D"
        if cohort_role == "w1_diag"
        else "FROZEN_W0_RAW_DIRECT_OBSERVED_2D"
    )
    if prompt.get("provider") != expected_provider:
        raise ContractError(f"prompt provider must be {expected_provider}")
    if prompt.get("image_domain") != EXPECTED_IMAGE_DOMAIN:
        raise ContractError("hand prompt image-domain mismatch")
    admission = prompt.get("side_admission")
    if not isinstance(admission, dict) or set(admission) != {"left", "right"}:
        raise ContractError("hand prompt must declare left/right admission")
    return {
        side: "ATTEMPT_HAND" if admission[side] == ADMITTED_SIDE else "UNKNOWN_BLOCKED_NO_DIRECT_ANCHOR"
        for side in ("left", "right")
    }


def plan_session(row: Mapping[str, Any]) -> dict[str, Any]:
    """Pure routing decision used by CPU fixtures and the GPU preflight."""

    prompt = row.get("raw_direct_prompt")
    if prompt is not None and not isinstance(prompt, dict):
        raise ContractError("raw_direct_prompt must be object or null")
    hand_sides = side_plan(prompt, str(row["cohort_role"]))
    if row["cohort_role"] == "regression":
        return {
            "session_id": row["session_id"],
            "execution_mode": "CACHE_REGRESSION_ONLY",
            "hand_sides": hand_sides,
            "hand_attempt": False,
            "object_attempt": False,
            "object_depends_on_hand_terminal": False,
            "gpu_model_consumed": False,
        }
    return {
        "session_id": row["session_id"],
        "execution_mode": "FRESH_W1_DIAG",
        "hand_sides": hand_sides,
        "hand_attempt": any(value == "ATTEMPT_HAND" for value in hand_sides.values()),
        "object_attempt": True,
        "object_depends_on_hand_terminal": False,
        "gpu_model_consumed": True,
    }


def validate_identity_evidence_ref(
    value: Mapping[str, Any] | None, session_id: str, task: str,
) -> dict[str, Any] | None:
    """Identity authority must be an immutable external file ref, never inline text."""

    if value is None:
        return None
    if task != "playing_cards":
        raise ContractError("independent same-card identity evidence is only defined for Poker")
    if not isinstance(value, Mapping) or not {"path", "bytes", "sha256"}.issubset(value):
        raise ContractError("object identity evidence must be a SHA-bound file ref or null")
    path = verify_ref(value, f"{session_id} object identity evidence")
    payload = load_json(path)
    if payload.get("session_id") != session_id:
        raise ContractError("object identity evidence session mismatch")
    if payload.get("status") not in {
        "PASSED_INDEPENDENT_SAME_CARD_SAME_FACE",
        "UNKNOWN_INSUFFICIENT_IDENTITY_EVIDENCE",
    }:
        raise ContractError("unrecognized Poker physical-identity evidence status")
    return dict(value)


def poker_identity_admission(
    instance_ids: list[str], identity_evidence: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Fail closed unless independent same-card/same-face evidence names each ID."""

    if not identity_evidence or identity_evidence.get("status") != "PASSED_INDEPENDENT_SAME_CARD_SAME_FACE":
        return {instance_id: "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY" for instance_id in instance_ids}
    admitted = set(identity_evidence.get("admitted_instance_ids", []))
    return {
        instance_id: (
            "ADMITTED_VISIBLE_INSTANCE_DIAGNOSTIC"
            if instance_id in admitted
            else "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY"
        )
        for instance_id in instance_ids
    }


def chips_instance_admission(
    instances: list[Mapping[str, Any]], union_mask_created: bool,
) -> dict[str, str]:
    """Map at most three separate tracks to three physical slots; missing slots stay UNKNOWN."""

    if union_mask_created or len(instances) > 3:
        raise ContractError("Chips union or more than three candidate instances is forbidden")
    result = {f"potato_chip_{index:02d}": "UNKNOWN_NO_SEPARATE_VISIBLE_INSTANCE" for index in range(3)}
    for row in instances:
        instance_id = str(row.get("instance_id", ""))
        if instance_id not in result:
            raise ContractError(f"unversioned Chips instance ID: {instance_id}")
        if row.get("union_mask_created") is True:
            raise ContractError("per-instance Chips union flag is forbidden")
        quality = row.get("quality", {})
        if row.get("object_mask_consumer_allowed") is True and quality.get("stable_instance") is True:
            result[instance_id] = "ADMITTED_SEPARATE_VISIBLE_INSTANCE_DIAGNOSTIC"
        else:
            result[instance_id] = "UNKNOWN_QUALITY_OR_IDENTITY_AMBIGUITY"
    return result


def validate_runtime_refs(repo_root: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, descriptor in policy.get("runtime_refs", {}).items():
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
            raise ContractError(f"invalid runtime ref: {name}")
        path = (repo_root / str(descriptor["path"])).resolve(strict=True)
        actual = file_ref(path)
        expected = descriptor.get("sha256")
        if expected is not None and actual["sha256"] != expected:
            raise ContractError(f"runtime SHA drift for {name}")
        result[name] = actual
    return result


def validate_a1_input_authority(
    policy: Mapping[str, Any], manifest: Mapping[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind W1-DIAG inputs to one immutable A1 candidate and its ledgers."""

    policy_a1 = policy.get("a1_input_authority")
    manifest_a1 = manifest.get("a1_input_authority")
    if not isinstance(policy_a1, Mapping) or not isinstance(manifest_a1, Mapping):
        raise ContractError("A1 input authority is missing")
    signature = str(policy_a1.get("candidate_signature_sha256", ""))
    if len(signature) != 64 or signature != manifest_a1.get("candidate_signature_sha256"):
        raise ContractError("A1 candidate signature drift")
    verified: dict[str, Any] = {"candidate_signature_sha256": signature}
    for name in ("candidate_freeze", "prepared_manifest", "consumer_admission"):
        if policy_a1.get(name) != manifest_a1.get(name):
            raise ContractError(f"A1 {name} differs between policy and input manifest")
        descriptor = policy_a1.get(name)
        if not isinstance(descriptor, Mapping):
            raise ContractError(f"A1 {name} ref missing")
        path = verify_ref(descriptor, f"A1 {name}")
        verified[name] = file_ref(path)
    freeze = load_json(Path(verified["candidate_freeze"]["path"]))
    if freeze.get("candidate_signature_sha256") != signature:
        raise ContractError("A1 candidate freeze semantic signature mismatch")
    prepared = load_json(Path(verified["prepared_manifest"]["path"]))
    consumer = load_json(Path(verified["consumer_admission"]["path"]))
    if prepared.get("candidate_signature_sha256") != signature or prepared.get("status") != "PASS":
        raise ContractError("A1 prepared manifest is not a passing member of the frozen candidate")
    if consumer.get("candidate_signature_sha256") != signature:
        raise ContractError("A1 consumer admission candidate signature mismatch")
    w1_rows = [row for row in rows if row.get("cohort_role") == "w1_diag"]
    prepared_rows = prepared.get("results")
    consumer_rows = consumer.get("sessions")
    if not isinstance(prepared_rows, list) or not isinstance(consumer_rows, list):
        raise ContractError("A1 prepared/consumer session ledgers are malformed")
    prepared_by_id = {str(row.get("session_id")): row for row in prepared_rows if isinstance(row, Mapping)}
    consumer_by_id = {str(row.get("session_id")): row for row in consumer_rows if isinstance(row, Mapping)}
    if tuple(prepared_by_id) != FIXED_W1_DIAG or tuple(consumer_by_id) != FIXED_W1_DIAG:
        raise ContractError("A1 ledgers do not contain exactly the frozen W1-DIAG cohort")
    expected_domain = {
        "lens_undistortion": False,
        "operation": "CROP_THEN_RESIZE_ONLY",
        "output_size": [1280, 960],
        "physical_left_source_index": 1,
        "physical_right_source_index": 0,
        "pico26_hand_consumed": False,
        "remap_applied": False,
        "trackingData_hand_consumed": False,
    }
    for row in w1_rows:
        session_id = str(row["session_id"])
        prepared_row = prepared_by_id[session_id]
        consumer_row = consumer_by_id[session_id]
        if any(prepared_row.get(name) != row.get(name) for name in ("task", "source_group", "frame_count")):
            raise ContractError(f"A1 prepared session identity drift: {session_id}")
        if prepared_row.get("input_domain") != expected_domain:
            raise ContractError(f"A1 prepared image-domain drift: {session_id}")
        rgb = row.get("prepared_rgb")
        if not isinstance(rgb, Mapping) or prepared_row.get("prepared_video") != {
            name: rgb.get(name) for name in ("path", "bytes", "sha256")
        }:
            raise ContractError(f"A1 prepared RGB is not ledger-bound: {session_id}")
        prompt = row.get("raw_direct_prompt")
        if not isinstance(prompt, Mapping) or consumer_row.get("npz") != {
            name: prompt.get(name) for name in ("path", "bytes", "sha256")
        }:
            raise ContractError(f"A1 prompt NPZ is not consumer-ledger-bound: {session_id}")
        if consumer_row.get("runtime_terminal") != prompt.get("whole_session_quality"):
            raise ContractError(f"A1 prompt terminal drift: {session_id}")
        sides = consumer_row.get("sides")
        if not isinstance(sides, list) or [side.get("side") for side in sides] != ["left", "right"]:
            raise ContractError(f"A1 consumer side ledger drift: {session_id}")
        for side_row in sides:
            side = str(side_row["side"])
            if prompt.get("side_admission", {}).get(side) == ADMITTED_SIDE:
                if side_row.get("diagnostic_review_admitted") is not True:
                    raise ContractError(f"A1 side lacks diagnostic admission: {session_id}/{side}")
    return verified


def validate_w0_cache_regression(row: Mapping[str, Any]) -> dict[str, Any]:
    """Verify old W0 Hand/Object terminals without invoking SAM or changing them."""

    if row.get("cohort_role") != "regression":
        raise ContractError("cache regression validation is W0-only")
    cache = row.get("w0_cache_regression")
    if not isinstance(cache, Mapping):
        raise ContractError(f"W0 cache refs missing: {row.get('session_id')}")
    result: dict[str, Any] = {
        "execution_mode": "CACHE_REGRESSION_ONLY",
        "gpu_model_consumed": False,
        "sam_process_session_called": False,
    }
    for capability in ("hand", "object"):
        descriptor = cache.get(f"{capability}_result")
        if not isinstance(descriptor, Mapping):
            raise ContractError(f"W0 {capability} cache ref missing: {row.get('session_id')}")
        path = verify_ref(descriptor, f"{row.get('session_id')} W0 cached {capability}")
        payload = load_json(path)
        result[capability] = {
            "result": file_ref(path),
            "reported_status": payload.get("status"),
        }
    return result


def prepare_or_verify_rgb(
    repo_root: Path, output_root: Path, row: dict[str, Any],
) -> dict[str, Any]:
    prepared = row.get("prepared_rgb")
    if isinstance(prepared, dict):
        validate_image_descriptor(prepared)
        verify_ref(prepared, f"{row['session_id']} prepared_rgb")
        return dict(prepared)
    if row.get("cohort_role") != "w1_diag":
        raise ContractError(f"regression RGB missing: {row['session_id']}")
    source = row.get("source_stereo")
    camera = row.get("camera_params")
    if not isinstance(source, dict) or not isinstance(camera, dict):
        raise ContractError(f"W1-DIAG raw source binding missing: {row['session_id']}")
    verify_ref(source, f"{row['session_id']} source_stereo")
    verify_ref(camera, f"{row['session_id']} camera_params")

    # Reuse the frozen W0 sourceIndex1 crop+resize-only implementation exactly.
    sys.path.insert(0, str((repo_root / "src").resolve()))
    preparer = importlib.import_module("chaoyang.ops.run_0915_robot15h_hawor_wave0_v1")
    destination = output_root / "prepared" / str(row["session_id"])
    prepared_row = preparer.prepare_session(row, destination)
    descriptor = dict(prepared_row["prepared_video"])
    descriptor.update({"image_domain": EXPECTED_IMAGE_DOMAIN, "remap_applied": False})
    validate_image_descriptor(descriptor)
    verify_ref(descriptor, f"{row['session_id']} newly prepared RGB")
    return descriptor


def derive_admitted_prompt_view(
    destination: Path, row: Mapping[str, Any], prompt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Create a SHA-bound per-side view; never invent a missing anchor."""

    plan = side_plan(prompt, str(row["cohort_role"]))
    if prompt is None:
        return None, {"side_plan": plan, "source": None}
    source_path = verify_ref(prompt, f"{row['session_id']} raw_direct_prompt")
    import numpy as np

    with np.load(source_path, allow_pickle=False) as archive:
        required = {"joints_2d", "observed", "detector_confidence", "anatomical_side_names"}
        if not required.issubset(archive.files):
            raise ContractError(f"prompt keys missing: {row['session_id']}")
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        confidence = np.asarray(archive["detector_confidence"], np.float64)
        sides = np.asarray(archive["anatomical_side_names"]).astype(str)
    expected_frames = int(row["frame_count"])
    if joints.shape != (2, expected_frames, 21, 2):
        raise ContractError(f"prompt joints shape drift: {row['session_id']}")
    if observed.shape != (2, expected_frames) or confidence.shape != (2, expected_frames):
        raise ContractError(f"prompt observation axis drift: {row['session_id']}")
    if sides.tolist() != ["left", "right"]:
        raise ContractError(f"prompt side order drift: {row['session_id']}")
    admitted = np.asarray([plan[side] == "ATTEMPT_HAND" for side in ("left", "right")], bool)
    observed_counts = {"left": int(observed[0].sum()), "right": int(observed[1].sum())}
    if prompt.get("direct_observed_frames") != observed_counts:
        raise ContractError(f"prompt direct-observed counts drift: {row['session_id']}")
    for index, side in enumerate(("left", "right")):
        if admitted[index] and observed_counts[side] == 0:
            raise ContractError(f"admitted prompt side has no direct observation: {row['session_id']}/{side}")
    view_observed = observed & admitted[:, None]
    view_confidence = np.where(admitted[:, None], confidence, 0.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}.npz")
    np.savez_compressed(
        temporary,
        joints_2d=joints,
        observed=view_observed,
        detector_confidence=view_confidence,
        anatomical_side_names=sides,
    )
    os.replace(temporary, destination)
    evidence = {
        "source": file_ref(source_path),
        "view": file_ref(destination),
        "side_plan": plan,
        "direct_observed_frames": {
            "left": int(view_observed[0].sum()),
            "right": int(view_observed[1].sum()),
        },
        "whole_session_hawor_quality_consumed": False,
        "hawor_3d_consumed": False,
    }
    return evidence["view"], evidence


def build_freeze(
    repo_root: Path,
    output_root: Path,
    policy_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = load_json(policy_path)
    manifest = load_json(manifest_path)
    rows = validate_cohort(policy, manifest)
    verify_ref(manifest["cohort_manifest"], "frozen cohort manifest")
    a1_authority = validate_a1_input_authority(policy, manifest, rows)
    runtimes = validate_runtime_refs(repo_root, policy)
    prepared_rows: list[dict[str, Any]] = []
    prompt_audit: list[dict[str, Any]] = []
    for row in rows:
        prepared = prepare_or_verify_rgb(repo_root, output_root, row)
        cache_regression = None
        if row["cohort_role"] == "regression":
            prompt = row.get("raw_direct_prompt")
            if not isinstance(prompt, Mapping):
                raise ContractError(f"W0 prompt ref missing: {row['session_id']}")
            prompt_path = verify_ref(prompt, f"{row['session_id']} frozen W0 raw prompt")
            prompt_view = None
            prompt_evidence = {
                "source": file_ref(prompt_path),
                "view": None,
                "side_plan": side_plan(prompt, "regression"),
                "execution_mode": "CACHE_REGRESSION_ONLY",
                "gpu_model_consumed": False,
            }
            cache_regression = validate_w0_cache_regression(row)
        else:
            prompt_view, prompt_evidence = derive_admitted_prompt_view(
                output_root / "prompt_views" / f"{row['session_id']}.npz",
                row,
                row.get("raw_direct_prompt"),
            )
            prompt_evidence["execution_mode"] = "FRESH_W1_DIAG"
            prompt_evidence["gpu_model_consumed"] = True
        prompt_audit.append({"session_id": row["session_id"], **prompt_evidence})
        identity_evidence = validate_identity_evidence_ref(
            row.get("object_identity_evidence"), str(row["session_id"]), str(row["task"]),
        )
        prepared_rows.append({
            **row,
            "prepared_rgb": prepared,
            "admitted_prompt_view": prompt_view,
            "object_identity_evidence": identity_evidence,
            "execution_plan": plan_session(row),
            "w0_cache_regression": cache_regression,
        })
    atomic_json(output_root / "PREPARED_RGB_MANIFEST.json", {
        "schema_version": f"{SCHEMA}-prepared-rgb-manifest-v1",
        "sessions": [{
            "session_id": row["session_id"],
            "prepared_rgb": row["prepared_rgb"],
        } for row in prepared_rows],
    })
    atomic_json(output_root / "PER_SIDE_PROMPT_ADMISSION.json", {
        "schema_version": f"{SCHEMA}-per-side-prompt-admission-v1",
        "sessions": prompt_audit,
    })
    payload = {
        "schema_version": f"{SCHEMA}-candidate-freeze-v1",
        "candidate_id": policy["policy_id"],
        "task_id": TASK_ID,
        "policy": file_ref(policy_path),
        "input_manifest": file_ref(manifest_path),
        "child_runner": file_ref(Path(__file__)),
        "runtime_refs": runtimes,
        "a1_input_authority": a1_authority,
        "gpu_lease": policy["gpu_lease"],
        "gpu_execution_cohort": list(FIXED_W1_DIAG),
        "cache_regression_cohort": list(FIXED_REGRESSION),
        "sessions": [{
            "session_id": row["session_id"],
            "task": row["task"],
            "cohort_role": row["cohort_role"],
            "frame_count": row["frame_count"],
            "source_group": row["source_group"],
            "prepared_rgb": row["prepared_rgb"],
            "admitted_prompt_view": row["admitted_prompt_view"],
            "execution_plan": row["execution_plan"],
            "w0_cache_regression": row.get("w0_cache_regression"),
            "object_identity_evidence": row.get("object_identity_evidence"),
        } for row in prepared_rows],
        "claim_limit": (
            "Development diagnostic SAM Hand/Object evidence only; no Mask accuracy, physical "
            "identity truth, hidden surface, Contact, training, control or deployment authority."
        ),
    }
    freeze = {**payload, "candidate_freeze_sha256": canonical_sha(payload)}
    atomic_json(output_root / "CANDIDATE_FREEZE_V1.json", freeze)
    return freeze, prepared_rows


def validate_claim(
    claim_path: Path, freeze_sha: str, run_signature_sha: str,
    expected_output_root: Path, expected_visual_root: Path, expected_executor_epoch: int,
    require_descendant: bool,
) -> dict[str, Any]:
    claim = load_json(claim_path)
    pid = claim.get("pid")
    valid = (
        claim.get("task_id") == TASK_ID
        and claim.get("status") == "CLAIMED"
        and claim.get("candidate_freeze_sha256") == freeze_sha
        and claim.get("run_signature_sha256") == run_signature_sha
        and claim.get("unique_write_root") == str(expected_output_root.resolve())
        and claim.get("visual_write_root") == str(expected_visual_root.resolve())
        and claim.get("executor_epoch") == expected_executor_epoch
        and isinstance(claim.get("fencing_token_sha256"), str)
        and len(claim["fencing_token_sha256"]) == 64
        and isinstance(pid, int)
        and process_start_ticks(pid) == claim.get("proc_start_ticks")
    )
    if not valid:
        raise ContractError("B1 writer claim mismatch")
    if require_descendant and not process_has_ancestor(os.getpid(), int(pid)):
        raise ContractError("B1 GPU worker is outside writer ancestry")
    return claim


def object_admission_sidecar(
    task: str,
    object_manifest: Mapping[str, Any],
    identity_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    instances = object_manifest.get("instances", [])
    if not isinstance(instances, list):
        raise ContractError("object instance manifest lacks instances")
    if object_manifest.get("union_mask_created") is not False:
        raise ContractError("union mask is forbidden")
    instance_ids = [str(row.get("instance_id", "")) for row in instances if isinstance(row, Mapping)]
    if len(instance_ids) != len(instances) or len(instance_ids) != len(set(instance_ids)):
        raise ContractError("object instance IDs must be present and unique")
    if task == "playing_cards":
        allowed = {f"playing_card_{index:02d}" for index in range(3)}
        if len(instance_ids) > 3 or any(instance_id not in allowed for instance_id in instance_ids):
            raise ContractError("Poker instance IDs exceed the frozen three-instance namespace")
        statuses = poker_identity_admission(instance_ids, identity_evidence)
        unresolved_task_level_instances = 1 if not instances else 0
    elif task == "potato_chips":
        statuses = chips_instance_admission(instances, False)
        unresolved_task_level_instances = 0
    else:
        raise ContractError(f"unsupported task: {task}")
    return {
        "schema_version": f"{SCHEMA}-object-admission-v1",
        "task": task,
        "instance_status": statuses,
        "union_mask_created": False,
        "visible_surface_only": True,
        "hidden_shape_inferred": False,
        "consumer_allowed_instances": sum(value.startswith("ADMITTED_") for value in statuses.values()),
        "unknown_instances": (
            sum(value.startswith("UNKNOWN_") for value in statuses.values())
            + unresolved_task_level_instances
        ),
        "unresolved_task_level_instances": unresolved_task_level_instances,
        "mask_accuracy_claimed": False,
        "physical_identity_ground_truth": False,
    }


def run_gpu_worker(
    repo_root: Path,
    output_root: Path,
    visual_root: Path,
    claim_path: Path,
    freeze_path: Path,
) -> int:
    freeze = load_json(freeze_path)
    stored = freeze.pop("candidate_freeze_sha256", None)
    if stored != canonical_sha(freeze):
        raise ContractError("candidate freeze digest mismatch")
    freeze["candidate_freeze_sha256"] = stored
    if freeze.get("task_id") != TASK_ID:
        raise ContractError("candidate freeze task identity drift")
    if tuple(row.get("session_id") for row in freeze.get("sessions", [])) != FIXED_ALL:
        raise ContractError("candidate freeze cohort drift")
    signature = load_json(output_root / "RUN_SIGNATURE.json")
    signature_sha = signature.pop("run_signature_sha256", None)
    if signature_sha != canonical_sha(signature):
        raise ContractError("run signature digest mismatch")
    if signature.get("candidate_freeze", {}).get("sha256") != sha256_file(freeze_path):
        raise ContractError("run signature candidate-freeze reference drift")
    if signature.get("task_id") != TASK_ID or signature.get("gpu_lease") != freeze.get("gpu_lease"):
        raise ContractError("run signature task/lease drift")
    executor_epoch = signature.get("executor_epoch")
    if not isinstance(executor_epoch, int) or executor_epoch < 1:
        raise ContractError("run signature executor epoch is invalid")
    if signature.get("output_root") != str(output_root.resolve()):
        raise ContractError("run signature output-root drift")
    if signature.get("visual_root") != str(visual_root.resolve()):
        raise ContractError("run signature visual-root drift")
    if tuple(signature.get("fixed_cohort", ())) != FIXED_ALL:
        raise ContractError("run signature cohort drift")
    validate_claim(
        claim_path, str(stored), str(signature_sha), output_root, visual_root,
        executor_epoch,
        require_descendant=True,
    )
    if visual_root.exists() or visual_root.is_symlink():
        raise ContractError("fresh B1 visual root is required at worker start")
    verify_ref(freeze["child_runner"], "frozen child runner")
    for name, descriptor in freeze.get("runtime_refs", {}).items():
        verify_ref(descriptor, f"frozen runtime {name}")

    # W0 is a cache/contract regression fence only.  Validate and record those
    # immutable terminals before any model is loaded; never call process_session.
    results: list[dict[str, Any]] = []
    for row in freeze["sessions"][:4]:
        if row.get("cohort_role") != "regression":
            raise ContractError("first four frozen rows must be W0 regression")
        plan = row.get("execution_plan", {})
        if plan.get("execution_mode") != "CACHE_REGRESSION_ONLY":
            raise ContractError(f"W0 execution-mode drift: {row.get('session_id')}")
        verify_ref(row["prepared_rgb"], f"{row['session_id']} cached prepared RGB")
        cache = row.get("w0_cache_regression")
        if not isinstance(cache, Mapping):
            raise ContractError(f"W0 cache closure missing: {row.get('session_id')}")
        for capability in ("hand", "object"):
            ref = cache.get(capability, {}).get("result")
            if not isinstance(ref, Mapping):
                raise ContractError(f"W0 cached {capability} result missing: {row.get('session_id')}")
            verify_ref(ref, f"{row['session_id']} W0 cached {capability}")
        results.append({
            "session_id": row["session_id"],
            "task": row["task"],
            "execution_mode": "CACHE_REGRESSION_ONLY",
            "gpu_model_consumed": False,
            "sam_process_session_called": False,
            "cache_regression": cache,
        })

    gpu_rows = freeze["sessions"][4:]
    if tuple(row.get("session_id") for row in gpu_rows) != FIXED_W1_DIAG:
        raise ContractError("GPU cohort must be exactly Poker044/Chips097")
    for row in gpu_rows:
        if row.get("execution_plan", {}).get("execution_mode") != "FRESH_W1_DIAG":
            raise ContractError(f"W1 execution-mode drift: {row.get('session_id')}")

    sys.path.insert(0, str((repo_root / "src").resolve()))
    adapter_module = importlib.import_module("chaoyang.pipeline.sam31_compat_adapter_v1")
    hand_module = importlib.import_module("chaoyang.ops.run_0915_robot15h_sam31_temporal_identity_v1")
    object_module = importlib.import_module("chaoyang.ops.run_0915_robot15h_sam31_task_object_wave0_v1")

    checkpoint = Path(str(freeze["runtime_refs"]["sam31_checkpoint"]["path"]))
    code_root = repo_root / "vendor/SAM3"
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=code_root,
        checkpoint_path=checkpoint,
    )
    model = adapter.model
    hand_module.OUTPUT = output_root / "hand"
    hand_module.VISUAL = visual_root / "hand"
    object_module.OUTPUT = output_root / "object"
    object_module.VISUAL = visual_root / "object"
    hand_module.OUTPUT.mkdir(parents=True, exist_ok=True)
    object_module.OUTPUT.mkdir(parents=True, exist_ok=True)

    runtime_failures = 0
    for row in gpu_rows:
        session_id = str(row["session_id"])
        verify_ref(row["prepared_rgb"], f"{session_id} prepared RGB at worker start")
        if row.get("admitted_prompt_view") is not None:
            verify_ref(row["admitted_prompt_view"], f"{session_id} prompt view at worker start")
        runtime_row = {
            "session_id": session_id,
            "task": row["task"],
            "source_group": row["source_group"],
            "frame_count": int(row["frame_count"]),
            "prepared_video": row["prepared_rgb"]["path"],
        }
        session_result: dict[str, Any] = {
            "session_id": session_id,
            "task": row["task"],
            "execution_mode": "FRESH_W1_DIAG",
            "gpu_model_consumed": True,
            "object_attempt_independent_of_hand": True,
            "quality_rejections": [],
        }
        prompt = row.get("admitted_prompt_view")
        if prompt is None:
            blocked = {
                "schema_version": f"{SCHEMA}-hand-blocked-v1",
                "session_id": session_id,
                "status": "BLOCKED_PROMPT_SOURCE",
                "side_status": row["execution_plan"]["hand_sides"],
                "whole_session_hawor_quality_consumed": False,
            }
            path = output_root / "hand" / "sessions" / row["task"] / session_id / "RESULT.json"
            atomic_json(path, blocked)
            session_result["hand"] = file_ref(path)
            session_result["quality_rejections"].extend(
                f"HAND_{side}_{status}"
                for side, status in row["execution_plan"]["hand_sides"].items()
            )
        else:
            try:
                runtime_row["hawor_npz"] = prompt["path"]
                hand_result = hand_module.process_session(model, runtime_row)
                verify_ref_under(
                    hand_result["result"], f"{session_id} Hand result", hand_module.OUTPUT,
                )
                session_result["hand"] = hand_result["result"]
                side_results = hand_result.get("side_results")
                if not isinstance(side_results, list) or len(side_results) != 2:
                    raise ContractError("legacy Hand result lacks exactly two side terminals")
                if [side.get("role") for side in side_results] != ["left_hand", "right_hand"]:
                    raise ContractError("legacy Hand role identity/order drift")
                session_result["hand_reported_status"] = hand_result.get("status")
                session_result["hand_side_statuses"] = {
                    str(side.get("role")): str(side.get("status")) for side in side_results
                }
                session_result["quality_rejections"].extend(
                    f"HAND_{side.get('role')}_{side.get('status')}"
                    for side in side_results
                    if not str(side.get("status", "")).startswith("PASS_HAND_")
                )
            except Exception as error:  # independent Object path must still run
                runtime_failures += 1
                path = output_root / "hand" / "failures" / f"{session_id}.json"
                atomic_json(path, {
                    "schema_version": f"{SCHEMA}-capability-failure-v1",
                    "session_id": session_id,
                    "capability": "hand",
                    "status": "FAILED_RUNTIME",
                    "error": f"{type(error).__name__}: {error}",
                })
                session_result["hand"] = file_ref(path)

        try:
            object_result = object_module.process_session(model, runtime_row)
            manifest_ref = object_result["instance_manifest"]
            manifest = load_json(verify_ref_under(
                manifest_ref, f"{session_id} object manifest", object_module.OUTPUT,
            ))
            evidence = row.get("object_identity_evidence")
            if evidence is not None:
                evidence = load_json(verify_ref(evidence, f"{session_id} object identity evidence"))
            admission = object_admission_sidecar(str(row["task"]), manifest, evidence)
            admission_path = output_root / "object_admission" / row["task"] / session_id / "B1_OBJECT_ADMISSION.json"
            atomic_json(admission_path, {"session_id": session_id, **admission})
            verify_ref_under(
                object_result["result"], f"{session_id} Object result", object_module.OUTPUT,
            )
            session_result["object"] = object_result["result"]
            session_result["object_admission"] = file_ref(admission_path)
            session_result["object_reported_status"] = object_result.get("status")
            if object_result.get("status") != "PASSED_VISIBLE_OBJECT_MASK_PROXY":
                session_result["quality_rejections"].append(
                    f"OBJECT_{object_result.get('status', 'MISSING_TERMINAL')}"
                )
            if int(admission["unknown_instances"]):
                session_result["quality_rejections"].append("OBJECT_ADMISSION_UNKNOWN_REMAINS")
        except Exception as error:
            runtime_failures += 1
            path = output_root / "object" / "failures" / f"{session_id}.json"
            atomic_json(path, {
                "schema_version": f"{SCHEMA}-capability-failure-v1",
                "session_id": session_id,
                "capability": "object",
                "status": "FAILED_RUNTIME",
                "error": f"{type(error).__name__}: {error}",
            })
            session_result["object"] = file_ref(path)
        results.append(session_result)

    quality_rejection_count = sum(len(row.get("quality_rejections", ())) for row in results)
    batch = {
        "schema_version": f"{SCHEMA}-batch-result-v1",
        "status": "COMPLETED_WITH_RUNTIME_FAILURE" if runtime_failures else "COMPLETED_ALL_TERMINAL",
        "model_load_count": 1,
        "gpu_execution_sessions": list(FIXED_W1_DIAG),
        "gpu_execution_session_count": 2,
        "cache_regression_sessions": list(FIXED_REGRESSION),
        "cache_regression_session_count": 4,
        "w0_sam_rerun_performed": False,
        "fresh_state_per_role_prompt": True,
        "runtime_failures": runtime_failures,
        "quality_rejection_count": quality_rejection_count,
        "sessions": results,
        "mask_accuracy_claimed": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "model_build_evidence": build_evidence,
    }
    atomic_json(output_root / "BATCH_RESULT.json", batch)
    return 2 if runtime_failures else 0


def write_terminal(
    output_root: Path,
    receipt_path: Path,
    status: str,
    first_blocker: str | None,
) -> None:
    result = {
        "schema_version": f"{SCHEMA}-result-v1",
        "task_id": TASK_ID,
        "status": status,
        "first_blocker": first_blocker,
        "candidate_freeze": file_ref(output_root / "CANDIDATE_FREEZE_V1.json")
        if (output_root / "CANDIDATE_FREEZE_V1.json").is_file() else None,
        "run_signature": file_ref(output_root / "RUN_SIGNATURE.json")
        if (output_root / "RUN_SIGNATURE.json").is_file() else None,
        "batch_result": file_ref(output_root / "BATCH_RESULT.json")
        if (output_root / "BATCH_RESULT.json").is_file() else None,
        "gpu_command_receipt": file_ref(output_root / "GPU_COMMAND_RECEIPT.json")
        if (output_root / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "mask_accuracy_claimed": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Development diagnostic only; no Mask accuracy, identity ground truth, hidden "
            "surface, Contact, training, control or deployment authority."
        ),
    }
    atomic_json(output_root / "RESULT.json", result)
    receipt = {**result, "result": file_ref(output_root / "RESULT.json")}
    atomic_json(receipt_path, receipt)
    atomic_json(output_root / "RUN_RECEIPT.json", {
        "schema_version": f"{SCHEMA}-run-receipt-v1",
        "task_id": TASK_ID,
        "status": status,
        "result": file_ref(output_root / "RESULT.json"),
        "terminal_receipt": file_ref(receipt_path),
    })


def classify_terminal(batch: Mapping[str, Any]) -> tuple[str, str | None]:
    runtime_failures = batch.get("runtime_failures")
    quality_rejections = batch.get("quality_rejection_count")
    if not isinstance(runtime_failures, int) or runtime_failures < 0:
        raise ContractError("batch runtime-failure count is invalid")
    if not isinstance(quality_rejections, int) or quality_rejections < 0:
        raise ContractError("batch quality-rejection count is invalid")
    if runtime_failures:
        return "FAILED_RUNTIME_FINAL", "ONE_OR_MORE_CAPABILITIES_FAILED_RUNTIME"
    if quality_rejections:
        return "COMPLETED_WITH_QUALITY_REJECTION", "FAIL_CLOSED_QUALITY_OR_UNKNOWN_REMAINS"
    return "PASSED_DIAGNOSTIC", None


def run_orchestrator(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    visual_root = args.visual_root.resolve()
    receipt_path = args.receipt.resolve()
    if (output_root.exists() or output_root.is_symlink()
            or visual_root.exists() or visual_root.is_symlink()
            or receipt_path.exists() or receipt_path.is_symlink()):
        raise ContractError("fresh B1 output, visual and receipt paths are required")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise ContractError("positive executor epoch and fencing token >=16 characters required")
    output_root.mkdir(parents=True)
    try:
        freeze, _rows = build_freeze(
            args.repo_root.resolve(strict=True), output_root,
            args.policy.resolve(strict=True), args.input_manifest.resolve(strict=True),
        )
    except Exception as error:
        write_terminal(output_root, receipt_path, "BLOCKED_INPUT", f"{type(error).__name__}: {error}")
        return 3
    signature_payload = {
        "schema_version": f"{SCHEMA}-run-signature-v1",
        "task_id": TASK_ID,
        "executor_epoch": args.executor_epoch,
        "candidate_freeze": file_ref(output_root / "CANDIDATE_FREEZE_V1.json"),
        "output_root": str(output_root),
        "visual_root": str(visual_root),
        "gpu_lease": freeze["gpu_lease"],
        "fixed_cohort": list(FIXED_ALL),
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    atomic_json(output_root / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": f"{SCHEMA}-writer-claim-v1",
        "task_id": TASK_ID,
        "status": "CLAIMED",
        "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output_root),
        "visual_write_root": str(visual_root),
    }
    atomic_json(output_root / "CLAIM.json", claim)
    lease_policy = freeze["gpu_lease"]
    wrapper = Path(str(freeze["runtime_refs"]["gpu_lease_wrapper"]["path"]))
    worker = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--repo-root", str(args.repo_root.resolve()),
        "--output-root", str(output_root), "--visual-root", str(visual_root),
        "--claim", str(output_root / "CLAIM.json"),
        "--freeze", str(output_root / "CANDIDATE_FREEZE_V1.json"),
    ]
    lease = [
        sys.executable, str(wrapper),
        "--task-id", TASK_ID, "--attempt-id", output_root.name,
        "--executor-epoch", str(args.executor_epoch), "--priority", "CANARY",
        "--gpu-id", str(lease_policy["gpu_id"]),
        "--min-free-mib", str(lease_policy["minimum_free_mib"]),
        "--wait-seconds", str(lease_policy["wait_seconds"]),
        "--wall-seconds", str(lease_policy["wall_seconds"]),
        "--receipt", str(output_root / "GPU_COMMAND_RECEIPT.json"),
        "--claim-limit", freeze["claim_limit"], "--", *worker,
    ]
    atomic_json(output_root / "COMMAND.json", {"worker_command": worker, "lease_command": lease})
    log_path = output_root / "GPU_WRAPPER.log"
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        completed = subprocess.run(lease, cwd=args.repo_root, stdout=log, stderr=subprocess.STDOUT, check=False)
    gpu = load_json(output_root / "GPU_COMMAND_RECEIPT.json") if (output_root / "GPU_COMMAND_RECEIPT.json").is_file() else {}
    if gpu.get("status") == "BLOCKED_RESOURCE":
        write_terminal(output_root, receipt_path, "BLOCKED_RESOURCE", str(gpu.get("error") or gpu.get("reason")))
        return 3
    if completed.returncode != 0 or gpu.get("status") != "PASSED":
        write_terminal(output_root, receipt_path, "FAILED_RUNTIME_FINAL", str(gpu.get("error") or "B1_GPU_WORKER_FAILED"))
        return 2
    batch = load_json(output_root / "BATCH_RESULT.json")
    status, blocker = classify_terminal(batch)
    write_terminal(output_root, receipt_path, status, blocker)
    return 0 if status in {"PASSED_DIAGNOSTIC", "COMPLETED_WITH_QUALITY_REJECTION"} else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--executor-epoch", type=int, default=1)
    parser.add_argument("--fencing-token", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.claim is None or args.freeze is None:
            raise ContractError("worker requires --claim and --freeze")
        return run_gpu_worker(
            args.repo_root.resolve(strict=True), args.output_root.resolve(), args.visual_root.resolve(),
            args.claim.resolve(strict=True), args.freeze.resolve(strict=True),
        )
    if args.policy is None or args.input_manifest is None or args.receipt is None:
        raise ContractError("orchestrator requires --policy, --input-manifest and --receipt")
    return run_orchestrator(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(json.dumps({"status": "BLOCKED_INPUT", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
