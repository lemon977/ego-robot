#!/usr/bin/env python3
"""Bounded Poker044 Hand-only successor for the B1 tracker-frame-state failure.

The parent B1 semantic staging is evidence only and is never consumed.  This
runner rebuilds the prompt view from the original A1 NPZ, reloads the pinned
SAM3.1 model, and calls the frozen legacy Hand process_session.  Its only
algorithmic interception converts the exact right-fallback reverse-propagation
KeyError(164), before any frame is yielded, into UNKNOWN_DIRECTION_INCOMPLETE.
Every unreturned mask remains zero/UNKNOWN and the affected side is forced to
the named quality rejection REJECTED_TRACKER_DIRECTION_INCOMPLETE.
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
from typing import Any, Callable, Mapping
import uuid


TASK_ID = "0915_robot_quality_recovery_v21_sam31_b1r_poker044_hand"
SCHEMA = "0915-robot-recovery-v21-sam31-b1r"
SESSION_ID = "play_cards_0915_044"
TASK = "playing_cards"
FRAME_COUNT = 166
SOURCE_GROUP = "0915:cards_120_0915:044:2fae5fbe35ad430fae7a45320ee03529"
IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
RUNTIME_NAMES = {
    "sam31_checkpoint",
    "sam31_adapter",
    "legacy_hand_runner",
    "sam31_tracker_source",
    "gpu_lease_wrapper",
}
TERMINALS = (
    "COMPLETED_WITH_QUALITY_REJECTION",
    "BLOCKED_INPUT",
    "BLOCKED_RESOURCE",
    "FAILED_RUNTIME_FINAL",
)


class ContractError(RuntimeError):
    """Fail-closed B1R contract error."""


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
    if not {"path", "bytes", "sha256"}.issubset(value):
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
        raise ContractError(f"{label} escapes B1R write root: {path}") from error
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
        current = int(Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()[3])
    return False


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != "0915-robot-recovery-v21-sam31-b1r-task-policy-v1":
        raise ContractError("B1R policy schema drift")
    fixed = policy.get("fixed_session", {})
    expected_fixed = {
        "session_id": SESSION_ID,
        "task": TASK,
        "frame_count": FRAME_COUNT,
        "source_group": SOURCE_GROUP,
        "capability": "HAND_ONLY",
    }
    if fixed != expected_fixed:
        raise ContractError("B1R fixed Hand-only session drift")
    recovery = policy.get("recovery_contract", {})
    expected_recovery = {
        "parent_terminal": "FAILED_RUNTIME_FINAL",
        "parent_failure": "KeyError: 164",
        "located_stage": "RIGHT_HAND_FALLBACK_REVERSE_PROPAGATION_FIRST_FRAME",
        "fallback_anchor_frame": 165,
        "reverse_first_frame": 164,
        "allowed_exception_class": "KeyError",
        "allowed_exception_args": [164],
        "allowed_direction": "backward",
        "allowed_yielded_frames_before_exception": 0,
        "conversion_status": "UNKNOWN_DIRECTION_INCOMPLETE",
        "affected_side_terminal": "REJECTED_TRACKER_DIRECTION_INCOMPLETE",
        "uncovered_frame_semantics": "UNKNOWN_NOT_ABSENT",
        "all_other_exceptions": "FAILED_RUNTIME_FINAL",
        "maximum_matching_conversions": 1,
    }
    if recovery != expected_recovery:
        raise ContractError("B1R recovery exception signature drift")
    immutable = policy.get("immutability", {})
    required_false = (
        "reuse_parent_failed_semantic_as_input",
        "prompt_changed",
        "thresholds_changed",
        "weights_changed",
        "legacy_hand_algorithm_changed",
    )
    if any(immutable.get(name) is not False for name in required_false):
        raise ContractError("B1R may not change inputs, prompts, thresholds, weights or legacy Hand")
    if set(policy.get("runtime_refs", {})) != RUNTIME_NAMES:
        raise ContractError("B1R runtime closure drift")
    lease = policy.get("gpu_lease", {})
    if (lease.get("single_lease") is not True
            or lease.get("persistent_model_load_count") != 1
            or lease.get("gpu_execution_session_count") != 1
            or lease.get("fresh_state_per_role_prompt") is not True):
        raise ContractError("B1R must reload one model for one fresh Hand session")
    if tuple(policy.get("terminal_statuses", ())) != TERMINALS:
        raise ContractError("B1R terminal vocabulary drift")


def validate_parent_and_a1(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != "0915-robot-recovery-v21-sam31-b1r-input-v1":
        raise ContractError("B1R input schema drift")
    session = manifest.get("session")
    if session != {
        "session_id": SESSION_ID,
        "task": TASK,
        "frame_count": FRAME_COUNT,
        "source_group": SOURCE_GROUP,
    }:
        raise ContractError("B1R session identity drift")
    forbidden = manifest.get("forbidden_inputs", {})
    if set(forbidden) != {
        "parent_failed_runtime_staging",
        "parent_hand_semantic",
        "parent_object_semantic",
        "parent_prompt_view",
    } or any(value is not True for value in forbidden.values()):
        raise ContractError("B1R forbidden parent-derived input list drift")

    parent = manifest.get("parent_b1")
    a1 = manifest.get("a1_authority")
    if not isinstance(parent, Mapping) or not isinstance(a1, Mapping):
        raise ContractError("B1R parent/A1 authority missing")
    verified: dict[str, Any] = {"parent_b1": {}, "a1_authority": {}}
    for name in ("result", "batch_result", "candidate_freeze", "hand_failure", "runner"):
        descriptor = parent.get(name)
        if not isinstance(descriptor, Mapping):
            raise ContractError(f"parent B1 {name} ref missing")
        path = verify_ref(descriptor, f"parent B1 {name}")
        verified["parent_b1"][name] = file_ref(path)
    for name in ("consumer_admission", "prepared_rgb", "raw_direct_prompt"):
        descriptor = a1.get(name)
        if not isinstance(descriptor, Mapping):
            raise ContractError(f"A1 {name} ref missing")
        path = verify_ref(descriptor, f"A1 {name}")
        verified["a1_authority"][name] = file_ref(path)

    parent_result = load_json(Path(verified["parent_b1"]["result"]["path"]))
    failure = load_json(Path(verified["parent_b1"]["hand_failure"]["path"]))
    batch = load_json(Path(verified["parent_b1"]["batch_result"]["path"]))
    freeze = load_json(Path(verified["parent_b1"]["candidate_freeze"]["path"]))
    if parent_result.get("task_id") != "0915_robot_quality_recovery_v21_sam31_b1":
        raise ContractError("parent B1 task identity drift")
    if parent_result.get("status") != "FAILED_RUNTIME_FINAL":
        raise ContractError("B1R requires the frozen failed parent terminal")
    if failure != {
        "capability": "hand",
        "error": "KeyError: 164",
        "schema_version": "0915-robot-recovery-v21-sam31-b1-capability-failure-v1",
        "session_id": SESSION_ID,
        "status": "FAILED_RUNTIME",
    }:
        raise ContractError("parent B1 Hand failure signature drift")
    if batch.get("runtime_failures") != 1:
        raise ContractError("parent B1 must have exactly one runtime failure")
    parent_session = next(
        (row for row in batch.get("sessions", []) if row.get("session_id") == SESSION_ID), None,
    )
    if not isinstance(parent_session, Mapping) or parent_session.get("hand") != parent.get("hand_failure"):
        raise ContractError("parent batch does not bind the frozen Poker044 Hand failure")
    if freeze.get("child_runner") != parent.get("runner"):
        raise ContractError("parent B1 runner ref drift")
    frozen_session = next(
        (row for row in freeze.get("sessions", []) if row.get("session_id") == SESSION_ID), None,
    )
    if not isinstance(frozen_session, Mapping):
        raise ContractError("parent B1 Poker044 freeze row missing")
    rgb = a1["prepared_rgb"]
    if frozen_session.get("prepared_rgb") != rgb:
        raise ContractError("B1R A1 RGB differs from the parent frozen source")
    if rgb.get("image_domain") != IMAGE_DOMAIN or rgb.get("remap_applied") is not False:
        raise ContractError("B1R RGB domain drift")

    prompt = a1["raw_direct_prompt"]
    if (prompt.get("provider") != "A1_SAME_DOMAIN_RAW_DIRECT_OBSERVED_2D"
            or prompt.get("image_domain") != IMAGE_DOMAIN
            or prompt.get("side_admission") != {
                "left": "ADMITTED_DIRECT_OBSERVED", "right": "ADMITTED_DIRECT_OBSERVED",
            }
            or prompt.get("direct_observed_frames") != {"left": 16, "right": 74}
            or prompt.get("whole_session_quality") != "FAILED_QUALITY_C"):
        raise ContractError("B1R A1 prompt contract drift")
    consumer = load_json(Path(verified["a1_authority"]["consumer_admission"]["path"]))
    consumer_session = next(
        (row for row in consumer.get("sessions", []) if row.get("session_id") == SESSION_ID), None,
    )
    prompt_ref = {name: prompt[name] for name in ("path", "bytes", "sha256")}
    if not isinstance(consumer_session, Mapping) or consumer_session.get("npz") != prompt_ref:
        raise ContractError("B1R prompt is not bound by A1 consumer admission")
    return verified


def validate_runtime_refs(repo_root: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, descriptor in policy["runtime_refs"].items():
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
            raise ContractError(f"invalid runtime ref: {name}")
        path = (repo_root / str(descriptor["path"])).resolve(strict=True)
        actual = file_ref(path)
        if actual["sha256"] != descriptor.get("sha256"):
            raise ContractError(f"runtime SHA drift: {name}")
        result[name] = actual
    return result


def derive_fresh_prompt_view(destination: Path, prompt: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the four-key legacy view from A1, never from parent B1 output."""

    import numpy as np

    source = verify_ref(prompt, "A1 raw direct prompt")
    with np.load(source, allow_pickle=False) as archive:
        required = {"joints_2d", "observed", "detector_confidence", "anatomical_side_names"}
        if not required.issubset(archive.files):
            raise ContractError("A1 prompt lacks the legacy Hand keys")
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        confidence = np.asarray(archive["detector_confidence"], np.float64)
        sides = np.asarray(archive["anatomical_side_names"]).astype(str)
    if joints.shape != (2, FRAME_COUNT, 21, 2):
        raise ContractError("B1R prompt joint shape drift")
    if observed.shape != (2, FRAME_COUNT) or confidence.shape != (2, FRAME_COUNT):
        raise ContractError("B1R prompt observation/confidence shape drift")
    if sides.tolist() != ["left", "right"]:
        raise ContractError("B1R prompt side order drift")
    counts = {"left": int(observed[0].sum()), "right": int(observed[1].sum())}
    if counts != prompt.get("direct_observed_frames"):
        raise ContractError("B1R prompt observed-count drift")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}.npz")
    np.savez_compressed(
        temporary,
        joints_2d=joints,
        observed=observed,
        detector_confidence=confidence,
        anatomical_side_names=sides,
    )
    os.replace(temporary, destination)
    return {
        "source": file_ref(source),
        "view": file_ref(destination),
        "direct_observed_frames": counts,
        "parent_b1_prompt_view_consumed": False,
        "hawor_3d_consumed": False,
    }


def match_tracker_frame_state_keyerror(
    error: BaseException,
    *,
    anchor: int,
    reverse: bool,
    yielded_frame_indices: list[int],
    frame_count: int,
) -> dict[str, Any] | None:
    """Match only the observed fallback-165 -> reverse-frame-164 failure."""

    if type(error) is not KeyError or error.args != (164,):
        return None
    if (anchor != 165 or reverse is not True or yielded_frame_indices
            or frame_count != FRAME_COUNT):
        return None
    return {
        "direction": "backward",
        "status": "UNKNOWN_DIRECTION_INCOMPLETE",
        "reason": "TRACKER_FRAME_STATE_KEYERROR",
        "exception": "KeyError: 164",
        "anchor_frame": 165,
        "first_missing_frame": 164,
        "frames_yielded_before_hold": 0,
        "uncovered_direction_frame_count": 165,
        "uncovered_frame_semantics": "UNKNOWN_NOT_ABSENT",
        "semantic_admission_for_direction": False,
    }


def build_bounded_collector(
    hand_module: Any, recovery_events: list[dict[str, Any]],
) -> Callable[..., tuple[Any, list[dict[str, Any]]]]:
    """Return the legacy collector with one exact fail-closed KeyError conversion."""

    def collect(
        model: Any,
        state: dict[str, Any],
        *,
        anchor: int,
        raw_id: int,
        frame_count: int,
    ) -> tuple[Any, list[dict[str, Any]]]:
        import numpy as np

        result = np.zeros((frame_count, hand_module.HEIGHT, hand_module.WIDTH), bool)
        directions: list[dict[str, Any]] = []
        routes = [(False, frame_count - anchor)]
        if anchor:
            routes.append((True, anchor + 1))
        for reverse, maximum in routes:
            yielded_indices: list[int] = []
            try:
                for frame_index, outputs in model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=maximum,
                    reverse=reverse,
                    output_prob_thresh=0.5,
                ):
                    yielded_indices.append(int(frame_index))
                    masks, _scores, ids = hand_module.legacy.normalize(
                        outputs, hand_module.HEIGHT, hand_module.WIDTH,
                    )
                    matches = np.flatnonzero(ids == raw_id)
                    if len(matches) == 1 and 0 <= int(frame_index) < frame_count:
                        result[int(frame_index)] = masks[int(matches[0])]
                directions.append({
                    "direction": "backward" if reverse else "forward",
                    "status": "COMPLETE",
                    "frames_yielded": len(yielded_indices),
                })
            except RuntimeError as error:
                if str(error) != "No points are provided; please add points first":
                    raise
                directions.append({
                    "direction": "backward" if reverse else "forward",
                    "status": "UNKNOWN_DIRECTION_TRACKER_HAS_NO_CONFIRMED_INSTANCE",
                    "frames_yielded_before_hold": len(yielded_indices),
                    "reason": str(error),
                })
            except KeyError as error:
                recovery = match_tracker_frame_state_keyerror(
                    error,
                    anchor=anchor,
                    reverse=reverse,
                    yielded_frame_indices=yielded_indices,
                    frame_count=frame_count,
                )
                if recovery is None or recovery_events:
                    raise
                recovery = {**recovery, "raw_id": int(raw_id)}
                directions.append(recovery)
                recovery_events.append(dict(recovery))
        return result, directions

    return collect


def validate_legacy_anchor_plan(hand_module: Any, prompt_path: Path) -> dict[str, Any]:
    """Use the frozen legacy selectors to prove the failing route is still reachable."""

    import numpy as np

    with np.load(prompt_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        confidence = np.asarray(archive["detector_confidence"], np.float64)
    result: dict[str, Any] = {}
    for side, role in ((0, "left_hand"), (1, "right_hand")):
        candidates = hand_module.select_anchor_candidates(
            joints, observed, confidence, side,
        )
        primary, fallback = hand_module.choose_primary_fallback(candidates, FRAME_COUNT)
        result[role] = {
            "candidate_count": len(candidates),
            "primary_anchor": primary.get("frame_index") if primary else None,
            "fallback_anchor": fallback.get("frame_index") if fallback else None,
        }
    expected = {
        "left_hand": {"candidate_count": 2, "primary_anchor": 69, "fallback_anchor": 150},
        "right_hand": {"candidate_count": 63, "primary_anchor": 89, "fallback_anchor": 165},
    }
    if result != expected:
        raise ContractError(f"B1R legacy anchor plan drift: {result}")
    return result


def build_recovery_admission(
    hand_result: Mapping[str, Any], output_root: Path,
    recovery_events: list[dict[str, Any]],
) -> dict[str, Any]:
    side_results = hand_result.get("side_results")
    if not isinstance(side_results, list) or [row.get("role") for row in side_results] != [
        "left_hand", "right_hand",
    ]:
        raise ContractError("legacy Hand side terminal/order drift")
    sides: list[dict[str, Any]] = []
    ledger_incomplete_count = 0
    for side in side_results:
        ledger_path = verify_ref_under(
            side["quality_ledger"], f"{side['role']} quality ledger", output_root / "hand",
        )
        ledger = load_json(ledger_path)
        incomplete: list[dict[str, Any]] = []
        for seed_name in ("primary_seed", "fallback_seed"):
            seed = ledger.get(seed_name)
            if not isinstance(seed, Mapping):
                continue
            for direction in seed.get("direction_evidence", []):
                if direction.get("status") == "UNKNOWN_DIRECTION_INCOMPLETE":
                    incomplete.append({"seed": seed_name, **direction})
        ledger_incomplete_count += len(incomplete)
        raw_status = str(side.get("status", ""))
        if incomplete:
            b1r_status = "REJECTED_TRACKER_DIRECTION_INCOMPLETE"
            consumer_allowed = False
        else:
            b1r_status = raw_status
            consumer_allowed = bool(side.get("consumer_allowed")) and raw_status.startswith("PASS_HAND_")
        sides.append({
            "role": side["role"],
            "legacy_status": raw_status,
            "b1r_status": b1r_status,
            "consumer_allowed": consumer_allowed,
            "tracker_direction_incomplete": bool(incomplete),
            "incomplete_direction_evidence": incomplete,
            "quality_ledger": side["quality_ledger"],
        })
    if ledger_incomplete_count != len(recovery_events):
        raise ContractError("recovery-event/quality-ledger count mismatch")
    quality_rejections = sum(not row["consumer_allowed"] for row in sides)
    if quality_rejections == 0:
        raise ContractError("B1R all-pass publication is not authorized")
    return {
        "schema_version": f"{SCHEMA}-hand-admission-v1",
        "session_id": SESSION_ID,
        "status": "COMPLETED_WITH_QUALITY_REJECTION",
        "sides": sides,
        "quality_rejection_count": quality_rejections,
        "matching_keyerror_conversion_count": len(recovery_events),
        "recovery_events": recovery_events,
        "parent_failed_semantic_consumed": False,
        "uncovered_frame_semantics": "UNKNOWN_NOT_ABSENT",
        "mask_accuracy_claimed": False,
        "training_eligible": False,
    }


def build_freeze(
    repo_root: Path,
    output_root: Path,
    policy_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    policy = load_json(policy_path)
    manifest = load_json(manifest_path)
    validate_policy(policy)
    inputs = validate_parent_and_a1(manifest)
    runtimes = validate_runtime_refs(repo_root, policy)
    prompt_evidence = derive_fresh_prompt_view(
        output_root / "prompt_views" / f"{SESSION_ID}.npz",
        manifest["a1_authority"]["raw_direct_prompt"],
    )
    payload = {
        "schema_version": f"{SCHEMA}-candidate-freeze-v1",
        "task_id": TASK_ID,
        "candidate_id": policy["policy_id"],
        "policy": file_ref(policy_path),
        "input_manifest": file_ref(manifest_path),
        "child_runner": file_ref(Path(__file__)),
        "runtime_refs": runtimes,
        "gpu_lease": policy["gpu_lease"],
        "session": {
            **manifest["session"],
            "prepared_rgb": dict(manifest["a1_authority"]["prepared_rgb"]),
            "fresh_prompt_view": prompt_evidence["view"],
        },
        "parent_failure_evidence": inputs["parent_b1"],
        "a1_input_authority": inputs["a1_authority"],
        "prompt_derivation": prompt_evidence,
        "recovery_contract": policy["recovery_contract"],
        "parent_failed_semantic_consumed": False,
        "prompt_changed": False,
        "thresholds_changed": False,
        "weights_changed": False,
        "claim_limit": (
            "Poker044 Hand-only runtime recovery; incomplete tracker direction is UNKNOWN and "
            "quality-rejected. No Mask accuracy, training, control or deployment authority."
        ),
    }
    freeze = {**payload, "candidate_freeze_sha256": canonical_sha(payload)}
    atomic_json(output_root / "CANDIDATE_FREEZE_V1.json", freeze)
    return freeze


def validate_claim(
    path: Path,
    freeze_sha: str,
    signature_sha: str,
    output_root: Path,
    visual_root: Path,
    executor_epoch: int,
) -> None:
    claim = load_json(path)
    pid = claim.get("pid")
    valid = (
        claim.get("task_id") == TASK_ID
        and claim.get("status") == "CLAIMED"
        and claim.get("candidate_freeze_sha256") == freeze_sha
        and claim.get("run_signature_sha256") == signature_sha
        and claim.get("unique_write_root") == str(output_root.resolve())
        and claim.get("visual_write_root") == str(visual_root.resolve())
        and claim.get("executor_epoch") == executor_epoch
        and isinstance(claim.get("fencing_token_sha256"), str)
        and len(claim["fencing_token_sha256"]) == 64
        and isinstance(pid, int)
        and process_start_ticks(pid) == claim.get("proc_start_ticks")
    )
    if not valid:
        raise ContractError("B1R writer claim mismatch")
    if not process_has_ancestor(os.getpid(), int(pid)):
        raise ContractError("B1R worker is outside writer ancestry")


def run_worker(
    repo_root: Path,
    output_root: Path,
    visual_root: Path,
    claim_path: Path,
    freeze_path: Path,
) -> int:
    freeze = load_json(freeze_path)
    freeze_sha = freeze.pop("candidate_freeze_sha256", None)
    if freeze_sha != canonical_sha(freeze):
        raise ContractError("B1R candidate freeze digest mismatch")
    freeze["candidate_freeze_sha256"] = freeze_sha
    if freeze.get("task_id") != TASK_ID or freeze.get("parent_failed_semantic_consumed") is not False:
        raise ContractError("B1R candidate identity/input prohibition drift")
    signature = load_json(output_root / "RUN_SIGNATURE.json")
    signature_sha = signature.pop("run_signature_sha256", None)
    if signature_sha != canonical_sha(signature):
        raise ContractError("B1R run signature digest mismatch")
    if signature.get("candidate_freeze", {}).get("sha256") != sha256_file(freeze_path):
        raise ContractError("B1R run signature freeze-ref drift")
    if signature.get("output_root") != str(output_root.resolve()):
        raise ContractError("B1R output-root drift")
    if signature.get("visual_root") != str(visual_root.resolve()):
        raise ContractError("B1R visual-root drift")
    epoch = signature.get("executor_epoch")
    if not isinstance(epoch, int) or epoch < 1:
        raise ContractError("B1R executor epoch invalid")
    validate_claim(claim_path, str(freeze_sha), str(signature_sha), output_root, visual_root, epoch)
    if visual_root.exists() or visual_root.is_symlink():
        raise ContractError("fresh B1R visual root required")
    verify_ref(freeze["child_runner"], "frozen B1R runner")
    for name, descriptor in freeze["runtime_refs"].items():
        verify_ref(descriptor, f"frozen runtime {name}")
    verify_ref(freeze["session"]["prepared_rgb"], "B1R prepared RGB at worker start")
    verify_ref(freeze["session"]["fresh_prompt_view"], "B1R fresh prompt view at worker start")

    sys.path.insert(0, str((repo_root / "src").resolve()))
    adapter_module = importlib.import_module("chaoyang.pipeline.sam31_compat_adapter_v1")
    hand_module = importlib.import_module(
        "chaoyang.ops.run_0915_robot15h_sam31_temporal_identity_v1",
    )
    checkpoint = Path(str(freeze["runtime_refs"]["sam31_checkpoint"]["path"]))
    anchor_plan = validate_legacy_anchor_plan(
        hand_module, Path(str(freeze["session"]["fresh_prompt_view"]["path"])),
    )
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=repo_root / "vendor/SAM3",
        checkpoint_path=checkpoint,
    )
    hand_module.OUTPUT = output_root / "hand"
    hand_module.VISUAL = visual_root / "hand"
    hand_module.OUTPUT.mkdir(parents=True, exist_ok=True)
    recovery_events: list[dict[str, Any]] = []
    original_collector = hand_module._collect_bidirectional
    hand_module._collect_bidirectional = build_bounded_collector(hand_module, recovery_events)
    try:
        row = {
            "session_id": SESSION_ID,
            "task": TASK,
            "source_group": SOURCE_GROUP,
            "frame_count": FRAME_COUNT,
            "prepared_video": freeze["session"]["prepared_rgb"]["path"],
            "hawor_npz": freeze["session"]["fresh_prompt_view"]["path"],
        }
        hand_result = hand_module.process_session(adapter.model, row)
    finally:
        hand_module._collect_bidirectional = original_collector
        close = getattr(adapter, "close", None)
        close() if callable(close) else adapter.predictor.shutdown()

    verify_ref_under(hand_result["result"], "B1R legacy Hand result", hand_module.OUTPUT)
    admission = build_recovery_admission(hand_result, output_root, recovery_events)
    admission_path = output_root / "B1R_HAND_ADMISSION.json"
    atomic_json(admission_path, admission)
    batch = {
        "schema_version": f"{SCHEMA}-batch-result-v1",
        "task_id": TASK_ID,
        "status": "COMPLETED_WITH_QUALITY_REJECTION",
        "session_id": SESSION_ID,
        "capability": "HAND_ONLY",
        "model_load_count": 1,
        "gpu_execution_session_count": 1,
        "legacy_hand_result": hand_result["result"],
        "b1r_hand_admission": file_ref(admission_path),
        "matching_keyerror_conversion_count": len(recovery_events),
        "legacy_anchor_plan": anchor_plan,
        "quality_rejection_count": admission["quality_rejection_count"],
        "parent_failed_semantic_consumed": False,
        "prompt_changed": False,
        "thresholds_changed": False,
        "weights_changed": False,
        "model_build_evidence": build_evidence,
        "mask_accuracy_claimed": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
    }
    atomic_json(output_root / "BATCH_RESULT.json", batch)
    return 0


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
        "parent_failed_semantic_consumed": False,
        "mask_accuracy_claimed": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Poker044 Hand-only runtime recovery; incomplete tracker direction is UNKNOWN and "
            "quality-rejected. No Mask accuracy, training, control or deployment authority."
        ),
    }
    atomic_json(output_root / "RESULT.json", result)
    atomic_json(receipt_path, {**result, "result": file_ref(output_root / "RESULT.json")})
    atomic_json(output_root / "RUN_RECEIPT.json", {
        "schema_version": f"{SCHEMA}-run-receipt-v1",
        "task_id": TASK_ID,
        "status": status,
        "result": file_ref(output_root / "RESULT.json"),
        "terminal_receipt": file_ref(receipt_path),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    visual_root = args.visual_root.resolve()
    receipt_path = args.receipt.resolve()
    if (output_root.exists() or output_root.is_symlink()
            or visual_root.exists() or visual_root.is_symlink()
            or receipt_path.exists() or receipt_path.is_symlink()):
        raise ContractError("fresh B1R output, visual and receipt paths are required")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise ContractError("positive epoch and fencing token >=16 characters required")
    output_root.mkdir(parents=True)
    try:
        freeze = build_freeze(
            args.repo_root.resolve(strict=True),
            output_root,
            args.policy.resolve(strict=True),
            args.input_manifest.resolve(strict=True),
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
        "session_id": SESSION_ID,
        "capability": "HAND_ONLY",
        "gpu_lease": freeze["gpu_lease"],
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
    worker = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repo-root", str(args.repo_root.resolve()),
        "--output-root", str(output_root),
        "--visual-root", str(visual_root),
        "--claim", str(output_root / "CLAIM.json"),
        "--freeze", str(output_root / "CANDIDATE_FREEZE_V1.json"),
    ]
    lease = [
        sys.executable,
        str(freeze["runtime_refs"]["gpu_lease_wrapper"]["path"]),
        "--task-id", TASK_ID,
        "--attempt-id", output_root.name,
        "--executor-epoch", str(args.executor_epoch),
        "--priority", "CANARY",
        "--gpu-id", str(lease_policy["gpu_id"]),
        "--min-free-mib", str(lease_policy["minimum_free_mib"]),
        "--wait-seconds", str(lease_policy["wait_seconds"]),
        "--wall-seconds", str(lease_policy["wall_seconds"]),
        "--receipt", str(output_root / "GPU_COMMAND_RECEIPT.json"),
        "--claim-limit", freeze["claim_limit"],
        "--",
        *worker,
    ]
    atomic_json(output_root / "COMMAND.json", {"worker_command": worker, "lease_command": lease})
    with (output_root / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        completed = subprocess.run(
            lease,
            cwd=args.repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    gpu_path = output_root / "GPU_COMMAND_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    if gpu.get("status") == "BLOCKED_RESOURCE":
        write_terminal(output_root, receipt_path, "BLOCKED_RESOURCE", str(gpu.get("reason")))
        return 3
    if completed.returncode != 0 or gpu.get("status") != "PASSED":
        write_terminal(
            output_root,
            receipt_path,
            "FAILED_RUNTIME_FINAL",
            str(gpu.get("error") or "B1R_GPU_WORKER_FAILED"),
        )
        return 2
    batch = load_json(output_root / "BATCH_RESULT.json")
    if (batch.get("status") != "COMPLETED_WITH_QUALITY_REJECTION"
            or not isinstance(batch.get("quality_rejection_count"), int)
            or batch["quality_rejection_count"] < 1):
        write_terminal(output_root, receipt_path, "FAILED_RUNTIME_FINAL", "B1R_FAIL_CLOSED_TERMINAL_DRIFT")
        return 2
    blocker = (
        "REJECTED_TRACKER_DIRECTION_INCOMPLETE"
        if batch.get("matching_keyerror_conversion_count")
        else "LEGACY_HAND_QUALITY_REJECTION"
    )
    write_terminal(output_root, receipt_path, "COMPLETED_WITH_QUALITY_REJECTION", blocker)
    return 0


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
        return run_worker(
            args.repo_root.resolve(strict=True),
            args.output_root.resolve(),
            args.visual_root.resolve(),
            args.claim.resolve(strict=True),
            args.freeze.resolve(strict=True),
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
