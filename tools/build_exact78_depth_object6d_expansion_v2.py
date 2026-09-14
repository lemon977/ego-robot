#!/usr/bin/env python3
"""Freeze the remaining exact78 calibrated Depth/Object6D expansion.

This is a CPU-only admission builder.  It snapshots the current exact78
indices, requires the exact 26-session calibrated join, excludes the four
already-complete Poker sessions, and emits fresh/no-clobber manifests for the
remaining 22.  Poker is routed as one action-conditioned physical card;
Chips is routed as three independent physical instances (never a union).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
CONTROL = PROJECT / "tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
FROZEN = PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json"
JOIN_STATE = CONTROL / "MASK_JOIN_READY_STATE.json"
HAWOR_INDEX = CONTROL / "HAWOR_TERMINAL_INDEX.json"
ROLE_INDEX = CONTROL / "MASK_ROLE_BOUNDED_V2_1_TERMINAL_INDEX.json"
TASK_OBJECT_INDEX = CONTROL / "MASK_TASK_OBJECT_TERMINAL_INDEX.json"
OLD_DEPTH_INDEX = CONTROL / "depth_object6d_stream_v1/DEPTH_TERMINAL_INDEX.json"
OLD_OBJECT_INDEX = CONTROL / "depth_object6d_stream_v1/OBJECT6D_TERMINAL_INDEX.json"
CALIBRATION = PROJECT / "tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json"
CHECKPOINT = PROJECT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
DEPTH_WORKER = PROJECT / "tools/run_exact78_foundationstereo_corrected_depth_worker.py"
DEPTH_LAUNCHER = PROJECT / "tools/launch_exact78_corrected_depth_after_training.py"
OBJECT_WORKER = PROJECT / "tools/run_visual_fixed_instance_object6d.py"
LEASE = PROJECT / "_run/GPU_LEASE.json"

CHIPS_CANARY = "get_potato_chips_0903_050"
EXPECTED_COMPLETED = {
    "play_cards_0903_203",
    "play_cards_0903_224",
    "play_cards_0903_243",
    "play_cards_0903_245",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular artifact required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify(value: dict[str, Any], label: str) -> Path:
    path = Path(str(value["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(value["bytes"])
        or sha(path) != value["sha256"]
    ):
        raise RuntimeError(f"{label}: artifact closure failed")
    return path


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"fresh/no-clobber output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_for(session_id: str) -> str:
    if session_id.startswith("get_potato_chips_"):
        return "chips"
    if session_id.startswith("play_cards_"):
        return "poker"
    raise RuntimeError(f"unknown exact78 task prefix: {session_id}")


def choose_expansion_sessions(
    calibrated_ready: Iterable[str], completed: Iterable[str]
) -> dict[str, Any]:
    calibrated = list(calibrated_ready)
    completed_set = set(completed)
    if len(calibrated) != 26 or len(set(calibrated)) != 26:
        raise RuntimeError("current calibrated join must contain exactly 26 unique sessions")
    if completed_set != EXPECTED_COMPLETED:
        raise RuntimeError(
            f"completed predecessor must be exact frozen four: {sorted(completed_set)}"
        )
    if not completed_set.issubset(calibrated):
        raise RuntimeError("completed predecessor is not a subset of calibrated-ready 26")
    remaining = [session for session in calibrated if session not in completed_set]
    chips = [session for session in remaining if task_for(session) == "chips"]
    poker = [session for session in remaining if task_for(session) == "poker"]
    if len(remaining) != 22 or len(chips) != 7 or len(poker) != 15:
        raise RuntimeError(
            f"unexpected exact78 expansion split: remaining={len(remaining)} "
            f"chips={len(chips)} poker={len(poker)}"
        )
    if CHIPS_CANARY not in remaining:
        raise RuntimeError(f"fresh Chips canary missing: {CHIPS_CANARY}")
    successors = [session for session in remaining if session != CHIPS_CANARY]
    return {
        "calibrated_ready": calibrated,
        "already_complete": sorted(completed_set),
        "remaining": remaining,
        "canary": [CHIPS_CANARY],
        "successors": successors,
        "counts": {
            "calibrated_ready": 26,
            "already_complete": 4,
            "remaining": 22,
            "chips_remaining": 7,
            "poker_remaining": 15,
        },
    }


def adapt_task_object_frames(
    source: dict[str, Any], task: str, global_instance_id: int
) -> dict[str, Any]:
    if source.get("task") != task:
        raise RuntimeError("task-object manifest task mismatch")
    if source.get("semantic_type") != "TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK":
        raise RuntimeError("task-object semantic type mismatch")
    expected_ids = {0} if task == "poker" else {0, 1, 2}
    if global_instance_id not in expected_ids:
        raise RuntimeError(f"invalid {task} global instance id: {global_instance_id}")
    rows: list[dict[str, Any]] = []
    for expected, frame in enumerate(source.get("frames", [])):
        if int(frame["source_frame"]) != expected:
            raise RuntimeError("task-object frame identity is not contiguous from zero")
        instances = frame.get("physical_instances", {})
        if set(map(int, instances)) != expected_ids:
            raise RuntimeError(f"{task} physical instance set mismatch at frame {expected}")
        item = instances[str(global_instance_id)]
        observed = bool(item.get("observed") and item.get("valid"))
        rows.append(
            {
                "frame_id": expected,
                "source_rgb_decoded_sha256": frame["selected_rgb_decoded_sha256"],
                "observed_identity": (
                    "ACTION_CONDITIONED_SAME_PHYSICAL_CARD_0"
                    if task == "poker"
                    else f"PHYSICAL_CHIP_{global_instance_id}"
                ),
                "physical_instance_id": 0 if observed else -1,
                "global_physical_instance_id": global_instance_id,
                "valid": observed,
                "observed": observed,
                "mask": item["mask"],
            }
        )
    return {
        "schema_version": "rgb-object-mask-manifest-v1",
        "created_at": now(),
        "session_id": source["session"],
        "task": task,
        "semantic_type": (
            "POKER_ACTION_CONDITIONED_SAME_CARD_OBSERVED_ONLY"
            if task == "poker"
            else "CHIPS_ONE_OF_THREE_FIXED_PHYSICAL_INSTANCES_OBSERVED_ONLY"
        ),
        "global_physical_instance_id": global_instance_id,
        "local_runner_instance_id": 0,
        "union_used_as_instance": False,
        "frames": rows,
        "claim_limit": (
            "Adapter only. Unobserved frames remain empty/invalid; no propagation, "
            "identity switch, hand/pinch substitution, or multi-instance union."
        ),
    }


def by_session(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["session_id"]): row for row in payload["terminals"]}


def camera_baseline(path: Path) -> float:
    params = load(path)
    left = np.asarray(params["extrinsics"]["left"], dtype=np.float64)
    right = np.asarray(params["extrinsics"]["right"], dtype=np.float64)
    left_center = -left[:3, :3].T @ left[:3, 3]
    right_center = -right[:3, :3].T @ right[:3, 3]
    return float(np.linalg.norm(right_center - left_center))


def require_grade_ab(row: dict[str, Any], label: str) -> None:
    if row.get("grade") not in {"A", "B"} or not row.get("downstream_authorized"):
        raise RuntimeError(f"{label}: current terminal is not downstream-authorized A/B")
    verify(row["result"], f"{label}:RESULT")
    verify(row["agent_review"], f"{label}:AGENT_REVIEW")


def completed_four() -> set[str]:
    depth = by_session(load(OLD_DEPTH_INDEX))
    objects = by_session(load(OLD_OBJECT_INDEX))
    common = set(depth) & set(objects)
    if common != EXPECTED_COMPLETED:
        raise RuntimeError(f"old stream terminal set drift: {sorted(common)}")
    for session in sorted(common):
        require_grade_ab(depth[session], f"{session}:old_depth")
        require_grade_ab(objects[session], f"{session}:old_object6d")
    return common


def build(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh/no-clobber root exists: {output_root}")
    output_root.mkdir(parents=True)

    sources = {
        "frozen_exact78": FROZEN,
        "mask_join_ready": JOIN_STATE,
        "hawor_index": HAWOR_INDEX,
        "role_index": ROLE_INDEX,
        "task_object_index": TASK_OBJECT_INDEX,
        "old_depth_index": OLD_DEPTH_INDEX,
        "old_object6d_index": OLD_OBJECT_INDEX,
    }
    payloads = {name: load(path) for name, path in sources.items()}
    snapshot_refs: dict[str, Any] = {}
    source_refs: dict[str, Any] = {}
    for name, path in sources.items():
        source_refs[name] = artifact(path)
        snapshot_path = output_root / "input_snapshots" / f"{name.upper()}.json"
        atomic_json(snapshot_path, payloads[name])
        snapshot_refs[name] = artifact(snapshot_path)

    join = payloads["mask_join_ready"]
    progress = join.get("progress", {})
    if (
        int(progress.get("role_terminal", -1)) != 156
        or int(progress.get("task_object_terminal", -1)) != 156
        or int(progress.get("calibrated_ready_for_depth_object6d_clean", -1)) != 26
    ):
        raise RuntimeError("two-Mask join is not the frozen 156/156, calibrated-ready=26 state")
    selection = choose_expansion_sessions(
        join["calibrated_ready_sessions"], completed_four()
    )

    frozen_by = {row["session_id"]: row for row in payloads["frozen_exact78"]["sessions"]}
    hawor_by = by_session(payloads["hawor_index"])
    role_by = by_session(payloads["role_index"])
    task_object_by = by_session(payloads["task_object_index"])
    common_sources = {
        "source_files": source_refs,
        "immutable_snapshots": snapshot_refs,
        "calibration": artifact(CALIBRATION),
        "foundationstereo_checkpoint": artifact(CHECKPOINT),
        "persistent_corrected_depth_worker": artifact(DEPTH_WORKER),
        "central_lease_launcher": artifact(DEPTH_LAUNCHER),
        "visual_fixed_instance_object6d_worker": artifact(OBJECT_WORKER),
    }

    depth_rows: list[dict[str, Any]] = []
    object_routes: list[dict[str, Any]] = []
    for session_id in selection["remaining"]:
        frozen = frozen_by.get(session_id)
        hawor = hawor_by.get(session_id)
        role = role_by.get(session_id)
        task_object = task_object_by.get(session_id)
        if not all((frozen, hawor, role, task_object)):
            raise RuntimeError(f"{session_id}: missing frozen/current terminal")
        task = task_for(session_id)
        if frozen["task"] != task or frozen["stereo_calibration_state"] != "CALIBRATED_METRIC_STEREO":
            raise RuntimeError(f"{session_id}: frozen task/calibration mismatch")
        require_grade_ab(hawor, f"{session_id}:hawor")
        require_grade_ab(role, f"{session_id}:role_mask")
        require_grade_ab(task_object, f"{session_id}:task_object_mask")

        task_result = load(Path(task_object["result"]["path"]))
        task_manifest_ref = task_result["artifacts"]["manifest"]
        task_manifest_path = verify(task_manifest_ref, f"{session_id}:task_object_manifest")
        task_manifest = load(task_manifest_path)
        if len(task_manifest.get("frames", [])) != int(frozen["frame_count"]):
            raise RuntimeError(f"{session_id}: task-object frame count mismatch")

        raw = Path(frozen["raw_path"])
        selected_rgb = raw / f"CameraRecord_{session_id}.mp4"
        raw_stereo = raw / "source_stereo" / f"CameraRecord_{session_id}_stereo.mp4"
        camera_params = raw / "camera_params.json"
        clip_manifest = raw / "clip_manifest.json"
        hawor_result = load(Path(hawor["result"]["path"]))
        hawor_npz = hawor_result["outputs"]["npz"]
        verify(hawor_npz, f"{session_id}:hawor_npz")

        adapters: list[dict[str, Any]] = []
        instance_ids = [0] if task == "poker" else [0, 1, 2]
        for instance_id in instance_ids:
            adapter_payload = adapt_task_object_frames(task_manifest, task, instance_id)
            for frame_index, frame in enumerate(adapter_payload["frames"]):
                verify(frame["mask"], f"{session_id}:object{instance_id}:mask{frame_index}")
            adapter_path = (
                output_root
                / "object6d_input_adapters"
                / task
                / session_id
                / f"physical_object_{instance_id}"
                / "OBJECT_MASK_MANIFEST.json"
            )
            adapter_payload["source_task_object_manifest"] = task_manifest_ref
            atomic_json(adapter_path, adapter_payload)
            adapters.append(
                {
                    "global_physical_instance_id": instance_id,
                    "adapter": artifact(adapter_path),
                    "destination": str(
                        output_root
                        / "object6d_observed_only_v1"
                        / task
                        / session_id
                        / (
                            f"physical_object_{instance_id}"
                            if task == "chips"
                            else "physical_object_0"
                        )
                    ),
                }
            )

        destination = output_root / "depth_corrected_metric_v1" / task / session_id
        base_inputs = {
            "session_id": session_id,
            "task": task,
            "frame_count": int(frozen["frame_count"]),
            "immutable_input_identity_sha256": frozen["immutable_input_identity_sha256"],
            "stereo_calibration_state": frozen["stereo_calibration_state"],
            "cohort": artifact(FROZEN),
            "raw_stereo": artifact(raw_stereo),
            "selected_rgb": artifact(selected_rgb),
            "camera_params": artifact(camera_params),
            "clip_manifest": artifact(clip_manifest),
            "hawor_result": hawor["result"],
            "hawor_agent_review": hawor["agent_review"],
            "hawor_npz": hawor_npz,
            "role_mask_result": role["result"],
            "role_mask_agent_review": role["agent_review"],
            "task_object_result": task_object["result"],
            "task_object_agent_review": task_object["agent_review"],
            "task_object_manifest": task_manifest_ref,
            "rgb_object_mask_adapter": adapters[0]["adapter"],
            "calibration": common_sources["calibration"],
            "foundationstereo_checkpoint": common_sources["foundationstereo_checkpoint"],
            "upstream_grades": {
                "hawor": hawor["grade"],
                "role_mask": role["grade"],
                "task_object": task_object["grade"],
            },
            "metric_baseline_m": camera_baseline(camera_params),
            "destination": str(destination),
            "atomic_temp_destination": str(destination.parent / f".{session_id}.partial"),
        }
        closure = canonical_sha(base_inputs)
        depth_rows.append(
            {
                **base_inputs,
                "input_closure_sha256": closure,
                "resume_no_clobber": {
                    "decision": "RUN_NEW_ATOMIC_TEMP",
                    "root": str(destination),
                },
            }
        )
        object_routes.append(
            {
                "session_id": session_id,
                "task": task,
                "frame_count": int(frozen["frame_count"]),
                "depth_input_closure_sha256": closure,
                "depth_destination": str(destination),
                "instance_count": len(instance_ids),
                "instances": adapters,
                "identity_policy": (
                    "ACTION_CONDITIONED_SAME_PHYSICAL_CARD_0"
                    if task == "poker"
                    else "THREE_INDEPENDENT_FIXED_PHYSICAL_CHIP_IDS_NO_UNION"
                ),
                "unobserved_pose_policy": "KEEP_INVALID",
                "aggregate_destination": str(
                    output_root / "object6d_observed_only_v1" / task / session_id
                ),
            }
        )

    by_id = {row["session_id"]: row for row in depth_rows}
    common_manifest = {
        "schema_version": "exact78-corrected-depth-input-manifest-v1",
        "created_at": now(),
        "mode": "CURRENT_EXACT78_EXPANSION_GPU_UNDER_CENTRAL_LEASE",
        "resume_requested": True,
        "join_contract": "IMMUTABLE_EXACT26_MASK_TWO_LANE_A_OR_B_EXACT_REFS",
        "sources": common_sources,
        "errors": [],
        "status": "PASS_CPU_INPUT_CLOSURE",
        "execution_authorized": True,
    }
    canary_path = output_root / "DEPTH_INPUT_MANIFEST_CHIPS_CANARY1.json"
    successors_path = output_root / "DEPTH_INPUT_MANIFEST_SUCCESSOR21.json"
    atomic_json(
        canary_path,
        {
            **common_manifest,
            "phase": "fresh_chips_three_instance_canary1",
            "sessions": [by_id[CHIPS_CANARY]],
        },
    )
    atomic_json(
        successors_path,
        {
            **common_manifest,
            "phase": "successor21_after_chips_canary_ab",
            "sessions": [by_id[session] for session in selection["successors"]],
        },
    )
    routes_path = output_root / "OBJECT6D_ROUTE_PLAN.json"
    atomic_json(
        routes_path,
        {
            "schema_version": "exact78-object6d-expansion-route-plan-v2",
            "created_at": now(),
            "sessions": object_routes,
            "poker_contract": "one action-conditioned same card; KEEP_INVALID",
            "chips_contract": "three independent fixed IDs; three child trajectories; no union",
            "robot_contact_authorized": False,
        },
    )
    selection_path = output_root / "EXACT26_SELECTION_AND_EXCLUSION.json"
    atomic_json(
        selection_path,
        {
            "schema_version": "exact78-calibrated-expansion-selection-v2",
            "created_at": now(),
            **selection,
            "source_join_snapshot": snapshot_refs["mask_join_ready"],
            "old_completed_depth_index": snapshot_refs["old_depth_index"],
            "old_completed_object6d_index": snapshot_refs["old_object6d_index"],
        },
    )
    authority_path = output_root / "AUTHORITY.json"
    authority = {
        "schema_version": "exact78-depth-object6d-expansion-authority-v2",
        "created_at": now(),
        "status": "PASS_CPU_VALIDATED_READY_FOR_CHIPS_CANARY_LAUNCH",
        "selection": artifact(selection_path),
        "canary_session": CHIPS_CANARY,
        "successor_count": 21,
        "manifests": {
            "chips_canary1": artifact(canary_path),
            "successor21": artifact(successors_path),
            "object6d_routes": artifact(routes_path),
        },
        "implementations": {
            "persistent_corrected_depth_worker": common_sources["persistent_corrected_depth_worker"],
            "central_lease_launcher": common_sources["central_lease_launcher"],
            "visual_fixed_instance_object6d_worker": common_sources["visual_fixed_instance_object6d_worker"],
            "admission_builder": artifact(Path(__file__)),
        },
        "gpu_policy": {
            "depth": "one persistent FoundationStereo model under central GPU lease; launcher requires >=61440 MiB free",
            "object6d": "CPU only and starts only after same-session Depth Grade B",
            "canary_gate": "Chips050 Depth and all three independent Object6D children must be Grade A/B before successor21",
        },
        "authorized_scope": "remaining exact78 calibrated-ready 22 Depth plus observed-only Object6D",
        "forbidden_scope": [
            "Clean execution",
            "Robot/contact authority",
            "HumanEgo training",
            "calibration-missing 10",
            "task-object union as a Chips instance",
        ],
    }
    atomic_json(authority_path, authority)
    lease = load(LEASE)
    receipt_path = output_root / "CPU_VALIDATE_ONLY_RECEIPT.json"
    receipt = {
        "schema_version": "exact78-depth-object6d-expansion-cpu-validation-v2",
        "created_at": now(),
        "status": "PASS_CPU_VALIDATE_ONLY_NO_GPU_STARTED",
        "fresh_root": str(output_root),
        "selection": artifact(selection_path),
        "authority": artifact(authority_path),
        "counts": selection["counts"],
        "routing": {
            "poker_sessions": 15,
            "poker_instances_per_session": 1,
            "chips_sessions": 7,
            "chips_instances_per_session": 3,
            "chips_union_used_as_instance": False,
        },
        "central_lease_observed": {
            "status": lease.get("status"),
            "holder": lease.get("holder"),
        },
        "gpu_started": False,
        "object6d_started": False,
        "next_gate": "launch Chips050 canary; require Depth B and three Object6D child A/B before successor21",
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Required safety flag: build/validate immutable CPU plan; never launch GPU.",
    )
    args = parser.parse_args()
    if not args.validate_only:
        raise SystemExit("--validate-only is required; this admission tool never launches GPU")
    result = build(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
