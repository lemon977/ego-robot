#!/usr/bin/env python3
"""Build the 58-row exact78 Contact hypothesis readiness matrix.

The audit validates pinned Role/Object Mask, Depth, observed-only Object6D,
camera-coordinate arrays, frame identity, and direct hand/object overlap.  It
does not execute CONTACT-10 and never promotes Contact authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_clean_runtime_recovery_v53/EXACT78_WAVE0_SELECTION.json"
DEFAULT_STATUS = ROOT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_ref(reference: dict[str, Any], label: str) -> Path:
    if not {"path", "bytes", "sha256"}.issubset(reference):
        raise ValueError(f"{label}: incomplete reference")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label}: missing absolute path {path}")
    if path.stat().st_size != int(reference["bytes"]) or sha256_file(path) != reference["sha256"]:
        raise ValueError(f"{label}: bytes/SHA mismatch {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_role_ready(value: dict[str, Any]) -> bool:
    return value.get("grade") in {"A", "B"} and value.get("downstream_authorized") is True


def is_object_mask_ready(value: dict[str, Any]) -> bool:
    gates = value.get("gates") or {}
    return (
        value.get("grade") in {"A", "B"}
        and value.get("downstream_authorized") is True
        and all(bool(v) for v in gates.values())
    )


def is_depth_ready(value: dict[str, Any]) -> bool:
    return (
        value.get("grade") in {"A", "B"}
        and value.get("consumption_authorized") is True
        and "VISUAL_OBJECT6D_CANDIDATE_INPUT" in value.get("authorized_scopes", [])
    )


def is_object6d_ready(value: dict[str, Any]) -> bool:
    authorized = value.get("downstream_authorized") is True or value.get("consumption_authorized") is True
    return value.get("grade") in {"A", "B"} and authorized and value.get("unobserved_pose_policy") == "KEEP_INVALID"


def object_children(value: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return (child result, trajectory descriptor) for both Object6D schemas."""
    children: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if isinstance(value.get("instances"), dict):
        for instance in value["instances"].values():
            child_path = validate_ref(instance["result"], "object6d.child_result")
            child = load_json(child_path)
            children.append((child, instance["trajectory"]))
    else:
        trajectory = (value.get("artifacts") or {}).get("trajectory")
        if not isinstance(trajectory, dict):
            raise ValueError("single-instance Object6D result lacks trajectory descriptor")
        children.append((value, trajectory))
    return children


def audit_coordinates(hawor: dict[str, Any], object6d: dict[str, Any]) -> dict[str, Any]:
    output = hawor.get("outputs", {}).get("npz")
    if not isinstance(output, dict):
        return {"consistent": False, "reason": "HAWOR_CAMERA_ARRAY_MISSING", "direct_overlap_frames": 0}
    hawor_path = validate_ref(output, "hawor.npz")
    with np.load(hawor_path, allow_pickle=False) as hand:
        required = {"joints_3d_camera", "observed", "original_frame_indices", "mano_tip_indices", "mano_wrist_index"}
        if not required.issubset(hand.files):
            return {"consistent": False, "reason": "HAWOR_REQUIRED_CAMERA_FIELDS_MISSING", "direct_overlap_frames": 0}
        hand_frames = np.asarray(hand["original_frame_indices"], dtype=np.int64)
        hand_observed = np.asarray(hand["observed"], dtype=bool).any(axis=0)
        joints = np.asarray(hand["joints_3d_camera"])
        hand_finite = bool(np.isfinite(joints[np.asarray(hand["observed"], dtype=bool)]).all())
        hand_count = int(hand_observed.sum())

    object_observed = np.zeros(hand_frames.shape[0], dtype=bool)
    object_finite = True
    frame_identity_exact = True
    direct_only = True
    child_count = 0
    observed_sum_by_instance: list[int] = []
    for child, trajectory_ref in object_children(object6d):
        trajectory_path = validate_ref(trajectory_ref, "object6d.trajectory")
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            needed = {"frame_indices", "T_object_to_camera"}
            if not needed.issubset(trajectory.files):
                return {"consistent": False, "reason": "OBJECT6D_CAMERA_FIELDS_MISSING", "direct_overlap_frames": 0}
            frames = np.asarray(trajectory["frame_indices"], dtype=np.int64)
            observed_key = "observed" if "observed" in trajectory.files else "valid"
            observed = np.asarray(trajectory[observed_key], dtype=bool)
            transforms = np.asarray(trajectory["T_object_to_camera"])
            same_frames = frames.shape == hand_frames.shape and np.array_equal(frames, hand_frames)
            frame_identity_exact = frame_identity_exact and same_frames
            if same_frames:
                object_observed |= observed
            else:
                object_finite = False
            if observed.any():
                object_finite = object_finite and bool(np.isfinite(transforms[observed]).all())
            observed_sum_by_instance.append(int(observed.sum()))
        direct_only = direct_only and int(child.get("propagated_frames") or 0) == 0 and child.get("unobserved_pose_policy") == "KEEP_INVALID"
        child_count += 1

    overlap = hand_observed & object_observed
    consistent = hand_finite and object_finite and frame_identity_exact and child_count > 0
    return {
        "consistent": consistent,
        "reason": "CAMERA_METERS_FIELDS_AND_FRAME_IDENTITY_CLOSED_FOR_DEVELOPMENT" if consistent else "CAMERA_DOMAIN_OR_FRAME_IDENTITY_MISMATCH",
        "coordinate_frame": "selected_left_camera",
        "unit": "meter",
        "hand_field": "joints_3d_camera",
        "object_field": "T_object_to_camera",
        "frame_identity_exact": frame_identity_exact,
        "hand_camera_values_finite": hand_finite,
        "object_camera_values_finite": object_finite,
        "hand_observed_frames": hand_count,
        "object_observed_union_frames": int(object_observed.sum()),
        "object_observed_frames_by_instance": observed_sum_by_instance,
        "direct_hand_object_overlap_frames": int(overlap.sum()),
        "object6d_direct_observed_only": direct_only,
        "claim_limit": "Field/schema/frame identity closure for a development hypothesis; not external metric truth or Contact accuracy.",
    }


def first_blocker(row: dict[str, Any]) -> str:
    priorities = (
        (not row["role_mask_ready"], "ROLE_MASK_NOT_READY"),
        (not row["object_mask_ready"], "OBJECT_MASK_NOT_READY"),
        (not row["depth_ready"], "DEPTH_NOT_READY"),
        (not row["object6d_ready"], "OBJECT6D_NOT_READY"),
        (not row["object6d_direct_observed_only"], "OBJECT6D_NOT_DIRECT_OBSERVED_ONLY"),
        (not row["coordinate_domain_consistent"], "HAND_OBJECT_CAMERA_COORDINATE_DOMAIN_NOT_CLOSED"),
        (row["direct_hand_object_overlap_frames"] <= 0, "NO_DIRECT_HAND_OBJECT_OBSERVATION_OVERLAP"),
    )
    for failed, name in priorities:
        if failed:
            return name
    return "R3_PER_FINGER_CONTACT_PRODUCER_NOT_RUN"


def build_row(selected: dict[str, Any]) -> dict[str, Any]:
    session = str(selected["session_id"])
    upstream = selected.get("upstream") or {}
    docs: dict[str, dict[str, Any]] = {}
    refs: dict[str, dict[str, Any]] = {}
    for name in ("hawor", "role_mask", "task_object_mask", "depth", "object6d"):
        path = validate_ref(upstream[name], f"{session}.{name}")
        docs[name] = load_json(path)
        refs[name] = ref(path)
    for name, value in docs.items():
        evidence_session = value.get("session_id") or value.get("session")
        if evidence_session != session:
            raise ValueError(f"{session}.{name}: session identity mismatch {evidence_session}")

    coord = audit_coordinates(docs["hawor"], docs["object6d"])
    depth_closure = docs["depth"].get("input_closure_sha256")
    object_closure = docs["object6d"].get("input_closure_sha256")
    if object_closure is not None:
        lineage_closed = depth_closure == object_closure
    else:
        object_depth_ref = (docs["object6d"].get("inputs") or {}).get("depth_result") or {}
        lineage_closed = object_depth_ref.get("sha256") == refs["depth"]["sha256"]

    row = {
        "position": int(selected["position"]),
        "session_id": session,
        "task": selected["task"],
        "frame_count": int(selected["frame_count"]),
        "artifact_revision": "R7_0",
        "role_mask_ready": is_role_ready(docs["role_mask"]),
        "object_mask_ready": is_object_mask_ready(docs["task_object_mask"]),
        "depth_ready": is_depth_ready(docs["depth"]),
        "object6d_ready": is_object6d_ready(docs["object6d"]),
        "depth_object6d_lineage_closed": lineage_closed,
        "coordinate_domain_consistent": bool(coord["consistent"] and lineage_closed),
        "coordinate_evidence": coord,
        "object6d_direct_observed_only": bool(coord.get("object6d_direct_observed_only")),
        "direct_hand_object_overlap_frames": int(coord.get("direct_hand_object_overlap_frames") or 0),
        "evidence": refs,
        "current_contact_result_present": False,
        "contact_authority_promotable": False,
        "external_accuracy": "UNKNOWN",
    }
    row["hypothesis_only_input_possible"] = all(
        (
            row["role_mask_ready"],
            row["object_mask_ready"],
            row["depth_ready"],
            row["object6d_ready"],
            row["depth_object6d_lineage_closed"],
            row["coordinate_domain_consistent"],
            row["object6d_direct_observed_only"],
            row["direct_hand_object_overlap_frames"] > 0,
        )
    )
    row["primary_blocker"] = first_blocker(row)
    row["terminal_readiness"] = "READY_TO_RUN_HYPOTHESIS_ONLY" if row["hypothesis_only_input_possible"] else "BLOCKED_PREREQ"
    row["claim_limit"] = "Input readiness for HYPOTHESIS_ONLY only; CONTACT-10 has not run and no physical/Gold accuracy is claimed."
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--status-min", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"immutable attempt exists: {output}")
    output.mkdir(parents=True)
    generated_at = now_iso()
    selection = load_json(args.selection.resolve())
    status = load_json(args.status_min.resolve())
    selected = selection.get("sessions", [])
    if len(selected) != 58 or len({row["session_id"] for row in selected}) != 58:
        raise ValueError("selection is not exact58 unique")

    rows = [build_row(row) for row in sorted(selected, key=lambda x: (int(x["position"]), x["session_id"]))]
    ready_count = sum(row["hypothesis_only_input_possible"] for row in rows)
    blockers = Counter(row["primary_blocker"] for row in rows)
    by_task = {
        task: {
            "sessions": sum(row["task"] == task for row in rows),
            "hypothesis_only_input_possible": sum(row["task"] == task and row["hypothesis_only_input_possible"] for row in rows),
        }
        for task in ("chips", "poker")
    }
    overlaps = np.asarray([row["direct_hand_object_overlap_frames"] for row in rows], dtype=np.int64)
    matrix = {
        "schema_version": "exact78-contact-readiness-matrix-r3-v1",
        "artifact_revision": "R7_3_CONTACT_READINESS",
        "generated_at": generated_at,
        "status": "PASSED_58_UNIQUE_READINESS_ROWS",
        "selection": ref(args.selection.resolve()),
        "counts": {
            "sessions": 58,
            "role_mask_ready": sum(row["role_mask_ready"] for row in rows),
            "object_mask_ready": sum(row["object_mask_ready"] for row in rows),
            "depth_ready": sum(row["depth_ready"] for row in rows),
            "object6d_ready": sum(row["object6d_ready"] for row in rows),
            "coordinate_domain_consistent_for_development": sum(row["coordinate_domain_consistent"] for row in rows),
            "direct_observed_overlap_nonzero": sum(row["direct_hand_object_overlap_frames"] > 0 for row in rows),
            "hypothesis_only_input_possible": ready_count,
            "contact_10_executed": 0,
            "contact_authority": 0,
            "by_primary_blocker": dict(blockers),
            "by_task": by_task,
        },
        "overlap_frames": {
            "minimum": int(overlaps.min()),
            "median": float(np.median(overlaps)),
            "maximum": int(overlaps.max()),
        },
        "policy": {
            "formal_object6d": "DIRECT_OBSERVED_ONLY_KEEP_INVALID",
            "hypothesis_scope": "HYPOTHESIS_ONLY",
            "attachment_feedback_forbidden": True,
            "gold_accuracy_reported": False,
            "external_accuracy": "UNKNOWN",
            "current_missing_execution": "R3_PER_FINGER_CONTACT_PRODUCER_NOT_RUN",
        },
        "rows": rows,
        "authority_promoted": False,
        "claim_limit": "CPU input-readiness audit. It proves which pinned exact58 sessions can enter a future HYPOTHESIS_ONLY producer; it does not create Contact labels, accuracy, authority, or physical truth.",
    }
    matrix_path = output / "CONTACT_READINESS_MATRIX.json"
    write_json(matrix_path, matrix)
    csv_path = output / "CONTACT_READINESS_MATRIX.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fields = [
            "position", "session_id", "task", "frame_count", "role_mask_ready", "object_mask_ready",
            "depth_ready", "object6d_ready", "depth_object6d_lineage_closed", "coordinate_domain_consistent",
            "object6d_direct_observed_only", "direct_hand_object_overlap_frames", "hypothesis_only_input_possible",
            "terminal_readiness", "primary_blocker", "contact_authority_promotable", "external_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    metrics_path = output / "METRICS.json"
    write_json(metrics_path, {
        "schema_version": "exact78-contact-readiness-r3-metrics-v1",
        "generated_at": generated_at,
        "counts": matrix["counts"],
        "overlap_frames": matrix["overlap_frames"],
        "all_input_refs_verified": True,
        "all_rows_have_one_primary_blocker_or_execution_gap": all(bool(row["primary_blocker"]) for row in rows),
        "gpu_calls": 0,
        "contact_10_executed": False,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    decision_path = output / "DECISION.md"
    decision_path.write_text(
        "# exact78 Contact readiness 决定 R3\n\n"
        "状态：`PASSED`（58 行输入就绪审计）。\n\n"
        f"- Role/Object Mask、Depth、observed-only Object6D 与开发级相机坐标/帧身份闭合：{ready_count}/58 可进入 `HYPOTHESIS_ONLY` producer。\n"
        f"- 直接人手—物体观测重叠帧：每条最少 {int(overlaps.min())}，中位数 {float(np.median(overlaps)):.1f}，最多 {int(overlaps.max())}。\n"
        "- 当前 58 条均未执行 R3 逐指 CONTACT-10；因此主阻塞是 `R3_PER_FINGER_CONTACT_PRODUCER_NOT_RUN`。\n"
        "- `robot_contact_authorized=false`、Contact authority=0、external accuracy=UNKNOWN。\n\n"
        "该矩阵只回答能否启动假设生成，不能回答接触是否真实或精度是多少。\n",
        encoding="utf-8",
    )
    next_path = output / "NEXT_ACTION.json"
    write_json(next_path, {
        "schema_version": "exact78-contact-readiness-r3-next-v1",
        "status": "PASSED",
        "next_task_id": "CONTACT-10-REAL-BOUNDED-CANARY",
        "selection_rule": "One Poker thin-card session and one Chips three-instance session from rows with nonzero direct overlap; 4 frames then 24 frames before any full session.",
        "required_claim_status": "HYPOTHESIS_ONLY",
        "forbidden": ["attachment_feedback", "Object6D authority promotion", "Gold accuracy without frozen labels"],
        "authority_promoted": False,
    })
    packet_path = output / "TASK_PACKET.json"
    write_json(packet_path, {
        "schema_version": "exact78-contact-readiness-r3-task-packet-v1",
        "task_id": "CONTACT-READINESS-MATRIX-R3",
        "attempt_id": output.name,
        "execution_mode": "CPU_READ_ONLY_AUDIT",
        "read_set": [str(args.status_min.resolve()), str(args.selection.resolve())],
        "write_set": [str(output)],
        "gpu_seconds": 0,
        "current_governance_update_allowed": False,
        "robot_v76_touched": False,
    })
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    manifest_members = [matrix_path, csv_path, metrics_path, decision_path, next_path, packet_path]
    write_json(manifest_path, {
        "schema_version": "exact78-contact-readiness-r3-manifest-v1",
        "task_id": "CONTACT-READINESS-MATRIX-R3",
        "attempt_id": output.name,
        "artifacts": [ref(path) for path in manifest_members],
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    result_path = output / "RESULT.json"
    write_json(result_path, {
        "schema_version": "exact78-contact-readiness-r3-result-v1",
        "task_id": "CONTACT-READINESS-MATRIX-R3",
        "attempt_id": output.name,
        "terminal_status": "PASSED",
        "generated_at": generated_at,
        "counts": matrix["counts"],
        "matrix_json": ref(matrix_path),
        "matrix_csv": ref(csv_path),
        "metrics": ref(metrics_path),
        "decision": ref(decision_path),
        "artifact_manifest": ref(manifest_path),
        "current_governance_updated": False,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "exact78-contact-readiness-r3-receipt-v1",
        "task_id": "CONTACT-READINESS-MATRIX-R3",
        "attempt_id": output.name,
        "terminal_status": "PASSED",
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "gpu_used": False,
        "governance_revision_observed": status.get("governance_revision"),
        "result": ref(result_path),
        "artifact_manifest": ref(manifest_path),
        "current_governance_updated": False,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    print(json.dumps({"status": "PASSED", "counts": matrix["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
