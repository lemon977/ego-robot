#!/usr/bin/env python3
"""Run one real, direct-observation-only CONTACT-10 development canary.

The producer measures HaWoR MANO tip points against the observed Object6D
oriented box in the same selected-left camera coordinate system.  It never
propagates Object6D through an occlusion and never creates attachment evidence.
Consequently the output is a cross-system hypothesis, not contact truth.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.contact_geometry_v2 import oriented_box_sdf
from chaoyang.pipeline.contact_occlusion_contracts_r3 import validate_contact_hypothesis
from chaoyang.pipeline.object_contact_evidence_v1 import validate_evidence_dag


FINGERS = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES = (4, 8, 12, 16, 20)


def digest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    sha = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha.hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_result(path: Path, session_id: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = value.get("session_id", value.get("session"))
    if identity != session_id:
        raise ValueError(f"session mismatch in {path}: {identity}")
    return value


def state_for(
    distance_mm: float,
    radial_velocity_mm_s: float | None,
    tangential_velocity_mm_s: float | None,
    previous_touch: bool,
) -> str:
    if abs(distance_mm) <= 8.0:
        if previous_touch and tangential_velocity_mm_s is not None and tangential_velocity_mm_s >= 20.0:
            return "SLIDE_CANDIDATE"
        return "TOUCH_CANDIDATE"
    if 8.0 < distance_mm <= 30.0 and radial_velocity_mm_s is not None:
        if radial_velocity_mm_s <= -5.0:
            return "APPROACH"
        if previous_touch and radial_velocity_mm_s >= 5.0:
            return "RELEASE"
    return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--object6d-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    hawor_result_path = args.hawor_result.resolve(strict=True)
    object_result_path = args.object6d_result.resolve(strict=True)
    hawor_result = load_result(hawor_result_path, args.session_id)
    object_result = load_result(object_result_path, args.session_id)
    if object_result.get("unobserved_pose_policy") != "KEEP_INVALID":
        raise ValueError("CONTACT-10 requires observed-only KEEP_INVALID Object6D")
    if int(object_result.get("propagated_frames", -1)) != 0:
        raise ValueError("propagated Object6D is forbidden")

    hawor_npz = Path(hawor_result["outputs"]["npz"]["path"]).resolve(strict=True)
    object_npz = Path(object_result["artifacts"]["trajectory"]["path"]).resolve(strict=True)
    with np.load(hawor_npz, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_3d_camera"], dtype=np.float64)
        hand_observed = np.asarray(archive["observed"], dtype=np.bool_)
        confidence = np.asarray(archive["detector_confidence"], dtype=np.float64)
        fps = float(archive["fps"])
        sides = tuple(str(value) for value in archive["anatomical_side_names"].tolist())
        original_frames = np.asarray(archive["original_frame_indices"], dtype=np.int64)
    with np.load(object_npz, allow_pickle=False) as archive:
        object_frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        object_valid = np.asarray(archive["valid"], dtype=np.bool_)
        object_observed = np.asarray(archive["observed"], dtype=np.bool_)
        object_visibility = np.asarray(archive["visibility"], dtype=np.float64)
        object_transforms = np.asarray(archive["T_object_to_camera"], dtype=np.float64)
        object_size = np.asarray(archive["object_size_m"], dtype=np.float64)

    if joints.shape[:3] != (2, len(original_frames), 21):
        raise ValueError("unexpected HaWoR MANO21 shape")
    if sides != ("left", "right"):
        raise ValueError(f"anatomical side contract mismatch: {sides}")
    if not np.array_equal(original_frames, object_frames):
        raise ValueError("HaWoR and Object6D frame identity mismatch")
    direct_valid = object_valid & object_observed
    if not np.any(direct_valid):
        raise ValueError("no direct Object6D observations")

    direct_id = f"direct_object6d:{args.session_id}:object_0"
    seed_id = f"contact_seed:{args.session_id}:object_0"
    unknown_id = f"unknown:{args.session_id}"
    evidence_nodes = [
        {
            "evidence_id": direct_id,
            "evidence_type": "DIRECT_OBJECT6D",
            "parent_evidence_ids": [],
            "evidence_depth": 0,
            "may_support_contact_authority": True,
            "may_support_object6d_authority": True,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
        {
            "evidence_id": seed_id,
            "evidence_type": "CONTACT_SEED",
            "parent_evidence_ids": [direct_id],
            "evidence_depth": 1,
            "may_support_contact_authority": True,
            "may_support_object6d_authority": False,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
        {
            "evidence_id": unknown_id,
            "evidence_type": "UNKNOWN",
            "parent_evidence_ids": [],
            "evidence_depth": 0,
            "may_support_contact_authority": False,
            "may_support_object6d_authority": False,
            "may_support_gold_contact_accuracy": False,
            "may_upgrade_tactile_supported_contact": False,
        },
    ]
    graph = validate_evidence_dag(evidence_nodes)

    hypotheses: list[dict[str, Any]] = []
    previous_distance: dict[tuple[int, int], float] = {}
    previous_local_xy: dict[tuple[int, int], np.ndarray] = {}
    previous_touch: dict[tuple[int, int], bool] = {}
    for local_frame, source_frame in enumerate(original_frames.tolist()):
        transform = object_transforms[local_frame]
        rotation = transform[:3, :3]
        for side_index, side in enumerate(sides):
            for finger_index, (finger, joint_index) in enumerate(zip(FINGERS, TIP_INDICES)):
                key = (side_index, finger_index)
                measurement_available = bool(
                    direct_valid[local_frame]
                    and hand_observed[side_index, local_frame]
                    and np.isfinite(joints[side_index, local_frame, joint_index]).all()
                    and np.isfinite(transform).all()
                )
                state = "UNKNOWN"
                distance_mm: float | None = None
                radial_velocity: float | None = None
                tangential_velocity: float | None = None
                slip_score: float | None = None
                uncertainty = 1.0
                if measurement_available:
                    tip = joints[side_index, local_frame, joint_index]
                    distance_mm = float(oriented_box_sdf(tip[None], transform, object_size)[0] * 1000.0)
                    local = (tip - transform[:3, 3]) @ rotation
                    if key in previous_distance:
                        radial_velocity = (distance_mm - previous_distance[key]) * fps
                        tangential_velocity = float(
                            np.linalg.norm(local[:2] - previous_local_xy[key]) * 1000.0 * fps
                        )
                    state = state_for(
                        distance_mm,
                        radial_velocity,
                        tangential_velocity,
                        previous_touch.get(key, False),
                    )
                    previous_distance[key] = distance_mm
                    previous_local_xy[key] = local[:2].copy()
                    previous_touch[key] = state in {"TOUCH_CANDIDATE", "SLIDE_CANDIDATE"}
                    if state != "UNKNOWN":
                        slip_score = float(np.clip((tangential_velocity or 0.0) / 100.0, 0.0, 1.0))
                        evidence_strength = min(
                            float(np.clip(object_visibility[local_frame], 0.0, 1.0)),
                            float(np.clip(confidence[side_index, local_frame], 0.0, 1.0)),
                        )
                        uncertainty = float(np.clip(1.0 - evidence_strength, 0.05, 0.95))

                known = state != "UNKNOWN"
                hypotheses.append(
                    {
                        "frame_id": int(source_frame),
                        "hand_side": side,
                        "finger_id": finger,
                        "pad_link": f"{side}_{finger}_tip_surface_proxy",
                        "object_instance_id": "playing_card_0" if known else None,
                        "state": state,
                        "surface_distance_mm": distance_mm if known else None,
                        "relative_velocity": radial_velocity if known else None,
                        "slip_score": slip_score if known else None,
                        "uncertainty": uncertainty,
                        "valid": known,
                        "evidence_id": seed_id if known else unknown_id,
                        "parent_evidence_ids": [direct_id] if known else [],
                    }
                )

    contact = {
        "schema_version": "CONTACT_HYPOTHESIS_R3",
        "claim_status": "HYPOTHESIS_ONLY",
        "external_accuracy": "UNKNOWN",
        "evidence_nodes": evidence_nodes,
        "hypotheses": hypotheses,
        "claim_limit": (
            "HaWoR MANO fingertip versus directly observed Object6D thin-box proximity in "
            "selected-left camera metres. This is cross-system development evidence, not "
            "anatomical contact truth; no attachment pose is used."
        ),
    }
    validate_contact_hypothesis(contact)
    contact_path = output / "CONTACT_HYPOTHESIS.json"
    write_json(contact_path, contact)

    state_counts = Counter(row["state"] for row in hypotheses)
    known_rows = sum(count for state, count in state_counts.items() if state != "UNKNOWN")
    direct_frames = int(np.count_nonzero(direct_valid))
    metrics = {
        "schema_version": "CONTACT10_REAL_CANARY_METRICS_R22",
        "session_id": args.session_id,
        "frame_count": int(len(original_frames)),
        "direct_object6d_frames": direct_frames,
        "direct_object6d_coverage": direct_frames / len(original_frames),
        "hypothesis_rows": len(hypotheses),
        "known_rows": known_rows,
        "known_row_coverage": known_rows / len(hypotheses),
        "state_counts": dict(sorted(state_counts.items())),
        "distance_definition": "signed closed oriented-box SDF at MANO tip proxy",
        "relative_velocity_unit": "mm/s",
        "slip_definition": "clipped local-card-plane tip speed / 100 mm/s",
        "attachment_evidence_nodes": 0,
        "evidence_topological_order": list(graph.topological_order),
        "evidence_dag_acyclic": True,
        "external_accuracy": "UNKNOWN",
    }
    metrics_path = output / "METRICS.json"
    write_json(metrics_path, metrics)
    (output / "DECISION.md").write_text(
        "# CONTACT-10 Poker245真实canary\n\n"
        f"终态：`PASSED`（开发假设生成）  \n直接Object6D观测：`{direct_frames}/{len(original_frames)}`帧  \n"
        f"逐指已知假设：`{known_rows}/{len(hypotheses)}`行\n\n"
        "结果只由直接观测Object6D与HaWoR指尖产生。没有创建或消费Attachment，证据图无环。"
        "`HYPOTHESIS_ONLY`和`external_accuracy=UNKNOWN`是不可升级边界。\n",
        encoding="utf-8",
    )
    write_json(
        output / "NEXT_ACTION.json",
        {
            "schema_version": "CONTACT10_REAL_CANARY_NEXT_ACTION_R22",
            "status": "PASSED",
            "next_task_id": "CONTACT10_POKER243_AND_CHIPS_BOUNDED_EXPANSION",
            "prerequisites": ["same coordinate-domain closure", "DIRECT_OBJECT6D only"],
            "authority_promoted": False,
        },
    )
    inputs = [digest(hawor_result_path), digest(hawor_npz), digest(object_result_path), digest(object_npz)]
    run_receipt = {
        "schema_version": "CONTACT10_REAL_CANARY_RUN_RECEIPT_R22",
        "task_id": "CONTACT-10-REAL-BOUNDED-CANARY",
        "attempt_id": output.name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "artifact_revision": "R7_2_CONTACT10_DIRECT_ONLY",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": inputs,
        "producer": digest(Path(__file__)),
        "authority_promoted": False,
    }
    write_json(output / "RUN_RECEIPT.json", run_receipt)
    manifest = {
        "schema_version": "CONTACT10_REAL_CANARY_ARTIFACT_MANIFEST_R22",
        "artifacts": [
            digest(contact_path),
            digest(metrics_path),
            digest(output / "DECISION.md"),
            digest(output / "NEXT_ACTION.json"),
            digest(output / "RUN_RECEIPT.json"),
        ],
        "authority_promoted": False,
    }
    write_json(output / "ARTIFACT_MANIFEST.json", manifest)
    result = {
        "schema_version": "CONTACT10_REAL_CANARY_RESULT_R22",
        "task_id": "CONTACT-10-REAL-BOUNDED-CANARY",
        "attempt_id": output.name,
        "session_id": args.session_id,
        "terminal_status": "PASSED",
        "status": "PASS_DEVELOPMENT_HYPOTHESIS_ONLY",
        "claim_status": "HYPOTHESIS_ONLY",
        "external_accuracy": "UNKNOWN",
        "evidence_dag_acyclic": True,
        "attachment_used": False,
        "metrics": digest(metrics_path),
        "contact_hypothesis": digest(contact_path),
        "authority_promoted": False,
        "control_ground_truth": False,
        "claim_limit": contact["claim_limit"],
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
