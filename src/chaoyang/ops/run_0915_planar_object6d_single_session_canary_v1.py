#!/usr/bin/env python3
"""Run the CPU-only observed planar Object6D canary for two stable cards."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.pipeline.object6d_planar_observability_v1 import (
    DEPTH_REFERENCE,
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    PlanarFrameInput,
    build_observability_document,
    estimate_planar_frame,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_planar_object6d_single_session_canary_v1"
PHASE = "0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY"
SESSION_ID = "play_cards_0915_001"
INSTANCE_IDS = ("playing_card_00", "playing_card_01")
DEPTH_ROOT = (
    ROOT / "_run/current/0915_foundationstereo_single_session_canary_v1/"
    "attempts/attempt_0001"
)
SAM_ROOT = (
    ROOT / "_run/current/0915_sam31_strict_role_canary_v5/"
    "attempts/attempt_0001"
)
OUTPUT_NAMESPACE = ROOT / "_run/current" / TASK_ID
SCHEMA = ROOT / "contracts/object6d_planar_observability_v1.schema.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_output_namespace(output: Path) -> None:
    if not output.resolve().is_relative_to(OUTPUT_NAMESPACE.resolve()):
        raise RuntimeError(f"output must stay inside task namespace: {OUTPUT_NAMESPACE}")


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen specification")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("planar Object6D canary must have weights=ABSENT")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("planar Object6D canary is not current next_task")
    task = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING",
    }:
        raise RuntimeError("planar Object6D canary is not executable")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize planar Object6D canary")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet


def heartbeat(pid: int) -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(pid),
            "--status", "RUNNING", "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )


@dataclass(frozen=True)
class PackedMask:
    packed: np.ndarray
    frame_count: int
    height: int
    width: int
    bitorder: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.frame_count, self.height, self.width

    def unpack(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        return np.unpackbits(
            self.packed[frame_index], bitorder=self.bitorder,
            count=self.height * self.width,
        ).reshape(self.height, self.width).astype(bool)


def _load_packed(path: Path) -> PackedMask:
    with np.load(path, allow_pickle=False) as archive:
        packed = np.asarray(archive["packed"], dtype=np.uint8)
        frame_count = int(archive["frame_count"])
        height = int(archive["height"])
        width = int(archive["width"])
        bitorder = str(archive["bitorder"])
    expected = (frame_count, (height * width + 7) // 8)
    if packed.shape != expected or bitorder != "big":
        raise RuntimeError(f"packed SAM mask contract drift: {path}")
    return PackedMask(packed, frame_count, height, width, bitorder)


def _ref_matches(path: Path, expected: dict[str, Any]) -> bool:
    return (
        str(path.resolve(strict=True)) == expected.get("path")
        and path.stat().st_size == expected.get("bytes")
        and sha256(path) == expected.get("sha256")
    )


def _validate_depth_upstream(depth_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = load_json(depth_root / "RESULT.json")
    contract = load_json(depth_root / "DEPTH_CONTRACT.json")
    registration = load_json(depth_root / "REGISTRATION.json")
    if result.get("status") != "PASSED" or result.get("depth_admission") != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_DEPTH_NOT_ADMITTED")
    if result.get("external_accuracy") != "UNVERIFIED":
        raise RuntimeError("depth external-accuracy boundary drift")
    if contract.get("external_accuracy") != "UNVERIFIED":
        raise RuntimeError("depth contract external-accuracy boundary drift")
    if contract.get("frame_arrays", {}).get("depth_reference") != DEPTH_REFERENCE:
        raise RuntimeError("depth contract is not rectified-left optical-Z metres")
    maps_path = depth_root / "REGISTRATION_MAPS.npz"
    if not maps_path.is_file():
        raise RuntimeError("BLOCKED_UNPROVEN_MASK_DEPTH_REGISTRATION")
    maps_ref = registration.get("nonlinear_registration_maps")
    if not isinstance(maps_ref, dict) or not _ref_matches(maps_path, maps_ref):
        raise RuntimeError("registration maps are not SHA-bound by REGISTRATION.json")
    join = registration.get("depth_to_sam_resize_map")
    if not isinstance(join, dict):
        raise RuntimeError("BLOCKED_UNPROVEN_MASK_DEPTH_REGISTRATION")
    if registration.get("source_domain") != "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS":
        raise RuntimeError("mask/depth registration source-domain drift")
    if join.get("source") != "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS":
        raise RuntimeError("mask/depth registration source-domain drift")
    if join.get("target") != "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960_PIXEL_CENTERS":
        raise RuntimeError("mask/depth registration target-domain drift")
    if join.get("array") != "depth_to_sam_resize_xy" or join.get("identity_assumed") is not False:
        raise RuntimeError("mask/depth join must use the explicit nonlinear map")
    if "IDENTITY_JOIN_FORBIDDEN" not in str(registration.get("mask_join_policy")):
        raise RuntimeError("identity mask/depth join was not explicitly forbidden")
    return registration, {
        "depth_result": ref(depth_root / "RESULT.json"),
        "depth_contract": ref(depth_root / "DEPTH_CONTRACT.json"),
        "depth_registration": ref(depth_root / "REGISTRATION.json"),
        "depth_registration_maps": ref(maps_path),
    }


def _load_registration_maps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"depth_to_sam_resize_xy", "sam_resize_in_bounds"}
        if not required.issubset(archive.files):
            raise RuntimeError("registration maps are missing SAM join arrays")
        mapping = np.asarray(archive["depth_to_sam_resize_xy"], np.float64)
        valid = np.asarray(archive["sam_resize_in_bounds"], bool)
    if mapping.ndim != 3 or mapping.shape[2] != 2 or valid.shape != mapping.shape[:2]:
        raise RuntimeError("registration map geometry is invalid")
    return mapping, valid


def _load_sam_inputs(
    sam_root: Path,
) -> tuple[dict[str, PackedMask], dict[str, list[str]], dict[str, Any]]:
    manifest_path = sam_root / "ROLE_MANIFEST.json"
    temporal_path = sam_root / "TEMPORAL_STATE_LEDGER.json"
    manifest = load_json(manifest_path)
    temporal = load_json(temporal_path)
    if manifest.get("image_domain") != "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP":
        raise RuntimeError("v5 SAM image-domain drift")
    by_id = {str(row["instance_id"]): row for row in manifest.get("instances", [])}
    states_by_id = {
        str(row["instance_id"]): [str(item["state"]) for item in row["frames"]]
        for row in temporal.get("instances", [])
    }
    masks: dict[str, PackedMask] = {}
    mask_refs: dict[str, Any] = {}
    for instance_id in INSTANCE_IDS:
        row = by_id.get(instance_id)
        if row is None or row.get("role") != "task_object":
            raise RuntimeError(f"missing stable v5 task-object instance: {instance_id}")
        path = sam_root / str(row["mask_archive"])
        masks[instance_id] = _load_packed(path)
        mask_refs[instance_id] = ref(path)
        states = states_by_id.get(instance_id)
        if states is None or len(states) != masks[instance_id].frame_count:
            raise RuntimeError(f"v5 temporal-state axis drift: {instance_id}")
    if masks[INSTANCE_IDS[0]].shape != masks[INSTANCE_IDS[1]].shape:
        raise RuntimeError("v5 stable-card mask geometry differs")
    return masks, states_by_id, {
        "sam_role_manifest": ref(manifest_path),
        "sam_temporal_state_ledger": ref(temporal_path),
        "sam_masks": mask_refs,
    }


def _frame_paths(depth_root: Path, expected_count: int) -> list[Path]:
    paths = sorted((depth_root / "frames").glob("*.npz"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"depth frame count differs from SAM: {len(paths)} != {expected_count}"
        )
    return paths


def build_depth_frame_input_manifest(depth_root: Path) -> dict[str, Any]:
    paths = sorted((depth_root / "frames").glob("*.npz"))
    if not paths:
        raise RuntimeError("no immutable depth frames were published")
    return {
        "schema_version": "0915-planar-object6d-depth-frame-input-manifest-v1",
        "session_id": SESSION_ID,
        "frame_count": len(paths),
        "frames": [ref(path) for path in paths],
    }


def build_run_signature(
    packet_path: Path, depth_frame_manifest: dict[str, Any],
) -> dict[str, Any]:
    fixed_inputs = [
        DEPTH_ROOT / name for name in (
            "RESULT.json", "CALIBRATION.json", "DEPTH_CONTRACT.json",
            "DEPTH_SUMMARY.json", "REGISTRATION.json", "REGISTRATION_MAPS.npz",
        )
    ] + [
        SAM_ROOT / "ROLE_MANIFEST.json",
        SAM_ROOT / "TEMPORAL_STATE_LEDGER.json",
        SAM_ROOT / "masks/playing_card_00.npz",
        SAM_ROOT / "masks/playing_card_01.npz",
    ]
    signature = {
        "schema_version": "0915-planar-object6d-run-signature-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "weights": "ABSENT",
        "calibration": ref(DEPTH_ROOT / "CALIBRATION.json"),
        "input_artifacts": [ref(path) for path in fixed_inputs],
        "depth_frame_input_manifest": depth_frame_manifest,
        "code": [
            ref(Path(__file__)),
            ref(ROOT / "src/chaoyang/pipeline/object6d_planar_observability_v1.py"),
        ],
        "schema": ref(SCHEMA),
        "task_packet": ref(packet_path),
        "config": {
            "instances": list(INSTANCE_IDS),
            "depth_reference": DEPTH_REFERENCE,
            "registration_authority": MASK_REGISTRATION_AUTHORITY,
            "hidden_geometry_policy": "INVALID_NOT_COMPLETED",
        },
    }
    return {**signature, "run_signature_sha256": canonical_sha256(signature)}


def _load_depth_frame(
    path: Path, expected_index: int, expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        frame_id = int(archive["frame_id"])
        depth = np.asarray(archive["depth_m"], np.float64)
        valid = np.asarray(archive["valid"], bool)
        intrinsics = np.asarray(archive["depth_intrinsics"], np.float64)
        reference = str(archive["depth_reference"])
    if frame_id != expected_index:
        raise RuntimeError(f"non-contiguous depth frame id at {path}")
    if depth.shape != expected_shape or valid.shape != expected_shape:
        raise RuntimeError(f"depth frame geometry drift at {path}")
    if reference != DEPTH_REFERENCE:
        raise RuntimeError(f"depth-reference drift at {path}")
    return depth, valid, intrinsics


def _summarize(document: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for obj in document["objects"]:
        rows = obj["frames"]
        objects.append({
            "instance_id": obj["instance_id"],
            "frame_count": len(rows),
            "mask_state_counts": dict(Counter(row["mask_state"] for row in rows)),
            "observable_frame_counts": {
                name: sum(row[name]["observability"] != "UNOBSERVABLE" for row in rows)
                for name in ("translation", "plane_normal", "in_plane_rotation")
            },
            "unobservable_reason_counts": {
                name: dict(Counter(
                    row[name]["reason"] for row in rows
                    if row[name]["observability"] == "UNOBSERVABLE"
                ))
                for name in ("translation", "plane_normal", "in_plane_rotation")
            },
        })
    return {
        "schema_version": "0915-planar-object6d-summary-v1",
        "status": "COMPLETED_DEVELOPMENT_OBSERVABILITY",
        "session_id": SESSION_ID,
        "objects": objects,
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "unified_confidence_emitted": False,
        "contact_authority": "NONE",
        "robot_authority": "NONE",
    }


def _render_timeline(document: dict[str, Any], destination: Path) -> None:
    frame_count = int(document["frame_count"])
    scale = 5
    left = 245
    row_height = 28
    components = ("translation", "plane_normal", "in_plane_rotation")
    canvas = np.full(
        (65 + len(INSTANCE_IDS) * len(components) * row_height, left + frame_count * scale + 15, 3),
        245, np.uint8,
    )
    cv2.putText(
        canvas, "Planar Object6D: independent observability (no confidence)",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "green=observable | gray=unknown/occluded/insufficient",
        (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1, cv2.LINE_AA,
    )
    y = 65
    for obj in document["objects"]:
        for component in components:
            cv2.putText(
                canvas, f"{obj['instance_id']} {component}", (8, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA,
            )
            for row in obj["frames"]:
                x0 = left + int(row["frame_index"]) * scale
                color = (
                    (55, 170, 55)
                    if row[component]["observability"] != "UNOBSERVABLE"
                    else (155, 155, 155)
                )
                cv2.rectangle(canvas, (x0, y + 3), (x0 + scale - 1, y + 23), color, -1)
            y += row_height
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError("failed to write Object6D observability timeline")


def _input_snapshot(inputs: dict[str, Any]) -> dict[str, str]:
    refs = [
        inputs["depth_result"], inputs["depth_contract"],
        inputs["depth_registration"], inputs["depth_registration_maps"],
        inputs["sam_role_manifest"], inputs["sam_temporal_state_ledger"],
        *inputs["sam_masks"].values(),
    ]
    return {row["path"]: row["sha256"] for row in refs}


def assert_inputs_unchanged(snapshot: dict[str, str]) -> None:
    for name, expected in snapshot.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"read-only upstream changed during Object6D run: {path}")


def run_canary(
    output: Path, visual_root: Path, depth_frame_manifest: dict[str, Any],
) -> dict[str, Any]:
    _registration, depth_inputs = _validate_depth_upstream(DEPTH_ROOT)
    mapping, registration_valid = _load_registration_maps(
        DEPTH_ROOT / "REGISTRATION_MAPS.npz"
    )
    masks, states, sam_inputs = _load_sam_inputs(SAM_ROOT)
    atomic_json(output / "DEPTH_FRAME_INPUT_MANIFEST.json", depth_frame_manifest)
    inputs = {
        **depth_inputs,
        "depth_frame_manifest": ref(output / "DEPTH_FRAME_INPUT_MANIFEST.json"),
        **sam_inputs,
    }
    snapshot = _input_snapshot(inputs)
    snapshot.update({
        str(row["path"]): str(row["sha256"])
        for row in depth_frame_manifest["frames"]
    })
    frame_count = masks[INSTANCE_IDS[0]].frame_count
    frame_paths = _frame_paths(DEPTH_ROOT, frame_count)
    if depth_frame_manifest["frame_count"] != frame_count:
        raise RuntimeError("signed depth-frame manifest differs from SAM frame axis")
    for path, expected in zip(frame_paths, depth_frame_manifest["frames"], strict=True):
        if not _ref_matches(path, expected):
            raise RuntimeError(f"signed depth-frame artifact drift: {path}")
    if mapping.shape[:2] != registration_valid.shape:
        raise RuntimeError("depth-to-SAM registration-map axes differ")
    object_frames: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in INSTANCE_IDS
    }
    for frame_index, path in enumerate(frame_paths):
        depth, valid, intrinsics = _load_depth_frame(
            path, frame_index, mapping.shape[:2],
        )
        for instance_id in INSTANCE_IDS:
            mask_state = states[instance_id][frame_index]
            visibility = "UNKNOWN" if mask_state == "unknown" else DIRECT_VISIBILITY
            object_frames[instance_id].append(estimate_planar_frame(PlanarFrameInput(
                frame_index=frame_index,
                mask=masks[instance_id].unpack(frame_index),
                mask_state=mask_state,
                visibility_state=visibility,
                depth_m=depth,
                depth_valid=valid,
                depth_intrinsics=intrinsics,
                depth_to_mask_xy=mapping,
                registration_valid=registration_valid,
            )))
    document = build_observability_document(
        session_id=SESSION_ID, object_frames=object_frames, inputs=inputs,
    )
    schema = load_json(SCHEMA)
    jsonschema.Draft202012Validator(schema).validate(document)
    assert document["unified_confidence_emitted"] is False
    atomic_json(output / "OBJECT6D_OBSERVABILITY.json", document)
    summary = _summarize(document)
    atomic_json(output / "OBJECT6D_SUMMARY.json", summary)
    visual = visual_root / "0915_PLANAR_OBJECT6D_OBSERVABILITY.png"
    _render_timeline(document, visual)
    assert_inputs_unchanged(snapshot)
    return {
        "document": ref(output / "OBJECT6D_OBSERVABILITY.json"),
        "summary": ref(output / "OBJECT6D_SUMMARY.json"),
        "visual": ref(visual),
        "input_snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()

    packet = validate_route()
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    validate_output_namespace(output)
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("valid executor epoch and fencing token are required")
    for path in (output, visual):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)
    depth_frame_manifest = build_depth_frame_input_manifest(DEPTH_ROOT)
    run_signature = build_run_signature(packet_path, depth_frame_manifest)
    claim = {
        "schema_version": "0915-planar-object6d-writer-claim-v1",
        "task_id": TASK_ID,
        "pid": os.getpid(),
        "proc_start_ticks": int(
            Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()[21]
        ),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(
            args.fencing_token.encode("utf-8")
        ).hexdigest(),
        "run_signature_sha256": run_signature["run_signature_sha256"],
        "unique_write_root": str(output),
        "weights": "ABSENT",
    }
    atomic_json(output / "CLAIM.json", claim)
    atomic_json(output / "RUN_SIGNATURE.json", run_signature)
    heartbeat(os.getpid())
    artifacts = run_canary(output, visual, depth_frame_manifest)
    result = {
        "schema_version": "0915-planar-object6d-single-session-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "session_id": SESSION_ID,
        "weights": "ABSENT",
        "object6d_admission": "PASS_DEVELOPMENT_VISIBLE_SURFACE_ONLY",
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "unified_confidence_emitted": False,
        "contact_authority": "NONE",
        "robot_authority": "NONE",
        "artifacts": artifacts,
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "source_mutated": False,
        "gpu_used": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-planar-object6d-run-receipt-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "result": ref(output / "RESULT.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "gpu_used": False,
    })
    print(json.dumps({
        "status": "PASSED",
        "result": str(output / "RESULT.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
