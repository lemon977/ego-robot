#!/usr/bin/env python3
"""CPU-only planar Object6D V2 canary in the encoded physical-left domain."""

from __future__ import annotations

import argparse
from collections import Counter
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

from chaoyang.pipeline.object6d_planar_observability_v2 import (
    COMPONENT_NAMES,
    DEPTH_REFERENCE,
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    OBJECT_INSTANCE_IDS,
    PlanarFrameInput,
    build_observability_document,
    estimate_planar_frame,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_planar_object6d_observability_canary_v2"
PHASE = "0915_PLANAR_OBJECT6D_ENCODED_OBSERVABILITY_CANARY_V2"
SESSION_ID = "play_cards_0915_001"
DEPTH_TASK_ID = "0915_foundationstereo_encoded_domain_canary_v1"
AUTHORIZED_SCOPE = "VISUAL_OBJECT6D_CANDIDATE_INPUT"
INSTANCE_IDS = OBJECT_INSTANCE_IDS
DEPTH_ROOT = (
    ROOT / "_run/current/0915_foundationstereo_encoded_domain_canary_v1/"
    "attempts/attempt_0001"
)
SAM_ROOT = (
    ROOT / "_run/current/0915_sam31_strict_role_canary_v5/"
    "attempts/attempt_0001"
)
OUTPUT_NAMESPACE = ROOT / "_run/current" / TASK_ID
FIXED_ATTEMPT = OUTPUT_NAMESPACE / "attempts/attempt_0001"
VISUAL_NAMESPACE = (
    ROOT / "docs/current/visuals/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2"
)
TERMINAL_RECEIPT = (
    ROOT / "tasks/receipts/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2_RESULT.json"
)
SCHEMA = ROOT / "contracts/object6d_planar_observability_v2.schema.json"
DEPTH_SHAPE = (480, 640)
MASK_SHAPE = (960, 1280)
MAPPING_CONTRACT = {
    "type": "ANALYTIC_PIXEL_CENTER_2X",
    "formula_x": "x_sam = 2*x_depth + 0.5",
    "formula_y": "y_sam = 2*y_depth + 0.5",
    "source": "ORIGINAL_PHYSICAL_LEFT_640x480",
    "target": "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960",
}


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
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
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


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_writer_claim(
    path: Path, *, output: Path, signature_sha: str, executor_epoch: int,
) -> dict[str, Any]:
    claim = load_json(path)
    pid = claim.get("pid")
    ticks = claim.get("proc_start_ticks")
    if (
        claim.get("schema_version")
        != "0915-planar-object6d-observability-writer-claim-v2"
        or claim.get("task_id") != TASK_ID
        or claim.get("session_id") != SESSION_ID
        or claim.get("attempt_id") != output.name
        or claim.get("status") != "CLAIMED"
        or claim.get("weights") != "ABSENT"
        or not isinstance(pid, int)
        or not isinstance(ticks, int)
        or process_start_ticks(pid) != ticks
        or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("unique_write_root") != str(output)
        or not isinstance(claim.get("fencing_token_sha256"), str)
        or len(claim.get("fencing_token_sha256", "")) != 64
    ):
        raise RuntimeError("encoded Object6D V2 writer claim/fence mismatch")
    return claim


def validate_output_namespace(output: Path, visual: Path, receipt: Path) -> None:
    if output.resolve() != FIXED_ATTEMPT.resolve():
        raise RuntimeError(f"output must equal fixed attempt: {FIXED_ATTEMPT}")
    if visual.resolve() != VISUAL_NAMESPACE.resolve():
        raise RuntimeError(f"visual root must equal: {VISUAL_NAMESPACE}")
    if receipt.resolve() != TERMINAL_RECEIPT.resolve():
        raise RuntimeError(f"receipt must equal: {TERMINAL_RECEIPT}")


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet.get("task_id") != TASK_ID or packet.get("weights") != "ABSENT":
        raise RuntimeError("encoded Object6D V2 task packet identity drift")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("encoded Object6D V2 is not current next_task")
    task = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING",
    }:
        raise RuntimeError("encoded Object6D V2 is not executable")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if (
        route is None
        or route.get("execution_allowed") is not True
        or route.get("packet_sha256") != sha256(packet_path)
    ):
        raise RuntimeError("task index does not authorize encoded Object6D V2")
    return packet


def heartbeat(pid: int) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            TASK_ID,
            "--pid",
            str(pid),
            "--status",
            "RUNNING",
            "--phase",
            PHASE,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )


def validate_encoded_depth_admission(result_path: Path) -> dict[str, Any]:
    """Call the governance-owned encoded-domain downstream admission gate."""

    from chaoyang.governance.register_0915_campaign_task_v1 import (
        validate_encoded_foundationstereo_object6d_admission,
    )

    return validate_encoded_foundationstereo_object6d_admission(result_path)


def depth_to_sam_pixel_center_map(
    *, height: int = DEPTH_SHAPE[0], width: int = DEPTH_SHAPE[1],
) -> tuple[np.ndarray, np.ndarray]:
    """Map 640x480 depth centres to the same physical-left 1280x960 domain."""

    yy, xx = np.indices((height, width), dtype=np.float64)
    mapping = np.stack((2.0 * xx + 0.5, 2.0 * yy + 0.5), axis=-1)
    in_bounds = (
        (mapping[..., 0] >= 0.0)
        & (mapping[..., 0] < MASK_SHAPE[1])
        & (mapping[..., 1] >= 0.0)
        & (mapping[..., 1] < MASK_SHAPE[0])
    )
    return mapping, in_bounds


def validate_depth_upstream(depth_root: Path) -> dict[str, Any]:
    admission = validate_encoded_depth_admission(depth_root / "RESULT.json")
    result = admission.get("result")
    contract = admission.get("contract")
    summary = admission.get("summary")
    if not isinstance(result, dict):
        result = load_json(depth_root / "RESULT.json")
    if not isinstance(contract, dict):
        contract = load_json(depth_root / "DEPTH_CONTRACT.json")
    if not isinstance(summary, dict):
        summary = load_json(depth_root / "DEPTH_SUMMARY.json")
    if (
        result.get("task_id") != DEPTH_TASK_ID
        or result.get("session_id") != SESSION_ID
        or result.get("status") != "PASSED"
        or result.get("depth_admission") != "PASS"
        or result.get("consumption_authorized") is not True
        or result.get("authorized_scopes") != [AUTHORIZED_SCOPE]
        or result.get("source_mutated") is not False
        or result.get("external_accuracy") != "UNVERIFIED"
    ):
        raise RuntimeError("BLOCKED_UPSTREAM_ENCODED_DEPTH_NOT_ADMITTED")
    if (
        contract.get("schema_version")
        != "0915-foundationstereo-encoded-depth-contract-v1"
        or contract.get("task_id") != DEPTH_TASK_ID
        or contract.get("session_id") != SESSION_ID
        or contract.get("frame_count") != 150
        or contract.get("frame_geometry") != [640, 480]
        or contract.get("depth_reference") != DEPTH_REFERENCE
        or contract.get("depth_to_sam_resize_homography")
        != [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]
        or contract.get("native_model_confidence") != "ABSENT_NOT_FABRICATED"
        or contract.get("occluded_or_hidden_geometry") != "INVALID_NOT_COMPLETED"
        or contract.get("consumption_authorized") is not True
        or contract.get("authorized_scopes") != [AUTHORIZED_SCOPE]
    ):
        raise RuntimeError("BLOCKED_UPSTREAM_ENCODED_DEPTH_CONTRACT_DRIFT")
    if (
        summary.get("schema_version")
        != "0915-foundationstereo-encoded-depth-summary-v1"
        or summary.get("task_id") != DEPTH_TASK_ID
        or summary.get("session_id") != SESSION_ID
        or summary.get("status") != "PASS"
        or summary.get("depth_admission") != "PASS"
        or summary.get("frame_count") != 150
        or summary.get("model_load_count") != 1
        or summary.get("model_inference_count") != 300
        or summary.get("quality", {}).get("passed") is not True
        or summary.get("review_decode", {}).get("full_decode") is not True
        or summary.get("review_decode", {}).get("frame_count") != 150
        or summary.get("source_mutated") is not False
    ):
        raise RuntimeError("BLOCKED_UPSTREAM_ENCODED_DEPTH_SUMMARY_DRIFT")
    return {
        "result": result,
        "contract": contract,
        "summary": summary,
        "references": {
            "depth_result": ref(depth_root / "RESULT.json"),
            "depth_adapter_contract": ref(depth_root / "ADAPTER_CONTRACT.json"),
            "depth_rgb_alignment_qa": ref(depth_root / "RGB_ALIGNMENT_QA.json"),
            "depth_contract": ref(depth_root / "DEPTH_CONTRACT.json"),
            "depth_summary": ref(depth_root / "DEPTH_SUMMARY.json"),
        },
    }


class PackedMask:
    def __init__(
        self,
        packed: np.ndarray,
        frame_count: int,
        height: int,
        width: int,
        bitorder: str,
    ) -> None:
        self.packed = packed
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.bitorder = bitorder

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.frame_count, self.height, self.width

    def unpack(self, frame_index: int) -> np.ndarray:
        return np.unpackbits(
            self.packed[frame_index],
            bitorder=self.bitorder,
            count=self.height * self.width,
        ).reshape(self.height, self.width).astype(bool)


def load_packed_mask(path: Path) -> PackedMask:
    with np.load(path, allow_pickle=False) as archive:
        mask = PackedMask(
            np.asarray(archive["packed"], np.uint8),
            int(archive["frame_count"]),
            int(archive["height"]),
            int(archive["width"]),
            str(archive["bitorder"]),
        )
    expected = (mask.frame_count, (mask.height * mask.width + 7) // 8)
    if mask.packed.shape != expected or mask.bitorder != "big":
        raise RuntimeError(f"packed SAM mask contract drift: {path}")
    return mask


def load_sam_inputs(
    sam_root: Path,
) -> tuple[dict[str, PackedMask], dict[str, list[str]], dict[str, Any]]:
    manifest_path = sam_root / "ROLE_MANIFEST.json"
    temporal_path = sam_root / "TEMPORAL_STATE_LEDGER.json"
    manifest = load_json(manifest_path)
    temporal = load_json(temporal_path)
    if manifest.get("image_domain") != (
        "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP"
    ):
        raise RuntimeError("SAM image-domain drift")
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
            raise RuntimeError(f"missing physical task-object instance: {instance_id}")
        path = sam_root / str(row["mask_archive"])
        masks[instance_id] = load_packed_mask(path)
        mask_refs[instance_id] = ref(path)
        states = states_by_id.get(instance_id)
        if states is None or len(states) != masks[instance_id].frame_count:
            raise RuntimeError(f"SAM temporal-state axis drift: {instance_id}")
    if len({mask.shape for mask in masks.values()}) != 1:
        raise RuntimeError("three physical card mask geometries differ")
    if next(iter(masks.values())).shape[1:] != MASK_SHAPE:
        raise RuntimeError("SAM masks are not 1280x960")
    return masks, states_by_id, {
        "sam_role_manifest": ref(manifest_path),
        "sam_temporal_state_ledger": ref(temporal_path),
        "sam_masks": mask_refs,
    }


def load_depth_frame(
    path: Path, expected_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        frame_id = int(archive["frame_id"])
        depth = np.asarray(archive["depth_m"], np.float64)
        valid = np.asarray(archive["valid"], bool)
        intrinsics = np.asarray(archive["physical_left_intrinsics"], np.float64)
        reference = str(archive["depth_reference"])
    if frame_id != expected_index:
        raise RuntimeError(f"non-contiguous encoded depth frame: {path}")
    if depth.shape != DEPTH_SHAPE or valid.shape != DEPTH_SHAPE:
        raise RuntimeError(f"encoded depth geometry drift: {path}")
    if intrinsics.shape != (3, 3) or reference != DEPTH_REFERENCE:
        raise RuntimeError(f"encoded depth coordinate contract drift: {path}")
    return depth, valid, intrinsics


def build_depth_frame_manifest(depth_root: Path) -> dict[str, Any]:
    paths = sorted((depth_root / "frames").glob("*.npz"))
    if len(paths) != 150:
        raise RuntimeError(f"encoded depth frame denominator drift: {len(paths)}")
    return {
        "schema_version": "0915-planar-object6d-encoded-depth-input-manifest-v1",
        "session_id": SESSION_ID,
        "frame_count": 150,
        "frames": [ref(path) for path in paths],
    }


def build_run_signature(
    packet_path: Path, depth_manifest: dict[str, Any], executor_epoch: int,
) -> dict[str, Any]:
    fixed_inputs = [
        DEPTH_ROOT / "RESULT.json",
        DEPTH_ROOT / "ADAPTER_CONTRACT.json",
        DEPTH_ROOT / "RGB_ALIGNMENT_QA.json",
        DEPTH_ROOT / "DEPTH_CONTRACT.json",
        DEPTH_ROOT / "DEPTH_SUMMARY.json",
        DEPTH_ROOT / "DEPTH_WORKER_RESULT.json",
        SAM_ROOT / "ROLE_MANIFEST.json",
        SAM_ROOT / "TEMPORAL_STATE_LEDGER.json",
        *(SAM_ROOT / f"masks/{instance_id}.npz" for instance_id in INSTANCE_IDS),
    ]
    payload = {
        "schema_version": "0915-planar-object6d-observability-run-signature-v2",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "executor_epoch": executor_epoch,
        "weights": "ABSENT",
        "coordinate_domain": DEPTH_REFERENCE,
        "depth_to_sam_mapping": MAPPING_CONTRACT,
        "input_artifacts": [ref(path) for path in fixed_inputs],
        "depth_frame_input_manifest": depth_manifest,
        "code": [
            ref(Path(__file__)),
            ref(ROOT / "src/chaoyang/pipeline/object6d_planar_observability_v2.py"),
            ref(ROOT / "src/chaoyang/pipeline/object6d_planar_observability_v1.py"),
        ],
        "schema": ref(SCHEMA),
        "task_packet": ref(packet_path),
        "config": {
            "instances": list(INSTANCE_IDS),
            "support_entity": "black_card_tray",
            "semantic_group": "card_set",
            "hidden_geometry_policy": "INVALID_NOT_COMPLETED",
        },
    }
    return {**payload, "run_signature_sha256": canonical_sha256(payload)}


def summarize(document: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for obj in document["objects"]:
        rows = obj["frames"]
        objects.append({
            "instance_id": obj["instance_id"],
            "frame_count": len(rows),
            "mask_state_counts": dict(Counter(row["mask_state"] for row in rows)),
            "observable_frame_counts": {
                name: sum(
                    row[name]["observability"] != "UNOBSERVABLE" for row in rows
                )
                for name in COMPONENT_NAMES
            },
            "unobservable_reason_counts": {
                name: dict(Counter(
                    row[name]["reason"]
                    for row in rows
                    if row[name]["observability"] == "UNOBSERVABLE"
                ))
                for name in COMPONENT_NAMES
            },
        })
    return {
        "schema_version": "0915-planar-object6d-observability-summary-v2",
        "status": "COMPLETED_DEVELOPMENT_OBSERVABILITY",
        "session_id": SESSION_ID,
        "coordinate_domain": DEPTH_REFERENCE,
        "objects": objects,
        "support_entities": document["support_entities"],
        "semantic_groups": document["semantic_groups"],
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "unified_validity_emitted": False,
        "gpu_used": False,
    }


def render_timeline(document: dict[str, Any], destination: Path) -> None:
    frame_count = int(document["frame_count"])
    scale = 5
    left = 250
    row_height = 26
    height = 68 + len(INSTANCE_IDS) * len(COMPONENT_NAMES) * row_height
    canvas = np.full((height, left + frame_count * scale + 15, 3), 245, np.uint8)
    cv2.putText(
        canvas,
        "Object6D V2 encoded domain: field observability",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green=observable | gray=unknown; tray mask absent; card_set semantic-only",
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    y = 65
    for obj in document["objects"]:
        for component in COMPONENT_NAMES:
            cv2.putText(
                canvas,
                f"{obj['instance_id']} {component}",
                (8, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            for row in obj["frames"]:
                x0 = left + int(row["frame_index"]) * scale
                color = (
                    (55, 170, 55)
                    if row[component]["observability"] != "UNOBSERVABLE"
                    else (155, 155, 155)
                )
                cv2.rectangle(
                    canvas, (x0, y + 3), (x0 + scale - 1, y + 22), color, -1,
                )
            y += row_height
    destination.parent.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError("failed to write encoded Object6D V2 timeline")


def assert_inputs_unchanged(snapshot: dict[str, str]) -> None:
    for raw, expected_sha in snapshot.items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"read-only Object6D V2 input changed: {path}")


def run_canary(
    output: Path, visual: Path, depth_manifest: dict[str, Any],
) -> dict[str, Any]:
    depth_admission = validate_depth_upstream(DEPTH_ROOT)
    mapping, mapping_valid = depth_to_sam_pixel_center_map()
    masks, states, sam_inputs = load_sam_inputs(SAM_ROOT)
    atomic_json(output / "DEPTH_FRAME_INPUT_MANIFEST.json", depth_manifest)
    inputs = {
        **depth_admission["references"],
        "depth_frame_manifest": ref(output / "DEPTH_FRAME_INPUT_MANIFEST.json"),
        "depth_to_sam_mapping": MAPPING_CONTRACT,
        **sam_inputs,
    }
    bound_refs = [
        value for value in inputs.values()
        if isinstance(value, dict) and {"path", "sha256"}.issubset(value)
    ] + list(sam_inputs["sam_masks"].values())
    snapshot = {str(value["path"]): str(value["sha256"]) for value in bound_refs}
    snapshot.update({
        str(value["path"]): str(value["sha256"])
        for value in depth_manifest["frames"]
    })
    frame_paths = sorted((DEPTH_ROOT / "frames").glob("*.npz"))
    if any(mask.frame_count != len(frame_paths) for mask in masks.values()):
        raise RuntimeError("encoded depth and three-card SAM frame axes differ")
    object_frames: dict[str, list[dict[str, Any]]] = {
        instance_id: [] for instance_id in INSTANCE_IDS
    }
    for frame_index, path in enumerate(frame_paths):
        expected = depth_manifest["frames"][frame_index]
        if ref(path) != expected:
            raise RuntimeError(f"signed encoded depth frame drift: {path}")
        depth, valid, intrinsics = load_depth_frame(path, frame_index)
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
                registration_valid=mapping_valid,
                depth_reference=DEPTH_REFERENCE,
                registration_authority=MASK_REGISTRATION_AUTHORITY,
            )))
    document = build_observability_document(
        session_id=SESSION_ID,
        object_frames=object_frames,
        inputs=inputs,
    )
    jsonschema.Draft202012Validator(load_json(SCHEMA)).validate(document)
    atomic_json(output / "OBJECT6D_OBSERVABILITY_V2.json", document)
    summary = summarize(document)
    atomic_json(output / "OBJECT6D_SUMMARY_V2.json", summary)
    timeline = visual / "0915_PLANAR_OBJECT6D_OBSERVABILITY_TIMELINE_V2.png"
    render_timeline(document, timeline)
    assert_inputs_unchanged(snapshot)
    return {
        "document": ref(output / "OBJECT6D_OBSERVABILITY_V2.json"),
        "summary": ref(output / "OBJECT6D_SUMMARY_V2.json"),
        "timeline": ref(timeline),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()

    packet = validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    validate_output_namespace(output, visual, receipt)
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive executor epoch and fencing token >=16 chars required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    depth_manifest = build_depth_frame_manifest(DEPTH_ROOT)
    signature = build_run_signature(packet_path, depth_manifest, args.executor_epoch)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": "0915-planar-object6d-observability-writer-claim-v2",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(
            args.fencing_token.encode("utf-8")
        ).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output),
        "task_packet": ref(packet_path),
    }
    atomic_json(output / "CLAIM.json", claim)
    validate_writer_claim(
        output / "CLAIM.json",
        output=output,
        signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch,
    )
    heartbeat(os.getpid())
    artifacts = run_canary(output, visual, depth_manifest)
    validate_writer_claim(
        output / "CLAIM.json",
        output=output,
        signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch,
    )
    result = {
        "schema_version": "0915-planar-object6d-observability-canary-result-v2",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": "PASSED",
        "object6d_admission": "PASS_DEVELOPMENT_VISIBLE_SURFACE_ONLY",
        "coordinate_domain": DEPTH_REFERENCE,
        "weights": "ABSENT",
        "source_mutated": False,
        "gpu_used": False,
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "unified_validity_emitted": False,
        "artifacts": artifacts,
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-planar-object6d-observability-run-receipt-v2",
        "task_id": TASK_ID,
        "status": "PASSED",
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
        "gpu_used": False,
    })
    print(json.dumps({"status": "PASSED", "result": str(output / "RESULT.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
