#!/usr/bin/env python3
"""Run the bounded 0915 stereo-preflight and Interaction-v0a CPU lanes.

One governance task owns two disjoint writer lanes.  Each child receives a
lane-specific fencing token, writes into a private staging directory, and
atomically publishes exactly one immutable terminal.  The parent is the only
process allowed to heartbeat the global task or create the joined result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops.analyze_0915_stereo_domain_preflight_v1 import run_preflight
from chaoyang.pipeline.interaction_evidence_v0a import (
    estimate_interaction_evidence_v0a,
    interaction_evidence_v0a_to_record,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_stereo_interaction_cpu_canary_v1"
PHASE = "0915_STEREO_INTERACTION_CPU_CANARY"
SESSION = (
    Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915")
    / "cleaned/playing_cards/play_cards_0915_001"
)
HAWOR = (
    ROOT / "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/"
    "attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/"
    "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
)
SAM_ROOT = (
    ROOT / "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001"
)
INTERACTION_SCHEMA = ROOT / "contracts/interaction_evidence_v0a.schema.json"
LANES = ("stereo_preflight", "interaction_v0a")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def ref_at(path: Path, published_path: Path) -> dict[str, Any]:
    """Hash ``path`` while binding the immutable post-rename location."""

    candidate = path.resolve(strict=True)
    return {
        "path": str(published_path.resolve()),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
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


def process_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    return int(fields[21])


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen specification")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("parallel CPU evidence task must not bind a weight")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("CPU evidence task is not current next_task")
    task = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING",
    }:
        raise RuntimeError("CPU evidence task is not executable")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize CPU evidence task")
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


def _load_packed_masks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        packed = np.asarray(archive["packed"], np.uint8)
        frame_count = int(archive["frame_count"])
        height = int(archive["height"])
        width = int(archive["width"])
        bitorder = str(archive["bitorder"])
    return np.unpackbits(
        packed, axis=1, count=height * width, bitorder=bitorder,
    ).reshape(frame_count, height, width).astype(bool)


def _interaction_inputs() -> tuple[dict[str, Any], list[str], list[str]]:
    manifest = load_json(SAM_ROOT / "ROLE_MANIFEST.json")
    temporal = load_json(SAM_ROOT / "TEMPORAL_STATE_LEDGER.json")
    state_by_id = {
        str(row["instance_id"]): [str(item["state"]) for item in row["frames"]]
        for row in temporal["instances"]
    }
    objects = [
        row for row in manifest["instances"] if row.get("role") == "task_object"
    ]
    object_ids = [str(row["instance_id"]) for row in objects]
    object_masks = np.stack(
        [_load_packed_masks(SAM_ROOT / str(row["mask_archive"])) for row in objects],
        axis=1,
    )
    object_states = np.asarray([state_by_id[value] for value in object_ids]).T

    with np.load(HAWOR, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        provenance = np.asarray(archive["provenance"]).astype(str)
        tip_indices = np.asarray(archive["mano_tip_indices"], np.int64)
        sides = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
        fps = float(archive["fps"])
    if sides != ["left", "right"] or tip_indices.tolist() != [4, 8, 12, 16, 20]:
        raise RuntimeError("HaWoR side or MANO fingertip contract drift")
    if joints.shape[:3] != (2, object_masks.shape[0], 21):
        raise RuntimeError("HaWoR and SAM frame axes differ")
    tips = joints[:, :, tip_indices].transpose(1, 0, 2, 3).reshape(
        object_masks.shape[0], 10, 2,
    )
    tip_states = np.full((object_masks.shape[0], 10), "unknown", dtype="<U16")
    direct = observed & (provenance == "BOUNDED_PARAMETER_FIT")
    direct_tips = np.repeat(direct.T[:, :, None], 5, axis=2).reshape(
        object_masks.shape[0], 10,
    )
    finite_in_frame = (
        np.isfinite(tips).all(axis=2)
        & (tips[:, :, 0] >= 0.0) & (tips[:, :, 0] < object_masks.shape[3])
        & (tips[:, :, 1] >= 0.0) & (tips[:, :, 1] < object_masks.shape[2])
    )
    tip_states[direct_tips & finite_in_frame] = "direct_observed"
    fingertip_ids = [f"{side}_{finger}" for side in sides for finger in FINGERS]

    frames = sorted((SESSION / "preprocess/all_data").glob("*/training_data.json"))
    if len(frames) != object_masks.shape[0]:
        raise RuntimeError("processed tactile frame count differs from SAM")
    tactile_active = np.zeros((len(frames), 10), bool)
    tactile_valid = np.zeros((len(frames), 10), bool)
    tactile_times = np.full((len(frames), 10), np.nan, np.float64)
    frame_times = np.arange(len(frames), dtype=np.float64) / fps
    for frame_index, path in enumerate(frames):
        data = load_json(path)
        tactile = data.get("entities", {}).get("tactile")
        if not isinstance(tactile, dict):
            raise RuntimeError(f"missing entities.tactile: {path}")
        for side_index, side in enumerate(sides):
            row = tactile.get(side)
            if not isinstance(row, dict):
                raise RuntimeError(f"missing entities.tactile.{side}: {path}")
            grid = np.asarray(row.get("finger_grid_5x4x8"), np.float64)
            valid_grid = np.asarray(row.get("valid_mask_5x4x8"), bool)
            if grid.shape != (5, 4, 8) or valid_grid.shape != (5, 4, 8):
                raise RuntimeError(f"invalid tactile grid shape: {path}")
            slot = slice(side_index * 5, (side_index + 1) * 5)
            tactile_active[frame_index, slot] = np.any(
                (grid > 0.0) & valid_grid, axis=(1, 2),
            )
            source_valid = bool(row.get("offline_source_valid"))
            tactile_valid[frame_index, slot] = source_valid
            if source_valid:
                offset_ms = float(row.get("offline_source_offset_ms"))
                tactile_times[frame_index, slot] = (
                    frame_times[frame_index] + offset_ms / 1000.0
                )
    return {
        "fingertip_xy_px": tips,
        "fingertip_observation_state": tip_states,
        "object_masks": object_masks,
        "object_mask_state": object_states,
        "frame_timestamps_s": frame_times,
        "tactile_active": tactile_active,
        "tactile_source_valid": tactile_valid,
        "tactile_timestamps_s": tactile_times,
    }, object_ids, fingertip_ids


def _render_interaction_timeline(
    record: dict[str, Any], destination: Path, *, frame_count: int,
    object_ids: list[str], fingertip_ids: list[str],
) -> None:
    scale = 6
    left = 260
    top = 75
    row_height = 18
    rows = len(object_ids) * len(fingertip_ids)
    canvas = np.full((top + rows * row_height + 45, left + frame_count * scale + 25, 3), 245, np.uint8)
    cv2.putText(canvas, "Interaction v0a: image-plane evidence only", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "green=adjacent | blue=not adjacent | gray=unknown | yellow=tactile-supported hypothesis",
                (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
    lookup = {
        (row["frame_index"], row["object_instance_id"], row["fingertip_id"]): row
        for row in record["observations"]
    }
    for object_index, object_id in enumerate(object_ids):
        for tip_index, tip_id in enumerate(fingertip_ids):
            row_index = object_index * len(fingertip_ids) + tip_index
            y0 = top + row_index * row_height
            cv2.putText(canvas, f"{object_id[-2:]} {tip_id}", (8, y0 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA)
            for frame in range(frame_count):
                value = lookup[(frame, object_id, tip_id)]
                status = value["adjacency"]
                color = ((55, 175, 55) if status == "ADJACENT_2D"
                         else (130, 80, 30) if status == "NOT_ADJACENT_2D"
                         else (165, 165, 165))
                x0 = left + frame * scale
                cv2.rectangle(canvas, (x0, y0), (x0 + scale - 1, y0 + row_height - 2), color, -1)
                if value["tactile"] == "TACTILE_SUPPORTED_HYPOTHESIS":
                    cv2.circle(canvas, (x0 + scale // 2, y0 + row_height // 2), 2,
                               (0, 230, 255), -1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError("interaction timeline write failed")


def run_interaction_lane(output: Path) -> dict[str, Any]:
    inputs, object_ids, fingertip_ids = _interaction_inputs()
    result = estimate_interaction_evidence_v0a(**inputs)
    record = interaction_evidence_v0a_to_record(
        result, object_ids=object_ids, fingertip_ids=fingertip_ids,
    )
    schema = load_json(INTERACTION_SCHEMA)
    jsonschema.Draft202012Validator(schema).validate(record)
    atomic_json(output / "INTERACTION_EVIDENCE_V0A.json", record)
    adjacency = Counter(row["adjacency"] for row in record["observations"])
    approach = Counter(row["approach"] for row in record["observations"])
    co_motion = Counter(row["co_motion"] for row in record["observations"])
    tactile = Counter(row["tactile"] for row in record["observations"])
    summary = {
        "schema_version": "0915-interaction-v0a-single-session-summary-v1",
        "status": "COMPLETED_DEVELOPMENT_EVIDENCE",
        "session_id": SESSION.name,
        "frame_count": int(inputs["object_masks"].shape[0]),
        "object_ids": object_ids,
        "fingertip_ids": fingertip_ids,
        "counts": {
            "adjacency": dict(adjacency), "approach": dict(approach),
            "co_motion": dict(co_motion), "tactile": dict(tactile),
        },
        "inputs": {
            "hawor": ref(HAWOR),
            "role_manifest": ref(SAM_ROOT / "ROLE_MANIFEST.json"),
            "temporal_state_ledger": ref(SAM_ROOT / "TEMPORAL_STATE_LEDGER.json"),
            "processed_tactile_scope": "entities.tactile only",
        },
        "access_contract": {
            "pico26": "NOT_CONSUMED", "controller_pose": "NOT_CONSUMED",
            "trackingData_hand": "NOT_CONSUMED", "source_mutated": False,
        },
        "authority": "DEVELOPMENT_WEAK_EVIDENCE_ONLY",
        "relative_z": False,
        "occlusion_order": False,
        "contact_ground_truth": False,
    }
    atomic_json(output / "SUMMARY.json", summary)
    _render_interaction_timeline(
        record, output / "INTERACTION_V0A_TIMELINE.png",
        frame_count=summary["frame_count"], object_ids=object_ids,
        fingertip_ids=fingertip_ids,
    )
    return summary


def lane_signature(lane: str) -> dict[str, Any]:
    common = {
        "schema_version": "0915-parallel-cpu-lane-signature-v1",
        "parent_task_id": TASK_ID,
        "lane_id": lane,
        "session_id": SESSION.name,
        "weights": "ABSENT",
        "source_mutation_allowed": False,
    }
    if lane == "stereo_preflight":
        common["inputs"] = {
            "session": str(SESSION),
            "code": str(ROOT / "src/chaoyang/ops/analyze_0915_stereo_domain_preflight_v1.py"),
        }
    elif lane == "interaction_v0a":
        common["inputs"] = {
            "hawor": str(HAWOR), "sam_root": str(SAM_ROOT),
            "processed_tactile": str(SESSION / "preprocess/all_data"),
            "code": str(ROOT / "src/chaoyang/pipeline/interaction_evidence_v0a.py"),
        }
    else:
        raise ValueError(f"unsupported lane: {lane}")
    return common


def run_lane(args: argparse.Namespace) -> int:
    lane = str(args.lane)
    if lane not in LANES:
        raise RuntimeError(f"invalid lane: {lane}")
    final = args.lane_output.resolve()
    expected = (args.attempt_root.resolve() / "lanes" / lane).resolve()
    if final != expected:
        raise RuntimeError("lane write root escapes its fixed writer fence")
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"fresh lane output required: {final}")
    staging = args.attempt_root.resolve() / ".lane_staging" / f"{lane}-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"fresh lane staging required: {staging}")
    staging.mkdir(parents=True)
    signature = lane_signature(lane)
    signature_sha = _canonical_sha(signature)
    expected_token = hashlib.sha256(
        f"{args.parent_fencing_token}:{lane}:{signature_sha}".encode("utf-8")
    ).hexdigest()
    if args.lane_fencing_token != expected_token:
        raise RuntimeError("lane fencing token mismatch")
    claim = {
        "schema_version": "0915-parallel-cpu-lane-claim-v1",
        "parent_task_id": TASK_ID,
        "lane_id": lane,
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "run_signature_sha256": signature_sha,
        "fencing_token_sha256": hashlib.sha256(
            args.lane_fencing_token.encode("utf-8")
        ).hexdigest(),
        "unique_write_root": str(final),
    }
    atomic_json(staging / "RUN_SIGNATURE.json", {
        **signature, "run_signature_sha256": signature_sha,
    })
    atomic_json(staging / "LANE_CLAIM.json", claim)
    try:
        if lane == "stereo_preflight":
            result = run_preflight(SESSION)
            atomic_json(staging / "STEREO_PREFLIGHT.json", result)
            status = result["status"]
            admission = bool(result["decision"]["gpu_depth_allowed"])
            lane_result = {
                "schema_version": "0915-stereo-preflight-lane-result-v1",
                "status": status,
                "gpu_depth_allowed": admission,
                "stereo_preflight": ref_at(
                    staging / "STEREO_PREFLIGHT.json",
                    final / "STEREO_PREFLIGHT.json",
                ),
            }
        else:
            result = run_interaction_lane(staging)
            status = result["status"]
            lane_result = {
                "schema_version": "0915-interaction-v0a-lane-result-v1",
                "status": status,
                "interaction_summary": ref_at(
                    staging / "SUMMARY.json", final / "SUMMARY.json",
                ),
                "interaction_evidence": ref_at(
                    staging / "INTERACTION_EVIDENCE_V0A.json",
                    final / "INTERACTION_EVIDENCE_V0A.json",
                ),
                "timeline": ref_at(
                    staging / "INTERACTION_V0A_TIMELINE.png",
                    final / "INTERACTION_V0A_TIMELINE.png",
                ),
            }
        atomic_json(staging / "RESULT.json", lane_result)
        terminal = {
            "schema_version": "0915-parallel-cpu-lane-terminal-v1",
            "parent_task_id": TASK_ID,
            "lane_id": lane,
            "status": status,
            "run_signature_sha256": signature_sha,
            "fencing_token_sha256": claim["fencing_token_sha256"],
            "result": ref_at(staging / "RESULT.json", final / "RESULT.json"),
        }
        atomic_json(staging / "LANE_TERMINAL.json", terminal)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return 0
    except Exception as exc:
        atomic_json(staging / "FAILED_RUNTIME.json", {
            "schema_version": "0915-parallel-cpu-lane-failure-v1",
            "parent_task_id": TASK_ID, "lane_id": lane,
            "status": "FAILED_RUNTIME_FINAL",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def validate_lane_terminal(
    lane: str, root: Path, *, parent_token: str, executor_epoch: int,
    expected_pid: int,
) -> dict[str, Any]:
    claim = load_json(root / "LANE_CLAIM.json")
    terminal = load_json(root / "LANE_TERMINAL.json")
    signature = load_json(root / "RUN_SIGNATURE.json")
    signature_payload = dict(signature)
    declared_signature_sha = str(signature_payload.pop("run_signature_sha256"))
    if declared_signature_sha != _canonical_sha(signature_payload):
        raise RuntimeError(f"{lane}: run signature SHA mismatch")
    token = hashlib.sha256(
        f"{parent_token}:{lane}:{declared_signature_sha}".encode("utf-8")
    ).hexdigest()
    token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expected_root = str(root.resolve())
    checks = {
        "claim_parent": claim.get("parent_task_id") == TASK_ID,
        "claim_lane": claim.get("lane_id") == lane,
        "claim_pid": claim.get("pid") == expected_pid,
        "claim_startticks": isinstance(claim.get("proc_start_ticks"), int),
        "claim_epoch": claim.get("executor_epoch") == executor_epoch,
        "claim_signature": claim.get("run_signature_sha256") == declared_signature_sha,
        "claim_token": claim.get("fencing_token_sha256") == token_sha,
        "claim_write_root": claim.get("unique_write_root") == expected_root,
        "terminal_parent": terminal.get("parent_task_id") == TASK_ID,
        "terminal_lane": terminal.get("lane_id") == lane,
        "terminal_signature": terminal.get("run_signature_sha256") == declared_signature_sha,
        "terminal_token": terminal.get("fencing_token_sha256") == token_sha,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{lane}: lane terminal fence mismatch: {failed}")
    result_ref = terminal.get("result")
    if not isinstance(result_ref, dict):
        raise RuntimeError(f"{lane}: terminal result ref missing")
    result_path = Path(str(result_ref.get("path")))
    if result_path != (root / "RESULT.json").resolve():
        raise RuntimeError(f"{lane}: terminal result path escapes lane")
    if ref(result_path) != result_ref:
        raise RuntimeError(f"{lane}: terminal result ref mismatch")
    return {"claim": claim, "terminal": terminal, "result": load_json(result_path)}


def _render_join_visual(stereo: dict[str, Any], timeline: Path, output: Path) -> None:
    chart = cv2.imread(str(timeline), cv2.IMREAD_COLOR)
    if chart is None:
        raise RuntimeError("cannot decode interaction timeline")
    header = np.full((230, chart.shape[1], 3), 248, np.uint8)
    raw = stereo["candidates"]["raw_resize"]["quality"]
    rect = stereo["candidates"]["calibrated_rectified"]["quality"]
    lines = [
        "0915 play_cards_0915_001 | CPU evidence join",
        f"Stereo admission: {stereo['status']} | selected: {stereo['decision']['selected_candidate']}",
        f"raw resize vertical px: median {raw['median_vertical_error_px']:.3f} | P90 {raw['p90_vertical_error_px']:.3f} | P95 {raw['p95_vertical_error_px']:.3f}",
        f"rectified vertical px: median {rect['median_vertical_error_px']:.3f} | P90 {rect['p90_vertical_error_px']:.3f} | P95 {rect['p95_vertical_error_px']:.3f}",
        f"baseline {stereo['camera_calibration']['baseline_m']:.6f} m | external accuracy UNVERIFIED",
        "Interaction v0a below is IMAGE_2D_ONLY; no relative-Z, occlusion order, contact truth, Object6D or Robot authority.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(header, line, (16, 34 + index * 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58 if index else 0.70,
                    (25, 25, 25), 2 if index == 0 else 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.vstack((header, chart))
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("CPU evidence review image write failed")


def run_parent(args: argparse.Namespace) -> int:
    packet = validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable attempt required: {output}")
    if visual.exists() or visual.is_symlink():
        raise RuntimeError(f"fresh visual root required: {visual}")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive executor epoch and fencing token >=16 chars required")
    output.mkdir(parents=True)
    (output / "lanes").mkdir()
    (output / ".lane_staging").mkdir()
    started = time.time()
    parent_signature = {
        "schema_version": "0915-stereo-interaction-cpu-run-signature-v1",
        "task_id": TASK_ID,
        "lanes": list(LANES),
        "weights": "ABSENT",
        "session_id": SESSION.name,
        "executor_epoch": args.executor_epoch,
    }
    parent_signature_sha = _canonical_sha(parent_signature)
    atomic_json(output / "RUN_SIGNATURE.json", {
        **parent_signature, "run_signature_sha256": parent_signature_sha,
    })
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-stereo-interaction-cpu-claim-v1",
        "task_id": TASK_ID, "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(
            args.fencing_token.encode("utf-8")
        ).hexdigest(),
        "run_signature_sha256": parent_signature_sha,
        "task_packet": ref(ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"),
    })
    heartbeat(os.getpid())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    processes: dict[str, subprocess.Popen[str]] = {}
    logs: dict[str, Any] = {}
    for lane in LANES:
        signature_sha = _canonical_sha(lane_signature(lane))
        lane_token = hashlib.sha256(
            f"{args.fencing_token}:{lane}:{signature_sha}".encode("utf-8")
        ).hexdigest()
        log = (output / f"{lane.upper()}.log").open(
            "w", encoding="utf-8", buffering=1,
        )
        logs[lane] = log
        command = [
            sys.executable, "-m",
            "chaoyang.ops.run_0915_stereo_interaction_cpu_canary_v1",
            "--lane", lane,
            "--attempt-root", str(output),
            "--lane-output", str(output / "lanes" / lane),
            "--executor-epoch", str(args.executor_epoch),
            "--parent-fencing-token", args.fencing_token,
            "--lane-fencing-token", lane_token,
        ]
        processes[lane] = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        while any(process.poll() is None for process in processes.values()):
            time.sleep(max(2, args.heartbeat_seconds))
            heartbeat(os.getpid())
            atomic_json(output / "PROGRESS.json", {
                "schema_version": "0915-stereo-interaction-cpu-progress-v1",
                "task_id": TASK_ID,
                "elapsed_seconds": time.time() - started,
                "lanes": {
                    lane: {"pid": process.pid, "returncode": process.poll()}
                    for lane, process in processes.items()
                },
            })
    finally:
        for log in logs.values():
            log.close()
    returncodes = {lane: process.returncode for lane, process in processes.items()}
    if any(value != 0 for value in returncodes.values()):
        result = {
            "schema_version": "0915-stereo-interaction-cpu-join-v1",
            "task_id": TASK_ID, "status": "FAILED_RUNTIME_FINAL",
            "returncodes": returncodes,
            "first_blocker": "CPU_LANE_RUNTIME_FAILURE",
            "weights": packet["weights"], "source_mutated": False,
        }
        atomic_json(output / "RESULT.json", result)
        atomic_json(output / "RUN_RECEIPT.json", {
            "schema_version": "0915-stereo-interaction-cpu-run-receipt-v1",
            "status": result["status"], "result": ref(output / "RESULT.json"),
        })
        return 1
    validated = {
        lane: validate_lane_terminal(
            lane, output / "lanes" / lane,
            parent_token=args.fencing_token,
            executor_epoch=args.executor_epoch,
            expected_pid=processes[lane].pid,
        )
        for lane in LANES
    }
    stereo_path = output / "lanes/stereo_preflight/STEREO_PREFLIGHT.json"
    stereo = load_json(stereo_path)
    interaction_summary_path = output / "lanes/interaction_v0a/SUMMARY.json"
    timeline = output / "lanes/interaction_v0a/INTERACTION_V0A_TIMELINE.png"
    visual.mkdir(parents=True)
    review = visual / "0915_STEREO_INTERACTION_CPU_REVIEW.png"
    _render_join_visual(stereo, timeline, review)
    result = {
        "schema_version": "0915-stereo-interaction-cpu-join-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "weights": packet["weights"],
        "session_id": SESSION.name,
        "returncodes": returncodes,
        "lane_fences_valid": True,
        "lane_results": {
            lane: ref(output / "lanes" / lane / "RESULT.json") for lane in LANES
        },
        "stereo_preflight": ref(stereo_path),
        "stereo_gpu_depth_allowed": bool(stereo["decision"]["gpu_depth_allowed"]),
        "interaction_summary": ref(interaction_summary_path),
        "visual": ref(review),
        "source_mutated": False,
        "gpu_used": False,
        "first_blocker": None,
        "claim_limit": (
            "Completed CPU evidence publication. A PASSED outer task does not by "
            "itself authorize Depth; a successor must separately require "
            "stereo_gpu_depth_allowed=true. Interaction remains 2D weak evidence."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-stereo-interaction-cpu-run-receipt-v1",
        "status": result["status"], "result": ref(output / "RESULT.json"),
        "visual": ref(review),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--lane-output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--visual-root", type=Path)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token")
    parser.add_argument("--parent-fencing-token")
    parser.add_argument("--lane-fencing-token")
    parser.add_argument("--heartbeat-seconds", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lane is not None:
        required = {
            "attempt_root": args.attempt_root,
            "lane_output": args.lane_output,
            "parent_fencing_token": args.parent_fencing_token,
            "lane_fencing_token": args.lane_fencing_token,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"lane arguments missing: {missing}")
        return run_lane(args)
    required = {
        "output_root": args.output_root,
        "visual_root": args.visual_root,
        "fencing_token": args.fencing_token,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"parent arguments missing: {missing}")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
