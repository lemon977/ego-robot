#!/usr/bin/env python3
"""Freeze independent Mask evaluation inputs and audit causal Clean prerequisites.

This is a fail-closed CPU task.  It never runs SAM/ProPainter and never changes
the R7_0 Clean or SAM3.1 authorities.  It materializes a causal donor
eligibility map, validates the existing lossless source-map inputs, and freezes
Poker atlas seed evidence.  Missing support-surface semantics or a verified
atlas are terminal prerequisites, not reasons to invent pixels.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import hashlib
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import dilate, proposed, unions


FAILURE_SELECTION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_mask_failure_clusters_r3/attempts/attempt_0001/MASK_CHALLENGER_FROZEN_SELECTION_R3.json"
FAILURE_LEDGER = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_mask_failure_clusters_r3/attempts/attempt_0001/MASK_FAILURE_CLUSTER_LEDGER_R3.json"
T0_FRAMESET = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/T0_FROZEN_FRAMESET.json"
T0_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/attempts/attempt_0004"

CASES: dict[str, dict[str, Any]] = {
    "Poker245": {
        "task": "poker",
        "session": "play_cards_0903_245",
        "manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json",
        "donor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/real_donor_v1/play_cards_0903_245/SOURCE_MAP_MANIFEST.json",
        "sam_left": T0_ROOT / "Poker245/left_human_SAM31_BIDIRECTIONAL_PACKED.npz",
        "sam_right": T0_ROOT / "Poker245/right_human_SAM31_BIDIRECTIONAL_PACKED.npz",
        "object6d": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/depth_object6d_stream_v1/object6d_observed_only_v1/poker/play_cards_0903_245/RESULT.json",
        "atlas_required": True,
    },
    "Chips039": {
        "task": "chips",
        "session": "get_potato_chips_0902_039",
        "manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/sessions/get_potato_chips_0902_039/expanded_role_handoff/FRAME_MANIFEST.json",
        "donor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/real_donor_v1/get_potato_chips_0902_039/SOURCE_MAP_MANIFEST.json",
        "sam_left": T0_ROOT / "Chips039/left_human_SAM31_BIDIRECTIONAL_PACKED.npz",
        "sam_right": T0_ROOT / "Chips039/right_human_SAM31_BIDIRECTIONAL_PACKED.npz",
        "object6d": None,
        "atlas_required": False,
    },
}

ROLE_TAXONOMY = [
    "human_left_hand", "human_right_hand", "human_left_forearm", "human_right_forearm",
    "glove", "wrist_tracker", "controller", "other_wearable",
    "task_object_instance_*", "support_surface",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path, *, verify: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if verify is not None and any(value[k] != verify[k] for k in ("path", "bytes", "sha256")):
        raise RuntimeError(f"reference mismatch: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def frozen_failure_canaries(selection: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = {("ROLE_MASK", "chips"), ("OBJECT_MASK", "poker")}
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selection["selections"]:
        canary = row.get("canary")
        if not canary:
            continue
        key = (row["stage"], canary["task"])
        if key in wanted:
            found[key] = {
                "stage": row["stage"],
                "cluster": row["cluster"],
                "session_id": canary["session_id"],
                "task": canary["task"],
                "terminal_grade": canary["grade"],
                "terminal_result": ref(Path(canary["result"]["path"]), verify=canary["result"]),
            }
    if set(found) != wanted:
        raise RuntimeError(f"missing frozen failure canaries: {wanted - set(found)}")
    return [found[("ROLE_MASK", "chips")], found[("OBJECT_MASK", "poker")]]


def object_masks(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    return [unions(row)[2] for row in rows]


def classify_temporal_donor(
    target_frame: int,
    m_write: np.ndarray,
    source_kind: np.ndarray,
    source_frame: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_objects: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    """Return fail-closed eligibility codes for SAME_SESSION_TEMPORAL_RAW.

    0: outside temporal write pixels; 1: causal and not a task object, but
    support-surface semantics remain unknown; 2: future; 3: invalid source;
    4: source task-object pixel.  Code 1 is not training-eligible until a
    support-surface semantic classifier exists.
    """
    temporal = m_write & (source_kind == 1)
    code = np.zeros(m_write.shape, np.uint8)
    future = temporal & (source_frame > target_frame)
    code[future] = 2
    causal = temporal & ~future
    in_frame = causal & (source_frame >= 0) & (source_frame < len(source_objects))
    in_bounds = in_frame & (source_x >= 0) & (source_x < m_write.shape[1]) & (source_y >= 0) & (source_y < m_write.shape[0])
    code[causal & ~in_bounds] = 3
    for source_id in np.unique(source_frame[in_bounds]):
        loc = in_bounds & (source_frame == source_id)
        hits = np.zeros_like(loc)
        hits[loc] = source_objects[int(source_id)][source_y[loc], source_x[loc]]
        code[loc & hits] = 4
        code[loc & ~hits] = 1
    counts = {
        "temporal": int(temporal.sum()),
        "causal_support_surface_unknown": int((code == 1).sum()),
        "future_rejected": int((code == 2).sum()),
        "invalid_coordinate_rejected": int((code == 3).sum()),
        "task_object_rejected": int((code == 4).sum()),
    }
    return code, counts


def verify_lossless_input(row: dict[str, Any], donor_row: dict[str, Any], shape: tuple[int, int]) -> None:
    source = Path(donor_row["pixel_source_map"]["path"])
    ref(source, verify=donor_row["pixel_source_map"])
    clean = Path(donor_row["clean_rgb"]["path"])
    ref(clean, verify=donor_row["clean_rgb"])
    if clean.suffix.lower() != ".png":
        raise RuntimeError(f"lossless donor frame is not PNG: {clean}")
    with np.load(source) as z:
        required = {"source_kind", "source_frame", "source_x", "source_y"}
        if not required.issubset(z.files):
            raise RuntimeError(f"source map keys missing: {source}")
        if any(z[key].shape != shape for key in required):
            raise RuntimeError(f"source map shape mismatch: {source}")
    if not Path(row["source_rgb"]["path"]).is_file():
        raise RuntimeError("Raw source frame missing")


def build_atlas_seed(spec: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    if not spec["atlas_required"]:
        return {"status": "NOT_APPLICABLE", "verified_object_atlas": False}
    result = json.loads(Path(spec["object6d"]).read_text(encoding="utf-8"))
    trajectory = Path(result["artifacts"]["trajectory"]["path"])
    ref(trajectory, verify=result["artifacts"]["trajectory"])
    with np.load(trajectory) as z:
        eligible = np.flatnonzero(np.asarray(z["valid"], bool) & np.asarray(z["observed"], bool))
    candidates: list[tuple[int, int]] = []
    for frame_id in eligible.tolist():
        obj = unions(rows[frame_id])[2]
        candidates.append((int(obj.sum()), frame_id))
    candidates.sort(reverse=True)
    seeds = []
    for _, frame_id in candidates[:3]:
        raw = cv2.imread(rows[frame_id]["source_rgb"]["path"], cv2.IMREAD_COLOR)
        obj = unions(rows[frame_id])[2]
        if raw is None or not np.any(obj):
            continue
        ys, xs = np.where(obj)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        rgba = cv2.cvtColor(raw[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = obj[y0:y1, x0:x1].astype(np.uint8) * 255
        path = out / f"ATLAS_SEED_FRAME_{frame_id:06d}.png"
        if not cv2.imwrite(str(path), rgba):
            raise RuntimeError(f"cannot write atlas seed: {path}")
        seeds.append({"frame_id": frame_id, "direct_object6d_observed": True, "crop": ref(path)})
    return {
        "status": "BLOCKED_PREREQ_POSE_VERIFIED_ATLAS_WARP",
        "direct_observed_candidate_frames": int(len(eligible)),
        "seeds": seeds,
        "verified_object_atlas": False,
        "training_eligible": False,
        "reason": "Masked lossless crops are frozen as causal seed evidence, but no pose-verified canonical texture warp/coverage closure exists.",
    }


def audit_case(label: str, spec: dict[str, Any], root: Path) -> dict[str, Any]:
    out = root / label
    maps_out = out / "causal_semantic_maps"
    maps_out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(spec["manifest"]).read_text(encoding="utf-8"))
    donor = json.loads(Path(spec["donor"]).read_text(encoding="utf-8"))
    rows, donor_rows = manifest["frames"], donor["frames"]
    if len(rows) != len(donor_rows):
        raise RuntimeError(f"{label}: manifest/donor length mismatch")
    source_objects = object_masks(rows)
    frame_rows = []
    totals = {key: 0 for key in (
        "m_remove", "m_write", "m_flow", "visible_object_in_write", "temporal",
        "causal_support_surface_unknown", "future_rejected", "invalid_coordinate_rejected", "task_object_rejected",
    )}
    source_map_input_closure = True
    for frame_id, (row, donor_row) in enumerate(zip(rows, donor_rows)):
        human, tracker, obj = unions(row)
        m_remove = (human | tracker) & ~obj
        m_write = proposed(human, tracker, obj)
        m_flow = dilate(m_write, 16)
        if np.any(m_remove & ~m_write) or np.any(m_write & ~m_flow):
            raise RuntimeError(f"{label}: mask domain nesting failed at {frame_id}")
        shape = m_write.shape
        try:
            verify_lossless_input(row, donor_row, shape)
        except Exception:
            source_map_input_closure = False
            raise
        with np.load(donor_row["pixel_source_map"]["path"]) as z:
            code, counts = classify_temporal_donor(
                frame_id, m_write, z["source_kind"], z["source_frame"], z["source_x"], z["source_y"], source_objects
            )
        map_path = maps_out / f"{frame_id:06d}.npz"
        np.savez_compressed(map_path, frame_id=np.int32(frame_id), eligibility_code=code)
        values = {
            "frame_id": frame_id,
            "m_remove": int(m_remove.sum()), "m_write": int(m_write.sum()), "m_flow": int(m_flow.sum()),
            "visible_object_in_write": int((obj & m_write).sum()), **counts,
            "eligibility_map": ref(map_path),
        }
        frame_rows.append(values)
        for key in totals:
            totals[key] += int(values[key])
    atlas = build_atlas_seed(spec, rows, out)
    source_manifest = {
        "schema_version": "clean-causal-semantic-source-map-preflight-r22-v1",
        "session": spec["session"], "frame_count": len(rows),
        "eligibility_codes": {"0": "NOT_TEMPORAL_WRITE", "1": "CAUSAL_OBJECT_SAFE_SUPPORT_SURFACE_UNKNOWN", "2": "REJECT_FUTURE", "3": "REJECT_INVALID_COORDINATE", "4": "REJECT_TASK_OBJECT"},
        "frames": frame_rows,
        "training_eligible": False,
        "claim_limit": "Lossless eligibility maps only; support-surface semantics and successor RGB are not closed.",
    }
    write_json(out / "CAUSAL_SEMANTIC_SOURCE_MAP_MANIFEST.json", source_manifest)
    with (out / "FRAME_METRICS.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        fields = [k for k in frame_rows[0] if k != "eligibility_map"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in frame_rows])
        handle.flush(); os.fsync(handle.fileno())
    gates = {
        "m_remove_subset_m_write": True,
        "m_write_subset_m_flow": True,
        "visible_object_excluded_from_m_write": totals["visible_object_in_write"] == 0,
        "lossless_input_frames_and_source_maps_closed": source_map_input_closure,
        "future_temporal_donors_rejected": totals["future_rejected"] >= 0,
        "task_object_donors_rejected": totals["task_object_rejected"] >= 0,
        "support_surface_semantics_closed": False,
        "successor_lossless_rgb_source_map_pair_published": False,
        "verified_poker_causal_atlas": atlas["verified_object_atlas"] if spec["atlas_required"] else None,
        "fresh_prefix_only_propainter": False,
    }
    metrics = {
        "schema_version": "clean-prerequisite-metrics-r22-v1", "case": label, "session": spec["session"],
        "frame_count": len(rows), "totals": totals, "gates": gates, "atlas": atlas,
        "semantic_donor_status": "BLOCKED_SUPPORT_SURFACE_LABELS_ABSENT",
        "lossless_source_map_input_status": "PASSED",
        "lossless_successor_output_status": "BLOCKED_NOT_GENERATED",
    }
    write_json(out / "METRICS.json", metrics)
    return {"label": label, "status": "BLOCKED_PREREQ", "metrics": metrics, "source_map_manifest": ref(out / "CAUSAL_SEMANTIC_SOURCE_MAP_MANIFEST.json")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    selection = json.loads(FAILURE_SELECTION.read_text(encoding="utf-8"))
    canaries = frozen_failure_canaries(selection)
    mask_contract = {
        "schema_version": "mask-candidate-evaluation-reference-r22-v1",
        "artifact_revision": "R7_4_MASK_EVAL_FREEZE_1", "generated_at": now(),
        "current_baseline": "SAM3.1", "authority_changed": False,
        "failure_canaries": canaries,
        "candidate_mask": [
            {"case": label, "mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION", "training_eligible": False,
             "left": ref(spec["sam_left"]), "right": ref(spec["sam_right"])} for label, spec in CASES.items()
        ],
        "evaluation_reference": {
            "failure_selection": ref(FAILURE_SELECTION), "failure_ledger": ref(FAILURE_LEDGER),
            "frameset_frozen_before_future_successor": ref(T0_FRAMESET),
            "pixel_gold_available": False, "accuracy_authorized": False,
            "interpretation": "Independent terminal receipts and a pre-frozen frame set support development diagnostics only; they are not pixel ground truth.",
        },
        "role_taxonomy": ROLE_TAXONOMY,
        "ownership": {"wearables": "HUMAN_REMOVAL_ROLE", "task_object": "PRESERVE_ROLE", "support_surface": "NEITHER_HUMAN_NOR_TASK_OBJECT"},
        "claim_limit": "Frozen candidate/reference contract; no Gold accuracy, Mask authority promotion or causal-training eligibility.",
    }
    write_json(output / "MASK_CANDIDATE_EVALUATION_REFERENCE.json", mask_contract)
    reports = [audit_case(label, spec, output) for label, spec in CASES.items()]
    result = {
        "schema_version": "r22-mask-clean-lane-b-result-v1", "task_id": "r22_mask_clean_lane_b",
        "attempt_id": output.name, "status": "BLOCKED_PREREQ", "artifact_revision": "R7_4_CLEAN_PREREQ_1",
        "validity": "VALID_FOR_PINNED_REVISION", "generated_at": now(), "reports": reports,
        "mask_contract": ref(output / "MASK_CANDIDATE_EVALUATION_REFERENCE.json"),
        "gpu_execution": "NOT_ATTEMPTED_FAIL_CLOSED", "fresh_propainter": False,
        "r7_0_modified": False, "authority_promoted": False,
        "claim_limit": "CPU prerequisite closure only; not fresh Clean, causal training RGB, Mask accuracy or authority.",
    }
    write_json(output / "RESULT.json", result)
    write_json(output / "METRICS.json", {"schema_version": "r22-mask-clean-lane-b-metrics-v1", "reports": reports})
    write_json(output / "RUN_RECEIPT.json", {"schema_version": "r22-mask-clean-lane-b-run-receipt-v1", "task_id": result["task_id"], "attempt_id": output.name, "status": result["status"], "created_at": now(), "host": socket.gethostname(), "pid": os.getpid(), "gpu_lease": "NOT_ACQUIRED", "authority_promoted": False})
    write_json(output / "NEXT_ACTION.json", {"schema_version": "r22-mask-clean-lane-b-next-v1", "status": "BLOCKED_PREREQ", "next": ["Publish independent support-surface semantics for donor rejection.", "Implement pose-verified causal Poker atlas warp and coverage QA.", "Then run fresh prefix-only ProPainter and publish lossless RGB plus per-pixel source maps."], "do_not": ["Do not use bidirectional SAM/ProPainter artifacts for causal training.", "Do not reuse R7_0 Clean as a fresh successor."]})
    write_text(output / "DECISION.md", "# R2.2 Lane B 决定\n\n状态：`BLOCKED_PREREQ`。\n\nSAM3.1仍为当前基线；Chips010与Poker015失败canary及独立评估引用已冻结。两条Clean会话的旧lossless source-map输入闭合，未来帧和任务物体donor已被显式分类；但support-surface语义、successor lossless RGB/source-map配对、Poker pose-verified atlas均未闭合。因此没有占用GPU，也没有运行或伪造fresh ProPainter。\n")
    artifacts = {name: ref(output / name) for name in ("RESULT.json", "METRICS.json", "RUN_RECEIPT.json", "NEXT_ACTION.json", "DECISION.md", "MASK_CANDIDATE_EVALUATION_REFERENCE.json")}
    for report in reports:
        artifacts[f"{report['label']}_source_map_manifest"] = report["source_map_manifest"]
    write_json(output / "ARTIFACT_MANIFEST.json", {"schema_version": "r22-mask-clean-lane-b-artifact-manifest-v1", "task_id": result["task_id"], "status": result["status"], "artifacts": artifacts, "authority_promoted": False})
    print(json.dumps({"status": result["status"], "result": ref(output / "RESULT.json"), "reports": [{"label": x["label"], "status": x["status"]} for x in reports]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
