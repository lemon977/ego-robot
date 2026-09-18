#!/usr/bin/env python3
"""Audit frozen Human/Stereo alignment for hidden optimizer-bound saturation.

This successor is intentionally read-only with respect to all model evidence.  It
recomputes the bounded and unconstrained affine fits from the immutable non-contact
MANO/Stereo surface rows, revokes metric translation authority when the optimum is
outside the frozen scale interval, and carries the prior 5 mm finite-patch Contact
result forward without recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

PROJECT = Path(__file__).resolve().parents[3]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet


TASK_ID = "0915_human_stereo_alignment_bound_audit_v1"
PHASE = "0915_HUMAN_STEREO_ALIGNMENT_BOUND_AUDIT_V1"
SESSION_ID = "play_cards_0915_001"
SCALE_BOUNDS = (0.8, 1.2)
OFFSET_BOUNDS_M = (-0.15, 0.15)
CONTACT_DISTANCE_M = 0.005
INPUT_ROOT = PROJECT / (
    "_run/current/0915_human_stereo_surface_association_canary_v1/"
    "attempts/attempt_0001"
)
OUTPUT = PROJECT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = PROJECT / "docs/current/visuals/0915_HUMAN_STEREO_ALIGNMENT_BOUND_AUDIT_V1"
TERMINAL_RECEIPT = PROJECT / "tasks/receipts/0915_HUMAN_STEREO_ALIGNMENT_BOUND_AUDIT_V1_RESULT.json"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = PROJECT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current task packet differs from frozen specification")
    state = load_json(PROJECT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not current next_task")
    index = load_json(PROJECT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize this task")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA differs from current route")
    return packet, packet_path


def heartbeat() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", TASK_ID,
         "--pid", str(os.getpid()), "--status", "RUNNING", "--phase", PHASE],
        cwd=PROJECT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def _admitted_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row.get("whole_frame_hand_contact_excluded") is False
        and int(row.get("support_count", 0)) >= 20
        and np.isfinite([
            row.get("mano_surface_depth_m"), row.get("stereo_surface_depth_m")
        ]).all()
    ]


def _robust_fit(
    rows: Sequence[Mapping[str, Any]], *, bounded: bool,
) -> dict[str, Any]:
    x = np.asarray([row["mano_surface_depth_m"] for row in rows], np.float64)
    y = np.asarray([row["stereo_surface_depth_m"] for row in rows], np.float64)
    if x.size < 30:
        raise ValueError("at least 30 frozen training rows are required")
    keep = np.ones(x.size, bool)
    scale, offset = 1.0, 0.0
    iterations: list[dict[str, Any]] = []
    for index in range(4):
        design = np.column_stack((x[keep], np.ones(int(np.count_nonzero(keep)))))
        raw_estimate, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        raw_scale, raw_offset = map(float, raw_estimate)
        scale = float(np.clip(raw_scale, *SCALE_BOUNDS)) if bounded else raw_scale
        offset = float(np.clip(raw_offset, *OFFSET_BOUNDS_M)) if bounded else raw_offset
        residual = y - (scale * x + offset)
        robust = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        next_keep = np.abs(residual - np.median(residual)) <= max(0.010, 3.0 * robust)
        iterations.append({
            "iteration": index,
            "raw_scale": raw_scale,
            "raw_offset_m": raw_offset,
            "effective_scale": scale,
            "effective_offset_m": offset,
            "input_rows": int(np.count_nonzero(keep)),
            "next_rows": int(np.count_nonzero(next_keep)),
        })
        if np.array_equal(next_keep, keep) or int(np.count_nonzero(next_keep)) < 30:
            break
        keep = next_keep
    return {
        "scale": scale,
        "offset_m": offset,
        "retained_train_rows": int(np.count_nonzero(keep)),
        "iterations": iterations,
    }


def _heldout_metrics(rows: Sequence[Mapping[str, Any]], fit: Mapping[str, Any]) -> dict[str, Any]:
    x = np.asarray([row["mano_surface_depth_m"] for row in rows], np.float64)
    y = np.asarray([row["stereo_surface_depth_m"] for row in rows], np.float64)
    residual = np.abs(y - (float(fit["scale"]) * x + float(fit["offset_m"])))
    return {
        "median_abs_residual_m": float(np.median(residual)),
        "p90_abs_residual_m": float(np.percentile(residual, 90)),
        "max_abs_residual_m": float(np.max(residual)),
        "frame_ids": sorted({int(row["frame_id"]) for row in rows}),
        "row_ids_sha256": canonical_sha([
            [int(row["frame_id"]), str(row["hand_id"])] for row in rows
        ]),
    }


def audit_alignment_bounds(
    rows: Sequence[Mapping[str, Any]], prior_alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute both fits and reject an apparently passing boundary solution."""

    admitted = _admitted_rows(rows)
    train = [row for row in admitted if int(row["frame_id"]) % 5 != 0]
    holdout = [row for row in admitted if int(row["frame_id"]) % 5 == 0]
    if len(train) < 30 or len(holdout) < 8:
        return {
            "schema_version": "ALIGNMENT_BOUND_SATURATION_AUDIT_V1",
            "status": "BLOCKED_INSUFFICIENT_FROZEN_ROWS",
            "metric_translation_authorized": False,
            "train_row_count": len(train),
            "holdout_row_count": len(holdout),
        }
    bounded = _robust_fit(train, bounded=True)
    unconstrained = _robust_fit(train, bounded=False)
    bounded_heldout = _heldout_metrics(holdout, bounded)
    unconstrained_heldout = _heldout_metrics(holdout, unconstrained)
    prior_fit = prior_alignment.get("fit", {})
    prior_heldout = prior_alignment.get("heldout", {})
    reproduced = bool(
        abs(float(prior_fit.get("scale", np.nan)) - float(bounded["scale"])) <= 1e-12
        and abs(float(prior_fit.get("offset_m", np.nan)) - float(bounded["offset_m"])) <= 1e-12
        and abs(float(prior_heldout.get("p90_abs_residual_m", np.nan))
                - float(bounded_heldout["p90_abs_residual_m"])) <= 1e-12
        and prior_heldout.get("row_ids_sha256") == bounded_heldout["row_ids_sha256"]
    )
    unconstrained_scale = float(unconstrained["scale"])
    outside = not (SCALE_BOUNDS[0] <= unconstrained_scale <= SCALE_BOUNDS[1])
    at_bound = bool(
        abs(float(bounded["scale"]) - SCALE_BOUNDS[0]) <= 1e-12
        or abs(float(bounded["scale"]) - SCALE_BOUNDS[1]) <= 1e-12
    )
    saturated = bool(outside and at_bound)
    residual_gate_passed = bool(
        bounded_heldout["median_abs_residual_m"] <= 0.015
        and bounded_heldout["p90_abs_residual_m"] <= 0.030
    )
    passed = bool(reproduced and residual_gate_passed and not saturated)
    return {
        "schema_version": "ALIGNMENT_BOUND_SATURATION_AUDIT_V1",
        "session_id": SESSION_ID,
        "status": (
            "PASS_DEVELOPMENT_ALIGNMENT" if passed
            else "REJECTED_BOUNDED_FIT_SATURATION" if saturated
            else "REJECTED_ALIGNMENT_AUDIT"
        ),
        "metric_translation_authorized": passed,
        "frozen_scale_bounds": list(SCALE_BOUNDS),
        "frozen_offset_bounds_m": list(OFFSET_BOUNDS_M),
        "train_row_count": len(train),
        "holdout_row_count": len(holdout),
        "bounded_fit": bounded,
        "unconstrained_fit": unconstrained,
        "bounded_holdout": bounded_heldout,
        "unconstrained_holdout": unconstrained_heldout,
        "prior_bounded_result_reproduced": reproduced,
        "bounded_residual_gate_passed": residual_gate_passed,
        "bounded_scale_at_frozen_edge": at_bound,
        "unconstrained_scale_outside_frozen_interval": outside,
        "bound_saturation_detected": saturated,
        "contact_or_object_fit_used": False,
        "fixed_48mm_bias_used": False,
        "model_rerun_performed": False,
        "decision_rule": (
            "metric translation requires an unconstrained optimum inside the frozen "
            "scale interval; a clipped holdout pass cannot grant authority"
        ),
    }


def render_diagnostic(audit: Mapping[str, Any], contact: Mapping[str, Any], path: Path) -> None:
    canvas = np.full((720, 1280, 3), 246, np.uint8)
    cv2.putText(canvas, "0915 Human/Stereo alignment bound audit", (42, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.12, (25, 25, 25), 2, cv2.LINE_AA)
    bounded = audit["bounded_fit"]
    unconstrained = audit["unconstrained_fit"]
    lines = [
        "Immutable evidence only; no model rerun / no Contact fit / no 48 mm bias",
        f"Frozen scale interval: [{SCALE_BOUNDS[0]:.3f}, {SCALE_BOUNDS[1]:.3f}]",
        f"Bounded fit scale: {float(bounded['scale']):.6f}",
        f"Unconstrained fit scale: {float(unconstrained['scale']):.6f}",
        f"Bound saturation: {audit['bound_saturation_detected']}",
        f"Bounded holdout P90: {1000.0 * float(audit['bounded_holdout']['p90_abs_residual_m']):.2f} mm",
        f"Metric translation authorized: {audit['metric_translation_authorized']}",
        f"Closest finite-patch distance (unchanged): {1000.0 * float(contact['closest_finite_patch_distance_m']):.2f} mm",
        f"5 mm Contact windows (unchanged): {contact['admitted_window_count']}",
        "R0 retained; R1 blocked; q22/wrist unmodified",
    ]
    y = 125
    for line in lines:
        cv2.putText(canvas, line, (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (50, 50, 50), 2, cv2.LINE_AA)
        y += 50
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write diagnostic: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    packet, packet_path = validate_route()
    output, visual, receipt = args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve()
    if output != OUTPUT.resolve() or visual != VISUAL.resolve() or receipt != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("fixed output/visual/receipt namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive executor epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)

    source_path = INPUT_ROOT / "MANO_SURFACE_ASSOCIATION_V1.json"
    alignment_path = INPUT_ROOT / "HUMAN_STEREO_ALIGNMENT_CHECK_V2.json"
    contact_path = INPUT_ROOT / "INTERACTION_CONTACT_RECHECK_V1.json"
    predecessor_result_path = INPUT_ROOT / "RESULT.json"
    input_paths = [source_path, alignment_path, contact_path, predecessor_result_path]
    input_refs = [ref(path) for path in input_paths]
    input_snapshot = {item["path"]: item["sha256"] for item in input_refs}
    surface = load_json(source_path)
    prior_alignment = load_json(alignment_path)
    contact = load_json(contact_path)
    predecessor_result = load_json(predecessor_result_path)
    if (
        surface.get("schema_version") != "MANO_SURFACE_ASSOCIATION_V1"
        or prior_alignment.get("status") != "PASS_DEVELOPMENT_ALIGNMENT"
        or contact.get("distance_gate_m") != CONTACT_DISTANCE_M
        or contact.get("distance_gate_was_uncertainty_expanded") is not False
        or contact.get("finite_visible_patch_required") is not True
        or contact.get("admitted_window_count") != 0
        or predecessor_result.get("task_id")
        != "0915_human_stereo_surface_association_canary_v1"
    ):
        raise RuntimeError("immutable predecessor evidence contract drift")

    signature_payload = {
        "schema_version": "0915-human-stereo-alignment-bound-audit-run-signature-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "gpu_used": False,
        "model_rerun_performed": False,
        "contact_policy": "CARRY_FORWARD_UNCHANGED_5MM_FINITE_VISIBLE_PATCH",
        "forbidden_inputs": ["Removal", "Clean", "archive", "object_or_contact_fit", "48mm_bias"],
        "task_packet": ref(packet_path),
        "inputs": input_refs,
        "code": [ref(Path(__file__))],
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-human-stereo-alignment-bound-audit-writer-claim-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "attempt_id": output.name,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output),
        "run_signature_sha256": signature["run_signature_sha256"],
        "task_packet": ref(packet_path),
    })
    heartbeat()

    records = surface.get("records")
    if not isinstance(records, list):
        raise RuntimeError("surface association records are absent")
    audit = audit_alignment_bounds(records, prior_alignment)
    atomic_json(output / "ALIGNMENT_BOUND_SATURATION_AUDIT_V1.json", audit)
    decision = {
        "schema_version": "0915_SURFACE_ASSOCIATION_DOWNSTREAM_DECISION_V2",
        "session_id": SESSION_ID,
        "alignment_status": audit["status"],
        "metric_wrist_object_translation_authorized": audit["metric_translation_authorized"],
        "contact_result_carried_forward_unchanged": True,
        "contact_window_count": contact["admitted_window_count"],
        "contact_distance_gate_m": contact["distance_gate_m"],
        "r0_baseline_remains_authoritative": True,
        "r1_status": "BLOCKED_LOCAL_EVIDENCE",
        "r2_status": "NOT_RUN_R1_NOT_EXECUTED",
        "first_blocker": (
            "HUMAN_STEREO_ALIGNMENT_BOUND_SATURATION"
            if audit["status"] == "REJECTED_BOUNDED_FIT_SATURATION"
            else "NO_FIXED_PAIR_FIVE_FRAME_CONTACT_WINDOW_AT_UNCHANGED_5MM_GATE"
        ),
        "blockers": [
            "HUMAN_STEREO_ALIGNMENT_BOUND_SATURATION",
            "NO_FIXED_PAIR_FIVE_FRAME_CONTACT_WINDOW_AT_UNCHANGED_5MM_GATE",
        ],
        "q22_or_wrist_modified": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
    }
    atomic_json(output / "DOWNSTREAM_DECISION_V2.json", decision)
    metrics = {
        "schema_version": "0915-human-stereo-alignment-bound-audit-metrics-v1",
        "bounded_scale": audit.get("bounded_fit", {}).get("scale"),
        "unconstrained_scale": audit.get("unconstrained_fit", {}).get("scale"),
        "bound_saturation_detected": audit.get("bound_saturation_detected"),
        "bounded_holdout_median_mm": 1000.0 * audit["bounded_holdout"]["median_abs_residual_m"],
        "bounded_holdout_p90_mm": 1000.0 * audit["bounded_holdout"]["p90_abs_residual_m"],
        "closest_finite_patch_mm": 1000.0 * contact["closest_finite_patch_distance_m"],
        "contact_window_count": contact["admitted_window_count"],
        "r1_status": decision["r1_status"],
    }
    atomic_json(output / "METRICS.json", metrics)

    visual.mkdir(parents=True)
    render_diagnostic(audit, contact, visual / "ALIGNMENT_BOUND_SATURATION_AUDIT.png")
    readme = f"""# 0915 Human/Stereo 尺度边界审计 V1

本页只审计已封存的 MANO-surface ↔ Stereo-surface 对齐，不重跑模型，不消费 Removal/Clean，
也不改变 Contact 的 `5 mm` 有限可见面门。

- 有界拟合尺度：`{float(audit['bounded_fit']['scale']):.6f}`（冻结下界 `{SCALE_BOUNDS[0]:.3f}`）
- 无约束拟合尺度：`{float(audit['unconstrained_fit']['scale']):.6f}`
- 有界 hold-out P90：`{metrics['bounded_holdout_p90_mm']:.2f} mm`
- 审计结论：`{audit['status']}`
- 公制 wrist-object 平移授权：`{audit['metric_translation_authorized']}`
- 最近有限可见牌面距离：`{metrics['closest_finite_patch_mm']:.2f} mm`
- 固定配对连续 Contact 窗口：`{contact['admitted_window_count']}`
- Robot：R0 保留；R1 为 `BLOCKED_LOCAL_EVIDENCE`；`q22/wrist` 未修改。

数值 hold-out 改善仍是有用诊断，但最优尺度落在冻结区间外，裁剪到边界后的“通过”不能升级为
公制对齐权威。Contact 结果原样继承，没有把 6.60 mm 放宽成接触，也没有使用 48 mm 统一偏置。
"""
    (visual / "README_ZH.md").write_text(readme, encoding="utf-8")

    for raw, expected in input_snapshot.items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"immutable predecessor input changed: {path}")
    result = {
        "schema_version": "0915-human-stereo-alignment-bound-audit-result-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": "PASSED",
        "diagnostic_status": audit["status"],
        "metric_translation_authorized": audit["metric_translation_authorized"],
        "contact_window_count": contact["admitted_window_count"],
        "r0_status": "COMPLETED_DEVELOPMENT_BASELINE",
        "r1_status": decision["r1_status"],
        "r2_status": decision["r2_status"],
        "q22_or_wrist_modified": False,
        "weights": "ABSENT",
        "gpu_used": False,
        "model_rerun_performed": False,
        "source_mutated": False,
        "removal_or_clean_consumed": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
        "artifacts": {
            "bound_audit": ref(output / "ALIGNMENT_BOUND_SATURATION_AUDIT_V1.json"),
            "downstream_decision": ref(output / "DOWNSTREAM_DECISION_V2.json"),
            "metrics": ref(output / "METRICS.json"),
            "visual_readme": ref(visual / "README_ZH.md"),
            "diagnostic_png": ref(visual / "ALIGNMENT_BOUND_SATURATION_AUDIT.png"),
        },
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-human-stereo-alignment-bound-audit-run-receipt-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
        "gpu_used": False,
    })
    print(json.dumps({
        "status": "PASSED",
        "diagnostic": audit["status"],
        "metric_translation_authorized": audit["metric_translation_authorized"],
        "contact_windows": contact["admitted_window_count"],
        "r1": decision["r1_status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
