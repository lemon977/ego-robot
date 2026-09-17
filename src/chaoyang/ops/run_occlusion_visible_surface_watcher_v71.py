#!/usr/bin/env python3
"""Build bounded visible-surface occlusion canaries during Robot expansion.

The watcher consumes each immutable hard/soft Robot candidate as soon as it is
published and only waits for the upstream terminal marker before closing its
own index.  A sparse uniform
Robot z-buffer identifies the strongest observed Robot/object overlap; a local
24-frame window is then exported and audited.  Outputs remain development
evidence: hidden object appearance, Gold accuracy and authority are out of
scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def exact(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference["path"])).resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256(path) != reference["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_bounded(command: list[str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            handle.write(f"\nTIMEOUT_SECONDS={timeout}\n")
            return 124


def selection_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load(path).get("sessions")
    if not isinstance(rows, list):
        raise RuntimeError("Wave0 selection sessions missing")
    indexed = {str(row["session_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError("duplicate session in Wave0 selection")
    return indexed


def resolve_inputs(candidate_result: Path, row: dict[str, Any]) -> dict[str, Path]:
    candidate = load(candidate_result)
    lineage_path = candidate_result.parent / "LINEAGE.json"
    lineage = load(lineage_path)
    manifest = lineage["input_manifest"]
    object_result = load(exact(row["upstream"]["task_object_mask"], "task object result"))
    return {
        "hawor": exact(manifest["hawor_chain"]["temporal_output"], "temporal HaWoR"),
        "arm": exact(manifest["legacy_arm_states"], "arm states"),
        "hand": exact(manifest["legacy_hand_states"], "hand states"),
        "source": exact(manifest["source_video"], "source video"),
        "depth_result": exact(row["upstream"]["depth"], "depth result"),
        "object_manifest": exact(object_result["artifacts"]["manifest"], "object mask manifest"),
        "candidate_result": candidate_result.resolve(strict=True),
        "lineage": lineage_path.resolve(strict=True),
    }


def object_union(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.bool_)
    for instance in row["physical_instances"].values():
        if not instance.get("valid"):
            continue
        mask_path = exact(instance["mask"], "physical object mask")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot decode mask: {mask_path}")
        union |= cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return union


def select_focus_frames(zbuffer_result: Path, object_manifest: Path, arm_states: Path, count: int) -> tuple[list[int], dict[str, Any]]:
    z_result = load(zbuffer_result)
    archive_path = exact(z_result["outputs"][0], "sparse Robot z-buffer")
    manifest = load(object_manifest)
    rows = {int(row["source_frame"]): row for row in manifest["frames"]}
    with np.load(archive_path, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        labels = np.asarray(archive["render_label"], dtype=np.int32)
    overlaps: list[int] = []
    for slot, frame in enumerate(frame_ids):
        robot = labels[slot] >= 0
        mask = object_union(rows[int(frame)], robot.shape)
        overlaps.append(int(np.count_nonzero(robot & mask)))
    best_slot = int(np.argmax(overlaps))
    best_frame = int(frame_ids[best_slot])
    with np.load(arm_states, allow_pickle=False) as archive:
        valid = np.asarray(archive["valid_side_frame"], dtype=np.bool_)
    eligible = np.flatnonzero(np.all(valid, axis=0))
    if len(eligible) < count:
        raise RuntimeError(f"only {len(eligible)} bilateral-valid frames for focus window")
    ranked = sorted(eligible.tolist(), key=lambda frame: (abs(frame - best_frame), frame))[:count]
    focus = sorted(int(frame) for frame in ranked)
    return focus, {
        "sparse_frame_ids": frame_ids.tolist(),
        "sparse_overlap_pixels": overlaps,
        "best_sparse_frame": best_frame,
        "best_sparse_overlap_pixels": overlaps[best_slot],
        "focus_frame_ids": focus,
    }


def build_one(
    candidate_result: Path,
    row: dict[str, Any],
    output_root: Path,
    sparse_count: int,
    focus_count: int,
    timeout: int,
) -> dict[str, Any]:
    session = str(load(candidate_result)["session"])
    destination = output_root / "sessions" / session
    final_result = destination / "RESULT.json"
    if final_result.is_file():
        return load(final_result)
    if destination.exists():
        return {"session": session, "status": "FAILED_RUNTIME_FINAL", "reason": "NONTERMINAL_OUTPUT_EXISTS"}
    destination.mkdir(parents=True)
    inputs = resolve_inputs(candidate_result, row)
    sparse_dir = destination / "sparse_zbuffer"
    sparse_log = destination / "sparse_zbuffer.log"
    code = run_bounded([
        sys.executable,
        "src/chaoyang/ops/export_robot_unified_zbuffer_canary_v71.py",
        "--session-id", session,
        "--hawor", str(inputs["hawor"]),
        "--arm-states", str(inputs["arm"]),
        "--hand-states", str(inputs["hand"]),
        "--frame-count", str(sparse_count),
        "--output-dir", str(sparse_dir),
    ], sparse_log, timeout)
    sparse_result = sparse_dir / "RESULT.json"
    if code != 0 or not sparse_result.is_file():
        result = {"session": session, "status": "FAILED_RUNTIME_FINAL", "phase": "SPARSE_ZBUFFER", "return_code": code}
        atomic_json(final_result, result)
        return result
    focus, selection = select_focus_frames(sparse_result, inputs["object_manifest"], inputs["arm"], focus_count)
    atomic_json(destination / "FOCUS_SELECTION.json", selection)
    if selection["best_sparse_overlap_pixels"] <= 0:
        result = {
            "session": session,
            "status": "FAILED_QUALITY_C",
            "reason": "NO_OBSERVED_ROBOT_OBJECT_OVERLAP_IN_SPARSE_SCAN",
            "selection": selection,
            "claim_limit": "Sparse visible-surface diagnostic only; no contact/occlusion authority.",
        }
        atomic_json(final_result, result)
        return result
    focus_zbuffer = destination / "focus_zbuffer"
    focus_log = destination / "focus_zbuffer.log"
    code = run_bounded([
        sys.executable,
        "src/chaoyang/ops/export_robot_unified_zbuffer_canary_v71.py",
        "--session-id", session,
        "--hawor", str(inputs["hawor"]),
        "--arm-states", str(inputs["arm"]),
        "--hand-states", str(inputs["hand"]),
        "--frame-ids", ",".join(map(str, focus)),
        "--output-dir", str(focus_zbuffer),
    ], focus_log, timeout)
    focus_zbuffer_result = focus_zbuffer / "RESULT.json"
    if code != 0 or not focus_zbuffer_result.is_file():
        result = {"session": session, "status": "FAILED_RUNTIME_FINAL", "phase": "FOCUS_ZBUFFER", "return_code": code}
        atomic_json(final_result, result)
        return result
    ordering = destination / "focus_visible_surface_ordering"
    ordering_log = destination / "focus_visible_surface_ordering.log"
    code = run_bounded([
        sys.executable,
        "src/chaoyang/ops/build_visible_surface_occlusion_canary_v71.py",
        "--session-id", session,
        "--robot-zbuffer-result", str(focus_zbuffer_result),
        "--depth-result", str(inputs["depth_result"]),
        "--object-mask-manifest", str(inputs["object_manifest"]),
        "--source-video", str(inputs["source"]),
        "--output-dir", str(ordering),
    ], ordering_log, timeout)
    ordering_result = ordering / "RESULT.json"
    if code != 0 or not ordering_result.is_file():
        result = {"session": session, "status": "FAILED_RUNTIME_FINAL", "phase": "VISIBLE_SURFACE_ORDERING", "return_code": code}
        atomic_json(final_result, result)
        return result
    metrics = load(ordering_result)["metrics"]
    result = {
        "schema_version": "visible-surface-occlusion-session-result-v71-v1",
        "session": session,
        "task": row["task"],
        "status": "PASSED_DEVELOPMENT_VISIBLE_SURFACE_CANARY",
        "selection": selection,
        "metrics": metrics,
        "gates": {
            "known_coverage_70": metrics["known_decision_coverage"] >= 0.70,
            "unknown_ratio_30": metrics["unknown_pixel_ratio"] <= 0.30,
            "conditional_retention_99": metrics["protected_retention_pixel_weighted"] >= 0.99,
        },
        "inputs": [ref(path) for path in inputs.values()],
        "outputs": [ref(sparse_result), ref(destination / "FOCUS_SELECTION.json"), ref(focus_zbuffer_result), ref(ordering_result)],
        "authority": False,
        "accuracy_reported": False,
        "claim_limit": "Observed visible-surface Robot/object optical-Z development canary only; no hidden appearance, Gold accuracy, Contact/Robot control, Silver/Gold or physical authority.",
    }
    atomic_json(final_result, result)
    return result


def build_silver_report(terminals: dict[str, dict[str, Any]], created_at: str) -> dict[str, Any]:
    rows = [terminals[key] for key in sorted(terminals)]
    numeric_passed = sum(
        row.get("status") == "PASSED_DEVELOPMENT_VISIBLE_SURFACE_CANARY"
        and all(row.get("gates", {}).values())
        for row in rows
    )
    return {
        "schema_version": "occlusion-silver-report-v71-v1",
        "status": "FAILED_QUALITY_C",
        "reason": "SILVER_CONTRACT_NOT_CLOSED",
        "created_at": created_at,
        "counts": {
            "terminal_candidates": len(rows),
            "visible_surface_numeric_gate_passed": numeric_passed,
        },
        "evaluated_gates": [
            "known_decision_coverage_70",
            "unknown_pixel_ratio_30",
            "conditional_object_retention_99",
            "robot_object_visible_surface_zbuffer",
        ],
        "required_but_unclosed_gates": [
            "full_session_temporal_consistency",
            "authorized_band_outside_byte_exact",
            "bidirectional_closure_residual",
            "legal_hidden_object_appearance_source",
        ],
        "rows": rows,
        "silver_authority": False,
        "accuracy_reported": False,
        "authority": False,
        "claim_limit": "Terminal visible-surface development coverage only; full Silver quality contract is not closed and no Gold accuracy, Contact, Robot control or physical authority is granted.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-state", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sparse-count", type=int, default=64)
    parser.add_argument("--focus-count", type=int, default=24)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    parser.add_argument("--phase-timeout-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.sparse_count < args.focus_count or args.focus_count < 1:
        raise SystemExit("require sparse-count >= focus-count >= 1")
    candidate_root = args.candidate_root.resolve()
    candidate_state = args.candidate_state.resolve()
    selection_path = args.selection.resolve(strict=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = selection_rows(selection_path)
    state_path = output / "AUTOMATION_STATE.json"
    started = time.monotonic()
    terminals: dict[str, dict[str, Any]] = {}
    while True:
        upstream = load(candidate_state) if candidate_state.is_file() else {}
        candidates = sorted(candidate_root.glob("*/RESULT.json"))
        for candidate in candidates:
            session = candidate.parent.name
            if session in terminals:
                continue
            if session not in rows:
                terminals[session] = {"session": session, "status": "BLOCKED_PREREQ_NOT_IN_WAVE0"}
                continue
            atomic_json(state_path, {
                "schema_version": "visible-surface-occlusion-watcher-v71-v2",
                "status": "RUNNING_INCREMENTAL",
                "phase": "PROCESS_SESSION",
                "current_session": session,
                "pid": os.getpid(),
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "upstream_status": upstream.get("status", "MISSING"),
                "terminal_count": len(terminals),
                "claim_limit": "Automated development canaries only; no authority.",
            })
            terminals[session] = build_one(
                candidate, rows[session], output,
                args.sparse_count, args.focus_count, args.phase_timeout_seconds,
            )
            atomic_json(output / "INCREMENTAL_RESULT.json", {
                "schema_version": "visible-surface-occlusion-incremental-v71-v1",
                "rows": [terminals[key] for key in sorted(terminals)],
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "claim_limit": "Development canary receipts only; no authority.",
            })
        candidate_count = len(candidates)
        terminal = upstream.get("status") == "TERMINAL" and len(terminals) == candidate_count
        state = {
            "schema_version": "visible-surface-occlusion-watcher-v71-v2",
            "status": "TERMINAL" if terminal else ("RUNNING_INCREMENTAL" if candidate_count else "WAITING_UPSTREAM"),
            "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "upstream_status": upstream.get("status", "MISSING"),
            "candidate_count": candidate_count,
            "terminal_count": len(terminals),
            "passed_count": sum(row.get("status") == "PASSED_DEVELOPMENT_VISIBLE_SURFACE_CANARY" for row in terminals.values()),
            "claim_limit": "Automated development canaries only; no authority.",
        }
        atomic_json(state_path, state)
        if terminal or args.once:
            terminal_index = {
                "schema_version": "visible-surface-occlusion-index-v71-v1",
                "status": "PASSED_DEVELOPMENT_TERMINAL_COVERAGE" if terminal else "PASSED_INCREMENTAL_SNAPSHOT",
                "created_at": state["updated_at"],
                "selection": ref(selection_path),
                "candidate_state": ref(candidate_state) if candidate_state.is_file() else None,
                "counts": {"candidate": candidate_count, "terminal": len(terminals), "passed": state["passed_count"]},
                "rows": [terminals[key] for key in sorted(terminals)],
                "authority": False,
                "accuracy_reported": False,
                "claim_limit": state["claim_limit"],
            }
            atomic_json(output / "RESULT.json", terminal_index)
            if terminal:
                atomic_json(
                    output / "OCCLUSION_SILVER_REPORT.json",
                    build_silver_report(terminals, state["updated_at"]),
                )
            return 0
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="WAIT_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
