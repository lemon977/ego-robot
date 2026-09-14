#!/usr/bin/env python3
"""Build non-final, machine-readable exact78 V5.2 current-state matrices.

The outputs are explicitly CURRENT_DRAFT.  They never substitute for the final
156-row Robot terminal authority required by the execution plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/mnt/workspace/code/chaoyang")
BASE = REPO / "tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
ROLE_INDEX = REPO / "tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/MASK_ROLE_SUCCESSOR_V3_TERMINAL_INDEX.json"
COHORT = REPO / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json"
WAVE0 = REPO / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
CLEAN_ROOT = REPO / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
ROBOT_ROOT = REPO / "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1"
ROBOT_TERMINAL_ROOT = REPO / "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_terminals_v1"
OUT_ROOT = REPO / "tasks/control/runs/20260913_exact78_v52/matrices"

FROZEN_COHORT_SHA = "b6dcfec1fb41b9ffb0b9b19cc62d029f5cedf57f41552a611272cc9e5c6e43a8"
FROZEN_WAVE0_SHA = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"
FROZEN_ROLE_V3_SHA = "8aca6a99969f83851c57cf0b220a07c16f9707b79546763aa5b4112f180ba359"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def exact_ref(item: dict[str, Any], label: str) -> Path:
    p = Path(item["path"])
    if not p.is_file() or p.is_symlink() or p.stat().st_size != int(item["bytes"]) or sha256(p) != item["sha256"]:
        raise RuntimeError(f"{label} ref mismatch: {p}")
    return p


def keyed_index(path: Path, expected: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    data = load(path)
    rows = data.get("terminals", [])
    if len(rows) != expected:
        raise RuntimeError(f"{path}: expected {expected} rows, got {len(rows)}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row["session_id"]
        if sid in result:
            raise RuntimeError(f"duplicate {sid} in {path}")
        exact_ref(row["result"], f"{path.name}:{sid}")
        result[sid] = row
    return result, data


def clean_state(row: dict[str, Any], wave: dict[str, Any]) -> tuple[str, dict[str, Any] | None, list[str]]:
    sid = row["session_id"]
    existing = wave.get("existing_clean")
    candidates: list[tuple[Path, dict[str, Any] | None]] = []
    if isinstance(existing, dict):
        candidates.append((Path(existing["path"]), existing))
    candidates.append((CLEAN_ROOT / "propainter_v1" / sid / "RESULT.json", None))
    for p, pin in candidates:
        if not p.is_file():
            continue
        if pin is not None:
            exact_ref(pin, f"Clean:{sid}")
        d = load(p)
        if d.get("grade") == "B" and d.get("downstream_authorized") is True and int(d.get("frame_count", -1)) == int(row["frame_count"]):
            return "PASSED_GRADE_B", ref(p), []
    terminal = CLEAN_ROOT / "clean_terminals" / sid / "RESULT.json"
    if terminal.is_file():
        d = load(terminal)
        return str(d.get("status", "FAILED_RUNTIME_FINAL")), ref(terminal), list(d.get("reason_codes", []))
    attempts = CLEAN_ROOT / "orchestration_attempts" / sid
    if attempts.is_dir():
        return "IN_PROGRESS", None, []
    return "PENDING_WAVE0", None, []


def robot_candidate(sid: str) -> tuple[str | None, dict[str, Any] | None]:
    p = ROBOT_ROOT / sid / "RESULT.json"
    if not p.is_file():
        return None, None
    d = load(p)
    if d.get("terminal_mode") != "POSE_ONLY_VISUAL_ROBOT" or d.get("status") != "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY":
        raise RuntimeError(f"unexpected Robot candidate state: {p}")
    forbidden = ("authority", "metric_object_geometry", "contact_frame_valid", "control_ground_truth", "action_sidecar_published")
    if any(d.get(k) is not False for k in forbidden) or d.get("contact_state") != "UNKNOWN":
        raise RuntimeError(f"Robot claim boundary violation: {p}")
    return "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY", ref(p)


def robot_terminal(task: str, sid: str, frame_count: int) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    p = ROBOT_TERMINAL_ROOT / task / sid / "RESULT.json"
    if not p.is_file():
        return None, None, []
    d = load(p)
    allowed = {
        "METRIC_CONTACT_ROBOT", "POSE_ONLY_VISUAL_ROBOT", "BLOCKED_PREREQ",
        "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL",
    }
    mode = d.get("terminal_mode")
    if d.get("terminal") is not True or mode not in allowed or d.get("status") != mode:
        raise RuntimeError(f"unexpected Robot terminal state: {p}")
    if (d.get("task"), d.get("session"), int(d.get("frame_count", -1))) != (task, sid, frame_count):
        raise RuntimeError(f"Robot terminal identity mismatch: {p}")
    if d.get("control_ground_truth") is not False or d.get("action_sidecar_published") is not False:
        raise RuntimeError(f"Robot terminal control-truth boundary violation: {p}")
    if mode in {"BLOCKED_PREREQ", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL"}:
        if d.get("authority") is not False or d.get("downstream_authorized") is not False:
            raise RuntimeError(f"non-pass Robot terminal authority violation: {p}")
    for evidence in d.get("evidence", {}).values():
        if isinstance(evidence, dict) and {"path", "bytes", "sha256"} <= evidence.keys():
            exact_ref(evidence, f"Robot terminal evidence:{sid}")
    return str(mode), ref(p), list(d.get("reason_codes", []))


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    if sha256(COHORT) != FROZEN_COHORT_SHA or sha256(WAVE0) != FROZEN_WAVE0_SHA:
        raise RuntimeError("frozen cohort/Wave0 SHA mismatch")
    if sha256(ROLE_INDEX) != FROZEN_ROLE_V3_SHA:
        raise RuntimeError("frozen Role Successor V3 index SHA mismatch")
    cohort = load(COHORT)
    wave = load(WAVE0)
    cohort_rows = cohort["sessions"]
    if len(cohort_rows) != 156 or len({x["session_id"] for x in cohort_rows}) != 156:
        raise RuntimeError("exact78 cohort is not 156 unique sessions")
    wave_by_id = {x["session_id"]: x for x in wave["sessions"]}
    if len(wave_by_id) != 58:
        raise RuntimeError("Wave0 is not 58 unique sessions")

    hawor, _ = keyed_index(BASE / "HAWOR_TERMINAL_INDEX.json", 156)
    role, _ = keyed_index(ROLE_INDEX, 156)
    obj, _ = keyed_index(BASE / "MASK_TASK_OBJECT_TERMINAL_INDEX.json", 156)
    ids = {r["session_id"] for r in cohort_rows}
    if ids != set(hawor) or ids != set(role) or ids != set(obj):
        raise RuntimeError("stage index session sets differ from frozen cohort")

    rows = []
    for source in sorted(cohort_rows, key=lambda x: x["position"]):
        sid = source["session_id"]
        hs, rs, os_ = hawor[sid], role[sid], obj[sid]
        is_wave0 = sid in wave_by_id
        reasons: list[str] = []
        if not hs.get("downstream_authorized"):
            reasons.append("HAWOR_GRADE_C")
        if not rs.get("downstream_authorized"):
            reasons.append("ROLE_MASK_GRADE_C")
        if not os_.get("downstream_authorized"):
            reasons.append("OBJECT_MASK_GRADE_C")
        three_upstream_ab = not reasons
        if not is_wave0 and not reasons:
            reasons.append("MISSING_OR_INELIGIBLE_METRIC_CALIBRATION")

        c_state: str
        c_ref: dict[str, Any] | None
        c_reasons: list[str]
        if is_wave0:
            c_state, c_ref, c_reasons = clean_state(source, wave_by_id[sid])
        else:
            c_state, c_ref, c_reasons = "NOT_ELIGIBLE_WAVE0", None, []
        terminal_state, terminal_ref, terminal_reasons = robot_terminal(source["task"], sid, int(source["frame_count"]))
        candidate_state, candidate_ref = robot_candidate(sid)
        if terminal_state:
            current_robot_state = terminal_state
            candidate_ref = terminal_ref
        elif candidate_state:
            current_robot_state = candidate_state
        elif reasons:
            current_robot_state = "BLOCKED_PREREQ_CURRENT_DRAFT"
        elif c_state != "PASSED_GRADE_B":
            current_robot_state = "BLOCKED_PREREQ_CLEAN_CURRENT_DRAFT"
        else:
            current_robot_state = "READY_FOR_ROBOT_CURRENT_DRAFT"

        rows.append({
            "position": source["position"],
            "task": source["task"],
            "session_id": sid,
            "frame_count": source["frame_count"],
            "split": source.get("split"),
            "hawor": {"grade": hs.get("grade"), "status": hs["status"], "result": hs["result"]},
            "role_mask": {"grade": rs.get("grade"), "status": rs["status"], "result": rs["result"]},
            "object_mask": {"grade": os_.get("grade"), "status": os_["status"], "result": os_["result"]},
            "three_upstream_ab": three_upstream_ab,
            "metric_ready_wave0": is_wave0,
            "clean_state": c_state,
            "clean_result": c_ref,
            "robot_current_state": current_robot_state,
            "robot_candidate_result": candidate_ref,
            "reason_codes": reasons + c_reasons + terminal_reasons,
            "final_robot_terminal_proven": terminal_state is not None,
        })

    counts = {
        "sessions": len(rows),
        "tasks": dict(sorted(Counter(r["task"] for r in rows).items())),
        "hawor_grade": dict(sorted(Counter(r["hawor"]["grade"] for r in rows).items())),
        "role_mask_grade": dict(sorted(Counter(r["role_mask"]["grade"] for r in rows).items())),
        "object_mask_grade": dict(sorted(Counter(r["object_mask"]["grade"] for r in rows).items())),
        "three_upstream_ab": sum(r["three_upstream_ab"] for r in rows),
        "visual_tier_without_metric_calibration": sum(r["three_upstream_ab"] and not r["metric_ready_wave0"] for r in rows),
        "metric_ready_wave0": sum(r["metric_ready_wave0"] for r in rows),
        "clean_state": dict(sorted(Counter(r["clean_state"] for r in rows).items())),
        "robot_current_state": dict(sorted(Counter(r["robot_current_state"] for r in rows).items())),
        "final_robot_terminals_proven": sum(r["final_robot_terminal_proven"] for r in rows),
    }
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    matrix = {
        "schema_version": "exact78-v52-current-draft-cross-stage-matrix-v1",
        "created_at": now,
        "status": "CURRENT_DRAFT_NOT_FINAL_AUTHORITY",
        "authority": False,
        "sources": {
            "cohort": ref(COHORT), "wave0": ref(WAVE0),
            "hawor_index": ref(BASE / "HAWOR_TERMINAL_INDEX.json"),
            "role_index": ref(ROLE_INDEX),
            "object_index": ref(BASE / "MASK_TASK_OBJECT_TERMINAL_INDEX.json"),
        },
        "counts": counts,
        "rows": rows,
        "snapshot_policy": "CONTENT_ADDRESSED_IMMUTABLE_COPY",
        "claim_limit": "Live current-state draft only; not EXACT78_TERMINAL_COMPLETE and not final Robot authority.",
    }
    out = args.output_root.resolve()
    json_path = out / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
    csv_path = out / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.csv"
    json_payload = (json.dumps(matrix, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    atomic_write(json_path, json_payload)
    csv_fields = [
        "position", "task", "session_id", "frame_count", "split", "hawor_grade", "role_mask_grade",
        "object_mask_grade", "three_upstream_ab", "metric_ready_wave0", "clean_state", "robot_current_state", "reason_codes",
        "final_robot_terminal_proven",
    ]
    fd, tmp = tempfile.mkstemp(prefix=f".{csv_path.name}.", suffix=".tmp", dir=out)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "position": r["position"], "task": r["task"], "session_id": r["session_id"],
                    "frame_count": r["frame_count"], "split": r["split"], "hawor_grade": r["hawor"]["grade"],
                    "role_mask_grade": r["role_mask"]["grade"], "object_mask_grade": r["object_mask"]["grade"],
                    "three_upstream_ab": r["three_upstream_ab"], "metric_ready_wave0": r["metric_ready_wave0"], "clean_state": r["clean_state"],
                    "robot_current_state": r["robot_current_state"], "reason_codes": "|".join(r["reason_codes"]),
                    "final_robot_terminal_proven": r["final_robot_terminal_proven"],
                })
        os.replace(tmp, csv_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    json_ref = ref(json_path)
    csv_ref = ref(csv_path)
    snapshot_root = out / "snapshots"
    json_snapshot = snapshot_root / f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{json_ref['sha256']}.json"
    csv_snapshot = snapshot_root / f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{csv_ref['sha256']}.csv"
    for snapshot, payload in ((json_snapshot, json_payload), (csv_snapshot, csv_path.read_bytes())):
        if snapshot.exists():
            if snapshot.read_bytes() != payload:
                raise RuntimeError(f"content-addressed matrix snapshot collision: {snapshot}")
        else:
            atomic_write(snapshot, payload)
    print(json.dumps({
        "status": matrix["status"], "counts": counts,
        "json": json_ref, "csv": csv_ref,
        "immutable_json_snapshot": ref(json_snapshot),
        "immutable_csv_snapshot": ref(csv_snapshot),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
