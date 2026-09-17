#!/usr/bin/env python3
"""Create immutable start/terminal research indices; never publish formal current."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1"
GOV = ROOT / "docs/governance"
QUEUE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_native_b_regression_queue_v1"
LANES = {
    "A_MASK": ("lane_a_mask", ["play_cards_0901_001", "play_cards_0901_005"], "GPU_QUEUED_EXISTING", 1800),
    "B_CLEAN": ("lane_b_clean", ["play_cards_0903_245", "get_potato_chips_0902_039"], "CPU", 2700),
    "C_DEPTH": ("lane_c_depth", [], "CPU", 3600),
    "D_OBJECT_CONTACT": ("lane_d_object_contact", [], "CPU", 3600),
    "E_ROBOT30": ("lane_e_robot30", [], "CPU_INVENTORY_GPU_CONDITIONAL", 5400),
    "F_DOCS": ("lane_f_docs", [], "CPU", 1800),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def initial() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    start = RUN / "RUN_START_SNAPSHOT.json"
    matrix = RUN / "TASK_MATRIX_REV_0001.json"
    visual = RUN / "VISUAL_REVIEW_INDEX_REV_0001.json"
    for path in (start, matrix, visual):
        if path.exists():
            raise FileExistsError(path)
    receipt = GOV / "CURRENT_STATUS_RECEIPT.json"
    current = load(receipt)
    if current.get("freshness", {}).get("status") != "FRESH":
        raise RuntimeError("governance receipt not FRESH")
    git_state = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout
    q_start = QUEUE / "QUEUE_START.json"
    q_result = QUEUE / "QUEUE_RESULT.json"
    q_heartbeat = QUEUE / "QUEUE_HEARTBEAT.json"
    snapshot = {
        "schema_version": "rc1-multiline-run-start-v1",
        "created_at": now_iso(),
        "governance_revision": current["governance_revision"],
        "generation_id": current["generation_id"],
        "formal_current_modified": False,
        "refs": {
            "receipt": artifact_ref(receipt),
            "task_state": artifact_ref(GOV / "LONG_HORIZON_TASK_STATE.json"),
            "algorithm_contract": artifact_ref(GOV / "ALGORITHM_CONTRACT.json"),
            "plan": artifact_ref(RUN / "EXECUTION_PLAN_ZH.md"),
            "gpu_lease": artifact_ref(ROOT / "_run/current/GPU_LEASE.json"),
            "existing_sam_queue_start": artifact_ref(q_start),
            "existing_sam_queue_observation": artifact_ref(q_result if q_result.exists() else q_heartbeat),
            "script": artifact_ref(Path(__file__)),
        },
        "git_dirty_untracked_porcelain_sha256": sha_bytes(git_state),
        "git_dirty_untracked_record_count": len([x for x in git_state.split(b"\0") if x]),
        "claim_limit": "Point-in-time research input freeze; not a formal current state or authority promotion.",
    }
    atomic_json(start, snapshot)
    rows = []
    for task_id, (lane, sessions, resource, cap) in LANES.items():
        rows.append({
            "task_id": task_id,
            "lane": lane,
            "session_ids": sessions,
            "session_binding_status": "BOUND" if sessions else "PREFLIGHT_SELECTION_REQUIRED",
            "worker": "existing_sam_queue_observer" if task_id == "A_MASK" else "research_lane",
            "resource": resource,
            "wall_cap_s": cap,
            "max_runtime_attempts_per_signature": 2,
            "status": "EXISTING_QUEUE_RUNNING_OR_WAITING" if task_id == "A_MASK" else "PENDING",
            "output_root": str(RUN / lane),
            "allowed_claim": "RESEARCH_ONLY",
            "input_snapshot": artifact_ref(start),
        })
    atomic_json(matrix, {"schema_version": "rc1-multiline-task-matrix-v1", "created_at": now_iso(), "rows": rows})
    atomic_json(visual, {
        "schema_version": "rc1-multiline-visual-index-v1", "created_at": now_iso(),
        "videos": [], "status": "PENDING_EVIDENCE",
        "claim_limit": "The index lists verified complete videos only; missing videos remain missing.",
    })
    print(json.dumps({"run": str(RUN), "revision": snapshot["governance_revision"],
                      "matrix_rows": len(rows)}, ensure_ascii=False))


def aggregate() -> None:
    """Append a new version based only on immutable lane receipts found on disk."""
    matrix_files = sorted(RUN.glob("TASK_MATRIX_REV_*.json"))
    visual_files = sorted(RUN.glob("VISUAL_REVIEW_INDEX_REV_*.json"))
    if not matrix_files or not visual_files:
        raise RuntimeError("initial index missing")
    next_n = int(matrix_files[-1].stem.rsplit("_", 1)[1]) + 1
    rows = load(matrix_files[0])["rows"]
    videos: list[dict] = []
    for row in rows:
        lane = RUN / row["lane"]
        receipts = sorted(lane.glob("**/RESULT.json")) if lane.exists() else []
        row["results"] = [artifact_ref(path) for path in receipts]
        if receipts:
            # A lane may still be developing; do not infer algorithm PASS from mere file existence.
            last = load(receipts[-1])
            row["status"] = str(last.get("status", "RESULT_STATUS_UNSPECIFIED"))
            if last.get("session_id") and last["session_id"] not in row["session_ids"]:
                row["session_ids"] = [*row["session_ids"], last["session_id"]]
        for path in sorted(lane.glob("**/*.mp4")) if lane.exists() else []:
            videos.append({"lane": row["task_id"], "video": artifact_ref(path),
                           "verification": "NOT_FULL_DECODE_VERIFIED_BY_THIS_INDEX",
                           "claim_limit": "Research visualization; inspect lane receipt for quality and time mode."})
    atomic_json(RUN / f"TASK_MATRIX_REV_{next_n:04d}.json", {
        "schema_version": "rc1-multiline-task-matrix-v1", "created_at": now_iso(),
        "previous": artifact_ref(matrix_files[-1]), "rows": rows,
    })
    atomic_json(RUN / f"VISUAL_REVIEW_INDEX_REV_{next_n:04d}.json", {
        "schema_version": "rc1-multiline-visual-index-v1", "created_at": now_iso(),
        "previous": artifact_ref(visual_files[-1]), "videos": videos,
    })
    print(json.dumps({"matrix_revision": next_n, "results": sum(bool(r.get("results")) for r in rows),
                      "research_videos": len(videos)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initial", "aggregate"))
    args = parser.parse_args()
    (initial if args.mode == "initial" else aggregate)()


if __name__ == "__main__":
    main()
