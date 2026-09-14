#!/usr/bin/env python3
"""Audit current V5.2 projection candidates; never admit them for training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SPLIT = Path("/mnt/workspace/code/chaoyang/tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/SPLIT_AND_PAIRING_PLAN.json")
SPLIT_SHA = "8e9977f0fd96e71f485d44c84d0aed2c06033c8fe7b7a97ba5fa0bed40cc1064"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_ref(root: Path, item: dict[str, Any], label: str) -> Path:
    p = Path(item["path"])
    if not p.is_absolute():
        p = root / p
    if not p.is_file() or p.is_symlink() or p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
        raise RuntimeError(f"{label}: ref mismatch")
    return p


def atomic(path: Path, value: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    if sha256(SPLIT) != SPLIT_SHA:
        raise RuntimeError("frozen split SHA mismatch")
    split = json.loads(SPLIT.read_text())["session_plan"]
    rows = []
    for session_root in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        result_path = session_root / "RESULT.json"
        d = json.loads(result_path.read_text())
        if d.get("status") != "BLOCKED_PREREQ_ROBOTIZED_RGB_VALID_MASK_AND_GOLDSET" or d.get("control_ground_truth") is not False:
            raise RuntimeError(f"unexpected candidate status: {session_root}")
        candidate = check_ref(session_root, d["output"], "candidate")
        for name, item in d["inputs"].items():
            check_ref(Path("/"), item, f"input:{name}")
        with np.load(candidate, allow_pickle=False) as z:
            if bool(z["control_ground_truth"]):
                raise RuntimeError("control truth bit enabled")
            frames = np.asarray(z["frame_ids"])
            count = int(d["frame_count"])
            if frames.dtype != np.int64 or not np.array_equal(frames, np.arange(count)):
                raise RuntimeError("frame identity mismatch")
            original = z["future_2d_xy_original"]
            normalized = z["future_2d_xy_normalized"]
            valid = z["future_2d_valid"]
            current = z["current_frame_valid"]
            starts = z["projection_upper_bound_h50_starts"]
            if original.shape != (count, 50, 2, 2) or normalized.shape != original.shape or valid.shape != (count, 50, 2) or current.shape != (count,):
                raise RuntimeError("projection candidate shape mismatch")
            if not np.array_equal(np.isfinite(original).all(-1), valid) or not np.array_equal(np.isfinite(normalized).all(-1), valid):
                raise RuntimeError("NaN/valid mismatch")
            if len(starts) != d["projection"]["h50_window_upper_bound"]:
                raise RuntimeError("H50 count mismatch")
        sid = d["session"]
        rows.append({
            "task": d["task"], "session_id": sid, "split": split[sid]["split"],
            "frame_count": count, "h50_projection_upper_bound": int(len(starts)),
            "result": ref(result_path), "candidate": ref(candidate),
            "training_admitted": False,
        })
    by_task_split = Counter((r["task"], r["split"]) for r in rows)
    windows = Counter()
    for r in rows:
        windows[(r["task"], r["split"])] += r["h50_projection_upper_bound"]
    out = {
        "schema_version": "exact78-visual-retarget-projection-candidate-index-v52-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_CURRENT_PROJECTION_CANDIDATES_NOT_TRAINING_ADMITTED",
        "authority": False,
        "split": ref(SPLIT),
        "counts": {
            "sessions": len(rows),
            "sessions_by_task_split": {f"{k[0]}:{k[1]}": v for k, v in sorted(by_task_split.items())},
            "h50_upper_bound_by_task_split": {f"{k[0]}:{k[1]}": v for k, v in sorted(windows.items())},
            "training_admitted": 0,
        },
        "rows": rows,
        "claim_limit": "Projection-only upper bounds. Missing goldset-passed Robotized RGB and shared RGB valid masks; no optimizer/checkpoint authority.",
    }
    target = root / "CURRENT_PROJECTION_CANDIDATE_INDEX.json"
    atomic(target, out)
    print(json.dumps({"status": out["status"], "counts": out["counts"], "output": ref(target)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
