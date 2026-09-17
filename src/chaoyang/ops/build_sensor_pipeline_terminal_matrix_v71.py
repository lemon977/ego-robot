#!/usr/bin/env python3
"""Join H0-H4 into the 332-row V7.1 sensor terminal matrix."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FILES = {
    "h0": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/SENSOR_PIPELINE_ADMISSION_LEDGER.json",
    "h1": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h1_hand/CANONICAL_HAND_LEDGER.json",
    "h2": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h2_tactile/TACTILE_SIDECAR_LEDGER.json",
    "h3": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/attempts/attempt_0002/SENSOR_DEPTH_LEDGER.json",
    "h4": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/SENSOR_MASK_LEDGER.json",
}
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/SENSOR_PIPELINE_TERMINAL_MATRIX.json"
CANARY_TERMINALS = {
    "h3": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/attempts/attempt_0003/TERMINAL_RESULT.json",
    "h4": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/attempts/attempt_0002/TERMINAL_RESULT.json",
}


def ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    ledgers = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in FILES.items()}
    indexes = {name: {row["session_id"]: row for row in value["rows"]} for name, value in ledgers.items()}
    ids = list(indexes["h0"])
    if len(ids) != 332 or any(set(index) != set(ids) for index in indexes.values()):
        raise RuntimeError("H0-H4 session identities do not form one 332-row cohort")
    rows = []
    for sid in ids:
        h0, h1, h2, h3, h4 = (indexes[name][sid] for name in ("h0", "h1", "h2", "h3", "h4"))
        rows.append({
            "session_id": sid,
            "dataset_id": h0["dataset_id"],
            "task": h0["task"],
            "frame_count": h0["frame_count"],
            "h0_admission": h0["admission"],
            "h1_hand": h1["status"],
            "h2_tactile": h2["status"],
            "h3_stereo": h3["status"],
            "h3_cpu_preflight": h3.get("cpu_preflight"),
            "h4_mask": h4["status"],
            "h4_cpu_preflight": h4.get("cpu_preflight"),
            "object6d": "BLOCKED_PREREQ",
            "clean": "BLOCKED_PREREQ",
            "contact": "BLOCKED_PREREQ",
            "robot_geometry": "BLOCKED_PREREQ",
            "primary_blocker": (h0.get("primary_blockers") or [h1.get("primary_blocker") or h3.get("primary_blocker") or h4.get("primary_blocker")])[0],
        })
    canaries = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in CANARY_TERMINALS.items() if path.is_file()
    }
    value = {
        "schema_version": "SENSOR_PIPELINE_TERMINAL_MATRIX_V71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": (
            "H0_H4_FINITE_TERMINAL_WITH_BOUNDED_GPU_CANARIES"
            if len(canaries) == len(CANARY_TERMINALS)
            else "H0_H4_FINITE_TERMINAL_GPU_STAGES_BLOCKED_RESOURCE"
        ),
        "counts": {
            "total": 332,
            "h1_passed": sum(r["h1_hand"] == "PASSED" for r in rows),
            "h2_passed": sum(r["h2_tactile"] == "PASSED" for r in rows),
            "h3_cpu_preflight_passed": sum(r["h3_cpu_preflight"] == "PASSED" for r in rows),
            "h4_cpu_preflight_passed": sum(r["h4_cpu_preflight"] == "PASSED" for r in rows),
            "blocked_source": sum(r["h0_admission"] == "BLOCKED_SOURCE" for r in rows),
            "bounded_gpu_canaries_published": len(canaries),
        },
        "rows": rows,
        "inputs": {name: ref(path) for name, path in FILES.items()},
        "bounded_gpu_canaries": {
            name: {"terminal": ref(CANARY_TERMINALS[name]), "status": receipt.get("status"), "reason": receipt.get("reason")}
            for name, receipt in canaries.items()
        },
        "claim_limit": "H0-H4 finite terminal accounting. Bounded H3/H4 canaries, when present, are development evidence only; full Depth, Mask, Object6D, Clean, Contact and Robot authorities are not produced.",
    }
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(rows[0].keys())
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    result = {"schema_version": "SENSOR_PIPELINE_TERMINAL_MATRIX_V71_RESULT", "status": value["status"], "counts": value["counts"], "matrix": ref(OUT), "claim_limit": value["claim_limit"]}
    (OUT.parent / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value["counts"]))


if __name__ == "__main__":
    main()
