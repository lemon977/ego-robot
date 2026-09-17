#!/usr/bin/env python3
"""Append the actual worker/session bindings to the sealed research matrix."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1"
ASSIGNMENTS = {
    "A_MASK": ("/root/mask_chips_role_prep", [
        "play_cards_0901_001", "play_cards_0901_005", "play_cards_0901_015",
        "get_potato_chips_0902_034", "get_potato_chips_0902_023", "get_potato_chips_0902_039",
    ]),
    "B_CLEAN": ("/root", ["play_cards_0903_245", "get_potato_chips_0902_039"]),
    "C_DEPTH": ("/root/depth_registration_lane", ["get_potato_chips_0903_050"]),
    "D_OBJECT_CONTACT": ("/root/object_contact_lane", ["play_cards_0903_245"]),
    "E_ROBOT30": ("/root/robot30_video_inventory", None),
    "F_DOCS": ("/root", []),
}


def main() -> None:
    prior = RUN / "TASK_MATRIX_REV_0002.json"
    path = RUN / "TASK_MATRIX_REV_0003.json"
    if path.exists():
        raise FileExistsError(path)
    matrix = json.loads(prior.read_text(encoding="utf-8"))
    for row in matrix["rows"]:
        worker, sessions = ASSIGNMENTS[row["task_id"]]
        row["worker"] = worker
        row["worker_scope"] = "GLOBAL_INDEX" if row["task_id"] == "F_DOCS" else "BOUND_SESSIONS"
        if sessions is not None:
            row["session_ids"] = sessions
        row["session_binding_status"] = "GLOBAL_NON_SESSION_TASK" if row["task_id"] == "F_DOCS" else "BOUND_FULL_SESSION_IDS"
        row["evidence_closure"] = {
            "start_snapshot": artifact_ref(RUN / "RUN_START_SNAPSHOT.json"),
            "research_receipts": row.get("result_refs", []),
            "research_summaries": row.get("summary_refs", []),
        }
    matrix.update(schema_version="rc1-multiline-task-matrix-v3", created_at=now_iso(),
                  previous=artifact_ref(prior), generator=artifact_ref(Path(__file__)))
    atomic_json(path, matrix)
    print(json.dumps({"matrix": str(path), "tasks": len(matrix["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
