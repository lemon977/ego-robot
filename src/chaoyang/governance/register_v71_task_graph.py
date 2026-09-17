from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Register all immutable V7.1 packets as bounded task-state rows in one CAS."""

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import AUTHORITY_PATH, TASK_STATE_PATH, load_json, now_iso, publish_bundle


ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/TASK_PACKET_INDEX.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    index = load_json(INDEX)
    by_id = {row["task_id"]: row for row in state.get("tasks", [])}
    added: list[str] = []
    for entry in index["task_packets"]:
        task_id = entry["task_id"]
        if task_id in by_id:
            continue
        packet_path = ROOT / entry["packet_path"]
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        row = {
            "task_id": task_id,
            "phase": "v71_registered_pending",
            "attempt": 0,
            "status": "PENDING",
            "updated_at": now_iso(),
            "heartbeat_at": None,
            "session": None,
            "pid": None,
            "proc_start_ticks": None,
            "gpu_id": None,
            "plan_revision": "chaoyang-v7.1",
            "prerequisites": packet.get("prerequisites", []),
            "task_packet": entry,
        }
        state["tasks"].append(row)
        by_id[task_id] = row
        added.append(task_id)
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": "g0_core_governance_v71",
                "session": None,
                "attempt": 2,
                "status": "PASSED",
                "created_at": now_iso(),
                "message": f"registered {len(added)} V7.1 task packets without changing existing tasks",
            }
        ]
    )[-100:]
    receipt = publish_bundle(
        authority,
        state,
        event_type="V71_TASK_GRAPH_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({"added": added, "receipt": receipt}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
