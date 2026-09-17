#!/usr/bin/env python3
"""Publish an immutable V7.1 Task Packet index successor with valid read sets."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import atomic_write, canonical_bytes, sha256_bytes, sha256_file
from chaoyang.governance.v52_contracts import atomic_write_new, validate_task_packet


POINTER = ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json"
TARGETS = {"visual_aux_chips_pair_v1", "visual_aux_poker_pair_v1"}
OLD_LEDGER = "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robotized_causal/ROBOTIZED_RGB_LEDGER.json"
UPSTREAM_STATE = "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_robot_R7_2/AUTOMATION_STATE.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def successor_packet(packet: dict[str, Any], predecessor: dict[str, Any]) -> dict[str, Any]:
    updated = dict(packet)
    updated["read_set"] = [UPSTREAM_STATE if item == OLD_LEDGER else item for item in packet["read_set"]]
    updated["executor_epoch"] = max(2, int(packet.get("executor_epoch", 1)) + 1)
    updated["packet_revision"] = "R7_2"
    updated["supersedes_packet"] = predecessor
    updated["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    errors = validate_task_packet(updated)
    if errors:
        raise RuntimeError("invalid successor packet: " + "; ".join(errors))
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_task_packets_R2",
    )
    args = parser.parse_args()
    pointer = load(POINTER)
    predecessor_index = ROOT / pointer["index_path"]
    if sha256_file(predecessor_index) != pointer["index_sha256"]:
        raise RuntimeError("current Task Packet pointer SHA mismatch")
    old_index = load(predecessor_index)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in old_index["task_packets"]:
        task_id = str(entry["task_id"])
        if task_id not in TARGETS:
            entries.append(entry)
            continue
        old_path = ROOT / entry["packet_path"]
        if sha256_file(old_path) != entry["packet_sha256"]:
            raise RuntimeError(f"predecessor packet SHA mismatch: {task_id}")
        predecessor = {
            "path": str(old_path.relative_to(ROOT)),
            "bytes": old_path.stat().st_size,
            "sha256": sha256_file(old_path),
        }
        packet = successor_packet(load(old_path), predecessor)
        packet_path = output / "task_packets" / task_id / "TASK_PACKET.json"
        payload = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
        atomic_write_new(packet_path, payload)
        card = (
            f"# {task_id} R7_2\n\n"
            "当前 read_set 只读取已存在的 Visual Tier 自动状态；终态后沿其 result 引用读取冻结 eligibility。\n\n"
            f"证据边界：{packet['claim_limit']}\n"
        ).encode()
        atomic_write_new(packet_path.with_name("CONTEXT_CARD.md"), card)
        entries.append({
            "task_id": task_id,
            "packet_path": str(packet_path.relative_to(ROOT)),
            "packet_sha256": sha256_bytes(payload),
        })
    index = {
        "schema_version": "chaoyang-v71-task-packet-index-v2",
        "plan_revision": "chaoyang-v7.1",
        "packet_revision": "R7_2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS",
        "task_packets": entries,
        "supersedes_index": {
            "path": str(predecessor_index.relative_to(ROOT)),
            "bytes": predecessor_index.stat().st_size,
            "sha256": sha256_file(predecessor_index),
        },
        "claim_limit": "Execution routing only; no algorithm authority.",
    }
    index_path = output / "TASK_PACKET_INDEX.json"
    index_payload = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(index_path, index_payload)
    atomic_write(
        POINTER,
        canonical_bytes({
            "schema_version": "chaoyang-v71-task-packet-pointer-v1",
            "plan_revision": "chaoyang-v7.1",
            "packet_revision": "R7_2",
            "index_path": str(index_path.relative_to(ROOT)),
            "index_sha256": sha256_bytes(index_payload),
        }),
        mode=0o444,
    )
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
