"""Fenced, render-only Cards042 camera-direction candidate operation.

Not a current maintained operation.  A new finite Task Packet and contract
route must authorize this module before it can render a single frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, process_identity
from chaoyang.pipeline.exact78_robot_render_direction_v1 import run_render_only


TASK = "exact78_robot_render_direction_cards042_v1_20260930"


def authorize(spec_path: Path) -> dict:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    routes = [row for row in index["task_packets"]
              if row.get("execution_allowed") is True]
    if len(routes) != 1 or routes[0]["task_id"] != TASK:
        raise RuntimeError("RENDER_DIRECTION_NOT_ROUTABLE")
    packet_path = REPO_ROOT / routes[0]["packet_path"]
    if (packet_path != REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
            or artifact_ref(packet_path)["sha256"] != routes[0]["packet_sha256"]):
        raise RuntimeError("RENDER_DIRECTION_PACKET_SHA_DRIFT")
    packet = load_json(packet_path)
    if (packet.get("task_id") != TASK or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("inputs", {}).get("render_spec") != artifact_ref(spec_path)):
        raise RuntimeError("RENDER_DIRECTION_SPEC_NOT_PACKET_BOUND")
    writer = packet["writer"]
    identity = process_identity(writer["pid"])
    if (writer.get("scope") != TASK or not identity["alive"]
            or identity["start_ticks"] != writer["proc_start_ticks"]):
        raise RuntimeError("RENDER_DIRECTION_WRITER_FENCE_DRIFT")
    attempt = REPO_ROOT / packet["write_set"][0]
    if (attempt != REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
            or load_json(attempt / "WRITER.json") != writer
            or spec_path.resolve() != attempt / "RENDER_SPEC.json"):
        raise RuntimeError("RENDER_DIRECTION_WRITE_SET_DRIFT")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    authorize(args.spec)
    spec = load_json(args.spec)
    if spec.get("task_id") != TASK:
        raise RuntimeError("RENDER_DIRECTION_TASK_ID_DRIFT")
    result = run_render_only(spec)
    if result.get("status") != "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS":
        raise RuntimeError("RENDER_DIRECTION_RESULT_STATUS_DRIFT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
