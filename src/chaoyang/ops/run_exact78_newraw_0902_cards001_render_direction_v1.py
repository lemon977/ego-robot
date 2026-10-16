"""Fenced, render-only camera-direction operation for sealed new Raw Cards001.

The maintained CLI route and a live, exact-command Task Packet are required.
The algorithm is imported only after the gate and frozen-input checks. This
module changes neither IK nor the hand/flange parent-link mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_newraw_0902_cards001_render_direction_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("RENDER_DIRECTION_JSON_OBJECT_REQUIRED")
    return value


def run_candidate(spec_path: Path) -> dict[str, Any]:
    """Only a CLI-gated writer descendant may call this in a registered packet."""
    # Keep this guard at the callable boundary as well as main(): importing
    # this module and invoking run_candidate() must not bypass the writer gate.
    authorize_exact78_operation(OPERATION, ["--spec", str(spec_path)])
    from chaoyang.governance.common import REPO_ROOT, artifact_ref
    from chaoyang.governance import prepare_exact78_newraw_0902_cards001_render_direction_v1 as prep

    spec_path = spec_path.resolve(strict=True)
    if spec_path != prep.SPEC:
        raise ValueError("RENDER_DIRECTION_SPEC_PATH_DRIFT")
    index = _read_object(REPO_ROOT / "tasks/current/INDEX.json")
    allowed = [row for row in index.get("task_packets", [])
               if row.get("execution_allowed") is True]
    if len(allowed) != 1 or allowed[0].get("task_id") != prep.TASK_ID:
        raise ValueError("RENDER_DIRECTION_ACTIVE_PACKET_REQUIRED")
    packet_path = REPO_ROOT / allowed[0]["packet_path"]
    if (packet_path != prep.PACKET
            or artifact_ref(packet_path)["sha256"] != allowed[0]["packet_sha256"]):
        raise ValueError("RENDER_DIRECTION_PACKET_SHA_DRIFT")
    packet = _read_object(packet_path)
    spec = _read_object(spec_path)
    if (packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("config", {}).get("change_only") !=
            "EGO_CAMERA_INV_T_CAMERA_BASE_TO_T_CAMERA_BASE"
            or packet.get("config", {}).get("flange_parent_changed") is not False
            or packet.get("config", {}).get("q_base_mount_recomputed") is not False
            or packet.get("config", {}).get("numeric_gates_changed") is not False
            or packet.get("inputs", {}).get("render_spec") != artifact_ref(spec_path)
            or packet.get("inputs", {}).get("fixed_inputs") != spec.get("inputs")):
        raise ValueError("RENDER_DIRECTION_PACKET_SCOPE_OR_SPEC_DRIFT")
    expected, _, _ = prep.frozen_spec()
    if spec != expected:
        raise ValueError("RENDER_DIRECTION_FROZEN_SPEC_DRIFT")
    # The only renderer implementation is the already-pinned direction-only
    # successor; no monkeypatch or full Robot re-solve is reachable here.
    from chaoyang.pipeline.exact78_robot_render_direction_v1 import run_render_only

    result = run_render_only(spec)
    if (result.get("status") != "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("frame_count") != prep.FRAMES
            or result.get("q_base_mount_recomputed") is not False
            or result.get("numeric_gates_changed") is not False
            or result.get("ego_camera_rule") != "T_CAMERA_BASE_BASE_TO_CAMERA_NO_INVERSION"):
        raise ValueError("RENDER_DIRECTION_RESULT_SCOPE_DRIFT")
    return result


def main() -> int:
    authorize_exact78_operation(OPERATION, sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    result = run_candidate(args.spec)
    print(json.dumps({"status": result["status"],
                      "session_id": result["session_id"],
                      "frame_count": result["frame_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
