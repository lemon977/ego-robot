"""Fenced Cards042 Robot-only flange-link candidate; no model or old Robot input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_robot_flange_link_cards042_v1"
TASK = "exact78_robot_flange_link_cards042_v1_20261001"
HAWOR_RELATIVE = ("_run/current/exact78_raw_to_robot_t2_0902_cards042_v1_20260930/"
                  "attempts/attempt_0001/candidate/hawor")


def _allowed_input_paths(root: Path, refs: dict) -> None:
    """Reject historical/derived action paths before importing Robot code."""
    hawor = root / HAWOR_RELATIVE
    expected = {
        "hawor_result": hawor / "RESULT.json",
        "hawor_npz": hawor / "HAWOR_CAMERA_PREDICTIONS.npz",
        "frame_provenance": hawor / "FRAME_PROVENANCE.json",
        "input_audit": hawor / "INPUT_AUDIT.json",
        "hawor_seal": hawor.parent / "HAWOR_STAGE_SEAL.json",
        "native_open_audit": hawor.parent / "NATIVE_OPEN_AUDIT.json",
        "robot_asset_pin": root / "assets/robot/ROBOT_ASSET_PIN.json",
        "mount_proxy": root / ("assets/robot/hardware_handoff/kaihand_flange_adapter_v1/"
                               "excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json"),
        "flange_mesh": root / "assets/robot/V2相机固定法兰PRO版本.STL",
    }
    if any(Path(refs[role]["path"]).resolve() != path.resolve()
           for role, path in expected.items()):
        raise RuntimeError("FLANGE_CANDIDATE_SEALED_PATH_DRIFT")
    audit = json.loads((hawor / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    if refs["raw_video"] != audit.get("raw_source_ref"):
        raise RuntimeError("FLANGE_CANDIDATE_RAW_SOURCE_DRIFT")


def _packet_and_spec(spec_path: Path) -> dict:
    # Called only after the CLI gate (also invoked from main for direct -m use).
    from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json

    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active) != 1 or active[0].get("task_id") != TASK:
        raise RuntimeError("FLANGE_CANDIDATE_PACKET_NOT_ACTIVE")
    packet_path = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
    if (REPO_ROOT / active[0]["packet_path"]).resolve() != packet_path.resolve():
        raise RuntimeError("FLANGE_CANDIDATE_PACKET_PATH_DRIFT")
    if artifact_ref(packet_path)["sha256"] != active[0].get("packet_sha256"):
        raise RuntimeError("FLANGE_CANDIDATE_PACKET_SHA_DRIFT")
    packet = load_json(packet_path)
    attempt = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
    expected = attempt / "FLANGE_SPEC.json"
    if (packet.get("task_id") != TASK
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or str(attempt.relative_to(REPO_ROOT)) not in packet.get("write_set", [])
            or spec_path.resolve() != expected.resolve()
            or packet.get("inputs", {}).get("flange_spec") != artifact_ref(expected)):
        raise RuntimeError("FLANGE_CANDIDATE_SPEC_OR_WRITE_SET_DRIFT")
    spec = load_json(expected)
    if (spec.get("task_id") != TASK or spec.get("session_id") != "play_cards_0902_042"
            or spec.get("source_kind") != "tracker"
            or spec.get("output_root") != str(attempt / "candidate")
            or spec.get("inputs") != packet.get("inputs", {}).get("source_refs")):
        raise RuntimeError("FLANGE_CANDIDATE_SPEC_IDENTITY_DRIFT")
    if set(spec["inputs"]) != {"hawor_result", "hawor_npz", "frame_provenance",
                               "input_audit", "hawor_seal", "native_open_audit",
                               "raw_video", "robot_asset_pin", "mount_proxy", "flange_mesh"}:
        raise RuntimeError("FLANGE_CANDIDATE_INPUT_ROLE_DRIFT")
    _allowed_input_paths(REPO_ROOT, spec["inputs"])
    for role, ref in spec["inputs"].items():
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError(f"FLANGE_CANDIDATE_INPUT_SHA_DRIFT:{role}")
    if (attempt / "candidate").exists():
        raise RuntimeError("FLANGE_CANDIDATE_FRESH_OUTPUT_REQUIRED")
    return spec


def main() -> int:
    authorize_exact78_operation(OPERATION, sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = _packet_and_spec(args.spec)

    # The algorithm is imported only after the live-writer gate, exact spec,
    # source roles and SHAs have all been verified.
    from chaoyang.pipeline.exact78_fresh_robot_flange_successor_v1 import run_flange_robot_stage

    refs = spec["inputs"]
    result = run_flange_robot_stage(
        hawor_npz=Path(refs["hawor_npz"]["path"]),
        frame_provenance=Path(refs["frame_provenance"]["path"]),
        input_audit=Path(refs["input_audit"]["path"]),
        raw_video=Path(refs["raw_video"]["path"]),
        source_kind=spec["source_kind"], session_id=spec["session_id"],
        output_root=Path(spec["output_root"]))
    if (result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("frame_count") != 171
            or result.get("single_change") !=
            "IK_AND_RASTER_HAND_ROOT_PARENT_FLANGE_INSTEAD_OF_SIDE_TOOL"):
        raise RuntimeError("FLANGE_CANDIDATE_RESULT_DRIFT")
    print(json.dumps({"status": result["status"],
                      "result": str(Path(spec["output_root"]) / "RESULT.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
