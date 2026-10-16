"""Gated Cards042 CPU-only motion diagnostic; no model or renderer invocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation


OPERATION = "run_exact78_cards042_cpu_motion_diagnostic_v1"
TASK_MODES = {
    "exact78_cards042_first_anchor_cpu_v1_20261001": "FIRST_ANCHOR_ONLY",
    "exact78_cards042_continuity_base_cpu_v1_20261001": "BASE_RERANK_ONLY",
}
HAWOR_RELATIVE = ("_run/current/exact78_raw_to_robot_t2_0902_cards042_v1_20260930/"
                  "attempts/attempt_0001/candidate/hawor")


def _allowed_input_paths(root: Path, refs: dict) -> None:
    """Reject old HaWoR/Robot paths before importing the CPU solver."""
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
    }
    if set(refs) != {*expected, "raw_video"}:
        raise RuntimeError("CARDS042_CPU_INPUT_ROLES_DRIFT")
    if any(Path(refs[role]["path"]).resolve() != path.resolve()
           for role, path in expected.items()):
        raise RuntimeError("CARDS042_CPU_HISTORICAL_OR_WRONG_SOURCE_PATH")
    audit = json.loads((hawor / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    if refs["raw_video"] != audit.get("raw_source_ref"):
        raise RuntimeError("CARDS042_CPU_RAW_SOURCE_DRIFT")


def _packet_and_spec(spec_path: Path) -> dict:
    # This is called only after authorize_exact78_operation, including direct -m.
    from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json

    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active) != 1 or active[0].get("task_id") not in TASK_MODES:
        raise RuntimeError("CARDS042_CPU_PACKET_NOT_ACTIVE")
    task = active[0]["task_id"]
    mode = TASK_MODES[task]
    packet_path = REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json"
    if ((REPO_ROOT / active[0]["packet_path"]).resolve() != packet_path.resolve()
            or artifact_ref(packet_path)["sha256"] != active[0]["packet_sha256"]):
        raise RuntimeError("CARDS042_CPU_PACKET_PATH_OR_SHA_DRIFT")
    packet = load_json(packet_path)
    attempt = REPO_ROOT / "_run/current" / task / "attempts/attempt_0001"
    expected = attempt / "CPU_DIAGNOSTIC_SPEC.json"
    if (packet.get("task_id") != task or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or str(attempt.relative_to(REPO_ROOT)) not in packet.get("write_set", [])
            or spec_path.resolve() != expected.resolve()
            or packet.get("inputs", {}).get("cpu_spec") != artifact_ref(expected)):
        raise RuntimeError("CARDS042_CPU_SPEC_OR_WRITE_SET_DRIFT")
    spec = load_json(expected)
    refs = spec.get("inputs", {})
    if (spec.get("task_id") != task or spec.get("mode") != mode
            or spec.get("session_id") != "play_cards_0902_042"
            or spec.get("source_kind") != "tracker"
            or spec.get("frame_count") != 171 or spec.get("fps_nominal") != 30.0
            or spec.get("output_root") != str(attempt / "candidate")
            or refs != packet.get("inputs", {}).get("source_refs")):
        raise RuntimeError("CARDS042_CPU_SPEC_IDENTITY_DRIFT")
    _allowed_input_paths(REPO_ROOT, refs)
    for role, ref in refs.items():
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError(f"CARDS042_CPU_SOURCE_SHA_DRIFT:{role}")
    if (attempt / "candidate").exists():
        raise RuntimeError("CARDS042_CPU_FRESH_OUTPUT_REQUIRED")
    return spec


def main() -> int:
    authorize_exact78_operation(OPERATION, sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = _packet_and_spec(args.spec)
    # Import the algorithm only after CLI gate, live writer, exact spec and SHA checks.
    from chaoyang.pipeline.exact78_cards042_cpu_motion_diagnostic_v1 import run_cpu_diagnostic

    refs = spec["inputs"]
    result = run_cpu_diagnostic(
        hawor_npz=Path(refs["hawor_npz"]["path"]),
        frame_provenance=Path(refs["frame_provenance"]["path"]),
        input_audit=Path(refs["input_audit"]["path"]),
        raw_video=Path(refs["raw_video"]["path"]),
        session_id=spec["session_id"], source_kind=spec["source_kind"],
        output_root=Path(spec["output_root"]), mode=spec["mode"])
    if (result.get("status") != "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("mode") != spec["mode"] or result.get("frame_count") != 171
            or result.get("model_calls") != 0 or result.get("render_calls") != 0):
        raise RuntimeError("CARDS042_CPU_RESULT_DRIFT")
    print(json.dumps({"status": result["status"],
                      "result": str(Path(spec["output_root"]) / "RESULT.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
