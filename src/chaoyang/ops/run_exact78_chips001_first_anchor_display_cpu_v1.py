"""Unrouted 0901 Chips001 CPU display candidate; no model or renderer calls.

The only Robot q input is the sealed structural-v2 IK state freshly solved
from the pinned HaWoR prediction. It is used solely for diagnostic display.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline.exact78_chips001_first_anchor_display_safety_v1 import apply_display_policy


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "exact78_0901_chips001_first_anchor_display_cpu_v1_20261001"
OPERATION = "run_exact78_chips001_first_anchor_display_cpu_v1"
SESSION = "get_potato_chips_0901_001"
FRAMES, FPS = 799, 25.0
OUTPUT = ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001/candidate/display_cpu"
_V2 = "_run/current/exact78_newraw_robot_get_potato_chips_0901_001_051c98741044600d_v2/attempts/attempt_0001"
_HAWOR = "_run/current/exact78_newraw_hawor_get_potato_chips_0901_001_051c98741044600d_v4/attempts/attempt_0001"
PINNED = {
    "v2_packet": ("tasks/current/exact78_newraw_robot_get_potato_chips_0901_001_051c98741044600d_v2/TASK_PACKET.json", "188e396f507429992c158175fdbf212eb5492809ddc0b83903186bb92a7f7016"),
    "v2_receipt": ("tasks/receipts/EXACT78_NEWRAW_ROBOT_GET_POTATO_CHIPS_0901_001_051C98741044600D_V2.json", "a8817163c0463b8e0ecf692afc5acb1373c5b0bf076de2a46ba9876fe88f1d90"),
    "v2_outer": (f"{_V2}/RESULT.json", "95b27d8cb9c80cf61b8743485e33efb494e6e114c1c5a76f01f146a8d6c36358"),
    "v2_candidate": (f"{_V2}/candidate/robot_v2/RESULT.json", "ed69b239aa062ad33df5b4d5ee695a7a35e5e99e77a377ae221a6342bb3fdc87"),
    "stage_result": (f"{_V2}/candidate/robot_v2/structural_stage_ffmpeg44/RESULT.json", "81c66948ebf8b3b95621c3dba8b62959a5fa8f266258480961b3cb1b21a5a162"),
    "states": (f"{_V2}/candidate/robot_v2/structural_stage_ffmpeg44/ROBOT_CAMERA_DIAGNOSTIC_STATES.npz", "f601a04605c5e77c8d98fccb436cc76b55efb7280dff351bb303b1ba3ce9db08"),
    "frame_rows": (f"{_V2}/candidate/robot_v2/structural_stage_ffmpeg44/ROBOT_FRAME_STATES.json", "8b8eba6dc0bd460780a58cc6dfc22fa7aca106deb639d7832e63ea179e0d5717"),
    "hawor_outer": (f"{_HAWOR}/RESULT.json", "d78580a73496a3b63e4f007f7051f0340049d5de757b278023a5581aae881d38"),
    "hawor_npz": (f"{_HAWOR}/candidate/hawor/HAWOR_CAMERA_PREDICTIONS.npz", "f4ba83da13ded8eb0913df73959ea4351de68b708ed391fc10b0f9f53c8bff6e"),
    "robot_asset_pin": ("assets/robot/ROBOT_ASSET_PIN.json", "4bc508a45fd460610a73bc9fb6bf194d97a5cfdef5e863bd1069683921cd1330"),
}


def _refs() -> dict[str, dict[str, Any]]:
    refs = {}
    for role, (relative, expected_sha) in PINNED.items():
        path = ROOT / relative
        if (path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(ROOT.resolve(strict=True))):
            raise RuntimeError("CHIPS001_PINNED_INPUT_MISSING:" + role)
        ref = file_ref(path)
        if ref["sha256"] != expected_sha:
            raise RuntimeError("CHIPS001_PINNED_INPUT_SHA_DRIFT:" + role)
        refs[role] = ref
    return refs


def _json(ref: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("CHIPS001_JSON_OBJECT_REQUIRED")
    return value


def _payload() -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray,
                        np.ndarray, list[dict[str, Any]]]:
    refs = _refs()
    packet, receipt, outer, candidate, result, rows, hawor_outer = (
        _json(refs[key]) for key in
        ("v2_packet", "v2_receipt", "v2_outer", "v2_candidate",
         "stage_result", "frame_rows", "hawor_outer"))
    if (packet.get("config", {}).get("resume_policy") != "SEALED_HAWOR_ONLY_NO_PARTIAL"
            or packet.get("config", {}).get("frame_count") != FRAMES
            or receipt.get("result") != refs["v2_outer"]
            or outer.get("status") != "PASSED"
            or outer.get("visual_qualified") is not False
            or outer.get("stage", {}).get("result") != refs["v2_candidate"]
            or outer.get("stage", {}).get("stage_result") != refs["stage_result"]
            or candidate.get("old_robot_state_consumed") is not False
            or candidate.get("visual_qualified") is not False
            or candidate.get("source_hawor_result") != refs["hawor_outer"]
            or candidate.get("stage_result") != refs["stage_result"]
            or hawor_outer.get("status") != "PASSED"
            or result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("visual_qualified") is not False
            or result.get("session_id") != SESSION or result.get("frame_count") != FRAMES
            or result.get("inputs", {}).get("hawor_npz") != refs["hawor_npz"]
            or result.get("outputs", {}).get("states") != refs["states"]
            or result.get("outputs", {}).get("frame_states") != refs["frame_rows"]
            or rows.get("session_id") != SESSION or rows.get("frame_count") != FRAMES
            or not isinstance(rows.get("rows"), list)
            or len(rows["rows"]) != 2 * FRAMES):
        raise RuntimeError("CHIPS001_FRESH_V2_HAWOR_CHAIN_DRIFT")
    with np.load(refs["states"]["path"], allow_pickle=False) as sealed:
        required = {"q_arm", "q_hand", "prediction_valid", "original_frame_indices",
                    "fps_nominal"}
        if not required.issubset(sealed.files):
            raise RuntimeError("CHIPS001_FRESH_PER_FRAME_IK_Q_UNAVAILABLE")
        q = sealed["q_arm"].copy()
        hand = sealed["q_hand"].copy()
        valid = sealed["prediction_valid"].copy()
        indices = sealed["original_frame_indices"].copy()
        fps = float(sealed["fps_nominal"])
    with np.load(refs["hawor_npz"]["path"], allow_pickle=False) as hawor:
        if not {"prediction_valid", "original_frame_indices", "fps_nominal"}.issubset(hawor.files):
            raise RuntimeError("CHIPS001_HAWOR_TIMELINE_UNAVAILABLE")
        hawor_valid = hawor["prediction_valid"]
        hawor_indices = hawor["original_frame_indices"]
        hawor_fps = float(hawor["fps_nominal"])
    if (q.shape != (FRAMES, 2, 7) or hand.shape != (FRAMES, 2, 22)
            or valid.shape != (2, FRAMES) or valid.dtype != np.bool_
            or not np.isfinite(q).all() or not np.isfinite(hand).all()
            or not np.array_equal(valid, hawor_valid)
            or not np.array_equal(indices, hawor_indices)
            or not np.array_equal(indices, np.arange(FRAMES))
            or fps != FPS or hawor_fps != FPS):
        raise RuntimeError("CHIPS001_FRESH_PER_FRAME_IK_Q_UNAVAILABLE")
    return refs, q, hand, valid, rows["rows"]


def check_only() -> dict[str, Any]:
    refs, q, _, valid, frame_rows = _payload()
    return {
        "status": "PREP_ONLY_PINNED_FRESH_IK_AVAILABLE_NOT_EXECUTED",
        "session_id": SESSION, "frame_count": len(q),
        "frame_side_rows": len(frame_rows),
        "hawor_prediction_rows": int(valid.sum()),
        "v2_outer_sha256": refs["v2_outer"]["sha256"],
        "fresh_v2_ik_states_sha256": refs["states"]["sha256"],
        "old_robot_q_action_input": False,
        "model_calls": 0, "render_calls": 0,
        "visual_qualified": False, "training_eligible": False,
    }


def _active_packet() -> None:
    index = json.loads((ROOT / "tasks/current/INDEX.json").read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active) != 1 or active[0].get("task_id") != TASK_ID:
        raise RuntimeError("CHIPS001_ACTIVE_TASK_REQUIRED")
    packet_path = ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
    if active[0].get("packet_sha256") != file_ref(packet_path)["sha256"]:
        raise RuntimeError("CHIPS001_ACTIVE_PACKET_SHA_DRIFT")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if (packet.get("task_id") != TASK_ID
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("inputs") != _refs()
            or packet.get("config", {}).get("frame_count") != FRAMES
            or packet.get("config", {}).get("old_robot_q_action_input") is not False
            or packet.get("config", {}).get("render_calls") != 0):
        raise RuntimeError("CHIPS001_ACTIVE_PACKET_DRIFT")


def run_candidate(output_root: Path) -> dict[str, Any]:
    authorize_exact78_operation(OPERATION, ["--output-root", str(output_root)])
    _active_packet()
    if (output_root != OUTPUT or output_root.exists()
            or not output_root.resolve().is_relative_to(ROOT.resolve(strict=True))):
        raise RuntimeError("CHIPS001_FRESH_OUTPUT_REQUIRED")
    refs, q, hand, valid, rows = _payload()
    lower, upper = stage._arm_limits(stage.load_pinned_robot_assets(ROOT))
    display, display_rows = apply_display_policy(q, rows, .5 * (lower + upper), valid, FPS)
    if (display.shape != (FRAMES, 2, 7) or len(display_rows) != 2 * FRAMES
            or not np.isfinite(display).all()
            or any(not row["robot_drawable"] or row["visual_qualified"]
                   or row["training_eligible"] or row["motion_evidence_eligible"]
                   for row in display_rows)):
        raise RuntimeError("CHIPS001_DISPLAY_GRID_OR_CLAIM_DRIFT")
    output_root.mkdir(parents=True, exist_ok=False)
    states_path = output_root / "DISPLAY_Q_DIAGNOSTIC.npz"
    np.savez_compressed(states_path, q_arm_display=display,
                        q_hand_display_diagnostic=hand,
                        prediction_valid=valid, original_frame_indices=np.arange(FRAMES))
    rows_path = output_root / "DISPLAY_FRAME_STATES.json"
    with rows_path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "exact78-chips001-display-cpu-frames-v1",
                   "session_id": SESSION, "frame_count": FRAMES, "rows": display_rows},
                  stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    value = {
        "schema_version": "exact78-chips001-display-cpu-result-v1",
        "status": "COMPLETE_CPU_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": SESSION, "frame_count": FRAMES, "frame_side_rows": len(display_rows),
        "source_refs": refs,
        "outputs": {"display_q": file_ref(states_path), "frame_states": file_ref(rows_path)},
        "display_status_counts": dict(Counter(row["display_status"] for row in display_rows)),
        "old_robot_q_action_input": False, "control_ground_truth": False,
        "motion_evidence_eligible": False, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "model_calls": 0, "render_calls": 0, "training_steps": 0,
        "claim_limit": "CPU-only diagnostic display q and source ledger; no rendered or reviewed visual evidence and no action, control or training authority",
    }
    with (output_root / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def main(argv: list[str] | None = None) -> int:
    given = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(given)
    if args.check_only:
        if args.output_root is not None:
            raise RuntimeError("CHIPS001_CHECK_ONLY_OUTPUT_FORBIDDEN")
        value = check_only()
    else:
        if args.output_root is None:
            raise RuntimeError("CHIPS001_OUTPUT_REQUIRED")
        value = run_candidate(args.output_root)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
