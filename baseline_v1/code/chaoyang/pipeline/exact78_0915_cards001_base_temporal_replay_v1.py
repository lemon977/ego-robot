"""Finite, unregistered CPU replay of the frozen 0915 Cards001 arm diagnostic.

Only ``--check-only`` is a command-line entry. A fenced future operation may
call ``run_replay`` with a new output directory after registering a Task Packet.
No historical Robot q, video, or holdout data is a motion input.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ASSET_PIN_RELATIVE


SESSION = "play_cards_0915_001"
FRAMES = 150
SEALED = stage.ROOT / "_run/current/exact78_raw_to_robot_t2_0915_cards001_v1_20260930/attempts/attempt_0001/candidate"
HAWOR = SEALED / "hawor"
BASELINE = SEALED / "robot/v1_stage_with_ffmpeg44_encoder/BASE_SEARCH.json"
PINS = {
    HAWOR / "HAWOR_CAMERA_PREDICTIONS.npz": "3aaba1de7c112ebae21ed3f459ea186d6460e2847168b6f3c4062b4c5e17519c",
    HAWOR / "FRAME_PROVENANCE.json": "96c124586d17aeadc02c6cb51176bdce724d8e79f0f722b813329a76bfe43595",
    HAWOR / "INPUT_AUDIT.json": "a8260d32d11f3a2915178b52aaa8a4414b1cc4fa31b2395150ca50e093498451",
    BASELINE: "e12a911bb03a4a5278233a64337122738757bf685483974a95b46f6c635be961",
    Path(stage.__file__): "9018d49be9030ec84c48e470f5b600ac828665bfe7e974b64b5f0736375b196c",
    Path(stage.adapter.__file__): "14f8b1897536847e7c2e5e97e3ad54a37a45e3076ccd9c03913a385e253f97b0",
    Path(stage.scene.__file__): "8680b6d26dc4703f6e3c099fc8b7a937f92cfdbbe2ba3ed073f693b522839793",
    stage.ROOT / "src/chaoyang/pipeline/robot_renderer_cycles.py": "ff83deb3417b8183560e59d7f73504d5ed5161f9d3202292850009792bde2c8a",
    stage.ROOT / "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py": "9f4354812a990fce6a34b8da649c7fa81dcbcee4ca88d64dd16a6d7df395740f",
}


def _verified(path: Path, expected: str) -> dict[str, Any]:
    reference = file_ref(path)
    if reference["sha256"] != expected:
        raise stage.FreshRobotError(f"BASE_TEMPORAL_PIN_DRIFT:{path}")
    return reference


def check_only() -> dict[str, Any]:
    """Read sealed identities and dimensions; never solve IK or write output."""
    refs = {str(path): _verified(path, sha) for path, sha in PINS.items()}
    audit = json.loads((HAWOR / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    raw_ref = audit["raw_source_ref"]
    raw = Path(raw_ref["path"])
    if (audit.get("session_id") != SESSION or audit.get("source_role") != "development"
            or audit.get("source_kind") != "vst0915" or raw.name == ""):
        raise stage.FreshRobotError("BASE_TEMPORAL_SOURCE_DRIFT")
    if file_ref(raw) != raw_ref:
        raise stage.FreshRobotError("BASE_TEMPORAL_RAW_SHA_DRIFT")
    values, provenance, _ = stage.load_fresh_inputs(
        HAWOR / "HAWOR_CAMERA_PREDICTIONS.npz", HAWOR / "FRAME_PROVENANCE.json",
        HAWOR / "INPUT_AUDIT.json", raw, SESSION)
    valid = values["prediction_valid"]
    if (values["joints_3d_camera"].shape != (2, FRAMES, 21, 3)
            or valid.shape != (2, FRAMES) or not valid.all()
            or len(provenance["frame_rows"]) != FRAMES):
        raise stage.FreshRobotError("BASE_TEMPORAL_GRID_DRIFT")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    if (len(base["candidates"]) != 32 or base["selected_index"] != 24
            or base["sample_frames"] != stage._sample_frames(valid)
            or base["anchor_frames"] != [0, 0]):
        raise stage.FreshRobotError("BASE_TEMPORAL_SEARCH_DRIFT")
    return {"status": "READY_CHECK_ONLY_NO_IK", "session_id": SESSION,
            "frame_count": FRAMES, "input_refs": refs, "raw_source_ref": raw_ref,
            "robot_asset_pin": file_ref(stage.ROOT / ASSET_PIN_RELATIVE),
            "mount_proxy": file_ref(stage.MOUNT), "baseline_selected_index": 24,
            "baseline_arm_pass": {"left": 33, "right": 2}}


def _evaluate_base(joints: np.ndarray, valid: np.ndarray, assets: Any,
                   mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   neutral: np.ndarray, anchors: np.ndarray, midpoint: np.ndarray,
                   sampled: list[int], spec: tuple[float, float, float, float]) -> dict[str, Any]:
    """The stage v1 spatial evaluator, retained byte-for-byte in its decisions."""
    base = stage._candidate_base(midpoint, spec)
    inverse = np.linalg.inv(base)
    offsets = np.zeros((2, 3))
    q_anchor = neutral.copy()
    neutral_roots = np.asarray([base @ stage._tool_fk(assets, side, neutral[side]) @ mounts[side]
                                for side in range(2)])
    origin_basis: list[np.ndarray | None] = [None, None]
    for side, anchor in enumerate(anchors):
        if anchor < 0:
            continue
        try:
            origin_basis[side] = stage._palm_basis(joints[side, anchor], side)
            wanted = stage._pose(joints[side, anchor, 0], neutral_roots[side, :3, :3])
            solved, _, _, _ = stage._arm_attempt(
                assets, side, inverse @ wanted @ np.linalg.inv(mounts[side]),
                lower[side], upper[side], [neutral[side]])
            q_anchor[side] = solved
            actual = base @ stage._tool_fk(assets, side, solved) @ mounts[side]
            offsets[side] = actual[:3, 3] - joints[side, anchor, 0]
        except Exception:
            origin_basis[side] = None
    errors = []
    passed = 0
    for frame in sampled:
        for side in range(2):
            if not valid[side, frame] or origin_basis[side] is None:
                continue
            try:
                basis = stage._palm_basis(joints[side, frame], side)
                rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                wanted = stage._pose(joints[side, frame, 0] + offsets[side], rotation)
                _, pos, rot, margin = stage._arm_attempt(
                    assets, side, inverse @ wanted @ np.linalg.inv(mounts[side]),
                    lower[side], upper[side], [q_anchor[side], neutral[side]])
                passed += stage._arm_status(pos, rot, margin).startswith("PASS")
                errors.append(pos + rot)
            except Exception:
                continue
    score = (-passed, float(np.median(errors)) if errors else 1e9)
    return {"spec": spec, "score": score, "sample_rows": len(errors),
            "sample_passed": passed, "base": base, "offsets": offsets,
            "anchor_q": q_anchor, "neutral_roots": neutral_roots,
            "origin_basis": origin_basis}


def build_candidates(joints: np.ndarray, valid: np.ndarray, assets: Any,
                     mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     neutral: np.ndarray) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    anchors = np.asarray([int(np.flatnonzero(valid[side])[0]) if valid[side].any() else -1
                          for side in range(2)], dtype=int)
    points = [joints[side, anchor, 0] for side, anchor in enumerate(anchors) if anchor >= 0]
    midpoint = np.mean(points, axis=0) if points else np.asarray((0.0, 0.0, 0.8))
    sampled = stage._sample_frames(valid)
    specs = [(x, y, z, yaw) for x in (0.50, 0.65) for y in (-0.05, 0.05)
             for z in (0.60, 0.75) for yaw in (-15.0, 0.0, 15.0)]
    if not points:
        raise stage.FreshRobotError("BASE_TEMPORAL_NO_ANCHOR")
    def evaluate(spec: tuple[float, float, float, float]) -> dict[str, Any]:
        return _evaluate_base(joints, valid, assets, mounts, lower, upper,
                              neutral, anchors, midpoint, sampled, spec)
    coarse = [evaluate(spec) for spec in specs]
    first = min(coarse, key=lambda row: row["score"])
    x, y, z, yaw = first["spec"]
    fine = [(x + dx, y + dy, z + dz, yaw + dyaw)
            for dx, dy, dz, dyaw in ((-.025, 0, 0, 0), (.025, 0, 0, 0),
                                     (0, -.025, 0, 0), (0, .025, 0, 0),
                                     (0, 0, -.025, 0), (0, 0, .025, 0),
                                     (0, 0, 0, -5.0), (0, 0, 0, 5.0))]
    rows = [*coarse, *(evaluate(spec) for spec in fine)]
    if len(rows) != 32:
        raise stage.FreshRobotError("BASE_TEMPORAL_CANDIDATE_COUNT")
    top = select_top_indices(rows)
    return rows, top, sampled


def select_top_indices(rows: list[dict[str, Any]]) -> list[int]:
    if len(rows) != 32:
        raise stage.FreshRobotError("BASE_TEMPORAL_CANDIDATE_COUNT")
    ranked = sorted(range(32), key=lambda index: (*rows[index]["score"], index))
    if 24 not in ranked[:8]:
        raise stage.FreshRobotError("BASE_TEMPORAL_ORIGINAL_EXCLUDED")
    return ranked[:8]


def replay_candidate(row: dict[str, Any], display: np.ndarray,
                     sources: np.ndarray, assets: Any, mounts: np.ndarray,
                     lower: np.ndarray, upper: np.ndarray,
                     neutral: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    inverse = np.linalg.inv(row["base"])
    q_arm = np.tile(neutral[None], (FRAMES, 1, 1))
    rows: list[dict[str, Any]] = []
    for side in range(2):
        previous = previous_previous = None
        previous_frame = None
        for frame in range(FRAMES):
            item: dict[str, Any] = {"frame": frame, "side": stage.SIDES[side],
                                    "motion_source": str(sources[side, frame])}
            if not np.isfinite(display[side, frame]).all():
                item.update(status="DIAGNOSTIC_NEUTRAL_NO_ANCHOR", pass_numeric=False)
            elif row["origin_basis"][side] is None:
                item.update(status="HOLD_BASIS_UNVERIFIED", pass_numeric=False)
            else:
                try:
                    basis = stage._palm_basis(display[side, frame], side)
                    rotation = basis @ row["origin_basis"][side].T @ row["neutral_roots"][side, :3, :3]
                    target = stage._pose(display[side, frame, 0] + row["offsets"][side], rotation)
                    seeds = ([row["anchor_q"][side], neutral[side]] if previous is None
                             else [previous, row["anchor_q"][side]])
                    solved, position, orientation, margin = stage._arm_attempt(
                        assets, side, inverse @ target @ np.linalg.inv(mounts[side]),
                        lower[side], upper[side], seeds, previous=previous)
                    status = stage._arm_status(position, orientation, margin)
                    if status.startswith("PASS") and previous is not None and previous_frame == frame - 1:
                        if np.max(np.abs(solved - previous)) > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
                            status = "HOLD_ARM_VELOCITY"
                        elif (previous_previous is not None and
                              np.max(np.abs(solved - 2 * previous + previous_previous))
                              > stage.ARM_ACCELERATION_GATE_RAD + 1e-9):
                            status = "HOLD_ARM_ACCELERATION"
                    if status.startswith("PASS"):
                        q_arm[frame, side] = solved
                        previous_previous = (previous.copy() if previous is not None
                                             and previous_frame == frame - 1 else None)
                        previous = solved
                    else:
                        q_arm[frame, side] = previous if previous is not None else neutral[side]
                    previous_frame = frame
                    item.update(status=status, pass_numeric=status.startswith("PASS"),
                                position_mm=position, rotation_deg=orientation,
                                branch_margin_m_min=margin)
                except Exception as exc:
                    q_arm[frame, side] = previous if previous is not None else neutral[side]
                    item.update(status="HOLD_ARM_SOLVER", pass_numeric=False,
                                error_type=type(exc).__name__)
            rows.append(item)
    return sorted(rows, key=lambda item: (item["frame"], item["side"])), q_arm


def longest_hold(rows: list[dict[str, Any]], side: str) -> dict[str, int | None]:
    sequence = sorted((row for row in rows if row["side"] == side), key=lambda row: row["frame"])
    best_start, best_length = None, 0
    start, length = None, 0
    for row in sequence:
        if not row["pass_numeric"]:
            if start is None:
                start = row["frame"]
            length += 1
            if length > best_length:
                best_start, best_length = start, length
        else:
            start, length = None, 0
    return {"start_frame": best_start, "end_frame": None if best_start is None else best_start + best_length - 1,
            "frames": best_length}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for side in stage.SIDES:
        side_rows = [row for row in rows if row["side"] == side]
        if len(side_rows) != FRAMES:
            raise stage.FreshRobotError("BASE_TEMPORAL_INCOMPLETE_GRID")
        summary[side] = {"pass": sum(row["pass_numeric"] for row in side_rows),
                         "status_counts": dict(Counter(row["status"] for row in side_rows)),
                         "longest_hold": longest_hold(side_rows, side),
                         "first_frame": side_rows[0]}
    return summary


def run_replay(output_root: Path) -> dict[str, Any]:
    """Only a future fenced, registered operation should call this function."""
    preflight = check_only()
    if output_root.exists() or stage.ROOT not in output_root.resolve().parents:
        raise stage.FreshRobotError("BASE_TEMPORAL_OUTPUT_NOT_NEW_PROJECT_PATH")
    raw = Path(preflight["raw_source_ref"]["path"])
    values, _, _ = stage.load_fresh_inputs(HAWOR / "HAWOR_CAMERA_PREDICTIONS.npz",
                                            HAWOR / "FRAME_PROVENANCE.json",
                                            HAWOR / "INPUT_AUDIT.json", raw, SESSION)
    joints, valid = values["joints_3d_camera"], values["prediction_valid"]
    display, sources, _, _ = stage.display_tracks(joints, valid, float(values["fps_nominal"]))
    assets = stage.load_pinned_robot_assets(stage.ROOT)
    mounts, _ = stage._mounts()
    lower, upper = stage._arm_limits(assets)
    neutral = .5 * (lower + upper)
    candidates, selected, sampled = build_candidates(display, valid, assets, mounts, lower, upper, neutral)
    sealed = json.loads(BASELINE.read_text(encoding="utf-8"))
    for index, row in enumerate(candidates):
        old = sealed["candidates"][index]
        if (list(row["spec"]) != old["spec"] or row["sample_rows"] != old["sample_rows"]
                or row["sample_passed"] != old["sample_passed"]
                or not np.allclose(row["score"], old["score"], rtol=0, atol=1e-8)):
            raise stage.FreshRobotError(f"BASE_TEMPORAL_SPATIAL_REPLAY_DRIFT:{index}")
    if sampled != sealed["sample_frames"] or selected[0] != sealed["selected_index"]:
        raise stage.FreshRobotError("BASE_TEMPORAL_RANK_DRIFT")
    results = []
    frame_rows = []
    for index in selected:
        rows, _ = replay_candidate(candidates[index], display, sources, assets, mounts,
                                   lower, upper, neutral)
        summary = summarize(rows)
        results.append({"candidate_index": index, "spec": list(candidates[index]["spec"]),
                        "sample_score": list(candidates[index]["score"]),
                        "sample_passed": candidates[index]["sample_passed"],
                        "sample_rows": candidates[index]["sample_rows"], "sides": summary})
        frame_rows.extend({"candidate_index": index, **item} for item in rows)
    if len(frame_rows) != 8 * 2 * FRAMES:
        raise stage.FreshRobotError("BASE_TEMPORAL_REPLAY_INCOMPLETE")
    original = next(item for item in results if item["candidate_index"] == 24)
    if {side: original["sides"][side]["pass"] for side in stage.SIDES} != {"left": 33, "right": 2}:
        raise stage.FreshRobotError("BASE_TEMPORAL_ORIGINAL_PASS_DRIFT")
    original_longest = {side: original["sides"][side]["longest_hold"]["frames"]
                        for side in stage.SIDES}
    for item in results:
        item["improves_both_sides_and_shortens_both_longest_holds"] = all(
            item["sides"][side]["pass"] > original["sides"][side]["pass"]
            and item["sides"][side]["longest_hold"]["frames"] < original_longest[side]
            for side in stage.SIDES)
    result = {"schema_version": "exact78-0915-cards001-base-temporal-replay-v1",
              "status": "COMPLETE_CPU_NUMERIC_CANDIDATE_NOT_VISUAL_QUALIFICATION",
              "session_id": SESSION, "frame_count": FRAMES,
              "sample_frames": sampled, "selected_indices": selected,
              "all_spatial_candidates": [{"index": i, "spec": list(row["spec"]),
                                          "score": list(row["score"]),
                                          "sample_rows": row["sample_rows"],
                                          "sample_passed": row["sample_passed"]}
                                         for i, row in enumerate(candidates)],
              "candidates": results, "baseline_pass": {"left": 33, "right": 2},
              "baseline_longest_hold": original_longest,
              "base_only_hypothesis": ("NUMERIC_CANDIDATE_ONLY_REQUIRES_INDEPENDENT_REVIEW"
                                       if any(item["improves_both_sides_and_shortens_both_longest_holds"]
                                              for item in results)
                                       else "REJECT_BASE_SORT_ALONE_SUFFICES"),
              "input_refs": preflight["input_refs"],
              "raw_source_ref": preflight["raw_source_ref"],
              "robot_asset_pin": preflight["robot_asset_pin"],
              "mount_proxy": preflight["mount_proxy"],
              "executor_code": file_ref(Path(__file__)),
              "control_ground_truth": False, "world_pose_verified": False,
              "visual_qualified": False}
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output_root / "FRAME_STATUSES.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "exact78-0915-cards001-base-temporal-frames-v1",
                   "session_id": SESSION, "rows": frame_rows}, stream,
                  ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(check_only(), ensure_ascii=False, sort_keys=True))
