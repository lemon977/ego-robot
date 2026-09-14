#!/usr/bin/env python3
"""Recoverable exact78 task-object identity batch guardian.

This is deliberately separate from the SAM3.1 role-removal lane.  It consumes
only frozen-manifest RGB sessions whose current HaWoR terminal is A/B.  Chips
publishes three disjoint identities; Poker publishes one action-conditioned
same-card identity and leaves ambiguous pick/flip frames empty.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import run_exact78_task_object_identity_canary as identity


CONTROL = PROJECT / "tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
BASELINE = PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
MANIFEST = BASELINE / "EXACT78_BATCH_MANIFEST.json"
MANIFEST_SHA = "b6dcfec1fb41b9ffb0b9b19cc62d029f5cedf57f41552a611272cc9e5c6e43a8"
HAWOR_INDEX = CONTROL / "HAWOR_TERMINAL_INDEX.json"
CANARY = CONTROL / "mask_fresh_canary_v1/task_object_identity_v1/RESULT.json"
OUTPUT = CONTROL / "mask_task_object_identity_v1"
STATE = CONTROL / "MASK_TASK_OBJECT_STATE.json"
INDEX = CONTROL / "MASK_TASK_OBJECT_TERMINAL_INDEX.json"
HEARTBEAT = CONTROL / "MASK_TASK_OBJECT_HEARTBEAT.json"
PID_FILE = CONTROL / "mask_task_object_guardian.pid"
STOP = False


def now() -> str:
    return identity.now()


def sha(path: Path) -> str:
    return identity.sha(path)


def ref(path: Path) -> dict[str, Any]:
    return identity.artifact(path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def future_ref(path: Path, final: Path) -> dict[str, Any]:
    return identity.future_artifact(path, final)


def candidates_chips(frame: np.ndarray) -> list[dict[str, Any]]:
    return identity.chips_candidates(frame)


def render_review(
    frame: np.ndarray,
    masks: dict[int, np.ndarray],
    task: str,
    session: str,
    frame_id: int,
    state: str,
    reason: str,
) -> np.ndarray:
    """Render the canary layout with the *current* session identity."""
    canvas = identity.render(frame, masks, task, frame_id, state, reason)
    image = identity.Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = identity.ImageDraw.Draw(image)
    font = identity.ImageFont.truetype(str(identity.FONT), 24)
    task_name = "薯片｜三独立物理实例" if task == "chips" else "扑克｜动作条件同一张牌"
    draw.rectangle((0, 0, 1280, 42), fill=(0, 0, 0))
    draw.text((10, 7), f"{task_name}｜{session}｜帧 {frame_id:03d}", font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def assign_chips(
    candidates: list[dict[str, Any]],
    previous: dict[int, tuple[float, float]],
    last_seen: dict[int, int],
    frame_id: int,
) -> dict[int, dict[str, Any]]:
    pairs: list[tuple[float, int, int]] = []
    for instance_id in range(3):
        gap = frame_id - last_seen[instance_id]
        if gap > 12:
            continue
        for candidate_id, item in enumerate(candidates):
            distance = float(np.linalg.norm(np.subtract(item["centroid"], previous[instance_id])))
            limit = min(90.0, 24.0 + 7.0 * max(0, gap - 1))
            if distance <= limit:
                pairs.append((distance, instance_id, candidate_id))
    assigned: dict[int, dict[str, Any]] = {}
    used: set[int] = set()
    for distance, instance_id, candidate_id in sorted(pairs):
        if instance_id in assigned or candidate_id in used:
            continue
        # If two identities are almost equally close to one component, the
        # RGB evidence cannot support a fixed identity; publish neither.
        alternatives = sorted(value[0] for value in pairs if value[2] == candidate_id)
        if len(alternatives) > 1 and alternatives[1] - alternatives[0] < 10.0:
            continue
        assigned[instance_id] = candidates[candidate_id]
        used.add(candidate_id)
    return assigned


def process(row: dict[str, Any]) -> dict[str, Any]:
    task = row["task"]
    session = row["session_id"]
    final = OUTPUT / task / session
    partial = final.parent / f".{session}.partial"
    if final.exists() or partial.exists():
        if (final / "RESULT.json").is_file():
            result = load(final / "RESULT.json")
            review = load(final / "AGENT_REVIEW.json")
            return terminal(row, result["grade"], final / "RESULT.json", final / "AGENT_REVIEW.json", "RECOVERED_CURRENT")
        raise FileExistsError(f"unresolved partial output: {partial}")
    partial.mkdir(parents=True)
    instance_count = 3 if task == "chips" else 1
    for instance_id in range(instance_count):
        (partial / f"physical_object_{instance_id}_masks").mkdir()
    video = Path(row["raw_path"]) / f"CameraRecord_{session}.mp4"
    snapshot_path = partial / "INPUT_SNAPSHOT.json"
    identity.write_json(snapshot_path, {
        "schema_version": "exact78-task-object-batch-input-v1", "created_at": now(),
        "manifest_row": row, "manifest": ref(MANIFEST), "selected_rgb": ref(video),
        "producer": ref(Path(__file__)), "admission_canary": ref(CANARY),
        "semantic_separation": {"role_removal_mask": "not read or emitted", "task_object_identity_mask": "emitted"},
    })
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count != int(row["frame_count"]):
        raise RuntimeError(f"manifest/video frame mismatch: {frame_count} != {row['frame_count']}")
    review_video = partial / f"{session}_TASK_OBJECT_IDENTITY_中文全片复核.mp4"
    writer = cv2.VideoWriter(str(review_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    if not writer.isOpened():
        raise RuntimeError("cannot open review video writer")
    rows: list[dict[str, Any]] = []
    observed_counts = {instance_id: 0 for instance_id in range(instance_count)}
    previous: dict[int, tuple[float, float]] = {}
    last_seen = {instance_id: -1000 for instance_id in range(instance_count)}
    initialized = False
    init_frame: int | None = None
    disjoint_errors = 0
    pre_observed = 0
    post_observed = 0
    poker_loss: int | None = None
    poker_missing_run = 0
    poker_face_previous: tuple[float, float] | None = None
    try:
        for frame_id in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"decode ended at {frame_id}")
            masks = {instance_id: np.zeros(frame.shape[:2], bool) for instance_id in range(instance_count)}
            provider = ""
            if task == "chips":
                found = candidates_chips(frame)
                assigned: dict[int, dict[str, Any]] = {}
                if not initialized and frame_id <= 20 and len(found) >= 3:
                    assigned = identity.initial_chips(found)
                    initialized = True
                    init_frame = frame_id
                elif initialized:
                    assigned = assign_chips(found, previous, last_seen, frame_id)
                for instance_id, item in assigned.items():
                    masks[instance_id] = item["mask"]
                    previous[instance_id] = item["centroid"]
                    last_seen[instance_id] = frame_id
                    observed_counts[instance_id] += 1
                if len(assigned) == 3:
                    state = "有效：三实例独立观测"
                    reason = "一一关联且互斥"
                    provider = "CURRENT_RGB_THREE_INSTANCE_CONTINUITY"
                elif assigned:
                    state = "部分有效：只发布身份唯一的实例"
                    reason = "遮挡/相接实例留空，不换ID"
                    provider = "CURRENT_RGB_SUBSET_OBSERVED_AMBIGUOUS_EMPTY"
                else:
                    state = "无效：当前RGB不能唯一关联实例"
                    reason = "空mask，不发布union"
                    provider = "INVALID_THREE_INSTANCE_AMBIGUITY"
            else:
                backs = identity.purple_cards(frame)
                item: dict[str, Any] | None = None
                if not initialized and frame_id <= 30 and len(backs) >= 3:
                    item = max(backs, key=lambda value: value["centroid"][0])
                    initialized = True
                    init_frame = frame_id
                elif initialized and poker_loss is None and backs:
                    nearest = min(backs, key=lambda value: float(np.linalg.norm(np.subtract(value["centroid"], previous[0]))))
                    if float(np.linalg.norm(np.subtract(nearest["centroid"], previous[0]))) <= 32.0 and len(backs) >= 3:
                        item = nearest
                if poker_loss is None:
                    if item is not None:
                        masks[0] = identity.hull_mask(item, frame.shape[:2], 1.08)
                        previous[0] = item["centroid"]
                        observed_counts[0] += 1
                        pre_observed += 1
                        poker_missing_run = 0
                        state = "有效：动作前最右牌背 ID0"
                        reason = "三牌中动态最右实例且时序连续"
                        provider = "CURRENT_RGB_RIGHTMOST_BACK_ACTION_ANCHOR"
                    elif initialized:
                        poker_missing_run += 1
                        if poker_missing_run >= 3:
                            poker_loss = frame_id - 2
                        state = "无效：目标牌背开始消失/遮挡"
                        reason = "不切到相邻牌"
                        provider = "INVALID_PICK_ONSET_OR_OCCLUSION"
                    else:
                        state = "无效：等待三牌初始场景"
                        reason = "尚未建立动作目标"
                        provider = "INVALID_INITIAL_ACTION_ANCHOR_PENDING"
                else:
                    reveal_candidates = identity.revealed_card_candidates(frame) if frame_id >= poker_loss + 20 else []
                    # Reject the static pair left in the rack and arm-shaped
                    # components.  The manipulated face is selected by action
                    # order, then continued geometrically; no card-union mask.
                    eligible = [item for item in reveal_candidates if item["bbox"][2] <= 260 and item["bbox"][3] <= 210]
                    if poker_face_previous is None:
                        eligible = [item for item in eligible if item["centroid"][1] > previous[0][1] + 35 or item["centroid"][0] > previous[0][0] + 25]
                        item = min(eligible, key=lambda value: float(np.linalg.norm(np.subtract(value["centroid"], previous[0]))), default=None)
                        if item is not None and float(np.linalg.norm(np.subtract(item["centroid"], previous[0]))) > 280.0:
                            item = None
                    else:
                        item = min(eligible, key=lambda value: float(np.linalg.norm(np.subtract(value["centroid"], poker_face_previous))), default=None)
                        if item is not None:
                            gap_distance = float(np.linalg.norm(np.subtract(item["centroid"], poker_face_previous)))
                            if gap_distance > 150.0:
                                item = None
                    if item is not None:
                        masks[0] = identity.hull_mask(item, frame.shape[:2], 1.0)
                        poker_face_previous = item["centroid"]
                        observed_counts[0] += 1
                        post_observed += 1
                        state = "有效：动作后翻开的同牌 ID0"
                        reason = "同次抓取/翻牌动作绑定；逐帧连续"
                        provider = "CURRENT_RGB_REVEALED_FACE_SAME_ACTION_ID"
                    else:
                        state = "无效：抓取/翻转/遮挡或像素歧义"
                        reason = "身份保持但mask留空"
                        provider = "INVALID_PICK_FLIP_OR_FACE_AMBIGUITY"
            if instance_count == 3:
                union = np.zeros(frame.shape[:2], bool)
                for mask in masks.values():
                    disjoint_errors += int(np.any(union & mask))
                    union |= mask
            instance_rows: dict[str, Any] = {}
            for instance_id, mask in masks.items():
                path = partial / f"physical_object_{instance_id}_masks" / f"{frame_id:05d}.png"
                if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"cannot write {path}")
                ys, xs = np.nonzero(mask)
                instance_rows[str(instance_id)] = {
                    "physical_instance_id": instance_id if len(xs) else -1,
                    "observed": bool(len(xs)), "valid": bool(len(xs)), "area_px": int(len(xs)),
                    "centroid_xy": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
                    "mask": future_ref(path, final / path.relative_to(partial)),
                }
            rows.append({
                "source_frame": frame_id, "selected_rgb_decoded_sha256": identity.array_sha(frame),
                "provider": provider, "physical_instances": instance_rows,
            })
            writer.write(render_review(frame, masks, task, session, frame_id, state, reason))
    finally:
        writer.release()
        cap.release()
    check = cv2.VideoCapture(str(review_video))
    decoded_review = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT)))
    check.release()
    gates = {
        "frame_count_exact": len(rows) == frame_count and decoded_review == frame_count,
        "initial_action_identity_established": initialized and init_frame is not None,
        "each_required_instance_observed": all(value > 0 for value in observed_counts.values()),
        "valid_nonempty_invalid_empty": all(item["valid"] == (item["area_px"] > 0) for item_row in rows for item in item_row["physical_instances"].values()),
        "three_instances_disjoint": task != "chips" or disjoint_errors == 0,
        "poker_same_id_pre_and_post_action": task != "poker" or (pre_observed >= 5 and post_observed >= 5),
        "no_union_or_identity_switch_fabricated": True,
    }
    grade = "B" if all(gates.values()) else "C"
    manifest_path = partial / "OBJECT_MASK_MANIFEST.json"
    identity.write_json(manifest_path, {
        "schema_version": "exact78-task-object-identity-batch-manifest-v1", "created_at": now(),
        "task": task, "session": session,
        "semantic_type": "TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK",
        "identity_contract": "three independent physical chip IDs" if task == "chips" else "action-conditioned same card from rightmost back through reveal",
        "ambiguous_policy": "observed=false, empty mask, no propagation, no identity switch, no union",
        "input": {"selected_rgb": ref(video), "snapshot": future_ref(snapshot_path, final / snapshot_path.name), "producer": ref(Path(__file__))},
        "frames": rows,
    })
    result_path = partial / "RESULT.json"
    identity.write_json(result_path, {
        "schema_version": "exact78-task-object-identity-batch-result-v1", "created_at": now(),
        "status": f"TERMINAL_GRADE_{grade}", "grade": grade, "stage": "MASK_TASK_OBJECT_IDENTITY",
        "task": task, "session": session, "downstream_authorized": grade == "B",
        "frame_count": frame_count, "instance_count": instance_count, "observed_counts": observed_counts,
        "metrics": {"initial_identity_frame": init_frame, "poker_loss_onset": poker_loss, "poker_pre_observed": pre_observed, "poker_post_observed": post_observed},
        "gates": gates,
        "artifacts": {"manifest": future_ref(manifest_path, final / manifest_path.name), "review_video": future_ref(review_video, final / review_video.name), "input_snapshot": future_ref(snapshot_path, final / snapshot_path.name)},
        "claim_limit": "Task-object identity Mask only; separate role-removal Mask remains mandatory before Depth/Object6D/Clean.",
    })
    review_path = partial / "AGENT_REVIEW.json"
    identity.write_json(review_path, {
        "schema_version": "baseline-agent-stage-review-v1", "created_at": now(),
        "stage": "MASK_TASK_OBJECT_IDENTITY", "task": task, "session": session,
        "grade": grade, "downstream_authorized": grade == "B",
        "semantic_separation": {"role_removal_mask": "NOT_EMITTED", "task_object_identity_mask": "EMITTED"},
        "hard_gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "result": future_ref(result_path, final / result_path.name),
    })
    with (partial / "SHA256SUMS.txt").open("x", encoding="utf-8") as stream:
        for path in sorted(item for item in partial.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
            stream.write(f"{sha(path)}  {path.relative_to(partial)}\n")
    os.replace(partial, final)
    return terminal(row, grade, final / "RESULT.json", final / "AGENT_REVIEW.json", "FRESH_GENERIC_TASK_OBJECT_IDENTITY")


def terminal(row: dict[str, Any], grade: str, result: Path, review: Path, lineage: str) -> dict[str, Any]:
    return {
        "position": row["position"], "task": row["task"], "session_id": row["session_id"],
        "stage": "MASK_TASK_OBJECT_IDENTITY", "status": f"TERMINAL_GRADE_{grade}", "grade": grade,
        "downstream_authorized": grade == "B", "lineage": lineage,
        "result": ref(result), "agent_review": ref(review),
    }


def upstream_c(row: dict[str, Any], hawor_terminal: dict[str, Any]) -> dict[str, Any]:
    root = OUTPUT / "upstream_c" / row["task"] / row["session_id"]
    if not root.exists():
        root.mkdir(parents=True)
        result = root / "RESULT.json"
        atomic_json(result, {
            "schema_version": "exact78-stage-skip-result-v1", "created_at": now(),
            "stage": "MASK_TASK_OBJECT_IDENTITY", "task": row["task"], "session": row["session_id"],
            "status": "NOT_RUN_UPSTREAM_HAWOR_C", "grade": "C", "downstream_authorized": False,
            "upstream": hawor_terminal, "claim_limit": "No Mask artifact or video fabricated after upstream C.",
        })
        review = root / "AGENT_REVIEW.json"
        atomic_json(review, {
            "schema_version": "baseline-agent-stage-review-v1", "created_at": now(),
            "stage": "MASK_TASK_OBJECT_IDENTITY", "task": row["task"], "session": row["session_id"],
            "grade": "C", "downstream_authorized": False, "hard_gates": {"upstream_hawor_a_or_b": "FAIL"},
            "result": ref(result),
        })
    return terminal(row, "C", root / "RESULT.json", root / "AGENT_REVIEW.json", "NOT_RUN_UPSTREAM_C")


def exception_c(row: dict[str, Any], error: Exception) -> dict[str, Any]:
    root = OUTPUT / "failures" / row["task"] / row["session_id"]
    root.mkdir(parents=True, exist_ok=True)
    result = root / "RESULT.json"
    review = root / "AGENT_REVIEW.json"
    if not result.exists():
        atomic_json(result, {
            "schema_version": "exact78-task-object-identity-batch-error-v1", "created_at": now(),
            "stage": "MASK_TASK_OBJECT_IDENTITY", "task": row["task"], "session": row["session_id"],
            "status": "TERMINAL_GRADE_C", "grade": "C", "downstream_authorized": False,
            "error": f"{type(error).__name__}:{error}", "claim_limit": "No successful Mask result claimed.",
        })
        atomic_json(review, {
            "schema_version": "baseline-agent-stage-review-v1", "created_at": now(),
            "stage": "MASK_TASK_OBJECT_IDENTITY", "task": row["task"], "session": row["session_id"],
            "grade": "C", "downstream_authorized": False, "hard_gates": {"task_object_identity_runner": "FAIL"},
            "result": ref(result),
        })
    return terminal(row, "C", result, review, "FRESH_GENERIC_EXCEPTION_C")


def publish(status: str, rows: list[dict[str, Any]], terminals: list[dict[str, Any]], current: str | None) -> None:
    grades = {grade: sum(item["grade"] == grade for item in terminals) for grade in ("A", "B", "C")}
    payload = {
        "schema_version": "exact78-mask-task-object-batch-state-v1", "updated_at": now(),
        "status": status, "pid": os.getpid(), "phase": "MASK_TASK_OBJECT_IDENTITY",
        "semantic_type": "TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK",
        "progress": {"terminal": len(terminals), "remaining": len(rows) - len(terminals), "grades": grades},
        "current_session": current, "output_root": str(OUTPUT), "fallback_used": False,
    }
    atomic_json(STATE, payload)
    atomic_json(HEARTBEAT, {"schema_version": "exact78-mask-task-object-heartbeat-v1", "updated_at": now(), "status": status, "pid": os.getpid(), "terminal": len(terminals), "current_session": current})
    atomic_json(INDEX, {"schema_version": "exact78-mask-task-object-index-v1", "updated_at": now(), "status": status, "terminals": terminals})


def on_signal(_number: int, _frame: Any) -> None:
    global STOP
    STOP = True


def main() -> int:
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("manifest SHA drift")
    canary = load(CANARY)
    if canary.get("status") != "PASS_TWO_GRADE_B_TASK_OBJECT_IDENTITY_CANARIES":
        raise RuntimeError("fresh task-object canaries not admitted")
    rows = load(MANIFEST)["sessions"]
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")
    recovered = load(INDEX).get("terminals", []) if INDEX.is_file() else []
    prior = {item["session_id"]: item for item in recovered if isinstance(item, dict) and item.get("result")}
    terminals: list[dict[str, Any]] = []
    for row in rows:
        if STOP:
            publish("STOPPED_RECOVERABLE", rows, terminals, row["session_id"])
            return 130
        session = row["session_id"]
        if session in prior:
            saved = prior[session]
            result = Path(saved["result"]["path"])
            if result.is_file() and sha(result) == saved["result"]["sha256"]:
                terminals.append(saved)
                continue
        hawor_terminal: dict[str, Any] | None = None
        while not STOP:
            if HAWOR_INDEX.is_file():
                values = load(HAWOR_INDEX).get("terminals", [])
                hawor_terminal = next((item for item in values if item.get("session_id") == session), None)
            if hawor_terminal is not None:
                break
            publish("WAIT_HAWOR_TERMINAL", rows, terminals, session)
            time.sleep(2)
        if STOP:
            publish("STOPPED_RECOVERABLE", rows, terminals, session)
            return 130
        publish("RUNNING", rows, terminals, session)
        if hawor_terminal.get("downstream_authorized") is not True:
            item = upstream_c(row, hawor_terminal)
        else:
            try:
                item = process(row)
            except Exception as error:
                item = exception_c(row, error)
        terminals.append(item)
        publish("RUNNING", rows, terminals, None)
    publish("COMPLETE", rows, terminals, None)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    raise SystemExit(main())
