#!/usr/bin/env python3
"""Persistent SAM3.1-only Mask worker for prepared 0915 physical-left RGB."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import cv2
import numpy as np
import torch

from chaoyang.ops import run_0915_leftmono_sam31_masks_v1 as legacy
from chaoyang.pipeline.sam31_0915_prompt_contract_v2 import build_prompt_plan
from chaoyang.pipeline.visual_role_mask_v1 import ROLE_NAMES, validate_manifest


EXPECTED_SESSIONS = 220
EXPECTED_FRAMES = 58_686
CHECKPOINT = legacy.CHECKPOINT
CHECKPOINT_SHA256 = legacy.CHECKPOINT_SHA256
CODE_ROOT = legacy.CODE_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def select_distinct_human_ids(
    masks: np.ndarray, ids: np.ndarray, prompts: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, Any]]:
    chosen: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    used: set[int] = set()
    for prompt in prompts:
        role = prompt["role"]
        positives = np.asarray(prompt["positive_points_xy"], np.float64)
        candidates = []
        for index, mask in enumerate(masks):
            raw_id = int(ids[index])
            area = int(mask.sum())
            if not 2_000 <= area <= int(mask.size * 0.45):
                continue
            distances = [legacy.anchor_distance(mask, point) for point in positives]
            candidates.append({
                "raw_id": raw_id,
                "area_pixels": area,
                "positive_covered": int(sum(distance == 0 for distance in distances)),
                "median_positive_distance_px": float(np.median(distances)),
            })
        eligible = sorted(
            (row for row in candidates
             if row["raw_id"] not in used
             and (row["positive_covered"] >= 2
                  or row["median_positive_distance_px"] <= 35.0)),
            key=lambda row: (
                -row["positive_covered"], row["median_positive_distance_px"],
                -row["area_pixels"], row["raw_id"],
            ),
        )
        if eligible:
            chosen[role] = eligible[0]["raw_id"]
            used.add(eligible[0]["raw_id"])
        evidence[role] = {"candidates": candidates,
                          "chosen_raw_id": chosen.get(role)}
    return chosen, evidence


def _normalised_prompt(
    prompt: dict[str, Any], width: int, height: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
    positive = prompt["positive_points_xy"]
    negative = prompt["negative_points_xy"]
    points = [*positive, *negative]
    labels = [1] * len(positive) + [0] * len(negative)
    x, y, w, h = prompt["box_xywh"]
    return (
        torch.tensor([[px / width, py / height] for px, py in points],
                     dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int32),
        [[x / width, y / height, w / width, h / height]],
    )


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def _propagate_to_directories(
    model: Any, state: Any, *, anchor: int, frame_count: int,
    height: int, width: int, id_to_directory: dict[int, Path],
) -> dict[int, int]:
    present = {raw_id: 0 for raw_id in id_to_directory}
    written: dict[int, set[int]] = {raw_id: set() for raw_id in id_to_directory}
    directions = [(False, frame_count - anchor)]
    if anchor:
        directions.append((True, anchor + 1))
    for reverse, maximum in directions:
        for frame_index, outputs in model.propagate_in_video(
            inference_state=state, start_frame_idx=anchor,
            max_frame_num_to_track=maximum, reverse=reverse,
            output_prob_thresh=0.5,
        ):
            masks, _scores, ids = legacy.normalize(outputs, height, width)
            by_id = {int(value): masks[index] for index, value in enumerate(ids)}
            frame = int(frame_index)
            for raw_id, directory in id_to_directory.items():
                mask = by_id.get(raw_id, np.zeros((height, width), bool))
                _write_mask(directory / f"{frame:05d}.png", mask)
                written[raw_id].add(frame)
    for raw_id, directory in id_to_directory.items():
        for frame in range(frame_count):
            path = directory / f"{frame:05d}.png"
            if frame not in written[raw_id]:
                _write_mask(path, np.zeros((height, width), bool))
            else:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.any(mask):
                    present[raw_id] += 1
    return present


def track_hands(
    model: Any, frames: Path, plan: dict[str, Any], output: Path,
    frame_count: int, height: int, width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor = int(plan["anchor_frame"])
    state = model.init_state(
        resource_path=str(frames), offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    try:
        _, initial = model.add_prompt(
            inference_state=state, frame_idx=anchor,
            text_str="a person's hand and forearm", output_prob_thresh=0.5,
        )
        masks, _scores, ids = legacy.normalize(initial, height, width)
        chosen, selection = select_distinct_human_ids(
            masks, ids, plan["hand_spatial_prompts"],
        )
        if len(chosen) != 2:
            # On high-view 0915 frames the neutral text head can legitimately
            # emit both arms as one connected person instance.  That is not a
            # valid bilateral identity seed.  Fall back to one independently
            # initialised SAM3.1 state per anatomical side, with the frozen
            # HaWoR box/positive points and opposite-hand negative points.
            # The states never share an object ID or memory, so this does not
            # manufacture a tracker/controller identity or split model pixels
            # after inference.
            state.clear()
            instances: list[dict[str, Any]] = []
            present_by_role: dict[str, int] = {}
            fallback: dict[str, Any] = {}
            for side_index, prompt in enumerate(plan["hand_spatial_prompts"]):
                role = prompt["role"]
                side_state = model.init_state(
                    resource_path=str(frames), offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                try:
                    points, labels, boxes = _normalised_prompt(
                        prompt, width, height,
                    )
                    _, box_seeded = model.add_prompt(
                        inference_state=side_state, frame_idx=anchor,
                        text_str="a person's hand and forearm",
                        boxes_xywh=boxes,
                        box_labels=[1], clear_old_boxes=True,
                        output_prob_thresh=0.5,
                    )
                    masks, _scores, ids = legacy.normalize(
                        box_seeded, height, width,
                    )
                    selected, side_evidence = select_distinct_human_ids(
                        masks, ids, [prompt],
                    )
                    raw_id = selected.get(role)
                    if raw_id is None:
                        raise RuntimeError(
                            f"SAM3.1 independent spatial hand seed failed: "
                            f"{role}: {side_evidence}"
                        )
                    _, refined = model.add_prompt(
                        inference_state=side_state, frame_idx=anchor,
                        text_str=None, points=points, point_labels=labels,
                        clear_old_points=True, boxes_xywh=None,
                        box_labels=None, clear_old_boxes=False,
                        obj_id=raw_id, rel_coordinates=True,
                        output_prob_thresh=0.5,
                    )
                    masks, _scores, ids = legacy.normalize(
                        refined, height, width,
                    )
                    anchor_by_id = {
                        int(value): masks[index]
                        for index, value in enumerate(ids)
                    }
                    directory = output / "masks" / role / role
                    _write_mask(
                        directory / f"{anchor:05d}.png",
                        anchor_by_id.get(raw_id, np.zeros((height, width), bool)),
                    )
                    counts = _propagate_to_directories(
                        model, side_state, anchor=anchor,
                        frame_count=frame_count, height=height, width=width,
                        id_to_directory={raw_id: directory},
                    )
                    instances.append({
                        "instance_id": role, "role": role,
                        "physical_identity_policy": "SIDE_LOCKED_HUMAN_ROLE",
                        "mask_directory": str(directory.relative_to(output)),
                    })
                    present_by_role[role] = counts[raw_id]
                    fallback[role] = {
                        "raw_id": raw_id,
                        "selection": side_evidence[role],
                    }
                finally:
                    side_state.clear()
            return instances, {
                "prompt": "a person's hand and forearm",
                "selection": selection,
                "chosen_raw_ids": {
                    role: evidence["raw_id"]
                    for role, evidence in fallback.items()
                },
                "present_frames": present_by_role,
                "route": "INDEPENDENT_SAM31_SPATIAL_STATE_PER_SIDE_AFTER_TEXT_UNION",
                "fallback_evidence": fallback,
            }
        latest = initial
        for prompt in plan["hand_spatial_prompts"]:
            points, labels, boxes = _normalised_prompt(prompt, width, height)
            _, latest = model.add_prompt(
                inference_state=state, frame_idx=anchor, text_str=None,
                points=points, point_labels=labels, clear_old_points=True,
                boxes_xywh=boxes, box_labels=[1], clear_old_boxes=True,
                obj_id=chosen[prompt["role"]], rel_coordinates=True,
                output_prob_thresh=0.5,
            )
        anchor_masks, _anchor_scores, anchor_ids = legacy.normalize(
            latest, height, width,
        )
        anchor_by_id = {
            int(value): anchor_masks[index]
            for index, value in enumerate(anchor_ids)
        }
        directories = {
            raw_id: output / "masks" / role / role
            for role, raw_id in chosen.items()
        }
        for raw_id, directory in directories.items():
            _write_mask(
                directory / f"{anchor:05d}.png",
                anchor_by_id.get(raw_id, np.zeros((height, width), bool)),
            )
        present = _propagate_to_directories(
            model, state, anchor=anchor, frame_count=frame_count,
            height=height, width=width, id_to_directory=directories,
        )
        instances = [{
            "instance_id": role,
            "role": role,
            "physical_identity_policy": "SIDE_LOCKED_HUMAN_ROLE",
            "mask_directory": str(directories[raw_id].relative_to(output)),
        } for role, raw_id in chosen.items()]
        return instances, {
            "prompt": "a person's hand and forearm",
            "selection": selection,
            "chosen_raw_ids": chosen,
            "present_frames": {role: present[raw_id]
                               for role, raw_id in chosen.items()},
        }
    finally:
        state.clear()


def _seed_candidates(
    masks: np.ndarray, scores: np.ndarray, ids: np.ndarray, *,
    human_union: np.ndarray, minimum_area: int, maximum_fraction: float,
    maximum_human_overlap: float,
) -> list[dict[str, Any]]:
    rows = []
    for index, mask in enumerate(masks):
        area = int(mask.sum())
        overlap = int(np.count_nonzero(mask & human_union)) / max(area, 1)
        eligible = bool(
            minimum_area <= area <= int(mask.size * maximum_fraction)
            and overlap <= maximum_human_overlap
        )
        rows.append({
            "raw_id": int(ids[index]), "score": float(scores[index]),
            "area_pixels": area, "human_overlap_fraction": overlap,
            "maximum_human_overlap": maximum_human_overlap,
            "eligible": eligible, **legacy.geometry(mask),
        })
    return rows


def track_text_instances(
    model: Any, frames: Path, *, prompt: str, group: str,
    role_resolver: Any, human_union: np.ndarray, anchor: int,
    output: Path, frame_count: int, height: int, width: int,
    maximum_instances: int, maximum_human_overlap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = model.init_state(
        resource_path=str(frames), offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    try:
        _, initial = model.add_prompt(
            inference_state=state, frame_idx=anchor, text_str=prompt,
            output_prob_thresh=0.5,
        )
        masks, scores, ids = legacy.normalize(initial, height, width)
        rows = _seed_candidates(
            masks, scores, ids, human_union=human_union,
            minimum_area=80, maximum_fraction=0.25,
            maximum_human_overlap=maximum_human_overlap,
        )
        selected = sorted(
            (row for row in rows if row["eligible"]),
            key=lambda row: (-row["score"], -row["area_pixels"], row["raw_id"]),
        )[:maximum_instances]
        directories: dict[int, Path] = {}
        instances: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            raw_id = row["raw_id"]
            role = str(role_resolver(masks[int(np.flatnonzero(ids == raw_id)[0])]))
            if role not in ROLE_NAMES:
                raise RuntimeError(f"role resolver returned invalid role: {role}")
            instance_id = f"{group}_{index:02d}"
            directory = output / "masks" / role / instance_id
            directories[raw_id] = directory
            instances.append({
                "instance_id": instance_id, "role": role,
                "physical_identity_policy": (
                    "SEPARATE_VISIBLE_PHYSICAL_OBJECT"
                    if role == "task_object"
                    else "UNKNOWN_AFTER_EXIT_NEW_ID_UNLESS_REIDENTIFIED"
                ),
                "mask_directory": str(directory.relative_to(output)),
            })
        if directories:
            present = _propagate_to_directories(
                model, state, anchor=anchor, frame_count=frame_count,
                height=height, width=width, id_to_directory=directories,
            )
        else:
            present = {}
        return instances, {
            "prompt": prompt, "seed_candidates": rows,
            "selected_raw_ids": list(directories),
            "present_frames": {str(key): value for key, value in present.items()},
        }
    finally:
        state.clear()


def _nearest_side_role(
    mask: np.ndarray, plan: dict[str, Any], suffix: str,
) -> str:
    distances = []
    for prompt in plan["hand_spatial_prompts"]:
        points = np.asarray(prompt["positive_points_xy"], np.float64)
        distances.append((
            float(np.median([legacy.anchor_distance(mask, point)
                             for point in points])),
            prompt["role"].split("_", 1)[0],
        ))
    return f"{min(distances)[1]}_{suffix}"


def process_session(
    *, model: Any, task: str, session_id: str, prepared_root: Path,
    hawor_root: Path, output_root: Path, frame_count: int,
) -> dict[str, Any]:
    target = output_root / "sessions" / task / session_id
    prior = target / "RESULT.json"
    if prior.is_file():
        return {**json.loads(prior.read_text(encoding="utf-8")),
                "resume_status": "RESUMED"}
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"unreceipted Mask target exists: {target}")
    staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    started = time.time()
    try:
        prepared = prepared_root / "sessions" / task / session_id
        video = prepared / "leftmono/LEFT_MONO_RECTIFIED.mp4"
        hawor = hawor_root / "sessions" / task / session_id / "HAWOR_RAW_MANO21.npz"
        if not video.is_file() or not hawor.is_file():
            raise RuntimeError("prepared RGB or HaWoR artifact missing")
        frames = staging / "input_frames"
        decoded, width, height, fps = legacy.extract_frames(video, frames)
        if decoded != frame_count or (width, height) != (1280, 960):
            raise RuntimeError("dynamic video/HaWoR frame geometry mismatch")
        with np.load(hawor, allow_pickle=False) as archive:
            joints = np.asarray(archive["joints_2d"], np.float64)
            observed = np.asarray(archive["observed"], bool)
        plan = build_prompt_plan(
            joints, observed, task=task, width=width, height=height,
        )
        atomic_json(staging / "PROMPT_PLAN.json", plan)
        instances, human_evidence = track_hands(
            model, frames, plan, staging, frame_count, height, width,
        )
        anchor = int(plan["anchor_frame"])
        human_union = np.zeros((height, width), bool)
        for instance in instances:
            mask = cv2.imread(
                str(staging / instance["mask_directory"] / f"{anchor:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            human_union |= np.asarray(mask, bool)

        text_evidence: dict[str, Any] = {}
        task_prompt = " or ".join(plan["task_object_text_prompts"])
        task_instances, text_evidence["task_object"] = track_text_instances(
            model, frames, prompt=task_prompt, group="task_object",
            role_resolver=lambda _mask: "task_object",
            human_union=human_union, anchor=anchor, output=staging,
            frame_count=frame_count, height=height, width=width,
            maximum_instances=12, maximum_human_overlap=0.35,
        )
        instances.extend(task_instances)
        for group, suffix, maximum, maximum_human_overlap in (
            ("finger_sleeve_attachment", "finger_sleeve_attachment", 12, 1.0),
            ("cable", "cable", 4, 0.65),
        ):
            new, text_evidence[group] = track_text_instances(
                model, frames, prompt=plan["auxiliary_text_prompts"][group],
                group=group,
                role_resolver=lambda mask, suffix=suffix: _nearest_side_role(
                    mask, plan, suffix,
                ),
                human_union=human_union, anchor=anchor, output=staging,
                frame_count=frame_count, height=height, width=width,
                maximum_instances=maximum,
                maximum_human_overlap=maximum_human_overlap,
            )
            instances.extend(new)

        role_manifest = {
            "schema_version": "visual-role-mask-v1",
            "session_id": session_id,
            "frame_count": frame_count,
            "model": {"identity": "SAM3.1",
                      "weight_sha256": CHECKPOINT_SHA256},
            "roles": sorted(ROLE_NAMES),
            "instances": instances,
            "tracker_role_created": False,
            "pico26_consumed": False,
        }
        validate_manifest(role_manifest, root=staging)
        atomic_json(staging / "ROLE_MANIFEST.json", role_manifest)
        human_coverage = {
            role: count / frame_count
            for role, count in human_evidence["present_frames"].items()
        }
        passed = bool(
            len(task_instances) >= 1
            and len(human_coverage) == 2
            and min(human_coverage.values()) >= 0.90
        )
        result = {
            "schema_version": "0915-sam31-mask-session-v2",
            "status": "PASS" if passed else "REJECTED_QUALITY",
            "task": task, "session_id": session_id,
            "frame_count": frame_count,
            "model_identity": "SAM3.1_ONLY_USER_LOCKED",
            "human_coverage": human_coverage,
            "task_object_instance_count": len(task_instances),
            "human_evidence": human_evidence,
            "text_evidence": text_evidence,
            "tracker": "ABSENT_NO_ROLE_CREATED",
            "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
            "inputs": {
                "video": {"path": str(video), "bytes": video.stat().st_size,
                          "sha256": sha256(video)},
                "hawor": {"path": str(hawor), "bytes": hawor.stat().st_size,
                          "sha256": sha256(hawor)},
                "checkpoint": {"path": str(CHECKPOINT),
                               "bytes": CHECKPOINT.stat().st_size,
                               "sha256": CHECKPOINT_SHA256},
            },
            "prompt_plan_sha256": sha256(staging / "PROMPT_PLAN.json"),
            "role_manifest_sha256": sha256(staging / "ROLE_MANIFEST.json"),
            "fps": fps,
            "wall_seconds": time.time() - started,
            "claim_limit": (
                "Fresh SAM3.1-only development masks. Task-object IDs remain "
                "separate, missing/re-entry evidence stays explicit, and no "
                "tracker, controller, PICO hand, pixel ground truth, contact "
                "truth, or deployment authority is created."
            ),
        }
        atomic_json(staging / "RESULT.json", result)
        shutil.rmtree(frames)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return json.loads((target / "RESULT.json").read_text(encoding="utf-8"))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-allowlist", type=Path)
    args = parser.parse_args()
    prepared_manifest_path = args.prepared_manifest.resolve(strict=True)
    hawor_result_path = args.hawor_result.resolve(strict=True)
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    hawor_batch = json.loads(hawor_result_path.read_text(encoding="utf-8"))
    if (prepared.get("session_count") != EXPECTED_SESSIONS
            or prepared.get("frame_count") != EXPECTED_FRAMES
            or hawor_batch.get("session_count") != EXPECTED_SESSIONS):
        raise RuntimeError("0915 prepared/HaWoR cohort identity mismatch")
    allowed = None
    if args.session_allowlist:
        allowlist = json.loads(args.session_allowlist.resolve(strict=True).read_text())
        allowed = {(row["task"], row["session_id"])
                   for row in allowlist["sessions"]}
    prepared_by_id = {
        (row["task"], row["session_id"]): row for row in prepared["results"]
    }
    hawor_by_id = {
        (row["task"], row["session_id"]): row
        for row in hawor_batch["results"]
    }
    identities = list(prepared_by_id)
    if allowed is not None:
        unknown = allowed - set(identities)
        if unknown:
            raise RuntimeError(f"allowlist has unknown sessions: {sorted(unknown)}")
        identities = [identity for identity in identities if identity in allowed]
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter_module = legacy.load_adapter()
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT,
    )
    results: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task, session_id in identities:
            hawor = hawor_by_id.get((task, session_id))
            if hawor is None or hawor.get("status") != "PASS_DEVELOPMENT_HAWOR":
                result = {
                    "schema_version": "0915-sam31-mask-session-v2",
                    "status": "BLOCKED_UPSTREAM", "task": task,
                    "session_id": session_id,
                    "reason": "HAWOR_NOT_PASS",
                }
            else:
                try:
                    result = process_session(
                        model=adapter.model, task=task, session_id=session_id,
                        prepared_root=prepared_manifest_path.parent,
                        hawor_root=hawor_result_path.parent,
                        output_root=output,
                        frame_count=int(prepared_by_id[(task, session_id)]["frame_count"]),
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "schema_version": "0915-sam31-mask-session-v2",
                        "status": "FAILED_RUNTIME", "task": task,
                        "session_id": session_id, "error": repr(exc),
                    }
            results.append(result)
            atomic_json(output / "STATE.json", {
                "schema_version": "0915-sam31-mask-state-v2",
                "state": "RUNNING", "completed": len(results),
                "session_count": len(identities), "results": results,
            })
            print(json.dumps({
                "completed": len(results), "total": len(identities),
                "last": f"{task}/{session_id}", "status": result["status"],
            }, sort_keys=True), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        adapter.predictor.shutdown()
    summary = {
        "schema_version": "0915-sam31-mask-batch-v2",
        "status": "COMPLETED_ALL_TERMINAL",
        "model_identity": "SAM3.1_ONLY_USER_LOCKED",
        "session_count": len(results),
        "full_cohort": allowed is None,
        "passed": sum(row["status"] == "PASS" for row in results),
        "rejected_quality": sum(row["status"] == "REJECTED_QUALITY" for row in results),
        "blocked_upstream": sum(row["status"] == "BLOCKED_UPSTREAM" for row in results),
        "failed_runtime": sum(row["status"] == "FAILED_RUNTIME" for row in results),
        "build_evidence": build_evidence,
        "wall_seconds": time.time() - started,
        "results": results,
        "claim_limit": "SAM3.1-only development Mask; no model selection or challenger authority.",
    }
    atomic_json(output / "BATCH_RESULT.json", summary)
    return 0 if summary["failed_runtime"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
