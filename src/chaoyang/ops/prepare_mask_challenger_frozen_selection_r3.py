#!/usr/bin/env python3
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Prepare the two frozen R3 Mask challenger sets without changing authority.

Only task-object identity rows with an independently pinned current manifest are
materialized for GPU inference.  Role rows fail closed unless all four role
instances have a non-empty seed; task-object prompts must never be relabelled as
role prompts.
"""

import argparse
import json
import os
from pathlib import Path
import secrets
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]

from chaoyang.ops.build_s1_input_preflight_index_v71 import (
    _load_object,
    _write_json_exclusive,
    build_prompt_manifest,
    build_rgb_manifest,
    evidence,
    extract_frames_atomic,
    first_reliable_prompts,
    run_preflight,
    verify_evidence,
)


def _selection_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in selection.get("selections", []):
        if group.get("selection_eligible") is not True:
            continue
        for item in [group.get("canary"), *group.get("regressions", [])]:
            if not isinstance(item, Mapping):
                continue
            rows.append({
                "stage": str(group["stage"]),
                "cluster": str(group["cluster"]),
                "selection_role": str(item["role"]),
                "session_id": str(item["session_id"]),
                "task": str(item["task"]).upper(),
                "grade": str(item["grade"]),
                "result": dict(item["result"]),
            })
    return rows


def _role_seed_status(result: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    manifest_ref = result.get("artifacts", {}).get("frame_manifest")
    if not isinstance(manifest_ref, Mapping):
        return False, {"reason": "MISSING_FOUR_ROLE_FRAME_MANIFEST"}
    manifest_path = verify_evidence(manifest_ref, label="role frame manifest")
    manifest = _load_object(manifest_path)
    roles = ("left_human", "right_human", "left_tracker", "right_tracker")
    seeds: dict[str, Any] = {}
    for role in roles:
        found = None
        for frame in manifest.get("frames", []):
            record = frame.get("role_masks", {}).get(role)
            if not isinstance(record, Mapping):
                continue
            path = verify_evidence(record, label=f"{role} seed candidate")
            # The all-zero PNG used by the predecessor is 2837 bytes at 1280x960.
            # Decode through OpenCV to avoid treating a non-empty file as a mask.
            import cv2

            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and int((mask > 0).sum()) > 0:
                found = {"source_frame": int(frame["source_frame"]), "mask": dict(record)}
                break
        seeds[role] = found
    missing = [role for role, value in seeds.items() if value is None]
    return not missing, {
        "required_roles": list(roles),
        "seed_frames": seeds,
        "missing_roles": missing,
        "reason": None if not missing else "FOUR_ROLE_CAUSAL_PROMPT_SET_INCOMPLETE",
    }


def _object_manifest_from_result(result: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_ref = result.get("artifacts", {}).get("manifest")
    if not isinstance(manifest_ref, Mapping):
        raise ValueError("MISSING_OBJECT_MASK_MANIFEST")
    manifest_path = verify_evidence(manifest_ref, label="object mask manifest")
    return manifest_path, _load_object(manifest_path)


def build(*, selection_path: Path, output_root: Path, executor_epoch: int) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable output already exists: {output_root}")
    output_root.mkdir(parents=True)
    selection = _load_object(selection_path)
    selection_ref = evidence(selection_path)
    selected_rows = _selection_rows(selection)
    audit_manifest = {
        "schema_version": "mask-challenger-frozen-audit-r3-v1",
        "immutable": True,
        "authority": False,
        "gold_accuracy_authorized": False,
        "source_selection": selection_ref,
        "rows": [
            {
                "session_id": row["session_id"],
                "stage": row["stage"],
                "cluster": row["cluster"],
                "selection_role": row["selection_role"],
                "source_grade": row["grade"],
                "source_result": row["result"],
            }
            for row in selected_rows
        ],
        "claim_limit": (
            "Frozen predecessor-derived 1C+2A/B development audit; not Gold labels "
            "and not an accuracy denominator."
        ),
    }
    audit_path = output_root / "FROZEN_DEVELOPMENT_AUDIT.json"
    _write_json_exclusive(audit_path, audit_manifest)
    audit_ref = evidence(audit_path)
    fencing_token = secrets.token_hex(24)
    rows: list[dict[str, Any]] = []
    first_large_sha = True

    for selected in selected_rows:
        row: dict[str, Any] = {
            **selected,
            "status": "BLOCKED_PREREQ",
            "blocked_reason": None,
            "rgb_manifest": None,
            "prompt_manifest": None,
            "preflight_result": None,
            "inference_eligible": False,
        }
        try:
            result_path = verify_evidence(selected["result"], label="selected terminal result")
            result = _load_object(result_path)
            if str(result.get("session")) != selected["session_id"]:
                raise ValueError("SELECTED_RESULT_SESSION_MISMATCH")
            if selected["stage"] == "ROLE_MASK":
                ready, detail = _role_seed_status(result)
                row["role_prompt_preflight"] = detail
                if not ready:
                    raise ValueError(str(detail["reason"]))
                # A four-role GPU adapter is not part of the reviewed S1 object
                # adapter.  Do not silently coerce four roles into object IDs.
                raise ValueError("FOUR_ROLE_CHALLENGER_ADAPTER_NOT_REVIEWED")
            if selected["stage"] != "OBJECT_MASK":
                raise ValueError("UNSUPPORTED_SELECTED_STAGE")

            object_path, object_manifest = _object_manifest_from_result(result)
            if str(object_manifest.get("session")) != selected["session_id"]:
                raise ValueError("OBJECT_MANIFEST_SESSION_MISMATCH")
            frame_count = len(object_manifest.get("frames", []))
            if frame_count != int(result.get("frame_count", -1)):
                raise ValueError("OBJECT_MANIFEST_FRAME_COUNT_MISMATCH")
            prompts, error = first_reliable_prompts(
                object_manifest=object_manifest, task=selected["task"]
            )
            if error:
                raise ValueError(error)
            for prompt in prompts:
                for event in prompt["prompt_events"]:
                    event["source_semantics"] = (
                        "FIRST_CURRENT_BASELINE_OBSERVED_VALID_MASK_"
                        "DEVELOPMENT_SEED_NOT_GOLD"
                    )
            source_video = object_manifest.get("input", {}).get("selected_rgb")
            if not isinstance(source_video, Mapping):
                # Current object manifests keep the selected video in INPUT_SNAPSHOT.
                snapshot_ref = result.get("artifacts", {}).get("input_snapshot")
                snapshot_path = verify_evidence(snapshot_ref, label="object input snapshot")
                snapshot = _load_object(snapshot_path)
                source_video = snapshot.get("selected_rgb")
            if not isinstance(source_video, Mapping):
                raise ValueError("MISSING_SELECTED_RGB_EVIDENCE")
            video_path = verify_evidence(source_video, label="selected RGB")

            input_dir = output_root / "inputs" / selected["session_id"] / "R7_3"
            input_dir.mkdir(parents=True)
            frame_dir = extract_frames_atomic(video_path, input_dir / "frames", frame_count)
            rgb = build_rgb_manifest(
                session_id=selected["session_id"],
                frame_directory=frame_dir,
                frame_count=frame_count,
                source_video=source_video,
                source_kind="FRESH_LOSSLESS_DECODE_FROM_PINNED_SELECTED_RGB",
            )
            rgb["artifact_revision"] = "R7_3"
            rgb["frozen_selection"] = selection_ref
            prompt = build_prompt_manifest(
                session_id=selected["session_id"],
                task=selected["task"],
                prompts=prompts,
                audit_evidence=audit_ref,
                object_evidence=evidence(object_path),
            )
            prompt["artifact_revision"] = "R7_3"
            prompt["selection_role"] = selected["selection_role"]
            prompt["source_grade"] = selected["grade"]
            prompt["seed_is_gold"] = False
            rgb_path = input_dir / "CAUSAL_RGB_MANIFEST.json"
            prompt_path = input_dir / "CAUSAL_PROMPT_MANIFEST.json"
            _write_json_exclusive(rgb_path, rgb)
            _write_json_exclusive(prompt_path, prompt)
            row["rgb_manifest"] = evidence(rgb_path)
            row["prompt_manifest"] = evidence(prompt_path)

            preflight = run_preflight(
                rgb_manifest=rgb_path,
                prompt_manifest=prompt_path,
                output_root=output_root / "cpu_preflight",
                session_id=selected["session_id"],
                executor_epoch=executor_epoch,
                fencing_token=fencing_token,
                verify_large_sha=first_large_sha,
            )
            first_large_sha = False
            row["preflight_result"] = evidence(preflight)
            preflight_value = _load_object(preflight)
            row["status"] = str(preflight_value.get("payload", {}).get("status"))
            row["inference_eligible"] = row["status"] == "PASSED"
            if not row["inference_eligible"]:
                row["blocked_reason"] = "CPU_BACKEND_OR_INPUT_PREFLIGHT_FAILED"
        except Exception as exc:
            row["blocked_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    payload = {
        "schema_version": "mask-challenger-bounded-input-preflight-r3-v1",
        "status": "PASSED_WITH_LOCAL_BLOCKERS",
        "selection": selection_ref,
        "frozen_development_audit": audit_ref,
        "current_baseline": "SAM3.1",
        "challengers": ["SAM2.1", "Cutie"],
        "executor_epoch": executor_epoch,
        "fencing_token_sha256": __import__("hashlib").sha256(fencing_token.encode()).hexdigest(),
        "rows": rows,
        "counts": {
            "selected_rows": len(rows),
            "inference_eligible": sum(bool(row["inference_eligible"]) for row in rows),
            "blocked_prereq": sum(not bool(row["inference_eligible"]) for row in rows),
        },
        "authority_promoted": False,
        "claim_limit": (
            "Frozen 1C+2A/B input and asset preflight only. Role prompts are not "
            "substituted by task-object prompts; no Gold accuracy or Mask authority."
        ),
    }
    _write_json_exclusive(output_root / "INPUT_PREFLIGHT.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--executor-epoch", type=int, default=1)
    args = parser.parse_args()
    result = build(
        selection_path=args.selection,
        output_root=args.output_root,
        executor_epoch=args.executor_epoch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
