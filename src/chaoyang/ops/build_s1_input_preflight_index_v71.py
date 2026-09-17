#!/usr/bin/env python3
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Freeze complete causal RGB/prompt inputs and run bounded CPU preflights.

The builder consumes only the immutable S1 re-entry audit selection.  It never
runs SAM/Cutie inference.  A prompt is accepted only from an upstream Grade-B
task-object identity manifest.  Each physical Chips instance receives its own
first observed mask and no instance union is permitted.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_challenger_v71 import file_sha256


DEFAULT_AUDIT = ROOT / (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/"
    "s1_reentry_audit/FROZEN_REENTRY_AUDIT_FRAMES_V71.json"
)
DEFAULT_OUTPUT = ROOT / (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/"
    "s1_input_preflight"
)
LEGACY_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1"
CLAIM_LIMIT = (
    "CPU-only immutable input closure and backend asset preflight for the frozen S1 "
    "development audit sessions. No inference was executed; masks are modal prompts, "
    "not amodal/Contact/Object6D/Gold/physical truth."
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def verify_evidence(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label}: missing absolute file: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label}: byte mismatch: {path}")
    if file_sha256(path) != record.get("sha256"):
        raise ValueError(f"{label}: SHA mismatch: {path}")
    return path


def audit_sessions(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    if audit.get("immutable") is not True or audit.get("authority") is not False:
        raise ValueError("audit selection is not immutable development evidence")
    grouped: dict[str, dict[str, Any]] = {}
    for row in audit.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        session_id = str(row["session_id"])
        item = grouped.setdefault(
            session_id,
            {
                "session_id": session_id,
                "task": str(row["task"]).upper(),
                "source_manifests": set(),
                "object_mask_grades": set(),
                "groups": set(),
            },
        )
        item["source_manifests"].add(str(row["source_manifest"]["path"]))
        item["object_mask_grades"].add(str(row["object_mask_grade"]))
        item["groups"].add(str(row["group_id"]))
    result = []
    for item in grouped.values():
        item["source_manifests"] = sorted(item["source_manifests"])
        item["object_mask_grades"] = sorted(item["object_mask_grades"])
        item["groups"] = sorted(item["groups"])
        result.append(item)
    return sorted(result, key=lambda item: item["session_id"])


def _expected_frame_names(frame_count: int) -> list[str]:
    return [f"{frame_id:05d}.png" for frame_id in range(frame_count)]


def discover_complete_frame_directory(
    *, task_lower: str, session_id: str, frame_count: int, legacy_root: Path = LEGACY_ROOT
) -> Path | None:
    candidates: list[Path] = []
    for stage in sorted(legacy_root.glob("mask_role_removal_bounded_v2*")):
        candidate = stage / task_lower / session_id / "input_frames"
        if candidate.is_dir():
            candidates.append(candidate)
    expected = _expected_frame_names(frame_count)
    complete = []
    for candidate in candidates:
        names = sorted(
            entry.name
            for entry in os.scandir(candidate)
            if entry.is_file(follow_symlinks=False)
        )
        if names == expected:
            complete.append(candidate.resolve())
    if len(complete) > 1:
        # Identical paths after resolve are harmless; distinct closures are ambiguous.
        unique = sorted(set(complete))
        if len(unique) > 1:
            raise ValueError(f"multiple complete RGB frame closures: {unique}")
        complete = unique
    return complete[0] if complete else None


def extract_frames_atomic(video: Path, destination: Path, frame_count: int) -> Path:
    staging = destination.parent / f".{destination.name}.staging.{os.getpid()}"
    if destination.exists() or staging.exists():
        raise FileExistsError(f"fresh immutable extraction required: {destination}")
    staging.mkdir(parents=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror",
        "-i", str(video), "-map", "0:v:0", "-vsync", "0",
        "-start_number", "0", str(staging / "%05d.png"),
    ]
    try:
        subprocess.run(command, check=True)
        names = sorted(item.name for item in staging.iterdir() if item.is_file())
        if names != _expected_frame_names(frame_count):
            raise ValueError(
                f"decoded frame closure mismatch: expected={frame_count}, actual={len(names)}"
            )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination.resolve()


def build_rgb_manifest(
    *, session_id: str, frame_directory: Path, frame_count: int,
    source_video: Mapping[str, Any], source_kind: str,
) -> dict[str, Any]:
    frames = []
    for frame_id in range(frame_count):
        frame_path = frame_directory / f"{frame_id:05d}.png"
        if not frame_path.is_file():
            raise ValueError(f"missing RGB frame {frame_id}: {frame_path}")
        frames.append({"frame_id": frame_id, **evidence(frame_path)})
    return {
        "schema_version": "causal-rgb-manifest-v71",
        "artifact_id": f"S1_CAUSAL_RGB.{session_id}",
        "artifact_revision": "R7_1",
        "session_id": session_id,
        "frozen": True,
        "execution_mode": "CAUSAL_PROCESSING",
        "source_video": dict(source_video),
        "frame_source_kind": source_kind,
        "frame_order_contract": "COMPLETE_SESSION_STRICTLY_INCREASING_0_TO_N_MINUS_1",
        "future_observation_allowed": False,
        "frames": frames,
    }


def first_reliable_prompts(
    *, object_manifest: Mapping[str, Any], task: str,
) -> tuple[list[dict[str, Any]], str | None]:
    expected_ids = [0, 1, 2] if task == "CHIPS" else [0]
    prompts: list[dict[str, Any]] = []
    frames = object_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        return [], "OBJECT_MANIFEST_HAS_NO_FRAMES"
    for physical_id in expected_ids:
        selected: tuple[int, Mapping[str, Any]] | None = None
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            frame_id = frame.get("source_frame")
            instance = frame.get("physical_instances", {}).get(str(physical_id))
            if not isinstance(frame_id, int) or not isinstance(instance, Mapping):
                continue
            if (
                instance.get("physical_instance_id") == physical_id
                and instance.get("observed") is True
                and instance.get("valid") is True
                and int(instance.get("area_px", 0)) > 0
                and isinstance(instance.get("mask"), Mapping)
            ):
                selected = (frame_id, instance["mask"])
                break
        if selected is None:
            return [], f"NO_RELIABLE_INITIAL_MASK_FOR_INSTANCE_{physical_id}"
        frame_id, mask = selected
        try:
            verify_evidence(mask, label=f"instance {physical_id} initial mask")
        except ValueError as exc:
            return [], str(exc)
        prompts.append(
            {
                "instance_id": (
                    f"chips_{physical_id + 1}" if task == "CHIPS" else "poker_0"
                ),
                "physical_instance_id": physical_id,
                "prompt_events": [
                    {
                        "frame_id": frame_id,
                        "event_type": "INITIAL_MASK",
                        "mask": dict(mask),
                        "source_semantics": "FIRST_GRADE_B_OBSERVED_VALID_INSTANCE_MASK",
                    }
                ],
            }
        )
    return prompts, None


def build_prompt_manifest(
    *, session_id: str, task: str, prompts: list[dict[str, Any]],
    audit_evidence: Mapping[str, Any], object_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "causal-modal-mask-prompts-v71",
        "artifact_id": f"S1_CAUSAL_PROMPTS.{session_id}",
        "artifact_revision": "R7_1",
        "session_id": session_id,
        "task": task,
        "frozen": True,
        "allow_instance_union": False,
        "audit_selection": dict(audit_evidence),
        "source_object_manifest": dict(object_evidence),
        "prompt_policy": {
            "initial": "FIRST_RELIABLE_OBSERVED_INSTANCE_MASK",
            "correction": "CURRENT_OR_PAST_FRAME_ONLY",
            "future_frame_prompt_forbidden": True,
            "initial_mask_is_not_back_propagated_to_earlier_frames": True,
        },
        "instances": prompts,
    }


def _preflight_payload_status(result_path: Path) -> str:
    value = _load_object(result_path)
    return str(value.get("payload", {}).get("status", "BLOCKED_PREREQ"))


def run_preflight(
    *, rgb_manifest: Path, prompt_manifest: Path, output_root: Path,
    session_id: str, executor_epoch: int, fencing_token: str,
    verify_large_sha: bool,
) -> Path:
    command = [
        sys.executable, str(ROOT / "src/chaoyang/ops/run_causal_modal_mask_challenger_v71.py"),
        "--rgb-manifest", str(rgb_manifest),
        "--prompt-manifest", str(prompt_manifest),
        "--output-root", str(output_root),
        "--attempt-id", "attempt_0001",
        "--artifact-revision", "R7_1",
        "--executor-epoch", str(executor_epoch),
        "--fencing-token", fencing_token,
        "--backend", "all", "--preflight-only",
    ]
    if verify_large_sha:
        command.append("--verify-large-sha")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result = output_root / "sessions" / session_id / "attempts/attempt_0001/RESULT.json"
    if not result.is_file():
        raise RuntimeError(
            f"preflight did not publish receipt (rc={completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    if completed.returncode not in {0, 3}:
        raise RuntimeError(f"unexpected preflight return code {completed.returncode}")
    return result.resolve()


def build(
    *, audit_path: Path, output_root: Path, executor_epoch: int,
    fencing_token: str, extract_missing: bool = True,
) -> dict[str, Any]:
    audit_path = audit_path.resolve()
    output_root = output_root.resolve()
    if (output_root / "S1_INPUT_PREFLIGHT_INDEX.json").exists():
        raise FileExistsError(f"immutable S1 index already exists: {output_root}")
    audit = _load_object(audit_path)
    audit_ev = evidence(audit_path)
    rows: list[dict[str, Any]] = []
    shared_large_sha_preflight: dict[str, Any] | None = None

    for session in audit_sessions(audit):
        session_id = session["session_id"]
        task = session["task"]
        task_lower = task.lower()
        row: dict[str, Any] = {
            "session_id": session_id,
            "task": task,
            "status": "BLOCKED_PREREQ",
            "frame_count": 0,
            "instance_count": 0,
            "rgb_manifest": None,
            "prompt_manifest": None,
            "preflight_result": None,
            "blocked_reason": None,
            "audit_groups": session["groups"],
        }
        try:
            if len(session["source_manifests"]) != 1:
                raise ValueError("AMBIGUOUS_SOURCE_OBJECT_MANIFEST")
            source_manifest_path = Path(session["source_manifests"][0])
            source_row = next(
                item for item in audit["rows"] if item.get("session_id") == session_id
            )
            verify_evidence(source_row["source_manifest"], label="audit source manifest")
            object_manifest = _load_object(source_manifest_path)
            if object_manifest.get("session") != session_id:
                raise ValueError("OBJECT_MANIFEST_SESSION_MISMATCH")
            frame_count = len(object_manifest.get("frames", []))
            row["frame_count"] = frame_count
            if session["object_mask_grades"] != ["B"]:
                raise ValueError("OBJECT_MASK_NOT_GRADE_B_RELIABLE_PROMPT_SOURCE")
            prompts, prompt_error = first_reliable_prompts(
                object_manifest=object_manifest, task=task
            )
            if prompt_error:
                raise ValueError(prompt_error)
            row["instance_count"] = len(prompts)
            source_video = object_manifest.get("input", {}).get("selected_rgb")
            if not isinstance(source_video, Mapping):
                raise ValueError("MISSING_SELECTED_RGB_EVIDENCE")
            video_path = verify_evidence(source_video, label="selected RGB")

            input_dir = output_root / "sessions" / session_id / "input_revision_R7_1"
            frame_dir = discover_complete_frame_directory(
                task_lower=task_lower, session_id=session_id, frame_count=frame_count
            )
            source_kind = "PINNED_PREDECESSOR_LOSSLESS_FRAMES"
            if frame_dir is None:
                if not extract_missing:
                    raise ValueError("COMPLETE_RGB_FRAME_SEQUENCE_ABSENT")
                input_dir.mkdir(parents=True, exist_ok=False)
                frame_dir = extract_frames_atomic(video_path, input_dir / "frames", frame_count)
                source_kind = "FRESH_LOSSLESS_DECODE_FROM_PINNED_SELECTED_RGB"
            else:
                input_dir.mkdir(parents=True, exist_ok=False)

            rgb = build_rgb_manifest(
                session_id=session_id, frame_directory=frame_dir, frame_count=frame_count,
                source_video=source_video, source_kind=source_kind,
            )
            prompt_manifest = build_prompt_manifest(
                session_id=session_id, task=task, prompts=prompts,
                audit_evidence=audit_ev, object_evidence=evidence(source_manifest_path),
            )
            rgb_path = input_dir / "CAUSAL_RGB_MANIFEST.json"
            prompt_path = input_dir / "CAUSAL_PROMPT_MANIFEST.json"
            _write_json_exclusive(rgb_path, rgb)
            _write_json_exclusive(prompt_path, prompt_manifest)
            row["rgb_manifest"] = evidence(rgb_path)
            row["prompt_manifest"] = evidence(prompt_path)

            preflight_result = run_preflight(
                rgb_manifest=rgb_path, prompt_manifest=prompt_path,
                output_root=output_root, session_id=session_id,
                executor_epoch=executor_epoch, fencing_token=fencing_token,
                verify_large_sha=shared_large_sha_preflight is None,
            )
            row["preflight_result"] = evidence(preflight_result)
            if shared_large_sha_preflight is None:
                shared_large_sha_preflight = dict(row["preflight_result"])
                row["backend_large_sha_proof"] = dict(shared_large_sha_preflight)
                row["large_sha_verified_for_this_preflight"] = True
            else:
                row["backend_large_sha_proof"] = dict(shared_large_sha_preflight)
                row["large_sha_verified_for_this_preflight"] = False
            row["status"] = _preflight_payload_status(preflight_result)
            if row["status"] != "PASSED":
                row["blocked_reason"] = "BACKEND_OR_INPUT_PREFLIGHT_NOT_READY"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            row["blocked_reason"] = str(exc)
        rows.append(row)

    counts = {
        "sessions": len(rows),
        "input_ready": sum(row["rgb_manifest"] is not None for row in rows),
        "input_blocked": sum(row["rgb_manifest"] is None for row in rows),
        "preflight_passed": sum(row["status"] == "PASSED" for row in rows),
        "preflight_blocked": sum(row["status"] != "PASSED" for row in rows),
    }
    return {
        "schema_version": "s1-input-preflight-index-v71",
        "artifact_id": "S1_INPUT_PREFLIGHT_INDEX",
        "artifact_revision": "R7_1",
        "immutable": True,
        "audit_selection": audit_ev,
        "counts": counts,
        "rows": rows,
        "claim_limit": CLAIM_LIMIT,
    }


def write_outputs(index: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "S1_INPUT_PREFLIGHT_INDEX.json"
    csv_path = output_root / "S1_INPUT_PREFLIGHT_INDEX.csv"
    _write_json_exclusive(json_path, index)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "session_id", "task", "status", "frame_count", "instance_count",
                "blocked_reason", "rgb_manifest_path", "prompt_manifest_path",
                "preflight_result_path",
            ],
        )
        writer.writeheader()
        for row in index["rows"]:
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames[:6]},
                    "rgb_manifest_path": (row["rgb_manifest"] or {}).get("path"),
                    "prompt_manifest_path": (row["prompt_manifest"] or {}).get("path"),
                    "preflight_result_path": (row["preflight_result"] or {}).get("path"),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema_version": "s1-input-preflight-result-v71",
        "status": "PASSED_BOUNDED_INPUT_PREFLIGHT",
        "authority": False,
        "metrics": dict(index["counts"]),
        "outputs": [evidence(json_path), evidence(csv_path)],
        "claim_limit": CLAIM_LIMIT,
    }
    _write_json_exclusive(output_root / "RESULT.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument("--no-extract-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.executor_epoch < 1 or not args.fencing_token:
        parser.error("executor epoch and fencing token are mandatory")
    index = build(
        audit_path=args.audit,
        output_root=args.output_root,
        executor_epoch=args.executor_epoch,
        fencing_token=args.fencing_token,
        extract_missing=not args.no_extract_missing,
    )
    result = write_outputs(index, args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
