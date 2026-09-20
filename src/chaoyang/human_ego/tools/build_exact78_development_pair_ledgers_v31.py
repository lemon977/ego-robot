#!/usr/bin/env python3
"""Audit current Exact78 assets and publish causal DEVELOPMENT pair ledgers.

This builder never searches ``archive/`` and never promotes co-located review
videos into Robotized training inputs.  A pair is admitted only from an
explicit current bundle manifest plus a byte-bound suffix-invariance receipt.
When the current 156-row split/source-group authority or legal bundles are
missing, it preserves the raw inventory and emits a fail-closed blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from chaoyang.human_ego.tools import train_visual_aux_future2d_v53 as trainer
from chaoyang.human_ego.tools.validate_visual_aux_bundle_v52 import validate_bundle


COHORT_SCHEMA = "EXACT78_CURRENT_COHORT_V31"
CANDIDATE_SCHEMA = "EXACT78_CURRENT_VISUAL_AUX_PAIR_CANDIDATES_V31"
INVENTORY_SCHEMA = "EXACT78_CURRENT_RAW_INVENTORY_V31"
RESULT_SCHEMA = "EXACT78_DEVELOPMENT_PAIR_LEDGER_BUILD_RESULT_V31"
CONTRACT = trainer.V31_DEVELOPMENT_CONTRACT
EXPECTED_SPLITS = {"train": 60, "validation": 8, "test": 5, "heldout": 5}
ROOT_SPECS = (
    ("chips", "0901", "chips_cards_tracker_0901/potato_chips", 16),
    ("chips", "0902", "chips_cards_tracker_0902/potato_chips", 52),
    ("chips", "0903", "chips_cards_tracker_0903/potato_chips", 10),
    ("poker", "0901", "chips_cards_tracker_0901/playing_cards", 43),
    ("poker", "0902", "chips_cards_tracker_0902/playing_cards", 25),
    ("poker", "0903", "chips_cards_tracker_0903/playing_cards", 10),
)


class AdmissionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdmissionError(f"JSON object required: {path}")
    return value


def is_archive(path: Path) -> bool:
    return "archive" in path.resolve(strict=False).parts


def current_regular_ref(item: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(str(item["path"]))
        expected_bytes = int(item["bytes"])
        expected_sha = str(item["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise AdmissionError(f"{label}: invalid artifact reference") from error
    if not path.is_absolute() or is_archive(path):
        raise AdmissionError(f"{label}: current non-archive absolute path required")
    if path.is_symlink():
        raise AdmissionError(f"{label}: regular non-symlink file required")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise AdmissionError(f"{label}: regular non-symlink file required")
    if resolved.stat().st_size != expected_bytes or sha256(resolved) != expected_sha:
        raise AdmissionError(f"{label}: byte/SHA mismatch")
    return resolved


def discover_raw_inventory(data_root: Path) -> tuple[dict[str, Any], list[str]]:
    root = data_root.resolve(strict=False)
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    if is_archive(root):
        blockers.append("ARCHIVE_DATA_ROOT_FORBIDDEN")
    for task, date, relative, expected in ROOT_SPECS:
        task_root = root / relative
        prefix = "get_potato_chips_" if task == "chips" else "play_cards_"
        sessions = sorted(
            path
            for path in task_root.glob(f"{prefix}{date}_*")
            if path.is_dir() and not path.is_symlink()
        )
        if len(sessions) < expected:
            blockers.append(
                f"RAW_ROOT_CAPACITY_{task.upper()}_{date}:{len(sessions)}<{expected}"
            )
        for session in sessions:
            manifest = session / "clip_manifest.json"
            video = session / f"CameraRecord_{session.name}.mp4"
            row_blockers = []
            if not manifest.is_file() or manifest.is_symlink():
                row_blockers.append("CLIP_MANIFEST_MISSING")
            if not video.is_file() or video.is_symlink():
                row_blockers.append("RAW_RGB_VIDEO_MISSING")
            source_group_candidate: str | None = None
            if not row_blockers:
                payload = load_json(manifest)
                if payload.get("clip_name") != session.name:
                    row_blockers.append("CLIP_NAME_MISMATCH")
                run_id = payload.get("run_id")
                source_sessions = sorted(
                    {
                        str(piece.get("session"))
                        for piece in payload.get("pieces", [])
                        if isinstance(piece, dict) and piece.get("session") is not None
                    }
                )
                if isinstance(run_id, str) and run_id and source_sessions:
                    source_group_candidate = f"{run_id}:{'+'.join(source_sessions)}"
            rows.append(
                {
                    "task": task,
                    "date": date,
                    "session_id": session.name,
                    "raw_path": str(session.resolve()),
                    "clip_manifest": artifact(manifest) if manifest.is_file() else None,
                    "raw_rgb_video": {
                        "path": str(video.resolve(strict=False)),
                        "bytes": video.stat().st_size if video.is_file() else None,
                        "sha256": None,
                        "sha_status": "NOT_COMPUTED_IN_INVENTORY_AUDIT",
                    },
                    "source_group_candidate": source_group_candidate,
                    "independence": "UNKNOWN",
                    "split": "UNKNOWN",
                    "status": "PASS_RAW_PRESENT" if not row_blockers else "BLOCKED_RAW_ASSET",
                    "blockers": row_blockers,
                }
            )
    identities = [row["session_id"] for row in rows]
    if len(set(identities)) != len(identities):
        blockers.append("CURRENT_RAW_CANDIDATE_SESSION_IDS_NOT_UNIQUE")
    counts = {
        task: sum(row["task"] == task for row in rows) for task in ("chips", "poker")
    }
    if counts["chips"] < 78 or counts["poker"] < 78:
        blockers.append(f"EXACT78_RAW_CANDIDATE_CAPACITY_INSUFFICIENT:{counts}")
    return (
        {
            "schema_version": INVENTORY_SCHEMA,
            "status": (
                "PASS_RAW_CANDIDATE_POOL" if not blockers else "BLOCKED_RAW_INVENTORY"
            ),
            "data_root": str(root),
            "counts": {"sessions": len(rows), **counts},
            "rows": rows,
            "claim_limit": (
                "Current raw candidate pool, not the frozen Exact78 denominator. Candidate "
                "source groups are not independent-recording proof and split remains UNKNOWN "
                "without a current non-archive 156-row cohort authority."
            ),
        },
        sorted(set(blockers)),
    )


def validate_current_cohort(
    cohort_path: Path,
    inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if cohort_path.is_symlink():
        raise AdmissionError("current cohort must be a non-archive regular file")
    path = cohort_path.resolve(strict=True)
    if is_archive(path):
        raise AdmissionError("current cohort must be a non-archive regular file")
    cohort = load_json(path)
    if cohort.get("schema_version") != COHORT_SCHEMA:
        raise AdmissionError("current cohort schema mismatch")
    rows = cohort.get("sessions")
    if not isinstance(rows, list) or len(rows) != 156:
        raise AdmissionError("current cohort must contain 156 rows")
    inventory_rows = {
        str(row["session_id"]): row for row in inventory.get("rows", [])
    }
    indexed: dict[str, dict[str, Any]] = {}
    per_task_split: dict[str, dict[str, int]] = {
        task: {split: 0 for split in EXPECTED_SPLITS} for task in ("chips", "poker")
    }
    group_split: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AdmissionError("invalid current cohort row")
        session = str(row.get("session_id", ""))
        task = str(row.get("task", ""))
        split = str(row.get("split", ""))
        group = str(row.get("source_group_id", ""))
        if session in indexed or task not in per_task_split or split not in EXPECTED_SPLITS:
            raise AdmissionError(f"invalid/duplicate cohort identity: {session}")
        if not group or group == "UNKNOWN_SOURCE_GROUP":
            raise AdmissionError(f"unknown source group in current cohort: {session}")
        current = inventory_rows.get(session)
        if (
            current is None
            or current.get("task") != task
            or current.get("status") != "PASS_RAW_PRESENT"
            or Path(str(row.get("raw_path", ""))).resolve(strict=False)
            != Path(str(current["raw_path"])).resolve(strict=False)
        ):
            raise AdmissionError(f"cohort/raw inventory mismatch: {session}")
        previous = group_split.get(group)
        if previous is not None and previous != split:
            raise AdmissionError(f"source group crosses splits: {group}={previous}/{split}")
        group_split[group] = split
        per_task_split[task][split] += 1
        indexed[session] = row
    for task, counts in per_task_split.items():
        if counts != EXPECTED_SPLITS:
            raise AdmissionError(
                f"current cohort split counts invalid for {task}: {counts}"
            )
    return cohort, indexed


def validate_suffix_audit(path: Path) -> dict[str, Any]:
    if path.is_symlink() or is_archive(path):
        raise AdmissionError("suffix audit must be a current non-archive regular file")
    payload = load_json(path)
    if (
        payload.get("schema_version") != "EXACT78_SUFFIX_INVARIANCE_V31"
        or payload.get("status") != "PASS"
        or payload.get("causal_current_inputs_authorized") is not True
    ):
        raise AdmissionError("suffix-invariance audit did not pass")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not trainer.V31_REQUIRED_SUFFIX_FIELDS.issubset(
        fields
    ):
        raise AdmissionError("suffix-invariance required current-input fields missing")
    for name in trainer.V31_REQUIRED_SUFFIX_FIELDS:
        row = fields[name]
        if (
            not isinstance(row, dict)
            or row.get("suffix_invariant") is not True
            or row.get("effective_temporal_authority") != "CAUSAL_CURRENT"
        ):
            raise AdmissionError(f"noncausal current input: {name}")
    return payload


def _assert_current_artifact_refs(value: Any, root: Path, label: str) -> None:
    """Reject a current bundle that indirectly consumes an archive artifact."""

    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = Path(str(value["path"]))
            resolved = (path if path.is_absolute() else root / path).resolve(
                strict=False
            )
            if is_archive(resolved):
                raise AdmissionError(f"{label}: archive artifact reference forbidden")
        for key, item in value.items():
            _assert_current_artifact_refs(item, root, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_current_artifact_refs(item, root, f"{label}[{index}]")


def validate_current_bundle(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_symlink():
        raise AdmissionError("current bundle manifest path is invalid")
    path = manifest_path.resolve(strict=True)
    if is_archive(path) or path.name != "VISUAL_AUX_SESSION_MANIFEST.json":
        raise AdmissionError("current bundle manifest path is invalid")
    payload = load_json(path)
    _assert_current_artifact_refs(payload, path.parent, "bundle")
    branches = payload.get("branches")
    if isinstance(branches, dict):
        for branch, value in branches.items():
            if not isinstance(value, dict) or not isinstance(value.get("selector"), dict):
                continue
            selector_item = value["selector"]
            selector_path = Path(str(selector_item.get("path", "")))
            if not selector_path.is_absolute():
                selector_path = path.parent / selector_path
            selector_path = selector_path.resolve(strict=False)
            if selector_path.is_file():
                selector = load_json(selector_path)
                _assert_current_artifact_refs(
                    selector, path.parent, f"bundle.branches.{branch}.selector"
                )
    if (
        payload.get("input_mode") != "CAUSAL_TRAINING_INPUT"
        or payload.get("causal_proof", {}).get("enabled") is not True
        or payload.get("causal_proof", {}).get("raw_robotized_valid_mask_shared")
        is not True
    ):
        raise AdmissionError("bundle does not declare causal paired inputs")
    report = validate_bundle(path.parent)
    trainer._validate_manifest_training_gate(path, payload, report)
    if int(report["eligible_h50_window_count"]) < 1:
        raise AdmissionError("bundle has zero legal H50 windows")
    return {"payload": payload, "report": report}


def validate_candidates(
    candidate_path: Path,
    cohort: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if candidate_path.is_symlink():
        raise AdmissionError("candidate index must be a current non-archive regular file")
    path = candidate_path.resolve(strict=True)
    if is_archive(path):
        raise AdmissionError("candidate index must be a current non-archive regular file")
    payload = load_json(path)
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise AdmissionError("candidate index schema mismatch")
    values = payload.get("candidates")
    if not isinstance(values, list):
        raise AdmissionError("candidate list required")
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        session = str(item.get("session_id", "")) if isinstance(item, dict) else ""
        try:
            if not isinstance(item, dict) or session in seen:
                raise AdmissionError("invalid/duplicate candidate")
            seen.add(session)
            cohort_row = cohort.get(session)
            if cohort_row is None:
                raise AdmissionError("candidate is outside current cohort")
            manifest_path = current_regular_ref(item.get("bundle_manifest", {}), "bundle")
            suffix_path = current_regular_ref(item.get("suffix_invariance", {}), "suffix")
            validated = validate_current_bundle(manifest_path)
            suffix = validate_suffix_audit(suffix_path)
            manifest = validated["payload"]
            for key in ("task", "split"):
                if manifest.get(key) != cohort_row.get(key):
                    raise AdmissionError(f"bundle/cohort {key} mismatch")
            if manifest.get("session_id") != session:
                raise AdmissionError("bundle/cohort session mismatch")
            admitted.append(
                {
                    "session_id": session,
                    "task": manifest["task"],
                    "split": manifest["split"],
                    "source_group_id": cohort_row["source_group_id"],
                    "manifest": artifact(manifest_path),
                    "suffix_audit": artifact(suffix_path),
                    "eligible_h50_starts": validated["report"]["eligible_h50_starts"],
                    "eligible_h50_window_count": validated["report"][
                        "eligible_h50_window_count"
                    ],
                    "suffix_status": suffix["status"],
                }
            )
        except (AdmissionError, OSError, ValueError, KeyError) as error:
            rejected.append({"session_id": session, "reason": str(error)})
    return admitted, rejected


def _task_outputs(
    task: str,
    admitted: list[dict[str, Any]],
    output: Path,
    cohort_ref: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    rows = [row for row in admitted if row["task"] == task]
    blockers: list[str] = []
    if not rows:
        blockers.append(f"{task.upper()}_NO_CURRENT_LEGAL_RAW_ROBOTIZED_PAIR")
        return {"task": task, "pair_count": 0, "ledgers": None}, blockers
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation")
    }
    if not by_split["train"]:
        blockers.append(f"{task.upper()}_NO_LEGAL_TRAIN_PAIR")
    if not by_split["validation"]:
        blockers.append(f"{task.upper()}_NO_LEGAL_VALIDATION_PAIR")
    if blockers:
        return {"task": task, "pair_count": len(rows), "ledgers": None}, blockers

    hardset = {
        "schema_version": "VISUAL_AUX_OCCLUSION_HARDSET_V1",
        "status": "FROZEN_BEFORE_TRAINING",
        "task": task,
        "split": "validation",
        "sessions": [
            {
                "session_id": row["session_id"],
                "eligible_h50_starts": row["eligible_h50_starts"],
            }
            for row in by_split["validation"]
        ],
    }
    hardset_path = output / task / "OCCLUSION_HARDSET.json"
    atomic_json(hardset_path, hardset)
    suffix_bindings = [
        {
            "session_id": row["session_id"],
            "bundle_manifest_sha256": row["manifest"]["sha256"],
            "audit": row["suffix_audit"],
        }
        for row in sorted(rows, key=lambda value: value["session_id"])
        if row["split"] in {"train", "validation"}
    ]
    base = {
        "schema_version": "exact78-visual-aux-dataset-ledger-v53-v1",
        "status": "PASS_VISUAL_AUX_DATASET_LEDGER",
        "task": task,
        "seed": 7,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "release_id": "EXACT78_V31_DEVELOPMENT",
        "contract": CONTRACT,
        "cohort_authority": cohort_ref,
        "suffix_invariance_receipts": suffix_bindings,
        "train": [row["manifest"] for row in by_split["train"]],
        "validation": [row["manifest"] for row in by_split["validation"]],
        "occlusion_hardset": artifact(hardset_path),
    }
    base["pair_dataset_signature"] = trainer.pair_dataset_signature(base)
    paths = {}
    for branch in ("HUMAN_RAW_RGB", "ROBOTIZED_RGB"):
        ledger = {**base, "branch": branch}
        ledger_path = output / task / f"{branch}_LEDGER.json"
        atomic_json(ledger_path, ledger)
        paths[branch] = ledger_path
    trainer.validate_paired_ledgers(
        paths["HUMAN_RAW_RGB"], paths["ROBOTIZED_RGB"]
    )
    return (
        {
            "task": task,
            "pair_count": len(rows),
            "train_pairs": len(by_split["train"]),
            "validation_pairs": len(by_split["validation"]),
            "ledgers": {branch: artifact(path) for branch, path in paths.items()},
            "hardset": artifact(hardset_path),
        },
        [],
    )


def build(
    *,
    data_root: Path,
    cohort_manifest: Path | None,
    candidate_index: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh/no-clobber output root required: {output}")
    output.mkdir(parents=True)
    inventory, blockers = discover_raw_inventory(data_root)
    inventory_path = output / "CURRENT_RAW_INVENTORY.json"
    atomic_json(inventory_path, inventory)

    cohort_ref = None
    cohort_rows: dict[str, dict[str, Any]] = {}
    if cohort_manifest is None:
        blockers.append("MISSING_CURRENT_NON_ARCHIVE_156_COHORT_MANIFEST")
    else:
        try:
            _, cohort_rows = validate_current_cohort(cohort_manifest, inventory)
            cohort_ref = artifact(cohort_manifest)
        except (AdmissionError, OSError, ValueError) as error:
            blockers.append(f"CURRENT_COHORT_INVALID:{error}")

    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidate_ref = None
    if candidate_index is None:
        blockers.append("MISSING_CURRENT_PAIR_CANDIDATE_INDEX")
    elif not cohort_rows:
        blockers.append("PAIR_CANDIDATES_NOT_EVALUATED_WITHOUT_COHORT_AUTHORITY")
    else:
        try:
            candidate_ref = artifact(candidate_index)
            admitted, rejected = validate_candidates(candidate_index, cohort_rows)
        except (AdmissionError, OSError, ValueError) as error:
            blockers.append(f"CURRENT_PAIR_CANDIDATE_INDEX_INVALID:{error}")

    task_results = []
    if cohort_ref is not None and candidate_ref is not None:
        for task in ("chips", "poker"):
            try:
                task_result, task_blockers = _task_outputs(
                    task, admitted, output, cohort_ref
                )
            except Exception as error:
                task_result = {"task": task, "pair_count": 0, "ledgers": None}
                task_blockers = [f"{task.upper()}_LEDGER_VALIDATION_FAILED:{error}"]
            task_results.append(task_result)
            blockers.extend(task_blockers)
    else:
        task_results = [
            {"task": task, "pair_count": 0, "ledgers": None}
            for task in ("chips", "poker")
        ]
        blockers.extend(
            [
                "CHIPS_NO_CURRENT_LEGAL_RAW_ROBOTIZED_PAIR",
                "POKER_NO_CURRENT_LEGAL_RAW_ROBOTIZED_PAIR",
            ]
        )

    pair_audit = {
        "schema_version": "EXACT78_DEVELOPMENT_PAIR_AUDIT_V31",
        "status": "PASS" if admitted and not rejected else "PARTIAL_OR_BLOCKED",
        "cohort_manifest": cohort_ref,
        "candidate_index": candidate_ref,
        "admitted": admitted,
        "rejected": rejected,
        "archive_consumed": False,
    }
    pair_audit_path = output / "PAIR_AUDIT.json"
    atomic_json(pair_audit_path, pair_audit)
    blockers = sorted(set(blockers))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "READY_EXACT78_V31_DEVELOPMENT" if not blockers else "BLOCKED_INPUTS",
        "execution_allowed": not blockers,
        "training_started": False,
        "gpu_used": False,
        "archive_consumed": False,
        "blockers": blockers,
        "tasks": task_results,
        "outputs": {
            "raw_inventory": artifact(inventory_path),
            "pair_audit": artifact(pair_audit_path),
        },
        "claim_limit": (
            "A READY result authorizes only V3.1 DEVELOPMENT Visual Aux training. "
            "It does not create Robot control ground truth or physical deployment authority."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        data_root=args.data_root,
        cohort_manifest=args.cohort_manifest,
        candidate_index=args.candidate_index,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["execution_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
