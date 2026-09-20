#!/usr/bin/env python3
"""Rebind stale AI1 staging artifact refs without recomputing any evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator
import uuid

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import build_wiyh_wrist_dual_input_v1 as input_builder


TASK_ID = "ai1_artifact_ref_rebind_v31"
SCHEMA_VERSION = "AI1_ARTIFACT_REF_REBIND_V31_RESULT"
OUTPUT_ROOT_RELATIVE = f"_run/current/{TASK_ID}/attempts/attempt_0001"
SOURCE_ATTEMPT_RELATIVE = "_run/current/ai1_cpfs_publish_fix_v31/attempts/attempt_0001"
LANE_RELATIVE = Path("AI1_CURRENT_ONLY_LANE")
CORRECTED_ROOT_RELATIVE = Path("CORRECTED_EVIDENCE") / LANE_RELATIVE
SOURCE_DOCUMENTS = (
    LANE_RELATIVE / "input_bundle/PROVENANCE.json",
    LANE_RELATIVE / "dual_representation/RESULT.json",
    LANE_RELATIVE / "LANE_LEDGER.json",
    LANE_RELATIVE / "RESULT.json",
)
STALE_COMPONENT_PREFIX = ".attempt_0001.staging-"


class RebindError(RuntimeError):
    """Raised when a stale reference cannot be rebound uniquely and exactly."""


def _assert_routed(root: Path) -> None:
    index = load_json(root / "tasks/current/INDEX.json")
    routes = index.get("task_packets", [])
    if len(routes) != 1 or routes[0].get("task_id") != TASK_ID:
        raise RebindError("AI1 artifact-ref successor is not the sole current route")
    if routes[0].get("execution_allowed") is not True:
        raise RebindError("AI1 artifact-ref successor is not execution_allowed")


def _future_ref(source: Path, future: Path) -> dict[str, Any]:
    reference = artifact_ref(source)
    reference["path"] = str(future.absolute())
    return reference


def _tree_manifest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        reference = artifact_ref(path)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(reference["sha256"]))
        count += 1
        total += int(reference["bytes"])
    return {
        "root": str(root.resolve(strict=True)),
        "file_count": count,
        "total_bytes": total,
        "aggregate_sha256": digest.hexdigest(),
    }


def _is_artifact_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("bytes"), int)
        and isinstance(value.get("sha256"), str)
    )


def _walk_artifact_refs(value: Any, pointer: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    if _is_artifact_ref(value):
        yield pointer or "/", value
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _walk_artifact_refs(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_artifact_refs(child, f"{pointer}/{index}")


def _stale_suffix(stale_path: str) -> Path:
    path = Path(stale_path)
    components = path.parts
    staging = [
        index for index, value in enumerate(components) if value.startswith(STALE_COMPONENT_PREFIX)
    ]
    lane = [index for index, value in enumerate(components) if value == LANE_RELATIVE.name]
    if len(staging) != 1 or len(lane) != 1 or lane[0] != staging[0] + 1:
        raise RebindError(f"stale path has an ambiguous staging/lane suffix: {stale_path}")
    suffix = Path(*components[lane[0] + 1 :])
    if not suffix.parts or ".." in suffix.parts:
        raise RebindError(f"stale path has no safe relative suffix: {stale_path}")
    return suffix


def _unique_final_target(final_lane: Path, suffix: Path) -> Path:
    matches = [
        path
        for path in final_lane.rglob(suffix.name)
        if path.is_file() and path.relative_to(final_lane) == suffix
    ]
    if len(matches) != 1:
        raise RebindError(f"stale suffix does not map uniquely ({len(matches)} matches): {suffix}")
    target = matches[0]
    if target.is_symlink():
        raise RebindError(f"stale suffix maps to a symlink: {suffix}")
    return target.resolve(strict=True)


def _audit_stale_refs(source_attempt: Path) -> list[dict[str, Any]]:
    final_lane = (source_attempt / LANE_RELATIVE).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    documents: set[Path] = set()
    for document in sorted(source_attempt.rglob("*.json")):
        value = load_json(document)
        for pointer, reference in _walk_artifact_refs(value):
            stale_path = str(reference["path"])
            if STALE_COMPONENT_PREFIX not in stale_path:
                continue
            suffix = _stale_suffix(stale_path)
            target = _unique_final_target(final_lane, suffix)
            target_ref = artifact_ref(target)
            if int(reference["bytes"]) != target_ref["bytes"]:
                raise RebindError(f"artifact byte mismatch at {document}:{pointer}")
            if str(reference["sha256"]) != target_ref["sha256"]:
                raise RebindError(f"artifact SHA mismatch at {document}:{pointer}")
            relative_document = document.relative_to(source_attempt)
            documents.add(relative_document)
            rows.append(
                {
                    "source_document": relative_document.as_posix(),
                    "json_pointer": pointer,
                    "stale_path": stale_path,
                    "relative_suffix": suffix.as_posix(),
                    "verified_final_artifact": target_ref,
                }
            )
    expected = set(SOURCE_DOCUMENTS)
    if documents != expected:
        raise RebindError(
            "stale-reference document set drift: "
            f"actual={sorted(map(str, documents))}, expected={sorted(map(str, expected))}"
        )
    if not rows:
        raise RebindError("no stale staging artifact references were found")
    identities: dict[str, tuple[int, str, str]] = {}
    for row in rows:
        key = row["stale_path"]
        current = row["verified_final_artifact"]
        identity = (current["bytes"], current["sha256"], current["path"])
        if key in identities and identities[key] != identity:
            raise RebindError(f"one stale path maps to conflicting final artifacts: {key}")
        identities[key] = identity
    return rows


def _replace_stale_refs(
    value: Any,
    *,
    target_by_stale_path: dict[str, Path],
    corrected_by_final_path: dict[Path, dict[str, Any]],
) -> Any:
    if _is_artifact_ref(value) and STALE_COMPONENT_PREFIX in str(value["path"]):
        stale = str(value["path"])
        target = target_by_stale_path[stale]
        return copy.deepcopy(corrected_by_final_path.get(target, artifact_ref(target)))
    if isinstance(value, dict):
        return {
            key: _replace_stale_refs(
                child,
                target_by_stale_path=target_by_stale_path,
                corrected_by_final_path=corrected_by_final_path,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_stale_refs(
                child,
                target_by_stale_path=target_by_stale_path,
                corrected_by_final_path=corrected_by_final_path,
            )
            for child in value
        ]
    return copy.deepcopy(value)


def run_successor(*, root: Path = REPO_ROOT, require_route: bool = True) -> dict[str, Any]:
    """Publish a fully closed corrected JSON evidence tree and rebind receipt."""

    root = root.resolve(strict=True)
    if require_route:
        _assert_routed(root)
    source_attempt = (root / SOURCE_ATTEMPT_RELATIVE).resolve(strict=True)
    source_result = load_json(source_attempt / "RESULT.json")
    source_run_receipt = load_json(source_attempt / "RUN_RECEIPT.json")
    if (
        source_result.get("schema_version") != "AI1_CPFS_PUBLISH_FIX_V31_RESULT"
        or source_result.get("status") != "BLOCKED_PREREQ"
        or source_run_receipt.get("finalization_status") != "PASSED_CAS_PUBLISHED"
    ):
        raise RebindError("source AI1 successor is not the sealed BLOCKED_PREREQ terminal")
    source_manifest_before = _tree_manifest(source_attempt)
    rows = _audit_stale_refs(source_attempt)
    output_root = root / OUTPUT_ROOT_RELATIVE
    if output_root.exists() or output_root.is_symlink():
        raise RebindError(f"fresh successor attempt required: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        target_by_stale_path = {
            row["stale_path"]: Path(row["verified_final_artifact"]["path"]).resolve(strict=True)
            for row in rows
        }
        corrected_by_final_path: dict[Path, dict[str, Any]] = {}
        corrected_refs: dict[str, dict[str, Any]] = {}
        for relative in SOURCE_DOCUMENTS:
            source = (source_attempt / relative).resolve(strict=True)
            destination_relative = Path("CORRECTED_EVIDENCE") / relative
            destination = staging / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            corrected = _replace_stale_refs(
                load_json(source),
                target_by_stale_path=target_by_stale_path,
                corrected_by_final_path=corrected_by_final_path,
            )
            if any(
                STALE_COMPONENT_PREFIX in str(reference["path"])
                for _, reference in _walk_artifact_refs(corrected)
            ):
                raise RebindError(f"corrected document still has stale refs: {relative}")
            atomic_json(destination, corrected)
            future = output_root / destination_relative
            corrected_reference = _future_ref(destination, future)
            corrected_by_final_path[source] = corrected_reference
            corrected_refs[relative.as_posix()] = corrected_reference

        for row in rows:
            target = Path(row["verified_final_artifact"]["path"]).resolve(strict=True)
            row["corrected_reference"] = copy.deepcopy(
                corrected_by_final_path.get(target, artifact_ref(target))
            )
        receipt = {
            "schema_version": "ai1-artifact-ref-rebind-v31-receipt-v1",
            "task_id": TASK_ID,
            "status": "PASSED_EXACT_REBIND",
            "source_attempt": str(source_attempt),
            "source_tree_before": source_manifest_before,
            "stale_reference_occurrences": len(rows),
            "unique_stale_paths": len({row["stale_path"] for row in rows}),
            "mappings": rows,
            "corrected_documents": corrected_refs,
            "algorithm_recomputed": False,
            "model_calls": 0,
            "gpu_calls": 0,
            "old_bytes_modified": False,
        }
        receipt_path = staging / "ARTIFACT_REF_REBIND_RECEIPT.json"
        atomic_json(receipt_path, receipt)
        source_manifest_after = _tree_manifest(source_attempt)
        if source_manifest_after != source_manifest_before:
            raise RebindError("source AI1 successor tree changed during rebind")
        corrected_result_ref = corrected_refs[(LANE_RELATIVE / "RESULT.json").as_posix()]
        corrected_ledger_ref = corrected_refs[(LANE_RELATIVE / "LANE_LEDGER.json").as_posix()]
        result = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "status": "PASSED",
            "source_lane_status": "BLOCKED_ADOPTION_OBSERVATIONS",
            "source_successor_status": "BLOCKED_PREREQ",
            "source_successor_result": artifact_ref(source_attempt / "RESULT.json"),
            "source_successor_run_receipt": artifact_ref(source_attempt / "RUN_RECEIPT.json"),
            "source_tree_preserved": True,
            "source_tree_manifest": source_manifest_after,
            "corrected_lane_ledger": corrected_ledger_ref,
            "corrected_lane_result": corrected_result_ref,
            "rebind_receipt": _future_ref(
                receipt_path, output_root / "ARTIFACT_REF_REBIND_RECEIPT.json"
            ),
            "stale_reference_occurrences": len(rows),
            "unique_stale_paths": len({row["stale_path"] for row in rows}),
            "all_mappings_unique": True,
            "all_bytes_sha_exact": True,
            "algorithm_recomputed": False,
            "weights": "ABSENT",
            "model_calls": 0,
            "gpu_calls": 0,
            "pipeline_complete": False,
            "numeric_quality_pass": False,
            "visual_review_status": "NOT_REVIEWED",
            "training_complete": False,
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
        }
        atomic_json(staging / "RESULT.json", result)
        input_builder._publish_directory_no_clobber(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_json(output_root / "RESULT.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_successor(root=args.root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
