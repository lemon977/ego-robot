#!/usr/bin/env python3
"""Build V3.2 Exact78 pair ledgers from the frozen 59/77/20 cohort.

The historical manifest is provenance only.  Execution consumes the checked-in
compact cohort and current, byte-bound bundle candidates.  Extra current
sessions are reported but never substituted for a missing frozen identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from chaoyang.human_ego import exact78_v32 as contract
from chaoyang.human_ego.tools import (
    build_exact78_development_pair_ledgers_v31 as legacy,
)
from chaoyang.human_ego.tools import train_visual_aux_future2d_v53 as trainer


PROJECT = Path(__file__).resolve().parents[4]
COHORT_SCHEMA_PATH = PROJECT / "contracts/exact78_cohort_split_v32.schema.json"
LEDGER_SCHEMA_PATH = PROJECT / "contracts/exact78_pair_ledger_v32.schema.json"
CANDIDATE_SCHEMA = "EXACT78_CURRENT_VISUAL_AUX_PAIR_CANDIDATES_V32"
RESULT_SCHEMA = "EXACT78_PAIR_LEDGER_BUILD_RESULT_V32"
DEFAULT_COHORT = PROJECT / "manifests/human_ego/exact78_cohort_split_v32.json"


AdmissionError = legacy.AdmissionError
artifact = legacy.artifact
atomic_json = legacy.atomic_json
discover_raw_inventory = legacy.discover_raw_inventory
load_json = legacy.load_json


def _validate_json_schema(payload: Mapping[str, Any], path: Path) -> None:
    jsonschema.Draft202012Validator(load_json(path)).validate(payload)


def validate_current_cohort(
    cohort_path: Path,
    inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    """Rebind exactly the frozen identities and preserve missing rows."""

    if cohort_path.is_symlink() or legacy.is_archive(cohort_path):
        raise AdmissionError("V3.2 cohort must be a current regular file")
    payload = load_json(cohort_path.resolve(strict=True))
    _validate_json_schema(payload, COHORT_SCHEMA_PATH)
    try:
        frozen = contract.validate_cohort_split(payload)
    except contract.ContractError as error:
        raise AdmissionError(str(error)) from error
    current = {str(row["session_id"]): row for row in inventory.get("rows", [])}
    unexpected = sorted(set(current) - set(frozen))
    rebound: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for session, frozen_row in frozen.items():
        row = current.get(session)
        expected_group = frozen_row["source_group_id"]
        if (
            row is None
            or row.get("status") != "PASS_RAW_PRESENT"
            or row.get("task") != frozen_row["task"]
            or row.get("date") != frozen_row["capture_date"]
            or row.get("source_group_candidate") != expected_group
        ):
            unresolved.append(session)
            rebound[session] = {
                **frozen_row,
                "current_rebind_status": "UNRESOLVED_NO_SUBSTITUTION",
                "raw_path": None,
                "clip_manifest": None,
            }
            continue
        rebound[session] = {
            **frozen_row,
            "current_rebind_status": "RESOLVED_EXACT_FROZEN_IDENTITY",
            "raw_path": row["raw_path"],
            "clip_manifest": row["clip_manifest"],
        }
    audit = {
        "schema_version": "EXACT78_CURRENT_REBIND_AUDIT_V32",
        "status": "PASS" if not unresolved else "BLOCKED_UNRESOLVED_FROZEN_MEMBERS",
        "frozen_member_count": len(frozen),
        "resolved_member_count": len(frozen) - len(unresolved),
        "unresolved_members": unresolved,
        "unexpected_current_candidates": unexpected,
        "unexpected_candidates_consumed": False,
        "substitution_count": 0,
    }
    return audit, rebound, unresolved


def _selector_frame_refs(
    manifest_path: Path, payload: Mapping[str, Any], branch: str
) -> list[dict[str, Any]]:
    selector_item = payload.get("branches", {}).get(branch, {}).get("selector")
    if not isinstance(selector_item, dict):
        raise AdmissionError(f"{branch} selector receipt missing")
    selector_path = Path(str(selector_item.get("path", "")))
    if not selector_path.is_absolute():
        selector_path = manifest_path.parent / selector_path
    selector_path = legacy.current_regular_ref(
        {**selector_item, "path": str(selector_path)}, f"{branch} selector"
    )
    selector = load_json(selector_path)
    frames = selector.get("frames")
    if not isinstance(frames, list):
        raise AdmissionError(f"{branch} selector frames missing")
    refs: list[dict[str, Any]] = []
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or row.get("frame_id") != index:
            raise AdmissionError(f"{branch} frame order mismatch")
        item = row.get("rgb")
        if not isinstance(item, dict):
            raise AdmissionError(f"{branch} frame RGB receipt missing")
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = manifest_path.parent / path
        resolved = legacy.current_regular_ref(
            {**item, "path": str(path)}, f"{branch}:frame:{index}"
        )
        refs.append({**item, "path": str(resolved)})
    return refs


def validate_current_bundle(manifest_path: Path) -> dict[str, Any]:
    """Apply the legacy structural gate plus V3.2 real-pixel readiness."""

    validated = legacy.validate_current_bundle(manifest_path)
    path = manifest_path.resolve(strict=True)
    payload = validated["payload"]
    raw = _selector_frame_refs(path, payload, "HUMAN_RAW_RGB")
    robotized = _selector_frame_refs(path, payload, "ROBOTIZED_RGB")
    try:
        pixel_audit = contract.validate_pair_production_readiness(
            payload, raw_frame_refs=raw, robotized_frame_refs=robotized
        )
    except contract.ContractError as error:
        raise AdmissionError(str(error)) from error
    return {**validated, "pair_pixel_audit": pixel_audit}


def validate_suffix_audit(path: Path) -> dict[str, Any]:
    if path.is_symlink() or legacy.is_archive(path):
        raise AdmissionError("suffix audit must be current and non-archive")
    payload = load_json(path.resolve(strict=True))
    try:
        contract.validate_suffix_receipt(payload)
    except contract.ContractError as error:
        raise AdmissionError(str(error)) from error
    return payload


def validate_candidates(
    candidate_path: Path,
    cohort: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if candidate_path.is_symlink() or legacy.is_archive(candidate_path):
        raise AdmissionError("candidate index must be current and non-archive")
    payload = load_json(candidate_path.resolve(strict=True))
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise AdmissionError("V3.2 candidate index schema mismatch")
    values = payload.get("candidates")
    if not isinstance(values, list):
        raise AdmissionError("candidate list required")
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        session = str(item.get("session_id", "")) if isinstance(item, dict) else ""
        try:
            if not isinstance(item, dict) or not session or session in seen:
                raise AdmissionError("invalid/duplicate candidate")
            seen.add(session)
            cohort_row = cohort.get(session)
            if cohort_row is None:
                raise AdmissionError("candidate is outside frozen 156 identities")
            if cohort_row.get("current_rebind_status") != "RESOLVED_EXACT_FROZEN_IDENTITY":
                raise AdmissionError("frozen identity is unresolved; substitution forbidden")
            if cohort_row.get("split") == "development_final":
                raise AdmissionError("exposed development_final is not a training split")
            manifest_path = legacy.current_regular_ref(
                item.get("bundle_manifest", {}), "bundle"
            )
            suffix_path = legacy.current_regular_ref(
                item.get("suffix_invariance", {}), "suffix"
            )
            validated = validate_current_bundle(manifest_path)
            validate_suffix_audit(suffix_path)
            manifest = validated["payload"]
            if (
                manifest.get("session_id") != session
                or manifest.get("task") != cohort_row.get("task")
                or manifest.get("split") != cohort_row.get("split")
            ):
                raise AdmissionError("bundle/frozen-cohort identity mismatch")
            admitted.append(
                {
                    "session_id": session,
                    "task": manifest["task"],
                    "split": manifest["split"],
                    "source_group_id": cohort_row["source_group_id"],
                    "source_lineage": cohort_row["clip_manifest"],
                    "manifest": artifact(manifest_path),
                    "suffix_audit": artifact(suffix_path),
                    "pair_pixel_audit": validated["pair_pixel_audit"],
                    "eligible_h50_starts": validated["report"]["eligible_h50_starts"],
                    "eligible_h50_window_count": validated["report"][
                        "eligible_h50_window_count"
                    ],
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
    producer_identity: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    rows = [row for row in admitted if row["task"] == task]
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation")
    }
    blockers = [
        f"{task.upper()}_NO_LEGAL_{split.upper()}_PAIR"
        for split in ("train", "validation")
        if not by_split[split]
    ]
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
    consumed = sorted(
        by_split["train"] + by_split["validation"],
        key=lambda row: row["session_id"],
    )
    base = {
        "schema_version": contract.PAIR_LEDGER_SCHEMA,
        "status": "PASS_EXACT78_PAIR_LEDGER_V32",
        "contract": contract.PAIR_CONTRACT,
        "release_id": "EXACT78_V32_DEVELOPMENT",
        "task": task,
        "seed": 7,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "image_domain": "SOURCE_DOMAIN_BOUND_PER_BUNDLE",
        "units": "RGB_UINT8_AND_NORMALIZED_IMAGE_XY",
        "side": "BUNDLE_DECLARED_PHYSICAL_SIDE",
        "validity": "EXPLICIT_SHARED_VALID_MASK_NO_FILL",
        "observability": "OBSERVED_AND_INFERRED_EXPLICITLY_SEPARATED",
        "temporal_authority": "CAUSAL_CURRENT_INPUT_NONCAUSAL_TARGET",
        "producer_identity": producer_identity,
        "cohort_authority": cohort_ref,
        "source_bindings": [
            {
                "session_id": row["session_id"],
                "source_group_id": row["source_group_id"],
                "split": row["split"],
                "source_frame": "BUNDLE_FRAME_IDS",
                "timestamp": "BUNDLE_TIMESTAMPS",
                "current_source_lineage": row["source_lineage"]["path"],
                "source_lineage_receipt": row["source_lineage"],
                "input_sha256": row["manifest"]["sha256"],
                "status": "RESOLVED_EXACT_FROZEN_IDENTITY",
            }
            for row in consumed
        ],
        "suffix_invariance_receipts": [
            {
                "session_id": row["session_id"],
                "bundle_manifest_sha256": row["manifest"]["sha256"],
                "audit": row["suffix_audit"],
            }
            for row in consumed
        ],
        "pair_production_readiness": [
            {"session_id": row["session_id"], **row["pair_pixel_audit"]}
            for row in consumed
        ],
        "train": [row["manifest"] for row in by_split["train"]],
        "validation": [row["manifest"] for row in by_split["validation"]],
        "development_final": [],
        "occlusion_hardset": artifact(hardset_path),
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "external_metric_authority": False,
    }
    base["pair_dataset_signature"] = trainer.pair_dataset_signature(base)
    paths: dict[str, Path] = {}
    for branch in ("HUMAN_RAW_RGB", "ROBOTIZED_RGB"):
        ledger = {**base, "branch": branch}
        _validate_json_schema(ledger, LEDGER_SCHEMA_PATH)
        path = output / task / f"{branch}_LEDGER_V32.json"
        atomic_json(path, ledger)
        paths[branch] = path
    trainer.validate_paired_ledgers(paths["HUMAN_RAW_RGB"], paths["ROBOTIZED_RGB"])
    return (
        {
            "task": task,
            "pair_count": len(rows),
            "train_pairs": len(by_split["train"]),
            "validation_pairs": len(by_split["validation"]),
            "ledgers": {name: artifact(path) for name, path in paths.items()},
        },
        [],
    )


def build(
    *,
    data_root: Path,
    cohort_manifest: Path,
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

    cohort_ref: dict[str, Any] | None = None
    cohort_rows: dict[str, dict[str, Any]] = {}
    try:
        rebind, cohort_rows, unresolved = validate_current_cohort(
            cohort_manifest, inventory
        )
        cohort_ref = artifact(cohort_manifest)
        if unresolved:
            blockers.append(f"UNRESOLVED_FROZEN_MEMBERS:{len(unresolved)}")
    except (AdmissionError, OSError, ValueError, jsonschema.ValidationError) as error:
        rebind = {
            "schema_version": "EXACT78_CURRENT_REBIND_AUDIT_V32",
            "status": "BLOCKED_INVALID_COHORT",
            "reason": str(error),
        }
        blockers.append(f"CURRENT_COHORT_INVALID:{error}")
    rebind_path = output / "CURRENT_REBIND_AUDIT_V32.json"
    atomic_json(rebind_path, rebind)

    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidate_ref: dict[str, Any] | None = None
    if candidate_index is None:
        blockers.append("MISSING_CURRENT_V32_PAIR_CANDIDATE_INDEX")
    elif cohort_ref is not None and cohort_rows:
        try:
            candidate_ref = artifact(candidate_index)
            admitted, rejected = validate_candidates(candidate_index, cohort_rows)
            if rejected:
                blockers.append(f"PAIR_CANDIDATES_REJECTED:{len(rejected)}")
        except (AdmissionError, OSError, ValueError) as error:
            blockers.append(f"CURRENT_PAIR_CANDIDATE_INDEX_INVALID:{error}")

    producer_identity = {
        "producer_sha256": legacy.sha256(Path(__file__)),
        "config_sha256": cohort_ref["sha256"] if cohort_ref else "0" * 64,
        "input_sha256": candidate_ref["sha256"] if candidate_ref else "0" * 64,
    }
    task_results: list[dict[str, Any]] = []
    if cohort_ref is not None and candidate_ref is not None:
        for task in ("chips", "poker"):
            try:
                task_result, task_blockers = _task_outputs(
                    task, admitted, output, cohort_ref, producer_identity
                )
            except Exception as error:  # fail-closed publication boundary
                task_result = {"task": task, "pair_count": 0, "ledgers": None}
                task_blockers = [f"{task.upper()}_LEDGER_VALIDATION_FAILED:{error}"]
            task_results.append(task_result)
            blockers.extend(task_blockers)
    else:
        for task in ("chips", "poker"):
            task_results.append({"task": task, "pair_count": 0, "ledgers": None})
            blockers.append(f"{task.upper()}_NO_CURRENT_LEGAL_RAW_ROBOTIZED_PAIR")

    pair_audit = {
        "schema_version": "EXACT78_PAIR_AUDIT_V32",
        "status": "PASS" if admitted and not rejected else "PARTIAL_OR_BLOCKED",
        "cohort_manifest": cohort_ref,
        "candidate_index": candidate_ref,
        "admitted": admitted,
        "rejected": rejected,
        "archive_consumed": False,
    }
    pair_audit_path = output / "PAIR_AUDIT_V32.json"
    atomic_json(pair_audit_path, pair_audit)
    blockers = sorted(set(blockers))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "READY_EXACT78_V32_DEVELOPMENT" if not blockers else "BLOCKED_INPUTS",
        "execution_allowed": not blockers,
        "training_started": False,
        "gpu_used": False,
        "archive_consumed": False,
        "blockers": blockers,
        "tasks": task_results,
        "raw_only_fallback": {
            "schema_version": contract.RAW_ONLY_SCHEMA,
            "activation": "SEPARATE_MANIFEST_REQUIRED_AFTER_REPRODUCIBLE_4H_BLOCKER",
            "formal_ab_allowed": False,
            "four_model_completion": False,
        },
        "outputs": {
            "raw_inventory": artifact(inventory_path),
            "current_rebind_audit": artifact(rebind_path),
            "pair_audit": artifact(pair_audit_path),
        },
        "claim_limit": (
            "READY authorizes V3.2 development Visual Aux training only; Raw-only "
            "fallback is an independent non-A/B engineering run."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT)
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
