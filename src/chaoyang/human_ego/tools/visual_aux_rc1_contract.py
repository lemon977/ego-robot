#!/usr/bin/env python3
"""Validate the receipt-bound RC1 Visual Aux consumer eligibility sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema


PROJECT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = PROJECT / "contracts/visual_aux_rc1_eligibility.schema.json"
CURRENT_CONTRACT_PATH = PROJECT / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json"
RELEASE_ID = "chaoyang-visual-aux-rc1-final"


class RC1ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RC1ContractError(f"JSON object required: {path}")
    return value


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def exact_ref(item: dict[str, Any], label: str) -> Path:
    path = Path(str(item.get("path", "")))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RC1ContractError(f"{label}: absolute regular file required")
    if path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
        raise RC1ContractError(f"{label}: byte/SHA mismatch")
    return path


def _same_ref(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "bytes", "sha256"))


def _reject_review_media(path: Path, label: str) -> None:
    lowered = str(path).lower()
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise RC1ContractError(f"{label}: review/video media is forbidden")
    if "/docs/current/visuals/" in lowered:
        raise RC1ContractError(f"{label}: current_visuals is review-only")


def validate_rc1_eligibility(
    eligibility_path: Path,
    *,
    expected_manifest: Path | None = None,
    expected_contract: Path = CURRENT_CONTRACT_PATH,
) -> dict[str, Any]:
    eligibility_path = eligibility_path.resolve(strict=True)
    value = load_json(eligibility_path)
    schema = load_json(SCHEMA_PATH)
    jsonschema.Draft202012Validator(schema).validate(value)
    if value.get("release_id") != RELEASE_ID:
        raise RC1ContractError("RC1 release identity mismatch")

    contract_ref = artifact_ref(expected_contract)
    if not _same_ref(value["contract"], contract_ref):
        raise RC1ContractError("eligibility is bound to a stale RC1 contract")
    exact_ref(value["contract"], "RC1 contract")

    consumer = value["consumer_eligibility"]["visual_aux_rc1"]
    eligible = bool(consumer["eligible"])
    if eligible != bool(value["train_eligible"]):
        raise RC1ContractError("consumer eligibility and train_eligible disagree")
    if eligible and (
        value["execution_status"] != "COMPLETED"
        or value["visual_quality"]
        not in {"HIGH_COVERAGE_VISUAL", "PARTIAL_TRAINABLE"}
        or consumer["reasons"]
    ):
        raise RC1ContractError("eligible result has incompatible state or reasons")
    if not eligible:
        raise RC1ContractError("formal loader rejects ineligible RC1 sidecars")

    resolved: dict[str, Path] = {}
    for key, item in value["artifacts"].items():
        resolved[key] = exact_ref(item, key)
    if expected_manifest is not None and resolved["manifest"] != expected_manifest.resolve(strict=True):
        raise RC1ContractError("eligibility/manifest binding mismatch")

    manifest = load_json(resolved["manifest"])
    if (
        manifest.get("session_id") != value["session_id"]
        or manifest.get("task") != value["task"]
        or manifest.get("split") != value["split"]
        or manifest.get("input_mode") != "CAUSAL_TRAINING_INPUT"
    ):
        raise RC1ContractError("eligibility identity/input-mode mismatch")
    if manifest.get("control_ground_truth") is not False or manifest.get(
        "physical_deployment_authorized"
    ) is not False:
        raise RC1ContractError("RC1 manifest exceeds the visual-only claim boundary")

    for key in ("raw_selector", "robotized_selector"):
        selector = load_json(resolved[key])
        for row in selector.get("frames", []):
            rgb = row.get("rgb")
            if not isinstance(rgb, dict):
                raise RC1ContractError(f"{key}: missing RGB artifact")
            rgb_path = exact_ref(rgb, f"{key}:rgb")
            _reject_review_media(rgb_path, key)

    proof = load_json(resolved["input_payload_receipt"])
    required_true = (
        "repeatability_pass",
        "future_mutation_test_pass",
        "source_max_frame_le_target",
        "shared_neutralization_pass",
    )
    if proof.get("status") != "PASSED" or any(proof.get(key) is not True for key in required_true):
        raise RC1ContractError("input payload causal proof is incomplete")
    if proof.get("input_mode") != "CAUSAL_TRAINING_INPUT":
        raise RC1ContractError("input payload proof is not causal")

    return {
        "status": "PASS_VISUAL_AUX_RC1_ELIGIBILITY",
        "task": value["task"],
        "session_id": value["session_id"],
        "source_group_id": value["source_group_id"],
        "split": value["split"],
        "eligibility_sha256": sha256(eligibility_path),
        "contract_sha256": contract_ref["sha256"],
        "artifacts": {key: str(path) for key, path in resolved.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eligibility", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = validate_rc1_eligibility(
        args.eligibility,
        expected_manifest=args.manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
