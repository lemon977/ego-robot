#!/usr/bin/env python3
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Immutable CPU preflight/launcher boundary for the S1 mask challenger.

The current tool intentionally exposes only ``--preflight-only``.  Model
inference must be added as a separately reviewed leased adapter; silently
falling through to an unpinned SAM/Cutie environment is forbidden.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_challenger_v71 import (
    BACKENDS,
    FORBIDDEN_CLAIMS,
    canonical_sha256,
    file_sha256,
    preflight_backend,
    validate_prompt_manifest,
    validate_rgb_manifest,
)
from chaoyang.governance.v71_contracts import build_artifact_revision, publish_new_revision


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _artifact_pin(value: dict[str, Any], *, label: str) -> tuple[str, str]:
    artifact_id = value.get("artifact_id")
    revision = value.get("artifact_revision")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"{label} must pin artifact_id")
    if not isinstance(revision, str) or not revision.startswith("R7_"):
        raise ValueError(f"{label} must pin an R7_* artifact_revision")
    return artifact_id, revision


def build_preflight_receipt(
    *,
    rgb_path: Path,
    prompt_path: Path,
    requested_backends: list[str],
    artifact_revision: str,
    attempt_id: str,
    executor_epoch: int,
    fencing_token: str,
    repository_root: Path = ROOT,
    verify_large_sha: bool = False,
) -> dict[str, Any]:
    rgb = _load(rgb_path)
    prompts = _load(prompt_path)
    validation_errors = validate_rgb_manifest(rgb) + validate_prompt_manifest(prompts, rgb)
    try:
        input_artifacts = [
            _artifact_pin(rgb, label="rgb manifest"),
            _artifact_pin(prompts, label="prompt manifest"),
        ]
    except ValueError as exc:
        validation_errors.append(str(exc))
        input_artifacts = [
            ("INVALID_RGB_MANIFEST", "R7_0"),
            ("INVALID_PROMPT_MANIFEST", "R7_0"),
        ]

    backend_results = [
        preflight_backend(
            backend,
            repository_root=repository_root,
            verify_large_sha=verify_large_sha,
        )
        for backend in requested_backends
    ]
    if validation_errors:
        status = "BLOCKED_PREREQ"
    elif any(item["status"] == "BLOCKED_REFERENCE_PROOF" for item in backend_results):
        status = "BLOCKED_REFERENCE_PROOF"
    elif any(not item["execution_ready"] for item in backend_results):
        status = "BLOCKED_PREREQ"
    else:
        status = "PASSED"

    combined_input_sha = canonical_sha256(
        {
            "rgb_manifest_file_sha": file_sha256(rgb_path),
            "prompt_manifest_file_sha": file_sha256(prompt_path),
        }
    )
    source_sha = file_sha256(Path(__file__))
    pipeline_sha = file_sha256(ROOT / "src/chaoyang/pipeline/causal_modal_mask_challenger_v71.py")
    producer_signature = canonical_sha256(
        {
            "source_sha": source_sha,
            "pipeline_sha": pipeline_sha,
            "input_manifest_sha": combined_input_sha,
            "backends": backend_results,
            "artifact_revision": artifact_revision,
        }
    )
    payload = {
        "status": status,
        "session_id": str(rgb.get("session_id", "UNKNOWN")),
        "attempt_id": attempt_id,
        "executor_epoch": executor_epoch,
        "fencing_token_sha256": hashlib.sha256(fencing_token.encode("utf-8")).hexdigest(),
        "requested_backends": requested_backends,
        "backend_preflight": backend_results,
        "validation_errors": validation_errors,
        "execution_performed": False,
        "gpu_lease_acquired": False,
        "execution_contract": {
            "gpu_lease_schema": "chaoyang-gpu-lease-v71",
            "lease_required_for_inference": True,
            "task_attempt_and_fencing_must_match": True,
            "immutable_attempt_required": True,
            "current_ledger_write_allowed": False,
            "backend_environment_must_match_preflight": True,
        },
        "output_contract": {
            "schema": str((ROOT / "contracts/causal_modal_mask_output_v71.schema.json").resolve()),
            "per_frame_independent_instance_masks": True,
            "unknown_must_have_null_mask": True,
            "causal_prompt_policy": "only prompt events at the current frame are applied; no future event is visible",
        },
        "authority_limit": {
            "maximum_publication": "MODAL_MASK_SUCCESSOR_CANDIDATE",
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
        },
    }
    return build_artifact_revision(
        artifact_id=f"S1_MODAL_MASK_PREFLIGHT.{rgb.get('session_id', 'UNKNOWN')}.{attempt_id}",
        artifact_revision=artifact_revision,
        input_artifacts=input_artifacts,
        input_manifest_sha=combined_input_sha,
        producer_signature=producer_signature,
        payload=payload,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only immutable preflight for causal SAM2.1/Cutie modal-mask challengers"
    )
    parser.add_argument("--rgb-manifest", required=True, type=Path)
    parser.add_argument("--prompt-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--artifact-revision", default="R7_1")
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument(
        "--backend", choices=(*BACKENDS, "all"), default="all",
        help="Preflight one backend or the S1 comparison pair",
    )
    parser.add_argument("--verify-large-sha", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight_only:
        parser.error(
            "this reviewed launcher currently permits --preflight-only; GPU inference adapter is not enabled"
        )
    if args.executor_epoch < 1 or not args.fencing_token:
        parser.error("executor epoch and fencing token are mandatory")
    backends = list(BACKENDS) if args.backend == "all" else [args.backend]
    receipt = build_preflight_receipt(
        rgb_path=args.rgb_manifest.resolve(),
        prompt_path=args.prompt_manifest.resolve(),
        requested_backends=backends,
        artifact_revision=args.artifact_revision,
        attempt_id=args.attempt_id,
        executor_epoch=args.executor_epoch,
        fencing_token=args.fencing_token,
        verify_large_sha=args.verify_large_sha,
    )
    session_id = receipt["payload"]["session_id"]
    attempt_directory = (
        args.output_root.resolve() / "sessions" / session_id / "attempts" / args.attempt_id
    )
    if attempt_directory.exists():
        raise FileExistsError(f"immutable attempt already exists: {attempt_directory}")
    attempt_directory.mkdir(parents=True)
    try:
        publication = publish_new_revision(attempt_directory / "RESULT.json", receipt)
    except Exception:
        attempt_directory.rmdir()
        raise
    print(json.dumps({"receipt": publication, "payload": receipt["payload"]}, ensure_ascii=False, indent=2))
    return 0 if receipt["payload"]["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
