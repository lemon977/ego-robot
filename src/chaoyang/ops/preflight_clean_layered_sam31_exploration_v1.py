#!/usr/bin/env python3
"""Fail-closed preflight for the isolated SAM3.1 layered Clean exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[3]
EXPECTED = {
    "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt": (
        3_502_755_717,
        "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
    ),
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py": (
        40_110,
        "5c6513c0bdd8fc29bf9248c8ca007b492e0d3d3051baf16f5c1cf870d7c48be2",
    ),
    "vendor/ProPainter/weights/ProPainter.pth": (
        157_780_510,
        "12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591",
    ),
    "vendor/ProPainter/weights/raft-things.pth": (
        21_108_000,
        "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
    ),
}
BASELINE_ROOTS = (
    PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1",
    PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1",
)
EXPLORATION_ROOT = (
    PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> Any:
    return json.loads((PROJECT / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPLORATION_ROOT / "PREFLIGHT.json",
    )
    args = parser.parse_args()

    status = load_json("docs/governance/CURRENT_PROJECT_STATUS_MIN.json")
    registry = load_json("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
    packet = load_json(
        "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/"
        "TASK_PACKET.json"
    )
    matrix = load_json(
        "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/"
        "EXPERIMENT_MATRIX.json"
    )

    clean_entries = [row for row in registry["entries"] if row.get("stage") == "Clean"]
    errors: list[str] = []
    if len(clean_entries) != 1:
        errors.append(f"expected one Clean registry entry, got {len(clean_entries)}")
        clean_entry = {}
    else:
        clean_entry = clean_entries[0]
    if clean_entry.get("algorithm_id") != "same_pixel_temporal_donor_then_propainter_v1":
        errors.append("frozen Clean baseline algorithm_id drifted")
    if clean_entry.get("current_counts", {}).get("passed") != 58:
        errors.append("frozen Clean baseline passed count is not 58")
    if status.get("freshness", {}).get("status") not in {"FRESH", "DELAYED"}:
        errors.append("governance is not fresh enough for a new canary")
    if packet["write_set"] != [
        "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/attempts/"
    ]:
        errors.append("unexpected exploration write_set")
    if matrix["baseline_isolation"].get("allow_overwrite") is not False:
        errors.append("baseline overwrite must be false")

    output_resolved = args.output.resolve()
    if not output_resolved.is_relative_to(EXPLORATION_ROOT.resolve()):
        errors.append("preflight output escapes exploration root")
    for root in BASELINE_ROOTS:
        if output_resolved.is_relative_to(root.resolve()):
            errors.append(f"preflight output overlaps baseline root: {root}")

    assets = []
    for relative, (expected_bytes, expected_sha) in EXPECTED.items():
        path = PROJECT / relative
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha = sha256(path) if exists and actual_bytes == expected_bytes else None
        match = bool(
            exists and actual_bytes == expected_bytes and actual_sha == expected_sha
        )
        if not match:
            errors.append(f"asset identity mismatch: {relative}")
        assets.append(
            {
                "path": str(path),
                "exists": exists,
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "expected_bytes": expected_bytes,
                "expected_sha256": expected_sha,
                "exact_match": match,
            }
        )

    result = {
        "schema_version": "clean-layered-sam31-exploration-preflight-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS" if not errors else "FAIL",
        "governance_revision_observed": status.get("governance_revision"),
        "baseline": {
            "algorithm_id": clean_entry.get("algorithm_id"),
            "passed": clean_entry.get("current_counts", {}).get("passed"),
            "write_overlap": False,
            "modified_by_preflight": False,
        },
        "assets": assets,
        "model_policy": {
            "segmentation_required": "SAM3.1",
            "residual_inpaint_required": "ProPainter",
            "sam2_fallback": False,
            "e2fgvi_primary": False,
            "lama_or_telea_fallback": False,
        },
        "execution": {
            "gpu_inference_started": False,
            "frozen_frameset_published": False,
            "next": "publish frozen frameset, then acquire central GPU lease for T0",
        },
        "errors": errors,
        "claim_limit": (
            "Asset and isolation preflight only; no SAM3.1 inference, Clean successor "
            "quality claim, authority promotion, or hidden-object truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "errors": errors}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
