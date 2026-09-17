#!/usr/bin/env python3
"""Audit the rectified-vs-selected Object6D world-pose error over a cohort."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chaoyang.ops.adapt_object6d_rectified_to_selected_v71 import adapt_poses, rotation_angle_deg


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(reference: dict[str, object]) -> Path:
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"missing absolute artifact: {path}")
    if path.stat().st_size != reference["bytes"] or sha(path) != reference["sha256"]:
        raise RuntimeError(f"artifact closure mismatch: {path}")
    return path


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def audit(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    spec_path = exact(result["inputs"]["spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    trajectory_path = exact(result["artifacts"]["trajectory"])
    registration_path = exact(spec["registration_authority"])
    c2w_path = exact(spec["camera_to_world"])
    with np.load(trajectory_path, allow_pickle=False) as source:
        frames = source["frame_indices"].astype(np.int64)
        valid = source["valid"].astype(bool)
        rectified = source["T_object_to_camera"].astype(np.float64)
        legacy_world = source["T_object_to_world"].astype(np.float64)
    with np.load(registration_path, allow_pickle=False) as registration:
        transform = registration["T_stereo_rectified_camera_to_selected_camera"].astype(np.float64)
    with np.load(c2w_path, allow_pickle=False) as camera:
        c2w = camera[spec["camera_to_world_key"]].astype(np.float64)
    _, corrected_world = adapt_poses(rectified, transform, c2w, frames, valid)
    translation = np.linalg.norm(
        corrected_world[valid, :3, 3] - legacy_world[valid, :3, 3], axis=1
    )
    return {
        "session_id": result["session_id"],
        "task": result["task"],
        "physical_object": result_path.parent.name,
        "valid_frames": int(valid.sum()),
        "registration_rotation_deg": rotation_angle_deg(transform[:3, :3]),
        "registration_translation_m": float(np.linalg.norm(transform[:3, 3])),
        "legacy_vs_corrected_world_translation_median_m": float(np.median(translation)),
        "legacy_vs_corrected_world_translation_p95_m": float(np.percentile(translation, 95)),
        "legacy_vs_corrected_world_translation_max_m": float(np.max(translation)),
        "source_result": str(result_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object6d-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"no-clobber output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    discovered = {
        path.resolve()
        for root in args.object6d_root
        for pattern in ("*/*/RESULT.json", "*/*/physical_object_*/RESULT.json")
        for path in root.glob(pattern)
    }
    candidates = []
    for path in sorted(discovered):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value.get("inputs"), dict) and isinstance(value.get("artifacts", {}).get("trajectory"), dict):
            candidates.append(path)
    rows, failures = [], []
    for path in candidates:
        try:
            rows.append(audit(path))
        except Exception as error:  # keep the cohort audit finite and explicit
            failures.append({"path": str(path.resolve()), "error": f"{type(error).__name__}: {error}"})
    values = np.asarray(
        [row["legacy_vs_corrected_world_translation_median_m"] for row in rows], dtype=np.float64
    )
    summary = {
        "instances_discovered": len(candidates),
        "instances_audited": len(rows),
        "instances_failed": len(failures),
        "sessions_audited": len({row["session_id"] for row in rows}),
        "median_instance_world_translation_shift_m": float(np.median(values)) if len(values) else None,
        "p95_instance_world_translation_shift_m": float(np.percentile(values, 95)) if len(values) else None,
        "max_instance_world_translation_shift_m": float(np.max(values)) if len(values) else None,
        "instances_median_shift_over_10mm": int((values > 0.010).sum()),
        "instances_median_shift_over_30mm": int((values > 0.030).sum()),
    }
    csv_path = args.output_dir / "OBJECT6D_COORDINATE_DOMAIN_AUDIT.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "schema_version": "object6d-coordinate-domain-audit-v71-v1",
        "artifact_revision": "R7_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DEVELOPMENT_COHORT_AUDIT" if rows and not failures else "PARTIAL_EXPLICIT_FAILURES",
        "summary": summary,
        "rows": rows,
        "failures": failures,
        "source_roots": [str(root.resolve()) for root in args.object6d_root],
        "claim_limit": (
            "Code-path coordinate discrepancy audit, not external pose truth. Legacy pinned results are not "
            "modified; Robot/Contact must use an explicit rectified-to-selected adapter."
        ),
    }
    atomic_json(args.output_dir / "OBJECT6D_COORDINATE_DOMAIN_AUDIT.json", payload)
    atomic_json(args.output_dir / "RESULT.json", payload)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if rows and not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
