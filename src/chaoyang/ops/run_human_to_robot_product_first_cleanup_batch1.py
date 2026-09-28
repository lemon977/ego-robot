"""Audited first-batch purge of five terminal R2 ProPainter *intermediate* trees.

Final Clean PNGs, prep inputs/masks, logs, weights, source videos, review videos,
historical receipts and the complete current consumer chain are retained.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import cv2

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

OLD = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene"
NAMES = ("007_wave4", "031_wave4", "031_wave5", "042_wave4", "103_wave4")
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/cleanup/batch1"


def _json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _event(value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with (OUT / "DELETE_EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def _tree(path: Path) -> dict:
    digest = hashlib.sha256()
    count, total, allocated = 0, 0, 0
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise RuntimeError(f"SYMLINK_IN_DELETE_TREE:{item}")
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        info = item.stat()
        child = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                child.update(block)
        digest.update(f"{relative}\0{info.st_size}\0{child.hexdigest()}\n".encode())
        count += 1; total += info.st_size; allocated += info.st_blocks * 512
    return {"files": count, "logical_bytes": total, "allocated_bytes": allocated,
            "tree_sha256": digest.hexdigest()}


def isolate_verified(source: Path, destination: Path, identity: dict) -> None:
    """Fail closed if bytes changed or a previous isolation target exists."""
    if destination.exists() or destination.is_symlink() or not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"ISOLATION_PATH_STATE:{source}:{destination}")
    if _tree(source) != identity:
        raise RuntimeError(f"OBJECT_CHANGED_BEFORE_ISOLATION:{source}")
    os.replace(source, destination)


def rollback_verified(source: Path, destination: Path, identity: dict) -> None:
    """Never overwrite a newer object appearing at the original path."""
    if source.exists() or source.is_symlink() or not destination.is_dir() or _tree(destination) != identity:
        raise RuntimeError(f"ROLLBACK_BLOCKED_NEW_CONTENT_OR_IDENTITY:{source}")
    os.replace(destination, source)


def _consumer_check() -> dict:
    checks: dict[str, object] = {}
    for name in NAMES:
        parent = OLD / f"clean_candidate_{name}"
        prep = load_json(parent / "SCENE_PREP_MANIFEST.json")
        result = load_json(parent / "RESULT.json")
        count = int(prep["frame_count"])
        if int(result["frame_count"]) != count:
            raise RuntimeError(f"HISTORICAL_FRAME_COUNT:{name}")
        for local in (0, count // 2, count - 1):
            raw = cv2.imread(str(prep["rows"][local]["raw"]), cv2.IMREAD_COLOR)
            clean = cv2.imread(str(parent / "clean" / f"{local:06d}.png"), cv2.IMREAD_COLOR)
            model_mask = cv2.imread(str(parent / "prep/model_masks" / f"{local:06d}.png"), cv2.IMREAD_UNCHANGED)
            if raw is None or clean is None or model_mask is None or raw.shape != clean.shape:
                raise RuntimeError(f"REPRO_INPUT_NOT_LOADABLE:{name}:{local}")
        review = cv2.VideoCapture(str(parent / "SCENE_CLEAN_CANDIDATE_REVIEW.mp4"))
        decoded = 0
        while review.read()[0]:
            decoded += 1
        review.release()
        if decoded != count:
            raise RuntimeError(f"OLD_REVIEW_INCOMPLETE:{name}:{decoded}/{count}")
        checks[name] = {"prep_count": count, "review_decoded": decoded,
                        "clean_samples_loaded": [0, count // 2, count - 1]}
    new = ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
    for relative in ("lanes/scene/cable_007/window_v1/RESULT.json",
                     "lanes/scene/cable_007/clean_window_v1/RESULT.json",
                     "lanes/scene/object_patch_031/window_v1/RESULT.json",
                     "lanes/motion_product/robot_correspondence_031/window_v1/RESULT.json"):
        if not (new / relative).is_file():
            raise RuntimeError(f"CURRENT_CONSUMER_MISSING:{relative}")
    validation = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                       "TMPDIR": str(ROOT / ".cache/tmp"), "XDG_CACHE_HOME": str(ROOT / ".cache/xdg")},
        text=True, capture_output=True, check=False,
    )
    if validation.returncode or '"status": "PASS"' not in validation.stdout:
        raise RuntimeError(f"GOVERNANCE_AFTER_ISOLATION:{validation.stdout[-1000:]}:{validation.stderr[-1000:]}")
    checks["governance"] = "PASS"
    checks["current_outputs"] = "FOUR_RESULTS_PRESENT"
    return checks


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in NAMES:
        path = OLD / f"clean_candidate_{name}" / "upstream"
        if not path.is_dir() or path.is_symlink() or not (path.parent / "RESULT.json").is_file():
            raise RuntimeError(f"UNSAFE_TARGET:{path}")
    # No raw/processed/sealed/archive/project-external or current attempt target.
    OUT.mkdir(parents=True)
    (OUT / "isolation").mkdir()
    retention = {"schema_version": "PRODUCT_FIRST_CLEANUP_RETENTION_V1", "task_id": TASK,
                 "protected": ["raw", "processed", "sealed", "archive", "project_external",
                               "all_model_weights", "mount_and_robot_CAD", "current_attempt",
                               "all_five_historical_clean_frames", "all_five_historical_prep_inputs_masks",
                               "all_five_historical_reviews_receipts_logs"],
                 "conditional_future_dependencies": ["007 full Clean if fixed-window quality permits",
                    "031/007 product and 0902 regressions consume retained Clean, not terminal upstream frames",
                    "object patch, Robot correspondence and adapter collision consume retained geometry/assets"],
                 "target_rule": "Only terminal R2 ProPainter uncomposited upstream intermediates; no active or conditional task requires the deleted copy.",
                 "historical_result_status": "UNCHANGED_BYTES_BUT_UPSTREAM_PAYLOAD_TRIMMED"}
    comparison = {"schema_version": "PRODUCT_FIRST_CLEANUP_MINIMAL_REPRO_V1",
                  "old_new_comparison": ["R2 old Clean PNGs and MP4", "new 007 fixed-window Clean PNGs and MP4"],
                  "007_repro": ["Raw source frames", "R2 prep and candidate masks", "new cable prep", "pinned ProPainter weights/code/config", "old and new invocation logs"],
                  "031_repro": ["frozen target and q", "complete mount contract", "validity and camera", "retained old Clean and geometry"],
                  "geometry_repro": ["031 object masks", "Depth and K", "Robot q/FK", "STEP-derived mesh", "query settings"],
                  "authority_limit": "Historical upstream generated frames no longer byte-reloadable; full invocation remains reproducible but costs compute."}
    _json(OUT / "RETENTION_CURRENT.json", retention)
    _json(OUT / "COMPARISON_REPRO.json", comparison)
    selected = []
    for name in NAMES:
        source = OLD / f"clean_candidate_{name}" / "upstream"
        selected.append({"name": name, "source": str(source), "identity": _tree(source)})
    _json(OUT / "SELECTED_TARGETS.json", {"schema_version": "PRODUCT_FIRST_CLEANUP_SELECTED_V1", "targets": selected})
    for row in selected:
        _event({"name": row["name"], "state": "SELECTED", "source": row["source"], "identity": row["identity"]})
    isolated = []
    try:
        for row in selected:
            source = Path(row["source"])
            destination = OUT / "isolation" / row["name"]
            isolate_verified(source, destination, row["identity"])
            isolated.append((row, destination))
            _event({"name": row["name"], "state": "ISOLATED", "isolation": str(destination)})
        checks = _consumer_check()
        _json(OUT / "CONSUMER_CHECKS.json", checks)
        for row, _ in isolated:
            _event({"name": row["name"], "state": "CONSUMER_CHECKED", "checks": "PASS"})
    except Exception:
        for row, destination in reversed(isolated):
            source = Path(row["source"])
            try:
                rollback_verified(source, destination, row["identity"])
                _event({"name": row["name"], "state": "ROLLED_BACK"})
            except RuntimeError:
                _event({"name": row["name"], "state": "ROLLBACK_BLOCKED_NEW_CONTENT_OR_IDENTITY"})
        raise
    for row, destination in isolated:
        if _tree(destination) != row["identity"] or Path(row["source"]).exists():
            raise RuntimeError(f"PURGE_IDENTITY_OR_NEW_CONTENT:{row['name']}")
        shutil.rmtree(destination)
        _event({"name": row["name"], "state": "PURGED", "logical_bytes": row["identity"]["logical_bytes"]})
    result = {"schema_version": "PRODUCT_FIRST_CLEANUP_BATCH1_RECEIPT_V1", "task_id": TASK,
              "execution": "ACTUALLY_PURGED", "selected_count": len(selected),
              "purged": selected, "logical_deleted_bytes": sum(row["identity"]["logical_bytes"] for row in selected),
              "allocated_deleted_bytes_estimate": sum(row["identity"]["allocated_bytes"] for row in selected),
              "isolated_remaining_bytes": 0, "physical_free_space_change": "UNKNOWN_SHARED_CPFS",
              "project_net_occupancy_change": "NOT_MEASURED_CONCURRENT_WRITES",
              "retention": artifact_ref(OUT / "RETENTION_CURRENT.json"),
              "comparison_repro": artifact_ref(OUT / "COMPARISON_REPRO.json"),
              "selected_targets": artifact_ref(OUT / "SELECTED_TARGETS.json"),
              "consumer_checks": artifact_ref(OUT / "CONSUMER_CHECKS.json"),
              "historical_receipts_modified": False,
              "historical_payload_notice": "R2 ProPainter upstream uncomposited intermediates trimmed; old Clean/prep/reviews/logs and model replay inputs retained.",
              "claim_limit": "Only named, verified terminal intermediate trees purged; no raw, processed, sealed, archive, model, CAD, current output or final Clean deleted."}
    _json(OUT / "DELETE_RECEIPT.json", result)
    print(json.dumps({"status": result["execution"], "count": len(selected),
                      "logical_deleted_bytes": result["logical_deleted_bytes"],
                      "receipt": str(OUT / "DELETE_RECEIPT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
