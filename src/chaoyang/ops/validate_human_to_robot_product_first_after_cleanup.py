"""Recheck all 15 video slots, new evidence, and retained runtime after purges."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json, validate_artifact_ref
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

ATTEMPT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUT = ATTEMPT / "validation_after_cleanup/RESULT.json"
OLD_DELIVERY = ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/delivery/DELIVERY_MANIFEST.json"


def decode(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def _environment(name: str) -> dict:
    interpreter = ROOT / f"_run/current/environments/{name}/bin/python"
    if not interpreter.is_file():
        raise RuntimeError(f"ENVIRONMENT_MISSING:{name}")
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(ROOT / ".cache/tmp"),
                       XDG_CACHE_HOME=str(ROOT / ".cache/xdg"),
                       PYTHONPYCACHEPREFIX=str(ROOT / ".cache/pycache"))
    check = subprocess.run([str(interpreter), "-c", "import cv2, numpy, torch; print('MODULES_OK')"],
                           cwd=ROOT, env=environment, capture_output=True, text=True,
                           timeout=120, check=False)
    if check.returncode or "MODULES_OK" not in check.stdout:
        raise RuntimeError(f"ENVIRONMENT_IMPORT_FAILED:{name}:{check.stderr[-700:]}")
    return {"name": name, "python": str(interpreter), "modules": "cv2,numpy,torch",
            "exit_code": check.returncode}


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    old = load_json(OLD_DELIVERY)
    if old.get("slot_count") != 15 or len(old.get("slots", [])) != 15:
        raise RuntimeError("OLD_DELIVERY_SLOT_COUNT")
    formal = load_json(ATTEMPT / "lanes/motion_product/formal_007_current/attempt_0001/PRODUCT_RESULT.json")
    checked = []
    for row in old["slots"]:
        expected = int(row["expected_frames"])
        ref = (formal["product_video"] if row["slot_id"] == "PRODUCT_007" else row["video"])
        errors = validate_artifact_ref(ref)
        if errors:
            raise RuntimeError(f"SLOT_SHA_INVALID:{row['slot_id']}:{errors}")
        actual = decode(Path(ref["path"]))
        if actual != expected:
            raise RuntimeError(f"SLOT_DECODE:{row['slot_id']}:{actual}/{expected}")
        checked.append({"slot_id": row["slot_id"], "session_id": row["session_id"],
                        "status": "NEW_RENDER_ONLY_IDENTICAL_SHA" if row["slot_id"] == "PRODUCT_007" else "REUSED",
                        "quality": formal["quality"] if row["slot_id"] == "PRODUCT_007" else row["quality"],
                        "expected_frames": expected, "decoded_frames": actual, "video": ref})
    fixed = []
    for relative, expected in (
        ("lanes/scene/cable_007/clean_window_v1/007_CABLE_OLD_NEW_CLEAN_REVIEW.mp4", 16),
        ("lanes/motion_product/product_007/window_v1/007_OLD_NEW_CLEAN_ROBOT_WINDOW.mp4", 16),
    ):
        path = ATTEMPT / relative
        count = decode(path)
        if count != expected:
            raise RuntimeError(f"FIXED_WINDOW_DECODE:{relative}:{count}/{expected}")
        fixed.append({"video": artifact_ref(path), "decoded_frames": count})
    first = load_json(ATTEMPT / "cleanup/batch1/DELETE_RECEIPT.json")
    second = load_json(ATTEMPT / "cleanup/batch1b/DELETE_RECEIPT.json")
    if any(Path(row["source"]).exists() for row in first["purged"]) or Path(second["target"]).exists():
        raise RuntimeError("PURGED_TARGET_REAPPEARED")
    selection = load_json(Path(second["selection"]["path"]))
    for ref in (*selection["retained_samples"].values(), selection["retained_video"]):
        if validate_artifact_ref(ref):
            raise RuntimeError("FORENSIC_RETENTION_INVALID")
    for ref in selection["retained_samples"].values():
        with np.load(ref["path"], allow_pickle=False) as archive:
            if not archive.files:
                raise RuntimeError("FORENSIC_NPZ_EMPTY")
    if decode(Path(selection["retained_video"]["path"])) != selection["retained_video_decoded_frames"]:
        raise RuntimeError("FORENSIC_VIDEO_INCOMPLETE")
    envs = [_environment(name) for name in ("hawor-py310-v1", "foundationstereo-py311-v1")]
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "validate-governance"]
    validation = subprocess.run(command, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
             "TMPDIR": str(ROOT / ".cache/tmp"), "XDG_CACHE_HOME": str(ROOT / ".cache/xdg")},
        capture_output=True, text=True, check=False)
    if validation.returncode or '"status": "PASS"' not in validation.stdout:
        raise RuntimeError(f"GOVERNANCE_INVALID:{validation.stdout[-700:]}")
    result = {"schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_POST_CLEANUP_VALIDATION_V1",
              "task_id": TASK, "status": "PASS_FOR_CHECKED_SCOPE", "slot_count": len(checked),
              "slots": checked, "new_fixed_windows": fixed, "retained_environments": envs,
              "cleanup": artifact_ref(ATTEMPT / "cleanup/DELETE_RECEIPT.json"),
              "formal_resume": artifact_ref(ATTEMPT / "lanes/motion_product/formal_007_current/RESUME_RECEIPT.json"),
              "governance_status": "PASS", "protected_root_snapshot_before_after": "NOT_ESTABLISHED_AT_T0",
              "claim_limit": "This rechecks retained files, not product quality or a full pre/post source-tree hash. All products remain rejected quality.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    OUT.parent.mkdir(parents=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "slots": len(checked),
                      "result": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
