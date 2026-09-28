#!/usr/bin/env python3
"""Final structural validation for the bounded Human-to-Robot R2 attempt."""
from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chaoyang.ops.run_human_to_robot_r2_candidate_product import CASES, product_binding
from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, TASK, ref, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_video(item: dict, expected: int) -> dict:
    path = Path(item["path"])
    if ref(path) != item:
        raise ValueError(f"VIDEO_SHA_DRIFT:{path}")
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if process.returncode or process.stdout.strip() != str(expected):
        raise ValueError(f"VIDEO_FRAME_COUNT:{path}:{process.stdout.strip()}:{expected}")
    return {"video": item, "expected_frames": expected, "decoded_frames": expected}


def main() -> int:
    junit = ATTEMPT / "PYTEST_FULL.xml"
    root = ET.parse(junit).getroot()
    # JUnit permits either a single ``testsuite`` root or a ``testsuites``
    # container.  Pytest currently emits the latter, whose aggregate counts
    # are carried by its child suite rather than the container itself.
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError(f"PYTEST_JUNIT_WITHOUT_SUITES:{junit}")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    if failures or errors or tests - skipped != 1489:
        raise ValueError(f"PYTEST_RESULT:{tests}:{failures}:{errors}:{skipped}")

    videos = []
    scene_specs = (("031", "wave5", 149), ("007", "wave4", 378),
                   ("103", "wave4", 284), ("042", "wave4", 171))
    for short, wave, count in scene_specs:
        result = load(ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{wave}/RESULT.json")
        videos.append({"role": "SCENE_CANDIDATE", "session_id": result["session_id"],
                       **decode_video(result["review"]["video"], count)})
    for short, case in CASES.items():
        root = ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6"
        result = load(root / "RESULT.json")
        videos.append({"role": "PRODUCT_CANDIDATE", "session_id": result["session_id"],
                       **decode_video(result["video"], int(case["frames"]))})
        prep_root = ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{case['scene_wave']}"
        prep = load(prep_root / "SCENE_PREP_MANIFEST.json")
        current = product_binding(
            scene_path=prep_root / "RESULT.json",
            robot_path=ATTEMPT / f"lanes/lane2_motion/{case['motion']}/ROBOT_R0_V1.npz",
            domain_path=Path(prep["source_domain"]["path"]),
        )
        binding = load(root / "CACHE_BINDING_V2.json")
        if current["signature_sha256"] != binding.get("signature_sha256"):
            raise ValueError(f"PRODUCT_BINDING_DRIFT:{short}")
    for short, wave, count in (("097", "wave0", 165), ("098", "wave1", 179), ("101", "wave1", 122)):
        result = load(ATTEMPT / f"lanes/lane3_sensor/run_{short}_{wave}/RESULT.json")
        videos.append({"role": "SENSOR_BACKEND_REVIEW", "session_id": result["session_id"],
                       **decode_video(result["review"]["video"], count)})
        for name in ("HAND_MOTION_V1.npz", "KAI22_COMMON_BACKEND_V1.npz"):
            path = ATTEMPT / f"lanes/lane3_sensor/run_{short}_{wave}" / name
            with np.load(path, allow_pickle=False) as archive:
                if len(archive.files) == 0:
                    raise ValueError(f"EMPTY_NPZ:{path}")
    for short, count in (("031", 149), ("007", 378)):
        result = load(ATTEMPT / f"lanes/lane4_compare/same_rejected_background_{short}_wave10/RESULT.json")
        videos.append({"role": "LOCAL_HURO_SAME_REJECTED_BACKGROUND", "session_id": result["session_id"],
                       **decode_video(result["video"], count)})
    if len(videos) != 13:
        raise ValueError("DELIVERED_VIDEO_COUNT")

    required = [ATTEMPT / "RUN_SIGNATURE.json", ATTEMPT / "PROGRESS_2H.json",
                *(ATTEMPT / f"lanes/{lane}/STATE.json" for lane in
                  ("lane1_scene", "lane2_motion", "lane3_sensor", "lane4_compare"))]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = {
        "schema_version": "HUMAN_TO_ROBOT_R2_FINAL_VALIDATION_V1",
        "task_id": TASK,
        "created_at": now(),
        "status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
        "pytest": {"tests": tests, "passed": tests - skipped, "skipped": skipped,
                   "failures": failures, "errors": errors, "junit": ref(junit)},
        "videos": videos,
        "video_count": len(videos),
        "video_full_decode_pass": True,
        "product_cache_binding_pass": 4,
        "governance_expected": "PASS_FRESH",
        "quality_summary": {"scene_pass": 0, "product_pass": 0, "robot_r1_adopted": 0,
                            "sensor_kinematic_only": 3, "formal_clean": 0},
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    out = ATTEMPT / "FINAL_VALIDATION.json"
    write_json(out, validation)
    print(json.dumps({"status": validation["status"], "tests_passed": tests - skipped,
                      "videos": len(videos), "output": ref(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
