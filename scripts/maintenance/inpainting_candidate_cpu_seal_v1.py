"""Seal bounded candidate CPU evidence; never publish governance or run models."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


def owned_paths():
    paths = list((REPO_ROOT / "src/chaoyang/inpainting_candidate").glob("*.py"))
    paths += list((REPO_ROOT / "tests/unit").glob("test_inpainting_candidate_*.py"))
    paths += list((REPO_ROOT / "scripts/maintenance").glob("inpainting_candidate_*.py"))
    paths += [REPO_ROOT / "src/chaoyang/cli/inpainting_submit.py",
              REPO_ROOT / "src/chaoyang/governance/inpainting_candidate_registration.py"]
    return sorted(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args(argv)
    packet = load_json(args.packet)
    if packet.get("algorithm_execution_allowed") is not False or packet.get("deletion_allowed") is not False:
        raise ValueError("CPU_ONLY_PACKET_REQUIRED")
    root = Path(packet["output_dir"])
    if not root.is_relative_to(REPO_ROOT) or not args.junit.resolve().is_relative_to(root):
        raise ValueError("TASK_OUTPUT_REQUIRED")
    if (root / "ENGINEERING_RESULT.json").exists() or (root / "MAINTENANCE_DONE.json").exists():
        raise ValueError("IMMUTABLE_NEW_RESULT_REQUIRED")
    tree = ET.parse(args.junit)
    suites = list(tree.getroot().iter("testsuite"))
    counts = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
              for name in ("tests", "errors", "failures", "skipped")}
    if not counts["tests"] or any(counts[name] for name in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_CPU_TEST_PASS_REQUIRED")
    freeze = root / "input_freeze/freeze_0002/INPUT_FREEZE.json"
    value = load_json(freeze)
    if (value.get("verified_files") != 2388 or value.get("unique_raw_files") != 2388
            or value.get("regression_sessions") != 199 or value.get("regression_frames") != 29322
            or value.get("failures") != []):
        raise ValueError("FULL_FIXED_INPUT_FREEZE_REQUIRED")
    paths = owned_paths()
    snapshot = root / "CODE_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(REPO_ROOT)))
    result = {"schema": "inpainting-candidate-integration-evidence-v1",
        "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "status": "CPU_ENGINEERING_COMPLETE_WRAPPER_PUBLICATION_NEXT",
        "cpu_tests": counts, "junit": artifact_ref(args.junit),
        "input_freeze": artifact_ref(freeze),
        "input_freeze_status": value.get("verification"),
        "source_counts": {key: value[key] for key in ("verified_files", "verified_bytes", "regression_sessions", "regression_frames")},
        "retained_environment_diagnostic": artifact_ref(root / "CPU_TESTS.xml") if (root / "CPU_TESTS.xml").exists() and args.junit.name != "CPU_TESTS.xml" else None,
        "environment_note": "Sandbox PID/proc namespace mismatch retained separately; final live-writer resource tests run on host.",
        "code": [artifact_ref(path) for path in paths], "code_snapshot": artifact_ref(snapshot),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1",
        "model_calls": 0, "videos_generated": 0, "files_deleted": 0,
        "visual_quality": "NOT_EVALUATED", "training_eligible": False,
        "candidate_adopted": False, "completed_at": now_iso(),
        "next_task": "Register G0 publication, bind exact wrapper SHA, prepare native Source, then independently register G1 model execution."}
    atomic_json(root / "ENGINEERING_RESULT.json", result)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    old = load_json(index_path)
    index = {**old, "g0_integration": artifact_ref(root / "ENGINEERING_RESULT.json"),
        "g0_integration_task": artifact_ref(args.packet),
        "input_freeze": result["input_freeze"],
        "next_actions": {"source": "Seal six original RGB stages in publication packet",
                         "mask": "Run G1 under exact frozen operation and one physical H20 lease",
                         "clean": "Run independent single-reference-selection experiment with sealed Raw/Mask",
                         "occlusion": "Bind full link render support to raw visible evidence",
                         "publisher": "Publish explicit R03 wrapper supplement, preserve immutable V1"}}
    atomic_json(index_path, index)
    atomic_json(root / "MAINTENANCE_DONE.json", {
        "task_id": packet["task_id"], "run_signature": packet["run_signature"], "status": "PASS",
        "reason": "Finite CPU integration scope sealed; real model and wrapper publication remain separate tasks.",
        "artifacts": [artifact_ref(root / "ENGINEERING_RESULT.json"), result["junit"],
                      result["input_freeze"], result["code_snapshot"]],
        "cpu_tests_passed": counts["tests"], "model_calls": 0, "videos_generated": 0,
        "files_deleted": 0, "visual_quality": "NOT_EVALUATED", "candidate_adopted": False})
    print(artifact_ref(root / "ENGINEERING_RESULT.json"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
