"""Seal bounded CPU engineering evidence; never claim model or visual success."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


def seal(packet: dict) -> dict:
    output = Path(packet["output_dir"]).resolve()
    if not output.is_relative_to(REPO_ROOT) or (output / "ENGINEERING_RESULT.json").exists():
        raise ValueError("FRESH_PROJECT_EVIDENCE_REQUIRED")
    preflight = load_json(output / "preflight/PREFLIGHT.json")
    if preflight["errors"] or preflight["task_id"] != packet["task_id"]:
        raise ValueError("PROTECTED_SOURCE_SHA_DRIFT")
    suite = ET.parse(output / "CPU_TESTS.xml").getroot()
    cases = list(suite.iter("testcase"))
    failures = [case.attrib.get("name") for case in cases if case.find("failure") is not None or case.find("error") is not None]
    skipped = [case.attrib.get("name") for case in cases if case.find("skipped") is not None]
    if failures or skipped or len(cases) != 115:
        raise ValueError("FINAL_CPU_TEST_COUNTS_OR_RESULTS_CHANGED")
    parent = load_json(REPO_ROOT / "inpainting_v1/MANIFEST.json")
    old_runtime = [Path(row["path"]) for row in parent["runtime_closure"] if Path(row["path"]).suffix == ".py"]
    own_code = sorted((REPO_ROOT / "src/chaoyang/inpainting_candidate").glob("*.py"))
    own_tests = sorted((REPO_ROOT / "tests/unit").glob("test_inpainting_candidate_*.py"))
    docs = [REPO_ROOT / "inpainting_optimization" / name for name in ("G0_SPEC.json", "00_EXECUTION_ZH.md", "README.md", "IMPLEMENTATION_ZH.md")]
    paths = sorted(set(old_runtime + own_code + own_tests + docs + [REPO_ROOT / "scripts/maintenance/inpainting_optimization_task_v1.py"]))
    refs = [artifact_ref(path) for path in paths]
    # Preserve actual dirty-tree bytes, not just a HEAD which lacks these files.
    snapshot = output / "CODE_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for reference in refs:
            path = Path(reference["path"])
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != reference["sha256"]:
                raise ValueError("CODE_CHANGED_DURING_SEAL:" + str(path))
            archive.writestr(str(path.relative_to(REPO_ROOT)), body)
    for reference in packet["inputs"].values():
        if artifact_ref(Path(reference["path"])) != reference:
            raise ValueError("PROTECTED_INPUT_CHANGED_DURING_SEAL")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    source_diagnostic = artifact_ref(output / "SOURCE_0918_001.json")
    actual_source = load_json(output / "SOURCE_0918_001.json")
    result = {
        "schema": "inpainting-g0-engineering-evidence-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "created_at": now_iso(),
        "engineering_execution": "COMPLETED_CPU_ONLY",
        "g0_full_admission": "BLOCKED_STORAGE_INPUT_IO_AND_ENTRY_INTEGRATION_PENDING",
        "tests": {"total": len(cases), "passed": len(cases), "failed": 0, "skipped": 0, "junit": artifact_ref(output / "CPU_TESTS.xml"), "real_model_calls": 0},
        "baseline_code_checks": preflight["groups"], "preflight": artifact_ref(output / "preflight/PREFLIGHT.json"),
        "code_files": refs, "code_snapshot": artifact_ref(snapshot),
        "git": {"head": head, "commit_status": "NOT_COMMITTED_EXISTING_DIRTY_TREE_PRESERVED", "actual_dirty_bytes_archived": True, "git_seal_complete": False},
        "raw199": {"sessions": 199, "frames": 29322, "manifest_bound": parent["inputs"]["raw_manifest"], "actual_raw_file_sha_refresh": "NOT_COMPLETED_NAS_RPC_WAIT", "target_file_count": 2388, "partial_consumed": False},
        "source0918": {"diagnostic": source_diagnostic, "native_qpc_frames": actual_source["native_sidecar_frames"], "decoded_frames": actual_source["decoded_frames"], "physical_eye": actual_source["camera"]["physical_eye"], "sbs_index": actual_source["camera"]["sbs_index"], "motion_validation": "NOT_EVALUATED", "camera_projection": "UNVERIFIED", "timebase": "TIMEBASE_UNVERIFIED"},
        "resources": {"storage": preflight["storage"], "gpu": preflight["gpu_query"], "actual_gpu_count": 1, "multigpu": "SYNTHETIC_ISOLATION_TESTED_NOT_REAL_TWO_CARD_LOAD"},
        "entry": "NOT_ROUTED_NOT_PRODUCTION", "g1": "NOT_REGISTERED", "g2": "NOT_REGISTERED", "g3": "NOT_REGISTERED", "g4": "NOT_REGISTERED",
        "visual_qualified": "NOT_EVALUATED", "training_usable": "NOT_QUALIFIED_BY_THIS_TASK", "physical_deployment": "NOT_AUTHORIZED",
        "new_videos": 0, "new_checkpoints": 0, "algorithm_calls": 0, "training_steps": 0, "user_files_deleted": 0, "external_writes": 0,
        "v1_pointer_changed": False, "other_manus_0916_changed": False, "training_bash_changed": False,
        "candidate_adopted": False, "control_ground_truth": False,
        "next_actions": {"publisher": "Confirm project output capacity, recover NAS source reads, then separately register candidate CLI/operation and Git closure engineering before G1 model dispatch", "mask": "Run fixed six full sessions through collect/refine/build under a signed G1 packet; evaluation-only manual annotations remain separate", "clean": "Freeze Raw/Mask; derive actual target-warped flow evidence and register single-reference-selection A/B", "occlusion": "Bind complete rendered link depth to raw visible object support; compare fallback and geometric decisions separately", "source": "Use a fresh finite source freeze attempt after NAS I/O recovers; do not consume either stopped partial directory"},
    }
    atomic_json(output / "ENGINEERING_RESULT.json", result)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    if index_path.exists():
        raise ValueError("FRESH_RESULT_INDEX_REQUIRED")
    atomic_json(index_path, {"schema": "inpainting-optimization-result-index-v1", "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "parent": packet["inputs"]["v1_manifest"], "g0_task": artifact_ref(Path(packet["packet_path"])), "g0_engineering": artifact_ref(output / "ENGINEERING_RESULT.json"), "cpu_tests": result["tests"], "source0918": source_diagnostic, "next_actions": result["next_actions"], "candidate_adopted": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    packet = load_json(args.packet.resolve())
    if packet["task_id"] != "inpainting_generalization_g0_20261012" or packet["algorithm_execution_allowed"] is not False:
        raise ValueError("G0_CPU_PACKET_REQUIRED")
    result = seal(packet)
    print({"engineering": result["engineering_execution"], "g0_admission": result["g0_full_admission"], "cpu_tests": result["tests"]["passed"], "model_calls": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
