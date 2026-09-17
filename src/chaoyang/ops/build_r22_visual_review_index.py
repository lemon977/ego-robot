#!/usr/bin/env python3
"""Publish shallow R2.2 visual copies and a fail-closed experiment index."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, atomic_write, now_iso  # noqa: E402


RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/doc_r22/attempts/attempt_0001"
SHALLOW = ROOT / "docs/current/visuals"
SOURCES = {
    "sensor_wrist_depth_unavailable": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v1/depth10/play_cards_0910_001/attempt_0001/play_cards_0910_001_Controller_MANUS_HaWoR_Stereo不可用_绝对3D全片.mp4",
    "chips023_basic_occlusion": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_lane_cd_v1/occlusion_basic_object6d_gated/get_potato_chips_0902_023/attempt_0001/payload/get_potato_chips_0902_023_Clean底图_Robot基础几何遮挡_全片.mp4",
}
NAMES = {
    "sensor_wrist_depth_unavailable": "R22_手套_PlayCards0910_001_Controller_MANUS_HaWoR_Stereo不可用_绝对3D全片.mp4",
    "chips023_basic_occlusion": "R22_Chips023_Clean底图_Robot基础几何遮挡_全片.mp4",
}


def write_once_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable conflict: {path}")
        return
    atomic_write(path, data)


def copy_verified(source: Path, target: Path) -> dict:
    if not source.is_file():
        raise RuntimeError(f"missing visual source: {source}")
    source_ref = artifact_ref(source)
    if target.exists():
        target_ref = artifact_ref(target)
        if (target_ref["bytes"], target_ref["sha256"]) != (source_ref["bytes"], source_ref["sha256"]):
            raise RuntimeError(f"shallow visual conflict: {target}")
    else:
        shutil.copy2(source, target)
    target_ref = artifact_ref(target)
    if (target_ref["bytes"], target_ref["sha256"]) != (source_ref["bytes"], source_ref["sha256"]):
        raise RuntimeError(f"shallow copy verification failed: {source}")
    return {"source": source_ref, "shallow_copy": target_ref}


def main() -> int:
    if RUN.exists():
        raise RuntimeError(f"fresh immutable output required: {RUN}")
    RUN.mkdir(parents=True)
    SHALLOW.mkdir(parents=True, exist_ok=True)
    visuals = {
        key: copy_verified(source, SHALLOW / NAMES[key])
        for key, source in SOURCES.items()
    }
    matrix = {
        "schema_version": "chaoyang-r22-experiment-matrix-v1",
        "generated_at": now_iso(),
        "experiments": [
            {"id": "B0", "definition": "current complete baseline", "status": "HISTORICAL_REFERENCE_ONLY", "reason": "Not rerun under the frozen R2.2 input revision."},
            {"id": "B1", "definition": "new Mask + old Clean", "status": "BLOCKED_RESOURCE", "reason": "The two frozen SAM3.1 canaries were not executed in the CPU prerequisite attempt."},
            {"id": "B2", "definition": "new Mask + new Clean", "status": "BLOCKED_PREREQ", "reason": "Causal Clean prerequisites did not close; no fresh prefix-only ProPainter result exists."},
            {"id": "B3", "definition": "B2 + direct Robot overlay", "status": "BLOCKED_PREREQ", "reason": "B2 is unavailable."},
            {"id": "B4", "definition": "B2 + Robot + z-buffer/contact-aware occlusion", "status": "BLOCKED_PREREQ", "reason": "B2 is unavailable and same-session contact-aware intersection is absent."},
            {"id": "D_BASE_CHIPS023", "definition": "existing Clean + Robot + Object6D-gated base visible-surface occlusion", "status": "PASSED_DEVELOPMENT", "reason": "Contact-independent diagnostic only; not a substitute for B4.", "visual": visuals["chips023_basic_occlusion"]["shallow_copy"]},
        ],
        "authority_promoted": False,
    }
    assumptions = {
        "schema_version": "chaoyang-r22-assumption-register-v1",
        "generated_at": now_iso(),
        "assumptions": [
            {"id": "A_MASK", "statement": "SAM3.1 remains the formal Mask baseline; no independent pixel gold exists for these R2.2 canaries.", "claim_limit": "Do not report accuracy."},
            {"id": "A_CLEAN", "statement": "Clean 58/58 proves structural/provenance/decode closure only.", "claim_limit": "No contact-edge or semantic correctness authority."},
            {"id": "A_CONTACT", "statement": "CONTACT-10 is HYPOTHESIS_ONLY and attachment cannot prove contact.", "claim_limit": "external_accuracy=UNKNOWN"},
            {"id": "A_ROBOT", "statement": "All Robot trajectories and renders are digital development artifacts.", "claim_limit": "control_ground_truth=false; physical_deployment=false"},
            {"id": "A_CAUSAL", "statement": "Future donors and bidirectional ProPainter are forbidden for causal training inputs.", "claim_limit": "Offline visualization may use them only when explicitly labelled."},
        ],
    }
    matrix_path = RUN / "EXPERIMENT_MATRIX.json"
    assumption_path = RUN / "ASSUMPTION_REGISTER.json"
    write_once_json(matrix_path, matrix)
    write_once_json(assumption_path, assumptions)
    index_path = RUN / "VISUAL_REVIEW_INDEX.json"
    write_once_json(index_path, {
        "schema_version": "chaoyang-r22-visual-review-index-v1",
        "generated_at": now_iso(),
        "visuals": visuals,
        "blocked_expected_visuals": ["Poker245 fresh Mask/Clean full session", "Chips039 fresh Mask/Clean full session", "Poker same-session Robot+base Occlusion"],
        "experiment_matrix": artifact_ref(matrix_path),
        "assumption_register": artifact_ref(assumption_path),
        "authority_promoted": False,
    })
    result_path = RUN / "RESULT.json"
    write_once_json(result_path, {
        "schema_version": "chaoyang-r22-doc-result-v1",
        "task_id": "doc_r22",
        "status": "PASSED",
        "visual_review_index": artifact_ref(index_path),
        "experiment_matrix": artifact_ref(matrix_path),
        "assumption_register": artifact_ref(assumption_path),
        "authority_promoted": False,
    })
    write_once_json(RUN / "METRICS.json", {"status": "PASSED", "shallow_visuals": len(visuals), "blocked_expected_visuals": 3})
    atomic_write(RUN / "DECISION.md", b"# R2.2 visual index\n\nOnly verified development visuals were copied to the shallow review directory. Missing ablations remain explicit blockers.\n")
    write_once_json(RUN / "NEXT_ACTION.json", {"task_id": "r22_recovery_package", "status": "PENDING", "prerequisites": ["robot_batch001_terminal", "mask_canary_terminal"]})
    artifacts = [index_path, matrix_path, assumption_path, result_path, RUN / "METRICS.json", RUN / "DECISION.md", RUN / "NEXT_ACTION.json", *[Path(item["shallow_copy"]["path"]) for item in visuals.values()]]
    manifest_path = RUN / "ARTIFACT_MANIFEST.json"
    write_once_json(manifest_path, {"schema_version": "chaoyang-r22-doc-manifest-v1", "artifacts": [artifact_ref(path) for path in artifacts]})
    write_once_json(RUN / "RUN_RECEIPT.json", {"schema_version": "chaoyang-r22-doc-receipt-v1", "task_id": "doc_r22", "status": "PASSED", "result": artifact_ref(result_path), "manifest": artifact_ref(manifest_path)})
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
