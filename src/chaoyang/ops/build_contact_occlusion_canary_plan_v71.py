#!/usr/bin/env python3
"""Compile the V7.1 contact/occlusion research canaries without running models."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries"
REPORT = ROOT / "docs/research/current/CONTACT_OCCLUSION_METHODS_V71_ZH.md"
PLAN = ROOT / "docs/governance/PLAN_REVISION.json"
BASELINE = ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _evidence(path: str, role: str) -> dict[str, Any]:
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(f"required evidence missing: {absolute}")
    return {"path": str(absolute), "bytes": absolute.stat().st_size, "sha256": _sha(absolute), "role": role}


def _evidence_file(path: Path, role: str) -> dict[str, Any]:
    absolute = path.resolve()
    if not absolute.is_file():
        raise FileNotFoundError(f"required evidence missing: {absolute}")
    return {"path": str(absolute), "bytes": absolute.stat().st_size, "sha256": _sha(absolute), "role": role}


def _dep(dependency_id: str, status: str, reason: str, *paths: str, required: bool = True) -> dict[str, Any]:
    return {
        "dependency_id": dependency_id,
        "required": required,
        "status": status,
        "reason": reason,
        "evidence_paths": [str(ROOT / p) for p in paths],
    }


def _budget(gpu_seconds: int, wall_seconds: int = 14400) -> dict[str, Any]:
    return {
        "max_rounds": 2,
        "wall_seconds_per_round": wall_seconds,
        "gpu_seconds_per_round": gpu_seconds,
        "cpu_only_compilation": True,
    }


def _common_forbidden() -> list[str]:
    return [
        "EXTERNAL_PHYSICAL_TRUTH",
        "CONTACT_GROUND_TRUTH",
        "OCCLUSION_GOLD_ACCURACY_WITHOUT_INDEPENDENT_LABELS",
        "PHYSICAL_DEPLOYMENT_AUTHORITY",
    ]


def _definitions() -> list[dict[str, Any]]:
    report = _evidence("docs/research/current/CONTACT_OCCLUSION_METHODS_V71_ZH.md", "CANARY_METHOD_SOURCE")
    plan = _evidence("docs/governance/PLAN_REVISION.json", "PINNED_PLAN_REVISION")
    contact_packet = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/task_packets/contact_evidence_dag_v1/TASK_PACKET.json", "CONTACT_CONTRACT")
    object_packet = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/task_packets/object_pose_hypothesis_v2/TASK_PACKET.json", "OBJECT_HYPOTHESIS_CONTRACT")
    silver_packet = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/task_packets/occlusion_silver_v1/TASK_PACKET.json", "SILVER_CONTRACT")
    h1 = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h1_hand/RESULT.json", "SENSOR_HAND_RECEIPT")
    h2 = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h2_tactile/RESULT.json", "TACTILE_RECEIPT")
    h3 = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/attempts/attempt_0002/RESULT.json", "STEREO_PREFLIGHT_RECEIPT")
    h4 = _evidence("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/RESULT.json", "SENSOR_MASK_PREFLIGHT_RECEIPT")

    return [
        {
            "canary_id": "S1", "line": "STANDARD_EXACT78", "priority": 1,
            "objective": "Compare causal SAM 2.1 and Cutie identity memory on occlusion and re-entry.",
            "methods": ["SAM_2_1_CAUSAL_STREAMING", "CUTIE_OBJECT_MEMORY_CHALLENGER"],
            "scope": "One frozen failure canary plus two old A/B regressions; Poker cards and three Chips instances stay separate.",
            "execution_mode": "CAUSAL_PROCESSING",
            "causal_policy": "Only observations at or before target frame may affect the target modal mask.",
            "offline_policy": "Offline comparison may audit disagreement but cannot feed causal training input.",
            "algorithm_prerequisites": [
                _dep("SAM2_1_LOCAL_PIN", "VERIFIED_LOCAL", "Local implementation, config, weight pin and CPU load-smoke receipt exist; production promotion remains disabled by the asset pin.", "assets/models/sam2_1_hiera_large/ASSET_PIN.json"),
                _dep("CUTIE_CODE_AND_WEIGHTS", "MISSING_LOCAL_CODE", "No pinned local Cutie implementation, weights, code SHA or load-smoke receipt was found."),
                _dep("FROZEN_REENTRY_AUDIT_FRAMES", "BLOCKED_UPSTREAM", "A frozen per-instance re-entry audit set and threshold receipt has not been published."),
            ],
            "input_evidence": [report, plan, silver_packet], "budget": _budget(7200),
            "go_gates": ["identity switches, false non-empty masks, re-entry delay and boundary leakage all improve", "two frozen A/B regressions do not degrade", "each physical instance remains independent"],
            "no_go_gates": ["any physical instance swap", "persistent pseudo-mask through full occlusion", "two full sessions exceed the 2 GPU-hour round budget"],
            "terminal_on_no_go": "FAILED_QUALITY_C_OR_BLOCKED_PREREQ",
            "authority_limit": {"maximum_publication": "MODAL_MASK_SUCCESSOR_CANDIDATE", "forbidden_claims": _common_forbidden() + ["AMODAL_MASK_TRUTH", "OBJECT6D_AUTHORITY"]},
            "resource_status": "BLOCKED_PREREQ", "blockers": ["CUTIE_CODE_AND_WEIGHTS", "FROZEN_REENTRY_AUDIT_FRAMES"],
        },
        {
            "canary_id": "S2", "line": "STANDARD_EXACT78", "priority": 2,
            "objective": "Test object onboarding, direct pose anchors and separately recorded short-gap point tracking.",
            "methods": ["FOUNDATIONPOSE_DIRECT_ANCHOR", "COTRACKER3_ONLINE_SHORT_GAP"],
            "scope": "Rigid Poker and low-deformation Chips only; tracked gaps never overwrite observed-only Object6D.",
            "execution_mode": "DUAL_SEPARATED",
            "causal_policy": "Direct per-frame anchors and online point tracks may be computed causally.",
            "offline_policy": "Future-anchor closure is OFFLINE_BIDIRECTIONAL_VISUALIZATION and produces BIDIRECTIONAL_TRACKED hypothesis only.",
            "algorithm_prerequisites": [
                _dep("FOUNDATIONPOSE_PIN", "MISSING_LOCAL_CODE", "No pinned local FoundationPose code, weights, license receipt or load-smoke exists."),
                _dep("COTRACKER3_PIN", "MISSING_LOCAL_CODE", "No pinned local CoTracker3 code, weights, license receipt or load-smoke exists."),
                _dep("OBJECT_ONBOARDING_ASSETS", "BLOCKED_UPSTREAM", "Frozen thin double-sided Poker mesh/atlas and low-deformation Chips reference assets are absent."),
                _dep("DIRECT_DEPTH_MASK_K", "VERIFIED_INPUT_ONLY", "Wave0 has governed Depth/Object6D inputs, but no challenger-specific input manifest has been frozen.", "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"),
            ],
            "input_evidence": [report, plan, object_packet], "budget": _budget(7200),
            "go_gates": ["direct-anchor reprojection and visible Stereo residual pass frozen thresholds", "front/back pose closure passes", "short-gap coverage improves without instance switches"],
            "no_go_gates": ["Poker front/back flip", "rigid SE3 forced during Chips deformation", "endpoint closure failure"],
            "terminal_on_no_go": "FAILED_QUALITY_C_OR_BLOCKED_PREREQ",
            "authority_limit": {"maximum_publication": "DIRECT_OBSERVED_OBJECT6D_CANDIDATE_PLUS_SEPARATE_TRACKED_HYPOTHESIS", "forbidden_claims": _common_forbidden() + ["TRACKED_GAP_AS_DIRECT_OBJECT6D"]},
            "resource_status": "BLOCKED_PREREQ", "blockers": ["FOUNDATIONPOSE_PIN", "COTRACKER3_PIN", "OBJECT_ONBOARDING_ASSETS"],
        },
        {
            "canary_id": "S3", "line": "STANDARD_EXACT78", "priority": 3,
            "objective": "Bounded A/B of joint hand-object reconstruction on two difficult clips.",
            "methods": ["HOLD", "MAGICHOI", "BIGS_ONLY_IF_BOTH_FAIL"],
            "scope": "One 5-10 second Poker clip and one low-deformation Chips clip; no exact78 expansion.",
            "execution_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
            "causal_policy": "No output from this canary is eligible as causal training RGB or direct measurement.",
            "offline_policy": "All reconstruction and novel-view priors are hypothesis-only offline diagnostics.",
            "algorithm_prerequisites": [
                _dep("JOINT_HO_CODE_WEIGHTS", "MISSING_LOCAL_CODE", "HOLD, MagicHOI and BIGS have no pinned local code/weights/load-smoke receipts."),
                _dep("FROZEN_DIFFICULT_CLIPS", "BLOCKED_UPSTREAM", "The two immutable clip manifests and common initialization receipt are not frozen."),
                _dep("LICENSE_REVIEW", "LICENSE_REVIEW_REQUIRED", "Per-method code, model and data licenses must be frozen before execution."),
            ],
            "input_evidence": [report, plan, contact_packet], "budget": _budget(7200),
            "go_gates": ["held-out silhouette, visible Stereo residual, temporal reprojection and penetration all improve", "unseen geometry is stable across initializations"],
            "no_go_gates": ["any method exceeds 2 GPU-hours or 4 wall-hours", "improvement exists only on optimized frames", "unseen geometry is initialization-sensitive"],
            "terminal_on_no_go": "NO_GO_RESEARCH_BUDGET_OR_BLOCKED_PREREQ",
            "authority_limit": {"maximum_publication": "DEVELOPMENT_EVIDENCE_HYPOTHESIS_ONLY", "forbidden_claims": _common_forbidden() + ["HIDDEN_GEOMETRY_TRUTH", "TRUE_HAND_SHAPE"]},
            "resource_status": "BLOCKED_PREREQ", "blockers": ["JOINT_HO_CODE_WEIGHTS", "FROZEN_DIFFICULT_CLIPS", "LICENSE_REVIEW"],
        },
        {
            "canary_id": "N1", "line": "CONTROLLER_MANUS_SENSOR", "priority": 1,
            "objective": "Gate Controller/MANUS geometry contact hypotheses with synchronized finger tactile events.",
            "methods": ["HAND21_FROM_MANUS_CONTROLLER", "TACTILE_EVENT_GATE", "DIRECT_OBJECT6D_CONTACT_SEED"],
            "scope": "Sensor cohort; Controller anchors wrist, MANUS25 supplies fingers, tactile supplies timing and finger identity only.",
            "execution_mode": "DUAL_SEPARATED",
            "causal_policy": "Online tactile events may gate contact timing but cannot infer object identity, force or palm contact.",
            "offline_policy": "Before/after direct anchors may audit attachment continuity, never self-prove contact.",
            "algorithm_prerequisites": [
                _dep("CANONICAL_HAND_SIDECAR", "VERIFIED_INPUT_ONLY", "H1 published a sensor hand ledger with explicit non-truth claim limits.", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h1_hand/RESULT.json"),
                _dep("TACTILE_SIDECAR", "VERIFIED_INPUT_ONLY", "H2 published timing-only tactile evidence.", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h2_tactile/RESULT.json"),
                _dep("SENSOR_DIRECT_OBJECT6D", "BLOCKED_UPSTREAM", "No sensor-line directly observed Object6D anchor exists because Stereo and Mask inference are not published."),
                _dep("OBJECT_INSTANCE_DISAMBIGUATION", "BLOCKED_UPSTREAM", "Tactile events alone do not identify which task object was touched."),
            ],
            "input_evidence": [report, h1, h2, contact_packet], "budget": _budget(0),
            "go_gates": ["cross-sensor time residual passes a frozen threshold", "left/right and finger identity agree", "distance changes agree with tactile event timing"],
            "no_go_gates": ["missing transform or timestamp synchronization", "tactile event cannot be assigned to an object instance", "attachment is used to prove its own contact seed"],
            "terminal_on_no_go": "BLOCKED_PREREQ_OR_FAILED_QUALITY_C",
            "authority_limit": {"maximum_publication": "TACTILE_SUPPORTED_CONTACT_TIMING_EVIDENCE", "forbidden_claims": _common_forbidden() + ["CALIBRATED_FORCE", "OBJECT_ID_FROM_TACTILE", "PALM_TACTILE"]},
            "resource_status": "BLOCKED_PREREQ", "blockers": ["SENSOR_DIRECT_OBJECT6D", "OBJECT_INSTANCE_DISAMBIGUATION"],
        },
        {
            "canary_id": "N2", "line": "CONTROLLER_MANUS_SENSOR", "priority": 2,
            "objective": "Test HOT3D-style SBS multi-view object lifting with same-session calibration.",
            "methods": ["SBS_TRIANGULATION", "ROBUST_SURFACE_AGGREGATION", "FOUNDATIONSTEREO_CROSS_CHECK"],
            "scope": "4096x1536 SBS only; 1280x960 mono is display/prompt input and exact78 rectification constants are forbidden.",
            "execution_mode": "CAUSAL_PROCESSING",
            "causal_policy": "Only same-frame stereo observations and same-session calibration form geometry.",
            "offline_policy": "Offline comparisons may audit closure but cannot promote unobserved surfaces.",
            "algorithm_prerequisites": [
                _dep("SENSOR_STEREO_INFERENCE", "BLOCKED_UPSTREAM", "H3 passed CPU contract preflight but is BLOCKED_RESOURCE; no rectified images or depth were generated.", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/attempts/attempt_0002/RESULT.json"),
                _dep("SENSOR_INSTANCE_MASKS", "BLOCKED_UPSTREAM", "H4 passed CPU contract preflight but is BLOCKED_RESOURCE; no pixel masks were generated.", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/RESULT.json"),
                _dep("OBJECT_ONBOARDING_REFERENCE", "BLOCKED_UPSTREAM", "No frozen sensor-line per-instance onboarding reference/mesh receipt exists."),
            ],
            "input_evidence": [report, h3, h4, object_packet], "budget": _budget(7200),
            "go_gates": ["left-right cycle and registration pass", "triangulation baseline and direct-view reprojection pass", "occluded pixels are excluded"],
            "no_go_gates": ["exact78 calibration reused", "image domain mismatches K", "occluded pixels participate in triangulation"],
            "terminal_on_no_go": "FAILED_QUALITY_C_OR_BLOCKED_PREREQ",
            "authority_limit": {"maximum_publication": "SAME_SESSION_MULTIVIEW_GEOMETRY_CANDIDATE", "forbidden_claims": _common_forbidden() + ["MILLIMETER_EXTERNAL_ACCURACY", "CONTROLLER_WRIST_OVERWRITE"]},
            "resource_status": "BLOCKED_PREREQ", "blockers": ["SENSOR_STEREO_INFERENCE", "SENSOR_INSTANCE_MASKS", "OBJECT_ONBOARDING_REFERENCE"],
        },
        {
            "canary_id": "N3", "line": "CONTROLLER_MANUS_SENSOR", "priority": 3,
            "objective": "Run a shadow Video-to-Data or Do As I Do comparison on one rigid Poker and one Chips clip.",
            "methods": ["NVIDIA_VIDEO_TO_DATA_SHADOW", "DO_AS_I_DO_SHADOW"],
            "scope": "Interface and evidence-separation comparison only; it cannot write the current baseline.",
            "execution_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
            "causal_policy": "Shadow outputs are never eligible as causal training inputs.",
            "offline_policy": "Onboarding, memory, smoothing and inspection rendering remain development comparisons.",
            "algorithm_prerequisites": [
                _dep("SHADOW_PIPELINE_CODE_ASSETS", "MISSING_LOCAL_CODE", "No pinned local Video-to-Data/Do As I Do implementation, weights, container or load-smoke receipt exists."),
                _dep("LICENSE_AND_CONTAINER_REVIEW", "LICENSE_REVIEW_REQUIRED", "Container size, all transitive model licenses and model hashes are not closed."),
                _dep("FROZEN_SENSOR_CLIPS", "BLOCKED_UPSTREAM", "One immutable rigid Poker clip and one Chips clip have not been selected."),
            ],
            "input_evidence": [report, plan, h1, h3], "budget": _budget(7200),
            "go_gates": ["traceable mesh/pose/hand alignment completes within budget", "direct evidence and hypotheses stay separate", "local SBS and Controller/MANUS remain authoritative inputs"],
            "no_go_gates": ["dependency or license closure missing", "object identity swaps", "learned depth overwrites measured SBS geometry"],
            "terminal_on_no_go": "BLOCKED_PREREQ_OR_NO_GO_RESEARCH_BUDGET",
            "authority_limit": {"maximum_publication": "DEVELOPMENT_CROSS_SYSTEM_COMPARISON", "forbidden_claims": _common_forbidden() + ["CURRENT_BASELINE", "LEARNED_DEPTH_AS_MEASURED_SBS"]},
            "resource_status": "BLOCKED_PREREQ", "blockers": ["SHADOW_PIPELINE_CODE_ASSETS", "LICENSE_AND_CONTAINER_REVIEW", "FROZEN_SENSOR_CLIPS"],
        },
    ]


def build() -> dict[str, Any]:
    canaries = _definitions()
    source_evidence = [
        _evidence("docs/research/current/CONTACT_OCCLUSION_METHODS_V71_ZH.md", "RESEARCH_REPORT"),
        _evidence("docs/governance/PLAN_REVISION.json", "PLAN_REVISION"),
        _evidence("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json", "CURRENT_BASELINE_BOUNDARY"),
    ]
    manifest_hash = hashlib.sha256("".join(item["sha256"] for item in source_evidence).encode()).hexdigest()
    producer_hash = hashlib.sha256((Path(__file__).read_text(encoding="utf-8") + manifest_hash).encode()).hexdigest()
    return {
        "schema_version": "contact-occlusion-canary-plan-v71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_id": "CONTACT_OCCLUSION_CANARY_PLAN_V71",
        "artifact_revision": "R7_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "producer_signature": producer_hash,
        "input_manifest_sha": manifest_hash,
        "source_evidence": source_evidence,
        "canaries": canaries,
        "counts": {
            "total": len(canaries),
            "standard_exact78": sum(c["line"] == "STANDARD_EXACT78" for c in canaries),
            "controller_manus_sensor": sum(c["line"] == "CONTROLLER_MANUS_SENSOR" for c in canaries),
            "ready": sum(c["resource_status"] == "READY_FOR_BOUNDED_RUN" for c in canaries),
            "blocked_prereq": sum(c["resource_status"] == "BLOCKED_PREREQ" for c in canaries),
        },
        "authority": False,
        "claim_limit": "CPU-compiled bounded research plan only. No model was downloaded or run; every canary is blocked until its listed prerequisites are closed, and no current authority is granted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    value = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
    csv_path = args.output_dir / "CONTACT_OCCLUSION_CANARY_PLAN_V71.csv"
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["canary_id", "line", "priority", "methods", "execution_mode", "resource_status", "blockers", "gpu_seconds_per_round", "wall_seconds_per_round", "maximum_publication", "terminal_on_no_go"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in value["canaries"]:
            writer.writerow({
                "canary_id": item["canary_id"], "line": item["line"], "priority": item["priority"],
                "methods": "|".join(item["methods"]), "execution_mode": item["execution_mode"],
                "resource_status": item["resource_status"], "blockers": "|".join(item["blockers"]),
                "gpu_seconds_per_round": item["budget"]["gpu_seconds_per_round"],
                "wall_seconds_per_round": item["budget"]["wall_seconds_per_round"],
                "maximum_publication": item["authority_limit"]["maximum_publication"],
                "terminal_on_no_go": item["terminal_on_no_go"],
            })
    outputs = [_evidence_file(json_path, "CANARY_PLAN_JSON"), _evidence_file(csv_path, "CANARY_PLAN_CSV")]
    result = {
        "schema_version": "contact-occlusion-canary-plan-build-result-v1",
        "status": "PASSED",
        "artifact_revision": "R7_1",
        "outputs": outputs,
        "counts": value["counts"],
        "claim_limit": value["claim_limit"],
    }
    (args.output_dir / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
