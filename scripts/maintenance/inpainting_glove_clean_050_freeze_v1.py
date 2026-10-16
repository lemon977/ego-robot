"""Small metadata-only publication/freeze; existing consumers do all inference."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref, verify_ref
from chaoyang.inpainting_candidate.native_contracts import validate_native_clean_spec
from chaoyang.inpainting_candidate.resources import proc_start_ticks

TASK = "inpainting_glove_clean_050_freeze_20261013"
OUT = REPO_ROOT / "_run" / TASK / "attempt_0001"
PARENT = REPO_ROOT / "_run/inpainting_post_g1_glove_concept_mask_a01_20261013/attempt_0001"
SPEC_PATH = REPO_ROOT / "inpainting_optimization/GLOVE_050_NATIVE_CLEAN_SPEC.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(value, code):
    if not value:
        raise ValueError(code)


def once(path, value):
    require(path.is_relative_to(REPO_ROOT) and path.resolve() == path, "PROJECT_ONLY")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def authority():
    packet = load(REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json")
    require(proc_start_ticks(packet["writer"]["pid"]) == packet["writer"]["proc_start_ticks"]
        and datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]), "LIVE_CPU_SCOPE_REQUIRED")
    active = [r for r in load(REPO_ROOT / "tasks/current/INDEX.json")["task_packets"] if r.get("execution_allowed")]
    require(len(active) == 1 and active[0]["task_id"] == TASK
        and active[0]["packet_sha256"] == file_ref(Path(packet["packet_path"]))["sha256"], "ONE_FENCED_CPU_TASK_REQUIRED")
    for reference in packet["inputs"].values():
        verify_ref(reference)
    return packet


def prepare():
    packet = authority()
    parent = load(packet["inputs"]["parent_model_packet"]["path"])
    model = load(packet["inputs"]["parent_model_result"]["path"])
    review_path = PARENT / "review_cpu/lane_001_050/ACTUAL_REVIEW_001_050.json"
    review = load(review_path)
    require(load(PARENT / "RESULT.json")["status"] == "PASSED"
        and load(PARENT / "POST_VALIDATION.json")["exit_code"] == 0
        and model["completed_mask_sessions"] == 2
        and review["run_signature"] == model["run_signature"]
        and review["review_counts"]["actual_viewed_frames"] == 12
        and review["review_counts"]["new_HIGH_clusters"] == 1, "ACTUAL_MODEL_AND_REVIEW_REQUIRED")
    videos = []
    for row in model["sessions"]:
        reference = row["review"]["video"]
        verify_ref(reference)
        shallow = REPO_ROOT / "review/inpainting_optimization" / f"G1_GLOVE_{row['session_key']}_Raw_Mask.mp4"
        require(not shallow.exists(), "FRESH_SHALLOW_LINK_REQUIRED")
        os.link(reference["path"], shallow)
        videos.append({"session_key": row["session_key"], "immutable": reference, "shallow": file_ref(shallow)})
    source_row = next(r for r in parent["sessions"] if r["session_key"] == "050")
    session = next(r for r in model["sessions"] if r["session_key"] == "050")
    source = load(source_row["source_receipt"]["path"])
    refs = {Path(r["path"]).name: r for r in source["outputs"]}
    report = load(refs["SOURCE_REPORT.json"]["path"])
    mask = load(session["mask"]["path"])
    baseline = load(packet["inputs"]["v1_manifest"]["path"])
    weights = [r for r in baseline["models"]["weights"] if Path(r["path"]).name != "sam3.1_multiplex.pt"]
    task = "inpainting_native_clean_glove050_a01_20261013"
    spec = {"spec_schema": "inpainting-finite-native-clean-spec-v1",
        "task_id": task, "experiment_scope": "INPAINTING_NATIVE_BASELINE_CLEAN",
        "actual_stage_scope": "INPAINTING_NATIVE_BASELINE_CLEAN",
        "operation": "run_inpainting_candidate_visual_stages_v1", "model_stages": ["Clean"],
        "infrastructure": parent["inputs"]["spec"] and load(parent["inputs"]["spec"]["path"])["infrastructure"],
        "output_dir": str(REPO_ROOT / "_run" / task / "attempt_0001"), "physical_gpu_index": 0,
        "inputs": {"source_index": parent["inputs"]["source_index"]},
        "runtime_code": parent["runtime_code"], "model_weights": weights, "weights": {"files": weights},
        "config": {"stage": "NATIVE_BASELINE_CLEAN", "image_size": [640, 480], "manual_seed_inputs": 0},
        "calibration": {"metric_projection": "NOT_CONSUMED_CLEAN", "source_eye_qualification": "INHERITED_SOURCE_NOT_GEOMETRY"},
        "schema": {"source": "inpainting-source-stage-v1", "model_result": "inpainting-native-clean-batch-result-v1"},
        "clean_config": {"frozen_settings": baseline["models"]["clean_SETTINGS"], "reference_policy": "FROZEN_BASELINE", "candidates": ["A"]},
        "budgets": {"candidates_max": 1, "cpu_threads_per_worker": 2, "gpu_count": 1,
            "gpu_seconds_max": 900, "wall_seconds_max": 900, "model_calls_max": 1,
            "model_calls_unit": "PROPAINTER_INPAINT_API_INVOCATION", "training_steps_max": 0,
            "output_bytes_estimate": 1073741824},
        "sessions": [{"session_key": "050", "session_id": session["session_id"], "native_frames": 128,
            "upstream_status": "READY", "source_receipt": source_row["source_receipt"],
            "source_signature": source["component_signature"], "frame_map_ref": refs["FRAME_MAP.json"],
            "raw_source_ref": report["actual_inputs"]["video"], "mask_receipt": session["mask"],
            "mask_signature": mask["component_signature"]}],
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
            "inpainting_optimization/GLOVE_050_NATIVE_CLEAN_SPEC.json", "src/chaoyang/inpainting_candidate/clean_worker.py",
            "src/chaoyang/inpainting_candidate/model_stages.py", "src/chaoyang/inpainting_candidate/native_contracts.py",
            "inpainting_optimization/CONTINUE_HERE_ZH.md"],
        "read_set_expansions": ["Only exact frozen Source/Mask/Raw/map/code/three PP weight refs. No SAM or Motion/Render/Composite, no manual seeds or old Clean. One full-native050 unchanged baseline call with exact sealed Mask.E, all original PP/reference/quality settings unchanged.",
            "Evaluator may read completed parent actual12-frame review and exact SourceAware050 Mask refs only after fresh Clean sealed. This is local failure-cluster testing, not adoption of rejected global glove prompt. Keep original FAILED, hiddenUNKNOWN/timebase/humanPENDING. No fabricated background truth, geometry or training claim."],
        "evaluation_only": {"feeds_producer": False, "parent_review": file_ref(review_path),
            "fixed_anchors": [0, 28, 64, 91, 96, 127], "human_review": "PENDING"}}
    spec["sessions"][0]["clean_signature"] = canonical_sha({"stage": "Clean", "recipe": spec,
        "claim": "MODEL_GENERATED_OFFLINE_VISUAL_NOT_GEOMETRY"})
    validate_native_clean_spec(spec, repo_root=REPO_ROOT)
    require([r["session_key"] for r in spec["sessions"]] == ["050"]
        and len(weights) == 3, "SINGLE_050_NO_OTHER_EXECUTION")
    spec_ref = once(SPEC_PATH, spec)
    delivery = {"schema": "inpainting-glove-concept-local-outcome-v1", "task_id": model["task_id"],
        "model_result": file_ref(PARENT / "MODEL_RESULT.json"), "actual_review": file_ref(review_path),
        "binding_counterexample": packet["inputs"]["binding_counterexample"], "videos": videos,
        "execution": "COMPLETE_MASK_ONLY", "native_frames": 366, "actual_viewed": 12, "unviewed": 354,
        "visual_qualified_sessions": 0, "quality": "FAILED", "candidate_adopted": False,
        "HIGH_parent": 4, "HIGH_fresh": 5, "new_HIGH": 1, "MED": 1,
        "positive": {"001_gray_finger_anchors_E_restored": 6, "050_right_controller_anchors_E_restored": 5,
            "050127_previous_controls_preserved": True},
        "rejection": "001_BACKGROUND_REAL_ROBOT_REMOVED_BY_GLOVE_POINT_PROPAGATION",
        "do_not_implement": "Independent-human overlap also zero on correct left gloves; would destroy recovered support.",
        "next_consumer": spec_ref, "next_consumer_status": "SPEC_FROZEN_NOT_EXECUTED",
        "human_review": "PENDING", "hidden_regions": "UNKNOWN", "training_eligible": False,
        "files_deleted": 0, "algorithm_calls_this_CPU": 0}
    return once(OUT / "GLOVE_CONCEPT_DELIVERY.json", delivery)


def seal():
    packet = authority()
    paths = [OUT / "GLOVE_CONCEPT_DELIVERY.json", SPEC_PATH, Path(__file__).resolve(),
        REPO_ROOT / "inpainting_optimization/CONTINUE_HERE_ZH.md", REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json",
        REPO_ROOT / "review/inpainting_optimization/README_ZH.md"]
    return once(OUT / "MAINTENANCE_DONE.json", {"task_id": TASK, "run_signature": packet["run_signature"],
        "status": "PASS", "artifacts": [file_ref(p) for p in paths], "model_calls": 0, "files_deleted": 0,
        "producer_code_changed": False, "tests_repeated": 0, "next_action": "Register one existing NativeClean050/full128 task, not001 rejected Mask or full199."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "seal"])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else seal(), ensure_ascii=False))
