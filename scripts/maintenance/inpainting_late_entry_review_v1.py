"""CPU evaluator only: validate sealed late-query trial and publish rejection.

No model imports, historical manual seeds, algorithm mutation, or adoption.
Images listed here were actually viewed by Root (050) and its reviewer (001).
"""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
import os
from pathlib import Path

import av
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso

ROOT = Path(__file__).resolve().parents[2]
TASK = "inpainting_late_entry_review_20261013"
OUT = ROOT / "_run" / TASK / "attempt_0001"
MODEL = ROOT / "_run/inpainting_late_entry_mask_a02_20261013/attempt_0001"
PRIOR = ROOT / "_run/inpainting_auto_v1_seed_mask_20261013/attempt_0001"
SOURCE = ROOT / "_run/inpainting_generalization_g0_publication_20261013/attempt_0001/sessions"
FIXED = {"001": [0, 50, 103, 104, 119, 132, 202, 237],
         "050": [0, 28, 64, 91, 96, 127]}


def verify(ref):
    actual = artifact_ref(Path(ref["path"]))
    if actual != ref:
        raise RuntimeError("EVIDENCE_SHA_DRIFT:" + ref["path"])


def main():
    packet = json.loads((ROOT / "tasks/current" / TASK / "TASK_PACKET.json").read_text())
    spec = spec_from_file_location("_review_existing_publisher", ROOT / "scripts/humanego_ict/production_resume_runner_v1.py")
    infra = module_from_spec(spec)
    spec.loader.exec_module(infra)
    _, state, index = infra._snapshot()
    infra._own_row(packet, state, index)
    for ref in packet["inputs"].values():
        verify(ref)
    result = json.loads((MODEL / "MODEL_RESULT.json").read_text())
    executor = json.loads((MODEL / "RESULT.json").read_text())
    if executor["status"] != "PASSED" or not executor["model_result_consumable"]:
        raise RuntimeError("SEALED_MODEL_RESULT_REQUIRED")
    if result["completed_mask_sessions"] != 2 or result["model_call_accounting"]["attempted_api_calls"] != 25:
        raise RuntimeError("MODEL_ACCOUNTING_DRIFT")
    rows = []
    for entry in result["sessions"]:
        key, n = entry["session_key"], entry["native_frames"]
        current = MODEL / "sessions" / key / "Mask"
        old = PRIOR / "sessions" / key / "Mask"
        verify(entry["mask"])
        receipt = json.loads(Path(entry["mask"]["path"]).read_text())
        if receipt["status"] != "SEALED_COMPLETE":
            raise RuntimeError("MASK_SEAL_REQUIRED")
        for ref in receipt["outputs"]:
            verify(ref)
        mapping = json.loads((SOURCE / key / "Source/FRAME_MAP.json").read_text())
        if len(mapping) != n:
            raise RuntimeError("NATIVE_MAPPING_REQUIRED")
        qpc0 = mapping[0]["qpc_declared_ns"]
        comparison = {}
        probes = []
        with np.load(current / "MASK_ARRAYS.npz", allow_pickle=False) as a, np.load(old / "MASK_ARRAYS.npz", allow_pickle=False) as b:
            if a.files != b.files:
                raise RuntimeError("MASK_KEYS_DRIFT")
            for name in a.files:
                new, previous = a[name], b[name]
                if new.shape != (n, 480, 640) or new.dtype != np.bool_ or previous.shape != new.shape:
                    raise RuntimeError("MASK_NATIVE_DOMAIN_DRIFT")
                changed = int(np.count_nonzero(new != previous))
                comparison[name] = {"shape": list(new.shape), "different_elements": changed,
                                    "old_nonzero": int(previous.sum()), "new_nonzero": int(new.sum())}
                if changed:
                    raise RuntimeError("EXPECTED_NO_BENEFIT_DIAGNOSIS_MISMATCH")
            points = ([(0, 450, 289, "controller"), (104, 535, 270, "controller"), (132, 429, 210, "controller"),
                       (0, 335, 280, "visible_cup"), (104, 367, 270, "visible_cup"), (132, 220, 166, "visible_cup")]
                      if key == "001" else [(0, 20, 230, "left_edge_device"), (0, 535, 320, "right_mount"),
                                              (64, 20, 330, "left_mount"), (127, 434, 310, "right_gray_finger")])
            for frame, x, y, kind in points:
                probes.append({"frame": frame, "xy": [x, y], "kind": kind, "not_ground_truth": True,
                               "previous": {k: bool(b[k][frame, y, x]) for k in ("equipment", "effective_removal", "protected_visible", "unknown")},
                               "new": {k: bool(a[k][frame, y, x]) for k in ("equipment", "effective_removal", "protected_visible", "unknown")}})
        query = json.loads((current / "CONTROLLER_REQUERY.json").read_text())
        if query["added_proposals"] != 0 or len(query["queries"]) != 2 or any(q["tracks"] != 0 or q["propagation_aborted"] for q in query["queries"]):
            raise RuntimeError("REQUERY_ACCOUNTING_DRIFT")
        video = entry["review"]["video"]
        verify(video)
        expected_pts = [(m["qpc_declared_ns"] - qpc0 + 500) // 1000 / 1e6 for m in mapping]
        with av.open(video["path"]) as container:
            container.streams.video[0].codec_context.thread_count = 2
            pts = [float(f.pts * f.time_base) for f in container.decode(video=0)]
        if len(pts) != n or max(abs(a - b) for a, b in zip(pts, expected_pts)) > 1e-6:
            raise RuntimeError("NATIVE_DECODE_OR_PTS_MISMATCH")
        target = ROOT / "review/inpainting_optimization" / f"LATE_ENTRY_{key}_Raw_Mask.mp4"
        if target.exists():
            if artifact_ref(target)["sha256"] != video["sha256"]:
                raise RuntimeError("SHALLOW_TARGET_CONFLICT")
        else:
            os.link(video["path"], target)
        defects = []
        for frame in FIXED[key]:
            descriptions = (["左右白色控制器、安装件及腕盒仍有漏删，晚帧查询未改善。"] if key == "001" else
                            ["左侧画面边缘白色安装件仍有漏删。"])
            if key == "050" and frame in (0, 28, 64, 127):
                descriptions.append("右腕白盒／连接板局部没有进入去除域。")
            if key == "050" and frame == 127:
                descriptions.append("右手灰色指套仍有漏删。")
            defects.append({"native_frame": frame, "declared_video_seconds": expected_pts[frame],
                            "image": artifact_ref(current / f"anchor_{frame:04d}.png"),
                            "video": video, "arrays": artifact_ref(current / "MASK_ARRAYS.npz"),
                            "actual_image_view": True, "reviewer": "/root/late_device_eval" if key == "001" else "/root",
                            "severity": "HIGH", "visible_defects_zh": descriptions,
                            "hidden_regions": "UNKNOWN", "human_review": "PENDING"})
        rows.append({"session_key": key, "native_frames": n, "mask_receipt": entry["mask"],
                     "outputs_sha_verified": len(receipt["outputs"]), "source": entry["source"],
                     "native_frame_map": artifact_ref(SOURCE / key / "Source/FRAME_MAP.json"),
                     "prior_arrays": artifact_ref(old / "MASK_ARRAYS.npz"), "new_arrays": artifact_ref(current / "MASK_ARRAYS.npz"),
                     "array_comparison": comparison, "point_probes": probes, "queries": query,
                     "query_evidence": artifact_ref(current / "CONTROLLER_REQUERY.json"),
                     "native_decode_frames": len(pts), "pts_max_error_seconds": max(abs(a - b) for a, b in zip(pts, expected_pts)),
                     "shallow_video": artifact_ref(target), "ai_seen_frames": FIXED[key],
                     "ai_unseen_frames": n - len(FIXED[key]), "defects": defects,
                     "execution": "COMPLETE_MASK_ONLY", "visual_quality": "FAILED", "human_review": "PENDING",
                     "hidden_regions": "UNKNOWN", "timebase_status": entry["review"]["timebase_status"]})
    audit = {"schema": "inpainting-late-query-actual-review-v1", "task_id": TASK, "run_signature": packet["run_signature"],
             "model_result": artifact_ref(MODEL / "MODEL_RESULT.json"), "model_execution": artifact_ref(MODEL / "RESULT.json"),
             "sessions": rows, "execution_completed_sessions": 2, "native_frames": 366,
             "queries_added": 4, "controller_tracks_added": 0, "mask_array_different_elements": 0,
             "ai_seen_frames": 14, "ai_unseen_frames": 352, "visual_qualified_sessions": 0,
             "visual_quality": "FAILED", "quality_pass": False, "candidate_adopted": False,
             "high_reviewed_frames": 14, "high_failure_clusters": 3, "high_cluster_reduction": 0,
             "failure_clusters": ["001/050白色控制器及安装件漏删", "050右腕盒及连接板漏删", "050末帧灰指套漏删"],
             "clean_executions": 0, "robot_executions": 0, "training_steps": 0, "files_deleted": 0,
             "human_review": "PENDING", "hidden_regions": "UNKNOWN", "control_ground_truth": False,
             "production_default": "INPAINTING_RAW_TO_ROBOT_V1", "v1_manifest": artifact_ref(ROOT / "inpainting_v1/MANIFEST.json"),
             "diagnosis": "Actual pinned text tracker can discover new objects on propagated frames. Fixed controller concept returned no retained tracks at all four later anchors; no benefit. Semantic/score/confirmation causes remain unseparated.",
             "physical_late_entry_tested": False, "completed_at": now_iso()}
    atomic_json(OUT / "ACTUAL_REVIEW.json", audit)
    atomic_json(OUT / "DIAGNOSIS.json", {
        "local_actual_model": "Sam3MultiplexTrackingWithInteractivity",
        "source_evidence": [
            {"file": artifact_ref(ROOT / "vendor/SAM3/sam3/model/sam3_multiplex_tracking.py"), "lines": [348, 620, 1699, 2319]},
            {"file": artifact_ref(ROOT / "vendor/SAM3/sam3/model/sam3_multiplex_base.py"), "lines": [509, 1042]},
            {"file": artifact_ref(ROOT / "src/chaoyang/inpainting_candidate/late_entry_seeds.py"), "lines": [13, 20, 24, 51, 70, 212, 218]}],
        "official_sources": ["https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md", "https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_inference.py"],
        "raw_detector_scores_saved": False, "score_vs_concept_vs_confirmation": "NOT_DISTINGUISHED",
        "next_hypothesis_not_implemented": "associate current RGB object instances with wearer equipment identity; use projection for identity not pixel evidence; reject table/emergency-stop counterexamples",
        "tracker_diagnostic_only": {
            "point_subset": "xy[:,:,27:63,:]", "unchanged_radius_pixels": 8, "unchanged_hit_threshold": 0.25,
            "001_object2": {"born": 12, "left_hits": 226, "nonempty_frames": 226},
            "001_object4": {"born": 104, "right_hits": 129, "nonempty_frames": 134},
            "050_object1_table_counterexample": {"right_hits": 24, "nonempty_frames": 80},
            "050_object2_emergency_stop_counterexample": {"right_hits": 1, "nonempty_frames": 4},
            "reviewer": "/root/late_device_eval", "sufficient_for_device_promotion": False},
        "runtime_failure_preserved": artifact_ref(ROOT / "_run/inpainting_late_entry_mask_20261013/attempt_0001/RESULT.json"),
        "runtime_fix": artifact_ref(ROOT / "_run/inpainting_late_entry_runtime_fix_20261013/attempt_0001/RESULT.json"),
        "capacity_proof_not_capacity_claim": artifact_ref(ROOT / "_run/inpainting_late_entry_cpu_20261013/attempt_0001/CAPACITY_PROOF.json"),
        "claim_limit": "Not first-frame-only detection, not actual physical late-entry validation, not full-video visual approval; unchanged candidate rejected."})
    print(json.dumps({"status": "CPU_EVIDENCE_COMPLETE", "native_frames": 366, "array_diff": 0, "seen": 14, "unseen": 352, "visual_quality": "FAILED"}), flush=True)


if __name__ == "__main__":
    main()
