"""Fenced CPU composition of immutable candidates; no model or Robot rerender."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

NATIVE_SCOPE = "INPAINTING_NATIVE_COMPOSITE_CPU"
NATIVE_RESULT_SCHEMA = "inpainting-native-composite-batch-result-v1"


def _write_once(path, value):
    from .contracts import file_ref
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def main(argv=None):
    from chaoyang.governance.common import REPO_ROOT, load_json, now_iso
    from chaoyang.governance.inpainting_candidate_registration import (
        COMPOSITE_OPERATION, authorize_composite_operation)
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parsed = parser.parse_args(args)
    packet = authorize_composite_operation(COMPOSITE_OPERATION, args, repo_root=REPO_ROOT)
    if load_json(parsed.packet) != packet:
        raise ValueError("COMPOSITE_IN_MEMORY_PACKET_DRIFT")
    root = Path(packet["output_dir"])
    if root.resolve(strict=True) != root or not root.is_relative_to(REPO_ROOT) or root == REPO_ROOT:
        raise ValueError("COMPOSITE_PROJECT_OUTPUT_REQUIRED")
    cache = root / "cpu_cache"
    cache.mkdir(exist_ok=False)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"))
    tempfile.tempdir = str(cache)
    sys.dont_write_bytecode = True
    from .contracts import file_ref, verify_ref, verify_sealed_stage
    from .composite import build_composite_stage

    def authority(*, full=False):
        current = authorize_composite_operation(COMPOSITE_OPERATION, args,
            repo_root=REPO_ROOT, full=full)
        if current != packet:
            raise ValueError("COMPOSITE_TASK_OR_WRITER_IDENTITY_DRIFT")
        return {"execution_allowed": True, "algorithm_execution_allowed": True,
            "task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "fencing_token": packet["writer"]["fencing_token"], "gpu_count": 0,
            "model_calls_max": 0, "render_calls_max": 0, "cpu_threads_per_worker": 2,
            "deadline_at": packet["deadline_at"]}

    if packet.get("experiment_scope") == NATIVE_SCOPE:
        result = run_native_composite_batch(packet, authority=authority, repo_root=REPO_ROOT)
        print(json.dumps(file_ref(root / "COMPOSITE_RESULT.json"), ensure_ascii=False), flush=True)
        return 0 if result["execution_diagnostic_complete"] else 2

    rows = []
    for row in packet["sessions"]:
        authority(full=True)
        bindings = {}
        for name in ("source", "mask", "clean", "render"):
            reference = row[name + "_receipt"]
            verify_ref(reference)
            path = Path(reference["path"])
            stage = load_json(path)
            verify_sealed_stage(stage, row[name + "_signature"], output_root=path.parent)
            bindings[name + "_receipt"] = reference
            bindings[name + "_signature"] = row[name + "_signature"]
        source = load_json(Path(row["source_receipt"]["path"]))
        session = root / "sessions" / row["session_key"]
        session.mkdir(parents=True, exist_ok=False)
        target = session / "Composite"
        result = build_composite_stage(**bindings, output_dir=target,
            component_signature=row["composite_signature"], repo_root=REPO_ROOT,
            authorization=authority, upstream_quality_evidence=row["upstream_quality_evidence"])
        verify_sealed_stage(result, row["composite_signature"], output_root=target)
        authority(full=True)
        qualities = {name: load_json(Path(reference["path"]))["visual_quality"]
            for name, reference in row["upstream_quality_evidence"].items()}
        rows.append({"session_key": row["session_key"], "native_frames": row["native_frames"],
            **{name: row[name + "_receipt"] for name in ("source", "mask", "clean", "render")},
            "composite": file_ref(target / "RECEIPT.json"),
            "composite_receipt": file_ref(target / "RECEIPT.json"),
            "execution": "COMPLETE_COMPOSITE", "geometry": "UNKNOWN_SCENE_DEPTH_ABSENT",
            "object_support_qualification": "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED",
            "mask_quality": qualities["Mask"], "clean_quality": qualities["Clean"],
            "timebase_status": source["timebase_status"],
            "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "control_ground_truth": False, "training_eligible": False})
    result = {"schema": "inpainting-composite-batch-result-v1",
        "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "status": "PASSED", "planned_sessions": len(packet["sessions"]),
        "completed_sessions": len(rows), "completed_composite_sessions": len(rows),
        "sessions": rows, "native_frames": sum(row["native_frames"] for row in rows),
        "model_calls": 0, "render_calls": 0, "gpu_count": 0, "training_steps": 0,
        "files_deleted": 0, "candidate_adopted": False, "control_ground_truth": False,
        "visual_quality": "NOT_REVIEWED", "training_eligible": False,
        "completed_at": now_iso(),
        "claim_limit": "Offline candidate composition only. Known Mask/Clean FAIL retained; model-predicted foreground and display fallback do not establish correct scene geometry or trainable labels."}
    print(json.dumps(_write_once(root / "COMPOSITE_RESULT.json", result), ensure_ascii=False), flush=True)
    return 0


def run_native_composite_batch(packet, *, authority, repo_root):
    """Existing pixel compositor at true native N, with local upstream blocks.

    Only immutable READY stages enter the producer. BLOCKED rows do not import
    or call a producer, render a diagnostic substitute, or claim a video.
    """
    from chaoyang.governance.common import now_iso
    from .contracts import file_ref, verify_ref, verify_sealed_stage
    from .native_contracts import validate_native_composite_spec, load
    validate_native_composite_spec(packet, repo_root=repo_root)
    authority(full=True)
    root = Path(packet["output_dir"])
    rows, completed = [], 0
    for row in packet["sessions"]:
        authority(full=True)
        target = root / "sessions" / row["session_key"]
        target.mkdir(parents=True, exist_ok=False)
        status = {"session_key": row["session_key"], "session_id": row["session_id"],
            "native_frames": row["native_frames"], "experiment_scope": NATIVE_SCOPE,
            "model_calls": 0, "render_calls": 0, "gpu_count": 0,
            "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False, "partial_consumable": False}
        if row["upstream_status"] == "BLOCKED":
            verify_ref(row["diagnostic_ref"])
            status.update(execution="BLOCKED_UPSTREAM", upstream_status="BLOCKED",
                failure_code=row["failure_code"], diagnostic_ref=row["diagnostic_ref"])
        else:
            try:
                # Delayed until READY and authorized; no model or rerender.
                from .composite import build_composite_stage
                bindings = {name + suffix: row[name + suffix]
                    for name in ("source", "mask", "clean", "render")
                    for suffix in ("_receipt", "_signature")}
                composite_root = target / "Composite"
                stage = build_composite_stage(**bindings, output_dir=composite_root,
                    component_signature=row["composite_signature"], repo_root=repo_root,
                    authorization=authority, upstream_quality_evidence=row["upstream_quality_evidence"])
                verify_sealed_stage(stage, row["composite_signature"], output_root=composite_root)
                authority(full=True)
                qualities = {key: (load(row["upstream_quality_evidence"][key]["path"])["visual_quality"]
                    if key in row["upstream_quality_evidence"] else
                    load(row[key.lower() + "_receipt"]["path"]).get("visual_quality", "NOT_EVALUATED"))
                    for key in ("Mask", "Clean")}
                status.update(**{name: row[name + "_receipt"] for name in ("source", "mask", "clean", "render")},
                    composite=file_ref(composite_root / "RECEIPT.json"),
                    composite_receipt=file_ref(composite_root / "RECEIPT.json"),
                    execution="COMPLETE_COMPOSITE", geometry="UNKNOWN_SCENE_DEPTH_ABSENT",
                    object_support_qualification="MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED",
                    qualified_trusted_occlusion_pixels=0,
                    mask_quality=qualities["Mask"], clean_quality=qualities["Clean"],
                    timebase_status=load(row["source_receipt"]["path"])["timebase_status"])
                completed += 1
            except Exception as error:
                status.update(execution="FAILED_RUNTIME", failure_code=type(error).__name__ + ":" + str(error))
                _write_once(target / "SESSION_RESULT.json", status)
                if any(code in str(error) for code in ("SHA_DRIFT", "FENCE", "LEASE", "DEADLINE", "TASK_OR_WRITER")):
                    raise
                rows.append(status)
                continue
        _write_once(target / "SESSION_RESULT.json", status)
        rows.append(status)
    blocked = sum(row["execution"] == "BLOCKED_UPSTREAM" for row in rows)
    failed = sum(row["execution"] == "FAILED_RUNTIME" for row in rows)
    complete = completed > 0 and failed == 0 and completed + blocked == len(rows)
    result = {"schema": NATIVE_RESULT_SCHEMA, "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "experiment_scope": NATIVE_SCOPE,
        "status": "PASSED" if complete else "BLOCKED" if not failed else "FAILED_RUNTIME_FINAL",
        "planned_sessions": len(rows), "completed_sessions": completed,
        "completed_composite_sessions": completed, "blocked_sessions": blocked, "failed_sessions": failed,
        "execution_diagnostic_complete": complete, "sessions": rows,
        "planned_native_frames": sum(row["native_frames"] for row in rows),
        "native_frames": sum(row["native_frames"] for row in rows if row["execution"] == "COMPLETE_COMPOSITE"),
        "model_calls": 0, "render_calls": 0, "gpu_count": 0, "training_steps": 0,
        "files_deleted": 0, "candidate_adopted": False, "control_ground_truth": False,
        "visual_quality": "NOT_REVIEWED", "training_eligible": False,
        "completed_at": now_iso(),
        "claim_limit": "Only READY native pixel composition executed; upstream FAILED and scene UNKNOWN retained. No adoption or trainable-label claim."}
    _write_once(root / "COMPOSITE_RESULT.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
