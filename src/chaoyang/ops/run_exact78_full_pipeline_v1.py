"""Finite, manifest-driven Exact78 raw-to-visual-Robot execution.

The same command dispatches both task categories. Legacy names in maintained
numeric producers are implementation dependencies, not historical result inputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, process_identity
from chaoyang.pipeline.exact78_role_admission_v1 import check_required_roles


def admit_masks(spec, mask_path, root):
    report = check_required_roles(load_json(mask_path), spec.get('required_mask_roles'), spec['frame_count'])
    report['source_mask'] = artifact_ref(mask_path)
    atomic_json(root / 'MASK_ADMISSION.json', report)
    if report['missing_required_roles']:
        raise RuntimeError('MASK_QUALITY_HOLD_REQUIRED_ROLE_ABSENT:' + ','.join(report['missing_required_roles']))


def authorize(task_id):
    routes = [r for r in load_json(REPO_ROOT / "tasks/current/INDEX.json")["task_packets"]
              if r["task_id"] == task_id and r.get("execution_allowed")]
    if len(routes) != 1:
        raise RuntimeError("PIPELINE_NOT_ROUTABLE")
    path = REPO_ROOT / routes[0]["packet_path"]
    if artifact_ref(path)["sha256"] != routes[0]["packet_sha256"]:
        raise RuntimeError("PACKET_SHA_DRIFT")
    packet = load_json(path)
    identity = process_identity(packet["writer"]["pid"])
    if not identity["alive"] or identity["start_ticks"] != packet["writer"]["proc_start_ticks"]:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    root = Path(packet["write_set"][0]).resolve()
    if load_json(root / "WRITER.json") != packet["writer"]:
        raise RuntimeError("WRITER_TOKEN_DRIFT")
    return packet, root


def checked(ref):
    path = Path(ref["path"])
    if artifact_ref(path) != ref:
        raise RuntimeError(f"INPUT_SHA_DRIFT:{path}")
    return path


def execute(command, log, cwd=REPO_ROOT):
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"STAGE_RETURN_CODE:{result.returncode}:{log}")


def cli(operation, *args, hawor=False):
    if hawor:
        # The isolated environment intentionally clears PYTHONPATH. Restore
        # only this repository's package root, still dispatching via the CLI.
        bootstrap = f"import sys,runpy;sys.path.insert(0,{str(REPO_ROOT / 'src')!r});runpy.run_module('chaoyang.cli',run_name='__main__')"
        return [str(REPO_ROOT / "src/chaoyang/ops/hawor_python.sh"), "-B", "-c", bootstrap,
                "run", operation, *map(str,args)]
    return [sys.executable, "-B", "-m", "chaoyang.cli", "run", operation, *map(str,args)]


def verify_refs(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            checked(value)
        else:
            for child in value.values():
                verify_refs(child)
    elif isinstance(value, list):
        for child in value:
            verify_refs(child)


def gpu(command, root, task_id, name, free=60000):
    receipt = root / f"{name}_GPU.json"
    execute(cli("run_gpu_command_with_v71_lease", "--task-id", task_id,
                "--attempt-id", f"attempt_0001_{root.name}_{name}", "--priority", "CANARY",
                "--gpu-id", "0", "--min-free-mib", free, "--wait-seconds", "600",
                "--wall-seconds", "5400", "--receipt", receipt,
                "--claim-limit", "Offline baseline engineering; no training or physical authority", "--", *command),
            root / f"{name}_GPU.log")
    if load_json(receipt).get("status") != "PASSED":
        raise RuntimeError(f"GPU_NOT_PASSED:{name}")


def motion(spec, root, task_id):
    name, kind = spec["session_id"], spec["task"]
    raw = root / "hawor_raw"
    gpu(cli("run_exact78_hawor_raw_v1", "--session-root", spec["source_root"],
            "--session-id", name, "--source-video", checked(spec["mono"]), "--output-root", raw,
            hawor=True), root, task_id, "hawor")
    raw_npz = raw / "HAWOR_RAW_MANO21.npz"
    bounded_contract = root / "BOUNDED_CONTRACT.json"
    atomic_json(bounded_contract, {
        "schema_version": "hawor-bounded-parameter-two-task-baseline-contract-v1",
        "canaries": [{"task": kind, "session_id": name, "frame_count": spec["frame_count"], "fps": spec["fps"],
                      "raw_hawor_npz": str(raw_npz), "raw_hawor_sha256": artifact_ref(raw_npz)["sha256"],
                      "source_video": spec["mono"]["path"], "source_video_sha256": spec["mono"]["sha256"]}],
        "acceptance_thresholds": {
            "per_side_reprojection_p95_px_max": 12.0, "per_side_observed_fraction_drop_max": 0.001,
            "identity_switch_count_max": 0, "root_and_pose_rotation_orthogonality_max": 0.0001,
            "root_and_pose_rotation_determinant_min_exclusive": 0.0, "bone_length_cv_must_not_regress": True,
            "wrist_step_p95_must_improve_over_raw": True, "all_joint_acceleration_p95_must_improve_over_raw": True,
        },
        "policy": {"missing": "PRESERVE_NAN_NO_INTERPOLATION", "smoothing_change_allowed": False,
                   "stronger_ab_allowed": False, "same_inode_raw_world_legacy_allowed": False,
                   "human_gate_required": False, "functional_grade_b_allowed": True},
    })
    execute(cli("run_hawor_bounded_parameter_successor", "--contract", bounded_contract,
                "--output-root", root / "bounded", hawor=True), root / "bounded.log")
    bounded_npz = root / "bounded" / name / "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
    if load_json(root / "bounded" / name / "RESULT.json")["status"] != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise RuntimeError("BOUNDED_NUMERIC_HOLD")
    temporal_contract = root / "TEMPORAL_CONTRACT.json"
    atomic_json(temporal_contract, {"sessions": [{"task": kind, "session": name,
        "bounded_npz": str(bounded_npz), "bounded_sha256": artifact_ref(bounded_npz)["sha256"],
        "source_video": spec["mono"]["path"], "source_video_sha256": spec["mono"]["sha256"]}]})
    execute(cli("run_hawor_temporal_jerk_successor_backtrack_v2", "--contract", temporal_contract,
                "--output-root", root / "temporal", hawor=True), root / "temporal.log")
    temporal = root / "temporal" / kind / name
    hawor = temporal / "HAWOR_TEMPORAL_SO3_JERK_SUCCESSOR.npz"
    result = temporal / "RESULT.json"
    if load_json(result)["status"] != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise RuntimeError("TEMPORAL_NUMERIC_HOLD")
    return retarget(spec, root, hawor, result)


def retarget(spec, root, hawor, result):
    name, kind = spec["session_id"], spec["task"]
    for folder in ("arm", "hand"):
        (root / folder).mkdir()
    common = ["--task", kind, "--session", name, "--hawor", hawor, "--hawor-result", result,
              "--accepted-states", checked(spec["robot_prior"])]
    execute(cli("run_robot_motion_transfer_arm_canary_v3", *common,
                "--accepted-hawor", checked(spec["human_prior"]), "--output", root / "arm/RESULT.json",
                "--motion-gain", "1.0", "--task-base-backoff-m", "0.26"), root / "arm.log")
    if load_json(root / "arm/RESULT.json")["status"] != "PASS_NUMERIC_CANARY_NO_AUTHORITY":
        raise RuntimeError("ARM_NUMERIC_HOLD")
    execute(cli("run_robot_hand_fullsession_anatomy_repair_v3", *common,
                "--output", root / "hand/RESULT.json"), root / "hand.log")
    if load_json(root / "hand/RESULT.json")["status"] != "PASS_NUMERIC_CANARY_NO_AUTHORITY":
        raise RuntimeError("HAND_NUMERIC_HOLD")
    return hawor, root / "arm/ARM_CANARY_STATES.npz", root / "hand/HAND_STATES.npz"


def assemble(spec, root, hawor, arm, hand, front=None):
    front = root / "frontend" if front is None else Path(front)
    raw = load_json(front / "RAW.json")
    clean = load_json(front / "CLEAN.json")
    bundle = root / "bundle"
    bundle.mkdir()
    files = {}

    def copy(name, source):
        source = Path(source)
        destination = bundle / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        ref = artifact_ref(destination)
        if ref["sha256"] != artifact_ref(source)["sha256"]:
            raise RuntimeError("BUNDLE_COPY_DRIFT")
        files[name] = {"sha256": ref["sha256"], "bytes": ref["bytes"]}
        return name

    session = {"session_id": spec["session_id"], "fps": spec["fps"], "render_size": [640,480],
        "motion_image_size": [1280,960], "source_stereo_shape": [1536,4096], "source_eye_width": 2048,
        "source_stereo": copy("inputs/source_stereo.mp4", checked(spec["stereo"])),
        "camera_maps": copy("inputs/CAMERA_MAPS.npz", checked(spec["camera_maps"])),
        "arm": copy("inputs/ARM.npz", arm), "hand": copy("inputs/HAND.npz", hand), "hawor": copy("inputs/HAWOR.npz", hawor),
        "flange": copy("assets/flange.STL", REPO_ROOT / "assets/robot/V2相机固定法兰PRO版本.STL"),
        "frames": [], "files": files, "producer_mode": "FRESH_RAW_MASK_CLEAN_MOTION_INFERENCE"}
    if spec["source_index"] != 0:
        raise RuntimeError("RENDER_SOURCE_INDEX_ADAPTER_REQUIRED")
    for r, c in zip(raw["rows"], clean["rows"], strict=True):
        frame = r["frame_id"]
        row = {"frame_id": frame, "clean": copy(f"clean/{frame:06d}.png", checked(c["clean"])),
               "expected_raw_array_sha256": r["array_sha256"]}
        for target, source in (("visible_card","object"), ("hand","human"), ("unknown","unknown")):
            row[target] = copy(f"masks/{target}/{frame:06d}.png", checked(c["masks"][source]))
        session["frames"].append(row)
    atomic_json(bundle / "SESSION.json", session)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--render-root", type=Path)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    packet, task_root = authorize(manifest["task_id"])
    if artifact_ref(args.manifest) != packet["inputs"]["manifest"]:
        raise RuntimeError("MANIFEST_SHA_DRIFT")
    for ref in packet["runtime_code"]:
        checked(ref)
    if args.render_root:
        args.render_root.resolve().relative_to(task_root)
        cache = args.render_root.resolve() / "cache"
        cache.mkdir(exist_ok=True)
        for name in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
            os.environ[name] = str(cache)
        import tempfile
        tempfile.tempdir = str(cache)
        from chaoyang.ops.run_exact78_standalone_replay_v2 import replay
        replay({}, args.render_root.resolve())
        result_path = args.render_root / "REPLAY_RESULT.json"
        result = load_json(result_path)
        reused = next(s for s in manifest["sessions"] if Path(s["output_root"]).resolve() == args.render_root.resolve()).get("completed_stage_root")
        result.update(status="RAW_TO_ROBOT_EXECUTION_COMPLETE_PENDING_VISUAL_REVIEW",
                      mask_inference_fresh=not bool(reused), clean_inference_fresh=not bool(reused),
                      motion_inference_fresh=not bool(reused), completed_fresh_stage_source=reused,
                      reuse_receipt=artifact_ref(args.render_root / "STAGE_REUSE_RECEIPT.json") if reused else None)
        atomic_json(result_path, result)
        return 0
    task_id = manifest["task_id"]
    verify_refs(packet["inputs"])
    verify_refs(manifest)
    results = []
    for spec in manifest["sessions"]:
        root = Path(spec["output_root"]).resolve()
        root.relative_to(REPO_ROOT.resolve())
        root.mkdir(parents=True)
        front = {**spec, "task_id": task_id, "output_root": str(root / "frontend")}
        spec_path = root / "FRONTEND_SPEC.json"
        atomic_json(spec_path, front)
        try:
            if spec.get("completed_stage_root"):
                source = Path(spec["completed_stage_root"]).resolve(strict=True)
                source.relative_to(REPO_ROOT)
                refs = spec["completed_stage_refs"]
                verify_refs(refs)
                recompute_robot = spec.get("recompute_robot_with_task_prior", False)
                if not recompute_robot:
                    for stage in ("arm", "hand"):
                        if load_json(checked(refs[stage]))["status"] != "PASS_NUMERIC_CANARY_NO_AUTHORITY":
                            raise RuntimeError("REUSE_REQUIRES_COMPLETE_NUMERIC_STAGES")
                if load_json(checked(refs["bounded"]))["status"] != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
                    raise RuntimeError("REUSE_REQUIRES_BOUNDED_PASS")
                if load_json(checked(refs["temporal"]))["status"] != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
                    raise RuntimeError("REUSE_REQUIRES_TEMPORAL_PASS")
                for stage in ("raw", "mask", "clean"):
                    rows = load_json(checked(refs[stage]))["rows"]
                    if [r["frame_id"] for r in rows] != list(range(spec["frame_count"])):
                        raise RuntimeError("REUSE_REQUIRES_COMPLETE_FRAME_SET")
                if not load_json(checked(refs["mask"]))["fresh_model_inference"] or not load_json(checked(refs["clean"]))["fresh_mask_and_clean_inference"]:
                    raise RuntimeError("REUSE_REQUIRES_FRESH_SOURCE_INFERENCE")
                admit_masks(spec, checked(refs['mask']), root)
                if recompute_robot:
                    hawor, arm, hand = retarget(spec, root, checked(refs["temporal_npz"]), checked(refs["temporal"]))
                    assemble(spec, root, hawor, arm, hand, front=source / "frontend")
                else:
                    bundle_manifest = load_json(checked(refs["bundle"]))
                    if bundle_manifest["session_id"] != spec["session_id"]:
                        raise RuntimeError("REUSE_SESSION_IDENTITY")
                    for name, expected in bundle_manifest["files"].items():
                        path = source / "bundle" / name
                        path.resolve(strict=True).relative_to(source / "bundle")
                        actual = artifact_ref(path)
                        if path.is_symlink() or any(actual[k] != expected[k] for k in ("bytes", "sha256")):
                            raise RuntimeError("REUSE_BUNDLE_DRIFT")
                    shutil.copytree(source / "bundle", root / "bundle")
                atomic_json(root / "STAGE_REUSE_RECEIPT.json", {
                    "status": "PASS_COMPLETE_CURRENT_STAGE_REUSE", "source_root": str(source),
                    "completed_stage_refs": refs, "unchanged_stage_code": spec["unchanged_stage_code"],
                    "model_inference_rerun_here": False,
                    "robot_solve_rerun_here": recompute_robot,
                    "claim_limit": "Only independently complete fresh model stages are reused; failed or partial render files are not inputs."})
            else:
                for phase in ("raw", "masks", "prepare", "clean"):
                    command = cli("run_exact78_fresh_frontend_v1", "--spec", spec_path, "--phase", phase)
                    if phase in {"masks", "clean"}:
                        gpu(command, root, task_id, phase)
                    else:
                        execute(command, root / f"{phase}.log")
                    if phase == 'masks':
                        admit_masks(spec, root / 'frontend/MASK.json', root)
                hawor, arm, hand = motion(spec, root, task_id)
                assemble(spec, root, hawor, arm, hand)
            execute(cli("run_exact78_full_pipeline_v1", "--manifest", args.manifest,
                        "--render-root", root), root / "render.log")
            results.append({"session_id": spec["session_id"], "status": "EXECUTION_COMPLETE_VISUAL_REVIEW_PENDING",
                            "result": artifact_ref(root / "REPLAY_RESULT.json")})
        except Exception as error:
            results.append({"session_id": spec["session_id"], "status": "FAILED", "failure": str(error)})
            atomic_json(root / "FAILURE.json", results[-1])
            # One integration attempt per session; no quality-threshold relaxation.
        atomic_json(Path(manifest["output_root"]) / "PROGRESS.json", {"sessions": results})
    atomic_json(Path(manifest["output_root"]) / "RESULT.json", {
        "task_id": task_id, "sessions": results,
        "status": "EXECUTION_COMPLETE" if all(r["status"] != "FAILED" for r in results) else "INCOMPLETE",
        "deletion_authorized": False, "visual_quality_adopted": False, "training_eligible": False,
    })
    return 0 if all(r["status"] != "FAILED" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
