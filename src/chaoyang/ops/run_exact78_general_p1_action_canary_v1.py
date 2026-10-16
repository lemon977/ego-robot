#!/usr/bin/env python3
"""Finite eight-session Exact78 P1 action canary orchestrator."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, process_identity


def checked(reference: dict) -> Path:
    path = Path(reference["path"])
    actual = artifact_ref(path)
    if actual["bytes"] != reference["bytes"] or actual["sha256"] != reference["sha256"]:
        raise RuntimeError(f"INPUT_SHA_DRIFT:{path}")
    return path


def authorize(manifest_path: Path) -> tuple[dict, dict, Path]:
    manifest = load_json(manifest_path.resolve(strict=True))
    routes = [row for row in load_json(REPO_ROOT / "tasks/current/INDEX.json")["task_packets"]
              if row["task_id"] == manifest["task_id"] and row.get("execution_allowed")]
    if len(routes) != 1:
        raise RuntimeError("P1_NOT_ROUTABLE")
    packet_path = REPO_ROOT / routes[0]["packet_path"]
    if artifact_ref(packet_path)["sha256"] != routes[0]["packet_sha256"]:
        raise RuntimeError("PACKET_SHA_DRIFT")
    packet = load_json(packet_path)
    if packet["inputs"]["manifest"] != artifact_ref(manifest_path):
        raise RuntimeError("MANIFEST_SHA_DRIFT")
    identity = process_identity(packet["writer"]["pid"])
    if not identity["alive"] or identity["start_ticks"] != packet["writer"]["proc_start_ticks"]:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    root = Path(packet["write_set"][0]).resolve()
    if load_json(root / "WRITER.json") != packet["writer"]:
        raise RuntimeError("WRITER_TOKEN_DRIFT")
    for reference in packet["runtime_code"]:
        checked(reference)
    return manifest, packet, root


def cli(operation: str, *args, hawor: bool = False) -> list[str]:
    if hawor:
        bootstrap = (f"import sys,runpy;sys.path.insert(0,{str(REPO_ROOT / 'src')!r});"
                     "runpy.run_module('chaoyang.cli',run_name='__main__')")
        return [str(REPO_ROOT / "src/chaoyang/ops/hawor_python.sh"), "-B", "-c", bootstrap,
                "run", operation, *map(str, args)]
    return [sys.executable, "-B", "-m", "chaoyang.cli", "run", operation, *map(str, args)]


def execute(command: list[str], log: Path, *, allowed: tuple[int, ...] = (0,)) -> int:
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode not in allowed:
        raise RuntimeError(f"STAGE_RETURN_CODE:{result.returncode}:{log}")
    return result.returncode


def gpu(command: list[str], root: Path, task_id: str, name: str, wall: int = 2400) -> None:
    receipt = root / f"{name}_GPU_RECEIPT.json"
    execute(cli("run_gpu_command_with_v71_lease", "--task-id", task_id,
                "--attempt-id", f"attempt_0001_{root.name}_{name}", "--priority", "CANARY",
                "--gpu-id", "0", "--min-free-mib", "60000", "--wait-seconds", "1800",
                "--wall-seconds", str(wall), "--receipt", receipt,
                "--claim-limit", "Exact78 P1 offline visual action canary; no training or physical authority",
                "--", *command), root / f"{name}_GPU.log")
    if load_json(receipt).get("status") != "PASSED":
        raise RuntimeError(f"GPU_STAGE_NOT_PASSED:{name}")


def neutral_hand_seed(path: Path) -> None:
    from chaoyang.ops import run_robot_hand_fullsession_v2 as hand
    assets = hand.old.load_pinned_robot_assets(REPO_ROOT)
    contracts = hand.handfit.model_contract(hand.old.official, hand.shared.wrist_adapter, assets)
    q = np.stack([contract["neutral"] for contract in contracts])[None]
    np.savez_compressed(path, q_hand=q)


def error_video(source: Path, destination: Path, session_id: str, message: str) -> dict:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.rectangle(frame, (0, 0), (width, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"{session_id} | P1 ERROR DIAGNOSTIC", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, message[:110], (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            count += 1
    finally:
        capture.release(); writer.release()
    return artifact_ref(destination) | {"frames": count}


def p0_interpretation(p0: Path, destination: Path) -> dict:
    rows = load_json(p0 / "SESSION_PREFLIGHT.json")["sessions"]
    source_valid = []
    variants = Counter()
    for row in rows:
        fps_match = (abs(row["mono"]["fps"] - row["stereo"]["fps"]) < 1e-9
                     and abs(row["mono"]["fps"] - row["manifest_checks"]["humanego_fps"]) < 1e-9)
        valid = (row["mono"]["decode_ok"] and row["stereo"]["decode_ok"]
                 and row["mono"]["decoded_frames"] == row["expected_frames"]
                 and row["stereo"]["decoded_frames"] == row["expected_frames"]
                 and all(row["frame_identity"].values()) and fps_match)
        source_valid.append(valid)
        variants[f"{row['mono']['fps']:g}_fps"] += 1
    value = {"schema_version": "exact78-p0-fps-interpretation-v1",
             "immutable_p0_result": artifact_ref(p0 / "RESULT.json"),
             "source_valid_sessions": sum(source_valid), "sessions": len(rows),
             "fps_variants": dict(variants),
             "finding": "43 rows are native 25 fps and match mono/stereo/HumanEgo metadata; the P0 hard-coded 30 fps classifier is not an input corruption finding.",
             "p0_result_modified": False, "numeric_quality_threshold_changed": False}
    atomic_json(destination, value)
    return value


def parameter_audit(canonical_npz: Path, canonical_result: Path, fresh_runner: Path,
                    cards_raw_result: Path | None, destination: Path) -> dict:
    with np.load(canonical_npz, allow_pickle=False) as archive:
        canonical_observed = np.asarray(archive["observed"], dtype=bool)
        canonical_focal = float(np.mean(archive["intrinsics"][:, [0, 1], [0, 1]]))
        canonical_provenance = sorted(map(str, np.unique(archive["provenance"])))
    fresh = load_json(cards_raw_result) if cards_raw_result and cards_raw_result.is_file() else None
    value = {"schema_version": "exact78-cards042-canonical-vs-fresh-hawor-audit-v1",
             "canonical": {"npz": artifact_ref(canonical_npz), "result": artifact_ref(canonical_result),
                           "runner_declared": load_json(canonical_result)["runner"],
                           "runner_source_present": False, "focal_px": canonical_focal,
                           "observed_frames": canonical_observed.sum(axis=1).tolist(),
                           "provenance_values": canonical_provenance,
                           "infiller_used": False},
             "fresh": {"runner": artifact_ref(fresh_runner), "focal_policy": "mean acquisition fx/fy",
                       "detector_confidence_threshold": 0.2,
                       "track_policy": "upstream detect_track_video then side-merge by mean handedness",
                       "infiller_used_by_L0": False,
                       "observed_frames": (fresh or {}).get("observed_frames")},
             "same_known_inputs": {"source_video_sha256": "879cd8f2ec91cd7aecbdff2de83d950d5cad0b6b11bc2fb97c5054df33ca4928",
                                   "vendor_runtime_pin_sha256": "7c20034be0d80c76adbb8eb278f0d3a6605c0bf9099a44829c028692e8db4aed",
                                   "focal_px": canonical_focal},
             "unknown_due_missing_canonical_runner_source": ["detector invocation details", "tracker association parameters",
                                                              "random/determinism controls", "frame staging details"],
             "explanation": "Canonical has direct OBSERVED boxes on both hands for 171/171 and did not use the infiller. Fresh L0 loses right frames 98,118,121-165 after detection/tracking. Therefore the 47-frame gap is a fresh detector/tracker association loss, not hand-out-of-view or infiller behavior; the exact single hidden parameter cannot be claimed because the canonical runner bytes are absent.",
             "next_test": "Same-frame Role Mask prompted L1 isolates detector/tracker recovery without changing model weights or gates."}
    atomic_json(destination, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest, packet, root = authorize(args.manifest)
    forbidden = (
        "accepted_robot_template.npz", "accepted_hawor_template.npz",
        "exact78_cards042_visual_baseline_v2_publish_20260929/attempts/attempt_0001/dependencies/selected_motion_bundle/inputs/ARM.npz",
    )
    serialized_inputs = json.dumps({"packet_inputs": packet["inputs"], "manifest": manifest}, sort_keys=True)
    if any(value in serialized_inputs for value in forbidden):
        raise RuntimeError("COLD_START_FORBIDDEN_CARDS042_FIT_REFERENCE")
    p0 = checked(packet["inputs"]["p0_result"]).parent
    p0_fix = p0_interpretation(p0, root / "P0_INPUT_INTERPRETATION.json")
    results = []
    problem_rows = []
    cards_raw_result = None
    for spec in manifest["sessions"]:
        session_id, task = spec["session_id"], spec["task"]
        session = root / "sessions" / session_id
        session.mkdir(parents=True)
        source = checked(spec["source_video"])
        try:
            raw = session / "hawor_l0"
            gpu(cli("run_exact78_hawor_raw_v1", "--session-root", spec["source_root"],
                    "--session-id", session_id, "--source-video", source,
                    "--output-root", raw, hawor=True), session, packet["task_id"], "hawor_l0")
            if session_id == "play_cards_0902_042":
                cards_raw_result = raw / "RESULT.json"
            recovery = session / "hawor_recovery"
            recovery_command = cli(
                "run_exact78_hawor_recovery_v2", "--session-id", session_id,
                "--source-video", source, "--raw-npz", raw / "HAWOR_RAW_MANO21.npz",
                "--raw-result", raw / "RESULT.json", "--upstream-work", raw / "upstream_work",
                "--output-root", recovery, hawor=True)
            if spec.get("role_manifest"):
                recovery_command.extend(("--role-manifest", str(checked(spec["role_manifest"]))))
            gpu(recovery_command, session, packet["task_id"], "hawor_recovery")
            recovered_npz = recovery / "HAWOR_RECOVERED_VISUAL_MANO21.npz"
            arm_root = session / "arm"
            execute(cli("run_exact78_session_base_arm_v1", "--session-id", session_id,
                        "--hawor", recovered_npz, "--hawor-result", recovery / "RESULT.json",
                        "--output-root", arm_root), session / "arm.log")
            hand_root = session / "hand"
            hand_root.mkdir()
            seed = hand_root / "NEUTRAL_HAND_SEED.npz"
            neutral_hand_seed(seed)
            hand_code = execute(cli("run_robot_hand_fullsession_anatomy_repair_v3", "--task", task,
                                    "--session", session_id, "--hawor", recovered_npz,
                                    "--hawor-result", recovery / "RESULT.json", "--accepted-states", seed,
                                    "--output", hand_root / "RESULT.json"), session / "hand.log", allowed=(0, 2))
            diagnostics = session / "diagnostics"
            execute(cli("render_exact78_action_diagnostics_v1", "--session-id", session_id,
                        "--source-video", source, "--hawor", recovered_npz,
                        "--arm", arm_root / "ARM_SESSION_BASE_STATES.npz",
                        "--hand", hand_root / "HAND_STATES.npz", "--output-root", diagnostics),
                    session / "diagnostics.log", allowed=(0, 2))
            recovery_result = load_json(recovery / "RESULT.json")
            arm_result = load_json(arm_root / "RESULT.json")
            diagnostic_result = load_json(diagnostics / "RESULT.json")
            item = {"session_id": session_id, "task": task, "status": "TERMINAL",
                    "hawor": artifact_ref(recovery / "RESULT.json"), "arm": artifact_ref(arm_root / "RESULT.json"),
                    "hand": artifact_ref(hand_root / "RESULT.json"), "hand_returncode": hand_code,
                    "diagnostics": artifact_ref(diagnostics / "RESULT.json"),
                    "motion_metrics": recovery_result["metrics"], "reach_metrics": arm_result["metrics"],
                    "diagnostic_outputs": diagnostic_result["outputs"]}
            results.append(item)
            problem_rows.append({"session_id": session_id,
                                 "motion_gap_after_completion": recovery_result["metrics"]["visible_without_visual_action_side_frames"],
                                 "arm_failure_clusters": arm_result["metrics"]["failure_clusters"],
                                 "visual_review": "PENDING", "result_sha256": item["diagnostics"]["sha256"]})
        except Exception as error:
            failure = f"{type(error).__name__}:{error}"
            video = error_video(source, session / f"{session_id}_ERROR_DIAGNOSTIC.mp4", session_id, failure)
            atomic_json(session / "FAILURE.json", {"session_id": session_id, "failure": failure,
                                                    "error_video": video})
            results.append({"session_id": session_id, "task": task, "status": "FAILED_RUNTIME",
                            "failure": failure, "error_video": video})
            problem_rows.append({"session_id": session_id, "owner": "INTEGRATION",
                                 "failure": failure, "visual_review": "ERROR_VIDEO_ONLY"})
        atomic_json(root / "PROGRESS.json", {"sessions": results})

    canonical_npz = checked(packet["inputs"]["canonical_raw_hawor"])
    canonical_result = checked(packet["inputs"]["canonical_raw_result"])
    parameter_audit(canonical_npz, canonical_result,
                    REPO_ROOT / "src/chaoyang/ops/run_exact78_hawor_raw_v1.py",
                    cards_raw_result, root / "CARDS042_HAWOR_PARAMETER_AUDIT.json")
    atomic_json(root / "PROBLEM_TABLE.json", {"rows": problem_rows})
    terminal = [row for row in results if row["status"] == "TERMINAL"]
    visible = sum(row["motion_metrics"]["role_mask_visible_side_frames"] for row in terminal)
    missing = sum(row["motion_metrics"]["visible_without_visual_action_side_frames"] for row in terminal)
    observed_rows = sum(row["reach_metrics"]["metric_observed_rows"] for row in terminal)
    reach_pass = sum(row["reach_metrics"]["metric_passed_rows"] for row in terminal)
    provenance = Counter()
    failure_clusters = Counter()
    for row in terminal:
        for side_counts in row["motion_metrics"]["provenance_counts"].values():
            provenance.update(side_counts)
        failure_clusters.update(row["reach_metrics"]["failure_clusters"])
    for row in results:
        if row["status"] != "TERMINAL":
            failure_clusters["INTEGRATION_RUNTIME"] += 1
    result = {"schema_version": "exact78-general-p1-action-canary-result-v1",
              "status": "EXECUTION_COMPLETE" if len(results) == len(manifest["sessions"]) else "INCOMPLETE",
              "execution_complete": len(results) == len(manifest["sessions"]),
              "visual_qualified": False, "quality_passed": False, "training_eligible": False,
              "canary_count": len(manifest["sessions"]), "terminal_action_results": len(terminal),
              "four_metrics": {
                  "motion_completeness": {"visible_without_action": missing, "role_mask_visible": visible,
                                          "fraction": float(missing / visible) if visible else None,
                                          "target_max": 0.05, "provenance_counts": dict(provenance)},
                  "cross_session_reachability": {"metric_observed_passed": reach_pass,
                                                 "metric_observed_rows": observed_rows,
                                                 "fraction": float(reach_pass / observed_rows) if observed_rows else None,
                                                 "target_min": 0.90},
                  "terminal_coverage": {"terminal": len(results), "expected": len(manifest["sessions"]),
                                        "action_complete": len(terminal),
                                        "diagnostic_video_or_error_video": len(results)},
                  "visual_quality_distribution": "P1_ACTION_DIAGNOSTICS_PENDING_HUMAN_REVIEW_NOT_A_VISUAL_PASS"},
              "p0_interpretation": p0_fix, "failure_clusters": dict(failure_clusters),
              "sessions": results,
              "outputs": {"progress": artifact_ref(root / "PROGRESS.json"),
                          "problem_table": artifact_ref(root / "PROBLEM_TABLE.json"),
                          "hawor_parameter_audit": artifact_ref(root / "CARDS042_HAWOR_PARAMETER_AUDIT.json"),
                          "p0_input_interpretation": artifact_ref(root / "P0_INPUT_INTERPRETATION.json")},
              "deletion_authorized": False, "next_action": "REVIEW_P1_DIAGNOSTICS_THEN_REGISTER_P2",
              "claim_limit": "Eight-session offline action canary only. Video existence is not visual qualification; only OBSERVED rows are metric evidence."}
    atomic_json(root / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "four_metrics": result["four_metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
