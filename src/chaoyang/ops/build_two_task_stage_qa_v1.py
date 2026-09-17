#!/usr/bin/env python3
"""Freeze stage QA for the two-task offline visual baseline.

Usage: python src/chaoyang/ops/build_two_task_stage_qa_v1.py --output-root \
  archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1/stage_qa

Only SHA-pinned prior results and the new hypothesis-only Contact are consumed.
This does not promote any RC1, training, physical, or contact authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROUTE_DIR = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1/route/attempt_0001"
EXPECTED = {"get_potato_chips_0902_103": ("chips", 284, 3), "play_cards_0902_042": ("poker", 171, 1)}


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(ref: dict) -> Path:
    actual = artifact(Path(ref["path"]))
    if actual["bytes"] != ref["bytes"] or actual["sha256"] != ref["sha256"]:
        raise ValueError(f"SHA closure failed: {ref['path']}")
    return Path(actual["path"])


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    route_path = ROUTE_DIR / "BASELINE_ROUTE_V1.json"
    freeze_path = ROUTE_DIR / "SOURCE_FREEZE.json"
    route, freeze = read(route_path), read(freeze_path)
    if route["route_id"] != "chips_poker_exact78_offline_visual_v1":
        raise ValueError("route ID mismatch")
    checked(route["source_freeze"])
    if set(freeze["sessions"]) != set(EXPECTED):
        raise ValueError("source freeze session set mismatch")
    out_refs = {}
    for sid, (task, nframes, nobjects) in EXPECTED.items():
        frozen = freeze["sessions"][sid]
        if (frozen["task"], frozen["frame_count"]) != (task, nframes):
            raise ValueError(f"{sid} task/frame closure mismatch")
        stage_rows = {}
        for name, ref in frozen["stages"].items():
            stage_result = read(checked(ref))
            stage_session = stage_result.get("session_id", stage_result.get("session"))
            if stage_session != sid:
                raise ValueError(f"{sid} {name} session mismatch")
            stage_rows[name] = {"source_state": "PINNED_REUSED", "result": ref,
                                "reported_status": stage_result.get("status"),
                                "authority_promoted_by_this_run": False}
        raw = frozen["raw_video"]
        robot_video = frozen["robot_video"]
        robot_result = frozen["robot_result"]
        clean_ref = frozen["clean_result"]
        for ref in (raw, robot_video, robot_result, clean_ref):
            checked(ref)
        clean = read(Path(clean_ref["path"]))
        depth = read(Path(stage_rows["depth"]["result"]["path"]))
        role = read(Path(stage_rows["role_mask"]["result"]["path"]))
        obj_mask = read(Path(stage_rows["task_object_mask"]["result"]["path"]))
        obj = read(Path(stage_rows["object6d"]["result"]["path"]))
        if clean.get("frame_count") != nframes or obj_mask.get("frame_count") != nframes:
            raise ValueError(f"{sid} Clean or Object Mask frame count mismatch")
        if obj.get("physical_instance_count") != nobjects:
            raise ValueError(f"{sid} physical object count mismatch")
        if task == "chips" and obj.get("multi_instance_union_used") is not False:
            raise ValueError(f"{sid} three-bag union is forbidden")
        for key in ("source_map_manifest", "clean_synthetic_master", "chinese_full_review"):
            checked(clean["artifacts"][key])
        for key in ("registration_authority", "frame_manifest"):
            checked(depth[key])
        hand_result = read(Path(stage_rows["hawor"]["result"]["path"]))
        checked(hand_result["outputs"]["npz"])
        contact_path = output_root / ("chips103_contact" if task == "chips" else "poker042_contact") / "attempt_0001/RESULT.json"
        contact = read(contact_path)
        if contact.get("session_id") != sid or contact.get("frame_count") != nframes:
            raise ValueError(f"{sid} Contact result frame/session mismatch")
        if contact.get("claim_status") != "HYPOTHESIS_ONLY" or contact.get("contact_ground_truth") is not False:
            raise ValueError(f"{sid} Contact authority violation")
        if contact.get("total_rows") != nframes * nobjects * 10:
            raise ValueError(f"{sid} Contact row count mismatch")
        hypothesis_path = checked(contact["outputs"]["contact_hypotheses"])
        frame_state = defaultdict(Counter)
        row_count = 0
        with hypothesis_path.open(encoding="utf-8") as stream:
            for text in stream:
                row = json.loads(text)
                f = int(row["frame_id"])
                if not 0 <= f < nframes or row["claim_status"] != "HYPOTHESIS_ONLY":
                    raise ValueError(f"{sid} Contact frame or claim violation")
                frame_state[f][row["state"]] += 1
                row_count += 1
        if row_count != contact["total_rows"] or set(frame_state) != set(range(nframes)):
            raise ValueError(f"{sid} Contact row/frame coverage mismatch")
        instance_frames = {key: int(value["direct_object_frames"]) for key, value in contact["object_instances"].items()}
        if len(instance_frames) != nobjects:
            raise ValueError(f"{sid} Contact object identity count mismatch")

        attempt_dir = output_root / ("chips103" if task == "chips" else "poker042") / "attempt_0001"
        if attempt_dir.exists():
            raise FileExistsError(f"immutable QA attempt exists: {attempt_dir}")
        attempt_dir.mkdir(parents=True)
        flags_path = attempt_dir / "FRAME_FLAGS.jsonl"
        with flags_path.open("w", encoding="utf-8") as sink:
            for f in range(nframes):
                counts = frame_state[f]
                flags = ["OFFLINE_VISUAL", "NOT_FOR_TRAINING", "CONTACT_HYPOTHESIS_ONLY"]
                if counts["UNKNOWN"]:
                    flags.append("CONTACT_UNKNOWN_PRESENT")
                sink.write(json.dumps({"frame_id": f, "contact_state_counts": dict(counts),
                                       "flags": flags}, ensure_ascii=False) + "\n")

        stages = {
            "raw": {"source_state": "PINNED_REUSED", "video": raw, "frame_count": nframes},
            **stage_rows,
            "clean": {"source_state": "PINNED_REUSED", "result": clean_ref,
                      "reported_status": clean["status"], "source_map": clean["artifacts"]["source_map_manifest"],
                      "source_map_sha_verified": True,
                      "visible_object_byte_exact_reported": clean["hard_gates"].get("protected_object_byte_exact") == "PASS",
                      "changed_outside_removal_zero_reported": clean["hard_gates"].get("changed_outside_removal_zero") == "PASS",
                      "semantic_contact_boundary_authority": False},
            "contact": {"source_state": "PRODUCED_THIS_RUN", "result": artifact(contact_path),
                        "reported_status": contact["status"], "claim_status": "HYPOTHESIS_ONLY",
                        "coordinate_domain": contact["coordinate_domain"],
                        "direct_object_frames_by_instance": instance_frames},
            "robot": {"source_state": "PINNED_REUSED", "result": robot_result,
                      "video": robot_video, "reported_hard_geometry_pass": frozen["robot_hard_geometry_pass"],
                      "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION", "control_ground_truth": False},
            "visible_surface_compositor": {"source_state": "NOT_PRODUCED_BY_STAGE_QA",
                                           "status": "READY_FOR_INTEGRATION_NOT_VERIFIED"},
        }
        warnings = [
            "HaWoR numeric closure and 3D hand depth are not external truth; full trajectory may use future frames.",
            "Depth is metric optical Z but same-session external depth accuracy and registration P90 are not established here.",
            "Object6D is direct-observed visual proxy only; invalid frames remain unknown.",
            "Clean Grade-B proves source-map/decoding/pixel gates, not semantic donor or hidden object recovery.",
            "Contact uses MANO fingertip joint centers and Object6D proxy, not pad surface or ground truth.",
            "Robot hard geometry is offline visualization, not control or causal training input.",
        ]
        if task == "chips":
            warnings.append("Role Mask left tracker has zero expected-visible denominator; terminal gate is not-applicable, not proven coverage.")
            warnings.append("All three-bag Contact states are UNKNOWN, even where distance is finite; do not interpret as confirmed no contact.")
        else:
            warnings.append("Poker same-face identity under real occlusion is unverified; hidden face remains UNKNOWN.")
        qa = {
            "schema_version": "two-task-stage-qa-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
            "route_id": route["route_id"], "route_version": route["route_version"],
            "route_manifest": artifact(route_path), "source_freeze": artifact(freeze_path),
            "session_id": sid, "task": task, "frame_count": nframes, "fps": 30,
            "status": "PASSED_PINNED_STAGE_EVIDENCE_WITH_DEVELOPMENT_CONTACT",
            "full_pipeline_optimized_quality_pass": False,
            "training_eligible": False, "control_ground_truth": False, "contact_authority": False,
            "stages": stages,
            "source_quality": {
                "role_mask_status": role["status"], "object_mask_status": obj_mask["status"],
                "clean_real_donor_fraction_within_removal": clean["pixel_fractions_within_removal"]["real_donor"],
                "clean_synthetic_fraction_within_removal": clean["pixel_fractions_within_removal"]["synthetic_propainter"],
                "contact_state_counts": contact["state_counts"],
                "contact_direct_object_frames_by_instance": instance_frames,
                "external_accuracy": "UNKNOWN",
            },
            "frame_flags": artifact(flags_path),
            "warnings": warnings,
            "claim_limit": "Stage and SHA closure only. Full visual compositor/video quality decided by separate integration receipt.",
        }
        write(attempt_dir / "STAGE_QA.json", qa)
        comparison = {
            "session_id": sid,
            "baseline_clean_grade": clean.get("grade"),
            "baseline_robot_hard_geometry_pass": frozen["robot_hard_geometry_pass"],
            "new_contact_selected_camera_hypothesis_produced": True,
            "new_mask_quality_comparison": "NOT_RUN",
            "new_clean_semantic_donor_comparison": "NOT_RUN",
            "new_depth_external_accuracy_comparison": "NOT_AVAILABLE",
            "optimization_claim": "UNPROVEN",
        }
        write(attempt_dir / "QUALITY_COMPARISON.json", comparison)
        out_refs[sid] = artifact(attempt_dir / "STAGE_QA.json")
    print(json.dumps(out_refs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
