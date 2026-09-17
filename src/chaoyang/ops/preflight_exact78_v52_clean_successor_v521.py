from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""V5.2.1 bounded CPU preflight for every frozen exact78 Wave0 Clean row."""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, sha256_file, validate_artifact_ref


ROOT = Path(__file__).resolve().parents[3]
WAVE_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
SELECTION = WAVE_ROOT / "EXACT78_WAVE0_SELECTION.json"
EXPECTED_SHA = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"
EXECUTION_AUTHORITY = WAVE_ROOT / "EXECUTION_AUTHORITY_V52_1.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        return {"status": "ERROR", "stderr": completed.stderr[-1000:]}
    streams = json.loads(completed.stdout).get("streams", [])
    return {"status": "PASS", **(streams[0] if streams else {})}


def check_ref(reference: dict[str, Any], errors: list[str], label: str) -> None:
    errors.extend(f"{label}: {error}" for error in validate_artifact_ref(reference))


def valid_existing_clean(reference: dict[str, Any], session: str, frames: int, errors: list[str]) -> None:
    check_ref(reference, errors, "existing_clean")
    if errors:
        return
    value = load(Path(reference["path"]))
    if value.get("session") != session or value.get("frame_count") != frames:
        errors.append("existing Clean identity/frame mismatch")
    if value.get("grade") not in {"A", "B"} or value.get("downstream_authorized") is not True:
        errors.append("existing Clean is not downstream-authorized A/B")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection = load(SELECTION)
    global_errors = []
    if sha256_file(SELECTION) != EXPECTED_SHA:
        global_errors.append("frozen Wave0 SHA mismatch")
    rows = selection.get("sessions", [])
    session_ids = [row["session_id"] for row in rows]
    if len(rows) != 58 or len(set(session_ids)) != 58:
        global_errors.append("Wave0 must contain 58 unique sessions")
    authority = load(EXECUTION_AUTHORITY)
    vendor_errors: list[str] = []
    for reference in authority.get("vendor", {}).get("weights", {}).values():
        check_ref(reference, vendor_errors, "ProPainter weight")
    if authority.get("vendor", {}).get("commit") != "e870e79321c31b733e2031af5aa2fb1fe3ac7eec":
        vendor_errors.append("ProPainter commit mismatch")

    matrix = []
    for row in rows:
        session = row["session_id"]
        frames = int(row["frame_count"])
        errors: list[str] = []
        warnings: list[str] = []
        refs = row.get("upstream", {})
        for stage in ("hawor", "role_mask", "task_object_mask", "depth", "object6d"):
            if stage not in refs:
                errors.append(f"missing upstream {stage}")
            else:
                check_ref(refs[stage], errors, f"upstream.{stage}")
        hawor = load(Path(refs["hawor"]["path"])) if "hawor" in refs and not any("upstream.hawor" in e for e in errors) else {}
        role = load(Path(refs["role_mask"]["path"])) if "role_mask" in refs and not any("upstream.role_mask" in e for e in errors) else {}
        obj = load(Path(refs["task_object_mask"]["path"])) if "task_object_mask" in refs and not any("upstream.task_object_mask" in e for e in errors) else {}
        object6d = load(Path(refs["object6d"]["path"])) if "object6d" in refs and not any("upstream.object6d" in e for e in errors) else {}

        rgb_ref = hawor.get("inputs", {}).get("source_video")
        if isinstance(rgb_ref, dict):
            check_ref(rgb_ref, errors, "selected_rgb")
            video_probe = ffprobe(Path(rgb_ref["path"]))
            if video_probe.get("status") != "PASS":
                errors.append("selected RGB ffprobe failed")
            elif str(video_probe.get("nb_frames", "N/A")).isdigit() and int(video_probe["nb_frames"]) != frames:
                errors.append(f"selected RGB frame mismatch={video_probe['nb_frames']} expected={frames}")
        else:
            errors.append("HaWoR source_video authority missing")
            video_probe = {"status": "MISSING"}
        if hawor.get("session_id") != session or hawor.get("validation", {}).get("full_video", {}).get("input_frames") != frames:
            errors.append("HaWoR identity/frame mismatch")
        if role.get("session") != session or role.get("frame_count") != frames or role.get("grade") not in {"A", "B"}:
            errors.append("Role Mask identity/frame/grade mismatch")
        role_lineage_mode = None
        predecessor = role
        predecessor_ref = role.get("pins", {}).get("immutable_predecessor_result")
        if isinstance(predecessor_ref, dict) and {"path", "bytes", "sha256"} <= set(predecessor_ref):
            check_ref(predecessor_ref, errors, "role immutable predecessor")
            if not any("role immutable predecessor" in error for error in errors):
                predecessor = load(Path(predecessor_ref["path"]))
                role_lineage_mode = "RECONCILED_WRAPPER_TO_IMMUTABLE_PREDECESSOR"
        else:
            role_lineage_mode = "DIRECT_ROLE_RESULT"
        role_config = predecessor.get("pins", {}).get("config")
        if isinstance(role_config, dict):
            check_ref(role_config, errors, "role config")
        else:
            errors.append("role lineage has no exact config reference")
        role_manifest = role.get("artifacts", {}).get("frame_manifest")
        if isinstance(role_manifest, dict):
            check_ref(role_manifest, errors, "role frame manifest")
        else:
            errors.append("Role Mask frame manifest missing")
        if obj.get("session") != session or obj.get("frame_count") != frames or obj.get("grade") not in {"A", "B"}:
            errors.append("Object Mask identity/frame/grade mismatch")
        object_manifest = obj.get("artifacts", {}).get("manifest")
        if isinstance(object_manifest, dict):
            check_ref(object_manifest, errors, "object mask manifest")
        else:
            errors.append("Object Mask manifest missing")
        chips_instance_gate = None
        if row["task"] == "chips":
            chips_instance_gate = bool(
                obj.get("instance_count") == 3
                and obj.get("gates", {}).get("three_instances_disjoint") is True
                and obj.get("gates", {}).get("no_union_or_identity_switch_fabricated") is True
                and object6d.get("physical_instance_count") == 3
                and object6d.get("multi_instance_union_used") is False
            )
            if not chips_instance_gate:
                errors.append("Chips three-instance independence gate failed")

        existing = row.get("existing_clean")
        canonical = WAVE_ROOT / "propainter_v1" / session / "RESULT.json"
        terminal_kind = None
        if existing:
            valid_existing_clean(existing, session, frames, errors)
            terminal_kind = "ADOPTED_EXISTING"
        elif canonical.is_file():
            valid_existing_clean(artifact_ref(canonical), session, frames, errors)
            terminal_kind = "VERIFIED_NEW_FINAL"

        prepare_receipt = WAVE_ROOT / "preparation_receipts" / f"{session}.json"
        prepare_state = "NOT_PREPARED_FRESH_REQUIRED"
        prepare_ref = None
        if prepare_receipt.is_file():
            prepare_ref = artifact_ref(prepare_receipt)
            prepared = load(prepare_receipt)
            if prepared.get("session") == session or prepared.get("session_id") == session:
                prepare_state = "EXISTING_PREPARE_IDENTITY_VERIFIED"
            else:
                errors.append("prepare receipt session mismatch")
        donor_result = WAVE_ROOT / "real_donor_v1" / session / "RESULT.json"
        donor_manifest = WAVE_ROOT / "real_donor_v1" / session / "SOURCE_MAP_MANIFEST.json"
        donor_state = "NOT_PREPARED_SAME_SESSION_REQUIRED"
        donor_refs = None
        if donor_result.is_file() or donor_manifest.is_file():
            if not donor_result.is_file() or not donor_manifest.is_file():
                errors.append("partial same-session donor artifacts")
            else:
                donor = load(donor_result)
                if donor.get("session") != session:
                    errors.append("donor session identity mismatch")
                donor_refs = {"result": artifact_ref(donor_result), "manifest": artifact_ref(donor_manifest)}
                donor_state = "EXISTING_SAME_SESSION_DONOR_VERIFIED"

        partial_paths = []
        for candidate in (
            WAVE_ROOT / "sessions" / session,
            WAVE_ROOT / "real_donor_v1" / session,
            WAVE_ROOT / "propainter_v1" / session,
        ):
            if candidate.exists() and not terminal_kind and candidate == WAVE_ROOT / "propainter_v1" / session:
                partial_paths.append(str(candidate))
        if partial_paths:
            errors.append("uncommitted canonical output conflict")
        errors.extend(vendor_errors)
        if terminal_kind and not errors:
            status = "TERMINAL_PASSED"
        elif errors:
            status = "CONFLICT" if any("conflict" in error or "mismatch" in error for error in errors) else "BLOCKED_PREREQ"
        else:
            status = "READY"
        matrix.append({
            "position": row["position"], "task": row["task"], "session_id": session,
            "frame_count": frames, "status": status, "terminal_kind": terminal_kind,
            "raw_path": row["raw_path"], "selected_rgb": rgb_ref,
            "rgb_probe": video_probe, "role_manifest": role_manifest,
            "object_manifest": object_manifest, "chips_three_instance_gate": chips_instance_gate,
            "prepare_state": prepare_state, "prepare_receipt": prepare_ref,
            "donor_state": donor_state, "donor": donor_refs,
            "vendor_pinned": not vendor_errors, "role_lineage_mode": role_lineage_mode,
            "partial_paths": partial_paths,
            "errors": errors, "warnings": warnings,
        })
    counts = {status: sum(item["status"] == status for item in matrix) for status in ("TERMINAL_PASSED", "READY", "BLOCKED_PREREQ", "CONFLICT")}
    result = {
        "schema_version": "exact78-wave0-clean-preflight-v52.1", "created_at": now_iso(),
        "status": "PASS" if not global_errors and not counts["BLOCKED_PREREQ"] and not counts["CONFLICT"] else "HOLD",
        "selection": artifact_ref(SELECTION), "execution_authority": artifact_ref(EXECUTION_AUTHORITY),
        "counts": counts, "sessions": matrix, "global_errors": global_errors,
        "claim_limit": "CPU readiness only; READY does not authorize GPU execution or claim Clean quality.",
    }
    json_path = output_root / "EXACT78_WAVE0_CLEAN_PREFLIGHT.json"
    csv_path = output_root / "EXACT78_WAVE0_CLEAN_PREFLIGHT.csv"
    atomic_json(json_path, result)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["position", "task", "session_id", "frame_count", "status", "terminal_kind", "prepare_state", "donor_state", "errors"])
        writer.writeheader()
        for item in matrix:
            writer.writerow({key: (" | ".join(item[key]) if key == "errors" else item.get(key)) for key in writer.fieldnames})
    print(json.dumps({"status": result["status"], "counts": counts, "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
