#!/usr/bin/env python3
"""Bound the PICO RGB timestamp/exporter proof gap for two exact sessions."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import subprocess

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, validate_artifact_ref


EXPECTED = {
    "get_potato_chips_0902_103": 284,
    "play_cards_0903_227": 196,
}


def inspect_audit(path: Path) -> dict:
    audit_ref = artifact_ref(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    manifest_ref = audit["inputs"]["clip_manifest"]
    errors = validate_artifact_ref(manifest_ref)
    if errors:
        raise RuntimeError("clip manifest closure failed: " + "; ".join(errors))
    manifest_path = Path(manifest_ref["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = manifest["clip_name"]
    if session not in EXPECTED or audit["frames"] != EXPECTED[session] or manifest["video"]["frame_count"] != EXPECTED[session]:
        raise RuntimeError(f"unexpected exact session/frame binding: {session}")
    clip_root = manifest_path.parent
    rgb = clip_root / manifest["files"]["video"]
    stereo = clip_root / manifest["files"]["source_stereo_video"]
    humanego = clip_root / "preprocess/pico_humanego_manifest.json"
    first_frame = clip_root / "preprocess/all_data/00000/training_data.json"
    asset_refs = {name: artifact_ref(item) for name, item in (
        ("clip_rgb", rgb), ("clip_stereo", stereo), ("humanego_manifest", humanego), ("first_frame_metadata", first_frame)
    )}
    frame_metadata = json.loads(first_frame.read_text(encoding="utf-8"))["metadata"]
    stream = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name,time_base,r_frame_rate,avg_frame_rate:format_tags", "-of", "json", str(rgb)],
        check=True, text=True, capture_output=True,
    ).stdout)
    source_video_paths = [str(piece["source_video"]) for piece in manifest["pieces"]]
    return {
        "session_id": session,
        "frames": EXPECTED[session],
        "corrected_c2w_audit": audit_ref,
        "clip_manifest": manifest_ref,
        "assets": asset_refs,
        "first_frame_metadata_fields": sorted(frame_metadata),
        "first_frame_metadata_ts": frame_metadata.get("ts"),
        "first_frame_metadata_video_time_s": frame_metadata.get("video_time_s"),
        "video_container_probe": stream,
        "merged_source_video_paths": source_video_paths,
        "merged_source_video_exists_on_host": [Path(item).is_file() for item in source_video_paths],
        "recorded_exporter_root": manifest.get("editor_source_root"),
        "recorded_exporter_root_exists_on_host": Path(str(manifest.get("editor_source_root", ""))).is_dir(),
        "rgb_per_frame_capture_time_evidence": "NOT_FOUND_IN_INSPECTED_CLIP_ASSETS",
        "rgb_to_original_source_frame_map_evidence": "ONLY_MERGED_VIDEO_PIECE_TIME_RANGE_NOT_FRAME_IDENTITY",
        "old_exporter_source_evidence": "RECORDED_PATH_UNAVAILABLE_ON_THIS_HOST",
        "producer_level_prefix_suffix_test": "NOT_EXECUTABLE_WITH_INSPECTED_ASSETS",
        "camera_pose_producer_causality": "UNKNOWN_VERIFICATION_REQUIRED",
        "robot_causal_eligible": False,
        "claim_limit": "Metadata.ts equals indexed tracker time, not measured RGB exposure time. Missing inspected proof is not proof the original recorder never produced it.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips-audit", type=Path, required=True)
    parser.add_argument("--poker-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [inspect_audit(path.resolve(strict=True)) for path in (args.chips_audit, args.poker_audit)]
    if {row["session_id"] for row in rows} != set(EXPECTED):
        raise RuntimeError("full session IDs are not the frozen pair")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    value = {
        "schema_version": "chaoyang-rc1-rgb-producer-evidence-v1",
        "created_at": now_iso(),
        "status": "UNKNOWN_VERIFICATION_REQUIRED",
        "authority_promoted": False,
        "claim_limit": "Exact exporter and original per-frame RGB capture time unavailable in inspected assets; no causal Robot eligibility or exposure-sync claim.",
        "rows": rows,
        "code": artifact_ref(Path(__file__)),
    }
    path = output / "RGB_PRODUCER_EVIDENCE_AUDIT.json"
    atomic_json(path, value)
    atomic_json(output / "RESULT.json", {
        "task_id": "research_rc1_rgb_producer_evidence",
        "status": value["status"],
        "created_at": value["created_at"],
        "audit": artifact_ref(path),
        "session_ids": sorted(EXPECTED),
        "authority_promoted": False,
        "claim_limit": value["claim_limit"],
    })
    print(json.dumps({"status": value["status"], "sessions": sorted(EXPECTED), "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
