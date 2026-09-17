"""Add verified offline diagnostic videos without changing formal Robot30 terminals."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def ref(path: Path) -> dict:
    blob = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def check_ref(item: dict) -> None:
    path = Path(item["path"])
    observed = ref(path)
    if observed["bytes"] != item["bytes"] or observed["sha256"] != item["sha256"]:
        raise ValueError(f"Artifact closure mismatch: {path}")


def probe_video(item: dict, expected_frames: int) -> dict:
    check_ref(item)
    path = item["path"]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json", path],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_frames"]) != expected_frames or stream["r_frame_rate"] != "30/1":
        raise ValueError(f"Video length/fps mismatch: {path}")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", path, "-f", "null", "-"], check=True, capture_output=True)
    return {"frames": expected_frames, "fps": "30/1", "full_decode": "PASS_FFMPEG_XERROR"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--chips-result", type=Path, required=True)
    parser.add_argument("--poker-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_matrix.read_text())
    if len(base["rows"]) != 60:
        raise ValueError("Frozen Robot30 matrix is not 60 rows")
    rows = base["rows"]
    keys = [(row["task"], row["session_id"]) for row in rows]
    if len(set(keys)) != 60 or sum(task == "chips" for task, _ in keys) != 30 or sum(task == "poker" for task, _ in keys) != 30:
        raise ValueError("Robot30 selection has duplicate or missing task rows")

    additions = (
        (args.chips_result, "chips", "get_potato_chips_0903_182", "full_video", 351),
        (args.poker_result, "poker", "play_cards_0901_001", "watermarked_video", 645),
    )
    for result_path, task, session, video_key, frames in additions:
        result = json.loads(result_path.read_text())
        if result["session_id"] != session or result.get("terminal_status", result.get("status")) != "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO":
            raise ValueError(f"Unexpected successor terminal: {result_path}")
        if result["training_eligible"] or result["input_mode"] != "OFFLINE_VISUAL":
            raise ValueError(f"Successor is not offline-only: {result_path}")
        video = result[video_key]
        qa = probe_video(video, frames)
        row = next(row for row in rows if row["task"] == task and row["session_id"] == session)
        if row.get("verified_video") is not None:
            raise ValueError(f"Refusing to replace an existing closed video: {session}")
        row["research_successor"] = {
            "terminal_status": "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO",
            "input_mode": "OFFLINE_VISUAL",
            "training_eligible": False,
            "hard_geometry_pass": False,
            "result": ref(result_path),
            "video": video,
            "video_qa": qa,
            "claim_limit": "Full diagnostic video only; no authority or training eligibility.",
        }

    counts = {}
    csv_rows = []
    for task in ("chips", "poker"):
        subset = [row for row in rows if row["task"] == task]
        base_count = sum(row.get("verified_video") is not None for row in subset)
        successor_count = sum("research_successor" in row for row in subset)
        counts[task] = {"selected": 30, "prior_closed_video": base_count, "new_c_diagnostic_video": successor_count, "all_watchable_full_video": base_count + successor_count, "missing_full_video": 30 - base_count - successor_count, "prior_hard_geometry_pass": sum(row.get("robot30_hard_geometry_pass") is True for row in subset)}
        for row in subset:
            research = row.get("research_successor")
            video = research["video"] if research else row.get("verified_video")
            csv_rows.append({"task": task, "rank": row["rank"], "session_id": row["session_id"], "full_video_status": "NEW_C_DIAGNOSTIC" if research else ("PRIOR_VERIFIED" if video else "MISSING"), "video_path": video["path"] if video else "", "video_sha256": video["sha256"] if video else "", "robot30_terminal_status_unchanged": row["robot30_terminal_status"], "research_training_eligible": "false" if research else "UNMODIFIED"})

    output = {
        "schema_version": "rc1-robot30-research-delivery-matrix-rev1",
        "mode": "OFFLINE_VISUAL_RESEARCH",
        "authority_promoted": False,
        "base_matrix": ref(args.base_matrix),
        "generator": ref(Path(__file__)),
        "successor_results": [ref(args.chips_result), ref(args.poker_result)],
        "counts": counts,
        "rows": rows,
        "claim_limit": "Exactly 60 selected rows. New full videos are C-grade diagnostics only; existing formal terminal and hard-geometry fields are unchanged. No causal or training claim.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0001.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0001.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
