#!/usr/bin/env python3
"""Validate and close the two-session OFFLINE visual delivery (no authority promotion).

Usage from the Chaoyang repository root::

    python -m tools.finalize_two_task_visual_baseline_v1

The command is idempotent for identical bytes and refuses to overwrite a
different final receipt. It does not mutate generated CURRENT_* files.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import hashlib
import json
from pathlib import Path
import subprocess
import sys

from chaoyang.ops.two_task_visual_baseline_v1 import ROOT, validate


RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1"
ROUTE = RUN / "route/attempt_0001/BASELINE_ROUTE_V1.json"
VIDEO = RUN / "visuals/attempt_0002/VIDEO_RECEIPT.json"
FRAME_MAP = RUN / "visuals/attempt_0002/FRAME_MAP_RECEIPT.json"
SHALLOW = ROOT / "docs/current/visuals/TWO_TASK_BASELINE_20260917"
ARCHIVE = RUN / "archive/attempts/attempt_0003_final/RESULT.json"
BACKLOG = RUN / "backlog/attempt_0001/RESULT.json"
GUIDE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/docs/stale-linked/docs/reference/pipeline/TWO_TASK_VISUAL_BASELINE_V1_ZH.md"
PACKET = RUN / "TASK_PACKET.json"
OUT = RUN / "final/attempt_0001/RESULT.json"
SESSIONS = {"get_potato_chips_0902_103": 284, "play_cards_0902_042": 171}
CODE = [
    ROOT / "src/chaoyang/ops/two_task_visual_baseline_v1.py",
    ROOT / "src/chaoyang/ops/build_two_task_stage_qa_v1.py",
    ROOT / "src/chaoyang/ops/build_two_task_chips103_contact_hypothesis_v1.py",
    ROOT / "src/chaoyang/ops/build_two_task_poker042_contact_hypothesis_v1.py",
    ROOT / "src/chaoyang/ops/build_two_task_baseline_backlog.py",
    Path(__file__),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"object expected: {path}")
    return result


def verified_ref(value: dict) -> dict:
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    actual = ref(path)
    if actual["sha256"] != value["sha256"]:
        raise ValueError(f"SHA differs: {path}")
    if "bytes" in value and actual["bytes"] != value["bytes"]:
        raise ValueError(f"byte count differs: {path}")
    return actual


def probe_video(path: Path, expected: int) -> dict:
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,avg_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(info.stdout)["streams"][0]
    if int(stream["nb_frames"]) != expected or stream["avg_frame_rate"] != "30/1":
        raise ValueError(f"video count/fps mismatch: {path}")
    subprocess.run(["ffmpeg", "-xerror", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True, capture_output=True)
    return {"decoded_frames": expected, "fps": "30/1", "full_decode": True}


def verify_frame_map(path: Path, expected: int, session: str) -> None:
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            if row.get("session") != session or row.get("frame_index") != index or row.get("source_frame_id") != index:
                raise ValueError(f"frame map mismatch: {path}:{index}")
            if row.get("raw_to_robot_output_frame_index") != index or row.get("clean_comparison_output_frame_index") != index:
                raise ValueError(f"output map mismatch: {path}:{index}")
            if any(source_index != index for source_index in row.get("stage_source_video_frame_index", {}).values()):
                raise ValueError(f"source index mismatch: {path}:{index}")
    if index + 1 != expected:
        raise ValueError(f"frame map length mismatch: {path}")


def main() -> None:
    route, frozen = validate(ROUTE)
    if route["output_contract"]["control_ground_truth"] is not False or route["output_contract"]["training_eligible"] is not False:
        raise ValueError("offline contract changed")
    video = read(VIDEO)
    fmap = read(FRAME_MAP)
    if video.get("source_freeze", {}).get("sha256") != sha(RUN / "route/attempt_0001/SOURCE_FREEZE.json"):
        raise ValueError("video source freeze differs")
    if fmap.get("video_receipt_sha256") != sha(VIDEO):
        raise ValueError("frame-map receipt not bound to video receipt")
    backlog = read(BACKLOG)
    if backlog.get("pending_count") != 154:
        raise ValueError("backlog not 154")
    verified_ref(backlog["source_manifest"])
    verified_ref(backlog["csv"])
    archive = read(ARCHIVE)
    if archive.get("candidate_count") != 1675 or archive.get("moved_count") != 0 or archive.get("archivable_count_proven") != 0:
        raise ValueError("archive terminal receipt differs from reviewed audit")
    verified_ref(archive["archive_audit"])
    gov = subprocess.run([sys.executable, "-m", "chaoyang.governance.validate_governance_state"], cwd=ROOT, check=True, capture_output=True, text=True)
    gov_result = json.loads(gov.stdout)
    if gov_result.get("status") != "PASS":
        raise ValueError("governance did not pass")
    sessions = {}
    for session, expected in SESSIONS.items():
        qa_path = RUN / ("stage_qa/chips103/attempt_0001/STAGE_QA.json" if session.startswith("get_") else "stage_qa/poker042/attempt_0001/STAGE_QA.json")
        qa = read(qa_path)
        if qa.get("route_id") != route["route_id"] or qa.get("frame_count") != expected:
            raise ValueError(f"stage QA differs: {session}")
        verified_ref(qa["source_freeze"])
        verified_ref(qa["frame_flags"])
        verified_ref(qa["stages"]["contact"]["result"])
        if qa.get("training_eligible") is not False or qa.get("control_ground_truth") is not False:
            raise ValueError(f"stage authority escalation: {session}")
        if qa["stages"]["visible_surface_compositor"]["source_state"] != "NOT_PRODUCED_BY_STAGE_QA":
            raise ValueError("unexpected compositor claim")
        item = video["sessions"][session]
        if item.get("training_eligible") is not False or item.get("control_ground_truth") is not False:
            raise ValueError("video authority escalation")
        if item.get("new_clean_candidate") != "UNKNOWN":
            raise ValueError("unregistered Clean candidate")
        outputs = item["outputs"]
        if len(outputs) != 2:
            raise ValueError(f"expected two planned review videos: {session}")
        out_refs = []
        for out in outputs:
            deep = verified_ref(out)
            if out.get("decoded_frames") != expected or out.get("fps") != "30/1" or out.get("ffmpeg_xerror_full_decode") is not True:
                raise ValueError(f"incomplete video receipt: {session}")
            shallow = SHALLOW / Path(out["path"]).name
            shallow_ref = ref(shallow)
            if shallow_ref["sha256"] != deep["sha256"]:
                raise ValueError(f"shallow video copy differs: {shallow}")
            probe_video(shallow, expected)
            out_refs.append({"deep": deep, "shallow": shallow_ref, "frames": expected, "fps": 30})
        map_item = fmap["frame_maps"][session]
        map_path = ROOT / map_item["path"]
        verified_ref(map_item)
        if map_item["rows"] != expected:
            raise ValueError("frame map row count differs")
        verify_frame_map(map_path, expected, session)
        sessions[session] = {
            "task": frozen["sessions"][session]["task"],
            "frame_count": expected,
            "stage_qa": ref(qa_path),
            "contact_result": verified_ref(qa["stages"]["contact"]["result"]),
            "frame_map": ref(map_path),
            "videos": out_refs,
            "compositor_status": "BLOCKED_PREREQ_NOT_PRODUCED",
            "optimization_status": "OPTIMIZATION_UNPROVEN",
            "training_eligible": False,
            "control_ground_truth": False,
        }
    output = {
        "schema_version": "two-task-visual-baseline-final-v1",
        "route_id": route["route_id"],
        "route_version": route["route_version"],
        "status": "TWO_TASK_FULL_VISUAL_REVIEW_DELIVERED_WITH_BLOCKED_COMPOSITOR",
        "optimized_full_pipeline_pass": False,
        "optimization_status": "OPTIMIZATION_UNPROVEN",
        "route_manifest": ref(ROUTE),
        "source_freeze": ref(RUN / "route/attempt_0001/SOURCE_FREEZE.json"),
        "task_packet": ref(PACKET),
        "guide": ref(GUIDE),
        "video_receipt": ref(VIDEO),
        "frame_map_receipt": ref(FRAME_MAP),
        "backlog": ref(BACKLOG),
        "archive_audit": ref(ARCHIVE),
        "archive_status": archive.get("status"),
        "archive_moved_count": archive.get("moved_count", 0),
        "archive_decision_counts": archive.get("hold_counts", {}),
        "code": [ref(path) for path in CODE],
        "sessions": sessions,
        "governance_validation": gov_result,
        "rc1_authority_changed": False,
        "formal_four_checkpoints_trained": False,
        "claim_limit": "Two full-session offline visual reviews with exact frame mapping. No new Clean/occlusion compositor successor, no causal training data, physical contact truth or real Robot action.",
    }
    data = (json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        if OUT.read_bytes() != data:
            raise FileExistsError(f"immutable final differs: {OUT}")
    else:
        with OUT.open("xb") as f:
            f.write(data)
    print(OUT)


if __name__ == "__main__":
    main()
