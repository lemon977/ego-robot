#!/usr/bin/env python3
"""Build the frozen Wave0 Clean/Robot/Depth/Object6D/appearance intersection.

This is a read-only evidence join.  It does not infer success from directory
names and does not promote any authority.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def digest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def load_ref(ref: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(ref["path"]).resolve(strict=True)
    actual = digest(path)
    for field in ("bytes", "sha256"):
        if field in ref and ref[field] != actual[field]:
            raise ValueError(f"reference mismatch for {path}: {field}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_map(authority: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage = next(row for row in authority["stages"] if row.get("stage") == "Clean")
    found: dict[str, dict[str, Any]] = {}
    for ref in stage["evidence"]:
        path = Path(ref["path"])
        if path.name != "RESULT.json" or not path.is_file():
            continue
        _, value = load_ref(ref)
        session = value.get("session") or value.get("session_id")
        if session and value.get("status") == "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B":
            found[str(session)] = digest(path)
    return found


def robot_map(results: list[Path]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for result_path in results:
        result_path = result_path.resolve(strict=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for row in result.get("rows", []):
            terminal_path, terminal = load_ref(row["result"])
            session = str(row["session"])
            if terminal.get("session") != session:
                raise ValueError(f"Robot terminal identity mismatch: {session}")
            found[session] = {
                "terminal_status": terminal.get("terminal_status"),
                "hard_geometry_pass": bool(terminal.get("hard_geometry_pass", False)),
                "robot_tier": terminal.get("robot_tier"),
                "result": digest(terminal_path),
                "review_result": terminal.get("evidence", {}).get("review_result"),
            }
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--authority-index", type=Path, required=True)
    parser.add_argument("--robot-batch-result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    selection = json.loads(args.selection.resolve(strict=True).read_text(encoding="utf-8"))
    authority = json.loads(args.authority_index.resolve(strict=True).read_text(encoding="utf-8"))
    clean = clean_map(authority)
    robots = robot_map(args.robot_batch_result)
    rows: list[dict[str, Any]] = []
    for selected in selection["sessions"]:
        session = str(selected["session_id"])
        upstream = selected["upstream"]
        task_object_path, task_object = load_ref(upstream["task_object_mask"])
        depth_path, depth = load_ref(upstream["depth"])
        object6d_path, object6d = load_ref(upstream["object6d"])
        raw_dir = Path(selected["raw_path"]).resolve(strict=True)
        raw_video = raw_dir / f"CameraRecord_{session}.mp4"
        existing_clean = selected.get("existing_clean")
        if existing_clean is not None:
            clean_path, existing_clean_value = load_ref(existing_clean)
            existing_status = str(existing_clean_value.get("status", ""))
            if "PASS" not in existing_status or existing_clean_value.get("session") != session:
                raise ValueError(f"frozen existing Clean is not a passed same-session result: {session}")
            clean_ref = digest(clean_path)
        else:
            clean_ref = clean.get(session)
        robot = robots.get(session)
        visible_object = (
            task_object.get("downstream_authorized") is True
            and int(task_object.get("instance_count", 0)) >= 1
            and sum(int(v) for v in task_object.get("observed_counts", {}).values()) > 0
        )
        geometry = (
            depth.get("status") == "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION"
            and object6d.get("unobserved_pose_policy") == "KEEP_INVALID"
            and object6d.get("downstream_authorized") is True
        )
        robot_pass = bool(robot and robot["terminal_status"] == "PASSED" and robot["hard_geometry_pass"])
        pre_zbuffer_ready = bool(clean_ref and robot_pass and geometry and visible_object and raw_video.is_file())
        blocker = None
        if not clean_ref:
            blocker = "CLEAN_NOT_IN_CURRENT_58"
        elif not robot:
            blocker = "ROBOT_NOT_IN_V77_PROCESSED_21"
        elif not robot_pass:
            blocker = "ROBOT_HARD_GEOMETRY_NOT_PASSED"
        elif not geometry:
            blocker = "DEPTH_OR_OBJECT6D_GATE_FAILED"
        elif not visible_object:
            blocker = "NO_DIRECT_VISIBLE_OBJECT_APPEARANCE"
        elif not raw_video.is_file():
            blocker = "RAW_VIDEO_MISSING"
        else:
            blocker = "READY_FOR_FULL_ZBUFFER_EXPORT"
        rows.append({
            "position": int(selected["position"]),
            "task": selected["task"],
            "session": session,
            "frame_count": int(selected["frame_count"]),
            "clean_ready": clean_ref is not None,
            "robot_processed_v77": robot is not None,
            "robot_hard_geometry_pass": robot_pass,
            "depth_ready": depth.get("status") == "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION",
            "object6d_direct_observed_only": geometry,
            "legal_object_appearance": "RAW_VISIBLE_OBJECT_PIXELS_ONLY" if visible_object else "NONE",
            "hidden_object_appearance": "UNAVAILABLE",
            "pre_zbuffer_occlusion_ready": pre_zbuffer_ready,
            "blocker_or_next_gate": blocker,
            "clean_result": clean_ref,
            "robot_result": robot["result"] if robot else None,
            "depth_result": digest(depth_path),
            "object6d_result": digest(object6d_path),
            "task_object_result": digest(task_object_path),
            "raw_video": digest(raw_video) if raw_video.is_file() else None,
        })

    counts = {
        "wave0_sessions": len(rows),
        "clean_ready": sum(row["clean_ready"] for row in rows),
        "robot_processed_v77": sum(row["robot_processed_v77"] for row in rows),
        "robot_hard_geometry_pass": sum(row["robot_hard_geometry_pass"] for row in rows),
        "pre_zbuffer_occlusion_ready": sum(row["pre_zbuffer_occlusion_ready"] for row in rows),
        "poker_pre_zbuffer_ready": sum(row["pre_zbuffer_occlusion_ready"] and row["task"] == "poker" for row in rows),
    }
    matrix = {
        "schema_version": "R22_OCCLUSION_INTERSECTION_MATRIX_V1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": counts,
        "rows": rows,
        "authority_promoted": False,
        "claim_limit": "Evidence intersection only; pre-zbuffer readiness is not Occlusion authority.",
    }
    write_json(output / "INTERSECTION_MATRIX.json", matrix)
    csv_fields = [
        "position", "task", "session", "frame_count", "clean_ready", "robot_processed_v77",
        "robot_hard_geometry_pass", "depth_ready", "object6d_direct_observed_only",
        "legal_object_appearance", "hidden_object_appearance", "pre_zbuffer_occlusion_ready",
        "blocker_or_next_gate",
    ]
    with (output / "INTERSECTION_MATRIX.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})
    write_json(output / "METRICS.json", {"schema_version": "R22_OCCLUSION_INTERSECTION_METRICS_V1", **counts})
    ready_poker = [row["session"] for row in rows if row["task"] == "poker" and row["pre_zbuffer_occlusion_ready"]]
    (output / "DECISION.md").write_text(
        "# R2.2 Occlusion交集决定\n\n"
        f"冻结Wave0共{len(rows)}条；v77已处理{counts['robot_processed_v77']}条，硬几何通过"
        f"{counts['robot_hard_geometry_pass']}条。完整Robot z-buffer仍需逐会话导出。"
        f"Poker候选：{', '.join(ready_poker) if ready_poker else '无'}。\n\n"
        "合法外观仅指当前Raw中由任务物体Mask直接观测到的像素；隐藏牌面仍为UNAVAILABLE，"
        "不得用Clean背景或未来帧伪造。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "R22_OCCLUSION_INTERSECTION_NEXT_V1",
        "status": "PASSED",
        "next_task_id": "POKER_FULL_ZBUFFER_AND_BASIC_OCCLUSION" if ready_poker else "BLOCK_NO_POKER_INTERSECTION",
        "preferred_session": ready_poker[0] if ready_poker else None,
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "R22_OCCLUSION_INTERSECTION_RECEIPT_V1",
        "selection": digest(args.selection),
        "authority_index": digest(args.authority_index),
        "robot_batch_results": [digest(path) for path in args.robot_batch_result],
        "producer": digest(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "authority_promoted": False,
    })
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "R22_OCCLUSION_INTERSECTION_ARTIFACTS_V1",
        "artifacts": [
            digest(output / "INTERSECTION_MATRIX.json"), digest(output / "INTERSECTION_MATRIX.csv"),
            digest(output / "METRICS.json"), digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json"),
        ],
    })
    write_json(output / "RESULT.json", {
        "schema_version": "R22_OCCLUSION_INTERSECTION_RESULT_V1",
        "terminal_status": "PASSED",
        "status": "PASS_DEVELOPMENT_EVIDENCE_INTERSECTION",
        "counts": counts,
        "preferred_poker_session": ready_poker[0] if ready_poker else None,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    print(json.dumps({"status": "PASSED", "counts": counts, "poker": ready_poker}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
