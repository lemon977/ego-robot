"""Verify, index and terminalize the governed Robot100 development-video batch."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)


TASK_ID = "robot100_batch_baseline_20260928"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SELECTION = ATTEMPT / "SELECTION.json"
CURRENT_DOC = REPO_ROOT / "docs/current/ROBOT100_BATCH_ZH.md"
ACTIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19])


def _write_outputs() -> tuple[dict, dict]:
    if (ATTEMPT / "RESULT.json").exists():
        raise RuntimeError("FINAL_RESULT_ALREADY_EXISTS")
    selection = load_json(SELECTION)
    if selection.get("count") != 100 or len(selection.get("rows", [])) != 100:
        raise RuntimeError("SELECTION_NOT_100")
    rows = []
    for selected in selection["rows"]:
        session_id = selected["session_id"]
        result_path = ATTEMPT / "sessions" / session_id / "RESULT.json"
        if not result_path.is_file():
            raise RuntimeError(f"MISSING_SESSION_RESULT:{session_id}")
        result = load_json(result_path)
        if (
            result.get("schema_version") != "CHAOYANG_ROBOT100_SESSION_RESULT_V1"
            or result.get("session_id") != session_id
            or result.get("status") != "COMPLETE_DEVELOPMENT_VISUAL_BASELINE"
        ):
            raise RuntimeError(f"SESSION_RESULT_IDENTITY_OR_STATUS:{session_id}")
        video = result["video"]
        errors = validate_artifact_ref(video)
        if errors:
            raise RuntimeError(f"VIDEO_REF_INVALID:{session_id}:{errors}")
        if result["review"]["decoded_frames"] != result["review"]["frames"]:
            raise RuntimeError(f"VIDEO_DECODE_MISMATCH:{session_id}")
        if result["review"]["frames"] != selected["frames"]:
            raise RuntimeError(f"SELECTION_FRAME_MISMATCH:{session_id}")
        rows.append({
            "index": len(rows) + 1,
            "session_id": session_id,
            "task": selected["task"],
            "frames": selected["frames"],
            "fps": selected["fps"],
            "video": video,
            "session_result": artifact_ref(result_path),
            "arm_tolerance_pass_physical_left_right": result["arm_tolerance_pass_physical_left_right"],
            "human_review": result["human_review"],
            "quality_adopted": result["algorithm_quality_adopted"],
        })
    ids = [row["session_id"] for row in rows]
    hashes = [row["video"]["sha256"] for row in rows]
    if len(set(ids)) != 100:
        raise RuntimeError("SESSION_IDS_NOT_100_UNIQUE")
    if len(set(hashes)) != 100:
        raise RuntimeError("VIDEO_CONTENT_NOT_100_UNIQUE")
    index = {
        "schema_version": "CHAOYANG_ROBOT100_VIDEO_INDEX_V1",
        "task_id": TASK_ID,
        "created_at": now_iso(),
        "count": 100,
        "unique_sessions": 100,
        "unique_video_sha256": 100,
        "by_task": {
            "playing_cards": sum(row["task"] == "playing_cards" for row in rows),
            "potato_chips": sum(row["task"] == "potato_chips" for row in rows),
        },
        "rows": rows,
        "layout": "1280x720: source acquisition video at left; fixed-view Tianji+KaiHand+real adapter at right; Chinese provenance/limits below",
        "claim_limit": "100 distinct, fully decoded development visualisations; not physical calibration, contact truth, control truth, training labels, deployment approval, or human quality acceptance.",
    }
    atomic_json(ATTEMPT / "VIDEO_INDEX.json", index)
    result = {
        "schema_version": "CHAOYANG_ROBOT100_BATCH_RESULT_V1",
        "task_id": TASK_ID,
        "status": "COMPLETE_100_DISTINCT_ROBOT_VIDEOS",
        "created_at": now_iso(),
        "selection": artifact_ref(SELECTION),
        "video_index": artifact_ref(ATTEMPT / "VIDEO_INDEX.json"),
        "counts": {
            "requested_minimum": 100,
            "delivered": 100,
            "unique_sessions": 100,
            "fully_decoded": 100,
            "playing_cards": index["by_task"]["playing_cards"],
            "potato_chips": index["by_task"]["potato_chips"],
            "quality_adopted": 0,
            "human_reviewed": 0,
        },
        "gpu_used": False,
        "training_executed": False,
        "source_data_modified": False,
        "processed_data_modified": False,
        "quality": "PENDING_HUMAN_REVIEW_DEVELOPMENT_BASELINE",
        "adoption": "NOT_AUTOMATIC",
        "claim_limit": index["claim_limit"],
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    lines = [
        "# 先看这里：100条标准 Robot 开发基线视频",
        "",
        "本批次已经生成并完整解码 **100 条不同会话** 的 Robot 视频：扑克牌50条、薯片50条。",
        "",
        "- 左侧：同会话采集视频。",
        "- 右侧：同一固定视角下的 Tianji 双臂、KaiHand 双手与真实连接件网格。",
        "- 驱动：0916同会话、同时间轴的 PICO Controller 腕位姿＋MANUS腕局部骨架。",
        "- 固定项：同一资产、同一虚拟安装、同一求解与渲染参数、同一中文布局。",
        "- 边界：这是开发可视化基线，不证明物理标定、接触、控制、训练标签或部署安全。",
        "",
        "## 查看",
        "",
        "机器索引：`VIDEO_INDEX.json`；每条视频与其 `RESULT.json` 位于 `sessions/<session_id>/`。",
        "",
        "## 验证",
        "",
        "100条均绑定源视频/HDF5 bytes+SHA，输出内容SHA互不相同，且逐条完整解码帧数与冻结选择一致。人工效果审阅仍为待完成，不能把批量生成完成写成算法质量采用。",
        "",
    ]
    atomic_write(ATTEMPT / "先看这里.md", "\n".join(lines).encode("utf-8"))
    return result, index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    result, video_index = _write_outputs()
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    matches = [row for row in state["tasks"] if row.get("task_id") == TASK_ID]
    if len(matches) != 1 or matches[0].get("status") not in ACTIVE:
        raise RuntimeError("TASK_NOT_EXACTLY_ONCE_ACTIVE")
    created = now_iso()
    task = matches[0]
    writer_pid = int(task["pid"])
    writer_ticks = int(task["proc_start_ticks"])
    if _ticks(writer_pid) != writer_ticks:
        raise RuntimeError("WRITER_IDENTITY_DRIFT")
    final_ref = artifact_ref(ATTEMPT / "RESULT.json")
    task.update(
        status="PASSED",
        phase="ROBOT100_BATCH_COMPLETE",
        phase_detail="100_DISTINCT_FULLY_DECODED_DEVELOPMENT_ROBOT_VIDEOS",
        updated_at=created,
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        result=final_ref,
        quality="PENDING_HUMAN_REVIEW_DEVELOPMENT_BASELINE",
        adoption="NOT_AUTOMATIC",
        review="NOT_REVIEWED",
        last_attempt_terminal="PASSED",
        last_attempt_reason="ENGINEERING_DELIVERY_COMPLETE_QUALITY_NOT_PROMOTED",
    )
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "attempt": 1,
        "status": "PASSED",
        "created_at": created,
        "message": "Delivered 100 distinct fully decoded Robot development videos; no physical or quality authority promoted.",
        "result": final_ref,
    }])[-100:]
    old_index = load_json(INDEX)
    closed_index = dict(old_index)
    closed_index.update(
        packet_revision="ROBOT100_BATCH_BASELINE_V1_TERMINAL",
        plan_revision="ROBOT100_BATCH_BASELINE_V1_20260928",
        execution_revision="ROBOT100_BATCH_BASELINE_V1_20260928",
        status="PASS_NO_ACTIVE_TASKS",
        supersedes_index=artifact_ref(INDEX),
        task_packets=[],
        claim_limit=result["claim_limit"],
    )
    doc = "\n".join([
        "# Robot100 批量开发基线",
        "",
        f"状态：已完成100条不同会话视频；机器结果：`{ATTEMPT / 'RESULT.json'}`。",
        "",
        "范围：0916 PICO Controller＋MANUS，扑克牌50条、薯片50条。每条使用固定 Tianji＋KaiHand＋真实连接件资产和统一中文布局。",
        "",
        "采用边界：工程批量交付已完成；人工视觉审阅与算法质量采用尚未执行。不得解释为物理标定、接触真值、训练标签、控制真值或部署许可。",
        "",
        f"视频索引：`{ATTEMPT / 'VIDEO_INDEX.json'}`。",
        "",
    ])
    atomic_write(CURRENT_DOC, doc.encode("utf-8"))
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="ROBOT100_BATCH_BASELINE_COMPLETED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=closed_index,
    )
    try:
        os.kill(writer_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    print(json.dumps({
        "status": result["status"],
        "revision": published["governance_revision"],
        "videos": video_index["count"],
        "unique_sessions": video_index["unique_sessions"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
