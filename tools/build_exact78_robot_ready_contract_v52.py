from __future__ import annotations

"""从 current matrix 生成不可覆盖的 exact78 Robot 输入合同。

中文用法：

  # 为刚通过 Clean 的一条或多条会话冻结 Robot 输入
  PYTHONPATH=. python3 tools/build_exact78_robot_ready_contract_v52.py \
    --matrix tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json \
    --sessions get_potato_chips_0902_019 \
    --output tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/ROBOT_READY_BATCH_019_INPUT_V1.json

  # 只打印并校验，不写文件
  PYTHONPATH=. python3 tools/build_exact78_robot_ready_contract_v52.py ... --dry-run

默认会扫描 lane_c_contact_robot 下已有 ROBOT_READY_*_INPUT*.json；同一 session
已被冻结时 fail closed，避免重复求解。只有人工审计确认是同一合同的重建时才可加
--allow-existing-contract。该脚本只生成输入合同，不发布 Robot/contact/action authority。
后续 preflight 的 `--matrix` 必须传合同内 `matrix_snapshot.path` 指向的不可变文件，
不能再传会继续刷新的 CURRENT_DRAFT 路径。
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LANE_C_ROOT = ROOT / "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot"

TEMPLATES = {
    "chips": {
        "states": ROOT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/WORLD_FIRST_SAME_SIDE_48FRAME_STATES.npz",
        "hawor": ROOT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/get_potato_chips_0902_034/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
    },
    "poker": {
        "states": ROOT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_same_side_world_temporal48_v3/WORLD_FIRST_SAME_SIDE_48FRAME_STATES.npz",
        "hawor": ROOT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1/play_cards_0902_042/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_ref(ref: dict[str, Any], label: str) -> Path:
    required = {"path", "bytes", "sha256"}
    if not isinstance(ref, dict) or not required.issubset(ref):
        raise ValueError(f"{label}: missing artifact reference fields")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = ROOT / path
    actual = artifact_ref(path)
    if actual["bytes"] != ref["bytes"] or actual["sha256"] != ref["sha256"]:
        raise ValueError(f"{label}: bytes/SHA mismatch for {path}")
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def existing_contract_sessions(output: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(LANE_C_ROOT.glob("ROBOT_READY_*_INPUT*.json")):
        if path.resolve() == output.resolve():
            continue
        try:
            obj = load_json(path)
        except Exception:
            continue
        for row in obj.get("sessions", []):
            if isinstance(row, dict) and isinstance(row.get("session"), str):
                found.setdefault(row["session"], []).append(str(path))
    return found


def session_contract(row: dict[str, Any]) -> dict[str, Any]:
    session = row["session_id"]
    task = row["task"]
    if row.get("clean_state") != "PASSED_GRADE_B":
        raise ValueError(f"{session}: clean_state is not PASSED_GRADE_B")
    if row.get("robot_current_state") != "READY_FOR_ROBOT_CURRENT_DRAFT":
        raise ValueError(f"{session}: robot_current_state is {row.get('robot_current_state')}")
    if row.get("final_robot_terminal_proven"):
        raise ValueError(f"{session}: immutable Robot terminal already exists")

    clean_ref = row.get("clean_result")
    verify_ref(clean_ref, f"{session}.clean_result")
    bounded_ref = row.get("hawor", {}).get("result")
    bounded_path = verify_ref(bounded_ref, f"{session}.bounded_result")
    bounded = load_json(bounded_path)
    if not str(bounded.get("status", "")).startswith("PASS_"):
        raise ValueError(f"{session}: bounded HaWoR is not PASS")
    if bounded.get("session_id") != session or bounded.get("task") != task:
        raise ValueError(f"{session}: bounded HaWoR identity mismatch")
    bounded_npz = bounded.get("outputs", {}).get("npz")
    source_video = bounded.get("inputs", {}).get("source_video")
    verify_ref(bounded_npz, f"{session}.bounded_npz")
    verify_ref(source_video, f"{session}.source_video")
    frame_count = bounded.get("validation", {}).get("full_video", {}).get("input_frames")
    if int(frame_count) != int(row["frame_count"]):
        raise ValueError(f"{session}: frame count mismatch matrix={row['frame_count']} bounded={frame_count}")
    return {
        "task": task,
        "session": session,
        "frame_count": int(row["frame_count"]),
        "split": row["split"],
        "clean_result": clean_ref,
        "bounded_result": bounded_ref,
        "bounded_npz": bounded_npz,
        "source_video": source_video,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable contract: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-existing-contract", action="store_true")
    args = ap.parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = ROOT / matrix_path
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    matrix = load_json(matrix_path)
    if matrix.get("status") != "CURRENT_DRAFT_NOT_FINAL_AUTHORITY":
        raise ValueError("matrix is not the current exact78 draft schema")
    matrix_live_ref = artifact_ref(matrix_path)
    matrix_snapshot = (
        matrix_path.parent / "snapshots" /
        f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{matrix_live_ref['sha256']}.json"
    )
    if not matrix_snapshot.is_file() or matrix_snapshot.read_bytes() != matrix_path.read_bytes():
        raise ValueError("current matrix has no matching immutable content-addressed snapshot; rebuild it first")
    rows = {r["session_id"]: r for r in matrix.get("rows", [])}
    requested = args.sessions
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate --sessions values")
    missing = [s for s in requested if s not in rows]
    if missing:
        raise ValueError(f"sessions absent from matrix: {missing}")
    existing = existing_contract_sessions(output)
    conflicts = {s: existing[s] for s in requested if s in existing}
    if conflicts and not args.allow_existing_contract:
        raise ValueError(f"sessions already frozen in Robot input contracts: {conflicts}")

    sessions = [session_contract(rows[s]) for s in requested]
    templates = {
        task: {name: artifact_ref(path) for name, path in values.items()}
        for task, values in TEMPLATES.items()
    }
    contract = {
        "schema_version": "exact78-robot-ready-input-v52-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "FROZEN_PREFLIGHT_PENDING",
        "matrix_snapshot": artifact_ref(matrix_snapshot),
        "placement_policy": {
            "mode": "per_session_same_algorithm",
            "regularization_prior_m": 0.26,
            "chips_candidates_m": [0.2, 0.24, 0.258, 0.2595, 0.26, 0.28, 0.3],
            "poker_candidates_m": [0.24, 0.258, 0.26, 0.28],
            "fixed_for_full_session": True,
            "target_hand_roots_unchanged": True,
            "selection_order": [
                "strict_numeric_pass", "failed_rows", "normalized_gate_excess",
                "distance_to_0.26m_prior", "backoff_m",
            ],
        },
        "method_budget": {
            "arm_round_1": "forward full trajectory with frozen placement candidate selection",
            "arm_round_2": "segment-local bidirectional lookahead",
            "hand_round_1": "forward full trajectory with independent thumb q[0:6]",
            "hand_round_2": "segment-local bidirectional lookahead",
            "max_new_method_rounds": 2,
        },
        "sessions": sessions,
        "accepted_templates": templates,
        "claim_limit": (
            "Frozen newly Clean-joined numeric Robot inputs only; this contract is not "
            "Robot/contact/action/deployment authority and does not authorize another GPU owner."
        ),
    }
    if args.dry_run:
        print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        atomic_write_json(output, contract)
        print(json.dumps({"status": "PASS_CONTRACT_FROZEN", "sessions": requested, "output": artifact_ref(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
