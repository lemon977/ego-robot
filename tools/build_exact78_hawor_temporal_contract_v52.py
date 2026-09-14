from __future__ import annotations

"""从冻结 Robot 输入合同生成 HaWoR temporal successor 合同。

中文用法：

  PYTHONPATH=. python3 tools/build_exact78_hawor_temporal_contract_v52.py \
    --robot-contract tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/ROBOT_READY_BATCH_019_INPUT_V1.json \
    --output tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/HAWOR_TEMPORAL_BATCH_019_CONTRACT_V1.json

脚本只投影已经冻结并复核过的 bounded NPZ 与源视频引用，不重新选数据，不覆盖已有
合同，也不启动 HaWoR。正式运行 temporal 时必须显式使用已验证的
`/root/.venvs/wilor/bin/python`；系统 Python 缺少 smplx。
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ref(ref: dict[str, Any], label: str) -> Path:
    if not isinstance(ref, dict) or not {"path", "bytes", "sha256"}.issubset(ref):
        raise ValueError(f"{label}: invalid artifact reference")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
        raise ValueError(f"{label}: bytes/SHA mismatch for {path}")
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable contract: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
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
    ap.add_argument("--robot-contract", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    source = Path(args.robot_contract)
    if not source.is_absolute():
        source = ROOT / source
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    robot = load_json(source)
    if robot.get("schema_version") != "exact78-robot-ready-input-v52-v1":
        raise ValueError("unexpected Robot input contract schema")
    if robot.get("status") != "FROZEN_PREFLIGHT_PENDING":
        raise ValueError("Robot input contract is not frozen")
    sessions = []
    seen = set()
    for row in robot.get("sessions", []):
        session = row.get("session")
        task = row.get("task")
        if not isinstance(session, str) or task not in {"chips", "poker"} or session in seen:
            raise ValueError(f"invalid or duplicate session identity: {session}")
        seen.add(session)
        bounded = verify_ref(row.get("bounded_npz"), f"{session}.bounded_npz")
        video = verify_ref(row.get("source_video"), f"{session}.source_video")
        sessions.append({
            "task": task,
            "session": session,
            "bounded_npz": str(bounded),
            "bounded_sha256": row["bounded_npz"]["sha256"],
            "source_video": str(video),
            "source_video_sha256": row["source_video"]["sha256"],
        })
    if not sessions:
        raise ValueError("Robot input contract contains no sessions")
    result = {
        "schema_version": "exact78-ready-batch-hawor-temporal-successor-contract-v1",
        "sessions": sessions,
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        atomic_json(output, result)
        print(json.dumps({
            "status": "PASS_TEMPORAL_CONTRACT_FROZEN",
            "sessions": sorted(seen),
            "output": {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
