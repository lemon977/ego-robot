from __future__ import annotations

"""核验 exact78 HaWoR temporal batch，并发布不可覆盖的审计收据。

中文用法：

  PYTHONPATH=. python3 tools/audit_exact78_hawor_temporal_batch_v52.py \
    --preflight .../ready_batch_019_preflight_v1/RESULT.json \
    --contract .../HAWOR_TEMPORAL_BATCH_019_CONTRACT_V1.json \
    --aggregate .../hawor_temporal_candidates_v1/batch_019_v1/RESULT.json \
    --output .../hawor_temporal_candidates_v1/batch_019_v1/AUDIT.json

若此前有运行失败，可重复传入 `--failed-attempt <ATTEMPT.json>`。审计会复算所有
path/bytes/SHA，完整执行 ffmpeg -xerror，核对全片/48帧视频和 frame manifest；
不修改 temporal 数值，也不授予 Robot/contact/action authority。
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def verify_ref(ref: dict[str, Any], label: str) -> Path:
    if not isinstance(ref, dict) or not {"path", "bytes", "sha256"}.issubset(ref):
        raise ValueError(f"{label}: invalid artifact reference")
    path = resolve(ref["path"])
    actual = artifact_ref(path)
    if (actual["bytes"], actual["sha256"]) != (ref["bytes"], ref["sha256"]):
        raise ValueError(f"{label}: bytes/SHA mismatch for {path}")
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def video_probe(path: Path, expected_frames: int) -> dict[str, Any]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path),
    ])
    stream = json.loads(raw)["streams"][0]
    frames = int(stream["nb_read_frames"])
    if frames != expected_frames:
        raise ValueError(f"{path}: decoded frames {frames} != {expected_frames}")
    check = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if check.returncode:
        raise ValueError(f"{path}: ffmpeg -xerror failed: {check.stderr.decode('utf-8', 'replace')[-2000:]}")
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": stream["avg_frame_rate"], "decoded_frames": frames, "ffmpeg_xerror": "PASS",
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--failed-attempt", action="append", default=[])
    args = ap.parse_args()
    preflight_path, contract_path, aggregate_path = map(resolve, [args.preflight, args.contract, args.aggregate])
    output = resolve(args.output)
    preflight, contract, aggregate = map(load_json, [preflight_path, contract_path, aggregate_path])
    if not str(preflight.get("status", "")).startswith("PASS_READY"):
        raise ValueError("preflight is not PASS_READY")
    if contract.get("schema_version") != "exact78-ready-batch-hawor-temporal-successor-contract-v1":
        raise ValueError("unexpected temporal contract schema")
    if aggregate.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise ValueError("temporal aggregate is not numeric PASS")
    expected = {(r["task"], r["session"]): r for r in contract["sessions"]}
    got = {(r["task"], r["session"]): r for r in aggregate["sessions"]}
    if set(expected) != set(got):
        raise ValueError("aggregate session set differs from contract")

    sessions = []
    for identity in sorted(expected):
        row = got[identity]
        result_path = verify_ref(row["result"], f"{identity}.result")
        result = load_json(result_path)
        if result.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
            raise ValueError(f"{identity}: result not numeric PASS")
        if (result.get("task"), result.get("session")) != identity:
            raise ValueError(f"{identity}: result identity mismatch")
        frames = int(result["frame_count"])
        refs = {}
        for key in ["npz", "frame_manifest", "bounded_only_video", "ab48_video"]:
            refs[key] = artifact_ref(verify_ref(result["outputs"][key], f"{identity}.{key}"))
        manifest_rows = sum(1 for _ in refs_path(result["outputs"]["frame_manifest"]).open("r", encoding="utf-8"))
        if manifest_rows != frames:
            raise ValueError(f"{identity}: frame manifest {manifest_rows} != {frames}")
        full_probe = video_probe(refs_path(result["outputs"]["bounded_only_video"]), frames)
        ab_frames = min(48, frames)
        ab_probe = video_probe(refs_path(result["outputs"]["ab48_video"]), ab_frames)
        sessions.append({
            "task": identity[0], "session": identity[1], "frames": frames,
            "selected_alpha": result["selected_alpha"], "numeric_status": result["status"],
            "result": artifact_ref(result_path), "npz": refs["npz"],
            "frame_manifest": refs["frame_manifest"], "frame_manifest_rows": manifest_rows,
            "full_review_video": refs["bounded_only_video"], "full_video_probe": full_probe,
            "ab48_video": refs["ab48_video"], "ab48_video_probe": ab_probe,
        })
    failed = []
    for item in args.failed_attempt:
        path = resolve(item)
        obj = load_json(path)
        if obj.get("status") != "FAILED_RUNTIME" or obj.get("partial_reused") is not False:
            raise ValueError(f"invalid failed-attempt receipt: {path}")
        failed.append(artifact_ref(path))
    program = ROOT / "tools/run_hawor_temporal_jerk_successor.py"
    auditor = Path(__file__).resolve()
    receipt = {
        "schema_version": "exact78-hawor-temporal-batch-audit-v52-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "PASS_NUMERIC_AND_FULL_VIDEO_DECODE_NEEDS_HUMAN_REVIEW",
        "runtime": {"python_executable": "/root/.venvs/wilor/bin/python", "cuda_visible_devices": "", "gpu_calls": 0},
        "attempts": {"failed_runtime": failed, "passing_attempt_partial_reused": False},
        "preflight": artifact_ref(preflight_path), "contract": artifact_ref(contract_path),
        "program": artifact_ref(program), "auditor": artifact_ref(auditor),
        "aggregate": artifact_ref(aggregate_path), "sessions": sessions,
        "claim_limit": "Temporal smoothing and full-video decode evidence only; human visual approval is still required and no Robot/contact/action/deployment authority is implied.",
    }
    atomic_json(output, receipt)
    print(json.dumps({"status": receipt["status"], "sessions": len(sessions), "output": artifact_ref(output)}))
    return 0


def refs_path(ref: dict[str, Any]) -> Path:
    return resolve(ref["path"]).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
