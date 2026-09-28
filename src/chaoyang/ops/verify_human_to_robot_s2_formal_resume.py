#!/usr/bin/env python3
"""Verify that every formal S2 product is reused by the public CLI unchanged."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUTPUT = ROOT / "formal_entry_resume/attempt_0002"
PRODUCTS = {
    "get_potato_chips_0915_007": ROOT / "lanes/motion_product/formal_product_007/attempt_0002",
    "play_cards_0915_031": ROOT / "lanes/motion_product/formal_product_031/attempt_0004",
    "get_potato_chips_0902_103": ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002",
    "play_cards_0902_042": ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(REPO_ROOT)
    return resolved


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"IMMUTABLE_RESUME_EVIDENCE_EXISTS:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    runtime = REPO_ROOT / ".tmp/s2/formal_resume"
    runtime.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for session_id, product_root in PRODUCTS.items():
        result_path = product_root / "PRODUCT_RESULT.json"
        before_result = artifact_ref(result_path)
        result = load_json(result_path)
        if result.get("session_id") != session_id:
            raise RuntimeError(f"SESSION_MISMATCH:{session_id}")
        if result.get("execution") != "EXECUTED" or result.get("structure") != "PASS":
            raise RuntimeError(f"PRODUCT_NOT_STRUCTURAL:{session_id}")
        video_path = _inside_repo(Path(result["product_video"]["path"]))
        before_video = artifact_ref(video_path)
        binding_path = _inside_repo(Path(result["binding"]["path"]))
        domain_path = _inside_repo(Path(result["domain_manifest"]["path"]))
        domain = load_json(domain_path)
        input_path = Path(domain["frames"][0]["rgb"]).resolve(strict=True).parent
        # The public resolver accepts either the session directory or a file
        # whose parent is the session directory.  Walk only to that named root;
        # inputs remain read-only and may live outside the repository.
        while input_path.name != session_id and input_path != input_path.parent:
            input_path = input_path.parent
        if input_path.name != session_id:
            raise RuntimeError(f"SESSION_INPUT_ROOT_NOT_FOUND:{session_id}")
        command = [
            sys.executable,
            "-m",
            "chaoyang.ops.run_human_to_robot_baseline_v1",
            "--input",
            str(input_path),
            "--motion-source",
            "hawor",
            "--output",
            str(product_root),
            "--config",
            str(binding_path),
            "--resume",
        ]
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(REPO_ROOT / "src"),
            TMPDIR=str(runtime),
            XDG_CACHE_HOME=str(REPO_ROOT / ".tmp/s2/xdg"),
            PYTHONPYCACHEPREFIX=str(REPO_ROOT / ".tmp/s2/cache"),
        )
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"FORMAL_RESUME_FAILED:{session_id}:{process.returncode}:{process.stderr[-1000:]}"
            )
        try:
            payload = json.loads(process.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"FORMAL_RESUME_OUTPUT_INVALID:{session_id}") from exc
        if payload.get("status") != "REUSED":
            raise RuntimeError(f"FORMAL_RESUME_NOT_REUSED:{session_id}:{payload}")
        after_result = artifact_ref(result_path)
        after_video = artifact_ref(video_path)
        if before_result != after_result or before_video != after_video:
            raise RuntimeError(f"FORMAL_RESUME_MUTATED_PRODUCT:{session_id}")
        rows.append(
            {
                "session_id": session_id,
                "status": "REUSED",
                "exit_code": process.returncode,
                "command": command,
                "binding": artifact_ref(binding_path),
                "product_result_before_after": before_result,
                "product_video_before_after": before_video,
                "stdout_last_json": payload,
                "model_or_solver_reexecution": False,
            }
        )
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_FORMAL_ENTRY_RESUME_V1",
        "task_id": TASK,
        "created_at": _now(),
        "status": "PASS",
        "formal_cli": "chaoyang.ops.run_human_to_robot_baseline_v1",
        "sessions": rows,
        "reused": len(rows),
        "mutated": 0,
        "base_model_reexecutions": 0,
        "claim_limit": (
            "Same-signature formal-entry cache reuse only; this does not promote product quality, "
            "adoption, training, metric, control, or deployment authority."
        ),
    }
    atomic_json(OUTPUT / "RESULT.json", result)
    print(json.dumps({"status": "PASS", "reused": len(rows), "result": artifact_ref(OUTPUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
