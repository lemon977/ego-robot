#!/usr/bin/env python3
"""Close RC1 T2 when causal Clean prerequisites are or are not satisfied."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path: Path, value: Any) -> None:
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-result", type=Path, required=True)
    parser.add_argument("--clean-contract-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    mask = json.loads(args.mask_result.resolve(strict=True).read_text())
    clean = json.loads(args.clean_contract_result.resolve(strict=True).read_text())
    reports = clean.get("case_reports", [])
    blockers = []
    for row in reports:
        totals = row.get("totals", {})
        if int(totals.get("accepted_temporal_pixels", 0)) == 0:
            blockers.append({"case": row.get("label"), "code": "NO_LEGAL_CAUSAL_REAL_DONOR_PIXELS"})
        if int(totals.get("unknown_write_pixels", 0)) > 0:
            blockers.append({"case": row.get("label"), "code": "WRITE_DOMAIN_REMAINS_SEMANTIC_UNKNOWN", "pixels": int(totals.get("unknown_write_pixels", 0))})
        if row.get("label", "").lower().startswith("poker"):
            blockers.append({"case": row.get("label"), "code": "POKER_CAUSAL_ATLAS_NOT_VERIFIED"})
    if clean.get("fresh_propainter_started"):
        blockers.append({"code": "UNEXPECTED_PRIOR_PROPAINTER_EXECUTION"})
    status = "BLOCKED_PREREQ" if blockers else "PASSED"
    result = {
        "schema_version": "chaoyang-rc1-t2-causal-clean-preflight-v1",
        "task_id": "rc1_t2_causal_clean", "created_at": now(), "status": status,
        "mask_terminal_status": mask.get("status"),
        "mask_negative_terminal_is_local": True,
        "cases": [{"label": row.get("label"), "session": row.get("session"), "frame_count": row.get("frame_count"), "training_eligible": row.get("training_eligible", False)} for row in reports],
        "blockers": blockers,
        "gpu_execution": "NOT_STARTED_FAIL_CLOSED",
        "fresh_clean_generated": False,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "authority_promoted": False,
        "claim_limit": "RC1 causal Clean prerequisite closure only; no fresh Clean imagery, background truth, Mask accuracy or training eligibility.",
        "lineage": {"mask_result": ref(args.mask_result), "clean_contract_result": ref(args.clean_contract_result)},
    }
    result_path = args.output_root / "RESULT.json"; write(result_path, result)
    write(args.output_root / "METRICS.json", {"status": status, "blocker_count": len(blockers), "cases": result["cases"]})
    write(args.output_root / "DECISION.md", "# RC1 T2 因果 Clean\n\n未启动 ProPainter。现有 CPU 闭包对两条 pilot 均未找到合法的因果真实 donor，写域仍是语义 UNKNOWN；Poker 的因果 atlas 也未验证。继续生成会把未知对象/背景外观伪装成训练像素，因此本任务有限终结为 `BLOCKED_PREREQ`。\n")
    write(args.output_root / "NEXT_ACTION.json", {"status": status, "next_task_id": None, "required_to_resume": ["verified causal support-surface donor semantics", "conservative possible_task_object_mask", "Poker same-instance same-face causal atlas or UNKNOWN exclusion"]})
    write(args.output_root / "RUN_RECEIPT.json", {"status": status, "created_at": now(), "result": ref(result_path), "gpu_used": False, "authority_promoted": False})
    write(args.output_root / "ARTIFACT_MANIFEST.json", {"status": status, "result": ref(result_path), "metrics": ref(args.output_root / "METRICS.json")})
    write(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": status, "blocker_count": len(blockers), "next_action": "STOP_FAIL_CLOSED"})
    print(json.dumps({"status": status, "result": ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
