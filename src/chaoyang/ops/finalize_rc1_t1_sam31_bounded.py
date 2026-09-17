#!/usr/bin/env python3
"""Seal the two bounded SAM3.1 revisions with corrected stream semantics."""
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
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha(value)}


def write(path: Path, value: Any) -> None:
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-1", type=Path, required=True)
    parser.add_argument("--revision-2", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    revisions = []
    for label, result_path in (("detector_raw_id", args.revision_1), ("stable_primed_point_id", args.revision_2)):
        result = json.loads(result_path.resolve(strict=True).read_text())
        metrics_path = Path(result["metrics"]["path"])
        metrics = json.loads(metrics_path.read_text())
        targets = {}
        for name, row in metrics.get("target_results", {}).items():
            fraction = float(row["gates"]["development_quality_gates"].get("accepted_id_presence_fraction", 0.0))
            targets[name] = {
                "accepted_id_presence_fraction": fraction,
                "functional_stream_presence_pass": fraction >= 0.05,
                "visibility_conditioned_presence_95pct": "NOT_MEASURABLE_NO_INDEPENDENT_VISIBLE_FRAME_REFERENCE",
            }
        revisions.append({"revision": label, "reported_status": result.get("status"), "result": ref(result_path), "metrics": ref(metrics_path), "targets": targets, "corrected_pass": bool(targets) and all(x["functional_stream_presence_pass"] for x in targets.values())})
    passed = any(row["corrected_pass"] for row in revisions)
    status = "PASSED" if passed else "FAILED_QUALITY_C"
    result = {
        "schema_version": "chaoyang-rc1-t1-sam31-bounded-final-v1",
        "task_id": "rc1_t1_sam31_mask_bounded", "created_at": now(), "status": status,
        "bounded_revisions_exhausted": True, "revisions": revisions,
        "decision": "STOP_NO_THIRD_SEGMENTATION_MODEL" if not passed else "ALLOW_FROZEN_REGRESSION_ONLY",
        "authority_promoted": False, "external_accuracy": "UNKNOWN",
        "claim_limit": "Bounded SAM3.1 functional temporal-stream evidence only. The 5% gate detects a broken stream and is not pixel accuracy or the 95% visibility-conditioned quality metric.",
    }
    result_path = args.output_root / "RESULT.json"; write(result_path, result)
    write(args.output_root / "METRICS.json", {"status": status, "revisions": revisions})
    write(args.output_root / "DECISION.md", "# RC1 T1 SAM3.1 有界结论\n\n两轮均只在 seed 帧产生目标，三条输出率约 0.2%。旧评估器漏掉持续输出门，第二轮的 `PASSED_DEVELOPMENT` 不可用于晋升。CPU 复核将本任务封为 `FAILED_QUALITY_C`；不接第三种分割模型。\n")
    write(args.output_root / "NEXT_ACTION.json", {"status": status, "next_task_id": "rc1_t2_causal_clean", "routing": "use_existing_passed_mask_sessions_only; failed cluster remains C"})
    write(args.output_root / "RUN_RECEIPT.json", {"status": status, "created_at": now(), "result": ref(result_path), "gpu_used": False, "authority_promoted": False})
    write(args.output_root / "ARTIFACT_MANIFEST.json", {"status": status, "result": ref(result_path), "metrics": ref(args.output_root / "METRICS.json")})
    write(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": status, "decision": result["decision"], "next_action": "REGISTER_T2"})
    print(json.dumps({"status": status, "result": ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
