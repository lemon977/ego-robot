#!/usr/bin/env python3
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Single-session S1 GPU adapter.  Default mode is import/config preflight."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import real_attempt, runtime_import_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pinned causal SAM2.1/Cutie single-session development adapter"
    )
    parser.add_argument("--backend", choices=["sam2_1", "cutie"], required=True)
    parser.add_argument("--runtime-import-preflight-only", action="store_true")
    parser.add_argument("--execute-development-canary", action="store_true")
    parser.add_argument("--rgb-manifest", type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--artifact-revision", default="R7_1")
    parser.add_argument("--executor-epoch", type=int)
    parser.add_argument("--fencing-token")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--lease-path", type=Path, default=ROOT / "_run/current/GPU_LEASE.json")
    parser.add_argument("--lease-lock-path", type=Path, default=ROOT / "_run/current/GPU_LEASE.lock")
    args = parser.parse_args(argv)
    if args.runtime_import_preflight_only == args.execute_development_canary:
        parser.error("select exactly one of import preflight or development execution")
    if args.runtime_import_preflight_only:
        result = runtime_import_preflight(args.backend)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["execution_ready"] else 3
    required = {
        "rgb_manifest": args.rgb_manifest,
        "prompt_manifest": args.prompt_manifest,
        "output_root": args.output_root,
        "task_id": args.task_id,
        "attempt_id": args.attempt_id,
        "executor_epoch": args.executor_epoch,
        "fencing_token": args.fencing_token,
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        parser.error(f"execution requires: {', '.join(missing)}")
    if len(args.fencing_token) < 16:
        parser.error("fencing token must contain at least 16 characters")
    receipt = real_attempt(
        rgb_manifest_path=args.rgb_manifest.resolve(),
        prompt_manifest_path=args.prompt_manifest.resolve(),
        output_root=args.output_root.resolve(),
        backend_name=args.backend,
        task_id=args.task_id,
        attempt_id=args.attempt_id,
        artifact_revision=args.artifact_revision,
        executor_epoch=args.executor_epoch,
        fencing_token=args.fencing_token,
        gpu_id=args.gpu_id,
        lease_path=args.lease_path.resolve(),
        lease_lock_path=args.lease_lock_path.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["payload"]["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
