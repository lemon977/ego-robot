#!/usr/bin/env python3
"""Bounded read-only GPU/CPU sampler for a pinned research worker PID."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def sample(pid: int) -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5).strip().splitlines()[0]
    used, free, util = (int(value.strip()) for value in output.split(","))
    statm = Path(f"/proc/{pid}/statm")
    rss_pages = int(statm.read_text().split()[1]) if statm.exists() else None
    return {"elapsed_s": None, "gpu_used_mib": used, "gpu_free_mib": free,
            "gpu_util_percent": util, "cpu_rss_pages": rss_pages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--wall-cap-s", type=int, default=2700)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not Path(f"/proc/{args.pid}/statm").exists():
        raise ProcessLookupError(args.pid)
    start = time.monotonic()
    rows = []
    while time.monotonic() - start < args.wall_cap_s:
        if not Path(f"/proc/{args.pid}/statm").exists():
            break
        try:
            row = sample(args.pid)
            row["elapsed_s"] = round(time.monotonic() - start, 3)
            rows.append(row)
        except (OSError, subprocess.SubprocessError, ValueError):
            rows.append({"elapsed_s": round(time.monotonic() - start, 3), "sample_error": True})
        time.sleep(args.interval_s)
    payload = {
        "schema_version": "research-gpu-process-samples-v1", "worker_pid": args.pid,
        "sample_count": len(rows), "samples": rows,
        "peak_gpu_used_mib": max((row["gpu_used_mib"] for row in rows if "gpu_used_mib" in row), default=None),
        "peak_cpu_rss_pages": max((row["cpu_rss_pages"] for row in rows if row.get("cpu_rss_pages") is not None), default=None),
        "claim_limit": "Physical GPU total includes other processes; sampling may miss transient peaks. This is not per-model CUDA allocated/reserved memory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": payload["sample_count"], "peak_gpu_used_mib": payload["peak_gpu_used_mib"]}))


if __name__ == "__main__":
    main()
