#!/usr/bin/env python3
"""Apply the frozen local attachment tracker to the existing 0902_042 seeds."""
from __future__ import annotations

import json
from pathlib import Path

from chaoyang.ops.run_human_to_robot_quality_closure_s1_attachment_canary import (
    ATTEMPT,
    V5,
    now,
    run_case,
    write_once,
)
from chaoyang.pipeline.attachment_tracker_v1 import read_binary_mask

REPO = Path("/mnt/workspace/code/chaoyang")
SESSION = "play_cards_0902_042"


def main() -> int:
    masks = V5 / "masks_042_v1"
    device = masks / "capture_device"
    seeds = [
        int(path.stem) for path in sorted(device.glob("*.png"))
        if read_binary_mask(str(path)).any()
    ]
    if len(seeds) != 59:
        raise RuntimeError(f"FROZEN_SEED_COUNT_DRIFT:{len(seeds)}")
    raw = (
        REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
        "lanes/exact78/exact_domain_v1/play_cards_0902_042/raw"
    )
    root = ATTEMPT / "lanes/scene_evidence/attachment_track_042_canary_v1"
    if root.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{root}")
    output = run_case(SESSION, {"short": "042", "raw": raw, "masks": masks, "seeds": seeds}, root)
    result = {
        "schema_version": "LOCAL_ATTACHMENT_TRACK_042_CANARY_SUMMARY_V1",
        "created_at": now(), "status": "EXECUTED_PENDING_VISUAL_REVIEW",
        "session_id": SESSION, "seed_count": len(seeds), "output": output,
        "long_gap_policy": "UNKNOWN_NO_FORWARD_FILL",
        "same_signature_scene_retry": False, "new_model_invocations": 0,
    }
    write_once(root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
