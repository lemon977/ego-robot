#!/usr/bin/env python3
"""Pin every protected raw RGB frame used by a full SAM3.1 research regression."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
from pathlib import Path

import cv2

from chaoyang.governance.common import artifact_ref, atomic_json


SESSIONS = {"play_cards_0901_001": 645, "play_cards_0901_005": 520}
RAW_ROOT = Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", choices=tuple(SESSIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable source closure required")
    count = SESSIONS[args.session_id]
    rows = []
    aggregate = hashlib.sha256()
    for frame in range(count):
        path = RAW_ROOT / args.session_id / "preprocess/all_data" / f"{frame:05d}" / "rgb.png"
        ref = artifact_ref(path)
        if frame in {0, count - 1}:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (960, 1280):
                raise ValueError(f"raw RGB image domain/shape mismatch: {frame}")
        rows.append({"source_frame_id": frame, **ref})
        aggregate.update(f"{frame}\0{ref['bytes']}\0{ref['sha256']}\n".encode())
    payload = {
        "schema_version": "sam31-full-raw-rgb-source-closure-v1",
        "session_id": args.session_id,
        "frame_count": count,
        "frame_ids": "EXACT_0_TO_N_MINUS_1",
        "image_domain": "raw_poker_rgb_1280x960_png",
        "coordinate_semantics": "Original source RGB pixels; no crop, resize, registration or camera/world pose is inferred by this manifest.",
        "ordered_manifest_sha256": aggregate.hexdigest(),
        "frames": rows,
        "code": artifact_ref(Path(__file__)),
        "claim_limit": "Exact RGB file identity only; does not prove physical card instance or mask quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps({"session_id": args.session_id, "frame_count": count,
                      "ordered_manifest_sha256": aggregate.hexdigest(), "output": str(args.output)}))


if __name__ == "__main__":
    main()
