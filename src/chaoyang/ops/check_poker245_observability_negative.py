#!/usr/bin/env python3
"""Frozen-threshold CPU rejection check for Poker245 frames 45..60."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cv2

from chaoyang.ops.audit_poker245_donor_observability import read_prefix, ref, track_order, verify, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.packet.resolve(strict=True)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    parent_path = args.parent_result.resolve(strict=True)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("session_id") != "play_cards_0903_245" or parent.get("formal_written_pixels") != 0:
        raise RuntimeError("parent identity/strict-write preflight failed")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(packet["write_root"]).resolve()):
        raise RuntimeError("fresh immutable output under original write root required")
    for name, pin in packet["inputs"].items():
        verify(pin, name)
    for name, pin in packet["code"].items():
        verify(pin, f"code:{name}")
    raw = Path(packet["inputs"]["raw_video"]["path"])
    objects = json.loads(Path(packet["inputs"]["object_manifest"]["path"]).read_text())
    source_pin = objects["frames"][12]["physical_instances"]["0"]["mask"]
    source_path = verify(source_pin, "source_mask")
    source_mask = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE) > 0
    frames = read_prefix(raw, 60)
    ledger, cue = track_order(frames, source_mask, packet["config"], 60)
    selected = [item for item in ledger if 45 <= item["frame"] <= 60]
    if len(selected) != 16:
        raise RuntimeError("negative frame selection mismatch")
    video_pin = json.loads(Path(packet["inputs"]["v4_result"]["path"]).read_text())["outputs"]["review_video"]
    inherited_video = ref(verify(video_pin, "inherited_v4_review_video"))
    output.mkdir(parents=True)
    metadata = {
        "schema_version": "poker245-observability-negative-v1",
        "session_id": "play_cards_0903_245", "task_id": "poker245_observability_negative_45_60",
        "created_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(), "process_startticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
        "parent_result": ref(parent_path), "packet": ref(packet_path),
        "runner": ref(Path(__file__)), "inherited_v4_review_video": inherited_video,
        "input_mode": "OFFLINE_VISUAL", "authority_promoted": False, "training_eligible": False,
        "frame_ids": list(range(45, 61)),
        "thresholds_identical_to_parent": True,
        "prefix_cue_at_60": cue,
        "rows": [{"frame": item["frame"], "cue_pass": item["cue_pass"],
                  "ordered": item["ordered"], "center_stable": item["center_stable"],
                  "three_visible_back_components": item["three_visible_back_components"],
                  "left_back_area_px": item["left"]["area_px"] if item["left"] else 0,
                  "middle_back_area_px": item["middle"]["area_px"] if item["middle"] else 0,
                  "rightmost_back_area_px": item["rightmost"]["area_px"] if item["rightmost"] else 0}
                 for item in selected],
        "rejected_count": sum(not item["cue_pass"] for item in selected),
        "decision": "REJECT_ALL_16_DEVELOPMENT_WRITES" if not any(item["cue_pass"] for item in selected) else "NEGATIVE_CHECK_FAILED",
        "formal_written_pixels": 0,
        "claim_limit": "One same-session negative check only; rejection is an internal colour/order cue, not hidden-pixel truth or independent held-out accuracy.",
    }
    signature_payload = {"packet_sha": metadata["packet"]["sha256"],
                         "parent_sha": metadata["parent_result"]["sha256"],
                         "runner_sha": metadata["runner"]["sha256"],
                         "config": packet["config"], "range": [45, 60]}
    metadata["run_signature"] = hashlib.sha256(json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    write_json(output / "RESULT.json", metadata)
    print(json.dumps({"decision": metadata["decision"], "rejected_count": metadata["rejected_count"],
                      "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0 if metadata["rejected_count"] == 16 else 2


if __name__ == "__main__":
    raise SystemExit(main())
