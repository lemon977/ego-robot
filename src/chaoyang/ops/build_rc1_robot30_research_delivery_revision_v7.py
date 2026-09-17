#!/usr/bin/env python3
"""Immutable rev7: recompute Robot30 video counts from the frozen 60 rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_independent_placement_successor_v1/delivery_index_rev_0006/ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0006.json"
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_independent_placement_successor_v1/delivery_index_rev_0007/ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0007.json"
ADDENDUM = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_independent_placement_gate_v1/REFERENCE_CLOSURE_ADDENDUM.json"


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def delivered(row: dict) -> bool:
    evidence = row.get("delivery_full_video")
    if evidence is None:
        return False
    if not isinstance(evidence, dict):
        raise RuntimeError(f"{row['session_id']}: invalid video evidence")
    artifact = evidence.get("artifact", evidence)
    if not all(key in artifact for key in ("path", "bytes", "sha256")):
        raise RuntimeError(f"{row['session_id']}: video lacks path/bytes/SHA")
    actual = ref(Path(artifact["path"]))
    if (actual["bytes"], actual["sha256"]) != (artifact["bytes"], artifact["sha256"]):
        raise RuntimeError(f"{row['session_id']}: video SHA closure mismatch")
    other = row.get("verified_video")
    if other is not None and (not isinstance(other, dict) or other.get("sha256") != actual["sha256"]):
        raise RuntimeError(f"{row['session_id']}: verified video does not match delivery")
    decoder = evidence.get("full_decode_receipt_pass") is True or row.get("full_video_decode") == "PASS_FFMPEG_XERROR"
    if not decoder:
        raise RuntimeError(f"{row['session_id']}: no bound full-decode receipt")
    known_count = row.get("verified_frame_count")
    if known_count is not None and known_count != evidence.get("frames", known_count):
        raise RuntimeError(f"{row['session_id']}: frame count disagreement")
    if evidence.get("frames", known_count) is None:
        raise RuntimeError(f"{row['session_id']}: no closed full-session frame count")
    return True


def rebuild_counts(rows: list[dict]) -> tuple[dict, dict]:
    if len(rows) != 60 or len({row["session_id"] for row in rows}) != 60:
        raise RuntimeError("Robot30 must have 60 unique full session IDs")
    counts = {}
    invariant = {}
    for task in ("chips", "poker"):
        selected = [row for row in rows if row["task"] == task]
        if len(selected) != 30 or sorted(row["rank"] for row in selected) != list(range(1, 31)):
            raise RuntimeError(f"{task}: frozen rank 1..30 closure failed")
        watched = [row for row in selected if delivered(row)]
        prior = sum(isinstance(row["delivery_full_video"], dict) and
                    row["delivery_full_video"].get("source_kind") == "PRIOR_VERIFIED" for row in watched)
        c_video = sum(row["delivery_class"] == "C_WATERMARKED_FULL_DIAGNOSTIC_VIDEO" for row in watched)
        counts[task] = {
            "selected": 30,
            "all_watchable_full_video": len(watched),
            "missing_full_video": 30 - len(watched),
            "prior_closed_video": prior,
            "new_c_diagnostic_video": c_video,
            "prior_hard_geometry_pass": sum(row.get("robot30_hard_geometry_pass") is True for row in selected),
        }
        invariant[task] = {
            "frozen_rows": 30,
            "unique_session_ids": 30,
            "delivery_rows_with_verified_artifact_and_decode": len(watched),
            "row_boolean_sum_equals_counts": len(watched) == counts[task]["all_watchable_full_video"],
            "extra_diagnostic_videos_outside_frozen_30": "NOT_COUNTED",
        }
    return counts, invariant


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    revised = copy.deepcopy(original)
    counts, invariant = rebuild_counts(revised["rows"])
    revised.update({
        "schema_version": "rc1-robot30-research-delivery-matrix-rev7",
        "counts": counts,
        "normalized_watchable_counts": {task: counts[task]["all_watchable_full_video"] for task in counts},
        "count_invariant": invariant,
        "supersedes": ref(SOURCE),
        "generator": ref(Path(__file__)),
        "independent_placement_reference_closure_addendum": ref(ADDENDUM),
        "claim_limit": "Fixed-30-row research watchable-video count only. Chips30/30, Poker21/30 includes Poker008 C-watermarked diagnostic. No Robot quality, causal training, control or deployment authority changed.",
    })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(revised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": ref(OUTPUT)["sha256"],
                      "counts": revised["normalized_watchable_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
