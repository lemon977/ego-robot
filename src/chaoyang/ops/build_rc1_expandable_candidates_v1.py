#!/usr/bin/env python3
"""Classify excluded exact78 clips without treating source IDs as acquisitions."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, validate_artifact_ref


def connected_components(rows: list[dict]) -> tuple[dict[str, str], int]:
    """Return source-session component per clip; an upper bound, not RGB proof."""
    parent: dict[str, str] = {}

    def root(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = root(parent[value])
        return parent[value]

    for row in rows:
        sources = row.get("claimed_source_sessions", [])
        if sources:
            anchor = root(sources[0])
            for other in sources[1:]:
                parent[root(other)] = anchor
    clip_root = {
        row["session_id"]: root(row["claimed_source_sessions"][0])
        for row in rows if row.get("claimed_source_sessions")
    }
    return clip_root, len(set(clip_root.values()))


def expansion_route(first_blocker: str) -> str:
    return {
        "CALIBRATION_MISSING": "VISUAL_TIER_REVIEW_NO_METRIC_CONTACT",
        "METRIC_GEOMETRY_READY": "AUDIT_SELECTION_SPLIT_AND_EXISTING_ROBOT_RESULT",
        "HAWOR_C": "BOUNDED_HAWOR_SUCCESSOR",
        "ROLE_MASK_C": "BOUNDED_SAM31_ROLE_SUCCESSOR",
        "OBJECT_MASK_C": "BOUNDED_SAM31_OBJECT_IDENTITY_SUCCESSOR",
    }.get(first_blocker, "BLOCKED_CLASSIFICATION_REQUIRED")


def build(master: dict, source: dict, conversion: dict) -> dict:
    masters = {row["session_id"]: row for row in master["rows"]}
    sources = {row["session_id"]: row for row in source["rows"]}
    causes = {row["session_id"]: row for row in conversion["rows"]}
    if len(masters) != 156 or len(sources) != 156 or len(causes) != 156:
        raise ValueError("exact78 inputs do not each contain 156 unique sessions")
    if set(masters) != set(sources) or set(masters) != set(causes):
        raise ValueError("exact78 input session sets disagree")
    result_rows = []
    summary = {}
    for task in ("chips", "poker"):
        subset = [sources[session] for session, row in masters.items() if row["task"] == task]
        if len(subset) != 78:
            raise ValueError(f"{task} denominator is not 78")
        component_by_session, all_components = connected_components(subset)
        candidate_components = {
            component_by_session[row["session_id"]]
            for row in subset
            if row["session_id"] in component_by_session
            and masters[row["session_id"]]["legacy_candidate_split"] != "not_candidate"
        }
        expandable_components = set()
        blocker_counts = Counter()
        route_counts = Counter()
        unknown_source_rows = 0
        for source_row in subset:
            session = source_row["session_id"]
            master_row = masters[session]
            cause = causes[session]
            if not source_row.get("claimed_source_sessions"):
                unknown_source_rows += 1
            if master_row["legacy_candidate_split"] != "not_candidate":
                continue
            blocker = cause["first_blocker"]
            route = expansion_route(blocker)
            blocker_counts[blocker] += 1
            route_counts[route] += 1
            component = component_by_session.get(session)
            if component is not None and component not in candidate_components:
                expandable_components.add(component)
            result_rows.append({
                "session_id": session,
                "task": task,
                "frame_count": master_row["frame_count"],
                "scheduled_h50_start_count": master_row["scheduled_h50_start_count"],
                "first_blocker": blocker,
                "blocker_class": cause["blocker_class"],
                "successor_route_from_conversion": cause["successor_route"],
                "research_expansion_route": route,
                "legacy_split": cause.get("split"),
                "candidate_exclusion_is_quality_result": False,
                "claimed_source_sessions": source_row.get("claimed_source_sessions", []),
                "source_component_upper_bound_key": component,
                "source_component_already_in_shortlist": component in candidate_components if component is not None else None,
                "original_rgb_source_proven": source_row.get("original_rgb_per_frame_source_proven", False),
                "independent_acquisition_proven": source_row.get("original_acquisition_independence_proven", False),
                "proof_status": source_row.get("proof_status"),
                "assets": source_row.get("assets", {}),
                "blocker_evidence": cause.get("evidence"),
                "claim_limit": "Research expansion candidate only; not quality passed, group-independent or train eligible.",
            })
        summary[task] = {
            "all_sessions": 78,
            "frozen_shortlist_sessions": sum(row["task"] == task and row["legacy_candidate_split"] != "not_candidate" for row in masters.values()),
            "excluded_sessions": sum(blocker_counts.values()),
            "excluded_by_first_blocker": dict(sorted(blocker_counts.items())),
            "excluded_by_research_route": dict(sorted(route_counts.items())),
            "source_session_components_all_upper_bound": all_components,
            "shortlist_components_upper_bound": len(candidate_components),
            "additional_nonshortlist_components_upper_bound": len(expandable_components),
            "unknown_source_sessions": unknown_source_rows,
            "verified_independent_rgb_acquisitions": 0,
            "claim_limit": "Connected sourceSession components are only possible group upper bounds; original RGB lineage and quality are not proven.",
        }
    result_rows.sort(key=lambda row: (row["task"], row["session_id"]))
    if len(result_rows) != 111:
        raise ValueError("frozen non-shortlist denominator is not 111")
    return {"summary": summary, "rows": result_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conversion", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    refs = {name: artifact_ref(getattr(args, name)) for name in ("master", "source", "conversion")}
    for ref in refs.values():
        if validate_artifact_ref(ref):
            raise RuntimeError("input closure failed")
    report = build(*[json.loads(Path(refs[name]["path"]).read_text(encoding="utf-8")) for name in ("master", "source", "conversion")])
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "EXPANDABLE_CANDIDATES.json"
    atomic_json(path, {
        "schema_version": "chaoyang-rc1-expandable-candidates-v1",
        "created_at": now_iso(),
        "status": "PASSED_RESEARCH_AUDIT",
        "authority_promoted": False,
        "claim_limit": "Only frozen-shortlist expansion potential; full exact78 is not proven insufficient or sufficient for formal checkpoint capacity.",
        "inputs": refs,
        **report,
    })
    csv_path = output / "EXPANDABLE_CANDIDATES.csv"
    fields = ["session_id", "task", "frame_count", "scheduled_h50_start_count", "first_blocker", "research_expansion_route", "source_component_upper_bound_key", "source_component_already_in_shortlist", "original_rgb_source_proven", "independent_acquisition_proven"]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in report["rows"])
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema_version": "chaoyang-rc1-expandable-candidates-result-v1",
        "created_at": now_iso(),
        "status": "PASSED_RESEARCH_AUDIT",
        "authority_promoted": False,
        "claim_limit": "No RC1 shortlist, split, terminal or checkpoint status changed.",
        "inputs": refs,
        "summary": report["summary"],
        "index": artifact_ref(path),
        "csv": artifact_ref(csv_path),
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "summary": result["summary"], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
