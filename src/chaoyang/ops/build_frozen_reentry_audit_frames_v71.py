#!/usr/bin/env python3
"""Freeze development-only per-instance visible/empty/re-entry audit windows.

The builder consumes governed terminal ledgers and their explicit Object Mask
manifests.  It never infers physical occlusion: an empty segment means only
``EMPTY_OR_UNOBSERVED`` in the predecessor manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
DEFAULT_SELECTION = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/SUCCESSOR_CANARY_SELECTION_V71.json"
DEFAULT_ROLE_LEDGER = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/MASK_ROLE_SUCCESSOR_V3_TERMINAL_INDEX.json"
DEFAULT_OBJECT_LEDGER = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_TASK_OBJECT_TERMINAL_INDEX.json"
DEFAULT_OUT = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_reentry_audit"
SCHEMA = REPO / "contracts/frozen_reentry_audit_frames_v71.schema.json"
ARCHIVED_TASKS = REPO / "archive/baseline-20260917-0aa69e9/content/history/tasks"


def _archived_path(value: str) -> Path:
    prefix = str(REPO / "tasks") + "/"
    return ARCHIVED_TASKS / value[len(prefix):] if value.startswith(prefix) else Path(value)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable output conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _index(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["session_id"]: row for row in ledger["terminals"]}


def _manifest_from_result(row: dict[str, Any]) -> Path | None:
    result_path = _archived_path(row["result"]["path"])
    if not result_path.is_file():
        return None
    result = _load(result_path)
    manifest = result.get("artifacts", {}).get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("path"), str):
        path = _archived_path(manifest["path"])
        return path if path.is_file() else None
    return None


def _bounded_empty_events(sequence: list[bool]) -> list[tuple[int, int, int, int]]:
    """Return (visible_before, empty_start, empty_end, visible_after)."""
    events: list[tuple[int, int, int, int]] = []
    i = 1
    while i < len(sequence) - 1:
        if sequence[i - 1] and not sequence[i]:
            start = i
            while i < len(sequence) and not sequence[i]:
                i += 1
            if i < len(sequence):
                events.append((start - 1, start, i - 1, i))
            continue
        i += 1
    return events


def _selection_groups(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile all Mask successor groups; HaWoR is outside S1 modal-mask scope."""
    groups = []
    for src in selection["groups"]:
        if src["stage"] not in {"role_mask", "object_identity"}:
            continue
        groups.append(
            {
                "group_id": f"S1_{src['stage'].upper()}_{src['task'].upper()}",
                "stage": src["stage"],
                "task": src["task"],
                "failure_canary": src["canary_session"],
                "regressions": list(src["regression_sessions"]),
            }
        )
    return groups


def build(plan_path: Path, selection_path: Path, role_path: Path, object_path: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    selection = _load(selection_path)
    role_rows = _index(_load(role_path))
    object_rows = _index(_load(object_path))
    s1 = next(row for row in plan["canaries"] if row["canary_id"] == "S1")
    groups = _selection_groups(selection)
    rows: list[dict[str, Any]] = []

    for group in groups:
        sessions = [(group["failure_canary"], "FAILURE_CANARY")]
        sessions.extend((x, "A_B_REGRESSION") for x in group["regressions"])
        expected_instances = range(3) if group["task"] == "chips" else range(1)
        for session_id, selection_role in sessions:
            role_row = role_rows.get(session_id, {})
            object_row = object_rows.get(session_id, {})
            manifest_path = _manifest_from_result(object_row) if object_row else None
            if manifest_path is None:
                for instance_id in expected_instances:
                    rows.append(
                        {
                            "group_id": group["group_id"], "stage": group["stage"], "task": group["task"],
                            "session_id": session_id, "selection_role": selection_role,
                            "physical_instance_id": instance_id,
                            "role_mask_grade": role_row.get("grade", "UNKNOWN"),
                            "object_mask_grade": object_row.get("grade", "UNKNOWN"),
                            "status": "BLOCKED_INSUFFICIENT_TRANSITION_EVIDENCE",
                            "blocked_reason": "CURRENT_OBJECT_MASK_LEDGER_HAS_NO_READABLE_PER_FRAME_MANIFEST",
                            "source_manifest": None, "event": None,
                        }
                    )
                continue

            manifest = _load(manifest_path)
            frames = manifest.get("frames", [])
            manifest_ev = _evidence(manifest_path)
            for instance_id in expected_instances:
                key = str(instance_id)
                sequence = []
                for frame in frames:
                    value = frame.get("physical_instances", {}).get(key, {})
                    sequence.append(bool(value.get("observed") and value.get("valid") and int(value.get("area_px", 0)) > 0))
                events = _bounded_empty_events(sequence)
                if not events:
                    rows.append(
                        {
                            "group_id": group["group_id"], "stage": group["stage"], "task": group["task"],
                            "session_id": session_id, "selection_role": selection_role,
                            "physical_instance_id": instance_id,
                            "role_mask_grade": role_row.get("grade", "UNKNOWN"),
                            "object_mask_grade": object_row.get("grade", "UNKNOWN"),
                            "status": "BLOCKED_INSUFFICIENT_TRANSITION_EVIDENCE",
                            "blocked_reason": "NO_VISIBLE_TO_EMPTY_OR_UNOBSERVED_TO_VISIBLE_SEQUENCE",
                            "source_manifest": manifest_ev, "event": None,
                        }
                    )
                    continue
                for event_index, (before, empty_start, empty_end, after) in enumerate(events):
                    midpoint = (empty_start + empty_end) // 2
                    audit_frames = sorted(set([before, empty_start, midpoint, empty_end, after]))
                    event_id = f"{group['group_id']}:{session_id}:instance_{instance_id}:event_{event_index:03d}"
                    rows.append(
                        {
                            "group_id": group["group_id"], "stage": group["stage"], "task": group["task"],
                            "session_id": session_id, "selection_role": selection_role,
                            "physical_instance_id": instance_id,
                            "role_mask_grade": role_row.get("grade", "UNKNOWN"),
                            "object_mask_grade": object_row.get("grade", "UNKNOWN"),
                            "status": "FROZEN_EVENT", "blocked_reason": None,
                            "source_manifest": manifest_ev,
                            "event": {
                                "event_id": event_id,
                                "last_visible_before": before,
                                "empty_or_unobserved_start": empty_start,
                                "empty_or_unobserved_end": empty_end,
                                "first_visible_after": after,
                                "empty_run_length": empty_end - empty_start + 1,
                                "window_start": max(0, before - 2),
                                "window_end": min(len(sequence) - 1, after + 2),
                                "audit_frames": audit_frames,
                                "transition_semantics": "VISIBLE_TO_EMPTY_OR_UNOBSERVED_TO_VISIBLE",
                            },
                        }
                    )

    rows.sort(key=lambda x: (x["group_id"], x["session_id"], x["physical_instance_id"], (x["event"] or {}).get("event_id", "")))
    evidence_paths = [plan_path, selection_path, role_path, object_path, SCHEMA, Path(__file__)]
    source_evidence = [_evidence(x) for x in evidence_paths]
    signature_payload = {
        "sources": [x["sha256"] for x in source_evidence],
        "event_rule": "ALL_MAXIMAL_BOUNDED_FALSE_RUNS",
        "window_padding_frames": 2,
        "audit_frame_rule": "BOUNDARIES_PLUS_EMPTY_MIDPOINT",
    }
    producer_signature = hashlib.sha256(_json_bytes(signature_payload)).hexdigest()
    frozen = sum(x["status"] == "FROZEN_EVENT" for x in rows)
    blocked = sum(x["status"] == "BLOCKED_INSUFFICIENT_TRANSITION_EVIDENCE" for x in rows)
    return {
        "schema_version": "frozen-reentry-audit-frames-v71",
        "artifact_id": "FROZEN_REENTRY_AUDIT_FRAMES_V71",
        "artifact_revision": "R7_1",
        "immutable": True,
        "authority": False,
        "audit_set_type": "DEVELOPMENT_AUDIT_FRAME_SELECTION",
        "transition_signal": "OBJECT_MASK_OBSERVED_AND_VALID_AND_NONEMPTY",
        "gold_accuracy_authorized": False,
        "selection_groups": groups,
        "rows": rows,
        "counts": {
            "selection_groups": len(groups),
            "row_records": len(rows),
            "frozen_events": frozen,
            "blocked_instance_records": blocked,
            "unique_sessions": len({x["session_id"] for x in rows}),
            "unique_event_audit_frames": len({(x["session_id"], f) for x in rows if x["event"] for f in x["event"]["audit_frames"]}),
        },
        "source_evidence": source_evidence,
        "producer_signature": producer_signature,
        "claim_limit": (
            "Development-only deterministic frame selection from predecessor modal Object Mask visibility. "
            "EMPTY_OR_UNOBSERVED is not a physical-occlusion label; this artifact is not Gold, does not "
            "support accuracy, Contact, amodal Mask, Object6D or physical-truth claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--role-ledger", type=Path, default=DEFAULT_ROLE_LEDGER)
    parser.add_argument("--object-ledger", type=Path, default=DEFAULT_OBJECT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    artifact = build(args.plan, args.selection, args.role_ledger, args.object_ledger)
    json_path = args.output_dir / "FROZEN_REENTRY_AUDIT_FRAMES_V71.json"
    csv_path = args.output_dir / "FROZEN_REENTRY_AUDIT_FRAMES_V71.csv"
    _write_immutable(json_path, _json_bytes(artifact))

    columns = [
        "group_id", "stage", "task", "session_id", "selection_role", "physical_instance_id",
        "role_mask_grade", "object_mask_grade", "status", "blocked_reason", "event_id",
        "last_visible_before", "empty_or_unobserved_start", "empty_or_unobserved_end",
        "first_visible_after", "empty_run_length", "window_start", "window_end", "audit_frames",
        "source_manifest_path", "source_manifest_sha256",
    ]
    csv_rows = []
    for row in artifact["rows"]:
        event = row["event"] or {}
        flat = {k: row.get(k) for k in columns}
        for key in (
            "event_id", "last_visible_before", "empty_or_unobserved_start",
            "empty_or_unobserved_end", "first_visible_after", "empty_run_length",
            "window_start", "window_end",
        ):
            flat[key] = event.get(key)
        flat["audit_frames"] = ";".join(map(str, event.get("audit_frames", [])))
        source_manifest = row["source_manifest"] or {}
        flat["source_manifest_path"] = source_manifest.get("path")
        flat["source_manifest_sha256"] = source_manifest.get("sha256")
        csv_rows.append(flat)
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(csv_rows)
    _write_immutable(csv_path, buf.getvalue().encode("utf-8"))

    result = {
        "schema_version": "frozen-reentry-audit-result-v71",
        "status": "PASSED_DEVELOPMENT_AUDIT_SELECTION",
        "authority": False,
        "metrics": artifact["counts"],
        "outputs": [_evidence(json_path), _evidence(csv_path)],
        "producer_signature": artifact["producer_signature"],
        "claim_limit": artifact["claim_limit"],
    }
    _write_immutable(args.output_dir / "RESULT.json", _json_bytes(result))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
