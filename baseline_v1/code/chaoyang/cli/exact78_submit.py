"""Thin Exact78 submission front end; T1-B owns candidate task registration.

The front end never imports a model or creates a second writer. It validates a
raw request and attaches only to an already fenced candidate of the same
input identity. T1-B will provide the first-run registration path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.exact78_cold_start_guard_v1 import (
    DATA_ROOT, PROJECT_ROOT, ColdStartInputError, file_ref, inspect_session,
    validate_manifest,
)

T0_FREEZE = PROJECT_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001/INPUT_FREEZE.json"
T0_FREEZE_SHA = "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88"
T0_CORRECTION = PROJECT_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/RESULT.json"
T0_CORRECTION_SHA = "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c"
T0_CAPABILITY_SHA = "5577a80426333b5bd3eb26eb378f2332b8a2827027f01be5b439090087e00475"
T0_SELECTED_IDS = frozenset({
    "get_potato_chips_0901_013", "get_potato_chips_0902_103",
    "play_cards_0901_001", "play_cards_0902_042",
    "play_cards_0915_001", "play_cards_0915_002",
})
FORBIDDEN_DERIVED_COMPONENTS = frozenset({
    "preprocess", "all_data", "trackingdata", "processed", "clean", "robot",
    "pico", "manus",
})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def build_request(*, session: Path | None = None, manifest: Path | None = None,
                  allowed_root: Path = DATA_ROOT) -> dict[str, Any]:
    if (session is None) == (manifest is None):
        raise ColdStartInputError("EXACTLY_ONE_SESSION_OR_MANIFEST_REQUIRED")
    manifest_ref: dict[str, Any] | None = None
    frozen_selection: dict[str, Any] | None = None
    corrected_calibration: dict[str, Any] | None = None
    if manifest is None:
        sessions = [session]
    else:
        manifest_path = manifest.resolve(strict=True)
        project = PROJECT_ROOT.resolve(strict=True)
        if manifest_path != project and project not in manifest_path.parents:
            raise ColdStartInputError("FROZEN_MANIFEST_OUTSIDE_PROJECT")
        manifest_ref = file_ref(manifest_path)
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("schema_version") == "exact78-raw-t0-input-freeze-v1":
            if manifest_path != T0_FREEZE or manifest_ref["sha256"] != T0_FREEZE_SHA:
                raise ColdStartInputError("T0_FREEZE_IDENTITY_DRIFT")
            if file_ref(T0_CORRECTION)["sha256"] != T0_CORRECTION_SHA:
                raise ColdStartInputError("T0_CORRECTION_SHA_DRIFT")
            correction = json.loads(T0_CORRECTION.read_text(encoding="utf-8"))
            capability_ref = correction["outputs"]["corrected_capability"]
            if capability_ref["sha256"] != T0_CAPABILITY_SHA or file_ref(Path(capability_ref["path"])) != capability_ref:
                raise ColdStartInputError("T0_CAPABILITY_SHA_DRIFT")
            if correction["upstream_input_freeze"]["sha256"] != T0_FREEZE_SHA:
                raise ColdStartInputError("T0_CORRECTION_BINDING_DRIFT")
            capability = json.loads(Path(capability_ref["path"]).read_text(encoding="utf-8"))
            corrected_calibration = capability["calibration_states"]
            frozen_selection = value["selected"]
            sessions = [Path(row["source_path"]) for _, row in sorted(frozen_selection.items())]
        else:
            sessions = validate_manifest(value)
    assert sessions is not None
    sources = [inspect_session(path, allowed_root=allowed_root) for path in sessions]
    if frozen_selection is not None:
        if set(frozen_selection) != {source["session_id"] for source in sources}:
            raise ColdStartInputError("T0_SELECTED_SESSION_ID_DRIFT")
        for source in sources:
            selected = frozen_selection[source["session_id"]]
            source["selection_role"] = selected["role"]
            source["source_group"] = selected["source_group"]
            assert corrected_calibration is not None
            expected = ("ABSENT" if corrected_calibration[source["session_id"]]["extrinsics"] == "ABSENT"
                        else "SOURCE_PRESENT_QUALITY_UNVERIFIED")
            if source["metric_calibration"] != expected:
                raise ColdStartInputError("CORRECTED_CALIBRATION_CONFLICT")
    identities = [source["session_id"] for source in sources]
    if len(identities) != len(set(identities)):
        raise ColdStartInputError("DUPLICATE_SESSION_ID")
    evidence = {"schema_version": "exact78-submission-input-v1", "manifest": manifest_ref,
                "sources": sources}
    return {**evidence, "submission_input_sha256": hashlib.sha256(_canonical(evidence)).hexdigest(),
            "status": "CANDIDATE_REQUEST_ONLY", "training_eligible": False}


def _raw_session_path(session: Path) -> Path:
    """Normalize a CLI session without admitting aliases or derived roots."""
    if session.is_symlink():
        raise ColdStartInputError("SYMLINK_SESSION_FORBIDDEN")
    source = session.resolve(strict=True)
    allowed = DATA_ROOT.resolve(strict=True)
    if not source.is_dir() or not source.is_relative_to(allowed):
        raise ColdStartInputError("SOURCE_OUTSIDE_ALLOWED_ROOT")
    if any(part.lower() in FORBIDDEN_DERIVED_COMPONENTS
           for part in source.relative_to(allowed).parts):
        raise ColdStartInputError("OLD_DERIVED_SESSION_FORBIDDEN")
    return source


def _frozen_t0_selected_paths() -> frozenset[Path]:
    """Classify only by the SHA-pinned six raw source paths, never by ID alone."""
    if file_ref(T0_FREEZE)["sha256"] != T0_FREEZE_SHA:
        raise ColdStartInputError("T0_FREEZE_SHA_DRIFT")
    freeze = json.loads(T0_FREEZE.read_text(encoding="utf-8"))
    selected = freeze.get("selected") if isinstance(freeze, dict) else None
    if (not isinstance(freeze, dict)
            or freeze.get("schema_version") != "exact78-raw-t0-input-freeze-v1"
            or not isinstance(selected, dict) or set(selected) != T0_SELECTED_IDS):
        raise ColdStartInputError("T0_SELECTED_SET_DRIFT")
    paths = []
    for session_id, row in selected.items():
        raw = row.get("source_path") if isinstance(row, dict) else None
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ColdStartInputError(f"T0_SELECTED_PATH_INVALID:{session_id}")
        path = Path(raw)
        resolved = _raw_session_path(path)
        if str(resolved) != raw:
            raise ColdStartInputError(f"T0_SELECTED_PATH_ALIAS:{session_id}")
        paths.append(resolved)
    if len(paths) != len(set(paths)):
        raise ColdStartInputError("T0_SELECTED_PATH_DUPLICATE")
    return frozenset(paths)


def _attach_existing(request: dict[str, Any]) -> dict[str, Any]:
    index_path = PROJECT_ROOT / "tasks/current/INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    active = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(active) != 1:
        raise ColdStartInputError("CANDIDATE_PACKET_REQUIRED")
    packet_path = PROJECT_ROOT / active[0]["packet_path"]
    if file_ref(packet_path)["sha256"] != active[0]["packet_sha256"]:
        raise ColdStartInputError("ACTIVE_PACKET_SHA_DRIFT")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if (packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("inputs", {}).get("submission_input_sha256")
            != request["submission_input_sha256"]):
        raise ColdStartInputError("ACTIVE_PACKET_SIGNATURE_CONFLICT")
    task_id = packet.get("task_id")
    if (not isinstance(task_id, str) or active[0].get("task_id") != task_id
            or packet_path != PROJECT_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"):
        raise ColdStartInputError("ACTIVE_PACKET_IDENTITY_DRIFT")
    state = json.loads((PROJECT_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json").read_text(encoding="utf-8"))
    matches = [task for task in state.get("tasks", []) if task.get("task_id") == task_id]
    if (len(matches) != 1 or matches[0].get("status") != "RUNNING"
            or (state.get("next_task") or {}).get("task_id") != task_id
            or matches[0].get("run_signature") != packet.get("run_signature")
            or matches[0].get("pid") != (packet.get("writer") or {}).get("pid")
            or matches[0].get("proc_start_ticks") != (packet.get("writer") or {}).get("proc_start_ticks")
            or (matches[0].get("task_packet") or {}).get("sha256") != active[0]["packet_sha256"]):
        raise ColdStartInputError("ACTIVE_TASK_STATE_DRIFT")
    writer = packet.get("writer")
    attempts = [(Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw)
                for raw in packet.get("write_set", [])
                if Path(raw).name.startswith("attempt_")]
    if not isinstance(writer, dict) or len(attempts) != 1:
        raise ColdStartInputError("ACTIVE_WRITER_FENCE_ABSENT")
    attempt = attempts[0].resolve()
    if attempt.parent != PROJECT_ROOT / "_run/current" / task_id / "attempts":
        raise ColdStartInputError("ACTIVE_WRITER_FENCE_ABSENT")
    writer_file = json.loads((attempt / "WRITER.json").read_text(encoding="utf-8"))
    keys = ("pid", "proc_start_ticks", "executor_epoch", "fencing_token", "scope")
    if any(writer.get(key) != writer_file.get(key) for key in keys):
        raise ColdStartInputError("ACTIVE_WRITER_FENCE_DRIFT")
    pid, ticks = writer.get("pid"), writer.get("proc_start_ticks")
    if not isinstance(pid, int) or not isinstance(ticks, int) or pid < 2:
        raise ColdStartInputError("ACTIVE_WRITER_IDENTITY_INVALID")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        live_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise ColdStartInputError("ACTIVE_WRITER_NOT_LIVE") from exc
    if ticks != live_ticks:
        raise ColdStartInputError("ACTIVE_WRITER_NOT_LIVE")
    return {"status": "ATTACHED_TO_ACTIVE_CANDIDATE", "task_id": packet["task_id"],
            "submission_input_sha256": request["submission_input_sha256"],
            "task_packet": file_ref(packet_path), "execution_started_here": False,
            "visual_qualified": False, "training_eligible": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chaoyang submit exact78-raw-to-robot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", type=Path)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--preview", action="store_true", help="validate raw request without registering it")
    args = parser.parse_args(argv)
    if args.manifest is not None:
        # Manifest behavior is unchanged; a batch is never silently expanded
        # into the new one-session CPU admission producer.
        request = build_request(manifest=args.manifest)
        result = request if args.preview else _attach_existing(request)
    else:
        assert args.session is not None
        source = _raw_session_path(args.session)
        frozen_paths = _frozen_t0_selected_paths()
        if source in frozen_paths:
            request = build_request(session=source)
            result = request if args.preview else _attach_existing(request)
        else:
            if any(source.is_relative_to(path) for path in frozen_paths):
                raise ColdStartInputError("T0_SELECTED_CHILD_NOT_RAW_SESSION")
            from chaoyang.cli.exact78_newraw_cpu_submit_candidate import submit_one

            result = submit_one(source, preview=args.preview)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
