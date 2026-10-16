"""Raw-auto CPU admission and explicit V1-compatible full-flow request seam.

No algorithm/model is imported here.  The separately published operation calls
``operation_main`` only after its finite Task Packet and writer gate pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .contracts import canonical_sha, file_ref, verify_ref
from .source import RAW_FILES, inspect_raw_session

RECIPE = "inpainting-raw-to-robot"
REQUEST_SCHEMA = "inpainting-candidate-submission-v1"
MANIFEST_SCHEMA = "inpainting-raw-session-manifest-v1"
ALLOWED_RAW_ROOTS = (Path("/path/to/private_storage/datasets/ego"), Path("/path/to/private_data/egodata/datasets/ego"))
FORBIDDEN_COMPONENTS = frozenset({"_run", "review", "processed", "attempts", "archive"})


class EntryError(ValueError):
    pass


def raw_session_path(path: str | Path, *, allowed_roots=None) -> Path:
    raw = Path(path).absolute()
    root = raw.resolve(strict=True)
    if raw != root or not root.is_dir():
        raise EntryError("RAW_SESSION_ALIAS_OR_NOT_DIRECTORY")
    allowed = [Path(item).resolve(strict=True) for item in (allowed_roots if allowed_roots is not None else ALLOWED_RAW_ROOTS)]
    matches = [item for item in allowed if root != item and root.is_relative_to(item)]
    if not matches:
        raise EntryError("RAW_SESSION_OUTSIDE_ALLOWED_CAPTURE_ROOTS")
    relative = root.relative_to(max(matches, key=lambda item: len(item.parts)))
    if {part.lower() for part in relative.parts} & FORBIDDEN_COMPONENTS:
        raise EntryError("DERIVED_SESSION_FORBIDDEN")
    return root


def build_request(*, project_root: Path, session: Path | None = None,
                  manifest: Path | None = None, allowed_roots=None,
                  inspector: Callable[..., dict] = inspect_raw_session,
                  mode: str = "raw-auto", repair_rounds: int = 1) -> dict[str, Any]:
    """Bind actual raw bytes, never historical per-session derived artifacts.

    The CPU request signature is not a full model recipe signature.  Model
    weights and production configuration belong to the later finite GPU packet.
    """
    if mode == "v1-compatible":
        from .automation import build_v1_request
        return build_v1_request(project_root=project_root, session=session, manifest=manifest, repair_rounds=repair_rounds)
    if mode != "raw-auto":
        raise EntryError("INPAINTING_SUBMISSION_MODE_UNSUPPORTED")
    if type(repair_rounds) is not int or repair_rounds != 1:
        raise EntryError("REPAIR_ROUNDS_REQUIRE_V1_COMPATIBLE_MODE")
    if (session is None) == (manifest is None):
        raise EntryError("EXACTLY_ONE_SESSION_OR_MANIFEST_REQUIRED")
    project = Path(project_root).resolve(strict=True)
    manifest_ref = None
    if manifest is None:
        paths = [session]
    else:
        raw_manifest = manifest.absolute()
        frozen = raw_manifest.resolve(strict=True)
        if raw_manifest != frozen or not frozen.is_relative_to(project):
            raise EntryError("FROZEN_MANIFEST_MUST_BE_PROJECT_LOCAL_WITHOUT_ALIAS")
        value = json.loads(frozen.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or set(value) != {"schema", "sessions"}
                or value.get("schema") != MANIFEST_SCHEMA
                or not isinstance(value.get("sessions"), list) or not value["sessions"]
                or any(not isinstance(item, str) or not Path(item).is_absolute() for item in value["sessions"])):
            raise EntryError("RAW_MANIFEST_SCHEMA_UNSUPPORTED_OR_TASK_CONFIG_PRESENT")
        manifest_ref = file_ref(frozen)
        paths = [Path(item) for item in value["sessions"]]
    normalized = [raw_session_path(path, allowed_roots=allowed_roots) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise EntryError("DUPLICATE_RAW_SESSION_PATH")
    sources = [inspector(path, decode_probe=False) for path in normalized]
    for path, source in zip(normalized, sources):
        if source.get("session_dir") != str(path) or not source.get("actual_inputs"):
            raise EntryError("RAW_INSPECTION_IDENTITY_OR_INPUTS_INVALID")
        if any(source.get(key) is not False for key in
               ("aligned_rows_read", "processed_video_read", "hdf5_datasets_read", "control_ground_truth")):
            raise EntryError("RAW_INSPECTION_DERIVED_INPUT_OR_TRUTH_FORBIDDEN")
        for reference in source["actual_inputs"].values():
            if not Path(reference["path"]).resolve(strict=True).is_relative_to(path):
                raise EntryError("RAW_INSPECTION_INPUT_ESCAPES_SESSION")
            verify_ref(reference)
        if source.get("status") != "SOURCE_PREFLIGHT_CANDIDATE":
            raise EntryError("RAW_INPUT_INCOMPLETE:" + str(source.get("status")))
    identities = [source.get("session_id") for source in sources]
    if any(not isinstance(item, str) or not item for item in identities) or len(identities) != len(set(identities)):
        raise EntryError("RAW_SESSION_IDENTITY_EMPTY_OR_DUPLICATE")
    payload = {"schema": REQUEST_SCHEMA, "recipe": RECIPE, "manifest": manifest_ref,
               "sources": sources, "motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
               "producer_policy": "RAW_AND_GENERIC_ONLY", "manual_session_edits": 0}
    return {**payload, "submission_input_sha256": canonical_sha(payload),
            "status": "CPU_ADMISSION_REQUEST_ONLY", "robot_video_produced": False,
            "visual_qualified": False, "training_eligible": False, "deployable": False}


def verify_request(request: dict) -> None:
    payload = {key: request[key] for key in ("schema", "recipe", "manifest", "sources",
               "motion_source", "producer_policy", "manual_session_edits")}
    if (request.get("schema") != REQUEST_SCHEMA or request.get("recipe") != RECIPE
            or request.get("submission_input_sha256") != canonical_sha(payload)
            or request.get("motion_source") != "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS"
            or request.get("producer_policy") != "RAW_AND_GENERIC_ONLY"
            or request.get("manual_session_edits") != 0):
        raise EntryError("REQUEST_SIGNATURE_OR_PRODUCER_POLICY_DRIFT")
    if request.get("manifest"):
        verify_ref(request["manifest"])
    if not isinstance(request.get("sources"), list) or not request["sources"]:
        raise EntryError("REQUEST_SOURCES_EMPTY")
    identities = []
    paths = []
    for source in request["sources"]:
        root = raw_session_path(source["session_dir"])
        identities.append(source.get("session_id"))
        paths.append(root)
        if source.get("status") != "SOURCE_PREFLIGHT_CANDIDATE" or any(source.get(key) is not False for key in
               ("aligned_rows_read", "processed_video_read", "hdf5_datasets_read", "control_ground_truth")):
            raise EntryError("REQUEST_DERIVED_SOURCE_OR_STATUS_FORBIDDEN")
        if not isinstance(source.get("actual_inputs"), dict) or not source["actual_inputs"]:
            raise EntryError("REQUEST_ACTUAL_RAW_INPUTS_EMPTY")
        for role, reference in source["actual_inputs"].items():
            if role not in RAW_FILES or reference.get("path") != str(root / RAW_FILES[role]):
                raise EntryError("REQUEST_NON_RAW_ROLE_OR_FILE_FORBIDDEN")
            verify_ref(reference)
    if (len(paths) != len(set(paths)) or len(identities) != len(set(identities))
            or any(not isinstance(item, str) or not item for item in identities)):
        raise EntryError("REQUEST_SOURCE_IDENTITIES_EMPTY_OR_DUPLICATE")


def operation_main(argv: list[str] | None = None) -> int:
    """Export for the future registered operation; fail closed before any I/O.

    The operation wrapper/route must be published separately.  Calling this
    function directly cannot skip the exact command, PID/startticks or SHA gate.
    """
    parser = argparse.ArgumentParser(prog="inpainting-candidate-cpu-admission")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    from chaoyang.governance.inpainting_candidate_registration import (
        authorize_cpu_operation, write_once_json,
    )
    from chaoyang.paths import get_paths
    root = get_paths().repo_root
    packet = authorize_cpu_operation(list(argv if argv is not None else __import__("sys").argv[1:]), repo_root=root)
    if args.output_root.resolve() != Path(packet["output_dir"]) or args.request.resolve() != Path(packet["request_path"]):
        raise EntryError("CPU_OPERATION_REQUEST_OR_OUTPUT_DRIFT")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    verify_request(request)
    outputs = []
    for index, source in enumerate(request["sources"]):
        fresh = inspect_raw_session(source["session_dir"], decode_probe=True)
        if fresh["actual_inputs"] != source["actual_inputs"]:
            raise EntryError("SOURCE_SHA_DRIFT_AFTER_ADMISSION")
        destination = args.output_root / "source" / f"{index:04d}.json"
        write_once_json(destination, fresh)
        outputs.append(file_ref(destination))
    stage = {"schema": "inpainting-candidate-cpu-admission-stage-v1", "task_id": packet["task_id"],
             "run_signature": packet["run_signature"], "status": "SEALED_COMPLETE_CPU_ADMISSION",
             "sources": outputs, "models_called": 0, "robot_video_produced": False,
             "visual_qualified": False, "training_eligible": False, "deployable": False,
             "next_action": "Register the separately frozen automatic-Mask GPU experiment; CPU admission is not Robot production."}
    write_once_json(args.output_root / "ADMISSION_STAGE.json", stage)
    return 0
