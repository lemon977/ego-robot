"""CPU-only handoff from terminal raw admission to the reusable HaWoR v3 core.

This module neither registers a task nor invokes ``extract_model_frames`` or
``run_hawor``. In particular, it never calls the T0-only v3 ``build_spec`` or
``main``. A separately registered, fenced GPU consumer is still required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chaoyang.governance.common import validate_artifact_ref
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    DATA_ROOT, PROJECT_ROOT, ColdStartInputError,
)
from chaoyang.governance.exact78_hawor_descriptor_adapter_candidate import (
    build_spec_from_sealed,
)


V3_MODEL_CORE = PROJECT_ROOT / "src/chaoyang/ops/run_exact78_hawor_raw_v3.py"
ADMISSION_ADAPTER = PROJECT_ROOT / "src/chaoyang/governance/exact78_hawor_descriptor_adapter_candidate.py"
HANDOFF_CODE = Path(__file__).resolve()
FORBIDDEN_INPUT_KEYS = frozenset({
    "t0_input_freeze", "t0_frame_time_map", "t0_correction", "t1a_result",
    "canonical_motion", "old_hawor", "prior_hawor", "accepted_template",
    "action_array", "motion_array", "hawor_npz", "hand_motion", "robot_motion",
})
FORBIDDEN_PATH_COMPONENTS = frozenset({
    "preprocess", "all_data", "trackingdata", "processed", "clean", "robot",
})
CORE_SPEC_FIELDS = frozenset({
    "source_video", "allowed_refs", "intrinsics", "fps_nominal", "frame_count",
    "crop_xywh", "time_mode", "physical_eye", "source_kind", "session_id",
    "frame_rows_sealed", "source_dimensions", "calibration_claim",
})


def _reject_legacy_inputs(packet: dict[str, Any]) -> None:
    inputs = packet.get("inputs")
    if not isinstance(inputs, dict):
        raise ColdStartInputError("CONSUMER_INPUTS_REQUIRED")
    forbidden = {str(key).lower() for key in inputs} & FORBIDDEN_INPUT_KEYS
    if forbidden:
        raise ColdStartInputError("OLD_MOTION_OR_T0_INPUT_FORBIDDEN:" + sorted(forbidden)[0])

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            path = value.get("path")
            if isinstance(path, str):
                parts = {part.lower() for part in Path(path).parts}
                if parts & FORBIDDEN_PATH_COMPONENTS:
                    raise ColdStartInputError("OLD_DERIVED_REFERENCE_FORBIDDEN")
                if any(part.startswith("exact78_raw_to_robot_t0_") for part in parts):
                    raise ColdStartInputError("T0_REFERENCE_FORBIDDEN")
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)

    inspect(inputs)
    read_set = packet.get("read_set", [])
    if not isinstance(read_set, list) or any(not isinstance(path, str) for path in read_set):
        raise ColdStartInputError("CONSUMER_READ_SET_INVALID")
    commands = packet.get("commands", [])
    if not isinstance(commands, list) or any(not isinstance(command, str) for command in commands):
        raise ColdStartInputError("CONSUMER_COMMANDS_INVALID")
    for path in [*read_set, *commands]:
        parts = {part.lower() for part in Path(path).parts}
        if (parts & FORBIDDEN_PATH_COMPONENTS
                or "exact78_raw_to_robot_t0_" in path.lower()
                or any(part.startswith("exact78_raw_to_robot_t0_") for part in parts)):
            raise ColdStartInputError("OLD_DERIVED_READ_SET_FORBIDDEN")


def _pinned_v3_core(packet: dict[str, Any]) -> dict[str, Any]:
    runtime = packet.get("runtime_code")
    if not isinstance(runtime, list):
        raise ColdStartInputError("CONSUMER_RUNTIME_CODE_REQUIRED")
    core_ref: dict[str, Any] | None = None
    for path, label in ((V3_MODEL_CORE, "V3_MODEL_CORE"),
                        (ADMISSION_ADAPTER, "ADMISSION_ADAPTER"),
                        (HANDOFF_CODE, "HANDOFF_CODE")):
        pinned = [ref for ref in runtime if isinstance(ref, dict)
                  and ref.get("path") == str(path)]
        if len(pinned) != 1:
            raise ColdStartInputError(f"{label}_SHA_PIN_REQUIRED")
        if errors := validate_artifact_ref(pinned[0]):
            raise ColdStartInputError(f"{label}_SHA_DRIFT:" + ";".join(errors))
        if path == V3_MODEL_CORE:
            core_ref = pinned[0]
    assert core_ref is not None
    return core_ref


def prepare_v3_handoff(
    *, packet: dict[str, Any], repo_root: Path = PROJECT_ROOT,
    allowed_root: Path = DATA_ROOT,
) -> dict[str, Any]:
    """Return a source-verified v3 core spec, not an executable GPU permit.

    The consumer packet must pin five references from one terminal producer.
    Only generic v3 ``extract_model_frames``, ``run_hawor`` and ``frame_status``
    may be reused by a future executor; v3 ``build_spec/main`` remain T0-only.
    """
    _reject_legacy_inputs(packet)
    code_ref = _pinned_v3_core(packet)
    inputs = packet["inputs"]
    spec = build_spec_from_sealed(
        packet=packet,
        descriptor_ref=inputs.get("upstream_admission_descriptor"),
        frame_map_ref=inputs.get("upstream_admission_frame_time_map"),
        seal_ref=inputs.get("upstream_admission_seal"),
        producer_result_ref=inputs.get("upstream_admission_result"),
        producer_receipt_ref=inputs.get("upstream_admission_receipt"),
        repo_root=repo_root, allowed_root=allowed_root,
    )
    if not CORE_SPEC_FIELDS.issubset(spec):
        raise ColdStartInputError("V3_MODEL_CORE_SPEC_INCOMPLETE")
    if (spec.get("frame_count") != len(spec["frame_rows_sealed"])
            or spec.get("native_open_audit_required") is not True
            or spec.get("world_pose_authority") is not False
            or spec.get("metric_authority") is not False):
        raise ColdStartInputError("V3_MODEL_CORE_CAPABILITY_DRIFT")
    if spec["source_kind"] == "vst0915" and spec["time_mode"] != "TIMEBASE_UNVERIFIED":
        raise ColdStartInputError("VST_TIMEBASE_UNVERIFIED_REQUIRED")
    return {
        "status": "CPU_HANDOFF_READY_GPU_RUNNER_NOT_REGISTERED",
        "v3_core_spec": spec,
        "v3_model_core": code_ref,
        "permitted_core_functions": ["extract_model_frames", "run_hawor", "frame_status"],
        "forbidden_v3_functions": ["build_spec", "main"],
        "frame_mapping": "SEALED_SAME_SESSION_DECODE_ORDER",
        "native_open_audit_required": True,
        "gpu_lease_verified": False,
        "model_invoked": False,
    }


__all__ = ["prepare_v3_handoff"]
