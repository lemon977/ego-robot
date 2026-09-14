"""Fail-closed GPU admission for the archive/legacy/task_cards/26 39-frame replay.

This module does not import torch or call a model.  It verifies the immutable
archive/legacy/task_cards/26 CPU evidence, an independent P0=0 QA record, a byte-bound GPU
preparation freeze, and a canonical mainline GPU handoff.  Only then does it
atomically create a direct child of ``_run`` and its O_EXCL owner marker.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from pipeline.lr_distributed_side_evidence import (
    FIXED_FRAME_INDICES,
    FORBIDDEN_OVERRIDE_KEYS,
    FORBIDDEN_SESSION,
    MAX_OBJECT_OVERLAP_OVER_INSTANCE,
    MIN_JOINT_SUPPORT_RATIO,
    MIN_SIDE_MARGIN,
    ROUTE_OF_EVIDENCE,
    TASK26_RELATIVE_PATH,
    TASK26_SHA256,
    TEXT_PROMPT,
    WRONG_SIDE_SENTINELS,
    DistributedSideEvidenceError,
    EvidenceRef,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


GPU_ADMISSION_RELATIVE_PATH = "pipeline/lr_distributed_side_gpu_admission.py"
GPU_RUNNER_RELATIVE_PATH = "tools/run_lr_distributed_side_gpu.py"
GPU_TESTS_RELATIVE_PATH = "tests/test_lr_distributed_side_gpu_admission.py"
CPU_MODULE_RELATIVE_PATH = "pipeline/lr_distributed_side_evidence.py"
CPU_RUNNER_RELATIVE_PATH = "tools/run_lr_distributed_side_evidence_cpu.py"
CPU_TESTS_RELATIVE_PATH = "tests/test_lr_distributed_side_evidence.py"
CPU_FREEZE_RELATIVE_PATH = "archive/audits/lr_distributed_side_evidence_cpu_v1/CPU_FREEZE.json"
COVARIATES_RELATIVE_PATH = (
    "archive/audits/lr_distributed_side_evidence_cpu_v1/INPUT_COVARIATES_FREEZE.json"
)
CPU_RESULT_RELATIVE_PATH = "archive/audits/lr_distributed_side_evidence_cpu_v1/RESULT.json"
SOURCE_INPUT_RELATIVE_PATH = (
    "archive/audits/a_prime_per_side_gpu_freeze_20260828_"
    "lr_per_side_candidate_t1_20260828_v3/INPUT_MANIFEST.json"
)
GPU_PREP_RELATIVE_PATH = "archive/audits/lr_distributed_side_gpu_prep_v5/GPU_PREP_FREEZE.json"
CANONICAL_QA_RELATIVE_PATH = (
    "archive/audits/INDEPENDENT_QA_LR_DISTRIBUTED_SIDE_EVIDENCE_T1_V1.json"
)
CANONICAL_RELEASE_RELATIVE_PATH = (
    "archive/audits/user_authorizations/TASK26_DISTRIBUTED_SIDE_GPU_RELEASE_V4.json"
)
CANONICAL_AUTHORITY_SOURCE_RELATIVE_PATH = (
    "archive/audits/user_authorizations/TASK26_DISTRIBUTED_SIDE_GPU_AUTHORITY_SOURCE_V4.json"
)
MODEL_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
MODEL_BYTES = 3502755717
OFFICIAL_CODE_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
INDEPENDENT_QA_SHA256 = "e7980b1c2bfad873acc8bd26ca2369178ba84a972fcbe42a7ee75ea2bc8eb3a6"

FIXED_SHA256: Mapping[str, str] = {
    TASK26_RELATIVE_PATH: TASK26_SHA256,
    CPU_MODULE_RELATIVE_PATH:
        "bca51f866de925a2e795c5257540abc9c06fb8d298f26dd049196fda14fb9ff3",
    CPU_RUNNER_RELATIVE_PATH:
        "66f2cb4eaf3c280ad4aa2483cfef3453990c2dc9a9934cd72291276bc34dd25f",
    CPU_TESTS_RELATIVE_PATH:
        "a31f177698e818f7e69bec4962a4a55be4aa6b88175b2ac7c55ee30eb2962e25",
    CPU_FREEZE_RELATIVE_PATH:
        "74c759eb1f3942447fb0217c5559678aaa998f57cb76ecd4aca1d9835e80e380",
    COVARIATES_RELATIVE_PATH:
        "17e82d678584277b68d2f25e518ecb924a04489454e94b2fddc45d5a148c2ea4",
    CPU_RESULT_RELATIVE_PATH:
        "89eeee2cd54f429d13ee7f08530dfaec2b71910b207e140c82e3f6536b178d8d",
    SOURCE_INPUT_RELATIVE_PATH:
        "864c6ec1fa3dc8605a0e81620c7c90794f09f5446c0beba6cd52705122097e86",
    "tools/run_lr_per_side_identity_t1.py":
        "68ac4a1ceff7a7db7f2ec13b436105f59a6a8613ea5b971093f8229ac83bbd84",
    "tools/run_sam31_detailed_text_prompt_probe.py":
        "1b93039675aece7f651beb6138da293a675855720652e93dc32e69e9a0f63854",
    "archive/legacy/task_cards/14_PARALLEL_EXECUTION_PLAN.md":
        "c0e2d454d8e2f01ce6e2ebebebdb253fa7455dfff1de5c6b519d7681d896522c",
}

GPU_CONFIG: Mapping[str, Any] = {
    "schema_version": "a-prime-distributed-side-gpu-config-v1",
    "route_of_evidence": ROUTE_OF_EVIDENCE,
    "prompt": TEXT_PROMPT,
    "fixed_frame_indices": {
        session: list(indices) for session, indices in FIXED_FRAME_INDICES.items()
    },
    "wrist_supported_role": "DIAGNOSTIC_ONLY",
    "authority_wrist_outside_image_role": "DIAGNOSTIC_ONLY",
    "side_routing_basis": "IN_IMAGE_HAWOR_JOINT_SUPPORT_DISTRIBUTION_ONLY",
    "min_joint_support_ratio": MIN_JOINT_SUPPORT_RATIO,
    "min_side_margin": MIN_SIDE_MARGIN,
    "max_object_overlap_over_instance": MAX_OBJECT_OVERLAP_OVER_INSTANCE,
    "boundary_supported_role": "DIAGNOSTIC_ONLY",
    "object6d_role": "REJECTION_PROTECTION_ONLY",
    "formal_pixel_semantics": "COPY_ONE_UNMODIFIED_RAW_SAM_INSTANCE_PER_ACCEPTED_SIDE",
    "forbidden_pixel_operations": [
        "union",
        "fill",
        "crop",
        "hard_crop",
        "morphology",
        "object6d_subtraction",
    ],
    "wrong_side_sentinels": [
        {
            "session_id": session_id,
            "frame_index": frame_index,
            "raw_instance_offset": offset,
            "expected_side": side,
        }
        for session_id, frame_index, offset, side in WRONG_SIDE_SENTINELS
    ],
    "forbidden_session": FORBIDDEN_SESSION,
    "candidate_requires_human_review": True,
    "next_bucket_blocked": True,
    "advancement_authorized": False,
    "formal_consumer_allowed": False,
}


class DistributedSideGpuAdmissionError(DistributedSideEvidenceError):
    """Raised before any GPU/run mutation when archive/legacy/task_cards/26 admission is incomplete."""


def _json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributedSideGpuAdmissionError(f"{label} is not JSON") from exc
    if not isinstance(value, Mapping):
        raise DistributedSideGpuAdmissionError(f"{label} must be a JSON object")
    return value


def _strict_equal(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise DistributedSideGpuAdmissionError(f"{label} drift")


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_OVERRIDE_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def _canonical_relative(ref: EvidenceRef, project_root: Path, relative: str) -> None:
    expected = project_root / relative
    observed = Path(ref.path)
    if not observed.is_absolute():
        observed = project_root / observed
    if Path(os.path.abspath(observed)) != Path(os.path.abspath(expected)):
        raise DistributedSideGpuAdmissionError(f"canonical path mismatch: {relative}")


def evidence_ref(project_root: Path, relative: str, source_kind: str) -> EvidenceRef:
    path = project_root / relative
    if path.is_symlink() or not path.is_file():
        raise DistributedSideGpuAdmissionError(f"missing regular evidence: {relative}")
    return EvidenceRef(str(path), path.stat().st_size, sha256_file(path), source_kind)


def _as_ref(record: Mapping[str, Any], label: str) -> EvidenceRef:
    if set(record) != {"path", "bytes", "sha256", "source_kind"}:
        raise DistributedSideGpuAdmissionError(f"{label} ref schema drift")
    if (
        not isinstance(record["path"], str)
        or type(record["bytes"]) is not int
        or record["bytes"] <= 0
        or not isinstance(record["sha256"], str)
        or len(record["sha256"]) != 64
        or not isinstance(record["source_kind"], str)
        or not record["source_kind"]
    ):
        raise DistributedSideGpuAdmissionError(f"{label} ref invalid")
    return EvidenceRef(
        record["path"], record["bytes"], record["sha256"], record["source_kind"]
    )


def _verify_fixed_sources(project_root: Path) -> dict[str, EvidenceRef]:
    refs: dict[str, EvidenceRef] = {}
    for relative, expected_sha in FIXED_SHA256.items():
        ref = evidence_ref(project_root, relative, "TASK26_FIXED_SOURCE")
        if ref.sha256 != expected_sha:
            raise DistributedSideGpuAdmissionError(f"fixed SHA drift: {relative}")
        refs[relative] = ref
    return refs


def _verify_prep(
    payload: bytes,
    *,
    project_root: Path,
    prep_ref: EvidenceRef,
    fixed_refs: Mapping[str, EvidenceRef],
) -> Mapping[str, Any]:
    prep = _json_object(payload, "GPU prep freeze")
    expected_keys = {
        "schema_version",
        "document_status",
        "task26_sha256",
        "gpu_execution_authorized",
        "gpu_started",
        "labels_read",
        "blind_scores_read",
        "grap_a_cap_025_frames_read",
        "fixed_source_refs",
        "live_gpu_code_refs",
        "gpu_config",
        "model_asset",
        "official_code",
        "python_runtime",
    }
    if set(prep) != expected_keys:
        raise DistributedSideGpuAdmissionError("GPU prep freeze schema drift")
    expected_header = {
        "schema_version": "a-prime-distributed-side-gpu-prep-freeze-v1",
        "document_status": "PREPARED_CPU_ONLY_AWAITING_INDEPENDENT_QA_AND_GPU_HANDOFF",
        "task26_sha256": TASK26_SHA256,
        "gpu_execution_authorized": False,
        "gpu_started": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
    }
    for key, expected in expected_header.items():
        _strict_equal(prep.get(key), expected, f"GPU prep {key}")
    if prep.get("gpu_config") != GPU_CONFIG:
        raise DistributedSideGpuAdmissionError("GPU prep config drift")
    if _find_forbidden_keys(prep):
        raise DistributedSideGpuAdmissionError("GPU prep contains forbidden overrides")
    expected_fixed = {
        relative: asdict(ref) for relative, ref in sorted(fixed_refs.items())
    }
    if prep.get("fixed_source_refs") != expected_fixed:
        raise DistributedSideGpuAdmissionError("GPU prep fixed-source binding drift")
    live_refs = prep.get("live_gpu_code_refs")
    if not isinstance(live_refs, Mapping) or set(live_refs) != {
        GPU_ADMISSION_RELATIVE_PATH,
        GPU_RUNNER_RELATIVE_PATH,
        GPU_TESTS_RELATIVE_PATH,
    }:
        raise DistributedSideGpuAdmissionError("GPU prep live-code ref set drift")
    for relative, record in live_refs.items():
        ref = _as_ref(record, f"live code {relative}")
        _canonical_relative(ref, project_root, relative)
        ref.read_verified(allowed_root=project_root)
        if ref.sha256 != sha256_file(project_root / relative):
            raise DistributedSideGpuAdmissionError(f"GPU live-code SHA drift: {relative}")
    for field, expected_identity, expected_bytes in (
        (
            "model_asset",
            MODEL_SHA256,
            MODEL_BYTES,
        ),
        (
            "official_code",
            OFFICIAL_CODE_COMMIT,
            None,
        ),
    ):
        value = prep.get(field)
        if not isinstance(value, Mapping):
            raise DistributedSideGpuAdmissionError(f"GPU prep {field} invalid")
        identity_key = "sha256" if field == "model_asset" else "commit"
        if value.get(identity_key) != expected_identity:
            raise DistributedSideGpuAdmissionError(f"GPU prep {field} identity drift")
        if expected_bytes is not None and value.get("bytes") != expected_bytes:
            raise DistributedSideGpuAdmissionError(f"GPU prep {field} byte count drift")
    model = prep["model_asset"]
    if set(model) != {"path", "bytes", "sha256"}:
        raise DistributedSideGpuAdmissionError("GPU prep model schema drift")
    model_path = Path(model["path"])
    if model_path.is_symlink() or not model_path.is_file():
        raise DistributedSideGpuAdmissionError("GPU model is not a regular file")
    if model_path.stat().st_size != MODEL_BYTES or sha256_file(model_path) != MODEL_SHA256:
        raise DistributedSideGpuAdmissionError("GPU model live identity drift")
    official = prep["official_code"]
    if set(official) != {"path", "commit"}:
        raise DistributedSideGpuAdmissionError("GPU prep official-code schema drift")
    code_path = Path(official["path"])
    if code_path.is_symlink() or not code_path.is_dir():
        raise DistributedSideGpuAdmissionError("official code root invalid")
    observed_commit = subprocess.run(
        ["git", "-C", str(code_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != OFFICIAL_CODE_COMMIT:
        raise DistributedSideGpuAdmissionError("official code live commit drift")
    runtime = prep["python_runtime"]
    if set(runtime) != {"path", "bytes", "sha256", "version", "torch", "cuda"}:
        raise DistributedSideGpuAdmissionError("GPU prep Python runtime schema drift")
    runtime_path = Path(runtime["path"])
    if runtime_path.is_symlink() or not runtime_path.is_file():
        raise DistributedSideGpuAdmissionError("GPU Python runtime invalid")
    if (
        runtime_path.stat().st_size != runtime["bytes"]
        or sha256_file(runtime_path) != runtime["sha256"]
    ):
        raise DistributedSideGpuAdmissionError("GPU Python runtime identity drift")
    runtime_probe = json.loads(
        subprocess.run(
            [
                str(runtime_path),
                "-c",
                "import json,sys,torch,iopath;print(json.dumps({'version':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if runtime_probe != {
        "version": runtime["version"],
        "torch": runtime["torch"],
        "cuda": runtime["cuda"],
    }:
        raise DistributedSideGpuAdmissionError("GPU Python runtime probe drift")
    _canonical_relative(prep_ref, project_root, GPU_PREP_RELATIVE_PATH)
    return prep


def _verify_independent_qa(
    payload: bytes,
    *,
    project_root: Path,
    qa_ref: EvidenceRef,
) -> Mapping[str, Any]:
    qa = _json_object(payload, "independent QA")
    required = {
        "schema_version": "a-prime-distributed-side-independent-qa-v1",
        "producer": "independent_cpu_qa",
        "qa_scope": "TASK26_CPU_EVIDENCE_AND_GPU_ADMISSION_PREREQUISITE_ONLY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "status": "PASS_CPU_ADMISSION_EXACT",
        "p0_findings": 0,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "gpu_started": False,
    }
    for key, expected in required.items():
        _strict_equal(qa.get(key), expected, f"independent QA {key}")
    # The exact QA SHA is authority-bound below.  Its verification section
    # intentionally contains zero-valued operation keys such as ``morphology``;
    # those are audit counters, not caller override configuration.
    if qa_ref.sha256 != INDEPENDENT_QA_SHA256:
        raise DistributedSideGpuAdmissionError("independent QA SHA mismatch")
    expected_ref_shas = {
        "task26": FIXED_SHA256[TASK26_RELATIVE_PATH],
        "module": FIXED_SHA256[CPU_MODULE_RELATIVE_PATH],
        "runner": FIXED_SHA256[CPU_RUNNER_RELATIVE_PATH],
        "tests": FIXED_SHA256[CPU_TESTS_RELATIVE_PATH],
        "cpu_freeze": FIXED_SHA256[CPU_FREEZE_RELATIVE_PATH],
        "input_covariates": FIXED_SHA256[COVARIATES_RELATIVE_PATH],
        "result": FIXED_SHA256[CPU_RESULT_RELATIVE_PATH],
    }
    frozen_refs = qa.get("frozen_refs")
    if not isinstance(frozen_refs, Mapping):
        raise DistributedSideGpuAdmissionError("independent QA frozen refs missing")
    for name, expected_sha in expected_ref_shas.items():
        record = frozen_refs.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != expected_sha:
            raise DistributedSideGpuAdmissionError(f"independent QA ref drift: {name}")
    wrong_side = qa.get("verification", {}).get("wrong_side_gate", {})
    expected_wrong_side = {
        "diagnostic_independent_counterexample":
            "FAIL_WRONG_SIDE_ACCEPTANCE_HAS_WHOLE_RUN_PRIORITY",
        "grap_a_cap_005_220_offset_1": "RIGHT_ONLY_PASS",
        "grap_a_cap_005_270_offset_1": "RIGHT_ONLY_PASS",
        "wrong_side_failed": False,
    }
    if wrong_side != expected_wrong_side:
        raise DistributedSideGpuAdmissionError("independent QA wrong-side gate mismatch")
    if qa.get("claim_limits", {}).get("gpu_execution_authorized_by_this_qa") is not False:
        raise DistributedSideGpuAdmissionError("independent QA claim limit drift")
    _canonical_relative(qa_ref, project_root, CANONICAL_QA_RELATIVE_PATH)
    return qa


def _verify_baseline_handoff(
    release: Mapping[str, Any],
    *,
    project_root: Path,
) -> None:
    state = release.get("task27_baseline_state")
    handoff_record = release.get("task27_handoff_ref")
    if state == "NOT_STARTED_NO_GPU_OWNERSHIP":
        if handoff_record is not None:
            raise DistributedSideGpuAdmissionError("unexpected task27 handoff ref")
        return
    if state != "PREEMPTED_AT_FRAME_BOUNDARY_GPU_RELEASED":
        raise DistributedSideGpuAdmissionError("task27 baseline handoff state invalid")
    if not isinstance(handoff_record, Mapping):
        raise DistributedSideGpuAdmissionError("task27 preemption ref missing")
    ref = _as_ref(handoff_record, "task27 handoff")
    data = ref.read_verified(allowed_root=project_root)
    record = _json_object(data, "task27 preemption record")
    required = {
        "schema_version": "task27-task26-preemption-v1",
        "status": "PREEMPTED_HIGHER_PRIORITY_TASK26_AT_FRAME_BOUNDARY",
        "partial_artifacts_preserved": True,
        "gpu_released": True,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
    }
    for key, expected in required.items():
        _strict_equal(record.get(key), expected, f"task27 handoff {key}")


def _verify_release(
    payload: bytes,
    *,
    project_root: Path,
    release_ref: EvidenceRef,
    qa_ref: EvidenceRef,
    prep_ref: EvidenceRef,
) -> Mapping[str, Any]:
    release = _json_object(payload, "mainline GPU release")
    required = {
        "schema_version": "a-prime-distributed-side-gpu-release-v1",
        "authorization_scope": "TASK26_FIXED_39_FRAME_GPU_REPLAY_ONLY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "release_written_by": "ROOT_PROJECT_MAINLINE_AFTER_QA",
        "gpu_release_authorized": True,
        "gpu_idle_verified_after_handoff": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
    }
    for key, expected in required.items():
        _strict_equal(release.get(key), expected, f"mainline release {key}")
    if release.get("independent_qa_ref") != asdict(qa_ref):
        raise DistributedSideGpuAdmissionError("mainline release QA binding drift")
    if release.get("gpu_prep_freeze_ref") != asdict(prep_ref):
        raise DistributedSideGpuAdmissionError("mainline release prep binding drift")
    source_record = release.get("authorization_source_ref")
    if not isinstance(source_record, Mapping):
        raise DistributedSideGpuAdmissionError("mainline authorization source missing")
    source_ref = _as_ref(source_record, "authorization source")
    _canonical_relative(
        source_ref, project_root, CANONICAL_AUTHORITY_SOURCE_RELATIVE_PATH
    )
    source = _json_object(
        source_ref.read_verified(allowed_root=project_root), "authorization source"
    )
    expected_source = {
        "schema_version": "task26-distributed-side-gpu-authority-source-v1",
        "record_type": "AUTHORIZATION_SOURCE_NOT_EXECUTION_RELEASE",
        "user_authorization_original":
            "可以全部都推进，我都同意，以任务进行为最高优先",
        "task_scope": "archive/legacy/task_cards/26 fixed 39-frame A-prime replay after independent P0=0 QA",
        "semantic_scope": "S1 wrist diagnostic-only plus S2 OOB wrist diagnostic-only; thresholds .20/.05/.12 unchanged",
        "governance_scope": "candidate-only new _run; no labels/blind/025/CLEAN/full460/promotion",
        "task26_sha256": TASK26_SHA256,
        "independent_qa_ref": asdict(qa_ref),
    }
    if source != expected_source:
        raise DistributedSideGpuAdmissionError("authorization source payload drift")
    if _find_forbidden_keys(release):
        raise DistributedSideGpuAdmissionError("mainline release contains forbidden overrides")
    _verify_baseline_handoff(release, project_root=project_root)
    _canonical_relative(release_ref, project_root, CANONICAL_RELEASE_RELATIVE_PATH)
    return release


def require_distributed_side_gpu_admission(
    *,
    project_root: Path,
    project_run_root: Path,
    run_root: Path,
    prep_ref: EvidenceRef,
    independent_qa_ref: EvidenceRef,
    owner_release_ref: EvidenceRef,
) -> EvidenceRef:
    """Verify every external gate, then create the run/OWNER using O_EXCL."""

    project = project_root.resolve(strict=True)
    expected_runs = project / "_run"
    if Path(os.path.abspath(project_run_root)) != Path(os.path.abspath(expected_runs)):
        raise DistributedSideGpuAdmissionError("project run root is not canonical _run")
    if project_run_root.is_symlink() or not project_run_root.is_dir():
        raise DistributedSideGpuAdmissionError("canonical _run is not a regular directory")

    fixed_refs = _verify_fixed_sources(project)
    prep_payload = prep_ref.read_verified(allowed_root=project)
    _verify_prep(
        prep_payload,
        project_root=project,
        prep_ref=prep_ref,
        fixed_refs=fixed_refs,
    )
    qa_payload = independent_qa_ref.read_verified(allowed_root=project)
    _verify_independent_qa(
        qa_payload,
        project_root=project,
        qa_ref=independent_qa_ref,
    )
    release_payload = owner_release_ref.read_verified(allowed_root=project)
    _verify_release(
        release_payload,
        project_root=project,
        release_ref=owner_release_ref,
        qa_ref=independent_qa_ref,
        prep_ref=prep_ref,
    )

    run_parent = project_run_root.resolve(strict=True)
    if Path(os.path.abspath(run_root.parent)) != Path(os.path.abspath(run_parent)):
        raise DistributedSideGpuAdmissionError("GPU run must be direct child of _run")
    if not run_root.name.startswith("lr_distributed_side_candidate_t1_"):
        raise DistributedSideGpuAdmissionError("GPU run name outside task26 namespace")
    parent_fd = os.open(
        run_parent,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        try:
            os.mkdir(run_root.name, 0o750, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise DistributedSideGpuAdmissionError("GPU run already exists; O_EXCL required") from exc
        run_fd = os.open(
            run_root.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    owner_payload = canonical_json(
        {
            "schema_version": "a-prime-distributed-side-gpu-owner-v1",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "task26_sha256": TASK26_SHA256,
            "prep_ref": asdict(prep_ref),
            "independent_qa_ref": asdict(independent_qa_ref),
            "owner_release_ref": asdict(owner_release_ref),
            "candidate_requires_human_review": True,
            "next_bucket_blocked": True,
            "advancement_authorized": False,
            "formal_consumer_allowed": False,
        }
    )
    marker = run_root / "OWNER.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open("OWNER.json", flags, 0o440, dir_fd=run_fd)
        written = 0
        while written < len(owner_payload):
            written += os.write(descriptor, owner_payload[written:])
        os.fsync(descriptor)
    except OSError as exc:
        raise DistributedSideGpuAdmissionError("GPU owner O_EXCL creation failed") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
        os.close(run_fd)
    return EvidenceRef(
        str(marker),
        len(owner_payload),
        sha256_bytes(owner_payload),
        "A_PRIME_DISTRIBUTED_SIDE_GPU_OWNER",
    )
