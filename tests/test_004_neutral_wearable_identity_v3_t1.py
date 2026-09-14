from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neutral_wearable_v2", ROOT / "pipeline/neutral_wearable_identity_v2.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


RUNNER_SPEC = importlib.util.spec_from_file_location(
    "task32_runner_v3", ROOT / "tools/run_004_neutral_wearable_identity_v3_t1.py"
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def contract_pair():
    path = ROOT / "contracts/neutral_wearable_prompts_v1.json"
    payload = path.read_bytes()
    ref = mod.EvidenceRef(
        str(path),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        "FROZEN_NEUTRAL_WEARABLE_PROMPT_CONTRACT",
    )
    return ref, mod.verify_prompt_contract(ref, ROOT)


def hawor_ref(kind: str):
    path = ROOT / "contracts/neutral_wearable_prompts_v1.json"
    payload = path.read_bytes()
    return mod.EvidenceRef(
        str(path),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        kind,
    )


def raw(offset: int, mask: np.ndarray, score: float = 0.9, prompt_id: str = "W1"):
    ref, _ = contract_pair()
    return mod.RawWearableInstance(
        prompt_id,
        mod.FROZEN_PROMPTS[prompt_id],
        ref,
        0,
        offset,
        10 + offset,
        score,
        mask,
        mod.mask_sha256(mask),
    )


def arm_identity(
    root: Path,
    mask: np.ndarray,
    *,
    session: str = "grap_a_cap_004",
    side: str = "right",
    frame: int = 0,
    offset: int = 2,
    instance_id: int = 12,
):
    evidence = {
        "frame_id": f"{session}:{frame}",
        "instance_id": instance_id,
        "mask_sha256": mod.task27_raw_mask_sha256(mask),
        "source_kind": "SAM_RAW_INSTANCE",
    }
    payload = json.dumps(evidence, separators=(",", ":"), sort_keys=True).encode()
    path = root / f"frame_{frame:05d}_offset_{offset:03d}.json"
    path.write_bytes(payload)
    ref = mod.EvidenceRef(
        str(path),
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        "TASK27_SELECTED_ARM_RAW_EVIDENCE_JSON",
    )
    return mod.AcceptedArmRawIdentity(
        side=side,
        session_id=session,
        frame_id=f"{session}:{frame}",
        frame_index=frame,
        status="ACCEPT",
        selected_offset=offset,
        selected_instance_id=instance_id,
        selected_raw_evidence_ref=ref,
        selected_raw_evidence_bytes=payload,
        selected_candidate_eligible=True,
        selected_candidate_rejection_reasons=tuple(),
        raw_mask_sha256=mod.task27_raw_mask_sha256(mask),
    )


def authorities():
    optimized = hawor_ref("HAWOR_OPTIMIZED_JOINTS_NPZ")
    raw_source = hawor_ref("HAWOR_RAW_PROJECTION_NPZ")
    return {
        "left": mod.WristAuthority(
            "left", 0, (3.0, 4.0), 4.0, "slot-0", 0, 0, optimized, raw_source
        ),
        "right": mod.WristAuthority(
            "right", 0, (16.0, 4.0), 4.0, "slot-1", 1, 1, optimized, raw_source
        ),
    }


def select(instances, auth, shape):
    return mod.select_wearable_per_side(
        instances, auth, shape, prompt_contract=contract_pair()[1]
    )


def test_routes_raw_instances_independently_by_side():
    left = np.zeros((12, 20), bool)
    left[3:6, 2:5] = True
    right = np.zeros((12, 20), bool)
    right[3:6, 15:18] = True
    decisions = select([raw(0, left), raw(1, right)], authorities(), left.shape)
    assert decisions["left"].selected_offset == 0
    assert decisions["right"].selected_offset == 1


def test_supporting_both_sides_is_not_accepted_as_both():
    mask = np.zeros((12, 20), bool)
    mask[3:6, 2:18] = True
    decisions = select([raw(0, mask)], authorities(), mask.shape)
    assert all(value.status == "HOLD" for value in decisions.values())


def test_missing_one_side_does_not_abort_other_side():
    mask = np.zeros((12, 20), bool)
    mask[3:6, 15:18] = True
    auth = authorities()
    auth["left"] = mod.WristAuthority(
        "left",
        0,
        None,
        None,
        "slot-0",
        0,
        0,
        hawor_ref("HAWOR_OPTIMIZED_JOINTS_NPZ"),
        hawor_ref("HAWOR_RAW_PROJECTION_NPZ"),
    )
    decisions = select([raw(0, mask)], auth, mask.shape)
    assert decisions["left"].status == "HOLD"
    assert decisions["right"].status == "ACCEPT"


def test_low_score_is_audited_not_filled():
    mask = np.zeros((12, 20), bool)
    mask[3:6, 2:5] = True
    decisions = select([raw(0, mask, 0.49)], authorities(), mask.shape)
    assert (
        "MODEL_SCORE_BELOW_FROZEN_THRESHOLD"
        in decisions["left"].audits[0].rejection_reasons
    )
    assert decisions["left"].status == "HOLD"


def test_identity_union_is_bit_exact_or_only_and_decision_bound(tmp_path: Path):
    arm = np.zeros((12, 20), bool)
    arm[0, 0] = True
    wearable = np.zeros((12, 20), bool)
    wearable[3:6, 15:18] = True
    instance = raw(0, wearable)
    decision = select([instance], authorities(), wearable.shape)["right"]
    result = mod.identity_union(
        arm,
        wearable,
        side="right",
        arm_identity=arm_identity(tmp_path, arm),
        wearable_decision=decision,
        wearable_instance=instance,
    )
    assert np.array_equal(result, np.logical_or(arm, wearable))


def test_arm_identity_join_is_session_generic(tmp_path: Path):
    arm = np.zeros((12, 20), bool)
    arm[0, 0] = True
    wearable = np.zeros((12, 20), bool)
    wearable[3:6, 15:18] = True
    instance = raw(0, wearable)
    decision = select([instance], authorities(), wearable.shape)["right"]
    result = mod.identity_union(
        arm,
        wearable,
        side="right",
        arm_identity=arm_identity(
            tmp_path,
            arm,
            session="grap_a_cap_012",
        ),
        wearable_decision=decision,
        wearable_instance=instance,
    )
    assert np.array_equal(result, np.logical_or(arm, wearable))


def test_arm_identity_join_rejects_session_alias_and_forbidden_025(tmp_path: Path):
    arm = np.zeros((12, 20), bool)
    arm[0, 0] = True
    wearable = np.zeros((12, 20), bool)
    wearable[3:6, 15:18] = True
    instance = raw(0, wearable)
    decision = select([instance], authorities(), wearable.shape)["right"]
    typed = arm_identity(tmp_path, arm, session="grap_a_cap_012")
    with pytest.raises(mod.WearableIdentityError, match="frame identity"):
        mod.identity_union(
            arm,
            wearable,
            side="right",
            arm_identity=replace(typed, session_id="grap_a_cap_004"),
            wearable_decision=decision,
            wearable_instance=instance,
        )
    for offset, forbidden_session in enumerate(
        (
            "grap_a_cap_025",
            "025",
            "grap_a_cap_025/",
            "grap_a_cap_004/../grap_a_cap_025",
        ),
        start=3,
    ):
        forbidden = arm_identity(
            tmp_path,
            arm,
            session=forbidden_session,
            offset=offset,
        )
        with pytest.raises(mod.WearableIdentityError, match="session identity"):
            mod.identity_union(
                arm,
                wearable,
                side="right",
                arm_identity=forbidden,
                wearable_decision=decision,
                wearable_instance=instance,
            )


def test_cross_prompt_pool_is_rejected():
    mask = np.zeros((12, 20), bool)
    with pytest.raises(mod.WearableIdentityError, match="cross-prompt"):
        select(
            [raw(0, mask, prompt_id="W1"), raw(1, mask, prompt_id="W2")],
            authorities(),
            mask.shape,
        )


def test_side_authority_alias_is_rejected():
    auth = authorities()
    left = auth["left"]
    auth["right"] = mod.WristAuthority(
        "right",
        0,
        (16.0, 4.0),
        4.0,
        left.lineage_id,
        0,
        0,
        left.optimized_source_ref,
        left.raw_hawor_source_ref,
    )
    with pytest.raises(mod.WearableIdentityError, match="alias"):
        select([], auth, (12, 20))


def test_union_rejects_unselected_instance_and_nonaccepted_arm(tmp_path: Path):
    mask = np.zeros((12, 20), bool)
    mask[3:6, 2:5] = True
    selected = raw(0, mask)
    decision = select([selected], authorities(), mask.shape)["left"]
    wrong = raw(1, mask)
    with pytest.raises(mod.WearableIdentityError, match="accepted decision"):
        mod.identity_union(
            mask,
            wrong.mask,
            side="left",
            arm_identity=arm_identity(tmp_path, mask, side="left"),
            wearable_decision=decision,
            wearable_instance=wrong,
        )
    with pytest.raises(mod.WearableIdentityError, match="arm identity"):
        held = replace(arm_identity(tmp_path, mask, side="left"), status="HOLD")
        mod.identity_union(
            mask,
            selected.mask,
            side="left",
            arm_identity=held,
            wearable_decision=decision,
            wearable_instance=selected,
        )


def test_prompt_contract_is_exact_neutral_and_four_are_mandatory():
    path = ROOT / "contracts/neutral_wearable_prompts_v1.json"
    payload = path.read_bytes()
    contract = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == mod.PROMPT_CONTRACT_SHA256
    assert contract["status"] == "FROZEN_BEFORE_INFERENCE"
    assert contract["prompt_count"] == 4 == len(contract["prompts"])
    assert (
        contract["execution_rule"]
        == "EXECUTE_AND_PERSIST_ALL_PROMPTS_NO_RESULT_DRIVEN_CHOICE"
    )
    forbidden = (
        "white",
        "black",
        "red",
        "blue",
        "plastic",
        "metal",
        "004",
        "left",
        "right",
    )
    assert not any(
        word in row["text"].lower() for row in contract["prompts"] for word in forbidden
    )


def test_contract_ref_replace_and_symlink_are_rejected(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "contract.json"
    source.write_bytes(
        (ROOT / "contracts/neutral_wearable_prompts_v1.json").read_bytes()
    )
    ref = mod.EvidenceRef(
        str(source),
        source.stat().st_size,
        mod.PROMPT_CONTRACT_SHA256,
        "FROZEN_NEUTRAL_WEARABLE_PROMPT_CONTRACT",
    )
    source.write_text("{}")
    with pytest.raises(mod.WearableIdentityError, match="bytes/SHA"):
        mod.verify_prompt_contract(ref, root)
    source.unlink()
    source.symlink_to(ROOT / "contracts/neutral_wearable_prompts_v1.json")
    with pytest.raises(mod.WearableIdentityError, match="nofollow"):
        mod.verify_prompt_contract(ref, root)


def npz_bytes(value: int) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, sentinel=np.asarray([value], dtype=np.int64))
    return stream.getvalue()


@pytest.mark.parametrize("filename", ["joint_optimized_3d.npz", "hawor_projection.npz"])
def test_joint_and_raw_hawor_replace_attack_consumes_verified_fd_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
):
    target = tmp_path / filename
    replacement = tmp_path / f"replacement-{filename}"
    payload_a, payload_b = npz_bytes(17), npz_bytes(99)
    target.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    original = runner.single_fd_bytes

    def replace_after_read(path: Path):
        payload, ref = original(path)
        os.replace(replacement, target)
        return payload, ref

    monkeypatch.setattr(runner, "single_fd_bytes", replace_after_read)
    arrays, ref = runner.verified_npz(target, hashlib.sha256(payload_a).hexdigest())
    assert arrays["sentinel"].tolist() == [17]
    assert ref["sha256"] == hashlib.sha256(payload_a).hexdigest()
    assert target.read_bytes() == payload_b
    assert ref["single_fd_identity_and_parse"] is True


def test_admission_hardblocks_before_run_creation_and_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    (tmp_path / "_run").mkdir()
    monkeypatch.setattr(runner, "frozen_input_refs", lambda: {"frozen": True})
    for bad in ("../escaped_task32", "/tmp/escaped", "must_not_exist", "x/y"):
        with pytest.raises(runner.Task32Error, match="run id"):
            runner.admit(bad)
    assert not (tmp_path / "escaped_task32").exists()
    with pytest.raises(runner.Task32Error):
        runner.admit(runner.TASK32_RUN_ID)
    assert not (tmp_path / "_run" / runner.TASK32_RUN_ID).exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("side", "left", "this side"),
        ("selected_offset", 3, "path/offset"),
        ("selected_instance_id", 99, "payload"),
        ("raw_mask_sha256", "f" * 64, "payload"),
        ("selected_candidate_eligible", False, "decision fields"),
    ],
)
def test_arm_join_rejects_wrong_side_offset_id_mask_and_eligibility(
    tmp_path: Path, field: str, value: object, match: str
):
    arm = np.zeros((12, 20), bool)
    arm[0, 0] = True
    wearable = np.zeros((12, 20), bool)
    wearable[3:6, 15:18] = True
    instance = raw(0, wearable)
    decision = select([instance], authorities(), wearable.shape)["right"]
    typed = replace(arm_identity(tmp_path, arm), **{field: value})
    with pytest.raises(mod.WearableIdentityError, match=match):
        mod.identity_union(
            arm,
            wearable,
            side="right",
            arm_identity=typed,
            wearable_decision=decision,
            wearable_instance=instance,
        )


def test_arm_join_rejects_wrong_evidence_bytes_and_survives_path_replacement(
    tmp_path: Path,
):
    arm = np.zeros((12, 20), bool)
    arm[0, 0] = True
    wearable = np.zeros((12, 20), bool)
    wearable[3:6, 15:18] = True
    instance = raw(0, wearable)
    decision = select([instance], authorities(), wearable.shape)["right"]
    typed = arm_identity(tmp_path, arm)
    Path(typed.selected_raw_evidence_ref.path).write_text("{}")
    # Consumption is pinned to the already verified descriptor bytes, not a reopen.
    result = mod.identity_union(
        arm,
        wearable,
        side="right",
        arm_identity=typed,
        wearable_decision=decision,
        wearable_instance=instance,
    )
    assert np.array_equal(result, np.logical_or(arm, wearable))
    forged = replace(typed, selected_raw_evidence_bytes=b"{}")
    with pytest.raises(mod.WearableIdentityError, match="verified-byte"):
        mod.identity_union(
            arm,
            wearable,
            side="right",
            arm_identity=forged,
            wearable_decision=decision,
            wearable_instance=instance,
        )


def cpu_test_payload(refs: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "task32-b2-cpu-test-record-v1",
        "status": "PASS_CURRENT_BYTES_CPU_TESTS",
        "command": ["python3", "-m", "pytest", "tests/task32.py"],
        "environment": {"CUDA_VISIBLE_DEVICES": "", "gpu_processes_started": 0},
        "tests": {"collected": 51, "passed": 51, "failures": 0, "errors": 0},
        "junit_ref": refs["cpu_pytest_xml"],
        "bound_sources": {"frozen": refs["frozen"]},
        "real_gpu_canary_required_after_admit": True,
        "candidate_requires_human_review": True,
        "created_at": "2026-08-30T00:00:00+08:00",
    }


@pytest.mark.parametrize("attack", ["bool_count", "nonzero_error", "unknown_key"])
def test_governance_rejects_fake_schema_bool_and_failed_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
):
    refs = {
        "frozen": {"path": "/f", "bytes": 1, "sha256": "a" * 64},
        "cpu_pytest_xml": {"path": "/j", "bytes": 1, "sha256": "b" * 64},
    }
    record = cpu_test_payload(refs)
    if attack == "bool_count":
        record["tests"]["collected"] = True  # type: ignore[index]
    elif attack == "nonzero_error":
        record["tests"]["errors"] = 1  # type: ignore[index]
    else:
        record["attacker"] = True
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "CPU_TEST_RECORD_REL", Path("record.json"))
    monkeypatch.setattr(runner, "CPU_TEST_BOUND_SOURCE_KEYS", ("frozen",))
    monkeypatch.setattr(runner, "require_task29_release", lambda: {})
    with pytest.raises(runner.Task32Error):
        runner.require_governance(refs)


def test_governance_parses_the_same_verified_bytes_under_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    refs = {
        "frozen": {"path": "/f", "bytes": 1, "sha256": "a" * 64},
        "cpu_pytest_xml": {"path": "/j", "bytes": 1, "sha256": "b" * 64},
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(cpu_test_payload(refs)))
    original = runner.single_fd_bytes

    def replace_after_read(target: Path):
        payload, ref = original(target)
        target.write_text("{}")
        return payload, ref

    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "CPU_TEST_RECORD_REL", Path("record.json"))
    monkeypatch.setattr(runner, "CPU_TEST_BOUND_SOURCE_KEYS", ("frozen",))
    monkeypatch.setattr(runner, "single_fd_bytes", replace_after_read)
    monkeypatch.setattr(runner, "require_task29_release", lambda: {})
    value = runner.require_governance(refs)
    assert value["cpu_test_record"]["sha256"] != hashlib.sha256(b"{}").hexdigest()


def test_task29_release_requires_exact_ref_chain_and_integer_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run = tmp_path / "_run/p1_v3"
    component = run / "component"
    audits = tmp_path / "audits"
    component.mkdir(parents=True)
    audits.mkdir()
    admission_path = run / "GPU_ADMISSION.json"
    terminal_path = component / "RUN_MANIFEST.json"
    artifact_path = component / "ARTIFACT_MANIFEST.json"
    video_002_path = component / "RAW_OLD_TASK26_grap_a_cap_002_FULL.mp4"
    video_012_path = component / "RAW_OLD_TASK26_grap_a_cap_012_FULL.mp4"
    upstream_qa_path = audits / "qa.json"
    result_qa_path = audits / "result_qa.json"
    admission_path.write_text(
        json.dumps(
            {
                "actual_run_filesystem": {
                    "block_size": 4096,
                    "free_bytes": 0,
                    "st_dev": 102,
                    "st_ino": 321,
                },
                "context_ref": {"path": "/context", "bytes": 1, "sha256": "a" * 64},
                "created_at": "2026-08-28T00:00:00+08:00",
                "directory_identity": {"st_dev": 102, "st_ino": 321},
                "disk_probe_ref": {"path": "/probe", "bytes": 1, "sha256": "b" * 64},
                "governance": {},
                "gpu_started": False,
                "owner_ref": {"path": "/owner", "bytes": 1, "sha256": "c" * 64},
                "schema_version": "task29-gpu-admission-v2",
                "status": "PASS_EXACT_FREEZE_OEXCL_ACTUAL_FS_GPU_READY",
            }
        )
    )
    terminal_path.write_text(
        json.dumps({"status": "COMPLETED_UNREVIEWED_AWAITING_INDEPENDENT_QA"})
    )
    artifact_path.write_text(json.dumps({"schema_version": "task29-artifact-manifest-v2"}))
    video_002_path.write_bytes(b"video-002")
    video_012_path.write_bytes(b"video-012")
    upstream_qa_path.write_text(
        json.dumps(
            {
                "producer": "INDEPENDENT_QA",
                "status": "PASS_TASK29_V3_CPU_QA_P0_ZERO_GPU_ADMISSION_ALLOWED",
            }
        )
    )
    admission_ref = runner.regular_ref(admission_path)
    terminal_ref = runner.regular_ref(terminal_path)
    release = {
        "schema_version": "task29-gpu-release-v3",
        "status": "GPU_RELEASED_AFTER_SUCCESS",
        "run_id": "p1_v3",
        "launcher_pid": 123,
        "gpu_child_has_exited": True,
        "release_observation_phase": "OUTER_LAUNCHER_AFTER_GPU_CHILD_EXIT",
        "exit_code": 0,
        "gpu_processes_remaining": 0,
        "gpu_processes": [],
        "nvidia_smi_command": "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits",
        "gpu_admission": {key: admission_ref[key] for key in ("path", "bytes", "sha256")},
        "terminal_kind": "RUN_MANIFEST",
        "terminal": {key: terminal_ref[key] for key in ("path", "bytes", "sha256")},
        "terminal_status": "COMPLETED_UNREVIEWED_AWAITING_INDEPENDENT_QA",
        "successful_p1_terminal": True,
        "unlocking_p2": True,
        "candidate_requires_human_review": True,
        "advancement_authorized": False,
        "created_at": "2026-08-28T00:00:00+08:00",
    }
    release_path = run / "GPU_RELEASE.json"
    release_path.write_text(json.dumps(release))
    def minimum(path: Path):
        ref = runner.regular_ref(path)
        return {key: ref[key] for key in ("path", "bytes", "sha256")}

    result_qa = {
        "schema_version": "task29-independent-result-qa-v3",
        "status": "PASS_TASK29_V3_RESULT_INTEGRITY_EXACT",
        "producer": "INDEPENDENT_RESULT_QA",
        "AUTH_TIER": "T1_INDEPENDENT_RESULT_QA",
        "WHY_NOT_BLOCKED": "TERMINAL_ARTIFACT_INTEGRITY_READ_ONLY",
        "p0_findings": 0,
        "p1_findings": 0,
        "access_counters": {key: 0 for key in runner.QA_ACCESS_KEYS},
        "frozen_refs": {
            "gpu_admission": minimum(admission_path),
            "run_manifest": minimum(terminal_path),
            "artifact_manifest": minimum(artifact_path),
            "video_grap_a_cap_002": minimum(video_002_path),
            "video_grap_a_cap_012": minimum(video_012_path),
            "gpu_release": minimum(release_path),
            "cpu_qa": minimum(upstream_qa_path),
        },
        "checks": {},
        "findings": [],
        "claim_limit": "RESULT_INTEGRITY_ONLY_NO_PROMOTION_NO_BASELINE_FREEZE",
        "created_at": "2026-08-28T00:00:00+08:00",
    }
    result_qa_path.write_text(json.dumps(result_qa))
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "TASK29_RUN_REL", Path("_run/p1_v3"))
    monkeypatch.setattr(runner, "TASK29_GPU_RELEASE_REL", release_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_ADMISSION_REL", admission_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_TERMINAL_REL", terminal_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_QA_REL", upstream_qa_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_RESULT_QA_REL", result_qa_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_ARTIFACT_REL", artifact_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_VIDEO_002_REL", video_002_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_VIDEO_012_REL", video_012_path.relative_to(tmp_path))
    assert runner.require_task29_release()["task29_gpu_release"]["bytes"] > 0
    release["gpu_processes_remaining"] = False
    release_path.write_text(json.dumps(release))
    with pytest.raises(runner.Task32Error, match="status/type"):
        runner.require_task29_release()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "task29-gpu-admission-v1"),
        ("status", "ADMITTED_AWAITING_EXECUTION"),
        ("gpu_started", 0),
        ("gpu_started", True),
    ],
)
def test_task29_real_admission_contract_rejects_stale_or_ambiguous_state(
    field: str,
    value: object,
):
    admission = {
        "actual_run_filesystem": {
            "block_size": 4096,
            "free_bytes": 0,
            "st_dev": 102,
            "st_ino": 321,
        },
        "context_ref": {"path": "/context", "bytes": 1, "sha256": "a" * 64},
        "created_at": "2026-08-28T00:00:00+08:00",
        "directory_identity": {"st_dev": 102, "st_ino": 321},
        "disk_probe_ref": {"path": "/probe", "bytes": 1, "sha256": "b" * 64},
        "governance": {},
        "gpu_started": False,
        "owner_ref": {"path": "/owner", "bytes": 1, "sha256": "c" * 64},
        "schema_version": "task29-gpu-admission-v2",
        "status": "PASS_EXACT_FREEZE_OEXCL_ACTUAL_FS_GPU_READY",
    }
    admission[field] = value
    with pytest.raises(runner.Task32Error):
        runner.validate_task29_admission(admission)


def test_v2_incompatible_release_fixture_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "task29-gpu-release-v3",
                "status": "GPU_RELEASED",
                "gpu_processes_remaining": 0,
                "gpu_admission_ref": {},
                "terminal_manifest_ref": {},
                "independent_qa_ref": {},
            }
        )
    )
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "TASK29_GPU_RELEASE_REL", Path("release.json"))
    with pytest.raises(runner.Task32Error, match="exact schema"):
        runner.require_task29_release()


def test_real_task27_frame63_arm_evidence_exact_join_both_sides():
    manifest, _ = runner.verified_json(
        ROOT / runner.BASELINE_ROOT_REL / "RUN_MANIFEST.json"
    )
    frame = manifest["frames"][63]
    for side in runner.SIDES:
        row = frame["sides"][side]
        typed = runner.accepted_arm_identity(row, 63, side)
        mask = runner.load_bool_png(
            Path(row["formal_mask"]["path"]),
            row["formal_mask"]["sha256"],
            row["formal_mask"]["bytes"],
        )
        evidence = typed.verify(mask)
        assert evidence["instance_id"] == row["selected_instance_id"]
        assert typed.selected_offset == row["selected_raw_instance_offset"]


def test_target_disk_probe_rejects_intermediate_symlink(tmp_path: Path):
    real = tmp_path / "real"
    (real / "_run").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        runner.target_disk_probe(link / "_run")
    assert not list((real / "_run").glob(".task32_probe_*"))


def test_secure_mkdir_rejects_intermediate_symlink(tmp_path: Path):
    real = tmp_path / "real"
    parent = real / "parent"
    parent.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(runner.Task32Error, match="secure direct-child"):
        runner.secure_mkdir_direct(link / "parent/child")
    assert not (parent / "child").exists()


def test_shared_strict_io_live_identity_is_frozen():
    path = ROOT / runner.STRICT_IO_REL
    payload = path.read_bytes()
    assert len(payload) == runner.STRICT_IO_BYTES
    assert hashlib.sha256(payload).hexdigest() == runner.STRICT_IO_SHA256


def test_digest_bound_shared_loader_rejects_drift_and_symlinks(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    source = real / "strict.py"
    payload = b"VALUE = 7\n"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    loaded = runner.load_digest_bound_source_module(
        "task32_test_strict_ok",
        source,
        expected_bytes=len(payload),
        expected_sha256=digest,
    )
    assert loaded.VALUE == 7
    with pytest.raises(runner.Task32Error, match="bytes/SHA drift"):
        runner.load_digest_bound_source_module(
            "task32_test_strict_drift",
            source,
            expected_bytes=len(payload),
            expected_sha256="0" * 64,
        )
    final_link = real / "final_link.py"
    final_link.symlink_to(source)
    with pytest.raises(OSError):
        runner.load_digest_bound_source_module(
            "task32_test_strict_final_link",
            final_link,
            expected_bytes=len(payload),
            expected_sha256=digest,
        )
    parent_link = tmp_path / "parent_link"
    parent_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        runner.load_digest_bound_source_module(
            "task32_test_strict_parent_link",
            parent_link / "strict.py",
            expected_bytes=len(payload),
            expected_sha256=digest,
        )


def test_source_loader_executes_the_same_bytes_it_hashes_when_path_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.py"
    replacement = tmp_path / "replacement.py"
    payload_a = b"VALUE = 'A'\n"
    payload_b = b"VALUE = 'B'\n"
    source.write_bytes(payload_a)
    replacement.write_bytes(payload_b)
    original = runner.single_fd_bytes

    def read_then_replace(path: Path):
        payload, ref = original(path)
        os.replace(replacement, source)
        return payload, ref

    monkeypatch.setattr(runner, "single_fd_bytes", read_then_replace)
    loaded, ref = runner.load_verified_source_module("task32_same_bytes_a", source)
    assert loaded.VALUE == "A"
    assert ref["sha256"] == hashlib.sha256(payload_a).hexdigest()
    assert source.read_bytes() == payload_b


def test_source_loader_records_consumed_b_when_live_path_returns_to_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.py"
    replacement = tmp_path / "replacement.py"
    payload_a = b"VALUE = 'A'\n"
    payload_b = b"VALUE = 'B'\n"
    source.write_bytes(payload_b)
    replacement.write_bytes(payload_a)
    original = runner.single_fd_bytes

    def read_then_restore_path(path: Path):
        payload, ref = original(path)
        os.replace(replacement, source)
        return payload, ref

    monkeypatch.setattr(runner, "single_fd_bytes", read_then_restore_path)
    loaded, ref = runner.load_verified_source_module("task32_same_bytes_b", source)
    assert loaded.VALUE == "B"
    assert ref["sha256"] == hashlib.sha256(payload_b).hexdigest()
    assert source.read_bytes() == payload_a


def test_source_loader_rejects_path_bytes_that_do_not_match_frozen_ref(
    tmp_path: Path
):
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 'A'\n")
    _, frozen = runner.single_fd_bytes(source)
    source.write_bytes(b"VALUE = 'B'\n")
    with pytest.raises(runner.Task32Error, match="frozen source identity drift"):
        runner.load_verified_source_module(
            "task32_frozen_a_path_b",
            source,
            expected_ref=frozen,
        )
    assert "task32_frozen_a_path_b" not in sys.modules


@pytest.mark.parametrize(
    ("consumed", "replacement"),
    [(b"VALUE = 'A'\n", b"VALUE = 'B'\n"), (b"VALUE = 'B'\n", b"VALUE = 'A'\n")],
)
def test_transitive_loader_executes_consumed_dependency_bytes_in_both_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumed: bytes,
    replacement: bytes,
):
    child = tmp_path / "child.py"
    parent = tmp_path / "parent.py"
    swap = tmp_path / "swap.py"
    child.write_bytes(consumed)
    swap.write_bytes(replacement)
    parent.write_text(
        "from pathlib import Path\n"
        "CHILD = __TASK32_VERIFIED_SOURCE_LOADER__(\n"
        "    'task32_nested_child', Path(__file__).with_name('child.py')\n"
        ")\n"
        "VALUE = CHILD.VALUE\n"
    )
    refs = {
        "runtime_dependency_parent": runner.regular_ref(parent),
        "runtime_dependency_child": runner.regular_ref(child),
    }
    original = runner.single_fd_bytes

    def read_then_replace(path: Path):
        payload, ref = original(path)
        if Path(path) == child:
            os.replace(swap, child)
        return payload, ref

    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(
        runner,
        "BASE_RUNTIME_DEPENDENCIES",
        {"parent": Path("parent.py"), "child": Path("child.py")},
    )
    monkeypatch.setattr(runner, "single_fd_bytes", read_then_replace)
    loader = runner.verified_dependency_loader(refs)
    loaded = loader("task32_nested_parent", parent)
    assert loaded.VALUE == consumed.decode().split("'")[1]
    assert child.read_bytes() == replacement


def test_real_pixel_dependency_bundle_loads_from_verified_bytes():
    refs = {
        f"runtime_dependency_{key}": runner.regular_ref(ROOT / relative)
        for key, relative in runner.BASE_RUNTIME_DEPENDENCIES.items()
    }
    loader = runner.verified_dependency_loader(refs)
    base, observed = runner.load_verified_source_module(
        "task32_verified_base_test",
        ROOT / runner.BASE_RUNNER_REL,
        expected_ref=runner.regular_ref(ROOT / runner.BASE_RUNNER_REL),
        initial_globals={"__TASK32_VERIFIED_SOURCE_LOADER__": loader},
    )
    assert observed["sha256"] == runner.BASE_RUNNER_SHA
    assert callable(base.build_pinned_adapter)
    assert callable(base.select_per_side_identities)


def test_live_task29_v3_release_chain_is_consumed_exactly():
    result = runner.require_task29_release()
    assert result["task29_gpu_admission"]["sha256"] == (
        "c068362ac4636ceba721c0949d52f523edc82914d960ee371878c4e0a4631234"
    )
    assert result["task29_gpu_release"]["sha256"] == (
        "fe26090ea5b8212b1dddd98c7ba195490cbf690532478b64e8fa58108f92ab8e"
    )


def test_review_video_output_rejects_preexisting_symlink(tmp_path: Path):
    run_root = tmp_path / "run"
    candidate = run_root / "candidate_W1"
    candidate.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"sentinel")
    video = candidate / "BEFORE_AFTER_FULL460.mp4"
    video.symlink_to(outside)
    with pytest.raises(FileExistsError):
        runner.start_review_video(video, run_root)
    assert outside.read_bytes() == b"sentinel"
    assert video.is_symlink()


def test_review_video_uses_fd_bound_stream_and_waits_for_child(tmp_path: Path):
    run_root = tmp_path / "run"
    candidate = run_root / "candidate_W1"
    candidate.mkdir(parents=True)
    video_path = candidate / "BEFORE_AFTER_FULL460.mp4"
    handle = runner.start_review_video(video_path, run_root)
    assert handle.stdin is not None
    handle.stdin.write(np.zeros((960, 2560, 3), dtype=np.uint8).tobytes())
    result = handle.wait_and_publish(timeout=30)
    assert result["child_exit_observed"] is True
    assert result["returncode"] == 0
    assert result["published"] is True
    assert result["bytes"] == video_path.stat().st_size
    assert result["sha256"] == hashlib.sha256(video_path.read_bytes()).hexdigest()


def success_terminal() -> dict[str, object]:
    return {
        "schema_version": "004-neutral-wearable-all-prompts-v1",
        "status": "COMPLETED_UNREVIEWED_AWAITING_INDEPENDENT_QA",
        "AUTH_TIER": "T1_CANDIDATE_ONLY",
        "prompt_results": [{"prompt_id": prompt_id} for prompt_id in ("W1", "W2", "W3", "W4")],
        "all_four_prompts_executed": True,
        "result_driven_prompt_choice": False,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
    }


def test_failure_terminal_is_o_excl_and_binds_partial_artifacts(tmp_path: Path):
    run_root = tmp_path / runner.TASK32_RUN_ID
    run_root.mkdir()
    (run_root / "GPU_ADMISSION.json").write_text("{}")
    partial = run_root / "partial.bin"
    partial.write_bytes(b"partial")
    terminal = runner.write_failure_terminal(
        run_root,
        error_type="SyntheticFailure",
        error_message="boom",
        error_traceback="trace",
    )
    assert terminal["status"] == "HOLD_EXECUTION_FAILED_PARTIAL_PRESERVED"
    refs = terminal["partial_artifact_snapshot"]["artifacts"]
    assert any(row["path"] == "partial.bin" for row in refs)
    with pytest.raises(FileExistsError):
        runner.write_failure_terminal(
            run_root,
            error_type="Overwrite",
            error_message="forbidden",
            error_traceback="",
        )


def test_failure_terminal_does_not_follow_preexisting_symlink(tmp_path: Path):
    run_root = tmp_path / runner.TASK32_RUN_ID
    run_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"sentinel")
    (run_root / "FAILURE_MANIFEST.json").symlink_to(outside)
    with pytest.raises(FileExistsError):
        runner.write_failure_terminal(
            run_root,
            error_type="SyntheticFailure",
            error_message="boom",
            error_traceback="trace",
        )
    assert outside.read_bytes() == b"sentinel"


def test_post_child_success_release_is_exact_and_does_not_unlock_p3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_root = tmp_path / runner.TASK32_RUN_ID
    run_root.mkdir()
    (run_root / "GPU_ADMISSION.json").write_text('{"status":"ADMITTED"}')
    (run_root / "RUN_MANIFEST.json").write_text(json.dumps(success_terminal()))
    monkeypatch.setattr(
        runner,
        "gpu_process_snapshot",
        lambda: ([], "nvidia-smi synthetic"),
    )
    release = runner.finalize_task32_gpu_release(
        run_root,
        exit_code=0,
        child_exit_observed=True,
    )
    assert release["status"] == "GPU_RELEASED_AFTER_SUCCESS"
    assert release["gpu_child_has_exited"] is True
    assert release["gpu_processes_remaining"] == 0
    assert release["gpu_resource_released"] is True
    assert release["unlocking_p3"] is False
    assert release["p3_requires_independent_result_qa"] is True
    with pytest.raises(FileExistsError):
        runner.finalize_task32_gpu_release(
            run_root,
            exit_code=0,
            child_exit_observed=True,
        )


def test_release_before_child_exit_is_forbidden(tmp_path: Path):
    run_root = tmp_path / runner.TASK32_RUN_ID
    run_root.mkdir()
    with pytest.raises(runner.Task32Error, match="before child exit"):
        runner.finalize_task32_gpu_release(
            run_root,
            exit_code=0,
            child_exit_observed=False,
        )
    assert not (run_root / "GPU_RELEASE.json").exists()


def test_missing_child_terminal_becomes_failure_then_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_root = tmp_path / runner.TASK32_RUN_ID
    run_root.mkdir()
    (run_root / "GPU_ADMISSION.json").write_text('{"status":"ADMITTED"}')
    monkeypatch.setattr(
        runner,
        "gpu_process_snapshot",
        lambda: ([], "nvidia-smi synthetic"),
    )
    release = runner.finalize_task32_gpu_release(
        run_root,
        exit_code=17,
        child_exit_observed=True,
    )
    failure = json.loads((run_root / "FAILURE_MANIFEST.json").read_text())
    assert failure["error"]["type"] == "MissingChildTerminal"
    assert release["status"] == "GPU_RELEASED_AFTER_FAILURE"
    assert release["terminal_kind"] == "FAILURE_MANIFEST"
    assert release["successful_task32_terminal"] is False


def test_remaining_gpu_process_keeps_release_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_root = tmp_path / runner.TASK32_RUN_ID
    run_root.mkdir()
    (run_root / "GPU_ADMISSION.json").write_text('{"status":"ADMITTED"}')
    (run_root / "RUN_MANIFEST.json").write_text(json.dumps(success_terminal()))
    monkeypatch.setattr(
        runner,
        "gpu_process_snapshot",
        lambda: ([{"pid": 123, "used_memory_mib": 10}], "nvidia-smi synthetic"),
    )
    release = runner.finalize_task32_gpu_release(
        run_root,
        exit_code=0,
        child_exit_observed=True,
    )
    assert release["status"] == "GPU_RELEASE_PENDING_PROCESSES"
    assert release["gpu_resource_released"] is False
    assert release["gpu_processes_remaining"] == 1
