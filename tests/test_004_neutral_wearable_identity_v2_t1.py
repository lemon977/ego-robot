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
    "task32_runner_v2", ROOT / "tools/run_004_neutral_wearable_identity_v2_t1.py"
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


def qa_payload(refs: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "004-neutral-wearable-identity-independent-qa-v2",
        "status": "PASS_CPU_ADMISSION_EXACT",
        "producer": "INDEPENDENT_QA",
        "AUTH_TIER": "T1_INDEPENDENT_CPU_QA",
        "WHY_NOT_BLOCKED": "AUTHORIZED_READ_ONLY_TASK32_V2_CPU_FREEZE_ADVERSARIAL_REPLAY",
        "p0_findings": 0,
        "p1_findings": 0,
        "access_counters": {key: 0 for key in runner.QA_ACCESS_KEYS},
        "frozen_refs": refs,
        "checks": {},
        "findings": [],
        "claim_limit": "CPU_QA_ONLY_GPU_ADMISSION_ALLOWED_CANDIDATE_STILL_UNREVIEWED",
        "created_at": "2026-08-28T00:00:00+08:00",
    }


@pytest.mark.parametrize("attack", ["bool_p0", "nonzero_access", "unknown_key"])
def test_governance_rejects_fake_schema_bool_and_nonzero_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
):
    refs = {"frozen": {"path": "/f", "bytes": 1, "sha256": "a" * 64}}
    qa = qa_payload(refs)
    if attack == "bool_p0":
        qa["p0_findings"] = False
    elif attack == "nonzero_access":
        qa["access_counters"]["labels_read"] = 1  # type: ignore[index]
    else:
        qa["attacker"] = True
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(qa))
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "QA_REL", Path("qa.json"))
    monkeypatch.setattr(runner, "require_task29_release", lambda: {})
    with pytest.raises(runner.Task32Error):
        runner.require_governance(refs)


def test_governance_parses_the_same_verified_bytes_under_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    refs = {"frozen": {"path": "/f", "bytes": 1, "sha256": "a" * 64}}
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(qa_payload(refs)))
    original = runner.single_fd_bytes

    def replace_after_read(target: Path):
        payload, ref = original(target)
        target.write_text("{}")
        return payload, ref

    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "QA_REL", Path("qa.json"))
    monkeypatch.setattr(runner, "single_fd_bytes", replace_after_read)
    monkeypatch.setattr(runner, "require_task29_release", lambda: {})
    value = runner.require_governance(refs)
    assert value["independent_cpu_qa"]["sha256"] != hashlib.sha256(b"{}").hexdigest()


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
    upstream_qa_path = audits / "qa.json"
    admission_path.write_text(json.dumps({"status": "ADMITTED_AWAITING_EXECUTION"}))
    terminal_path.write_text(
        json.dumps({"status": "COMPLETED_UNREVIEWED_AWAITING_INDEPENDENT_QA"})
    )
    upstream_qa_path.write_text(
        json.dumps(
            {
                "producer": "INDEPENDENT_QA",
                "p0_findings": 0,
                "access_counters": {"labels_read": 0},
            }
        )
    )
    admission_ref = runner.minimal_ref(
        runner.regular_ref(admission_path), "TASK29_GPU_ADMISSION"
    )
    terminal_ref = runner.minimal_ref(
        runner.regular_ref(terminal_path), "TASK29_TERMINAL_MANIFEST"
    )
    qa_ref = runner.minimal_ref(
        runner.regular_ref(upstream_qa_path), "INDEPENDENT_QA_JSON"
    )
    release = {
        "schema_version": "task29-gpu-release-v3",
        "status": "GPU_RELEASED",
        "gpu_processes_remaining": 0,
        "gpu_admission_ref": admission_ref,
        "terminal_manifest_ref": terminal_ref,
        "independent_qa_ref": qa_ref,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "partial_artifacts_preserved": True,
        "created_at": "2026-08-28T00:00:00+08:00",
    }
    release_path = run / "GPU_RELEASE.json"
    release_path.write_text(json.dumps(release))
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "TASK29_GPU_RELEASE_REL", release_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_ADMISSION_REL", admission_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_TERMINAL_REL", terminal_path.relative_to(tmp_path))
    monkeypatch.setattr(runner, "TASK29_QA_REL", upstream_qa_path.relative_to(tmp_path))
    assert runner.require_task29_release()["task29_gpu_release"]["bytes"] > 0
    release["gpu_processes_remaining"] = False
    release_path.write_text(json.dumps(release))
    with pytest.raises(runner.Task32Error, match="status/type"):
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
