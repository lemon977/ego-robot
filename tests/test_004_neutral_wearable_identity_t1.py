from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "neutral_wearable", ROOT / "pipeline/neutral_wearable_identity_v1.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


RUNNER_SPEC = importlib.util.spec_from_file_location(
    "task32_runner", ROOT / "tools/run_004_neutral_wearable_identity_t1.py"
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


def test_identity_union_is_bit_exact_or_only_and_decision_bound():
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
        arm_status="ACCEPT",
        arm_selected_raw_evidence_sha256="b" * 64,
        arm_raw_sha256=mod.mask_sha256(arm),
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


def test_union_rejects_unselected_instance_and_nonaccepted_arm():
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
            arm_status="ACCEPT",
            arm_selected_raw_evidence_sha256="b" * 64,
            arm_raw_sha256=mod.mask_sha256(mask),
            wearable_decision=decision,
            wearable_instance=wrong,
        )
    with pytest.raises(mod.WearableIdentityError, match="arm identity"):
        mod.identity_union(
            mask,
            selected.mask,
            side="left",
            arm_status="HOLD",
            arm_selected_raw_evidence_sha256="b" * 64,
            arm_raw_sha256=mod.mask_sha256(mask),
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


def test_admission_hardblocks_before_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    (tmp_path / "_run").mkdir()
    monkeypatch.setattr(runner, "frozen_input_refs", lambda: {"frozen": True})
    assert runner.INDEPENDENT_CPU_QA_SHA256 is None
    with pytest.raises(runner.Task32Error, match="independent CPU QA"):
        runner.admit("must_not_exist")
    assert not (tmp_path / "_run/must_not_exist").exists()
