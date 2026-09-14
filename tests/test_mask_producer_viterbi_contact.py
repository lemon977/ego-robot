from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import pipeline.mask_producer_viterbi_contact as producer_module

from pipeline.mask_producer_viterbi_contact import (
    LINEAGE,
    NOT_A_V9_HEURISTIC_PATCH,
    PRODUCER_ID,
    ProducerConfig,
    SamCandidate,
    assemble_current_frame_arm_candidates,
    cylinder_depth_map,
    freeze_inputs,
    local_forearm_direction,
    noncontact_changed_pixels,
    part_prompt_points,
    refine_contact_band,
    run_candidate,
    stable_holes_symmetric_sdf,
    stage3_ownership,
    validate_registered_model_asset,
    verify_frozen_inputs,
    verify_registered_model_inventory,
    viterbi_select,
)


def _candidate(mask: np.ndarray, unary: float) -> SamCandidate:
    return SamCandidate(mask, unary, 1.0, 1.0, 0.0, 0.0, 0.0)


def _zero_flows(count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    return [np.zeros((*shape, 2), np.float32) for _ in range(count)]


def test_new_identity_is_not_v9_and_viterbi_selects_only_current_candidates() -> None:
    assert PRODUCER_ID == "producer_p1"
    assert NOT_A_V9_HEURISTIC_PATCH is True
    assert "No inheritance from the v9" in LINEAGE
    first_a = np.zeros((12, 12), bool)
    first_b = np.zeros((12, 12), bool)
    second_a = np.zeros((12, 12), bool)
    second_b = np.zeros((12, 12), bool)
    first_a[2:6, 2:6] = True
    second_a[2:6, 2:6] = True
    first_b[7:10, 7:10] = True
    second_b[1:4, 8:11] = True
    candidates = [
        [_candidate(first_a, 1.0), _candidate(first_b, 0.0)],
        [_candidate(second_a, 0.2), _candidate(second_b, 0.3)],
    ]
    selected = viterbi_select(
        candidates, _zero_flows(1, first_a.shape), _zero_flows(1, first_a.shape), ProducerConfig()
    )
    assert selected == [0, 0]
    assert np.array_equal(candidates[1][selected[1]].mask, second_a)
    assert not np.shares_memory(candidates[1][selected[1]].mask, first_a)


def test_viterbi_fails_closed_when_any_frame_has_no_real_candidate() -> None:
    mask = np.ones((4, 4), bool)
    with pytest.raises(ValueError, match="at least one current-frame candidate"):
        viterbi_select([[_candidate(mask, 1.0)], []], _zero_flows(1, mask.shape), _zero_flows(1, mask.shape), ProducerConfig())


def test_unobservable_optional_sleeve_adds_no_pixels_or_pseudo_candidate() -> None:
    hand = np.zeros((8, 8), bool)
    wrist = np.zeros_like(hand)
    hand[2:5, 2:5] = True
    wrist[4:6, 3:5] = True
    candidates = assemble_current_frame_arm_candidates(
        {
            "hand": [_candidate(hand, 1.0)],
            "wrist_band": [_candidate(wrist, 0.8)],
            "wrist_device": [],
            "connected_sleeve": [],
        }
    )
    assert len(candidates) == 1
    assert np.array_equal(candidates[0].mask, hand | wrist)
    with pytest.raises(ValueError, match="hand anchor"):
        assemble_current_frame_arm_candidates({"hand": [], "connected_sleeve": []})


def test_local_prompts_are_finite_and_degenerate_direction_has_no_fallback() -> None:
    joints = np.zeros((21, 2), np.float32)
    joints[:, 0] = np.linspace(30, 70, 21)
    joints[:, 1] = np.linspace(40, 80, 21)
    joints[0] = [50, 90]
    joints[[5, 9, 13, 17]] = [[40, 65], [47, 62], [54, 62], [61, 65]]
    prompts = part_prompt_points(joints, "connected_sleeve")
    assert prompts.shape == (3, 2)
    assert np.isfinite(prompts).all()
    assert float(np.max(np.linalg.norm(prompts - joints[0], axis=1))) < 3 * 100

    degenerate = np.ones((21, 2), np.float32) * 5
    assert local_forearm_direction(degenerate) is None
    assert part_prompt_points(degenerate, "connected_sleeve").shape == (0, 2)


def test_cylinder_projection_is_metric_and_finite_only_on_projected_object() -> None:
    transform = np.eye(4)
    transform[2, 3] = 1.0
    depth = cylinder_depth_map(transform, 0.1, 0.3, 80.0, (40.0, 30.0), (60, 80))
    assert np.isfinite(depth[30, 40])
    assert depth[30, 40] < 1.0
    assert not np.isfinite(depth[0, 0])
    assert 0 < int(np.isfinite(depth).sum()) < depth.size


def test_stage3_ownership_is_mutually_exclusive() -> None:
    arm = np.zeros((20, 20), bool)
    pose = np.zeros_like(arm)
    observed = np.zeros_like(arm)
    stable = np.zeros_like(arm)
    arm[4:16, 4:16] = True
    pose[8:18, 8:18] = True
    observed[10:14, 10:14] = True
    stable[9:12, 9:12] = True
    final_arm, protect, hand_wins = stage3_ownership(arm, observed, pose, stable)
    assert np.array_equal(hand_wins, stable & pose)
    assert not np.logical_and(final_arm, protect).any()
    assert np.array_equal(final_arm, arm & ~protect)


def test_stage4_is_bit_exact_outside_contact_and_cannot_invent_non_sam_pixels() -> None:
    shape = (40, 40)
    arm_candidate = np.zeros(shape, bool)
    arm_candidate[10:30, 5:25] = True
    pose = np.zeros(shape, bool)
    pose[8:32, 16:32] = True
    observed = np.zeros(shape, bool)
    stable = np.zeros(shape, bool)
    digits = np.zeros(shape, bool)
    digits[16:24, 12:27] = True
    stage3, _, _ = stage3_ownership(arm_candidate, observed, pose, stable)
    hand_depth = np.full(shape, np.inf, np.float32)
    cad_depth = np.full(shape, np.inf, np.float32)
    hand_depth[digits] = 0.8
    cad_depth[pose] = 1.0
    refined, band = refine_contact_band(
        stage3,
        arm_candidate,
        observed,
        pose,
        digits,
        stable,
        hand_depth,
        cad_depth,
        0.1,
        ProducerConfig(),
    )
    assert noncontact_changed_pixels(stage3, refined, band) == 0
    assert not np.logical_and(refined, ~arm_candidate).any()
    adversarial = refined.copy()
    outside = np.argwhere(~band)[0]
    adversarial[tuple(outside)] = ~adversarial[tuple(outside)]
    assert noncontact_changed_pixels(refined, adversarial, band) == 1


def test_symmetric_sdf_identity_flow_keeps_supported_hole() -> None:
    masks = []
    for _ in range(5):
        value = np.zeros((24, 24), bool)
        value[9:15, 9:15] = True
        masks.append(value)
    flows = _zero_flows(4, masks[0].shape)
    stable = stable_holes_symmetric_sdf(masks, masks, masks, masks, flows, flows, 2)
    assert len(stable) == 5
    assert all(value[12, 12] for value in stable)


def test_input_freeze_has_t1_block_fields_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_root = tmp_path / "_run"
    candidate_root.mkdir()
    monkeypatch.setattr(producer_module, "CANDIDATE_RUNS_ROOT", candidate_root)
    monkeypatch.setattr(producer_module, "PROJECT_ROOT", tmp_path)
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    (contracts / "pipeline_contract_v1.yaml").write_text("model_pins: {}\n", encoding="utf-8")
    (contracts / "project_profile_v1.yaml").write_text("dependencies: {}\n", encoding="utf-8")
    raw = tmp_path / "raw.png"
    assert cv2.imwrite(str(raw), np.zeros((8, 8, 3), np.uint8))
    import hashlib

    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "session_digests": {"synthetic": "a" * 64},
                "sessions": [
                    {
                        "session_id": "synthetic",
                        "frame_count": 1,
                        "frames": [{"frame_index": 0, "image": {"path": str(raw), "sha256": raw_sha}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    implementation = tmp_path / "implementation"
    config = implementation / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: synthetic\n", encoding="utf-8")
    inventory = producer_module.implementation_inventory(implementation)
    registration = {
        "model_identifier": "SAM2_1_HIERA_LARGE", "asset_pin_sha256": "a" * 64,
        "pipeline_contract_sha256": "b" * 64, "project_profile_sha256": "c" * 64,
        "implementation_inventory_entries": inventory[0],
        "implementation_regular_bytes": inventory[1],
        "implementation_inventory_sha256": inventory[2],
        "config_path": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "config_sha256": "d" * 64, "weight_path": "weight", "weight_bytes": 1,
        "weight_sha256": "e" * 64,
    }
    monkeypatch.setattr(producer_module, "validate_registered_model_asset", lambda *_: registration)
    inputs = {}
    for name in ("hawor_projection", "object6d", "model_asset_pin", "model_weight", "fallback_evidence"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs[name] = str(path)
    arguments = argparse.Namespace(
        source_manifest=str(source),
        session_id="synthetic",
        frame_start=0,
        frame_count=1,
        output_root=str(candidate_root / "a2_test"),
        model_implementation=str(implementation),
        model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        **inputs,
    )
    result = freeze_inputs(arguments)
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert payload["candidate_policy"] == {
        "status": "candidate_requires_human_review",
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "processed_write_allowed": False,
    }
    assert payload["labels_read"] is False
    assert payload["held_out_session_149_read"] is False
    verify_frozen_inputs(payload)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_inputs(arguments)


def test_output_containment_partial_root_and_input_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_root = tmp_path / "_run"
    candidate_root.mkdir()
    monkeypatch.setattr(producer_module, "CANDIDATE_RUNS_ROOT", candidate_root)
    monkeypatch.setattr(producer_module, "PROJECT_ROOT", tmp_path)
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    (contracts / "pipeline_contract_v1.yaml").write_text("model_pins: {}\n", encoding="utf-8")
    (contracts / "project_profile_v1.yaml").write_text("dependencies: {}\n", encoding="utf-8")
    raw = tmp_path / "raw.png"
    assert cv2.imwrite(str(raw), np.zeros((8, 8, 3), np.uint8))
    import hashlib

    source = tmp_path / "source.json"
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    source.write_text(
        json.dumps(
            {
                "session_digests": {"synthetic": "b" * 64},
                "sessions": [{"session_id": "synthetic", "frame_count": 1, "frames": [
                    {"frame_index": 0, "image": {"path": str(raw), "sha256": raw_sha}}
                ]}],
            }
        ),
        encoding="utf-8",
    )
    implementation = tmp_path / "implementation"
    config = implementation / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: synthetic\n", encoding="utf-8")
    inventory = producer_module.implementation_inventory(implementation)
    registration = {
        "model_identifier": "SAM2_1_HIERA_LARGE", "asset_pin_sha256": "a" * 64,
        "pipeline_contract_sha256": "b" * 64, "project_profile_sha256": "c" * 64,
        "implementation_inventory_entries": inventory[0],
        "implementation_regular_bytes": inventory[1],
        "implementation_inventory_sha256": inventory[2],
        "config_path": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "config_sha256": "d" * 64, "weight_path": "weight", "weight_bytes": 1,
        "weight_sha256": "e" * 64,
    }
    monkeypatch.setattr(producer_module, "validate_registered_model_asset", lambda *_: registration)
    refs = {}
    for name in ("hawor_projection", "object6d", "model_asset_pin", "model_weight", "fallback_evidence"):
        value = tmp_path / name
        value.write_bytes(name.encode())
        refs[name] = str(value)
    common = dict(
        source_manifest=str(source), session_id="synthetic", frame_start=0, frame_count=1,
        model_implementation=str(implementation), model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        **refs,
    )
    with pytest.raises(ValueError, match="project/_run"):
        freeze_inputs(argparse.Namespace(output_root=str(tmp_path / "a2_escape"), **common))

    output_root = candidate_root / "a2_partial"
    frozen = freeze_inputs(argparse.Namespace(output_root=str(output_root), **common))
    freeze = json.loads(Path(frozen["path"]).read_text(encoding="utf-8"))
    (output_root / "unexpected.partial").write_text("x", encoding="utf-8")
    run_args = argparse.Namespace(
        output_root=str(output_root), expected_input_freeze_sha=frozen["sha256"], device="cpu"
    )
    with pytest.raises(FileExistsError, match="partial or already used"):
        run_candidate(run_args)

    drift_root = candidate_root / "a2_drift"
    drifted = freeze_inputs(argparse.Namespace(output_root=str(drift_root), **common))
    drift_payload = json.loads(Path(drifted["path"]).read_text(encoding="utf-8"))
    Path(refs["object6d"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="object6d byte identity drift"):
        verify_frozen_inputs(drift_payload)


def test_real_sam_asset_pin_contracts_and_inventory_are_one_registered_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_pin_path = root / "assets/models/sam2_1_hiera_large/ASSET_PIN.json"
    weight_path = root / "assets/models/sam2_1_hiera_large/sam2.1_hiera_large.pt"
    pipeline_path = root / "contracts/pipeline_contract_v1.yaml"
    profile_path = root / "contracts/project_profile_v1.yaml"
    pin = json.loads(asset_pin_path.read_text(encoding="utf-8"))["local"]
    refs = {
        "model_asset_pin": {"path": str(asset_pin_path), "sha256": producer_module.sha256_file(asset_pin_path), "bytes": asset_pin_path.stat().st_size},
        "model_weight": {"path": str(weight_path), "sha256": pin["weight_sha256"], "bytes": weight_path.stat().st_size},
        "pipeline_contract": {"path": str(pipeline_path), "sha256": producer_module.sha256_file(pipeline_path), "bytes": pipeline_path.stat().st_size},
        "project_profile": {"path": str(profile_path), "sha256": producer_module.sha256_file(profile_path), "bytes": profile_path.stat().st_size},
    }
    implementation = root / pin["implementation_ref"]
    config = implementation / pin["config_path"]
    registered = validate_registered_model_asset(refs, implementation, config)
    assert registered["implementation_inventory_entries"] == 38
    assert registered["implementation_regular_bytes"] == 1733828
    assert registered["implementation_inventory_sha256"] == pin["implementation_inventory_sha256"]


def test_bytecode_is_disabled_and_post_load_inventory_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sys.dont_write_bytecode is True
    implementation = tmp_path / "implementation"
    config = implementation / "sam2" / "configs" / "model.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: synthetic\n", encoding="utf-8")
    inventory = producer_module.implementation_inventory(implementation)
    registration = {"implementation_inventory_sha256": inventory[2]}
    monkeypatch.setattr(
        producer_module, "validate_registered_model_asset", lambda *_: registration
    )
    freeze = {
        "model": {
            "implementation_ref": str(implementation),
            "implementation_inventory_entries": inventory[0],
            "implementation_regular_bytes": inventory[1],
            "implementation_inventory_sha256": inventory[2],
            "config_ref": {"path": str(config)},
            "registered_pin": registration,
        },
        "inputs": {},
    }
    verify_registered_model_inventory(freeze)
    (implementation / "sam2" / "__pycache__").mkdir()
    (implementation / "sam2" / "__pycache__" / "pollution.pyc").write_bytes(b"drift")
    with pytest.raises(ValueError, match="implementation inventory drift"):
        verify_registered_model_inventory(freeze)
