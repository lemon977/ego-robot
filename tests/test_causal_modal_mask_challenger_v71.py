from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_challenger_v71 import (
    BackendFrameOutput,
    MaskChallengerError,
    preflight_backend,
    run_causal_sequence,
    validate_backend_environment,
    validate_execution_lease,
    validate_prompt_manifest,
    validate_rgb_manifest,
)
from chaoyang.ops.run_causal_modal_mask_challenger_v71 import build_preflight_receipt, main


def evidence(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def inputs(tmp_path: Path, *, correction_at: int | None = 2):
    frames = []
    for frame_id in range(3):
        path = tmp_path / f"rgb_{frame_id}.bin"
        path.write_bytes(f"rgb-{frame_id}".encode())
        frames.append({"frame_id": frame_id, **evidence(path)})
    rgb = {
        "schema_version": "causal-rgb-manifest-v71",
        "artifact_id": "raw.fixture",
        "artifact_revision": "R7_0",
        "session_id": "chips_fixture",
        "frozen": True,
        "execution_mode": "CAUSAL_PROCESSING",
        "frames": frames,
    }
    audit_path = tmp_path / "audit.json"
    write_json(
        audit_path,
        {
            "immutable": True,
            "authority": False,
            "gold_accuracy_authorized": False,
            "rows": [{"session_id": "chips_fixture"}],
        },
    )
    instances = []
    for index in range(3):
        mask0 = tmp_path / f"prompt_{index}_0.bin"
        mask0.write_bytes(f"mask-{index}-0".encode())
        events = [{"frame_id": 0, "event_type": "INITIAL_MASK", "mask": evidence(mask0)}]
        if correction_at is not None:
            mask2 = tmp_path / f"prompt_{index}_2.bin"
            mask2.write_bytes(f"mask-{index}-2".encode())
            events.append(
                {"frame_id": correction_at, "event_type": "CORRECTION_MASK", "mask": evidence(mask2)}
            )
        instances.append({"instance_id": f"chips_{index + 1}", "prompt_events": events})
    prompts = {
        "schema_version": "causal-modal-mask-prompts-v71",
        "artifact_id": "prompts.fixture",
        "artifact_revision": "R7_1",
        "session_id": "chips_fixture",
        "task": "CHIPS",
        "frozen": True,
        "allow_instance_union": False,
        "audit_selection": evidence(audit_path),
        "instances": instances,
    }
    return rgb, prompts


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_sam_asset(repo: Path):
    root = repo / "assets/models/sam2_1_hiera_large"
    root.mkdir(parents=True)
    implementation = root / "implementation"
    config = implementation / "sam2/configs/model.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"model: fixture")
    weight = root / "weight.bin"
    license_path = root / "LICENSE"
    weight.write_bytes(b"weight")
    license_path.write_bytes(b"license")
    pin = {
        "local": {
            "implementation_ref": "implementation",
            "implementation_inventory_sha256": "d" * 64,
            "config_path": "sam2/configs/model.yaml",
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weight_path": "weight.bin",
            "weight_bytes": weight.stat().st_size,
            "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "license_path": "LICENSE",
            "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        },
        "load_smoke": {"status": "PASS_CPU_MODEL_INSTANTIATION"},
        "execution": {"formal_production_allowed": False},
    }
    write_json(root / "ASSET_PIN.json", pin)


def test_frozen_manifests_and_three_independent_chips_instances(tmp_path):
    rgb, prompts = inputs(tmp_path)
    assert validate_rgb_manifest(rgb) == []
    assert validate_prompt_manifest(prompts, rgb) == []
    for name, value in (
        ("causal_rgb_manifest_v71.schema.json", rgb),
        ("causal_modal_mask_prompts_v71.schema.json", prompts),
    ):
        schema = json.loads((ROOT / "contracts" / name).read_text())
        jsonschema.Draft202012Validator(schema).validate(value)
    prompts["allow_instance_union"] = True
    assert "allow_instance_union must be false" in validate_prompt_manifest(prompts, rgb)
    prompts["allow_instance_union"] = False
    prompts["instances"].pop()
    assert "CHIPS requires exactly three independent physical instances" in validate_prompt_manifest(prompts, rgb)


def test_manifest_rejects_mutated_bytes_and_noncausal_frame_order(tmp_path):
    rgb, _ = inputs(tmp_path)
    Path(rgb["frames"][0]["path"]).write_bytes(b"changed")
    rgb["frames"] = [rgb["frames"][1], rgb["frames"][0], rgb["frames"][2]]
    errors = validate_rgb_manifest(rgb)
    assert any("bytes mismatch" in item or "SHA" in item for item in errors)
    assert any("strictly increasing" in item for item in errors)


def test_sam_asset_is_development_ready_but_not_formal_authority(tmp_path):
    make_sam_asset(tmp_path)
    result = preflight_backend("sam2_1", repository_root=tmp_path, verify_large_sha=True)
    assert result["status"] == "READY_DEVELOPMENT_CANARY"
    assert result["execution_ready"] is True
    assert result["formal_production_allowed"] is False
    assert result["weight_sha_verified_now"] is True


def test_cutie_embedded_code_or_partial_weight_without_pin_is_blocked(tmp_path):
    root = tmp_path / "assets/models/cutie"
    root.mkdir(parents=True)
    (root / "cutie-base-mega.pth.part").write_bytes(b"partial")
    result = preflight_backend("cutie", repository_root=tmp_path)
    assert result["status"] == "BLOCKED_REFERENCE_PROOF"
    assert result["execution_ready"] is False
    assert result["partial_files"]


def test_repository_cutie_pin_closes_and_requires_pinned_torch_home():
    result = preflight_backend("cutie", repository_root=ROOT)
    assert result["status"] == "READY_DEVELOPMENT_CANARY"
    assert result["execution_ready"] is True
    assert result["formal_production_allowed"] is False
    assert result["runtime_environment"]["TORCH_HOME"] == str(
        (ROOT / "assets/models/cutie/torch_home").resolve()
    )
    assert validate_backend_environment(result, result["runtime_environment"]) == []
    assert validate_backend_environment(result, {}) == ["backend environment mismatch: TORCH_HOME"]


class FakeBackend:
    def __init__(self):
        self.seen_prompts = {}

    def step(self, *, frame_id, rgb_path, prompt_masks):
        self.seen_prompts[frame_id] = sorted(prompt_masks)
        ids = [f"chips_{index}" for index in range(1, 4)]
        if frame_id == 1:
            return BackendFrameOutput(
                masks={},
                visibility={key: "FULLY_OCCLUDED_UNKNOWN" for key in ids},
            )
        masks = {}
        for index, key in enumerate(ids):
            value = np.zeros((3, 3), dtype=bool)
            value[index, index] = True
            masks[key] = value
        return BackendFrameOutput(masks=masks, visibility={key: "VISIBLE" for key in ids})


def test_runner_is_causal_and_full_occlusion_is_unknown_without_pseudomask(tmp_path):
    rgb, prompts = inputs(tmp_path)
    backend = FakeBackend()
    output = run_causal_sequence(rgb, prompts, backend, output_directory=tmp_path / "output")
    assert backend.seen_prompts[0] == ["chips_1", "chips_2", "chips_3"]
    assert backend.seen_prompts[1] == []
    assert backend.seen_prompts[2] == ["chips_1", "chips_2", "chips_3"]
    assert all(item["mask"] is None for item in output["frames"][1]["instances"])
    assert output["allow_instance_union"] is False
    assert "OBJECT6D_AUTHORITY" in output["authority_limit"]["forbidden_claims"]
    schema = json.loads((ROOT / "contracts/causal_modal_mask_output_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(output)


class BadUnknownBackend(FakeBackend):
    def step(self, *, frame_id, rgb_path, prompt_masks):
        ids = [f"chips_{index}" for index in range(1, 4)]
        value = np.ones((2, 2), dtype=bool)
        return BackendFrameOutput(
            masks={key: value for key in ids},
            visibility={key: "FULLY_OCCLUDED_UNKNOWN" for key in ids},
        )


def test_unknown_with_nonempty_mask_is_rejected(tmp_path):
    rgb, prompts = inputs(tmp_path)
    with pytest.raises(MaskChallengerError, match="pseudo-mask"):
        run_causal_sequence(rgb, prompts, BadUnknownBackend(), output_directory=tmp_path / "bad")


def test_v71_lease_requires_task_attempt_and_fencing_match():
    lease = {
        "schema_version": "chaoyang-gpu-lease-v71",
        "status": "ACQUIRED",
        "task_id": "s1",
        "attempt_id": "attempt_1",
        "fencing_token": "token",
        "executor_epoch": 4,
        "expires_at": (datetime.now().astimezone() + timedelta(minutes=1)).isoformat(),
    }
    assert validate_execution_lease(
        lease, task_id="s1", attempt_id="attempt_1", fencing_token="token", executor_epoch=4
    ) == []
    assert validate_execution_lease(lease, task_id="s1", attempt_id="attempt_1", fencing_token="wrong")
    assert "lease executor epoch mismatch" in validate_execution_lease(
        lease, task_id="s1", attempt_id="attempt_1", fencing_token="token", executor_epoch=5
    )


def test_preflight_receipt_blocks_pair_when_cutie_proof_is_missing(tmp_path):
    rgb, prompts = inputs(tmp_path)
    rgb_path, prompt_path = tmp_path / "rgb.json", tmp_path / "prompts.json"
    write_json(rgb_path, rgb)
    write_json(prompt_path, prompts)
    make_sam_asset(tmp_path)
    receipt = build_preflight_receipt(
        rgb_path=rgb_path,
        prompt_path=prompt_path,
        requested_backends=["sam2_1", "cutie"],
        artifact_revision="R7_1",
        attempt_id="attempt_0001",
        executor_epoch=7,
        fencing_token="secret",
        repository_root=tmp_path,
        verify_large_sha=True,
    )
    assert receipt["payload"]["status"] == "BLOCKED_REFERENCE_PROOF"
    assert receipt["payload"]["execution_performed"] is False
    assert "secret" not in json.dumps(receipt)
    schema = json.loads((ROOT / "contracts/causal_modal_mask_challenger_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(receipt)


def test_cli_preflight_is_immutable_and_never_runs_inference(tmp_path):
    rgb, prompts = inputs(tmp_path)
    rgb_path, prompt_path = tmp_path / "rgb.json", tmp_path / "prompts.json"
    write_json(rgb_path, rgb)
    write_json(prompt_path, prompts)
    output = tmp_path / "attempts"
    argv = [
        "--rgb-manifest", str(rgb_path), "--prompt-manifest", str(prompt_path),
        "--output-root", str(output), "--attempt-id", "attempt_0001",
        "--executor-epoch", "1", "--fencing-token", "token",
        "--backend", "cutie", "--preflight-only",
    ]
    assert main(argv) == 0
    receipt_path = output / "sessions/chips_fixture/attempts/attempt_0001/RESULT.json"
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text())["payload"]["execution_performed"] is False
    with pytest.raises(FileExistsError):
        main(argv)
