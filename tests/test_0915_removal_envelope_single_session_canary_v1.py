from __future__ import annotations

import json

import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops import run_0915_removal_envelope_single_session_canary_v1 as subject


def test_task_is_weightless_cpu_only_and_bounded() -> None:
    packet = build_packet(subject.TASK_ID)
    assert packet["weights"] == "ABSENT"
    assert packet["budgets"] == {"gpu_hours": 0, "runtime_attempts": 1}
    assert len(packet["read_set"]) == 8
    encoded = str(packet)
    assert "no_SAM_rerun" in encoded
    assert "no_inpaint" in encoded
    assert "no inpaint or batch" in packet["claim_limit"].lower()


def test_fixed_inputs_share_the_same_frame_and_image_domain() -> None:
    assert subject.VIDEO.is_file()
    assert subject.HAWOR.is_file()
    with np.load(subject.HAWOR, allow_pickle=False) as archive:
        assert archive["joints_2d"].shape == (2, 150, 21, 2)
        assert archive["observed"].shape == (2, 150)
        assert archive["anatomical_side_names"].astype(str).tolist() == ["left", "right"]
    for names in subject.STRICT_MASKS.values():
        for name in names:
            subject._load_packed(subject.strict_mask(name))
    for names in subject.WEAK_MASKS.values():
        for name in names:
            subject._load_packed(subject.weak_mask(name))


def test_packed_mask_roundtrip_is_exact() -> None:
    mask = np.zeros((subject.HEIGHT, subject.WIDTH), bool)
    mask[12:34, 56:89] = True
    assert np.array_equal(subject._unpack(subject._pack(mask)), mask)


def test_cable_color_is_a_replaceable_config_profile() -> None:
    raw = json.loads(subject.CONFIG_PATH.read_text())
    config = subject.load_config()
    assert config.cable_profile.profile_id == raw["cable_profile"]["profile_id"]
    assert raw["cable_profile"]["profile_semantics"] == (
        "CURRENT_INSTANCE_APPEARANCE_PRIOR_NOT_CABLE_CLASS_DEFINITION"
    )
    assert "yellow" not in type(config.cable_profile).__name__.lower()


def test_contract_requires_separate_layers_and_consumer_firewall() -> None:
    schema = json.loads(subject.SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    required_layers = set(schema["properties"]["layers"]["required"])
    assert required_layers == {
        "raw_candidate", "semantic_role", "removal_envelope", "feather_alpha",
    }
    forbidden = schema["properties"]["consumer_firewall"]["properties"]["forbidden_consumers"]["const"]
    assert forbidden == [
        "DEPTH", "OBJECT6D", "CONTACT", "ROBOT_GEOMETRY", "CONTROL_GROUND_TRUTH",
    ]


def test_runner_does_not_import_or_invoke_sam_or_inpaint() -> None:
    source = open(subject.__file__, encoding="utf-8").read().lower()
    assert "sam3_multiplex" not in source
    assert "propainter" not in source
    assert "lama" not in source
    assert "torch" not in source
