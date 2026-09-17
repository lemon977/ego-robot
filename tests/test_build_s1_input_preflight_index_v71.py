from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.build_s1_input_preflight_index_v71 import (
    audit_sessions,
    build_prompt_manifest,
    build_rgb_manifest,
    first_reliable_prompts,
)
from chaoyang.pipeline.causal_modal_mask_challenger_v71 import (
    validate_prompt_manifest,
    validate_rgb_manifest,
)


def ev(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_object_manifest(tmp_path: Path, task: str, *, missing: int | None = None):
    expected = range(3) if task == "CHIPS" else range(1)
    physical = {}
    for physical_id in expected:
        mask = tmp_path / f"mask_{physical_id}.png"
        mask.write_bytes(f"mask-{physical_id}".encode())
        physical[str(physical_id)] = {
            "physical_instance_id": physical_id,
            "observed": physical_id != missing,
            "valid": physical_id != missing,
            "area_px": 10 if physical_id != missing else 0,
            "mask": ev(mask),
        }
    return {"frames": [{"source_frame": 0, "physical_instances": physical}]}


def test_first_prompt_is_per_instance_and_never_future_backpropagated(tmp_path):
    prompts, error = first_reliable_prompts(
        object_manifest=make_object_manifest(tmp_path, "CHIPS"), task="CHIPS"
    )
    assert error is None
    assert [item["instance_id"] for item in prompts] == ["chips_1", "chips_2", "chips_3"]
    assert all(item["prompt_events"][0]["frame_id"] == 0 for item in prompts)
    assert len({item["prompt_events"][0]["mask"]["path"] for item in prompts}) == 3


def test_missing_chips_instance_fails_closed(tmp_path):
    prompts, error = first_reliable_prompts(
        object_manifest=make_object_manifest(tmp_path, "CHIPS", missing=2), task="CHIPS"
    )
    assert prompts == []
    assert error == "NO_RELIABLE_INITIAL_MASK_FOR_INSTANCE_2"


def test_generated_manifests_validate_and_pin_audit(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for frame_id in range(2):
        (frame_dir / f"{frame_id:05d}.png").write_bytes(f"rgb-{frame_id}".encode())
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "immutable": True, "authority": False, "gold_accuracy_authorized": False,
        "rows": [{"session_id": "fixture"}],
    }))
    obj = tmp_path / "object.json"
    obj.write_text("{}")
    rgb = build_rgb_manifest(
        session_id="fixture", frame_directory=frame_dir, frame_count=2,
        source_video=ev(video), source_kind="FIXTURE",
    )
    prompt_rows, _ = first_reliable_prompts(
        object_manifest=make_object_manifest(tmp_path, "POKER"), task="POKER"
    )
    prompts = build_prompt_manifest(
        session_id="fixture", task="POKER", prompts=prompt_rows,
        audit_evidence=ev(audit), object_evidence=ev(obj),
    )
    assert validate_rgb_manifest(rgb) == []
    assert validate_prompt_manifest(prompts, rgb) == []
    assert prompts["prompt_policy"]["future_frame_prompt_forbidden"] is True
    assert prompts["prompt_policy"]["initial_mask_is_not_back_propagated_to_earlier_frames"] is True


def test_audit_session_grouping_is_deterministic():
    audit = {
        "immutable": True,
        "authority": False,
        "rows": [
            {"session_id": "b", "task": "poker", "source_manifest": {"path": "/b"},
             "object_mask_grade": "B", "group_id": "g2"},
            {"session_id": "a", "task": "chips", "source_manifest": {"path": "/a"},
             "object_mask_grade": "B", "group_id": "g1"},
        ]
    }
    assert [item["session_id"] for item in audit_sessions(audit)] == ["a", "b"]


def test_repository_index_schema_accepts_final_output_when_present():
    path = ROOT / (
        "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/"
        "s1_input_preflight/S1_INPUT_PREFLIGHT_INDEX.json"
    )
    if not path.is_file():
        return
    schema = json.loads((ROOT / "contracts/s1_input_preflight_index_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(json.loads(path.read_text()))
