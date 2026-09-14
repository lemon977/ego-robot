from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import prepare_exact78_clean_expanded_role_v3 as producer


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_mask(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value.astype(np.uint8) * 255)
    return producer.artifact(path)


def test_accepted_poker042_parameters_are_still_the_v3_reuse_basis() -> None:
    value = producer.validate_baseline_reuse()
    assert value["reused_expansion"] == producer.EXPANSION
    assert value["reused_propainter_parameters"] == producer.PROPAINTER_PARAMETERS


def test_handoff_expands_roles_but_never_deletes_observed_object(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    role_frames = []
    task_frames = []
    shape = (17, 19)
    for frame_id in range(3):
        image = np.full((*shape, 3), 40 + frame_id, np.uint8)
        rgb_path = raw_root / f"{frame_id:05d}" / "rgb.png"
        rgb_path.parent.mkdir(parents=True)
        assert cv2.imwrite(str(rgb_path), image)
        roles = {}
        for role in ("left_human", "right_human", "left_tracker", "right_tracker"):
            value = np.zeros(shape, bool)
            value[8, 8] = True
            roles[role] = write_mask(tmp_path / "roles" / role / f"{frame_id:05d}.png", value)
        object_mask = np.zeros(shape, bool)
        object_mask[8, 8] = True
        task_ref = write_mask(tmp_path / "object" / f"{frame_id:05d}.png", object_mask)
        role_frames.append(
            {
                "source_frame": frame_id,
                "selected_rgb_decoded_sha256": producer.array_sha256(image),
                "role_masks": roles,
            }
        )
        task_frames.append(
            {
                "source_frame": frame_id,
                "selected_rgb_decoded_sha256": f"video-domain-{frame_id}",
                "physical_instances": {
                    "0": {"observed": True, "valid": True, "mask": task_ref}
                },
            }
        )
    role_manifest_path = tmp_path / "ROLE.json"
    task_manifest_path = tmp_path / "OBJECT.json"
    write_json(role_manifest_path, {"frames": role_frames})
    write_json(task_manifest_path, {"frames": task_frames})
    context = {
        "session": "play_cards_fixture",
        "task": "poker",
        "frame_count": 3,
        "fps": 30.0,
        "source_resolution": [shape[1], shape[0]],
        "raw_root": raw_root,
        "raw_video": {"path": "/fixture/video", "bytes": 1, "sha256": "x"},
        "role_result": {"path": "/fixture/role", "bytes": 1, "sha256": "r"},
        "task_object_result": {"path": "/fixture/object", "bytes": 1, "sha256": "o"},
        "object6d_result": {"path": "/fixture/object6d", "bytes": 1, "sha256": "6"},
        "role_manifest_path": role_manifest_path,
        "task_manifest_path": task_manifest_path,
        "role_manifest": {"frames": role_frames},
        "task_manifest": {"frames": task_frames},
        "observed_object_frames": 3,
    }
    temporary = tmp_path / "partial"
    final = tmp_path / "final"
    temporary.mkdir()
    handoff = producer.make_handoff(context, temporary, final)
    manifest = json.loads(
        (temporary / "sessions/play_cards_fixture/expanded_role_handoff/FRAME_MANIFEST.json").read_text()
    )
    removal_path = temporary / "sessions/play_cards_fixture/expanded_role_handoff/expanded_clean_removal/00000.png"
    removal = cv2.imread(str(removal_path), cv2.IMREAD_GRAYSCALE) > 0
    assert removal.sum() > 1
    assert not removal[8, 8]
    assert all(row["published_removal_object_overlap_pixels"] == 0 for row in manifest["frames"])
    assert handoff["metrics"]["protected_object_pixels"] == 3


def test_checked_ref_rejects_byte_tamper(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text("{}\n", encoding="utf-8")
    reference = producer.artifact(path)
    path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(producer.PrepareError, match="path/bytes/SHA mismatch"):
        producer.checked_ref(reference, "fixture")


def test_prepare_is_strictly_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(producer.PrepareError, match="no-clobber"):
        producer.prepare({"sessions": []}, output)
