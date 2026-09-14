from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json

import cv2
import numpy as np
import pytest

from training.FlowMatchingEvaluator import safe_imread_rgb as evaluator_imread_rgb
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions
from utils.utils_io import read_json, safe_imread_gray, safe_imread_rgb
import utils.source_contract as source_contract
import utils.frozen_contract as frozen_contract
import training.FlowMatchingDataloader as dataloader_module
from utils.source_contract import (
    ELIGIBLE68,
    ELIGIBLE68_FRAME_COUNTS,
    FORMAL_SELECTOR_HOLD,
    require_manifest_selector_ready,
    validate_frame_images,
    validate_selector_manifest_reference,
)


def file_ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_required_images_never_fall_back_to_black(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError):
        safe_imread_rgb(missing, 8, 8)
    with pytest.raises(FileNotFoundError):
        safe_imread_gray(missing, 8, 8)
    with pytest.raises(FileNotFoundError):
        evaluator_imread_rgb(missing, (8, 8))


def test_corrupt_images_and_json_fail_closed(tmp_path: Path) -> None:
    corrupt_image = tmp_path / "corrupt.png"
    corrupt_image.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot decode"):
        safe_imread_rgb(str(corrupt_image), 8, 8)

    corrupt_json = tmp_path / "frame.json"
    corrupt_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid required JSON"):
        read_json(str(corrupt_json))


def test_required_image_is_decoded_and_resized(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(path), np.full((4, 6, 3), 127, dtype=np.uint8))
    loaded = safe_imread_rgb(str(path), 8, 10)
    assert loaded.shape == (8, 10, 3)


def test_augmentation_rng_is_seeded_and_sample_specific() -> None:
    dataset = object.__new__(FlowMatchingDataloader)
    dataset.seed = 7
    dataset.epoch = 0
    dataset.use_legacy_rng = False
    first = dataset._rng_for_sample(4, 123).randn(8)
    repeated = dataset._rng_for_sample(4, 123).randn(8)
    other = dataset._rng_for_sample(5, 123).randn(8)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    dataset.set_epoch(1)
    next_epoch = dataset._rng_for_sample(4, 123).randn(8)
    assert not np.array_equal(first, next_epoch)


def test_augmentation_rng_uses_paired_logical_key_not_absolute_root() -> None:
    datasets = []
    for prefix in ("raw", "robot"):
        dataset = object.__new__(FlowMatchingDataloader)
        dataset.seed = 11
        dataset.epoch = 3
        dataset.use_legacy_rng = False
        dataset.samples = [
            f"/tmp/{prefix}/grap_a_cap_004/09_humanego_adapter/"
            "preprocess/all_data/00017/training_data.json"
        ]
        datasets.append(dataset)
    np.testing.assert_array_equal(
        datasets[0]._rng_for_sample(0, 9973).randn(8),
        datasets[1]._rng_for_sample(0, 9973).randn(8),
    )


def test_dataloader_consumes_only_paired_window_starts(tmp_path: Path) -> None:
    adapter = tmp_path / "grap_a_cap_004/09_humanego_adapter"
    for frame in (0, 1):
        path = adapter / f"preprocess/all_data/{frame:05d}/training_data.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    dataset = FlowMatchingDataloader(
        sessions=[MPSSessions(str(adapter))],
        single_hand=False,
        allowed_window_starts={"grap_a_cap_004": {1}},
    )
    assert [Path(path).parent.name for path in dataset.samples] == ["00001"]
    with pytest.raises(ValueError, match="window starts are not consumable"):
        FlowMatchingDataloader(
            sessions=[MPSSessions(str(adapter))],
            single_hand=False,
            allowed_window_starts={"grap_a_cap_004": {2}},
        )


def test_image_selector_cannot_escape_frame_directory() -> None:
    with pytest.raises(ValueError, match="frame-local basename"):
        FlowMatchingDataloader(sessions=[], img_name="../../legacy/rgb.png")


def test_frame_image_contract_detects_deleted_selector_target(tmp_path: Path) -> None:
    frame = tmp_path / "preprocess/all_data/00000"
    frame.mkdir(parents=True)
    raw = tmp_path / "raw/rgb.png"
    raw.parent.mkdir()
    (frame / "training_data.json").write_text(
        json.dumps({"obs": {"rgb_path": str(raw)}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="HOLD_MANIFEST_SELECTOR_REQUIRED"):
        validate_frame_images(tmp_path, "rgb.png")

    assert cv2.imwrite(str(raw), np.full((4, 6, 3), 127, dtype=np.uint8))
    metadata = frame / "training_data.json"
    records = {"00000": {"metadata": file_ref(metadata), "image": file_ref(raw)}}
    report = validate_frame_images(
        tmp_path,
        "rgb.png",
        selector_records=records,
        selector_root=raw.parent,
    )
    assert report["frames"] == 1
    assert report["manifest_bound"] is True

    metadata.write_text(
        json.dumps({"obs": {"rgb_path": str(raw)}, "changed": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="(byte-size|SHA256) mismatch"):
        validate_frame_images(
            tmp_path,
            "rgb.png",
            selector_records=records,
            selector_root=raw.parent,
        )
    metadata.write_text(json.dumps({"obs": {"rgb_path": str(raw)}}), encoding="utf-8")
    raw.write_bytes(b"changed after manifest")
    with pytest.raises(ValueError, match="(byte-size|SHA256) mismatch"):
        validate_frame_images(
            tmp_path,
            "rgb.png",
            selector_records=records,
            selector_root=raw.parent,
        )


def write_formal_robot_lineage(
    tmp_path: Path,
    selector_frames: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Create byte-real fixture bytes; they are untrusted until explicitly pinned."""
    session_id = "grap_a_cap_004"
    lineage_root = tmp_path / "formal_robot_lineage"
    evidence_root = lineage_root / "evidence"
    evidence_root.mkdir(parents=True)
    stage_references: dict[str, dict[str, object]] = {}
    for stage in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS:
        payload = {
            "schema_version": source_contract.ROBOT_RGB_STAGE_SCHEMA,
            "lineage_stage": stage,
            **source_contract._FORMAL_ROBOT_RGB_STATE,
            "session_id": session_id,
            "frame_count": len(selector_frames),
        }
        payload.update(
            {
                source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS[upstream_stage]: dict(
                    stage_references[upstream_stage]
                )
                for upstream_stage in source_contract.ROBOT_RGB_STAGE_UPSTREAM_STAGES[
                    stage
                ]
            }
        )
        if stage == "S8":
            outputs = {
                frame_key: dict(record["image"])
                for frame_key, record in selector_frames.items()
            }
            payload.update(
                {
                    "s8_authority": "FORMAL_S8_ROBOT_RGB",
                    "robot_rgb_outputs": outputs,
                    "success_evidence": list(outputs.values()),
                }
            )
        else:
            evidence_path = evidence_root / f"{stage.lower()}.bin"
            evidence_path.write_bytes(f"formal-{stage}".encode("ascii"))
            payload["success_evidence"] = [file_ref(evidence_path)]
            if stage == "SCENE_STATE":
                payload.update(
                    {
                        "scene_state_mode": "EXTERNAL_FORMAL_AUTHORITY",
                        "q_arm_and_camera_base_authoritative": True,
                    }
                )
            elif stage == "MOUNT_AUTHORITY":
                payload.update(
                    {
                        "mount_provenance": ("INDEPENDENT_BILATERAL_FORMAL_MOUNT"),
                        "independent_measurement": True,
                        "selected_by_ik_residual": False,
                    }
                )
            elif stage == "EEVEE_RENDER":
                payload["engine"] = "BLENDER_EEVEE_NEXT"
            elif stage == "CLEAN":
                payload["clean_authority"] = "FORMAL_CLEAN"
        stage_path = lineage_root / f"{stage.lower()}.json"
        stage_path.write_text(json.dumps(payload), encoding="utf-8")
        stage_references[stage] = file_ref(stage_path)

    session_row = {
        "frame_count": len(selector_frames),
        **{
            field: stage_references[stage]
            for stage, field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.items()
        },
    }
    authority = {
        "schema_version": source_contract.ROBOT_RGB_LINEAGE_SCHEMA,
        **source_contract._FORMAL_ROBOT_RGB_STATE,
        "cohort": "eligible68",
        "session_count": 1,
        "frame_count": len(selector_frames),
        "success_evidence": [
            session_row[field]
            for field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.values()
        ],
        "sessions": {session_id: session_row},
    }
    authority_path = lineage_root / "ROBOT_RGB_LINEAGE_AUTHORITY.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    return file_ref(authority_path)


def review_robot_lineage(robot_path: Path) -> None:
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF = dict(
        robot["robot_rgb_lineage_authority_ref"]
    )


def rewrite_robot_lineage_stage(
    robot_path: Path,
    stage: str,
    mutate,
) -> tuple[dict, dict, Path]:
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    session_id = "grap_a_cap_004"
    field = source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS[stage]
    stage_path = Path(authority["sessions"][session_id][field]["path"])
    stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
    mutate(stage_payload)
    stage_path.write_text(json.dumps(stage_payload), encoding="utf-8")
    authority["sessions"][session_id][field] = file_ref(stage_path)
    authority["success_evidence"] = [
        authority["sessions"][session_id][reference_field]
        for reference_field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.values()
    ]
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    robot["robot_rgb_lineage_authority_ref"] = file_ref(authority_path)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    return robot, authority, stage_path


def selector_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trust_robot_lineage: bool = True,
) -> tuple[dict, dict, Path, Path, Path]:
    monkeypatch.setattr(source_contract, "ELIGIBLE68", frozenset({"grap_a_cap_004"}))
    monkeypatch.setattr(
        source_contract, "ELIGIBLE68_FRAME_COUNTS", {"grap_a_cap_004": 51}
    )
    raw_root = tmp_path / "raw_artifacts"
    robot_root = tmp_path / "robot_artifacts"
    raw_root.mkdir()
    robot_root.mkdir()
    frames = {
        f"{frame:05d}": {
            "metadata": {
                "path": str(
                    (
                        tmp_path / f"production/grap_a_cap_004/09_humanego_adapter/"
                        f"preprocess/all_data/{frame:05d}/training_data.json"
                    ).resolve()
                ),
                "bytes": 1,
                "sha256": "0" * 64,
            },
            "image": {
                "path": str(
                    (raw_root / f"grap_a_cap_004/{frame:05d}/rgb.png").resolve()
                ),
                "bytes": 1,
                "sha256": "1" * 64,
            },
            # Deliberately asymmetric.  This remains in the paired cohort.
            "unresolved": False,
        }
        for frame in range(51)
    }
    raw = {
        "schema_version": "humanego-selector-manifest-v1",
        "immutable": True,
        "no_fallback": True,
        "product_line": "RAW",
        "image_name": "rgb.png",
        "artifact_root": str(raw_root.resolve()),
        "selector_root": str(raw_root.resolve()),
        "sessions": {"grap_a_cap_004": {"frames": frames}},
    }
    robot = json.loads(json.dumps(raw))
    robot["product_line"] = "ROBOT_RGB"
    robot["image_name"] = "robot.png"
    robot["artifact_root"] = str(robot_root.resolve())
    robot["selector_root"] = str(robot_root.resolve())
    for frame_key, record in robot["sessions"]["grap_a_cap_004"]["frames"].items():
        robot_image = robot_root / f"grap_a_cap_004/{frame_key}/robot.png"
        robot_image.parent.mkdir(parents=True)
        robot_image.write_bytes(f"robot-rgb-{frame_key}".encode("ascii"))
        record["image"] = {
            **file_ref(robot_image),
        }
        record["unresolved"] = False
    robot["robot_rgb_lineage_authority_ref"] = write_formal_robot_lineage(
        tmp_path,
        robot["sessions"]["grap_a_cap_004"]["frames"],
    )
    if trust_robot_lineage:
        monkeypatch.setattr(
            source_contract,
            "REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF",
            dict(robot["robot_rgb_lineage_authority_ref"]),
        )
    raw_path = tmp_path / "RAW_SELECTOR.json"
    robot_path = tmp_path / "ROBOT_SELECTOR.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    paired = {
        "schema_version": "humanego-paired-kept-manifest-v1",
        "immutable": True,
        "no_fallback": True,
        "cohort": "eligible68",
        "selection_policy": "SYMMETRIC_FRAME_AND_WINDOW_INTERSECTION",
        "unresolved_policy": "RETAIN_PAIRED_FRAME_REGARDLESS_OF_DOMAIN_UNRESOLVED",
        "selector_refs": {"RAW": file_ref(raw_path), "ROBOT_RGB": file_ref(robot_path)},
        "artifact_roots": {
            "RAW": str(raw_root.resolve()),
            "ROBOT_RGB": str(robot_root.resolve()),
        },
        "sessions": {
            "grap_a_cap_004": {
                "frames": list(range(51)),
                "window_starts": [0],
            }
        },
    }
    paired_path = tmp_path / "PAIRED_KEPT_MANIFEST.json"
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    return raw, paired, raw_path, robot_path, paired_path


def test_exact_eligible68_literal_scope() -> None:
    assert len(ELIGIBLE68) == len(ELIGIBLE68_FRAME_COUNTS) == 68
    assert set(ELIGIBLE68_FRAME_COUNTS) == ELIGIBLE68
    assert sum(ELIGIBLE68_FRAME_COUNTS.values()) == 28_265
    assert "grap_a_cap_025" not in ELIGIBLE68
    ranked = sorted(
        source_contract.TRAIN60,
        key=lambda session_id: hashlib.sha256(
            f"eligible68-dev-v1|7|{session_id}".encode("utf-8")
        ).hexdigest(),
    )
    assert tuple(ranked[:8]) == source_contract.DEV8
    assert len(source_contract.TRAIN52) == 52
    assert (
        sum(
            source_contract.ELIGIBLE68_FRAME_COUNTS[value]
            for value in source_contract.TRAIN52
        )
        == 21_740
    )
    assert (
        sum(
            source_contract.ELIGIBLE68_FRAME_COUNTS[value]
            for value in source_contract.DEV8
        )
        == 3_435
    )
    assert (
        sum(
            source_contract.ELIGIBLE68_FRAME_COUNTS[value]
            for value in source_contract.FINAL_TEST8
        )
        == 3_090
    )


def test_incomplete_selector_cannot_redefine_eligible68(tmp_path: Path) -> None:
    payload = {
        "schema_version": "humanego-selector-manifest-v1",
        "immutable": True,
        "no_fallback": True,
        "product_line": "RAW",
        "image_name": "rgb.png",
        "artifact_root": str(tmp_path.resolve()),
        "selector_root": str(tmp_path),
        "sessions": {"grap_a_cap_004": {"frames": {}}},
    }
    path = tmp_path / "INCOMPLETE.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact eligible68"):
        validate_selector_manifest_reference(file_ref(path), project_root=tmp_path)


def test_legacy_split_is_rejected_before_any_session_source_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_root = tmp_path / "HumanEgo/data_manifests"
    split_root.mkdir(parents=True)
    split_path = split_root / "legacy_78.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "humanego-eligible68-split-v1",
                "train": [
                    *source_contract.TRAIN60,
                    "grap_a_cap_025",
                    "grap_a_cap_149",
                ],
                "validation": list(source_contract.VALIDATION8),
                "test": [
                    "grap_a_cap_012",
                    "grap_a_cap_052",
                    "grap_a_cap_055",
                    "grap_a_cap_070",
                    "grap_a_cap_087",
                    "grap_a_cap_113",
                    "grap_a_cap_119",
                    "grap_a_cap_143",
                ],
            }
        ),
        encoding="utf-8",
    )
    session_source_loader_calls = 0
    forbidden_session_sidecar_open_count = 0
    original_path_open = Path.open

    def spy_path_open(path: Path, *args, **kwargs):
        nonlocal forbidden_session_sidecar_open_count
        if path.name == "sidecar.npz" and any(
            part in source_contract.FORBIDDEN10 for part in path.parts
        ):
            forbidden_session_sidecar_open_count += 1
        return original_path_open(path, *args, **kwargs)

    def forbidden_downstream_loader(*args, **kwargs):
        nonlocal session_source_loader_calls
        session_source_loader_calls += 1
        raise AssertionError("phase 2 must not start")

    monkeypatch.setattr(Path, "open", spy_path_open)
    monkeypatch.setattr(
        source_contract, "load_frozen_split", forbidden_downstream_loader
    )
    with pytest.raises(ValueError, match="not exact eligible68"):
        source_contract.load_eligible68_frozen_split(
            split_path,
            "kai22",
            verify_sidecars=True,
            project_root=tmp_path,
        )
    assert session_source_loader_calls == 0
    assert forbidden_session_sidecar_open_count == 0


def test_exact_eligible68_phase1_can_hand_off_to_phase2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_root = tmp_path / "HumanEgo/data_manifests"
    production_root = tmp_path / "production"
    sidecar_root = tmp_path / "sidecars"
    split_root.mkdir(parents=True)
    production_root.mkdir()
    sidecar_root.mkdir()
    sessions = {
        session_id: {
            "adapter": str(production_root / session_id / "09_humanego_adapter"),
            "production_status_sha256": "0" * 64,
            "embodiments": {
                "kai22": {
                    "frames": source_contract.ELIGIBLE68_FRAME_COUNTS[session_id],
                    "sidecar_sha256": "1" * 64,
                }
            },
        }
        for session_id in source_contract.ELIGIBLE68_ORDER
    }
    payload = {
        "schema_version": source_contract.ELIGIBLE68_SPLIT_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        **source_contract.eligible68_split_protocol_fields(),
        "train": list(source_contract.TRAIN52),
        "validation": list(source_contract.DEV8),
        "test": list(source_contract.FINAL_TEST8),
        "sessions": sessions,
        "production_root": str(production_root),
        "sidecar_root": str(sidecar_root),
    }
    split_path = split_root / "eligible68.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    phase2_calls = 0

    def downstream_loader(*args, **kwargs):
        nonlocal phase2_calls
        phase2_calls += 1
        return payload

    monkeypatch.setattr(source_contract, "load_frozen_split", downstream_loader)
    observed = source_contract.load_eligible68_frozen_split(
        split_path,
        "kai22",
        sidecar_root=sidecar_root,
        verify_sidecars=True,
        project_root=tmp_path,
    )
    assert observed is payload
    assert phase2_calls == 1


def test_eligible68_phase2_consumes_phase1_payload_not_replaced_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_root = tmp_path / "HumanEgo/data_manifests"
    production_root = tmp_path / "production"
    sidecar_root = tmp_path / "sidecars"
    split_root.mkdir(parents=True)
    production_root.mkdir()
    sidecar_root.mkdir()
    sidecar_digest = hashlib.sha256(b"sidecar").hexdigest()
    sessions = {}
    for session_id in source_contract.ELIGIBLE68_ORDER:
        sidecar = sidecar_root / "kai22" / session_id / "sidecar.npz"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(b"sidecar")
        sessions[session_id] = {
            "adapter": str(production_root / session_id / "09_humanego_adapter"),
            "production_status_sha256": "0" * 64,
            "embodiments": {
                "kai22": {
                    "frames": source_contract.ELIGIBLE68_FRAME_COUNTS[session_id],
                    "sidecar_sha256": sidecar_digest,
                }
            },
        }
    payload = {
        "schema_version": source_contract.ELIGIBLE68_SPLIT_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        **source_contract.eligible68_split_protocol_fields(),
        "train": list(source_contract.TRAIN52),
        "validation": list(source_contract.DEV8),
        "test": list(source_contract.FINAL_TEST8),
        "sessions": sessions,
        "production_root": str(production_root),
        "sidecar_root": str(sidecar_root),
    }
    split_path = split_root / "eligible68.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    forbidden = "grap_a_cap_025"
    replacement = json.loads(json.dumps(payload))
    replacement["train"].append(forbidden)
    replacement["sessions"][forbidden] = {
        "adapter": str(production_root / forbidden / "09_humanego_adapter"),
        "production_status_sha256": "0" * 64,
        "embodiments": {"kai22": {"frames": 1, "sidecar_sha256": sidecar_digest}},
    }
    original_loader = source_contract.load_frozen_split
    original_sha256_file = frozen_contract.sha256_file
    hashed_sidecar_sessions: list[str] = []

    def spy_sha256_file(path: Path) -> str:
        if path.name == "sidecar.npz":
            hashed_sidecar_sessions.append(path.parent.name)
        return original_sha256_file(path)

    def replace_before_phase2(*args, **kwargs):
        split_path.write_text(json.dumps(replacement), encoding="utf-8")
        assert kwargs["split_payload"] == payload
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        source_contract,
        "load_frozen_split",
        replace_before_phase2,
    )
    monkeypatch.setattr(frozen_contract, "sha256_file", spy_sha256_file)
    observed = source_contract.load_eligible68_frozen_split(
        split_path,
        "kai22",
        verify_sidecars=True,
        project_root=tmp_path,
    )
    assert observed["train"] == list(source_contract.TRAIN52)
    assert forbidden not in observed["sessions"]
    assert not (sidecar_root / "kai22" / forbidden / "sidecar.npz").exists()
    assert set(hashed_sidecar_sessions) == set(source_contract.TRAIN52) | set(
        source_contract.DEV8
    )
    assert not set(hashed_sidecar_sessions) & set(source_contract.FINAL_TEST8)


def test_selector_and_paired_kept_enable_symmetric_cpu_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    binding = require_manifest_selector_ready(
        file_ref(raw_path),
        file_ref(paired_path),
        artifact_root=tmp_path / "raw_artifacts",
        project_root=tmp_path,
    )
    assert binding["product_line"] == "RAW"
    assert binding["frame_ledger"]["grap_a_cap_004"] == frozenset(range(51))
    assert binding["window_starts"]["grap_a_cap_004"] == frozenset({0})


def test_selector_missing_or_staged_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match=FORMAL_SELECTOR_HOLD):
        require_manifest_selector_ready(project_root=tmp_path)
    _, _, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    staging = tmp_path / "_run"
    staging.mkdir()
    staged_manifest = staging / "candidate.json"
    staged_manifest.write_bytes(raw_path.read_bytes())
    with pytest.raises(ValueError, match="cannot be consumed from _run"):
        require_manifest_selector_ready(
            file_ref(staged_manifest),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_nested_selector_source_cannot_come_from_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, paired, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    raw["sessions"]["grap_a_cap_004"]["frames"]["00000"]["image"]["path"] = str(
        tmp_path / "_run/candidate.png"
    )
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    paired["selector_refs"]["RAW"] = file_ref(raw_path)
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot come from _run staging"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_selector_frame_or_window_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, paired, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    paired["sessions"]["grap_a_cap_004"]["frames"].pop()
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="window escapes frame ledger"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_robot_domain_cannot_alias_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, paired, raw_path, robot_path, paired_path = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    robot["image_name"] = "rgb.png"
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    paired["selector_refs"]["ROBOT_RGB"] = file_ref(robot_path)
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot bind the RAW"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_robot_domain_cannot_reuse_raw_image_content_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, paired, raw_path, robot_path, paired_path = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    for record in robot["sessions"]["grap_a_cap_004"]["frames"].values():
        record["image"]["sha256"] = "1" * 64
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    paired["selector_refs"]["ROBOT_RGB"] = file_ref(robot_path)
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="(selector/S8 output differs|image domain equals RAW)",
    ):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_raw_selector_remains_backward_compatible_without_robot_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _, raw_path, _, _ = selector_pair(tmp_path, monkeypatch)
    assert "robot_rgb_lineage_authority_ref" not in raw
    observed = validate_selector_manifest_reference(
        file_ref(raw_path),
        artifact_root=tmp_path / "raw_artifacts",
        project_root=tmp_path,
    )
    assert observed["product_line"] == "RAW"


def test_self_asserted_robot_lineage_holds_without_reviewed_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(
        tmp_path,
        monkeypatch,
        trust_robot_lineage=False,
    )
    assert source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF is None
    with pytest.raises(RuntimeError, match=source_contract.ROBOT_RGB_LINEAGE_HOLD):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_reasserted_candidate_copied_out_of_run_is_not_reviewed_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    reviewed_reference = dict(
        source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF or {}
    )
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    # This is the attacker case: after copying a candidate out of `_run`, all
    # flags and hashes can be rebuilt to look formal.  It still is not the
    # separately owner-reviewed exact publisher reference.
    reasserted_reference = write_formal_robot_lineage(
        tmp_path / "candidate_copied_out_of_run",
        robot["sessions"]["grap_a_cap_004"]["frames"],
    )
    assert reasserted_reference != reviewed_reference
    assert "_run" not in Path(str(reasserted_reference["path"])).parts
    robot["robot_rgb_lineage_authority_ref"] = reasserted_reference
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="differs from the owner-reviewed publisher authority",
    ):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


@pytest.mark.parametrize("inline_fake", [False, True])
def test_robot_selector_requires_external_aggregate_lineage_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inline_fake: bool,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    robot.pop("robot_rgb_lineage_authority_ref")
    if inline_fake:
        robot["robot_rgb_lineage_authority"] = {
            "status": "ARTIFACT_EXISTS",
            "formal_consumer_allowed": True,
            "self_attested": True,
        }
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    with pytest.raises(RuntimeError, match=source_contract.ROBOT_RGB_LINEAGE_HOLD):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_selector_rejects_inline_self_declared_lineage_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    robot["robot_rgb_lineage_authority"] = {
        "status": "ARTIFACT_EXISTS",
        "formal_consumer_allowed": True,
    }
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous self-declared lineage"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_candidate_stage_copied_out_of_staging_remains_nonformal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)

    def make_candidate(stage: dict) -> None:
        stage.update(
            {
                "status": "CANDIDATE_REQUIRES_HUMAN_REVIEW",
                "formal_consumer_allowed": False,
                "visual_only": True,
                "candidate_requires_human_review": True,
                "next_bucket_blocked": True,
                "advancement_authorized": False,
            }
        )

    _, _, stage_path = rewrite_robot_lineage_stage(
        robot_path,
        "EEVEE_RENDER",
        make_candidate,
    )
    assert "_run" not in stage_path.parts
    with pytest.raises(ValueError, match="status is not formal-consumable"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("completion_mode", "EXIT_ZERO"),
        ("development_only", True),
        ("provisional", True),
        ("visual_only", True),
        ("next_bucket_blocked", True),
        ("fallback_used", True),
        ("synthetic_fixture", True),
    ],
)
def test_robot_lineage_rejects_nonformal_or_self_attested_stage_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    rewrite_robot_lineage_stage(
        robot_path,
        "SCENE_STATE",
        lambda stage: stage.__setitem__(field, bad_value),
    )
    with pytest.raises(ValueError, match=f"{field} is not formal-consumable"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_requires_every_upstream_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["sessions"]["grap_a_cap_004"].pop("mount_authority_ref")
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    robot["robot_rgb_lineage_authority_ref"] = file_ref(authority_path)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match="session schema drift"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("stage", "upstream_field", "unrelated_field"),
    [
        ("EEVEE_RENDER", "scene_state_ref", "clean_ref"),
        ("EEVEE_RENDER", "mount_authority_ref", "clean_ref"),
        ("CLEAN", "eevee_render_ref", "scene_state_ref"),
        ("S8", "clean_ref", "scene_state_ref"),
    ],
)
def test_robot_stage_must_byte_join_each_upstream_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    upstream_field: str,
    unrelated_field: str,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    row = authority["sessions"]["grap_a_cap_004"]
    stage_field = source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS[stage]
    stage_path = Path(row[stage_field]["path"])
    stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
    stage_payload[upstream_field] = dict(row[unrelated_field])
    stage_path.write_text(json.dumps(stage_payload), encoding="utf-8")
    row[stage_field] = file_ref(stage_path)
    authority["success_evidence"] = [
        row[field]
        for field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.values()
    ]
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    robot["robot_rgb_lineage_authority_ref"] = file_ref(authority_path)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match="upstream .* differs from aggregate"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_stage_reference_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    row = authority["sessions"]["grap_a_cap_004"]
    row["clean_ref"]["sha256"] = "f" * 64
    authority["success_evidence"] = [
        row[field]
        for field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.values()
    ]
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    robot["robot_rgb_lineage_authority_ref"] = file_ref(authority_path)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_reopens_actual_s8_output_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    output = Path(
        robot["sessions"]["grap_a_cap_004"]["frames"]["00023"]["image"]["path"]
    )
    output.write_bytes(output.read_bytes() + b"post-publication-drift")
    with pytest.raises(ValueError, match="byte-size mismatch"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("session_count", 2, "session count must be exact eligible68"),
        ("frame_count", 50, "frame count must be exact 28265"),
        (
            "success_evidence",
            [],
            "aggregate success evidence must exactly bind",
        ),
    ],
)
def test_robot_aggregate_lineage_scope_and_evidence_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority[field] = bad_value
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    robot["robot_rgb_lineage_authority_ref"] = file_ref(authority_path)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match=message):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_selector_image_must_join_exact_s8_output_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    robot["sessions"]["grap_a_cap_004"]["frames"]["00017"]["image"]["sha256"] = "e" * 64
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    with pytest.raises(ValueError, match="selector/S8 output differs"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_s8_frame_ledger_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)

    def drop_frame(stage: dict) -> None:
        stage["robot_rgb_outputs"].pop("00050")
        stage["success_evidence"] = list(stage["robot_rgb_outputs"].values())

    rewrite_robot_lineage_stage(robot_path, "S8", drop_frame)
    with pytest.raises(ValueError, match="output frame ledger mismatch"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_stage_session_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    rewrite_robot_lineage_stage(
        robot_path,
        "CLEAN",
        lambda stage: stage.__setitem__("session_id", "grap_a_cap_005"),
    )
    with pytest.raises(ValueError, match="session mismatch"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_empty_success_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    rewrite_robot_lineage_stage(
        robot_path,
        "CLEAN",
        lambda stage: stage.__setitem__("success_evidence", []),
    )
    with pytest.raises(ValueError, match="success evidence must be non-empty"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_stage_path_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    row = authority["sessions"]["grap_a_cap_004"]
    row["clean_ref"] = dict(row["eevee_render_ref"])
    authority["success_evidence"] = [
        row[field]
        for field in source_contract.ROBOT_RGB_STAGE_REFERENCE_FIELDS.values()
    ]
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    robot["robot_rgb_lineage_authority_ref"] = file_ref(authority_path)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match="external path aliases logical records"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_noncanonical_reference_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_ref = robot["robot_rgb_lineage_authority_ref"]
    authority_path = Path(authority_ref["path"])
    alias_parent = authority_path.parent / "alias_parent"
    alias_parent.mkdir()
    authority_ref["path"] = str(alias_parent / ".." / authority_path.name)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_robot_lineage_rejects_hardlinked_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    hardlink = authority_path.with_name("ROBOT_RGB_LINEAGE_HARDLINK.json")
    hardlink.hardlink_to(authority_path)
    robot["robot_rgb_lineage_authority_ref"] = {
        "path": str(hardlink.resolve()),
        "bytes": hardlink.stat().st_size,
        "sha256": hashlib.sha256(hardlink.read_bytes()).hexdigest(),
    }
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match="singly-linked"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


@pytest.mark.parametrize("replacement", ["leaf", "ancestor"])
def test_robot_lineage_rejects_path_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    scene_ref = authority["sessions"]["grap_a_cap_004"]["scene_state_ref"]
    scene = json.loads(Path(scene_ref["path"]).read_text(encoding="utf-8"))
    target = Path(scene["success_evidence"][0]["path"])
    original_reader = source_contract.read_file_reference
    raced = False

    def replace_after_verified_read(*args, **kwargs):
        nonlocal raced
        result = original_reader(*args, **kwargs)
        if Path(args[0]["path"]) == target and not raced:
            raced = True
            payload = target.read_bytes()
            if replacement == "leaf":
                old = target.with_name(f"{target.name}.old")
                target.rename(old)
                target.write_bytes(payload)
            else:
                parent = target.parent
                old_parent = parent.with_name(f"{parent.name}_old")
                parent.rename(old_parent)
                parent.mkdir()
                target.write_bytes(payload)
        return result

    monkeypatch.setattr(
        source_contract,
        "read_file_reference",
        replace_after_verified_read,
    )
    with pytest.raises(ValueError, match="path changed around its verified read"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )
    assert raced is True


@pytest.mark.parametrize("namespace", ["_run", "labels", "grap_a_cap_025"])
def test_robot_lineage_rejects_external_forbidden_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
) -> None:
    _, _, _, robot_path, _ = selector_pair(tmp_path, monkeypatch)
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    authority_path = Path(robot["robot_rgb_lineage_authority_ref"]["path"])
    forbidden = tmp_path / namespace / authority_path.name
    forbidden.parent.mkdir()
    forbidden.write_bytes(authority_path.read_bytes())
    robot["robot_rgb_lineage_authority_ref"] = file_ref(forbidden)
    robot_path.write_text(json.dumps(robot), encoding="utf-8")
    review_robot_lineage(robot_path)
    with pytest.raises(ValueError, match="(forbidden.*namespace|_run staging)"):
        validate_selector_manifest_reference(
            file_ref(robot_path),
            artifact_root=tmp_path / "robot_artifacts",
            project_root=tmp_path,
        )


def test_selector_image_cannot_escape_manifest_root(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    frame = adapter / "preprocess/all_data/00000"
    frame.mkdir(parents=True)
    external = tmp_path / "external/rgb.png"
    external.parent.mkdir()
    assert cv2.imwrite(str(external), np.zeros((2, 2, 3), dtype=np.uint8))
    metadata = frame / "training_data.json"
    metadata.write_text(
        json.dumps({"obs": {"rgb_path": str(external)}}), encoding="utf-8"
    )
    records = {"00000": {"metadata": file_ref(metadata), "image": file_ref(external)}}
    with pytest.raises(ValueError, match="escapes approved roots"):
        validate_frame_images(
            adapter,
            "rgb.png",
            selector_records=records,
            selector_root=adapter,
        )

    adapter_link = tmp_path / "adapter_link"
    adapter_link.symlink_to(adapter, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        validate_frame_images(
            adapter_link,
            "rgb.png",
            selector_records=records,
            selector_root=external.parent,
        )


def test_dataloader_uses_exact_frame_image_reference_not_obs_basename(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "production/grap_a_cap_004/09_humanego_adapter"
    frame = adapter / "preprocess/all_data/00000"
    frame.mkdir(parents=True)
    decoy = tmp_path / "raw_source/00000/rgb.png"
    decoy.parent.mkdir(parents=True)
    selected = tmp_path / "external_artifacts/grap_a_cap_004/00000/rgb.png"
    selected.parent.mkdir(parents=True)
    assert cv2.imwrite(str(decoy), np.zeros((2, 2, 3), dtype=np.uint8))
    assert cv2.imwrite(str(selected), np.full((2, 2, 3), 255, dtype=np.uint8))
    metadata = frame / "training_data.json"
    payload = {"obs": {"rgb_path": str(decoy)}, "entities": {"hands": {}}}
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    dataset = FlowMatchingDataloader(
        sessions=[MPSSessions(str(adapter))],
        single_hand=False,
        img_name="rgb.png",
        selector_records={
            "grap_a_cap_004": {
                "00000": {
                    "metadata": file_ref(metadata),
                    "image": file_ref(selected),
                    "unresolved": False,
                }
            }
        },
        selector_root=str(tmp_path / "external_artifacts"),
    )
    assert dataset._resolve_image_path(payload, json_path=str(metadata)) == str(
        selected.resolve()
    )
    assert dataset._resolve_image_path(payload, json_path=str(metadata)) != str(decoy)
    selected.write_bytes(selected.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="byte-size mismatch"):
        dataset._load_image_tensor(payload, 0)


def test_dataloader_loads_the_exact_sidecar_bytes_that_were_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "grap_a_cap_004"
    adapter = tmp_path / "production" / session_id / "09_humanego_adapter"
    adapter.mkdir(parents=True)
    sidecar = tmp_path / "sidecars/kai22" / session_id / "sidecar.npz"
    sidecar.parent.mkdir(parents=True)

    def sidecar_bytes(value: float) -> bytes:
        encoded = io.BytesIO()
        np.savez(
            encoded,
            schema_version=np.asarray("humanego-robot-sidecar-v1"),
            embodiment=np.asarray("kai22"),
            frame_names=np.asarray(["00000"]),
            q=np.full((1, 2, 22), value, dtype=np.float32),
            valid=np.ones((1, 2), dtype=bool),
            confidence=np.ones((1, 2), dtype=np.float32),
            wrist_T_camera=np.broadcast_to(
                np.eye(4, dtype=np.float32), (1, 2, 4, 4)
            ).copy(),
            joint_names=np.asarray([[f"joint_{index:02d}" for index in range(22)]] * 2),
            joint_lower=np.full((2, 22), -2.0, dtype=np.float32),
            joint_upper=np.full((2, 22), 2.0, dtype=np.float32),
        )
        return encoded.getvalue()

    accepted = sidecar_bytes(0.25)
    replacement = sidecar_bytes(0.75)
    sidecar.write_bytes(accepted)
    accepted_digest = hashlib.sha256(accepted).hexdigest()
    original_reader = dataloader_module.read_ordinary_file_bytes

    def replace_after_verified_read(path, *, label):
        resolved, encoded = original_reader(path, label=label)
        sidecar.write_bytes(replacement)
        return resolved, encoded

    monkeypatch.setattr(
        dataloader_module,
        "read_ordinary_file_bytes",
        replace_after_verified_read,
    )
    dataset = object.__new__(FlowMatchingDataloader)
    dataset.sessions = [MPSSessions(str(adapter))]
    dataset.robot_sidecar_root = str(tmp_path / "sidecars")
    dataset.embodiment = "kai22"
    dataset.selector_records = {session_id: {}}
    dataset.sidecar_sha256_by_session = {session_id: accepted_digest}
    dataset.hand_command_dim = 22
    dataset._robot_sources = {}

    dataset._load_robot_sources()

    loaded = dataset._robot_sources[str(adapter.resolve())]["left_q"]
    np.testing.assert_array_equal(loaded, np.full((1, 22), 0.25, dtype=np.float32))
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() != accepted_digest


@pytest.mark.parametrize("replacement", ["symlink", "hardlink"])
def test_dataloader_revalidates_cached_image_path_at_the_read_fd(
    tmp_path: Path,
    replacement: str,
) -> None:
    adapter = tmp_path / "production/grap_a_cap_004/09_humanego_adapter"
    frame = adapter / "preprocess/all_data/00000"
    frame.mkdir(parents=True)
    selected = tmp_path / "external_artifacts/grap_a_cap_004/00000/rgb.png"
    selected.parent.mkdir(parents=True)
    external = tmp_path / "outside.png"
    pixels = np.full((2, 2, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(selected), pixels)
    external.write_bytes(selected.read_bytes())
    metadata = frame / "training_data.json"
    payload = {
        "obs": {"rgb_path": str(tmp_path / "decoy.png")},
        "entities": {"hands": {}},
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    dataset = FlowMatchingDataloader(
        sessions=[MPSSessions(str(adapter))],
        single_hand=False,
        img_name="rgb.png",
        selector_records={
            "grap_a_cap_004": {
                "00000": {
                    "metadata": file_ref(metadata),
                    "image": file_ref(selected),
                    "unresolved": False,
                }
            }
        },
        selector_root=str(tmp_path / "external_artifacts"),
    )
    assert dataset._resolve_image_path(payload, json_path=str(metadata)) == str(
        selected.resolve()
    )
    selected.unlink()
    if replacement == "symlink":
        selected.symlink_to(external)
        message = "symlink component"
    else:
        selected.hardlink_to(external)
        message = "hard-linked"
    with pytest.raises(ValueError, match=message):
        dataset._load_image_tensor(payload, 0)


def test_selector_json_parse_uses_the_same_verified_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _, raw_path, _, _ = selector_pair(tmp_path, monkeypatch)
    original_reader = source_contract.read_file_reference
    replacement_root = tmp_path / "replacement_artifacts"
    replacement_root.mkdir()
    replacement = json.loads(json.dumps(raw))
    replacement["artifact_root"] = str(replacement_root.resolve())
    replacement["selector_root"] = str(replacement_root.resolve())
    for record in replacement["sessions"]["grap_a_cap_004"]["frames"].values():
        record["image"]["path"] = str((replacement_root / "rgb.png").resolve())

    def replace_after_same_fd_read(*args, **kwargs):
        path, encoded = original_reader(*args, **kwargs)
        raw_path.write_text(json.dumps(replacement), encoding="utf-8")
        return path, encoded

    monkeypatch.setattr(
        source_contract,
        "read_file_reference",
        replace_after_same_fd_read,
    )
    observed = validate_selector_manifest_reference(
        file_ref(raw_path),
        artifact_root=tmp_path / "raw_artifacts",
        project_root=tmp_path,
    )
    assert observed["artifact_root"] == str((tmp_path / "raw_artifacts").resolve())


def test_selector_artifact_root_requires_caller_and_paired_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, paired, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="missing explicit artifact root"):
        require_manifest_selector_ready(
            file_ref(raw_path), file_ref(paired_path), project_root=tmp_path
        )
    other = tmp_path / "other_artifacts"
    other.mkdir()
    with pytest.raises(ValueError, match="caller-approved root"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=other,
            project_root=tmp_path,
        )

    subdirectory = tmp_path / "raw_artifacts/subdirectory"
    subdirectory.mkdir()
    noncanonical = subdirectory / ".."
    with pytest.raises(ValueError, match="not canonical"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=noncanonical,
            project_root=tmp_path,
        )

    paired["artifact_roots"]["RAW"] = str(other.resolve())
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_root differs"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_selector_manifest_rejects_outside_root_and_staging_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, paired, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    raw["sessions"]["grap_a_cap_004"]["frames"]["00000"]["image"]["path"] = str(
        (tmp_path / "outside/rgb.png").resolve()
    )
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    paired["selector_refs"]["RAW"] = file_ref(raw_path)
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="image escapes selector_root"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )

    staging = tmp_path / "_run/formal_artifacts"
    staging.mkdir(parents=True)
    raw["artifact_root"] = str(staging.resolve())
    raw["selector_root"] = str(staging.resolve())
    for record in raw["sessions"]["grap_a_cap_004"]["frames"].values():
        record["image"]["path"] = str((staging / "rgb.png").resolve())
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    paired["selector_refs"]["RAW"] = file_ref(raw_path)
    paired["artifact_roots"]["RAW"] = str(staging.resolve())
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact root cannot come from _run staging"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=staging,
            project_root=tmp_path,
        )


def test_selector_rejects_forbidden_source_namespace_before_image_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _, raw_path, _, _ = selector_pair(tmp_path, monkeypatch)
    forbidden_root = tmp_path / "labels"
    forbidden_root.mkdir()
    raw["artifact_root"] = str(forbidden_root.resolve())
    raw["selector_root"] = str(forbidden_root.resolve())
    for record in raw["sessions"]["grap_a_cap_004"]["frames"].values():
        record["image"]["path"] = str((forbidden_root / "rgb.png").resolve())
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden source namespace"):
        validate_selector_manifest_reference(
            file_ref(raw_path),
            artifact_root=forbidden_root,
            project_root=tmp_path,
        )


def test_selector_rejects_duplicate_cross_frame_image_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, paired, raw_path, _, paired_path = selector_pair(tmp_path, monkeypatch)
    frames = raw["sessions"]["grap_a_cap_004"]["frames"]
    frames["00001"]["image"] = dict(frames["00000"]["image"])
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    paired["selector_refs"]["RAW"] = file_ref(raw_path)
    paired_path.write_text(json.dumps(paired), encoding="utf-8")
    with pytest.raises(ValueError, match="image path aliases frames"):
        require_manifest_selector_ready(
            file_ref(raw_path),
            file_ref(paired_path),
            artifact_root=tmp_path / "raw_artifacts",
            project_root=tmp_path,
        )


def test_frame_reference_rejects_symlink_hardlink_and_path_alias(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    frame = adapter / "preprocess/all_data/00000"
    frame.mkdir(parents=True)
    metadata = frame / "training_data.json"
    root = tmp_path / "artifacts"
    root.mkdir()
    target = root / "target.png"
    assert cv2.imwrite(str(target), np.zeros((2, 2, 3), dtype=np.uint8))
    metadata.write_text(
        json.dumps({"obs": {"rgb_path": str(target)}}), encoding="utf-8"
    )

    symlink = root / "rgb.png"
    symlink.symlink_to(target)
    symlink_ref = {
        "path": str(symlink.absolute()),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="symlink component"):
        validate_frame_images(
            adapter,
            "rgb.png",
            selector_records={
                "00000": {"metadata": file_ref(metadata), "image": symlink_ref}
            },
            selector_root=root,
        )
    symlink.unlink()

    hardlink = root / "rgb.png"
    hardlink.hardlink_to(target)
    hardlink_ref = {
        "path": str(hardlink.resolve()),
        "bytes": hardlink.stat().st_size,
        "sha256": hashlib.sha256(hardlink.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="hard-linked"):
        validate_frame_images(
            adapter,
            "rgb.png",
            selector_records={
                "00000": {"metadata": file_ref(metadata), "image": hardlink_ref}
            },
            selector_root=root,
        )
    hardlink.unlink()
    target.unlink()

    canonical = root / "rgb.png"
    assert cv2.imwrite(str(canonical), np.zeros((2, 2, 3), dtype=np.uint8))
    (root / "sub").mkdir()
    alias_ref = file_ref(canonical)
    alias_ref["path"] = str(root / "sub/../rgb.png")
    with pytest.raises(ValueError, match="not canonical"):
        validate_frame_images(
            adapter,
            "rgb.png",
            selector_records={
                "00000": {"metadata": file_ref(metadata), "image": alias_ref}
            },
            selector_root=root,
        )
