from __future__ import annotations

from io import BytesIO
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "tools/run_eligible68_mask_producer_adapter.py"
LIVE = ROOT / "tools/run_d1_d4_live_producer.py"


def _module():
    spec = importlib.util.spec_from_file_location("eligible68_formal_adapter", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _png(mask: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return stream.getvalue()


def _write_fixture(path: Path, mask: np.ndarray) -> Path:
    path.write_bytes(_png(mask))
    return path


def _write_context_fixture(
    root: Path,
    *,
    session_id: str = "grap_a_cap_004",
    frame_count: int = 460,
    metadata_sha: str = "b" * 64,
    side_states: dict[int, dict[str, str]] | None = None,
) -> Path:
    root.mkdir()
    raw_path = root / "RAW_INPUT_MANIFEST.json"
    authority_path = root / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    object6d_path = root / "OBJECT6D.npz"
    geometry_paths = {
        "joints": root / "JOINTS.npz",
        "hand_axis_metadata": root / "HAND_AXIS.json",
        "per_frame_quality": root / "QUALITY.csv",
        "raw_hawor": root / "RAW_HAWOR.npz",
        "object6d": object6d_path,
        "cylinder": root / "CYLINDER.json",
    }

    def ref(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    np.savez(
        object6d_path,
        T_object_to_camera=np.repeat(
            np.eye(4, dtype=np.float64)[None], frame_count, axis=0
        ),
        valid=np.ones(frame_count, dtype=np.bool_),
    )
    geometry_paths["joints"].write_bytes(b"fixture-joints")
    geometry_paths["hand_axis_metadata"].write_bytes(b'{"fixture":"axis"}')
    geometry_paths["per_frame_quality"].write_bytes(b"fixture,quality\n")
    geometry_paths["raw_hawor"].write_bytes(b"fixture-raw-hawor")
    geometry_paths["cylinder"].write_bytes(b'{"fixture":"cylinder"}')
    raw_root = root / "raw"
    raw_root.mkdir()
    raw_payload = _png(np.zeros((960, 1280), dtype=np.bool_))
    raw_rows = []
    for index in range(frame_count):
        frame_path = raw_root / f"frame_{index:05d}.png"
        frame_path.write_bytes(raw_payload)
        raw_rows.append(
            {
                **ref(frame_path),
                "frame_index": index,
                "width": 1280,
                "height": 960,
            }
        )
    raw_path.write_text(
        json.dumps(
            {
                "schema_version": "eligible68-dual-raw-input-context-v1",
                "detection_mode": "dual",
                "sessions": {
                    session_id: {
                        "frame_count": frame_count,
                        "frames": raw_rows,
                    }
                },
                "session_count": 1,
                "frame_count": frame_count,
                "raw_png_full_decode_count": frame_count,
            }
        ),
        encoding="utf-8",
    )

    def side_record(frame_index: int, side: str, state: str) -> dict[str, object]:
        axis = ("left", "right").index(side)
        if state == "MISSING":
            valid = False
            wrist_xy = None
            outside_direction = None
            outside_distance = None
            center_oob = False
            center_direction = None
            center_distance = None
        elif state == "OUTSIDE_IMAGE":
            valid = True
            wrist_xy = [200.0 if side == "left" else 1000.0, 960.25]
            outside_direction = "BOTTOM"
            outside_distance = 1.25
            center_oob = True
            center_direction = "BOTTOM"
            center_distance = 1.25
        else:
            valid = True
            wrist_xy = [200.0 if side == "left" else 1000.0, 700.0]
            outside_direction = None
            outside_distance = None
            center_oob = False
            center_direction = None
            center_distance = None
        return {
            "identity": side,
            "physical_hand_axis_index": axis,
            "source_track_index": axis,
            "lineage_id": (f"{session_id}:final-v3-hand-axis-{axis}:{side}"),
            "source_array_key": "joints_2d_projection",
            "source_frame_index": frame_index,
            "source_state": "invalid" if state == "MISSING" else "tracked",
            "source_slot_observation": -1 if state == "MISSING" else axis,
            "numeric_session_id": -1 if state == "MISSING" else 0,
            "valid": valid,
            "measurement_accepted": state != "MISSING",
            "label_override_for_temporal_identity": False,
            "state": state,
            "wrist_xy": wrist_xy,
            "operational_bounds": "0<=x<1280,0<=y<960",
            "outside_image_direction": outside_direction,
            "outside_image_distance_px": outside_distance,
            "pixel_center_bounds": "0<=x<=1279,0<=y<=959",
            "pixel_center_oob": center_oob,
            "pixel_center_oob_direction": center_direction,
            "pixel_center_oob_distance_px": center_distance,
        }

    authority_path.write_text(
        json.dumps(
            {
                "schema_version": "eligible68-authority-physical-side-covariates-v1",
                "detection_mode": "dual",
                "sessions": {
                    session_id: {
                        "geometry": {
                            role: ref(path) for role, path in geometry_paths.items()
                        }
                    }
                },
                "frame_count": frame_count,
                "side_count": 2 * frame_count,
                "frame_side_rows": [
                    {
                        "session_id": session_id,
                        "frame_index": index,
                        "sides": {
                            side: side_record(
                                index,
                                side,
                                (side_states or {})
                                .get(index, {})
                                .get(side, "AVAILABLE"),
                            )
                            for side in ("left", "right")
                        },
                    }
                    for index in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )

    (root / "PREFLIGHT.json").write_text(
        json.dumps(
            {
                "schema_version": "eligible68-dual-context-preflight-v1",
                "session_id": session_id,
                "frame_count": frame_count,
                "input_metadata_bundle_sha256": metadata_sha,
                "completion_mode": "ARTIFACT_EXISTS",
                "artifacts": {
                    "raw_input_manifest": ref(raw_path),
                    "authority_covariates": ref(authority_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _artifact_ref(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _refresh_authority_binding(root: Path) -> None:
    preflight_path = root / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_bytes())
    preflight["artifacts"]["authority_covariates"] = _artifact_ref(
        root / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    )
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")


def _configure_formal_runtime(module, context_template: Path) -> None:
    live_payload = LIVE.read_bytes()
    module._RUNTIME = module.RuntimeConfig(
        command=(str(SOURCE),),
        cpu_fixture_left_png=None,
        cpu_fixture_right_png=None,
        live_producer=LIVE,
        live_producer_sha256=hashlib.sha256(live_payload).hexdigest(),
        context_root_template=str(context_template),
        expected_dependency_source_sha256="0" * 64,
    )


def test_fixed_adapter_is_accepted_by_b4_source_contract() -> None:
    from tools import run_eligible68_shards as shard

    payload = SOURCE.read_bytes()
    executable, reference = shard._bind_producer(
        SOURCE, hashlib.sha256(payload).hexdigest()
    )
    assert executable == SOURCE
    assert reference["bytes"] == len(payload)


def test_pre_side_schema_adapter_sha_and_admission_are_invalidated() -> None:
    from tools import run_eligible68_shards as shard

    stale_sha256 = "798bdefab39884c9cf220e926abbf37a4b8df9d2af94819b3a8e594b959194a7"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() != stale_sha256
    with pytest.raises(shard.ShardRunError, match="producer SHA-256 mismatch"):
        shard._bind_producer(SOURCE, stale_sha256)


@pytest.mark.parametrize(
    ("mode", "frame_count"),
    (("fixture", 1), ("formal", 460)),
)
def test_success_evidence_exactly_matches_shard_consumer_for_fixture_and_formal(
    tmp_path: Path,
    mode: str,
    frame_count: int,
) -> None:
    module = _module()
    from tools import run_eligible68_shards as shard

    root = tmp_path / mode
    manifest = root / "PRODUCER_MANIFEST.json"
    adapter_evidence = module._success_evidence(
        root / "masks/left",
        root / "masks/right",
        manifest,
        frame_count,
    )
    shard_evidence = shard._producer_success_evidence(
        root / "masks/left",
        root / "masks/right",
        manifest,
        frame_count,
    )

    assert adapter_evidence == shard_evidence
    assert {
        "pixels_created_or_modified",
        "retained_output_root",
    }.issubset(adapter_evidence[-1]["expected_keys"])


def test_production_command_is_strict_dual_generic_d4_with_d3_inventory(
    tmp_path: Path,
) -> None:
    module = _module()
    command = module._production_command(
        LIVE,
        tmp_path / "work",
        tmp_path / "context",
        "grap_a_cap_004",
        460,
    )
    assert command[command.index("--detection-mode") + 1] == "dual"
    assert "--dump-raw-inventory" in command
    assert command[command.index("--wearable-mode") + 1] == "per_frame_w1_w4"
    assert "--no-d4" not in command


def test_formal_path_fails_before_child_when_wearable_provider_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    child_called = False
    fail_closed_live = tmp_path / "live_without_wearable.py"
    fail_closed_payload = LIVE.read_bytes().replace(
        b'"wearable_w1_w4_raw_provider": True',
        b'"wearable_w1_w4_raw_provider": False',
        1,
    )
    assert fail_closed_payload != LIVE.read_bytes()
    fail_closed_live.write_bytes(fail_closed_payload)

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("child must not start")

    monkeypatch.setattr(module.subprocess, "run", forbidden_child)
    module._RUNTIME = module.RuntimeConfig(
        command=(str(SOURCE),),
        cpu_fixture_left_png=None,
        cpu_fixture_right_png=None,
        live_producer=fail_closed_live,
        live_producer_sha256=hashlib.sha256(fail_closed_payload).hexdigest(),
        context_root_template=str(tmp_path / "missing-{session_id}"),
    )
    with pytest.raises(
        module.AdapterError, match="WEARABLE_W1_W4_RAW_PROVIDER_REQUIRED"
    ):
        module.produce_eligible68_session(
            "grap_a_cap_004",
            1,
            "a" * 64,
            tmp_path / "left",
            tmp_path / "right",
            tmp_path / "PRODUCER_MANIFEST.json",
        )
    assert child_called is False
    assert not (tmp_path / "PRODUCER_MANIFEST.json").exists()
    assert not (tmp_path / "left").exists()
    assert not (tmp_path / "right").exists()


def test_one_frame_formal_canary_uses_full_context_with_exact_prefix_limit(
    tmp_path: Path,
) -> None:
    module = _module()
    metadata_sha = "b" * 64
    _write_context_fixture(
        tmp_path / "context-grap_a_cap_004", metadata_sha=metadata_sha
    )

    resolved = module._resolve_context(
        str(tmp_path / "context-{session_id}"),
        "grap_a_cap_004",
        1,
        metadata_sha,
    )
    command = module._production_command(
        LIVE,
        tmp_path / "work",
        resolved.origin_root,
        "grap_a_cap_004",
        1,
        resolved.frame_count,
    )
    assert resolved.frame_count == 460
    assert command[command.index("--frame-count") + 1] == "460"
    assert command[command.index("--limit-frames") + 1] == "1"


def test_frame_zero_missing_side_rejects_with_exact_evidence_before_child_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    metadata_sha = "b" * 64
    context = _write_context_fixture(
        tmp_path / "context-grap_a_cap_004",
        frame_count=2,
        metadata_sha=metadata_sha,
        side_states={0: {"right": "MISSING"}, 1: {"right": "MISSING"}},
    )
    _configure_formal_runtime(module, tmp_path / "context-{session_id}")
    child_called = False
    output_called = False

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("model/GPU child must not start")

    def forbidden_output(*_args, **_kwargs):
        nonlocal output_called
        output_called = True
        raise AssertionError("output/mkdir must not start")

    monkeypatch.setattr(module.subprocess, "run", forbidden_child)
    monkeypatch.setattr(module, "_ensure_output_directory", forbidden_output)
    monkeypatch.setattr(module, "_write_exclusive", forbidden_output)
    output_root = tmp_path / "adapter-output"
    with pytest.raises(
        module.AdapterError, match="^CONTEXT_SIDE_AUTHORITY_PREFIX_MISSING:"
    ) as captured:
        module.produce_eligible68_session(
            "grap_a_cap_004",
            1,
            metadata_sha,
            output_root / "masks/left",
            output_root / "masks/right",
            output_root / "PRODUCER_MANIFEST.json",
        )

    evidence = json.loads(str(captured.value).split(":", 1)[1])
    assert evidence["schema_version"] == (
        "eligible68-context-side-authority-prefix-preflight-v1"
    )
    assert evidence["status"] == "HOLD_CONTEXT_SIDE_AUTHORITY_PREFIX_MISSING"
    assert evidence["session_id"] == "grap_a_cap_004"
    assert evidence["context_frame_count"] == 2
    assert evidence["execution_frame_count"] == 1
    assert evidence["missing_count"] == 1
    assert evidence["first_missing"] == {
        "frame_index": 0,
        "side": "right",
        "state": "MISSING",
    }
    assert evidence["authority_covariates"] == _artifact_ref(
        context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    )
    assert evidence["outside_image_policy"] == "DIAGNOSTIC_ONLY_NOT_RECLASSIFIED"
    assert evidence["sam_acceptance_predicted"] is False
    assert evidence["model_called_during_preflight"] is False
    assert evidence["gpu_started_during_preflight"] is False
    assert evidence["output_created_during_preflight"] is False
    assert evidence["pixels_created_or_modified"] == 0
    assert child_called is False
    assert output_called is False
    assert not output_root.exists()


def test_missing_after_one_frame_limit_is_allowed_but_full_session_is_blocked(
    tmp_path: Path,
) -> None:
    module = _module()
    context = _write_context_fixture(
        tmp_path / "context-grap_a_cap_004",
        frame_count=2,
        side_states={1: {"right": "MISSING"}},
    )
    canary = module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)
    evidence = canary.side_authority_prefix_preflight
    assert evidence["status"] == "PASS_NO_MISSING_IN_EXECUTION_PREFIX"
    assert evidence["execution_prefix_state_counts"] == {
        "AVAILABLE": 2,
        "OUTSIDE_IMAGE": 0,
        "MISSING": 0,
    }
    assert evidence["full_context_state_counts"] == {
        "AVAILABLE": 3,
        "OUTSIDE_IMAGE": 0,
        "MISSING": 1,
    }

    with pytest.raises(
        module.AdapterError, match="^CONTEXT_SIDE_AUTHORITY_PREFIX_MISSING:"
    ) as captured:
        module._resolve_context(str(context), "grap_a_cap_004", 2, "b" * 64)
    failure = json.loads(str(captured.value).split(":", 1)[1])
    assert failure["execution_frame_count"] == 2
    assert failure["first_missing"] == {
        "frame_index": 1,
        "side": "right",
        "state": "MISSING",
    }


def test_missing_state_from_valid_but_nonfinite_source_still_holds_prefix(
    tmp_path: Path,
) -> None:
    module = _module()
    context = _write_context_fixture(
        tmp_path / "context-grap_a_cap_004",
        frame_count=1,
        side_states={0: {"right": "MISSING"}},
    )
    authority_path = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    authority = json.loads(authority_path.read_bytes())
    right = authority["frame_side_rows"][0]["sides"]["right"]
    right["valid"] = True
    right["source_state"] = "valid_source_with_nonfinite_wrist"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    _refresh_authority_binding(context)

    with pytest.raises(
        module.AdapterError, match="^CONTEXT_SIDE_AUTHORITY_PREFIX_MISSING:"
    ):
        module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)


def test_outside_image_remains_diagnostic_and_does_not_block_prefix(
    tmp_path: Path,
) -> None:
    module = _module()
    context = _write_context_fixture(
        tmp_path / "context-grap_a_cap_004",
        frame_count=1,
        side_states={0: {"right": "OUTSIDE_IMAGE"}},
    )
    resolved = module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)
    evidence = resolved.side_authority_prefix_preflight
    assert evidence["status"] == "PASS_NO_MISSING_IN_EXECUTION_PREFIX"
    assert evidence["execution_prefix_state_counts"] == {
        "AVAILABLE": 1,
        "OUTSIDE_IMAGE": 1,
        "MISSING": 0,
    }
    assert evidence["outside_image_policy"] == "DIAGNOSTIC_ONLY_NOT_RECLASSIFIED"
    assert evidence["sam_acceptance_predicted"] is False
    assert evidence["pixels_created_or_modified"] == 0


@pytest.mark.parametrize("failure", ["malformed_state", "coverage", "alias"])
def test_side_authority_preflight_rejects_malformed_coverage_and_alias_before_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=2)
    authority_path = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    if failure == "alias":
        authority_path.unlink()
        authority_path.hardlink_to(context / "RAW_INPUT_MANIFEST.json")
    else:
        authority = json.loads(authority_path.read_bytes())
        if failure == "malformed_state":
            # Even a malformed row outside the one-frame execution prefix is
            # a schema failure, while a well-formed MISSING tail is allowed.
            authority["frame_side_rows"][1]["sides"]["right"]["state"] = "UNKNOWN"
        else:
            authority["frame_side_rows"].pop()
        authority_path.write_text(json.dumps(authority), encoding="utf-8")
    _refresh_authority_binding(context)
    _configure_formal_runtime(module, context)
    child_called = False

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("model/GPU child must not start")

    monkeypatch.setattr(module.subprocess, "run", forbidden_child)
    output_root = tmp_path / "adapter-output"
    expected = {
        "malformed_state": "side-state schema mismatch",
        "coverage": "authority schema/coverage mismatch",
        "alias": "alias",
    }[failure]
    with pytest.raises(module.AdapterError, match=expected):
        module.produce_eligible68_session(
            "grap_a_cap_004",
            1,
            "b" * 64,
            output_root / "masks/left",
            output_root / "masks/right",
            output_root / "PRODUCER_MANIFEST.json",
        )
    assert child_called is False
    assert not output_root.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_key",
        "extra_key",
        "identity",
        "axis",
        "source_frame",
        "bool_type",
        "available_without_valid_wrist",
        "huge_wrist_number",
        "missing_with_wrist",
        "outside_with_inside_wrist",
        "outside_distance",
    ],
)
def test_side_authority_complete_record_schema_rejects_tail_attack_before_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=2)
    authority_path = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    authority = json.loads(authority_path.read_bytes())
    record = authority["frame_side_rows"][1]["sides"]["right"]
    if failure == "missing_key":
        record.pop("lineage_id")
    elif failure == "extra_key":
        record["unvalidated_authority"] = True
    elif failure == "identity":
        record["identity"] = "left"
    elif failure == "axis":
        record["physical_hand_axis_index"] = 0
    elif failure == "source_frame":
        record["source_frame_index"] = 0
    elif failure == "bool_type":
        record["measurement_accepted"] = 1
    elif failure == "available_without_valid_wrist":
        record["valid"] = False
        record["wrist_xy"] = None
    elif failure == "huge_wrist_number":
        record["wrist_xy"][0] = 10**400
    elif failure == "missing_with_wrist":
        record.update(
            {
                "state": "MISSING",
                "valid": False,
                "wrist_xy": [1000.0, 700.0],
                "outside_image_direction": None,
                "outside_image_distance_px": None,
                "pixel_center_oob": False,
                "pixel_center_oob_direction": None,
                "pixel_center_oob_distance_px": None,
            }
        )
    elif failure == "outside_with_inside_wrist":
        record["state"] = "OUTSIDE_IMAGE"
    else:
        record.update(
            {
                "state": "OUTSIDE_IMAGE",
                "wrist_xy": [1000.0, 960.25],
                "outside_image_direction": "BOTTOM",
                "outside_image_distance_px": 99.0,
                "pixel_center_oob": True,
                "pixel_center_oob_direction": "BOTTOM",
                "pixel_center_oob_distance_px": 1.25,
            }
        )
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    _refresh_authority_binding(context)
    _configure_formal_runtime(module, context)
    child_called = False

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("model/GPU child must not start")

    monkeypatch.setattr(module.subprocess, "run", forbidden_child)
    output_root = tmp_path / "adapter-output"
    with pytest.raises(module.AdapterError, match="context authority"):
        module.produce_eligible68_session(
            "grap_a_cap_004",
            1,
            "b" * 64,
            output_root / "masks/left",
            output_root / "masks/right",
            output_root / "PRODUCER_MANIFEST.json",
        )
    assert child_called is False
    assert not output_root.exists()


def test_noncanary_partial_context_is_rejected(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.AdapterError, match="prefix/frame-count"):
        module._production_command(
            LIVE,
            tmp_path / "work",
            tmp_path / "context",
            "grap_a_cap_004",
            2,
            460,
        )


def test_live_source_a_to_b_replacement_cannot_change_published_execution_bytes(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "live.py"
    payload_a = b"#!/usr/bin/env python3\nprint('A')\n"
    payload_b = b"#!/usr/bin/env python3\nprint('B')\n"
    source.write_bytes(payload_a)
    verified_a = module._read_regular(source, "test live source")
    executed = tmp_path / "run/EXECUTED_LIVE_PRODUCER.py"
    executed.parent.mkdir()
    reference = module._publish_immutable_live_source(executed, verified_a)
    source.write_bytes(payload_b)
    assert executed.read_bytes() == payload_a
    assert reference["sha256"] == hashlib.sha256(payload_a).hexdigest()
    command = module._production_command(
        executed,
        tmp_path / "run-root",
        tmp_path / "context",
        "grap_a_cap_004",
        1,
        1,
        reference["sha256"],
    )
    assert command[1] == str(executed)
    flag = command.index("--expected-executed-source-sha256")
    assert command[flag + 1] == reference["sha256"]


def test_inner_live_rename_restore_executes_the_held_verified_inode(
    tmp_path: Path,
) -> None:
    module = _module()
    executed = tmp_path / "EXECUTED_LIVE_PRODUCER.py"
    executed.write_text("print('ORIGINAL_HELD_LIVE')\n", encoding="utf-8")
    reference = {
        "path": str(executed),
        "bytes": executed.stat().st_size,
        "sha256": hashlib.sha256(executed.read_bytes()).hexdigest(),
    }
    descriptor, before = module._open_held_source(executed, reference, "test live")
    parked = tmp_path / "parked.py"
    malicious = tmp_path / "malicious.py"
    malicious.write_text("print('MALICIOUS_PATH_LIVE')\n", encoding="utf-8")
    executed.rename(parked)
    malicious.rename(executed)
    try:
        completed = subprocess.run(
            [sys.executable, f"/proc/self/fd/{descriptor}"],
            pass_fds=(descriptor,),
            capture_output=True,
            text=True,
            check=True,
        )
        module._assert_held_source_stable(descriptor, before, "test live")
    finally:
        executed.rename(malicious)
        parked.rename(executed)
        module.os.close(descriptor)
    assert completed.stdout.strip() == "ORIGINAL_HELD_LIVE"


def test_retained_attempt_dirfd_never_writes_into_rename_replacement(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "attempt"
    child = root / "child"
    child.mkdir(parents=True)
    scope = module.RetainedOutputScope.open(root)
    moved = tmp_path / "attempt-held"
    root.rename(moved)
    (root / "child").mkdir(parents=True)
    module._OUTPUT_SCOPE = scope
    try:
        module._write_exclusive(root / "child/artifact.bin", b"held", "attack")
        with pytest.raises(module.AdapterError, match="identity drift"):
            scope.verify_logical_identity()
    finally:
        module._OUTPUT_SCOPE = None
        scope.close()
    assert (moved / "child/artifact.bin").read_bytes() == b"held"
    assert not (root / "child/artifact.bin").exists()


def test_retained_attempt_open_rejects_ancestor_swap_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    ancestor = tmp_path / "ancestor"
    target = ancestor / "attempt"
    target.mkdir(parents=True)
    parked = tmp_path / "ancestor-held"
    real_open = module.os.open
    swapped = False

    def attacking_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "attempt" and dir_fd is not None and not swapped:
            swapped = True
            ancestor.rename(parked)
            target.mkdir(parents=True)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", attacking_open)
    with pytest.raises(module.AdapterError, match="identity drift"):
        module.RetainedOutputScope.open(target)

    assert swapped is True
    assert list((parked / "attempt").iterdir()) == []
    assert list(target.iterdir()) == []


def test_live_source_b_bytes_cannot_satisfy_frozen_a_sha_before_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "live.py"
    source.write_bytes(b"#!/usr/bin/env python3\nprint('B')\n")
    child_called = False

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("child must not start")

    monkeypatch.setattr(module.subprocess, "run", forbidden_child)
    module._RUNTIME = module.RuntimeConfig(
        command=(str(SOURCE),),
        cpu_fixture_left_png=None,
        cpu_fixture_right_png=None,
        live_producer=source,
        live_producer_sha256=hashlib.sha256(b"A").hexdigest(),
        context_root_template=str(tmp_path / "missing-{session_id}"),
    )
    with pytest.raises(module.AdapterError, match="SHA-256"):
        module.produce_eligible68_session(
            "grap_a_cap_004",
            1,
            "a" * 64,
            tmp_path / "left",
            tmp_path / "right",
            tmp_path / "manifest.json",
        )
    assert child_called is False


def test_context_rejects_replaced_or_self_consistent_stub_artifacts(
    tmp_path: Path,
) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=1)
    raw = context / "RAW_INPUT_MANIFEST.json"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(module.AdapterError, match="byte reference mismatch"):
        module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)

    preflight_path = context / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_bytes())
    payload = raw.read_bytes()
    preflight["artifacts"]["raw_input_manifest"] = {
        "path": str(raw),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(module.AdapterError, match="raw input schema/coverage"):
        module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)


def test_context_rejects_raw_authority_hardlink_alias(tmp_path: Path) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=1)
    raw = context / "RAW_INPUT_MANIFEST.json"
    authority = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    authority.unlink()
    authority.hardlink_to(raw)
    preflight_path = context / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_bytes())
    payload = authority.read_bytes()
    preflight["artifacts"]["authority_covariates"] = {
        "path": str(authority),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(module.AdapterError, match="alias"):
        module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)


@pytest.mark.parametrize("failure", ["invalid", "nonfinite"])
def test_formal_context_rejects_any_unmeasured_object6d_frame(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=2)
    object6d = context / "OBJECT6D.npz"
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    valid = np.ones(2, dtype=np.bool_)
    if failure == "invalid":
        valid[1] = False
        transforms[1] = np.nan
    else:
        transforms[1, 0, 0] = np.nan
    np.savez(object6d, T_object_to_camera=transforms, valid=valid)

    def ref(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    authority_path = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    authority = json.loads(authority_path.read_bytes())
    authority["sessions"]["grap_a_cap_004"]["geometry"]["object6d"] = ref(object6d)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    preflight_path = context / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_bytes())
    preflight["artifacts"]["authority_covariates"] = ref(authority_path)
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(module.AdapterError, match="valid finite geometry"):
        module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)


@pytest.mark.parametrize(
    "role",
    [
        "joints",
        "hand_axis_metadata",
        "per_frame_quality",
        "raw_hawor",
        "object6d",
        "cylinder",
    ],
)
def test_each_geometry_role_publishes_consumed_bytes_after_origin_swap(
    tmp_path: Path,
    role: str,
) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=1)
    authority_path = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    authority = json.loads(authority_path.read_bytes())
    source = Path(authority["sessions"]["grap_a_cap_004"]["geometry"][role]["path"])
    consumed = source.read_bytes()
    resolved = module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)
    source.write_bytes(b"replacement-B" if role != "object6d" else consumed + b"B")
    _, refs = module._publish_bound_context(resolved, tmp_path / "work")
    assert Path(refs[f"geometry_{role}"]["path"]).read_bytes() == consumed


def test_geometry_role_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=1)
    authority_path = context / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    authority = json.loads(authority_path.read_bytes())
    geometry = authority["sessions"]["grap_a_cap_004"]["geometry"]
    joints = Path(geometry["joints"]["path"])
    raw_hawor = Path(geometry["raw_hawor"]["path"])
    raw_hawor.unlink()
    raw_hawor.hardlink_to(joints)
    payload = raw_hawor.read_bytes()
    geometry["raw_hawor"] = {
        "path": str(raw_hawor),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    preflight_path = context / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_bytes())
    authority_payload = authority_path.read_bytes()
    preflight["artifacts"]["authority_covariates"] = {
        "path": str(authority_path),
        "bytes": len(authority_payload),
        "sha256": hashlib.sha256(authority_payload).hexdigest(),
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(module.AdapterError, match="alias"):
        module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)


def test_raw_frame_publishes_consumed_bytes_after_origin_swap(tmp_path: Path) -> None:
    module = _module()
    context = _write_context_fixture(tmp_path / "context-grap_a_cap_004", frame_count=1)
    raw_manifest = json.loads((context / "RAW_INPUT_MANIFEST.json").read_bytes())
    source = Path(raw_manifest["sessions"]["grap_a_cap_004"]["frames"][0]["path"])
    consumed = source.read_bytes()
    resolved = module._resolve_context(str(context), "grap_a_cap_004", 1, "b" * 64)
    replacement = _png(np.ones((960, 1280), dtype=np.bool_))
    source.write_bytes(replacement)
    bound_root, _ = module._publish_bound_context(resolved, tmp_path / "work")
    bound_manifest = json.loads((bound_root / "RAW_INPUT_MANIFEST.json").read_bytes())
    bound_path = Path(bound_manifest["sessions"]["grap_a_cap_004"]["frames"][0]["path"])
    assert bound_path.read_bytes() == consumed
    assert source.read_bytes() == replacement


def test_one_frame_no_model_fixture_writes_real_pngs_and_nonempty_evidence(
    tmp_path: Path,
) -> None:
    module = _module()
    left_mask = np.zeros((960, 1280), dtype=np.bool_)
    right_mask = np.zeros_like(left_mask)
    left_mask[10:30, 20:50] = True
    right_mask[100:140, 900:960] = True
    source_left = _write_fixture(tmp_path / "source_left.png", left_mask)
    source_right = _write_fixture(tmp_path / "source_right.png", right_mask)
    output_left = tmp_path / "out" / "masks" / "left"
    output_right = tmp_path / "out" / "masks" / "right"
    output_manifest = tmp_path / "out" / "PRODUCER_MANIFEST.json"
    command = (
        str(SOURCE),
        "--cpu-fixture-left-png",
        str(source_left),
        "--cpu-fixture-right-png",
        str(source_right),
    )
    module._RUNTIME = module.RuntimeConfig(
        command=command,
        cpu_fixture_left_png=source_left,
        cpu_fixture_right_png=source_right,
        live_producer=LIVE,
        live_producer_sha256=None,
        context_root_template=None,
    )

    value = module.produce_eligible68_session(
        "grap_a_cap_004",
        1,
        "a" * 64,
        output_left,
        output_right,
        output_manifest,
    )

    assert (output_left / "frame_00000.png").read_bytes() == source_left.read_bytes()
    assert (output_right / "frame_00000.png").read_bytes() == source_right.read_bytes()
    assert value == json.loads(output_manifest.read_bytes())
    assert value["status"] == value["completion_mode"] == "ARTIFACT_EXISTS"
    assert value["left_right_png_count"] == 2
    assert value["model_called"] is False
    assert value["gpu_started"] is False
    assert value["formal_admission_eligible"] is False
    assert value["strict_dual"] is False
    assert value["d4_applied"] is False
    assert value["wearable_w1_w4_applied"] is False
    assert value["command_manifest"]["command"] == list(command)
    evidence = value["command_manifest"]["success_evidence"]
    assert len(evidence) == 3
    assert all(Path(row["absolute_path"]).is_absolute() for row in evidence)
    assert all(row["minimum_bytes"] >= 1 for row in evidence)

    from tools import run_eligible68_shards as shard

    decoded = []
    for side, path in (
        ("left", output_left / "frame_00000.png"),
        ("right", output_right / "frame_00000.png"),
    ):
        payload = path.read_bytes()
        shard._decode_png(payload, str(path))
        decoded.append((side, 0, len(payload), hashlib.sha256(payload).hexdigest()))
    with pytest.raises(shard.ShardRunError, match="content mismatch"):
        shard._validate_producer_manifest(
            output_manifest,
            {
                "session_id": "grap_a_cap_004",
                "frame_count": 1,
                "input_metadata_bundle_sha256": "a" * 64,
            },
            {"combined_png_sha256": shard._combined_digest(decoded)},
        )


def test_fixture_rejects_nonbinary_source_before_any_output_write(
    tmp_path: Path,
) -> None:
    module = _module()
    valid = np.zeros((960, 1280), dtype=np.bool_)
    source_left = _write_fixture(tmp_path / "left.png", valid)
    invalid_array = np.zeros((960, 1280), dtype=np.uint8)
    invalid_array[0, 0] = 127
    invalid = BytesIO()
    Image.fromarray(invalid_array).save(invalid, format="PNG")
    source_right = tmp_path / "right.png"
    source_right.write_bytes(invalid.getvalue())
    module._RUNTIME = module.RuntimeConfig(
        command=(str(SOURCE),),
        cpu_fixture_left_png=source_left,
        cpu_fixture_right_png=source_right,
        live_producer=LIVE,
        live_producer_sha256=None,
        context_root_template=None,
    )
    left = tmp_path / "out-left"
    right = tmp_path / "out-right"
    with pytest.raises(module.AdapterError, match="exact binary"):
        module.produce_eligible68_session(
            "grap_a_cap_004",
            1,
            "a" * 64,
            left,
            right,
            tmp_path / "manifest.json",
        )
    assert list(left.iterdir()) == []
    assert list(right.iterdir()) == []
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("session", ["grap_a_cap_025", "grap_a_cap_149"])
def test_forbidden_session_rejected_before_output_directories(
    session: str,
    tmp_path: Path,
) -> None:
    module = _module()
    with pytest.raises(module.AdapterError, match="invalid or forbidden"):
        module.produce_eligible68_session(
            session,
            1,
            "a" * 64,
            tmp_path / "left",
            tmp_path / "right",
            tmp_path / "manifest.json",
        )
    assert not (tmp_path / "left").exists()
