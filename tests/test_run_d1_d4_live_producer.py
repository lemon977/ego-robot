from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import types
from typing import Any

import numpy as np
from PIL import Image
import pytest

from tools import run_d1_d4_live_producer as producer
from tools import immutable_artifact_io


class _Cuda:
    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def empty_cache() -> None:
        return None


class _V3:
    class _Torch:
        cuda = _Cuda()

    torch = _Torch()


@dataclass
class _Instances:
    masks: np.ndarray
    scores: np.ndarray
    instance_ids: np.ndarray

    def validated(self, _height: int, _width: int) -> _Instances:
        return self


class _Base:
    @staticmethod
    def outputs_to_instances(value: _Instances, _height: int, _width: int) -> _Instances:
        return value


class _Old:
    base = _Base()

    @staticmethod
    def exclusive_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _instances(
    value: bool,
    *,
    instance_id: int = 7,
    score: float = 0.75,
) -> _Instances:
    mask = np.zeros((1, 2, 3), dtype=bool)
    mask[0, 0, 0] = value
    mask[0, 1, 2] = not value
    return _Instances(
        masks=mask,
        scores=np.asarray([score], dtype=np.float64),
        instance_ids=np.asarray([instance_id], dtype=np.int64),
    )


class _Adapter:
    def __init__(
        self,
        anchor: _Instances,
        propagated: list[tuple[int, _Instances]],
        per_frame: list[_Instances],
    ) -> None:
        self.anchor = anchor
        self.propagated = propagated
        self.per_frame = per_frame

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request["type"]
        if request_type == "start_session":
            return {"parameter_mapping": {"session_id": request["session_id"]}}
        if request_type == "close_session":
            return {}
        if request_type != "add_prompt":
            raise AssertionError(request)
        if "-prop-" in request["session_id"]:
            return {"outputs": self.anchor, "frame_index": 0}
        index = int(request["frame_index"])
        return {"outputs": self.per_frame[index], "frame_index": index}

    def handle_stream_request(self, _request: dict[str, Any]):
        return iter(
            {"frame_index": index, "outputs": value}
            for index, value in self.propagated
        )


def _run_stream(adapter: _Adapter, frame_count: int):
    timing: dict[str, Any] = {}
    provenance: dict[int, dict[str, Any]] = {}
    output = list(
        producer.stream_dual_detection(
            adapter,
            _Old(),
            _V3(),
            producer.PROJECT,
            session="grap_a_cap_004",
            frame_count=frame_count,
            timing=timing,
            provenance=provenance,
        )
    )
    return output, timing, provenance


def test_dual_accepts_measured_frame0_raster_drift_and_records_exact_candidates() -> None:
    anchor = _instances(True)
    repeated = _instances(False)
    adapter = _Adapter(
        anchor,
        [(0, repeated), (1, _instances(True, instance_id=8))],
        [_instances(False, instance_id=20), _instances(True, instance_id=21)],
    )

    output, timing, provenance = _run_stream(adapter, 2)

    assert [index for index, _ in output] == [0, 1]
    assert timing["frames_streamed"] == 2
    duplicate = provenance[0]["propagation_frame0_duplicate"]
    assert duplicate["mask_exact_equal"] is False
    assert duplicate["instances"][0]["xor_pixels"] == 2
    assert set(provenance[1]["propagate_candidates"][0]) == {
        "raw_instance_offset",
        "instance_id",
        "score",
        "area",
        "mask_sha256",
    }


def test_dual_rejects_missing_propagation_frame_instead_of_falling_back() -> None:
    adapter = _Adapter(
        _instances(True),
        [(0, _instances(False)), (2, _instances(True, instance_id=9))],
        [_instances(True), _instances(True), _instances(True)],
    )
    with pytest.raises(RuntimeError, match="propagation missing exact frame: 1"):
        _run_stream(adapter, 3)


def test_dual_rejects_frame0_identity_drift() -> None:
    adapter = _Adapter(
        _instances(True, instance_id=7),
        [(0, _instances(False, instance_id=8)), (1, _instances(True))],
        [_instances(True), _instances(True)],
    )
    with pytest.raises(RuntimeError, match="frame 0 identity/score/shape drift"):
        _run_stream(adapter, 2)


def test_dual_rejects_duplicate_propagation_frame() -> None:
    adapter = _Adapter(
        _instances(True),
        [(0, _instances(False)), (1, _instances(True)), (1, _instances(True))],
        [_instances(True), _instances(True)],
    )
    with pytest.raises(RuntimeError, match="duplicate propagation frame: 1"):
        _run_stream(adapter, 2)


def test_generic_d4_provider_binds_exact_requested_sessions() -> None:
    captured: dict[str, Any] = {}

    class Connector:
        @staticmethod
        def FrozenContactGeometryProvider(
            refs: dict[str, Any],
            *,
            session_frame_counts: dict[str, int],
            allowed_root: Any,
        ) -> dict[str, Any]:
            captured.update(
                refs=refs,
                session_frame_counts=session_frame_counts,
                allowed_root=allowed_root,
            )
            return captured

    context = {
        "sessions": {
            "grap_a_cap_004": {
                "geometry": {"joints": {"path": "/literal/joints"}}
            },
            "grap_a_cap_005": {
                "geometry": {"joints": {"path": "/literal/other"}}
            },
        }
    }
    result = producer.build_requested_geometry_provider(
        Connector(), context, ["grap_a_cap_004"], {"grap_a_cap_004": 460}
    )

    assert result["session_frame_counts"] == {"grap_a_cap_004": 460}
    assert set(result["refs"]) == {"grap_a_cap_004"}
    assert (
        result["refs"]["grap_a_cap_004"]["physical_hand_axis"]
        == producer.PHYSICAL_AXIS_CONTRACT
    )


def test_generic_d4_provider_rejects_missing_context_session() -> None:
    with pytest.raises(RuntimeError, match="absent from context"):
        producer.build_requested_geometry_provider(
            object(), {"sessions": {}}, ["grap_a_cap_004"], {"grap_a_cap_004": 460}
        )


def test_formal_object6d_gate_rejects_invalid_even_when_generic_d4_can_unmeasure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "object6d.npz"
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    transforms[1] = np.nan
    np.savez(
        path,
        T_object_to_camera=transforms,
        valid=np.asarray([True, False], dtype=np.bool_),
    )
    payload = path.read_bytes()
    context = {
        "sessions": {
            "grap_a_cap_004": {
                "geometry": {
                    "object6d": {
                        "path": str(path),
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                }
            }
        }
    }
    with pytest.raises(RuntimeError, match="unmeasured/nonfinite"):
        producer.validate_formal_object6d_context(
            context,
            ["grap_a_cap_004"],
            {"grap_a_cap_004": 2},
            evidence_root=tmp_path,
        )


def test_six_geometry_roles_are_captured_once_and_all_consumers_use_same_payload(
    tmp_path: Path,
) -> None:
    session = "grap_a_cap_004"
    geometry: dict[str, dict[str, Any]] = {}
    originals: dict[str, bytes] = {}
    for role in producer.GEOMETRY_ROLES:
        path = tmp_path / f"{role}.npz"
        if role in {"hand_axis_metadata", "cylinder"}:
            path = path.with_suffix(".json")
            payload = json.dumps({"role": role}).encode()
            path.write_bytes(payload)
        elif role == "per_frame_quality":
            path = path.with_suffix(".csv")
            payload = b"hand,frame\nleft,0\nright,0\n"
            path.write_bytes(payload)
        else:
            np.savez(path, marker=np.asarray([len(role)], dtype=np.int64))
            payload = path.read_bytes()
        originals[role] = payload
        geometry[role] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    context = {"sessions": {session: {"geometry": geometry}}}
    captured, evidence = producer.capture_formal_geometry_context(
        context, [session], evidence_root=tmp_path
    )
    for role in producer.GEOMETRY_ROLES:
        Path(geometry[role]["path"]).write_bytes(b"PATH_MUTATED_AFTER_CAPTURE")

    helper = types.SimpleNamespace()
    connector = types.SimpleNamespace(artifact_io=immutable_artifact_io)
    producer.install_captured_geometry_loaders(helper, connector, captured)
    arrays, joints_ref = helper.verified_npz(Path(geometry["joints"]["path"]))
    d4_record = connector._read_ref(
        geometry["object6d"], allowed_root=tmp_path
    )
    assert arrays["marker"].tolist() == [len("joints")]
    assert joints_ref["sha256"] == geometry["joints"]["sha256"]
    assert d4_record.payload == originals["object6d"]
    assert evidence["capture_count"] == len(producer.GEOMETRY_ROLES)
    assert evidence["consumers"] == [
        "context_helper",
        "task29_runtime_geometry",
        "d4_provider",
        "manifest",
    ]


def test_live_retained_run_root_dirfd_rejects_path_replacement_without_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "live-root"
    (root / "frames").mkdir(parents=True)
    descriptor = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    info = os.fstat(descriptor)
    scope = producer.RetainedRunRoot(root, descriptor, info.st_dev, info.st_ino)
    moved = tmp_path / "live-root-held"
    root.rename(moved)
    (root / "frames").mkdir(parents=True)
    producer._RUN_OUTPUT_SCOPE = scope
    try:
        producer.output_exclusive_bytes(root / "frames/artifact.bin", b"held")
        with pytest.raises(RuntimeError, match="identity drift"):
            scope.verify_logical_identity()
    finally:
        producer._RUN_OUTPUT_SCOPE = None
        os.close(descriptor)
    assert (moved / "frames/artifact.bin").read_bytes() == b"held"
    assert not (root / "frames/artifact.bin").exists()


class _IdentityHelper:
    @staticmethod
    def identity(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "path": str(path.absolute()),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


class _WearableAdapter:
    def __init__(self, *, response_frame_delta: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stream_calls = 0
        self.response_frame_delta = response_frame_delta

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(request))
        if request["type"] == "start_session":
            return {"parameter_mapping": {"session_id": request["session_id"]}}
        if request["type"] == "close_session":
            return {}
        assert request["type"] == "add_prompt"
        prompt_id = request["session_id"].rsplit("-", 1)[1]
        prompt_number = int(prompt_id[1:])
        return {
            "frame_index": int(request["frame_index"]) + self.response_frame_delta,
            "outputs": _instances(
                prompt_number % 2 == 0,
                instance_id=100 + prompt_number,
                score=0.5 + prompt_number / 100.0,
            ),
        }

    def handle_stream_request(self, _request: dict[str, Any]):
        self.stream_calls += 1
        raise AssertionError("wearable provider must never propagate")


def test_w1_w4_provider_is_per_frame_independent_and_persists_bound_raw_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipeline import neutral_wearable_identity_v2 as wearable
    from pipeline import unified_d1_d4_wearable_selector as unified

    monkeypatch.setattr(producer, "HEIGHT", 2)
    monkeypatch.setattr(producer, "WIDTH", 3)
    adapter = _WearableAdapter()
    index, timing, prompt_ref = producer.persist_per_frame_w1_w4_raw_provider(
        adapter,
        _Old(),
        _V3(),
        _IdentityHelper(),
        unified,
        wearable,
        tmp_path,
        tmp_path / "component",
        session="grap_a_cap_004",
        frame_count=1,
        evidence_root=Path("/"),
    )

    operational = [
        call for call in adapter.calls if call["type"] in {"start_session", "add_prompt"}
    ]
    assert [call["type"] for call in operational] == [
        "start_session",
        "add_prompt",
    ] * 4
    assert [call["text"] for call in adapter.calls if call["type"] == "add_prompt"] == [
        text for _, text in producer.FROZEN_WEARABLE_PROMPTS
    ]
    assert adapter.stream_calls == 0
    assert timing["model_sessions_started"] == 4
    assert timing["add_prompt_calls"] == 4
    assert timing["propagation_calls"] == 0
    assert timing["method"] == "PER_FRAME_INDEPENDENT_TEXT_PROMPT_NO_PROPAGATION"

    root = tmp_path / "component" / "wearable_raw" / "grap_a_cap_004"
    assert len(list(root.glob("W*/frame_00000/*.png"))) == 4
    assert len(list(root.glob("W*/frame_00000/*.json"))) == 4
    provider_manifest = json.loads((root / "PROVIDER_MANIFEST.json").read_bytes())
    assert provider_manifest["status"] == "ARTIFACT_EXISTS"
    assert provider_manifest["prompt_order"] == ["W1", "W2", "W3", "W4"]
    assert provider_manifest["temporal_propagation_operations"] == 0

    bound = producer.load_bound_wearable_frame(
        index,
        unified,
        wearable,
        prompt_ref,
        frame=0,
        evidence_root=Path("/"),
    )
    assert list(bound) == ["W1", "W2", "W3", "W4"]
    assert all(len(bound[prompt_id]) == 1 for prompt_id in bound)
    assert all(
        item.instance.mask.dtype == np.bool_
        for rows in bound.values()
        for item in rows
    )


def test_w1_w4_provider_rejects_response_frame_drift_without_propagation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipeline import neutral_wearable_identity_v2 as wearable
    from pipeline import unified_d1_d4_wearable_selector as unified

    monkeypatch.setattr(producer, "HEIGHT", 2)
    monkeypatch.setattr(producer, "WIDTH", 3)
    adapter = _WearableAdapter(response_frame_delta=1)
    with pytest.raises(RuntimeError, match="response index drift"):
        producer.persist_per_frame_w1_w4_raw_provider(
            adapter,
            _Old(),
            _V3(),
            _IdentityHelper(),
            unified,
            wearable,
            tmp_path,
            tmp_path / "component",
            session="grap_a_cap_004",
            frame_count=1,
            evidence_root=Path("/"),
        )
    assert adapter.stream_calls == 0
    assert [call["type"] for call in adapter.calls] == [
        "start_session",
        "add_prompt",
        "close_session",
    ]


def test_prompt_contract_a_to_b_replacement_consumes_published_a_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipeline import neutral_wearable_identity_v2 as wearable
    from pipeline import unified_d1_d4_wearable_selector as unified

    monkeypatch.setattr(producer, "HEIGHT", 2)
    monkeypatch.setattr(producer, "WIDTH", 3)
    original = producer.WEARABLE_PROMPT_CONTRACT.read_bytes()
    mutable = tmp_path / "mutable_prompt.json"
    mutable.write_bytes(original)
    monkeypatch.setattr(producer, "WEARABLE_PROMPT_CONTRACT", mutable)

    class MutatingOld(_Old):
        @staticmethod
        def exclusive_bytes(path: Path, payload: bytes) -> None:
            if path.name == "FROZEN_PROMPT_CONTRACT.json":
                mutable.write_bytes(b'{"replaced":"B"}\n')
            _Old.exclusive_bytes(path, payload)

    _, _, prompt_ref = producer.persist_per_frame_w1_w4_raw_provider(
        _WearableAdapter(),
        MutatingOld(),
        _V3(),
        _IdentityHelper(),
        unified,
        wearable,
        tmp_path,
        tmp_path / "component",
        session="grap_a_cap_004",
        frame_count=1,
        evidence_root=Path("/"),
    )
    assert mutable.read_bytes() != original
    frozen = Path(prompt_ref.path)
    assert frozen.read_bytes() == original
    assert prompt_ref.sha256 == hashlib.sha256(original).hexdigest()


def test_prompt_contract_b_to_a_replacement_cannot_admit_b(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipeline import neutral_wearable_identity_v2 as wearable
    from pipeline import unified_d1_d4_wearable_selector as unified

    mutable = tmp_path / "mutable_prompt.json"
    mutable.write_bytes(b'{"replaced":"B"}\n')
    monkeypatch.setattr(producer, "WEARABLE_PROMPT_CONTRACT", mutable)
    with pytest.raises(RuntimeError, match="prompt contract SHA drift"):
        producer.persist_per_frame_w1_w4_raw_provider(
            _WearableAdapter(),
            _Old(),
            _V3(),
            _IdentityHelper(),
            unified,
            wearable,
            tmp_path,
            tmp_path / "component",
            session="grap_a_cap_004",
            frame_count=1,
            evidence_root=Path("/"),
        )
    assert not (tmp_path / "component/wearable_raw").exists()


def test_bound_wearable_loader_rejects_missing_prompt_coverage() -> None:
    from pipeline import neutral_wearable_identity_v2 as wearable
    from pipeline import unified_d1_d4_wearable_selector as unified

    incomplete = {0: {prompt_id: [] for prompt_id in ("W1", "W2", "W3")}}
    with pytest.raises(RuntimeError, match="coverage drift"):
        producer.load_bound_wearable_frame(
            incomplete,
            unified,
            wearable,
            object(),
            frame=0,
            evidence_root=Path("/"),
        )


def test_wearable_hand_scale_matches_frozen_task32_v3_rule() -> None:
    joints = np.full((21, 2), np.nan, dtype=np.float64)
    joints[:5] = np.asarray(
        [[10.0, 10.0], [13.0, 12.0], [18.0, 14.0], [16.0, 20.0], [11.0, 17.0]]
    )
    assert producer.wearable_hand_scale(joints) == 10.0
    assert producer.wearable_hand_scale(None) is None
    with pytest.raises(RuntimeError, match="shape drift"):
        producer.wearable_hand_scale(joints[:4])
    joints[4] = np.nan
    assert producer.wearable_hand_scale(joints) is None


def test_formal_final_h_rejects_one_frame_arm_hold_or_unbound_accept() -> None:
    accepted = {
        "grap_a_cap_004": {
            "frames_processed": 1,
            "final_h_left_right_png_count": 2,
            "final_h_counts": {
                side: {
                    "arm_accept": 1,
                    "arm_hold_zero_artifact": 0,
                    "accepted_raw_identity": 1,
                }
                for side in producer.SIDES
            },
        }
    }
    assert producer.validate_formal_final_h_coverage(accepted) == 1

    held = json.loads(json.dumps(accepted))
    held["grap_a_cap_004"]["final_h_counts"]["left"] = {
        "arm_accept": 0,
        "arm_hold_zero_artifact": 1,
        "accepted_raw_identity": 0,
    }
    with pytest.raises(RuntimeError, match="HOLD or unbound"):
        producer.validate_formal_final_h_coverage(held)

    unbound = json.loads(json.dumps(accepted))
    unbound["grap_a_cap_004"]["final_h_counts"]["right"][
        "accepted_raw_identity"
    ] = 0
    with pytest.raises(RuntimeError, match="HOLD or unbound"):
        producer.validate_formal_final_h_coverage(unbound)


@pytest.mark.parametrize(
    ("consumed", "replacement"),
    [(b"VALUE = 'A'\n", b"VALUE = 'B'\n"), (b"VALUE = 'B'\n", b"VALUE = 'A'\n")],
)
def test_dependency_bundle_executes_bound_bytes_after_origin_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumed: bytes,
    replacement: bytes,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    origin = project / "dep.py"
    origin.write_bytes(consumed)
    run_root = tmp_path / "run"
    bound = run_root / "bundle/dep.py"
    bound.parent.mkdir(parents=True)
    bound.write_bytes(consumed)

    def ref(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    sources = {
        "dep": {
            "relative_path": "dep.py",
            "origin": ref(origin),
            "bound": ref(bound),
        }
    }
    source_digest = producer._dependency_source_digest(sources)
    manifest = run_root / "bundle/MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "eligible68-runtime-dependency-bundle-v1",
                "status": "ARTIFACT_EXISTS",
                "source_bundle_sha256": source_digest,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    origin.write_bytes(replacement)
    monkeypatch.setattr(producer, "PROJECT", project)
    monkeypatch.setattr(producer, "RUNTIME_DEPENDENCIES", {"dep": Path("dep.py")})
    _, loader, consumed_names = producer.load_dependency_bundle(
        manifest, expected_sha256=manifest_sha, run_root=run_root
    )
    module = loader("bound_dep_test", origin)
    assert module.VALUE == consumed.decode().split("'")[1]
    assert origin.read_bytes() == replacement
    assert consumed_names == {"dep"}


@pytest.mark.parametrize("swap_direction", ["A_TO_B", "B_TO_A"])
def test_held_raw_model_and_overlay_consume_same_pre_swap_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_direction: str,
) -> None:
    monkeypatch.setattr(producer, "WIDTH", 3)
    monkeypatch.setattr(producer, "HEIGHT", 2)

    def png(value: int) -> bytes:
        stream = BytesIO()
        Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8)).save(
            stream, format="PNG"
        )
        return stream.getvalue()

    payload_a, payload_b = png(10), png(240)
    consumed, replacement = (
        (payload_a, payload_b)
        if swap_direction == "A_TO_B"
        else (payload_b, payload_a)
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    source = run_root / "frame.png"
    source.write_bytes(consumed)
    record = {
        "local_index": 0,
        "raw_path": str(source),
        "raw_bytes": len(consumed),
        "raw_sha256": hashlib.sha256(consumed).hexdigest(),
    }
    held, index_root, _ = producer.prepare_held_raw_index(
        [record], evidence_root=run_root
    )
    replacement_path = run_root / "replacement.png"
    replacement_path.write_bytes(replacement)
    os.replace(replacement_path, source)
    assert (index_root / "00000.png").read_bytes() == consumed
    assert held.payload(0) == consumed
    assert source.read_bytes() == replacement
    with pytest.raises(RuntimeError, match="held RAW inode changed"):
        held.verify_and_cleanup()


def test_formal_evidence_root_cannot_expand_to_root_or_outside_path(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-used")
    with pytest.raises(RuntimeError, match="escapes|strict descendant"):
        producer._strict_descendant(outside, Path("/"), "formal evidence")
    allowed = tmp_path / "run"
    allowed.mkdir()
    with pytest.raises(RuntimeError, match="escapes"):
        producer._strict_descendant(outside, allowed, "formal evidence")
