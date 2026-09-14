from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from pipeline import object6d_direct_observation_successor as successor


TIMESTAMP_NS = 1_987_654_321


def _authority_bundle(tmp_path: Path) -> dict[str, successor.AuthorityEvidence]:
    result: dict[str, successor.AuthorityEvidence] = {}
    for index, role in enumerate(successor.REQUIRED_AUTHORITY_ROLES):
        decision = f"authorized-{role.lower()}"
        if role == successor.TEMPORAL_POLICY_ROLE:
            decision = successor.DIRECT_OBSERVATION_TEMPORAL_POLICY
        if role == successor.REDETECTION_POLICY_ROLE:
            decision = successor.PER_FRAME_REDETECTION_POLICY
        authority_id = f"authority-{index}"
        source_payload = f"frozen-source-artifact-{index:02d}-{role}".encode("ascii")
        source_path = tmp_path / f"authority-source-{index:02d}.bin"
        source_path.write_bytes(source_payload)
        source_sha256 = hashlib.sha256(source_payload).hexdigest()
        record = {
            "schema": successor.AUTHORITY_RECORD_SCHEMA,
            "status": "ARTIFACT_EXISTS",
            "role": role,
            "authority_id": authority_id,
            "decision": decision,
            "source_artifact": {
                "path": str(source_path),
                "bytes": len(source_payload),
                "sha256": source_sha256,
            },
            "h5b_decision_authorized": True,
            "label_independent": True,
            "canary_only": True,
            "formal_consumer_allowed": False,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        authority_path = tmp_path / f"authority-{index:02d}.json"
        authority_path.write_bytes(payload)
        result[role] = successor.AuthorityEvidence(
            role=role,
            authority_id=authority_id,
            authority_path=authority_path,
            authority_bytes=len(payload),
            authority_sha256=hashlib.sha256(payload).hexdigest(),
            source_path=source_path,
            source_bytes=len(source_payload),
            source_sha256=source_sha256,
            decision=decision,
            artifact_exists=True,
            h5b_decision_authorized=True,
            label_independent=True,
        )
    return result


def _evidence_ref(
    tmp_path: Path,
    kind: str,
    suffix: int,
    *,
    camera_to_world: np.ndarray | None = None,
) -> successor.EvidenceDigest:
    if kind == "CAMERA_TO_WORLD":
        stream = io.BytesIO()
        np.save(stream, camera_to_world, allow_pickle=False)
        payload = stream.getvalue()
    else:
        payload = f"direct-evidence-{kind}-{suffix}".encode("ascii")
    path = tmp_path / f"evidence-{suffix:02d}.bin"
    path.write_bytes(payload)
    return successor.EvidenceDigest(
        kind=kind,
        source_path=path,
        source_bytes=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _row_evidence(
    tmp_path: Path,
    camera_to_world: np.ndarray,
    *,
    correspondences: int = 20,
    inliers: int = 9,
    ratio: float = 0.45,
    residual_m: float = 0.012,
    source_frame_indices: tuple[int, ...] = (371,),
    operation_counters: successor.OperationCounters | None = None,
    **flags: bool,
) -> successor.DirectObservationEvidence:
    values = {
        "used_neighbor_pose": False,
        "used_interpolation": False,
        "used_propagation": False,
        "used_fallback": False,
        "used_temporal_smoothing": False,
    }
    values.update(flags)
    return successor.DirectObservationEvidence(
        row_index=0,
        session_id=successor.CANARY_SESSION_ID,
        frame_index=successor.CANARY_FRAME_INDEX,
        timestamp_ns=TIMESTAMP_NS,
        direct_observation_attempted=True,
        source_frame_indices=source_frame_indices,
        positive_depth_correspondences=correspondences,
        ransac_inliers=inliers,
        ransac_inlier_ratio=ratio,
        residual_m=residual_m,
        raw_mask=_evidence_ref(tmp_path, "DIRECT_RAW_OBJECT_MASK", 1),
        tracks=_evidence_ref(tmp_path, "DIRECT_CANONICAL_TRACKS", 2),
        positive_depth=_evidence_ref(tmp_path, "POSITIVE_STEREO_DEPTH", 3),
        camera_to_world=_evidence_ref(
            tmp_path,
            "CAMERA_TO_WORLD",
            4,
            camera_to_world=camera_to_world,
        ),
        operation_counters=(
            operation_counters or successor.OperationCounters(0, 0, 0, 0, 0)
        ),
        **values,
    )


def _valid_arrays() -> tuple[dict[str, np.ndarray], np.ndarray]:
    transform_camera_to_world = np.eye(4, dtype=np.float64)
    transform_camera_to_world[0, 3] = 1.25
    transform_object_to_camera = np.eye(4, dtype=np.float64)
    transform_object_to_camera[2, 3] = 0.75
    transform_object_to_world = transform_camera_to_world @ transform_object_to_camera
    arrays = {
        "T_object_to_world": transform_object_to_world[None],
        "T_object_to_camera": transform_object_to_camera[None],
        "T_object_to_world_raw_center_corrected": transform_object_to_world[
            None
        ].copy(),
        "confidence": np.asarray([0.9], dtype=np.float64),
        "valid": np.asarray([True], dtype=np.bool_),
        "visual_observed": np.asarray([True], dtype=np.bool_),
        "timestamp_ns": np.asarray([TIMESTAMP_NS], dtype=np.int64),
        "cylinder_radius_m": np.asarray(0.03, dtype=np.float64),
        "cylinder_height_m": np.asarray(0.12, dtype=np.float64),
    }
    return arrays, transform_camera_to_world[None]


def _invalid_arrays() -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays, camera_to_world = _valid_arrays()
    for key in successor.POSE_KEYS:
        arrays[key] = np.full((1, 4, 4), np.nan, dtype=np.float64)
    arrays["confidence"] = np.asarray([0.0], dtype=np.float64)
    arrays["valid"] = np.asarray([False], dtype=np.bool_)
    arrays["visual_observed"] = np.asarray([False], dtype=np.bool_)
    return arrays, camera_to_world


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> tuple[int, str]:
    np.savez(path, **arrays)
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _npy_payload(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=allow_pickle)
    return stream.getvalue()


def _write_npz_with_member_override(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    key: str,
    member_payload: bytes,
) -> tuple[int, str]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for current_key, array in arrays.items():
            archive.writestr(
                f"{current_key}.npy",
                member_payload
                if current_key == key
                else _npy_payload(array, allow_pickle=False),
            )
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _validate(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
    camera_to_world: np.ndarray | None,
    *,
    evidence: successor.DirectObservationEvidence | None = None,
    authorities: dict[str, successor.AuthorityEvidence] | None = None,
    mapping: successor.CanaryRowMapping | None = None,
    path_name: str = "canary.npz",
) -> successor.ValidationCard:
    path = tmp_path / path_name
    size, digest = _write_npz(path, arrays)
    return successor.validate_canary_npz(
        path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            mapping
            or successor.CanaryRowMapping(
                row_index=0,
                session_id=successor.CANARY_SESSION_ID,
                frame_index=successor.CANARY_FRAME_INDEX,
                timestamp_ns=TIMESTAMP_NS,
                cylinder_radius_m=0.03,
                cylinder_height_m=0.12,
            ),
        ),
        direct_observation_evidence=(
            evidence
            or _row_evidence(
                tmp_path,
                camera_to_world
                if camera_to_world is not None
                else np.eye(4, dtype=np.float64)[None],
            ),
        ),
        authorities=(
            authorities if authorities is not None else _authority_bundle(tmp_path)
        ),
    )


def test_exact_nine_key_direct_observation_canary_passes_at_fixed_boundaries(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    card = _validate(tmp_path, arrays, camera_to_world)
    assert card.status == "CARD_VALIDATED_CPU_ONLY"
    assert card.failure_reasons == ()
    assert card.rows_total == 1
    assert card.rows_valid == 1
    assert card.session_id == "grap_a_cap_059"
    assert card.frame_index == 371
    assert card.minimum_3d_correspondences == 8
    assert card.minimum_ransac_inliers == 8
    assert card.minimum_ransac_inlier_ratio == 0.45
    assert card.maximum_residual_m == 0.012
    assert card.direct_observation_only is True
    assert card.formal_consumer_allowed is False
    assert card.gpu_executed is False
    assert card.model_loaded is False


@pytest.mark.parametrize("missing_role", successor.REQUIRED_AUTHORITY_ROLES)
def test_every_h5b_or_input_contract_is_a_required_authority(
    tmp_path: Path, missing_role: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    del authorities[missing_role]
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any(missing_role in reason for reason in card.failure_reasons)


def test_required_authority_role_set_is_exactly_the_eighteen_frozen_classes() -> None:
    assert len(successor.REQUIRED_AUTHORITY_ROLES) == 18
    assert len(set(successor.REQUIRED_AUTHORITY_ROLES)) == 18
    assert set(successor.REQUIRED_AUTHORITY_ROLES) == {
        "CANONICAL_OBJECT_FRAME_AND_YAW",
        "OBJECT_DETECTION_PROMPT_CONTRACT",
        "OBJECT_DETECTION_MODEL_AND_WEIGHT",
        "PER_FRAME_REDETECTION_POLICY",
        "FOUNDATION_STEREO_SOURCE",
        "FOUNDATION_STEREO_CHECKPOINT",
        "COTRACKER_SOURCE",
        "COTRACKER_CHECKPOINT",
        "STEREO_LEFT_RIGHT_LAYOUT",
        "STEREO_EXTRINSICS_FRAME_AXES_AND_DIRECTION",
        "MP4_PTS_TO_SLAM_TO_LEFT_CAMERA_MAPPING",
        "CAMERA_TO_WORLD_TRAJECTORY",
        "DIRECT_RAW_OBJECT_MASK_AND_CANONICAL_TRACKS",
        "MATCHER_SAMPLING_POLICY",
        "RANSAC_SEED_AND_TIEBREAK_POLICY",
        "TEMPORAL_OBSERVATION_AND_WRITER_POLICY",
        "CYLINDER_GEOMETRY_REFERENCE_AND_REUSE_POLICY",
        "SUCCESSOR_WRITER_AND_NPZ_LINEAGE",
    }


def test_extra_or_subclassed_authority_records_fail_closed(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    authorities["EXTRA_UNAUTHORIZED_ROLE"] = next(iter(authorities.values()))
    extra = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert extra.status == "CARD_FAILED"
    assert extra.authorities_valid is False
    assert any("unexpected authority" in value for value in extra.failure_reasons)

    class ForgedAuthority(successor.AuthorityEvidence):
        pass

    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    authorities[role] = ForgedAuthority(**authorities[role].__dict__)
    forged = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=authorities,
        path_name="subclassed-authority.npz",
    )
    assert forged.status == "CARD_FAILED"
    assert forged.authorities_valid is False
    assert any("wrong type" in value for value in forged.failure_reasons)


@pytest.mark.parametrize("bad_key", [7, ("tuple-key",)])
def test_non_string_authority_mapping_keys_are_card_failed(
    tmp_path: Path, bad_key: object
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities: dict[object, object] = dict(_authority_bundle(tmp_path))
    authorities[bad_key] = next(iter(authorities.values()))
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=authorities,  # type: ignore[arg-type]
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("exact strings" in value for value in card.failure_reasons)


def test_authority_mapping_values_are_exact_before_role_set_operations(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities: dict[str, object] = dict(_authority_bundle(tmp_path))
    authorities[successor.CANONICAL_FRAME_YAW_ROLE] = object()
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=authorities,  # type: ignore[arg-type]
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("wrong type" in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    "location",
    [
        "authority_role",
        "authority_id",
        "authority_decision",
        "authority_sha256",
        "authority_source_sha256",
        "evidence_kind",
        "evidence_sha256",
        "mapping_session",
        "evidence_session",
    ],
)
@pytest.mark.parametrize("bad_kind", ["subclass", "object"])
def test_all_public_string_fields_are_exact_before_untrusted_operations(
    tmp_path: Path, location: str, bad_kind: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    mapping = successor.CanaryRowMapping(
        0,
        successor.CANARY_SESSION_ID,
        successor.CANARY_FRAME_INDEX,
        TIMESTAMP_NS,
        0.03,
        0.12,
    )
    operation_called = False

    def explode() -> None:
        nonlocal operation_called
        operation_called = True
        raise RuntimeError("untrusted string operation must not run")

    class ExplodingString(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            explode()
            raise AssertionError("unreachable")

        def __ne__(self, other: object) -> bool:
            explode()
            raise AssertionError("unreachable")

        def __str__(self) -> str:
            explode()
            raise AssertionError("unreachable")

        def __format__(self, format_spec: str) -> str:
            explode()
            raise AssertionError("unreachable")

        def __len__(self) -> int:
            explode()
            raise AssertionError("unreachable")

        def strip(self, chars: str | None = None) -> str:
            explode()
            raise AssertionError("unreachable")

    class ExplodingObject:
        def __eq__(self, other: object) -> bool:
            explode()
            raise AssertionError("unreachable")

        def __ne__(self, other: object) -> bool:
            explode()
            raise AssertionError("unreachable")

        def __str__(self) -> str:
            explode()
            raise AssertionError("unreachable")

        def __format__(self, format_spec: str) -> str:
            explode()
            raise AssertionError("unreachable")

    bad_value: object = (
        ExplodingString("apparently-valid")
        if bad_kind == "subclass"
        else ExplodingObject()
    )
    role = successor.CANONICAL_FRAME_YAW_ROLE
    authority_field = {
        "authority_role": "role",
        "authority_id": "authority_id",
        "authority_decision": "decision",
        "authority_sha256": "authority_sha256",
        "authority_source_sha256": "source_sha256",
    }.get(location)
    if authority_field is not None:
        authorities[role] = replace(
            authorities[role],
            **{authority_field: bad_value},
        )
    elif location in {"evidence_kind", "evidence_sha256"}:
        assert evidence.raw_mask is not None
        evidence_field = "kind" if location == "evidence_kind" else "source_sha256"
        evidence = replace(
            evidence,
            raw_mask=replace(
                evidence.raw_mask,
                **{evidence_field: bad_value},
            ),
        )
    elif location == "mapping_session":
        mapping = replace(mapping, session_id=bad_value)
    else:
        evidence = replace(evidence, session_id=bad_value)
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=authorities,
        mapping=mapping,
        evidence=evidence,
        path_name=f"bad-string-{location}-{bad_kind}.npz",
    )
    assert operation_called is False
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False


def test_malformed_custom_mapping_iteration_and_access_are_card_failed(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    base = _authority_bundle(tmp_path)

    class ExplodingItems(dict[str, successor.AuthorityEvidence]):
        def items(self) -> object:
            raise RuntimeError("items exploded")

    exploded_items = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=ExplodingItems(base),
    )
    assert exploded_items.status == "CARD_FAILED"
    assert any(
        "cannot be snapshotted" in value for value in exploded_items.failure_reasons
    )

    class ExplodingAccess(Mapping[str, successor.AuthorityEvidence]):
        def __iter__(self) -> Iterator[str]:
            return iter(base)

        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, key: str) -> successor.AuthorityEvidence:
            raise RuntimeError(f"access exploded: {key}")

    exploded_access = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=ExplodingAccess(),
        path_name="exploding-access.npz",
    )
    assert exploded_access.status == "CARD_FAILED"
    assert any(
        "cannot be snapshotted" in value for value in exploded_access.failure_reasons
    )


def test_custom_mapping_duplicate_item_pairs_are_not_silently_overwritten(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    base = _authority_bundle(tmp_path)

    class DuplicateItems(Mapping[str, successor.AuthorityEvidence]):
        def __iter__(self) -> Iterator[str]:
            return iter(base)

        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, key: str) -> successor.AuthorityEvidence:
            return base[key]

        def items(self) -> object:
            first = successor.CANONICAL_FRAME_YAW_ROLE
            return ((first, base[first]), (first, base[first]), *base.items())

    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=DuplicateItems(),
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("duplicate key" in value for value in card.failure_reasons)


def test_unapproved_or_placeholder_authority_is_not_treated_as_a_value(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    authorities[role] = successor.AuthorityEvidence(
        **{
            **authorities[role].__dict__,
            "decision": "TBD",
            "h5b_decision_authorized": False,
        }
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False


def test_invalid_row_has_one_canonical_encoding_and_remains_card_failed(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _invalid_arrays()
    evidence = _row_evidence(
        tmp_path,
        camera_to_world,
        correspondences=7,
        inliers=7,
        ratio=1.0,
    )
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_FAILED"
    assert card.schema_valid is True
    assert card.rows_valid == 0
    assert card.failed_row_indices == (0,)
    assert any(
        "fixed direct-observation gates" in value for value in card.failure_reasons
    )


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("T_object_to_world", np.zeros((1, 4, 4), dtype=np.float64)),
        ("T_object_to_camera", np.full((1, 4, 4), np.inf, dtype=np.float64)),
        (
            "T_object_to_world_raw_center_corrected",
            np.zeros((1, 4, 4), dtype=np.float64),
        ),
        ("confidence", np.asarray([0.1], dtype=np.float64)),
        ("visual_observed", np.asarray([True], dtype=np.bool_)),
    ],
)
def test_invalid_row_rejects_pose_or_observation_fabrication(
    tmp_path: Path, key: str, replacement: np.ndarray
) -> None:
    arrays, camera_to_world = _invalid_arrays()
    arrays[key] = replacement
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=_row_evidence(
            tmp_path,
            camera_to_world,
            correspondences=7,
            inliers=7,
            ratio=1.0,
        ),
    )
    assert card.status == "CARD_FAILED"
    assert card.schema_valid is False


def test_valid_pose_must_be_right_handed_se3_and_obey_world_composition(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    arrays["T_object_to_world_raw_center_corrected"][0, 0, 0] = -1.0
    reflected = _validate(tmp_path, arrays, camera_to_world, path_name="reflected.npz")
    assert reflected.status == "CARD_FAILED"
    assert any("right-handed SE(3)" in value for value in reflected.failure_reasons)

    arrays, camera_to_world = _valid_arrays()
    arrays["T_object_to_world"][0, 1, 3] += 0.01
    drifted = _validate(tmp_path, arrays, camera_to_world, path_name="drifted.npz")
    assert drifted.status == "CARD_FAILED"
    assert any("T_W_O != T_W_C @ T_C_O" in value for value in drifted.failure_reasons)


@pytest.mark.parametrize(
    "evidence_kwargs",
    [
        {"correspondences": 7, "inliers": 7, "ratio": 1.0},
        {"correspondences": 20, "inliers": 7, "ratio": 0.35},
        {"correspondences": 20, "inliers": 8, "ratio": 0.40},
        {
            "correspondences": 20,
            "inliers": 9,
            "ratio": 0.45,
            "residual_m": 0.0120001,
        },
    ],
)
def test_valid_flag_cannot_bypass_any_fixed_quality_gate(
    tmp_path: Path, evidence_kwargs: dict[str, float | int]
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world, **evidence_kwargs)
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_FAILED"
    assert any(
        "valid row fails fixed direct-observation gates" in value
        for value in card.failure_reasons
    )


@pytest.mark.parametrize(
    "override",
    [
        {"source_frame_indices": (370, 371)},
        {"used_neighbor_pose": True},
        {"used_interpolation": True},
        {"used_propagation": True},
        {"used_fallback": True},
        {"used_temporal_smoothing": True},
    ],
)
def test_neighbor_temporal_and_fallback_paths_are_unconditionally_forbidden(
    tmp_path: Path, override: dict[str, object]
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world, **override)
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_FAILED"
    assert card.direct_observation_only is False


def test_npz_keys_mapping_and_timestamp_are_exact(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    arrays["hidden_pose_fallback"] = np.zeros((1, 4, 4), dtype=np.float64)
    extra = _validate(tmp_path, arrays, camera_to_world, path_name="extra.npz")
    assert extra.status == "CARD_FAILED"
    assert any("exactly nine keys" in value for value in extra.failure_reasons)

    arrays, camera_to_world = _valid_arrays()
    wrong_mapping = successor.CanaryRowMapping(
        row_index=0,
        session_id=successor.CANARY_SESSION_ID,
        frame_index=370,
        timestamp_ns=TIMESTAMP_NS,
        cylinder_radius_m=0.03,
        cylinder_height_m=0.12,
    )
    mapped = _validate(tmp_path, arrays, camera_to_world, mapping=wrong_mapping)
    assert mapped.status == "CARD_FAILED"
    assert any("059 frame 371" in value for value in mapped.failure_reasons)

    arrays, camera_to_world = _valid_arrays()
    arrays["timestamp_ns"][0] += 1
    timestamp = _validate(tmp_path, arrays, camera_to_world, path_name="timestamp.npz")
    assert timestamp.status == "CARD_FAILED"
    assert any("timestamp" in value for value in timestamp.failure_reasons)


def test_npz_bytes_are_same_fd_bound_and_symlinks_are_rejected(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "target.npz"
    size, digest = _write_npz(path, arrays)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    card = successor.validate_canary_npz(
        link,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert any("no-follow" in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        (
            "T_object_to_world",
            np.eye(4, dtype=np.float32)[None],
        ),
        ("confidence", np.asarray([0.9], dtype=np.float32)),
        ("timestamp_ns", np.asarray([TIMESTAMP_NS], dtype=np.int32)),
        ("cylinder_radius_m", np.asarray(0.03, dtype=np.float32)),
        ("valid", np.asarray([1], dtype=np.uint8)),
    ],
)
def test_canonical_npz_dtypes_are_exact(
    tmp_path: Path, key: str, replacement: np.ndarray
) -> None:
    arrays, camera_to_world = _valid_arrays()
    arrays[key] = replacement
    card = _validate(tmp_path, arrays, camera_to_world)
    assert card.status == "CARD_FAILED"
    assert card.schema_valid is False


def test_shape_nan_and_nonhomogeneous_pose_attacks_fail(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    arrays["confidence"] = np.asarray([[0.9]], dtype=np.float64)
    shape = _validate(tmp_path, arrays, camera_to_world, path_name="shape.npz")
    assert shape.status == "CARD_FAILED"

    arrays, camera_to_world = _valid_arrays()
    arrays["T_object_to_camera"][0, 0, 0] = np.nan
    nonfinite = _validate(tmp_path, arrays, camera_to_world, path_name="nonfinite.npz")
    assert nonfinite.status == "CARD_FAILED"
    assert any("right-handed SE(3)" in value for value in nonfinite.failure_reasons)

    arrays, camera_to_world = _valid_arrays()
    arrays["T_object_to_camera"][0, 3, 0] = 1e-12
    nonhomogeneous = _validate(
        tmp_path, arrays, camera_to_world, path_name="nonhomogeneous.npz"
    )
    assert nonhomogeneous.status == "CARD_FAILED"
    assert any(
        "right-handed SE(3)" in value for value in nonhomogeneous.failure_reasons
    )


def test_npz_huge_shape_is_rejected_before_array_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    malformed = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        malformed,
        {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (1_000_000_000,),
        },
    )
    path = tmp_path / "huge-shape-header.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            archive.writestr(
                f"{key}.npy",
                malformed.getvalue()
                if key == "T_object_to_world"
                else stream.getvalue(),
            )
    allocation_called = False

    def forbidden_frombuffer(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("malicious shape reached array allocation")

    monkeypatch.setattr(successor.np, "frombuffer", forbidden_frombuffer)
    payload = path.read_bytes()
    card = successor.validate_canary_npz(
        path,
        expected_npz_bytes=len(payload),
        expected_npz_sha256=hashlib.sha256(payload).hexdigest(),
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert allocation_called is False
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False
    assert any("shape" in reason for reason in card.failure_reasons)


def test_npy_member_header_and_payload_must_match_exact_fixed_schema(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    pose_key = "T_object_to_world"
    pose = arrays[pose_key]
    valid_payload = _npy_payload(pose)

    header_dict = {
        "descr": np.lib.format.dtype_to_descr(pose.dtype),
        "fortran_order": False,
        "shape": pose.shape,
    }
    header_core = repr(header_dict).encode("latin1")
    padding = successor.MAX_NPZ_HEADER_BYTES + 1 - len(header_core) - 1
    while (10 + len(header_core) + padding + 1) % 64:
        padding += 1
    overlong_header = header_core + (b" " * padding) + b"\n"
    overlong_payload = (
        b"\x93NUMPY\x01\x00"
        + len(overlong_header).to_bytes(2, "little")
        + overlong_header
        + pose.tobytes(order="C")
    )
    fortran_pose = np.asfortranarray(pose)
    assert fortran_pose.flags.f_contiguous
    assert not fortran_pose.flags.c_contiguous
    cases = (
        ("truncated", valid_payload[:-1], "byte length"),
        ("overlong-header", overlong_payload, "unreasonably large"),
        ("fortran", _npy_payload(fortran_pose), "C order"),
        (
            "object",
            _npy_payload(np.empty(pose.shape, dtype=object), allow_pickle=True),
            "dtype",
        ),
        ("complex", _npy_payload(pose.astype(np.complex128)), "dtype"),
        ("float32", _npy_payload(pose.astype(np.float32)), "dtype"),
    )
    for name, member_payload, expected_reason in cases:
        path = tmp_path / f"bad-member-{name}.npz"
        size, digest = _write_npz_with_member_override(
            path,
            arrays,
            key=pose_key,
            member_payload=member_payload,
        )
        card = successor.validate_canary_npz(
            path,
            expected_npz_bytes=size,
            expected_npz_sha256=digest,
            camera_to_world=camera_to_world,
            row_mapping=(
                successor.CanaryRowMapping(
                    0,
                    successor.CANARY_SESSION_ID,
                    successor.CANARY_FRAME_INDEX,
                    TIMESTAMP_NS,
                    0.03,
                    0.12,
                ),
            ),
            direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
            authorities=_authority_bundle(tmp_path),
        )
        assert card.status == "CARD_FAILED"
        assert card.formal_consumer_allowed is False
        assert any(expected_reason in reason for reason in card.failure_reasons)


def test_all_fixed_gate_boundaries_are_inclusive_and_invalid_cannot_claim_them(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    minimum_counts = _row_evidence(
        tmp_path,
        camera_to_world,
        correspondences=8,
        inliers=8,
        ratio=1.0,
        residual_m=0.012,
    )
    card = _validate(tmp_path, arrays, camera_to_world, evidence=minimum_counts)
    assert card.status == "CARD_VALIDATED_CPU_ONLY"

    arrays, camera_to_world = _invalid_arrays()
    passing_evidence = _row_evidence(
        tmp_path,
        camera_to_world,
        correspondences=20,
        inliers=9,
        ratio=0.45,
        residual_m=0.012,
    )
    invalid = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=passing_evidence,
        path_name="invalid-passing-gates.npz",
    )
    assert invalid.status == "CARD_FAILED"
    assert any("disagrees" in value for value in invalid.failure_reasons)


@pytest.mark.parametrize(
    "field",
    ["mapping_radius", "ransac_inlier_ratio", "residual_m", "correspondences"],
)
def test_oversized_python_integers_are_card_failed_without_conversion_escape(
    tmp_path: Path, field: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    oversized = 10**10000
    if field == "mapping_radius":
        mapping = successor.CanaryRowMapping(
            row_index=0,
            session_id=successor.CANARY_SESSION_ID,
            frame_index=successor.CANARY_FRAME_INDEX,
            timestamp_ns=TIMESTAMP_NS,
            cylinder_radius_m=oversized,
            cylinder_height_m=0.12,
        )
        card = _validate(tmp_path, arrays, camera_to_world, mapping=mapping)
    else:
        evidence = _row_evidence(tmp_path, camera_to_world)
        evidence_field = {
            "ransac_inlier_ratio": "ransac_inlier_ratio",
            "residual_m": "residual_m",
            "correspondences": "positive_depth_correspondences",
        }[field]
        card = _validate(
            tmp_path,
            arrays,
            camera_to_world,
            evidence=replace(evidence, **{evidence_field: oversized}),
        )
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False


def test_exact_numpy_integer_and_real_scalars_are_accepted_and_normalized(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    mapping = successor.CanaryRowMapping(
        row_index=np.int64(0),
        session_id=successor.CANARY_SESSION_ID,
        frame_index=np.int64(successor.CANARY_FRAME_INDEX),
        timestamp_ns=np.int64(TIMESTAMP_NS),
        cylinder_radius_m=np.float64(0.03),
        cylinder_height_m=np.float64(0.12),
    )
    evidence = _row_evidence(
        tmp_path,
        camera_to_world,
        correspondences=np.int64(20),
        inliers=np.int64(9),
        ratio=np.float64(0.45),
        residual_m=np.float64(0.012),
        source_frame_indices=(np.int64(successor.CANARY_FRAME_INDEX),),
    )
    evidence = replace(
        evidence,
        row_index=np.int64(0),
        frame_index=np.int64(successor.CANARY_FRAME_INDEX),
        timestamp_ns=np.int64(TIMESTAMP_NS),
        operation_counters=successor.OperationCounters(
            np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        ),
    )
    authorities = _authority_bundle(tmp_path)
    role = successor.REQUIRED_AUTHORITY_ROLES[0]
    authority = authorities[role]
    authorities[role] = replace(
        authority,
        authority_bytes=np.uint64(authority.authority_bytes),
        source_bytes=np.uint64(authority.source_bytes),
    )
    assert evidence.raw_mask is not None
    evidence = replace(
        evidence,
        raw_mask=replace(
            evidence.raw_mask,
            source_bytes=np.uint64(evidence.raw_mask.source_bytes),
        ),
    )
    accepted = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        mapping=mapping,
        evidence=evidence,
        authorities=authorities,
    )
    assert accepted.status == "CARD_VALIDATED_CPU_ONLY"


@pytest.mark.parametrize(
    "dtype_name",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "longlong",
        "ulonglong",
    ],
)
def test_every_exact_numpy_integer_scalar_dtype_is_accepted(
    tmp_path: Path, dtype_name: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    scalar_type = np.dtype(dtype_name).type
    evidence = _row_evidence(
        tmp_path,
        camera_to_world,
        correspondences=scalar_type(8),
        inliers=scalar_type(8),
        ratio=scalar_type(1),
        residual_m=scalar_type(0),
        operation_counters=successor.OperationCounters(
            scalar_type(0),
            scalar_type(0),
            scalar_type(0),
            scalar_type(0),
            scalar_type(0),
        ),
    )
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_VALIDATED_CPU_ONLY"


@pytest.mark.parametrize("dtype_name", ["float16", "float32", "float64", "longdouble"])
def test_every_exact_numpy_float_scalar_dtype_is_accepted(
    tmp_path: Path, dtype_name: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    scalar_type = np.dtype(dtype_name).type
    evidence = _row_evidence(
        tmp_path,
        camera_to_world,
        correspondences=8,
        inliers=8,
        ratio=scalar_type(1.0),
        residual_m=scalar_type(0.0),
    )
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_VALIDATED_CPU_ONLY"


@pytest.mark.parametrize("field", ["ratio", "residual", "cylinder"])
def test_extended_precision_values_cannot_collapse_across_fixed_boundaries(
    tmp_path: Path, field: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    if field == "ratio":
        value = np.nextafter(
            np.longdouble(successor.MINIMUM_RANSAC_INLIER_RATIO),
            np.longdouble(-np.inf),
        )
        assert value < np.longdouble(successor.MINIMUM_RANSAC_INLIER_RATIO)
        assert float(value) == successor.MINIMUM_RANSAC_INLIER_RATIO
        evidence = _row_evidence(
            tmp_path,
            camera_to_world,
            ratio=value,
        )
        card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    elif field == "residual":
        value = np.nextafter(
            np.longdouble(successor.MAXIMUM_RESIDUAL_M),
            np.longdouble(np.inf),
        )
        assert value > np.longdouble(successor.MAXIMUM_RESIDUAL_M)
        assert float(value) == successor.MAXIMUM_RESIDUAL_M
        evidence = _row_evidence(
            tmp_path,
            camera_to_world,
            residual_m=value,
        )
        card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    else:
        value = np.nextafter(np.longdouble(0.03), np.longdouble(np.inf))
        assert value > np.longdouble(0.03)
        assert float(value) == 0.03
        mapping = successor.CanaryRowMapping(
            row_index=0,
            session_id=successor.CANARY_SESSION_ID,
            frame_index=successor.CANARY_FRAME_INDEX,
            timestamp_ns=TIMESTAMP_NS,
            cylinder_radius_m=value,
            cylinder_height_m=0.12,
        )
        card = _validate(tmp_path, arrays, camera_to_world, mapping=mapping)
    assert card.status == "CARD_FAILED"
    assert any("losslessly" in reason for reason in card.failure_reasons)


def test_integer_real_above_binary64_exact_range_is_card_failed(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    lossy_integer = (1 << 53) + 1
    arrays["cylinder_radius_m"] = np.asarray(float(lossy_integer), dtype=np.float64)
    mapping = successor.CanaryRowMapping(
        row_index=0,
        session_id=successor.CANARY_SESSION_ID,
        frame_index=successor.CANARY_FRAME_INDEX,
        timestamp_ns=TIMESTAMP_NS,
        cylinder_radius_m=lossy_integer,
        cylinder_height_m=0.12,
    )
    card = _validate(tmp_path, arrays, camera_to_world, mapping=mapping)
    assert card.status == "CARD_FAILED"
    assert any("losslessly" in reason for reason in card.failure_reasons)


@pytest.mark.parametrize("dtype_name", ["longlong", "ulonglong"])
def test_numpy_alternate_integer_scalar_classes_bind_evidence_byte_counts(
    tmp_path: Path, dtype_name: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    assert evidence.raw_mask is not None
    scalar_type = np.dtype(dtype_name).type
    evidence = replace(
        evidence,
        raw_mask=replace(
            evidence.raw_mask,
            source_bytes=scalar_type(evidence.raw_mask.source_bytes),
        ),
    )
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_VALIDATED_CPU_ONLY"


@pytest.mark.parametrize(
    "source_bytes",
    [
        (1 << 53) + 1,
        np.uint64(1 << 63),
        np.int64(-1),
    ],
)
def test_evidence_byte_counts_preserve_exact_integer_boundaries(
    tmp_path: Path, source_bytes: object
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    assert evidence.raw_mask is not None
    evidence = replace(
        evidence,
        raw_mask=replace(evidence.raw_mask, source_bytes=source_bytes),
    )
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False


def test_python_and_numpy_numeric_subclasses_fail_closed_without_conversion(
    tmp_path: Path,
) -> None:
    class ExplodingIntSubclass(int):
        def __int__(self) -> int:
            raise RuntimeError("int conversion must not run")

    class ExplodingFloatSubclass(float):
        def __float__(self) -> float:
            raise RuntimeError("float conversion must not run")

    class ExplodingNumpyIntSubclass(np.int64):
        def __int__(self) -> int:
            raise RuntimeError("numpy int conversion must not run")

    class ExplodingNumpyFloatSubclass(np.float64):
        def __float__(self) -> float:
            raise RuntimeError("numpy float conversion must not run")

    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    values = (
        ("positive_depth_correspondences", ExplodingIntSubclass(20)),
        ("ransac_inlier_ratio", ExplodingFloatSubclass(0.45)),
        ("positive_depth_correspondences", ExplodingNumpyIntSubclass(20)),
        ("ransac_inlier_ratio", ExplodingNumpyFloatSubclass(0.45)),
    )
    for index, (field, value) in enumerate(values):
        rejected = _validate(
            tmp_path,
            arrays,
            camera_to_world,
            evidence=replace(evidence, **{field: value}),
            path_name=f"numeric-subclass-{index}.npz",
        )
        assert rejected.status == "CARD_FAILED"
        assert rejected.formal_consumer_allowed is False
        assert any(
            "exact integer" in reason or "finite real" in reason
            for reason in rejected.failure_reasons
        )


def test_bool_is_not_accepted_as_an_integer_count(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    rejected = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=replace(evidence, positive_depth_correspondences=True),
        path_name="bool-count.npz",
    )
    assert rejected.status == "CARD_FAILED"
    assert any("exact integer" in value for value in rejected.failure_reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mapping_radius", True),
        ("mapping_radius", np.bool_(True)),
        ("mapping_radius", 0.03 + 0.0j),
        ("mapping_radius", object()),
        ("ransac_inlier_ratio", np.bool_(True)),
        ("ransac_inlier_ratio", np.nan),
        ("residual_m", True),
        ("residual_m", np.inf),
        ("residual_m", 0.0 + 0.0j),
    ],
)
def test_bool_complex_object_nan_and_inf_public_reals_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    arrays, camera_to_world = _valid_arrays()
    if field == "mapping_radius":
        mapping = successor.CanaryRowMapping(
            row_index=0,
            session_id=successor.CANARY_SESSION_ID,
            frame_index=successor.CANARY_FRAME_INDEX,
            timestamp_ns=TIMESTAMP_NS,
            cylinder_radius_m=value,
            cylinder_height_m=0.12,
        )
        card = _validate(tmp_path, arrays, camera_to_world, mapping=mapping)
    else:
        evidence = _row_evidence(tmp_path, camera_to_world)
        card = _validate(
            tmp_path,
            arrays,
            camera_to_world,
            evidence=replace(evidence, **{field: value}),
        )
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False


@pytest.mark.parametrize("counter_index", range(5))
def test_every_forbidden_operation_counter_must_be_exact_zero(
    tmp_path: Path, counter_index: int
) -> None:
    arrays, camera_to_world = _valid_arrays()
    values: list[int | bool] = [0, 0, 0, 0, 0]
    values[counter_index] = True if counter_index == 0 else 1
    counters = successor.OperationCounters(*values)
    evidence = _row_evidence(tmp_path, camera_to_world, operation_counters=counters)
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_FAILED"
    assert card.direct_observation_only is False
    assert any("counter" in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    "record_kind",
    [
        "authority",
        "evidence_digest",
        "mapping",
        "direct_evidence",
        "operation_counters",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra", "replace", "malicious"])
def test_public_frozen_dataclass_state_requires_an_exact_safe_snapshot(
    tmp_path: Path, record_kind: str, mutation: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    counters = successor.OperationCounters(0, 0, 0, 0, 0)
    evidence = _row_evidence(
        tmp_path,
        camera_to_world,
        operation_counters=counters,
    )
    mapping = successor.CanaryRowMapping(
        0,
        successor.CANARY_SESSION_ID,
        successor.CANARY_FRAME_INDEX,
        TIMESTAMP_NS,
        0.03,
        0.12,
    )
    role = successor.CANONICAL_FRAME_YAW_ROLE
    if record_kind == "authority":
        target: object = authorities[role]
    elif record_kind == "evidence_digest":
        assert evidence.raw_mask is not None
        target = evidence.raw_mask
    elif record_kind == "mapping":
        target = mapping
    elif record_kind == "direct_evidence":
        target = evidence
    else:
        target = counters
    original_state = object.__getattribute__(target, "__dict__")
    assert type(original_state) is dict
    malicious_items_called = False

    class ExplodingDict(dict[str, object]):
        def items(self) -> object:
            nonlocal malicious_items_called
            malicious_items_called = True
            raise RuntimeError("untrusted dataclass state iteration")

    if mutation == "missing":
        del original_state[next(iter(original_state))]
    elif mutation == "extra":
        original_state["unexpected_public_field"] = 0
    elif mutation == "replace":
        object.__setattr__(target, "__dict__", {"replacement_field": 0})
    else:
        object.__setattr__(target, "__dict__", ExplodingDict(original_state))
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=authorities,
        mapping=mapping,
        evidence=evidence,
        path_name=f"dataclass-{record_kind}-{mutation}.npz",
    )
    assert malicious_items_called is False
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False


def test_cross_session_frame_timestamp_and_cylinder_mappings_fail(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = replace(
        _row_evidence(tmp_path, camera_to_world), session_id="grap_a_cap_050"
    )
    crossed = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert crossed.status == "CARD_FAILED"
    assert any("identity mismatch" in value for value in crossed.failure_reasons)

    mapping = successor.CanaryRowMapping(
        row_index=0,
        session_id=successor.CANARY_SESSION_ID,
        frame_index=successor.CANARY_FRAME_INDEX,
        timestamp_ns=TIMESTAMP_NS,
        cylinder_radius_m=0.031,
        cylinder_height_m=0.12,
    )
    cylinder = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        mapping=mapping,
        path_name="cylinder-mismatch.npz",
    )
    assert cylinder.status == "CARD_FAILED"
    assert any(
        "cylinder_radius_m mismatch" in value for value in cylinder.failure_reasons
    )


def test_row_and_evidence_sequence_iteration_or_access_exceptions_are_card_failed(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    mapping = successor.CanaryRowMapping(
        0,
        successor.CANARY_SESSION_ID,
        successor.CANARY_FRAME_INDEX,
        TIMESTAMP_NS,
        0.03,
        0.12,
    )
    evidence = _row_evidence(tmp_path, camera_to_world)

    class ExplodingIterator(Sequence[object]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            return mapping

        def __iter__(self) -> Iterator[object]:
            raise RuntimeError("iteration exploded")

    class ExplodingGetItem(Sequence[object]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            raise RuntimeError("getitem exploded")

    def run(
        row_values: Sequence[object],
        evidence_values: Sequence[object],
        name: str,
    ) -> successor.ValidationCard:
        path = tmp_path / name
        size, digest = _write_npz(path, arrays)
        return successor.validate_canary_npz(
            path,
            expected_npz_bytes=size,
            expected_npz_sha256=digest,
            camera_to_world=camera_to_world,
            row_mapping=row_values,  # type: ignore[arg-type]
            direct_observation_evidence=evidence_values,  # type: ignore[arg-type]
            authorities=authorities,
        )

    cards = (
        run(ExplodingIterator(), (evidence,), "row-iterator.npz"),
        run((mapping,), ExplodingIterator(), "evidence-iterator.npz"),
        run(ExplodingGetItem(), (evidence,), "row-getitem.npz"),
        run((mapping,), ExplodingGetItem(), "evidence-getitem.npz"),
    )
    assert all(card.status == "CARD_FAILED" for card in cards)
    assert all(card.authorities_valid is False for card in cards)
    assert all(
        any("finite sequence" in reason for reason in card.failure_reasons)
        for card in cards
    )


@pytest.mark.parametrize("bad_sequence", ["row", b"row", bytearray(b"row")])
def test_string_and_byte_sequences_are_rejected_before_snapshot(
    tmp_path: Path, bad_sequence: object
) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "bad-sequence.npz"
    size, digest = _write_npz(path, arrays)
    card = successor.validate_canary_npz(
        path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=bad_sequence,  # type: ignore[arg-type]
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("finite sequence" in reason for reason in card.failure_reasons)


def test_authority_record_bytes_not_caller_booleans_are_the_join(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    record = json.loads(original.authority_path.read_text(encoding="utf-8"))
    record["formal_consumer_allowed"] = True
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("formal_consumer_allowed" in value for value in card.failure_reasons)


def test_authority_json_oversized_integer_is_card_failed(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    payload = original.authority_path.read_bytes()
    needle = f'"bytes":{original.source_bytes}'.encode()
    assert payload.count(needle) == 1
    payload = payload.replace(needle, b'"bytes":' + (b"9" * 5000), 1)
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("strict UTF-8 JSON" in value for value in card.failure_reasons)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_authority_json_nonfinite_constants_are_explicitly_forbidden(
    tmp_path: Path, constant: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    payload = original.authority_path.read_bytes()
    needle = f'"bytes":{original.source_bytes}'.encode()
    assert payload.count(needle) == 1
    payload = payload.replace(needle, f'"bytes":{constant}'.encode(), 1)
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("non-finite JSON constant" in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    ("location", "operation"),
    [
        ("top", "extra"),
        ("top", "missing"),
        ("source", "extra"),
        ("source", "missing"),
    ],
)
def test_authority_json_top_and_source_keysets_are_exact(
    tmp_path: Path, location: str, operation: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    record = json.loads(original.authority_path.read_text(encoding="utf-8"))
    target = record if location == "top" else record["source_artifact"]
    if operation == "extra":
        target["unexpected"] = "not-authorized"
    else:
        del target["schema" if location == "top" else "path"]
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    expected = "schema mismatch" if location == "top" else "source artifact reference"
    assert any(expected in value for value in card.failure_reasons)


@pytest.mark.parametrize("location", ["top", "source"])
def test_authority_json_duplicate_keys_fail_at_parse_time(
    tmp_path: Path, location: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    text = original.authority_path.read_text(encoding="utf-8")
    if location == "top":
        text = text.replace(
            '"schema":',
            '"schema":"duplicate-value","schema":',
            1,
        )
    else:
        marker = '"source_artifact":{'
        assert text.count(marker) == 1
        text = text.replace(marker, marker + '"bytes":1,', 1)
    payload = text.encode("utf-8")
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("duplicate key" in value for value in card.failure_reasons)


def test_authority_record_must_bind_the_actual_source_artifact(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    record = json.loads(original.authority_path.read_text(encoding="utf-8"))
    record["source_artifact"]["sha256"] = "f" * 64
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    mismatched = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert mismatched.status == "CARD_FAILED"
    assert mismatched.authorities_valid is False
    assert any(
        "source artifact reference mismatch" in value
        for value in mismatched.failure_reasons
    )

    authorities = _authority_bundle(tmp_path)
    original = authorities[role]
    original.source_path.write_bytes(b"replaced-source-artifact")
    replaced_source = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        authorities=authorities,
        path_name="replaced-authority-source.npz",
    )
    assert replaced_source.status == "CARD_FAILED"
    assert replaced_source.authorities_valid is False
    assert any("byte count" in value for value in replaced_source.failure_reasons)


def test_same_canonical_path_record_source_swap_is_rejected_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    reused_path = tmp_path / "same-record-and-source.bin"
    later_source = b"post-record-read-replacement-source"
    later_source_sha256 = hashlib.sha256(later_source).hexdigest()
    record = {
        "schema": successor.AUTHORITY_RECORD_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "role": role,
        "authority_id": original.authority_id,
        "decision": original.decision,
        "source_artifact": {
            "path": str(reused_path),
            "bytes": len(later_source),
            "sha256": later_source_sha256,
        },
        "h5b_decision_authorized": True,
        "label_independent": True,
        "canary_only": True,
        "formal_consumer_allowed": False,
    }
    record_payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    reused_path.write_bytes(record_payload)
    authorities[role] = replace(
        original,
        authority_path=reused_path,
        authority_bytes=len(record_payload),
        authority_sha256=hashlib.sha256(record_payload).hexdigest(),
        source_path=reused_path,
        source_bytes=len(later_source),
        source_sha256=later_source_sha256,
    )
    replacement_path = tmp_path / "later-source.bin"
    replacement_path.write_bytes(later_source)
    original_loader = successor._load_authority_record
    swapped = False

    def swap_after_record_read(
        payload: bytes, *, expected_role: str
    ) -> dict[str, object]:
        nonlocal swapped
        parsed = original_loader(payload, expected_role=expected_role)
        if expected_role == role:
            os.replace(replacement_path, reused_path)
            swapped = True
        return parsed

    monkeypatch.setattr(successor, "_load_authority_record", swap_after_record_read)
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert swapped is False
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert card.formal_consumer_allowed is False
    assert any("globally unique canonical" in value for value in card.failure_reasons)


def test_different_path_replace_and_restore_after_first_read_fails_final_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    trigger = authorities[successor.CANONICAL_FRAME_YAW_ROLE].authority_path
    trigger_inode = trigger.stat().st_ino
    victim = authorities[successor.REQUIRED_AUTHORITY_ROLES[1]].source_path
    replacement_path = tmp_path / "unreferenced-replacement.bin"
    replacement_path.write_bytes(b"replacement-bytes-never-consumed")
    backup_path = tmp_path / "victim-backup.bin"
    original_read = successor.os.read
    restored = False

    def replace_and_restore(descriptor: int, count: int) -> bytes:
        nonlocal restored
        payload = original_read(descriptor, count)
        if not restored and os.fstat(descriptor).st_ino == trigger_inode:
            os.replace(victim, backup_path)
            os.replace(replacement_path, victim)
            os.replace(victim, replacement_path)
            os.replace(backup_path, victim)
            restored = True
        return payload

    monkeypatch.setattr(successor.os, "read", replace_and_restore)
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=evidence,
        authorities=authorities,
        path_name="replace-restore.npz",
    )
    assert restored is True
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False
    assert any("identity changed" in value for value in card.failure_reasons)


def test_ancestor_tree_swap_out_and_back_after_read_fails_final_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    stable = tmp_path / "stable-tree"
    stable.mkdir()
    replacement_tree = tmp_path / "replacement-tree"
    replacement_tree.mkdir()
    (replacement_tree / "canary.npz").write_bytes(b"replacement-never-consumed")
    moved_original = tmp_path / "moved-original-tree"
    trigger = authorities[successor.CANONICAL_FRAME_YAW_ROLE].authority_path
    trigger_inode = trigger.stat().st_ino
    original_read = successor.os.read
    restored = False

    def swap_ancestor_and_restore(descriptor: int, count: int) -> bytes:
        nonlocal restored
        payload = original_read(descriptor, count)
        if not restored and os.fstat(descriptor).st_ino == trigger_inode:
            os.replace(stable, moved_original)
            os.replace(replacement_tree, stable)
            os.replace(stable, replacement_tree)
            os.replace(moved_original, stable)
            restored = True
        return payload

    monkeypatch.setattr(successor.os, "read", swap_ancestor_and_restore)
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=evidence,
        authorities=authorities,
        path_name="stable-tree/canary.npz",
    )
    assert restored is True
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False
    assert any("identity changed" in value for value in card.failure_reasons)


def test_double_slash_path_alias_is_not_a_canonical_absolute_path(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    assert evidence.raw_mask is not None
    alias = Path("//" + str(evidence.raw_mask.source_path).lstrip("/"))
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=replace(
            evidence,
            raw_mask=replace(evidence.raw_mask, source_path=alias),
        ),
        path_name="double-slash-alias.npz",
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("canonical absolute" in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    ("role", "bad_decision", "expected_reason"),
    [
        (
            successor.REDETECTION_POLICY_ROLE,
            "SESSION_ANCHOR_THEN_TRACK",
            "direct re-detection",
        ),
        (
            successor.TEMPORAL_POLICY_ROLE,
            "ALLOW_NEIGHBOR_PROPAGATION_AND_SMOOTHING",
            "permits a neighbor",
        ),
    ],
)
def test_authority_cannot_approve_a_forbidden_temporal_policy(
    tmp_path: Path, role: str, bad_decision: str, expected_reason: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    original = authorities[role]
    record = json.loads(original.authority_path.read_text(encoding="utf-8"))
    record["decision"] = bad_decision
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    original.authority_path.write_bytes(payload)
    authorities[role] = replace(
        original,
        decision=bad_decision,
        authority_bytes=len(payload),
        authority_sha256=hashlib.sha256(payload).hexdigest(),
    )
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any(expected_reason in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    "placeholder", ["TODO later", "still_pending", "unknown value"]
)
def test_placeholder_variants_do_not_become_authority(
    tmp_path: Path, placeholder: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    original = authorities[role]
    authorities[role] = replace(original, decision=placeholder)
    card = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False


def test_authority_and_direct_evidence_require_real_distinct_bytes(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    authorities[role] = replace(authorities[role], source_sha256="f" * 64)
    forged = _validate(tmp_path, arrays, camera_to_world, authorities=authorities)
    assert forged.status == "CARD_FAILED"
    assert forged.authorities_valid is False

    evidence = _row_evidence(tmp_path, camera_to_world)
    assert evidence.raw_mask is not None
    same_bytes_tracks = successor.EvidenceDigest(
        kind="DIRECT_CANONICAL_TRACKS",
        source_path=evidence.raw_mask.source_path,
        source_bytes=evidence.raw_mask.source_bytes,
        source_sha256=evidence.raw_mask.source_sha256,
    )
    aliased = replace(evidence, tracks=same_bytes_tracks)
    card = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=aliased,
        path_name="aliased-evidence.npz",
    )
    assert card.status == "CARD_FAILED"
    assert any("distinct files" in value for value in card.failure_reasons)


def test_camera_array_is_exactly_bound_to_camera_evidence_bytes(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    different_camera = camera_to_world.copy()
    different_camera[0, 1, 3] = 0.25
    evidence = _row_evidence(tmp_path, different_camera)
    card = _validate(tmp_path, arrays, camera_to_world, evidence=evidence)
    assert card.status == "CARD_FAILED"
    assert any("bound evidence bytes" in value for value in card.failure_reasons)


def test_camera_snapshot_rejects_ragged_object_and_complex_inputs_as_cards(
    tmp_path: Path,
) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    authorities = _authority_bundle(tmp_path)
    mapping = successor.CanaryRowMapping(
        0,
        successor.CANARY_SESSION_ID,
        successor.CANARY_FRAME_INDEX,
        TIMESTAMP_NS,
        0.03,
        0.12,
    )

    def run(value: object, name: str) -> successor.ValidationCard:
        path = tmp_path / name
        size, digest = _write_npz(path, arrays)
        return successor.validate_canary_npz(
            path,
            expected_npz_bytes=size,
            expected_npz_sha256=digest,
            camera_to_world=value,  # type: ignore[arg-type]
            row_mapping=(mapping,),
            direct_observation_evidence=(evidence,),
            authorities=authorities,
        )

    cards = (
        run([[], [1]], "ragged-camera.npz"),
        run(np.empty((1, 4, 4), dtype=object), "object-camera.npz"),
        run(np.eye(4, dtype=np.complex128)[None], "complex-camera.npz"),
    )
    assert all(card.status == "CARD_FAILED" for card in cards)
    assert all(card.authorities_valid is False for card in cards)
    assert all(card.formal_consumer_allowed is False for card in cards)


@pytest.mark.parametrize(
    "ordinary_error",
    [
        TypeError("array type"),
        ValueError("array value"),
        OverflowError("array overflow"),
        RuntimeError("array runtime"),
    ],
)
def test_custom_array_conversion_errors_are_card_failed(
    tmp_path: Path, ordinary_error: Exception
) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "exploding-camera.npz"
    size, digest = _write_npz(path, arrays)

    class ExplodingArray:
        def __array__(self) -> np.ndarray:
            raise ordinary_error

    card = successor.validate_canary_npz(
        path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=ExplodingArray(),  # type: ignore[arg-type]
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("cannot be snapshotted" in value for value in card.failure_reasons)


def test_camera_snapshot_does_not_swallow_memory_error(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "memory-camera.npz"
    size, digest = _write_npz(path, arrays)

    class MemoryFailingArray:
        def __array__(self) -> np.ndarray:
            raise MemoryError("intentional memory boundary")

    with pytest.raises(MemoryError, match="intentional memory boundary"):
        successor.validate_canary_npz(
            path,
            expected_npz_bytes=size,
            expected_npz_sha256=digest,
            camera_to_world=MemoryFailingArray(),  # type: ignore[arg-type]
            row_mapping=(
                successor.CanaryRowMapping(
                    0,
                    successor.CANARY_SESSION_ID,
                    successor.CANARY_FRAME_INDEX,
                    TIMESTAMP_NS,
                    0.03,
                    0.12,
                ),
            ),
            direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
            authorities=_authority_bundle(tmp_path),
        )


def test_npz_and_authority_hardlinks_are_rejected(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "hardlinked.npz"
    size, digest = _write_npz(path, arrays)
    os.link(path, tmp_path / "second-name.npz")
    card = successor.validate_canary_npz(
        path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert any("nlink=1" in value for value in card.failure_reasons)

    clean_path = tmp_path / "clean.npz"
    clean_size, clean_digest = _write_npz(clean_path, arrays)
    authorities = _authority_bundle(tmp_path)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    os.link(authorities[role].source_path, tmp_path / "authority-hardlink.json")
    authority_card = successor.validate_canary_npz(
        clean_path,
        expected_npz_bytes=clean_size,
        expected_npz_sha256=clean_digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=authorities,
    )
    assert authority_card.status == "CARD_FAILED"
    assert any("nlink=1" in value for value in authority_card.failure_reasons)


def test_direct_evidence_symlinks_and_hardlinks_are_rejected(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    evidence = _row_evidence(tmp_path, camera_to_world)
    assert evidence.raw_mask is not None
    link = tmp_path / "raw-mask-link.bin"
    link.symlink_to(evidence.raw_mask.source_path)
    linked_ref = replace(evidence.raw_mask, source_path=link)
    linked = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=replace(evidence, raw_mask=linked_ref),
        path_name="evidence-symlink.npz",
    )
    assert linked.status == "CARD_FAILED"
    assert any("no-follow" in value for value in linked.failure_reasons)

    evidence = _row_evidence(tmp_path, camera_to_world)
    assert evidence.raw_mask is not None
    os.link(evidence.raw_mask.source_path, tmp_path / "raw-mask-hardlink.bin")
    hardlinked = _validate(
        tmp_path,
        arrays,
        camera_to_world,
        evidence=evidence,
        path_name="evidence-hardlink.npz",
    )
    assert hardlinked.status == "CARD_FAILED"
    assert any("nlink=1" in value for value in hardlinked.failure_reasons)


def test_nonregular_leaf_is_rejected_without_blocking(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    fifo = tmp_path / "canary.fifo"
    os.mkfifo(fifo)
    card = successor.validate_canary_npz(
        fifo,
        expected_npz_bytes=1,
        expected_npz_sha256="0" * 64,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert any("not a regular file" in value for value in card.failure_reasons)


def test_relative_paths_are_rejected_before_cwd_can_be_a_trust_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "relative.npz"
    size, digest = _write_npz(path, arrays)
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    monkeypatch.chdir(tmp_path)
    card = successor.validate_canary_npz(
        Path("relative.npz"),
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(evidence,),
        authorities=authorities,
    )
    assert card.status == "CARD_FAILED"
    assert any("must be absolute" in value for value in card.failure_reasons)


@pytest.mark.parametrize(
    "location",
    [
        "authorities",
        "row_mapping",
        "evidence_rows",
        "npz_path",
        "authority_record_path",
        "authority_source_path",
        "evidence_source_path",
    ],
)
def test_public_abc_and_path_type_checks_control_exploding_class_property(
    tmp_path: Path, location: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    npz_path = tmp_path / "class-property.npz"
    size, digest = _write_npz(npz_path, arrays)
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    mapping = successor.CanaryRowMapping(
        0,
        successor.CANARY_SESSION_ID,
        successor.CANARY_FRAME_INDEX,
        TIMESTAMP_NS,
        0.03,
        0.12,
    )
    class_accessed = False

    class ExplodingClass:
        @property
        def __class__(self) -> type:
            nonlocal class_accessed
            class_accessed = True
            raise RuntimeError("class lookup exploded")

    bad_value = ExplodingClass()
    public_npz_path: object = npz_path
    public_authorities: object = authorities
    public_row_mapping: object = (mapping,)
    public_evidence_rows: object = (evidence,)
    role = successor.CANONICAL_FRAME_YAW_ROLE
    if location == "authorities":
        public_authorities = bad_value
    elif location == "row_mapping":
        public_row_mapping = bad_value
    elif location == "evidence_rows":
        public_evidence_rows = bad_value
    elif location == "npz_path":
        public_npz_path = bad_value
    elif location == "authority_record_path":
        authorities[role] = replace(authorities[role], authority_path=bad_value)
    elif location == "authority_source_path":
        authorities[role] = replace(authorities[role], source_path=bad_value)
    else:
        assert evidence.raw_mask is not None
        evidence = replace(
            evidence,
            raw_mask=replace(evidence.raw_mask, source_path=bad_value),
        )
        public_evidence_rows = (evidence,)
    card = successor.validate_canary_npz(
        public_npz_path,  # type: ignore[arg-type]
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=public_row_mapping,  # type: ignore[arg-type]
        direct_observation_evidence=public_evidence_rows,  # type: ignore[arg-type]
        authorities=public_authorities,  # type: ignore[arg-type]
    )
    assert card.status == "CARD_FAILED"
    assert card.formal_consumer_allowed is False
    assert class_accessed is (
        location in {"authorities", "row_mapping", "evidence_rows"}
    )


@pytest.mark.parametrize("location", ["authorities", "row_mapping", "evidence_rows"])
def test_public_abc_eligibility_does_not_swallow_memory_error(
    tmp_path: Path, location: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    npz_path = tmp_path / f"memory-class-{location}.npz"
    size, digest = _write_npz(npz_path, arrays)
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    mapping = successor.CanaryRowMapping(
        0,
        successor.CANARY_SESSION_ID,
        successor.CANARY_FRAME_INDEX,
        TIMESTAMP_NS,
        0.03,
        0.12,
    )

    class MemoryClass:
        @property
        def __class__(self) -> type:
            raise MemoryError("intentional ABC memory boundary")

    bad_value = MemoryClass()
    public_authorities: object = bad_value if location == "authorities" else authorities
    public_row_mapping: object = bad_value if location == "row_mapping" else (mapping,)
    public_evidence_rows: object = (
        bad_value if location == "evidence_rows" else (evidence,)
    )
    with pytest.raises(MemoryError, match="intentional ABC memory boundary"):
        successor.validate_canary_npz(
            npz_path,
            expected_npz_bytes=size,
            expected_npz_sha256=digest,
            camera_to_world=camera_to_world,
            row_mapping=public_row_mapping,  # type: ignore[arg-type]
            direct_observation_evidence=public_evidence_rows,  # type: ignore[arg-type]
            authorities=public_authorities,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "location",
    [
        "npz",
        "authority_record",
        "authority_source",
        "raw_mask",
        "tracks",
        "positive_depth",
        "camera_to_world",
    ],
)
def test_all_public_path_classes_reject_path_subclasses_without_calling_fspath(
    tmp_path: Path, location: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    npz_path = tmp_path / "exact-path.npz"
    size, digest = _write_npz(npz_path, arrays)
    authorities = _authority_bundle(tmp_path)
    evidence = _row_evidence(tmp_path, camera_to_world)
    fspath_called = False
    concrete_path_type = type(Path(os.path.sep))

    class ExplodingPath(concrete_path_type):  # type: ignore[misc, valid-type]
        def __fspath__(self) -> str:
            nonlocal fspath_called
            fspath_called = True
            raise RuntimeError("untrusted fspath must not execute")

    if location == "npz":
        public_npz_path = ExplodingPath(str(npz_path))
    else:
        public_npz_path = npz_path
    role = successor.CANONICAL_FRAME_YAW_ROLE
    if location == "authority_record":
        authorities[role] = replace(
            authorities[role],
            authority_path=ExplodingPath(str(authorities[role].authority_path)),
        )
    elif location == "authority_source":
        authorities[role] = replace(
            authorities[role],
            source_path=ExplodingPath(str(authorities[role].source_path)),
        )
    elif location in {"raw_mask", "tracks", "positive_depth", "camera_to_world"}:
        record = getattr(evidence, location)
        assert record is not None
        evidence = replace(
            evidence,
            **{
                location: replace(
                    record, source_path=ExplodingPath(str(record.source_path))
                )
            },
        )
    card = successor.validate_canary_npz(
        public_npz_path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(evidence,),
        authorities=authorities,
    )
    assert fspath_called is False
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False
    assert any("exact concrete Path" in value for value in card.failure_reasons)


@pytest.mark.parametrize("kind", ["string", "pathlike"])
def test_string_and_generic_pathlike_inputs_are_not_public_paths(
    tmp_path: Path, kind: str
) -> None:
    arrays, camera_to_world = _valid_arrays()
    npz_path = tmp_path / "non-path.npz"
    size, digest = _write_npz(npz_path, arrays)

    class GenericPathLike:
        def __fspath__(self) -> str:
            return str(npz_path)

    public_path: object = str(npz_path) if kind == "string" else GenericPathLike()
    card = successor.validate_canary_npz(
        public_path,  # type: ignore[arg-type]
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert card.authorities_valid is False


def test_symlinked_ancestor_is_rejected(tmp_path: Path) -> None:
    arrays, camera_to_world = _valid_arrays()
    real = tmp_path / "real"
    real.mkdir()
    path = real / "canary.npz"
    size, digest = _write_npz(path, arrays)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    card = successor.validate_canary_npz(
        alias / "canary.npz",
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert card.status == "CARD_FAILED"
    assert any("ancestor no-follow" in value for value in card.failure_reasons)


def test_leaf_replacement_during_same_fd_read_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    path = tmp_path / "replace-target.npz"
    size, digest = _write_npz(path, arrays)
    replacement_arrays, _ = _valid_arrays()
    replacement_arrays["confidence"][0] = 0.8
    replacement = tmp_path / "replacement.npz"
    _write_npz(replacement, replacement_arrays)
    target_inode = path.stat().st_ino
    original_read = successor.os.read
    replaced = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, count)
        if not replaced and os.fstat(descriptor).st_ino == target_inode:
            os.replace(replacement, path)
            replaced = True
        return payload

    monkeypatch.setattr(successor.os, "read", replacing_read)
    card = successor.validate_canary_npz(
        path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert replaced is True
    assert card.status == "CARD_FAILED"
    assert any("identity changed" in value for value in card.failure_reasons)


def test_ancestor_replacement_during_held_dirfd_read_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, camera_to_world = _valid_arrays()
    stable = tmp_path / "stable"
    stable.mkdir()
    path = stable / "canary.npz"
    size, digest = _write_npz(path, arrays)
    replacement = tmp_path / "replacement-tree"
    replacement.mkdir()
    replacement_arrays, _ = _valid_arrays()
    replacement_arrays["confidence"][0] = 0.7
    _write_npz(replacement / "canary.npz", replacement_arrays)
    target_inode = path.stat().st_ino
    original_read = successor.os.read
    replaced = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, count)
        if not replaced and os.fstat(descriptor).st_ino == target_inode:
            stable.rename(tmp_path / "moved-original")
            replacement.rename(stable)
            replaced = True
        return payload

    monkeypatch.setattr(successor.os, "read", replacing_read)
    card = successor.validate_canary_npz(
        path,
        expected_npz_bytes=size,
        expected_npz_sha256=digest,
        camera_to_world=camera_to_world,
        row_mapping=(
            successor.CanaryRowMapping(
                0,
                successor.CANARY_SESSION_ID,
                successor.CANARY_FRAME_INDEX,
                TIMESTAMP_NS,
                0.03,
                0.12,
            ),
        ),
        direct_observation_evidence=(_row_evidence(tmp_path, camera_to_world),),
        authorities=_authority_bundle(tmp_path),
    )
    assert replaced is True
    assert card.status == "CARD_FAILED"
    assert any("ancestor identity changed" in value for value in card.failure_reasons)


def test_module_has_no_gpu_or_model_framework_imports() -> None:
    source = Path(successor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports.isdisjoint({"torch", "tensorflow", "jax", "transformers", "cv2"})
