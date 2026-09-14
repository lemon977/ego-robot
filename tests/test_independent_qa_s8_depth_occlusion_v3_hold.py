from __future__ import annotations

import inspect

from tools import run_depth_occlusion_v3_s8_t1 as runner


def _bound(path: str) -> dict[str, object]:
    return {"path": path, "bytes": 1, "sha256": "a" * 64}


def _manifest(*, session_id: str, object6d_path: str) -> dict[str, object]:
    return {
        "schema_version": runner.INPUT_MANIFEST_SCHEMA,
        "session_id": session_id,
        "frame_name": "00240",
        "frame_index": 240,
        "frame_bundle": _bound("_run/frame_bundle.npz"),
        "object6d": _bound(object6d_path),
        "source_lineage": {
            "clean": _bound("_run/unrelated-clean-record.bin"),
            "object_texture": _bound("_run/unrelated-texture-record.bin"),
            "robot_render": _bound("_run/unrelated-render-record.bin"),
        },
        "object_texture_source": "raw_observation",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "mount_provenance": runner.MOUNT_PROVENANCE,
        "contact_infeasible": runner.CONTACT_INFEASIBLE,
        "bundle_array_sha256": {
            "clean_rgb": "1" * 64,
            "object_rgb_2x": "2" * 64,
            "robot_rgb_2x": "3" * 64,
            "robot_range_m_2x": "4" * 64,
            "robot_alpha_2x": "5" * 64,
            "camera_intrinsics": "6" * 64,
        },
    }


def test_hold_replay_cross_session_object6d_alias_is_admitted() -> None:
    """A 002 output can currently name the frozen 004 Object6D as its input."""

    manifest = _manifest(
        session_id="grap_a_cap_002",
        object6d_path=(
            "data/benchmarks/hand/production_runs/final_v3_grap_a_cap_0812/"
            "grap_a_cap_004/10_object6d_v2/object_6dof_v2.npz"
        ),
    )
    runner.validate_input_manifest(manifest)


def test_hold_replay_lineage_records_do_not_bind_bundle_arrays() -> None:
    """Changing every declared pixel-array digest leaves lineage records untouched."""

    first = _manifest(
        session_id="grap_a_cap_004",
        object6d_path="_run/object6d.npz",
    )
    second = _manifest(
        session_id="grap_a_cap_004",
        object6d_path="_run/object6d.npz",
    )
    second["bundle_array_sha256"] = {
        name: "f" * 64 for name in first["bundle_array_sha256"]
    }
    assert first["source_lineage"] == second["source_lineage"]
    runner.validate_input_manifest(first)
    runner.validate_input_manifest(second)


def test_hold_replay_runtime_semantics_are_not_bound_to_freeze_bytes() -> None:
    """execute() consumes normally imported code and never consumes CPU_FREEZE_FINAL."""

    execute_source = inspect.getsource(runner.execute)
    require_source = inspect.getsource(runner._require_shared_io)
    assert "CPU_FREEZE_FINAL" not in execute_source
    assert "CORE" not in execute_source
    assert "read_bytes_nofollow" in require_source
    assert "exec(compile(" not in require_source
    assert runner.compose_s8_frame_bundle.__module__ == "pipeline.depth_occlusion_v3"
