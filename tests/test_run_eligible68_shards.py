from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT = Path("/mnt/workspace/code/chaoyang")
SOURCE = PROJECT / "tools/run_eligible68_shards.py"
PLAN = (
    PROJECT / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/checkpoints/"
    "ELIGIBLE68_BATCH_SHARD_PLAN_T0_V1.json"
)
SMOKE = (
    PROJECT / "archive/legacy_runs/unclassified/cpu_b4_eligible68_shard_smoke_20260829_v2/"
    "eligible68-train60-shard-01"
)


def _module():
    spec = importlib.util.spec_from_file_location("run_eligible68_shards", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_plan_population_and_first_shard() -> None:
    module = _module()
    plan, path, digest = module._load_plan(PLAN)
    shard = module._select_shard(plan, "eligible68-train60-shard-01")
    assert path == PLAN
    assert digest == module.PLAN_SHA256
    assert shard["frame_count"] == 2075
    assert [row["session_id"] for row in shard["sessions"]] == [
        "grap_a_cap_004",
        "grap_a_cap_005",
        "grap_a_cap_010",
        "grap_a_cap_021",
        "grap_a_cap_023",
    ]


@pytest.mark.parametrize(
    "path",
    [Path("/usr/bin/false"), Path("/usr/bin/true"), Path("/bin/sh"), Path("/bin/bash")],
)
def test_empty_or_shell_payload_is_rejected(path: Path) -> None:
    module = _module()
    with pytest.raises(module.ShardRunError, match="producer"):
        module._validate_executable(path)


def _write_contract_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """#!/usr/bin/python3
ELIGIBLE68_PRODUCER_ADAPTER_CONTRACT = "eligible68-mask-producer-adapter-v1"

def write_real_session_masks(session_id, expected_frames, input_metadata_sha256,
                             output_left_dir, output_right_dir, output_manifest):
    raise RuntimeError("test fixture is validated but never executed")

def produce_eligible68_session(session_id, expected_frames, input_metadata_sha256,
                               output_left_dir, output_right_dir, output_manifest):
    return write_real_session_masks(session_id, expected_frames,
                                    input_metadata_sha256, output_left_dir,
                                    output_right_dir, output_manifest)
""",
        encoding="utf-8",
    )
    adapter.chmod(0o750)
    return adapter


def test_production_adapter_is_bound_to_exact_executable_bytes(tmp_path: Path) -> None:
    module = _module()
    adapter = _write_contract_adapter(tmp_path)
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    executable, reference = module._bind_producer(adapter, digest)
    assert executable == adapter
    assert reference["sha256"] == digest

    adapter.write_bytes(adapter.read_bytes() + b"# drift\n")
    with pytest.raises(module.ShardRunError, match="SHA-256 mismatch"):
        module._bind_producer(adapter, digest)


def test_outer_adapter_rename_restore_executes_the_held_verified_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    attempt = module._ensure_directory(base / "attempt", contract)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "#!/usr/bin/env python3\nprint('ORIGINAL_HELD_ADAPTER')\n",
        encoding="utf-8",
    )
    adapter.chmod(0o750)
    malicious = tmp_path / "malicious.py"
    malicious.write_text(
        "#!/usr/bin/env python3\nprint('MALICIOUS_PATH_ADAPTER')\n",
        encoding="utf-8",
    )
    malicious.chmod(0o750)
    producer_ref = module._file_ref(adapter, "held adapter fixture")
    real_run = subprocess.run
    parked = tmp_path / "parked.py"

    def rename_attack(command, **kwargs):
        adapter.rename(parked)
        malicious.rename(adapter)
        try:
            return real_run(command, **kwargs)
        finally:
            adapter.rename(malicious)
            parked.rename(adapter)

    monkeypatch.setattr(module.subprocess, "run", rename_attack)
    module._run_producer(
        adapter,
        producer_ref,
        {"path": "/admission", "bytes": 1, "sha256": "a" * 64},
        [],
        attempt,
        {
            "session_id": "grap_a_cap_004",
            "frame_count": 1,
            "input_metadata_bundle_sha256": "b" * 64,
        },
        contract,
    )
    log = (attempt / "producer.log").read_text(encoding="utf-8")
    assert "ORIGINAL_HELD_ADAPTER" in log
    assert "MALICIOUS_PATH_ADAPTER" not in log


@pytest.mark.parametrize(
    "payload",
    [
        b":\n",
        b"#!/bin/sh\nexit 0\n",
        b"#!/usr/bin/python3\nraise SystemExit(0)\n",
        b"#!/usr/bin/python3\nimport sys\nsys.exit(0)\n",
    ],
)
def test_renamed_text_stub_payload_is_rejected(tmp_path: Path, payload: bytes) -> None:
    module = _module()
    adapter = tmp_path / "renamed_real_adapter"
    adapter.write_bytes(payload)
    adapter.chmod(0o750)
    with pytest.raises(module.ShardRunError, match="producer|stub|shebang"):
        module._bind_producer(adapter, hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("source", [Path("/bin/true"), Path("/usr/bin/false")])
def test_renamed_binary_stub_payload_is_rejected(tmp_path: Path, source: Path) -> None:
    module = _module()
    adapter = tmp_path / "renamed_real_adapter"
    shutil.copyfile(source, adapter)
    adapter.chmod(0o750)
    payload = adapter.read_bytes()
    with pytest.raises(module.ShardRunError, match="Python adapter"):
        module._bind_producer(adapter, hashlib.sha256(payload).hexdigest())


def _live_chain_fixture(module, root: Path) -> dict:
    declared = root / "declared_live.py"
    executed = root / "live_run/EXECUTED_LIVE_PRODUCER.py"
    executed.parent.mkdir(parents=True, exist_ok=True)
    payload = b"#!/usr/bin/env python3\nprint('real fixture')\n"
    declared.write_bytes(payload)
    executed.write_bytes(payload)
    live_manifest = root / "live_run/MANIFEST.json"
    origin = root / "origin_context"
    bound = root / "live_run/BOUND_CONTEXT"
    origin.mkdir()
    bound.mkdir()
    names = {
        "preflight": "PREFLIGHT.json",
        "raw_input_manifest": "RAW_INPUT_MANIFEST.json",
        "authority_covariates": "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json",
    }
    origin_refs = {}
    bound_refs = {}
    for key, name in names.items():
        if key in {"preflight", "authority_covariates", "raw_input_manifest"}:
            continue
        artifact_payload = (key + "\n").encode()
        (origin / name).write_bytes(artifact_payload)
        (bound / name).write_bytes(artifact_payload)
        origin_refs[key] = module._file_ref(origin / name, f"origin {key}")
        bound_refs[key] = module._file_ref(bound / name, f"bound {key}")
    origin_raw_png = origin / "raw_frame_00000.png"
    bound_raw_png = bound / "RAW_FRAMES/s/frame_00000.png"
    bound_raw_png.parent.mkdir(parents=True)
    raw_payload = (
        SMOKE / "sessions/grap_a_cap_004/attempt_001/masks/left/frame_00000.png"
    ).read_bytes()
    origin_raw_png.write_bytes(raw_payload)
    bound_raw_png.write_bytes(raw_payload)
    origin_raw_ref = module._file_ref(origin_raw_png, "origin RAW frame")
    bound_raw_ref = module._file_ref(bound_raw_png, "bound RAW frame")
    origin_raw_value = {
        "schema_version": "eligible68-dual-raw-input-context-v1",
        "detection_mode": "dual",
        "session_count": 1,
        "frame_count": 1,
        "raw_png_full_decode_count": 1,
        "sessions": {
            "s": {
                "frame_count": 1,
                "frames": [{"frame_index": 0, **origin_raw_ref}],
            }
        },
    }
    bound_raw_value = copy.deepcopy(origin_raw_value)
    bound_raw_value["sessions"]["s"]["frames"][0].update(
        {**bound_raw_ref, "origin_ref": origin_raw_ref}
    )
    origin_raw_manifest = origin / "RAW_INPUT_MANIFEST.json"
    bound_raw_manifest = bound / "RAW_INPUT_MANIFEST.json"
    origin_raw_manifest.write_text(json.dumps(origin_raw_value), encoding="utf-8")
    bound_raw_manifest.write_text(json.dumps(bound_raw_value), encoding="utf-8")
    origin_refs["raw_input_manifest"] = module._file_ref(
        origin_raw_manifest, "origin raw_input_manifest"
    )
    bound_refs["raw_input_manifest"] = module._file_ref(
        bound_raw_manifest, "bound raw_input_manifest"
    )
    for role in sorted(module.GEOMETRY_ROLES):
        origin_path = origin / f"{role}.bin"
        bound_path = bound / "geometry" / f"{role}.bin"
        bound_path.parent.mkdir(exist_ok=True)
        payload = f"geometry-{role}\n".encode()
        origin_path.write_bytes(payload)
        bound_path.write_bytes(payload)
        origin_refs[f"geometry_{role}"] = module._file_ref(
            origin_path, f"origin geometry {role}"
        )
        bound_refs[f"geometry_{role}"] = module._file_ref(
            bound_path, f"bound geometry {role}"
        )
    origin_refs["formal_object6d"] = origin_refs["geometry_object6d"]
    bound_refs["formal_object6d"] = bound_refs["geometry_object6d"]
    origin_authority = origin / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    bound_authority = bound / "AUTHORITY_PHYSICAL_SIDE_COVARIATES.json"
    origin_authority.write_text(
        json.dumps(
            {
                "sessions": {
                    "s": {
                        "geometry": {
                            role: origin_refs[f"geometry_{role}"]
                            for role in module.GEOMETRY_ROLES
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    bound_authority.write_text(
        json.dumps(
            {
                "sessions": {
                    "s": {
                        "geometry": {
                            role: bound_refs[f"geometry_{role}"]
                            for role in module.GEOMETRY_ROLES
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    origin_refs["authority_covariates"] = module._file_ref(
        origin_authority, "origin authority_covariates"
    )
    bound_refs["authority_covariates"] = module._file_ref(
        bound_authority, "bound authority_covariates"
    )
    origin_preflight = origin / "PREFLIGHT.json"
    bound_preflight = bound / "PREFLIGHT.json"
    origin_preflight.write_text(
        json.dumps({"artifacts": {"fixture": True}}), encoding="utf-8"
    )
    bound_preflight.write_text(
        json.dumps(
            {
                "artifacts": {
                    "raw_input_manifest": bound_refs["raw_input_manifest"],
                    "authority_covariates": bound_refs["authority_covariates"],
                }
            }
        ),
        encoding="utf-8",
    )
    origin_refs["preflight"] = module._file_ref(origin_preflight, "origin preflight")
    bound_refs["preflight"] = module._file_ref(bound_preflight, "bound preflight")

    dependency_root = root / "live_run/DEPENDENCY_BUNDLE"
    dependency_root.mkdir()
    dependency_sources = {}
    digest_rows = {}
    for name in sorted(module.RUNTIME_DEPENDENCY_NAMES):
        origin_path = root / "dependency_origin" / f"{name}.py"
        bound_path = dependency_root / f"{name}.py"
        origin_path.parent.mkdir(exist_ok=True)
        payload = f"# {name}\n".encode()
        origin_path.write_bytes(payload)
        bound_path.write_bytes(payload)
        origin_ref = module._file_ref(origin_path, f"dependency origin {name}")
        bound_ref = module._file_ref(bound_path, f"dependency bound {name}")
        dependency_sources[name] = {
            "relative_path": f"{name}.py",
            "origin": origin_ref,
            "bound": bound_ref,
        }
        digest_rows[name] = {
            "origin_path": origin_ref["path"],
            "bytes": origin_ref["bytes"],
            "sha256": origin_ref["sha256"],
        }
    dependency_manifest = dependency_root / "MANIFEST.json"
    dependency_manifest.write_text("{}\n", encoding="utf-8")
    dependency_bundle = {
        "schema_version": "eligible68-runtime-dependency-bundle-v1",
        "status": "ARTIFACT_EXISTS",
        "source_bundle_sha256": hashlib.sha256(
            json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "sources": dependency_sources,
        "manifest_ref": module._file_ref(dependency_manifest, "dependency manifest"),
        "consumed_sources": sorted(module.RUNTIME_DEPENDENCY_NAMES),
    }
    official = {
        "root": "/official",
        "commit": "1" * 40,
        "tracked_file_count": 1,
        "tracked_tree_sha256": "2" * 64,
        "git_status_clean": True,
        "tracked_files_regular_unaliased": True,
        "snapshot_root": "/official-snapshot",
        "snapshot_file_count": 2,
        "snapshot_source_bundle_sha256": "4" * 64,
        "snapshot_source": "PINNED_GIT_COMMIT_BLOBS",
    }
    live_manifest.write_text(
        json.dumps(
            {
                "governance_bypassed": [],
                "executed_dependency_bundle": dependency_bundle,
                "dependency_bundle_source_sha256": dependency_bundle[
                    "source_bundle_sha256"
                ],
                "formal_allowed_evidence_root": str(live_manifest.parent),
                "bound_context_document_refs": {
                    key: bound_refs[key]
                    for key in (
                        "preflight",
                        "raw_input_manifest",
                        "authority_covariates",
                    )
                },
                "held_geometry_context": {
                    "schema_version": "eligible68-held-six-role-geometry-v1",
                    "capture_count": len(module.GEOMETRY_ROLES),
                    "session_roles": {
                        "s": {
                            role: bound_refs[f"geometry_{role}"]
                            for role in module.GEOMETRY_ROLES
                        }
                    },
                    "capture_sha256": "5" * 64,
                    "consumers": [
                        "context_helper",
                        "task29_runtime_geometry",
                        "d4_provider",
                        "manifest",
                    ],
                },
                "retained_run_root": {
                    "path": str(live_manifest.parent),
                    "device": live_manifest.parent.stat().st_dev,
                    "inode": live_manifest.parent.stat().st_ino,
                    "method": "INHERITED_DIRECTORY_FD_OPENAT_MKDIRAT_O_EXCL",
                },
                "official_code_tree_before": official,
                "official_code_tree_after": official,
                "official_modules_before": {
                    "module_count": 2,
                    "loaded_source_sha256": "3" * 64,
                    "all_sources_under_official_root": True,
                    "all_code_objects_held_payload_bound": True,
                    "source_bundle_sha256": "4" * 64,
                },
                "official_modules_after": {
                    "module_count": 2,
                    "loaded_source_sha256": "3" * 64,
                    "all_sources_under_official_root": True,
                    "all_code_objects_held_payload_bound": True,
                    "source_bundle_sha256": "4" * 64,
                },
                "pinned_build": {
                    "checkpoint_same_held_fd_consumed": True,
                    "official_source_execution": {
                        "method": "HELD_PAYLOAD_META_PATH_LOADER",
                        "source_bundle_sha256": "4" * 64,
                        "snapshot_root": "/official-snapshot",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    executed_ref = module._file_ref(executed, "executed live fixture")
    return {
        "declared_source": module._file_ref(declared, "declared live fixture"),
        "executed_source": executed_ref,
        "command": [
            sys.executable,
            "/proc/self/fd/91",
            "--run-root",
            str(live_manifest.parent),
            "--run-root-fd",
            "92",
            "--expected-run-root-device",
            str(live_manifest.parent.stat().st_dev),
            "--expected-run-root-inode",
            str(live_manifest.parent.stat().st_ino),
            "--expected-executed-source-sha256",
            executed_ref["sha256"],
            "--expected-executed-source-path",
            executed_ref["path"],
            "--dependency-bundle-manifest",
            dependency_bundle["manifest_ref"]["path"],
            "--expected-dependency-bundle-sha256",
            dependency_bundle["manifest_ref"]["sha256"],
            "--expected-context-preflight-sha256",
            bound_refs["preflight"]["sha256"],
        ],
        "manifest": module._file_ref(live_manifest, "live manifest fixture"),
        "capabilities": {
            "schema_version": "d1-d4-live-producer-capabilities-v1",
            "strict_dual": True,
            "generic_d4": True,
            "d3_candidate_provenance": True,
            "wearable_w1_w4_raw_provider": True,
            "final_h_left_right_pngs": True,
        },
        "dependency_bundle": dependency_bundle,
        "context": {
            "origin_root": str(origin),
            "origin_refs": origin_refs,
            "bound_root": str(bound),
            "bound_refs": bound_refs,
        },
    }


def _producer_admission_fixture(module, tmp_path: Path) -> tuple[Path, dict]:
    adapter = _write_contract_adapter(tmp_path)
    producer_ref = module._file_ref(adapter, "test producer")
    canary_root = tmp_path / "canary"
    left = canary_root / "masks/left/frame_00000.png"
    right = canary_root / "masks/right/frame_00000.png"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(
        (
            SMOKE / "sessions/grap_a_cap_004/attempt_001/masks/left/frame_00000.png"
        ).read_bytes()
    )
    right.write_bytes(
        (
            SMOKE / "sessions/grap_a_cap_004/attempt_001/masks/right/frame_00000.png"
        ).read_bytes()
    )
    left_ref = module._file_ref(left, "test left canary")
    right_ref = module._file_ref(right, "test right canary")
    plan, plan_path, plan_sha = module._load_plan(PLAN)
    _, constituent_ref = module._load_constituent_manifest(plan)
    plan_ref = {
        "path": str(plan_path),
        "bytes": plan_path.stat().st_size,
        "sha256": plan_sha,
    }
    metadata_sha = module._select_shard(plan, "eligible68-train60-shard-01")[
        "sessions"
    ][0]["input_metadata_bundle_sha256"]
    combined = module._combined_digest(
        [
            ("left", 0, left_ref["bytes"], left_ref["sha256"]),
            ("right", 0, right_ref["bytes"], right_ref["sha256"]),
        ]
    )
    producer_manifest = canary_root / "PRODUCER_MANIFEST.json"
    live_chain = _live_chain_fixture(module, canary_root)
    producer_args = [
        "--expected-dependency-source-sha256",
        live_chain["dependency_bundle"]["source_bundle_sha256"],
    ]
    command = [
        sys.executable,
        "/proc/self/fd/90",
        *producer_args,
        "--session-id",
        "grap_a_cap_004",
        "--expected-frames",
        "1",
        "--input-metadata-sha256",
        metadata_sha,
        "--output-left-dir",
        str(left.parent),
        "--output-right-dir",
        str(right.parent),
        "--output-manifest",
        str(producer_manifest),
    ]
    producer_manifest.write_text(
        json.dumps(
            {
                "schema_version": module.PRODUCER_MANIFEST_SCHEMA,
                "status": "ARTIFACT_EXISTS",
                "completion_mode": "ARTIFACT_EXISTS",
                "session_id": "grap_a_cap_004",
                "frame_count": 1,
                "input_metadata_bundle_sha256": metadata_sha,
                "output_left_dir": str(left.parent),
                "output_right_dir": str(right.parent),
                "left_right_png_count": 2,
                "combined_png_sha256": combined,
                "producer_sha256": producer_ref["sha256"],
                "model_called": True,
                "gpu_started": True,
                "pixels_created_or_modified": True,
                "retained_output_root": {
                    "path": str(producer_manifest.parent),
                    "device": producer_manifest.parent.stat().st_dev,
                    "inode": producer_manifest.parent.stat().st_ino,
                    "method": "RETAINED_DIRECTORY_FD_OPENAT_MKDIRAT_O_EXCL",
                },
                "formal_admission_eligible": True,
                "detection_mode": "dual",
                "strict_dual": True,
                "d4_applied": True,
                "d4_scope": "GLOBAL_ALL_REQUESTED_SESSIONS_NO_SESSION_BRANCH",
                "d3_candidate_provenance_complete": True,
                "wearable_w1_w4_applied": True,
                "wearable_prompt_order": ["W1", "W2", "W3", "W4"],
                "wearable_combination_rule": module.PRODUCER_WEARABLE_COMBINATION_RULE,
                "wearable_detection_method": module.PRODUCER_WEARABLE_DETECTION_METHOD,
                "wearable_semantic_change_class": module.PRODUCER_WEARABLE_SEMANTIC_CHANGE_CLASS,
                "prompt_contract_sha256": module.PRODUCER_PROMPT_CONTRACT_SHA256,
                "live_producer": live_chain,
                "command_manifest": {
                    "command": command,
                    "completion_mode": "ARTIFACT_EXISTS",
                    "success_evidence": module._producer_success_evidence(
                        left.parent, right.parent, producer_manifest, 1
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    canary = {
        "session_id": "grap_a_cap_004",
        "frame_index": 0,
        "input_metadata_bundle_sha256": metadata_sha,
        "left_png": left_ref,
        "right_png": right_ref,
        "producer_manifest": module._file_ref(
            producer_manifest, "test canary producer manifest"
        ),
    }
    admission = {
        "schema_version": module.PRODUCER_ADMISSION_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "producer": producer_ref,
        "producer_args": producer_args,
        "plan": plan_ref,
        "constituent_manifest": constituent_ref,
        "command_manifest": {
            "command": command,
            "completion_mode": "ARTIFACT_EXISTS",
            "success_evidence": module._canary_success_evidence(canary),
        },
        "canary": canary,
    }
    return producer_manifest, admission


def test_resume_chain_redecodes_every_bound_raw_frame_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    module = _module()
    chain = _live_chain_fixture(module, tmp_path)
    module._validate_live_producer_chain(chain, None)
    bound_raw = json.loads(
        Path(chain["context"]["bound_refs"]["raw_input_manifest"]["path"]).read_text()
    )
    row = bound_raw["sessions"]["s"]["frames"][0]
    Path(row["path"]).write_bytes(b"not-a-decodable-png")
    with pytest.raises(module.ShardRunError, match="bound RAW frame"):
        module._validate_live_producer_chain(chain, None)


def _validate_test_admission(module, path: Path, admission: dict, producer=None):
    plan, plan_path, plan_sha = module._load_plan(PLAN)
    constituent, constituent_ref = module._load_constituent_manifest(plan)
    plan_ref = {
        "path": str(plan_path),
        "bytes": plan_path.stat().st_size,
        "sha256": plan_sha,
    }
    return module._validate_producer_admission(
        path,
        admission["producer"] if producer is None else producer,
        admission["producer_args"],
        plan,
        plan_ref,
        constituent,
        constituent_ref,
    )


def test_h4_admission_requires_real_decoded_canary_and_non_null_command_manifest(
    tmp_path: Path,
) -> None:
    module = _module()
    _, admission = _producer_admission_fixture(module, tmp_path)
    admission_path = tmp_path / "ADMISSION.json"
    admission_path.write_text(json.dumps(admission) + "\n", encoding="utf-8")
    observed = _validate_test_admission(module, admission_path, admission)
    assert observed["bytes"] == admission_path.stat().st_size

    missing_command = copy.deepcopy(admission)
    missing_command["command_manifest"] = None
    invalid_path = tmp_path / "INVALID_NULL_COMMAND.json"
    invalid_path.write_text(json.dumps(missing_command) + "\n", encoding="utf-8")
    with pytest.raises(module.ShardRunError, match="command_manifest"):
        _validate_test_admission(module, invalid_path, admission)


def test_h4_admission_rejects_exit_zero_command_or_unbound_producer(
    tmp_path: Path,
) -> None:
    module = _module()
    _, admission = _producer_admission_fixture(module, tmp_path)
    invalid_command = copy.deepcopy(admission)
    invalid_command["command_manifest"]["command"] = ["/usr/bin/true"]
    invalid_path = tmp_path / "INVALID_COMMAND.json"
    invalid_path.write_text(json.dumps(invalid_command) + "\n", encoding="utf-8")
    with pytest.raises(module.ShardRunError, match="command_manifest"):
        _validate_test_admission(module, invalid_path, admission)

    with pytest.raises(module.ShardRunError, match="identity/status"):
        _validate_test_admission(
            module,
            tmp_path / "INVALID_COMMAND.json",
            admission,
            {**admission["producer"], "sha256": "f" * 64},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "grap_a_cap_012"),
        ("input_metadata_bundle_sha256", "f" * 64),
        ("frame_index", 460),
    ],
)
def test_h4_canary_must_be_exact_frame_in_frozen_eligible68_row(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    module = _module()
    _, admission = _producer_admission_fixture(module, tmp_path)
    admission["canary"][field] = value
    path = tmp_path / "OUT_OF_SCOPE_ADMISSION.json"
    path.write_text(json.dumps(admission), encoding="utf-8")
    with pytest.raises(module.ShardRunError, match="frozen eligible68 plan"):
        _validate_test_admission(module, path, admission)


def test_producer_manifest_binds_real_run_flags_and_output_digest(
    tmp_path: Path,
) -> None:
    module = _module()
    live_chain = _live_chain_fixture(module, tmp_path)
    session = {
        "session_id": "grap_a_cap_004",
        "frame_count": 1,
        "input_metadata_bundle_sha256": "a" * 64,
    }
    mask_stats = {"combined_png_sha256": "b" * 64}
    manifest = tmp_path / "PRODUCER_MANIFEST.json"
    value = {
        "schema_version": module.PRODUCER_MANIFEST_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": session["session_id"],
        "frame_count": 1,
        "input_metadata_bundle_sha256": session["input_metadata_bundle_sha256"],
        "output_left_dir": str(tmp_path / "masks/left"),
        "output_right_dir": str(tmp_path / "masks/right"),
        "left_right_png_count": 2,
        "combined_png_sha256": mask_stats["combined_png_sha256"],
        "producer_sha256": "c" * 64,
        "model_called": True,
        "gpu_started": True,
        "pixels_created_or_modified": True,
        "retained_output_root": {
            "path": str(manifest.parent),
            "device": manifest.parent.stat().st_dev,
            "inode": manifest.parent.stat().st_ino,
            "method": "RETAINED_DIRECTORY_FD_OPENAT_MKDIRAT_O_EXCL",
        },
        "formal_admission_eligible": True,
        "detection_mode": "dual",
        "strict_dual": True,
        "d4_applied": True,
        "d4_scope": "GLOBAL_ALL_REQUESTED_SESSIONS_NO_SESSION_BRANCH",
        "d3_candidate_provenance_complete": True,
        "wearable_w1_w4_applied": True,
        "wearable_prompt_order": ["W1", "W2", "W3", "W4"],
        "wearable_combination_rule": module.PRODUCER_WEARABLE_COMBINATION_RULE,
        "wearable_detection_method": module.PRODUCER_WEARABLE_DETECTION_METHOD,
        "wearable_semantic_change_class": module.PRODUCER_WEARABLE_SEMANTIC_CHANGE_CLASS,
        "prompt_contract_sha256": module.PRODUCER_PROMPT_CONTRACT_SHA256,
        "live_producer": live_chain,
    }
    value["command_manifest"] = {
        "command": [
            sys.executable,
            "/proc/self/fd/90",
            "--session-id",
            session["session_id"],
            "--expected-frames",
            "1",
            "--input-metadata-sha256",
            session["input_metadata_bundle_sha256"],
            "--output-left-dir",
            value["output_left_dir"],
            "--output-right-dir",
            value["output_right_dir"],
            "--output-manifest",
            str(manifest),
        ],
        "completion_mode": "ARTIFACT_EXISTS",
        "success_evidence": module._producer_success_evidence(
            Path(value["output_left_dir"]),
            Path(value["output_right_dir"]),
            manifest,
            1,
        ),
    }
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
    reference, observed = module._validate_producer_manifest(
        manifest, session, mask_stats
    )
    assert reference["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert observed == value

    value["model_called"] = False
    invalid = tmp_path / "INVALID_PRODUCER_MANIFEST.json"
    invalid.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(module.ShardRunError, match="content mismatch"):
        module._validate_producer_manifest(invalid, session, mask_stats)


def _snapshot_files(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_actual_smoke_session_terminal_revalidates_read_only_twice() -> None:
    module = _module()
    plan, _, _ = module._load_plan(PLAN)
    session = module._select_shard(plan, "eligible68-train60-shard-01")["sessions"][0]
    session_dir = SMOKE / "sessions/grap_a_cap_004"
    before = _snapshot_files(SMOKE)
    first = module._validate_terminal(
        session_dir / "SESSION_COMPLETE_001.json", session, 1
    )
    middle = _snapshot_files(SMOKE)
    second = module._validate_terminal(
        session_dir / "SESSION_COMPLETE_001.json", session, 1
    )
    after = _snapshot_files(SMOKE)
    assert first == second
    assert before == middle == after
    assert first["session_id"] == "grap_a_cap_004"
    assert first["frame_count"] == 460
    assert first["video"]["frames"] == 460
    assert first["video"]["full_decode"] is True


def _fake_session_terminal(
    path: Path, *, gpu: bool = False, model: bool = False
) -> None:
    path.write_text(
        json.dumps({"gpu_started": gpu, "model_called": model}) + "\n",
        encoding="utf-8",
    )


def _fake_session_ref(module, path: Path, session_id: str) -> dict:
    reference = module._file_ref(path, "test session terminal")
    return {
        **reference,
        "session_id": session_id,
        "frame_count": 1,
        "attempt": str(path.parent / "attempt_001"),
        "mask_manifest": {
            "path": str(path.parent / "attempt_001/MASK_MANIFEST.json"),
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "video": {
            "path": str(path.parent / "attempt_001/MASK_LEFT_RIGHT.mp4"),
            "bytes": 1,
            "sha256": "2" * 64,
            "frames": 1,
            "full_decode": True,
        },
    }


def test_top_terminal_is_bound_to_exact_revalidated_session_refs(
    tmp_path: Path,
) -> None:
    module = _module()
    shard = {
        "shard_id": "eligible68-test-shard",
        "sessions": [{"session_id": "s1", "frame_count": 1}],
    }
    session_terminal = tmp_path / "sessions/s1/SESSION_COMPLETE_001.json"
    session_terminal.parent.mkdir(parents=True)
    _fake_session_terminal(session_terminal)
    session_ref = _fake_session_ref(module, session_terminal, "s1")
    written = module._write_run_terminal(
        tmp_path, "SHARD_COMPLETE", shard, [session_ref], False
    )
    before = _snapshot_files(tmp_path)
    assert (
        module._find_valid_run_terminal(
            tmp_path, "SHARD_COMPLETE", shard, [session_ref], False
        )
        == written
    )
    assert _snapshot_files(tmp_path) == before

    stale_ref = copy.deepcopy(session_ref)
    stale_ref["video"]["sha256"] = "f" * 64
    assert (
        module._find_valid_run_terminal(
            tmp_path, "SHARD_COMPLETE", shard, [stale_ref], False
        )
        is None
    )
    assert _snapshot_files(tmp_path) == before


def test_interrupted_resume_skips_complete_session_and_uses_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output_contract = module._bind_output_base(tmp_path)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    complete = {
        "path": str(sessions_dir / "s1/SESSION_COMPLETE_001.json"),
        "bytes": 1,
        "sha256": "a" * 64,
        "session_id": "s1",
    }
    partial_attempt = sessions_dir / "s2/attempt_001"
    partial_attempt.mkdir(parents=True)
    partial_marker = partial_attempt / "producer.log"
    partial_marker.write_bytes(b"interrupted\n")
    partial_before = (
        partial_marker.stat().st_mtime_ns,
        hashlib.sha256(partial_marker.read_bytes()).hexdigest(),
    )
    sessions = [
        {
            "session_id": "s1",
            "frame_count": 1,
            "input_metadata_bundle_sha256": "1" * 64,
        },
        {
            "session_id": "s2",
            "frame_count": 1,
            "input_metadata_bundle_sha256": "2" * 64,
        },
    ]
    calls: list[tuple[str, int]] = []

    def fake_find_completed(session_dir: Path, session: dict, contract, invocation):
        assert contract is output_contract
        assert invocation["run_mode"] == "PRODUCTION"
        return complete if session["session_id"] == "s1" else None

    def fake_run_session(
        session_dir: Path,
        session: dict,
        plan_ref: dict,
        smoke_inputs,
        producer,
        producer_ref,
        producer_admission_ref,
        producer_args,
        contract,
        invocation,
    ) -> dict:
        assert contract is output_contract
        assert invocation["run_mode"] == "PRODUCTION"
        index, attempt, _ = module._next_attempt(session_dir, contract)
        calls.append((session["session_id"], index))
        return {"session_id": session["session_id"], "attempt": str(attempt)}

    monkeypatch.setattr(module, "_find_completed", fake_find_completed)
    monkeypatch.setattr(module, "_run_session", fake_run_session)
    result = module._collect_completed_sessions(
        sessions_dir,
        sessions,
        True,
        {"sha256": module.PLAN_SHA256},
        None,
        Path("/unused/producer"),
        {"path": "/unused/producer", "bytes": 1, "sha256": "3" * 64},
        {"path": "/unused/admission", "bytes": 1, "sha256": "4" * 64},
        [],
        output_contract,
    )
    assert result.successful[0] is complete
    assert result.failed == []
    assert calls == [("s2", 2)]
    assert (partial_attempt.parent / "attempt_002").is_dir()
    assert (
        partial_marker.stat().st_mtime_ns,
        hashlib.sha256(partial_marker.read_bytes()).hexdigest(),
    ) == partial_before


def _failure_layout(module, tmp_path: Path, session_id: str = "s1"):
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_root = module._ensure_directory(base / "output", contract)
    shard_dir = module._ensure_directory(
        output_root / "eligible68-test-shard", contract
    )
    sessions_dir = module._ensure_directory(shard_dir / "sessions", contract)
    session_dir = module._ensure_directory(sessions_dir / session_id, contract)
    index, attempt, _ = module._next_attempt(session_dir, contract)
    session = {
        "session_id": session_id,
        "frame_count": 1,
        "input_metadata_bundle_sha256": "1" * 64,
    }
    plan_ref = {"path": "/frozen/plan.json", "bytes": 1, "sha256": "2" * 64}
    invocation = {
        "run_mode": "PRODUCTION",
        "producer_adapter": {
            "path": "/frozen/adapter.py",
            "bytes": 1,
            "sha256": "3" * 64,
        },
        "producer_admission": {
            "path": "/frozen/admission.json",
            "bytes": 1,
            "sha256": "4" * 64,
        },
        "producer_args": [],
    }
    return (
        contract,
        shard_dir,
        sessions_dir,
        session_dir,
        index,
        attempt,
        session,
        plan_ref,
        invocation,
    )


@pytest.mark.parametrize("failed_index", [0, 1])
def test_first_or_middle_failure_does_not_suppress_later_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_index: int,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    sessions_dir = module._ensure_directory(base / "sessions", contract)
    sessions = [
        {
            "session_id": f"s{index + 1}",
            "frame_count": 1,
            "input_metadata_bundle_sha256": f"{index + 1}" * 64,
        }
        for index in range(3)
    ]
    calls: list[str] = []

    def fake_run_session(
        session_dir: Path,
        session: dict,
        plan_ref: dict,
        smoke_inputs,
        producer,
        producer_ref,
        producer_admission_ref,
        producer_args,
        output_contract,
        run_invocation,
    ) -> dict:
        calls.append(session["session_id"])
        if session["session_id"] == sessions[failed_index]["session_id"]:
            raise module.RecordedSessionFailure(
                {
                    "path": str(session_dir / "SESSION_FAILED_001.json"),
                    "bytes": 1,
                    "sha256": "f" * 64,
                    "session_id": session["session_id"],
                    "frame_count": 1,
                    "attempt": str(session_dir / "attempt_001"),
                    "failure_stage": "PRODUCER_EXECUTION",
                    "error": {
                        "type": "RuntimeError",
                        "message": "fixture failure",
                        "producer_exit_code_diagnostic_only": None,
                    },
                    "run_mode": "PRODUCTION",
                    "failure_evidence": [],
                }
            )
        return {"session_id": session["session_id"]}

    monkeypatch.setattr(module, "_run_session", fake_run_session)
    outcomes = module._collect_completed_sessions(
        sessions_dir,
        sessions,
        False,
        {"sha256": module.PLAN_SHA256},
        None,
        Path("/unused/producer"),
        {"path": "/unused/producer", "bytes": 1, "sha256": "3" * 64},
        {"path": "/unused/admission", "bytes": 1, "sha256": "4" * 64},
        [],
        contract,
    )
    assert calls == ["s1", "s2", "s3"]
    assert [row["session_id"] for row in outcomes.successful] == [
        row["session_id"] for index, row in enumerate(sessions) if index != failed_index
    ]
    assert [row["session_id"] for row in outcomes.failed] == [
        sessions[failed_index]["session_id"]
    ]


@pytest.mark.parametrize("producer_failure", ["nonzero", "exception"])
def test_producer_nonzero_or_exception_gets_exact_failure_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer_failure: str,
) -> None:
    module = _module()
    (
        contract,
        _shard_dir,
        _sessions_dir,
        session_dir,
        _index,
        _attempt,
        session,
        plan_ref,
        invocation,
    ) = _failure_layout(module, tmp_path)
    # Reserve attempt_001 above to prove an abandoned attempt is not overwritten.
    producer = tmp_path / "producer.py"
    producer.write_text(
        "#!/usr/bin/env python3\nraise SystemExit(7)\n", encoding="utf-8"
    )
    producer.chmod(0o750)
    producer_ref = module._file_ref(producer, "test producer")
    if producer_failure == "exception":

        def explode(*args, **kwargs):
            raise RuntimeError("producer launch exploded")

        monkeypatch.setattr(module.subprocess, "run", explode)

    with pytest.raises(module.RecordedSessionFailure) as observed:
        module._run_session(
            session_dir,
            session,
            plan_ref,
            None,
            producer,
            producer_ref,
            invocation["producer_admission"],
            [],
            contract,
            {**invocation, "producer_adapter": producer_ref},
        )
    failure = observed.value.failure
    assert failure["attempt"].endswith("attempt_002")
    assert failure["failure_stage"] == "PRODUCER_EXECUTION"
    expected_type = (
        "ProducerProcessError" if producer_failure == "nonzero" else "RuntimeError"
    )
    assert failure["error"]["type"] == expected_type
    expected_exit = 7 if producer_failure == "nonzero" else None
    assert failure["error"]["producer_exit_code_diagnostic_only"] == expected_exit
    terminal = json.loads(Path(failure["path"]).read_text())
    assert terminal["completion_mode"] == "ARTIFACT_EXISTS"
    assert terminal["success_evidence"] == []
    assert terminal["artifact_success_claim"] is False
    assert terminal["pixels_successfully_produced_claim"] is False
    assert terminal["retry_requires_explicit_resume"] is True
    assert terminal["failure_evidence"]
    assert terminal["failure_evidence"][0]["bytes"] > 0


def test_mixed_shard_is_partial_hold_and_cannot_claim_complete(tmp_path: Path) -> None:
    module = _module()
    (
        contract,
        shard_dir,
        _sessions_dir,
        failure_session_dir,
        failure_index,
        failure_attempt,
        failure_session,
        plan_ref,
        invocation,
    ) = _failure_layout(module, tmp_path, "s2")
    failure = module._write_session_failure(
        module._failure_terminal_path(failure_session_dir, failure_index),
        failure_attempt,
        failure_session,
        failure_index,
        "PRODUCER_EXECUTION",
        module.ProducerProcessError(9),
        plan_ref,
        invocation,
        contract,
    )
    successful = []
    for session_id in ("s1", "s3"):
        session_dir = module._ensure_directory(
            shard_dir / "sessions" / session_id, contract
        )
        path = session_dir / "SESSION_COMPLETE_001.json"
        _fake_session_terminal(path, gpu=True, model=True)
        successful.append(_fake_session_ref(module, path, session_id))
    shard = {
        "shard_id": "eligible68-test-shard",
        "sessions": [
            {"session_id": "s1", "frame_count": 1},
            {"session_id": "s2", "frame_count": 1},
            {"session_id": "s3", "frame_count": 1},
        ],
    }
    with pytest.raises(module.ShardRunError, match="every planned session"):
        module._write_run_terminal(
            shard_dir,
            "SHARD_COMPLETE",
            shard,
            successful,
            False,
            contract,
        )
    partial_ref = module._write_partial_run_terminal(
        shard_dir,
        shard,
        successful,
        [failure],
        False,
        contract,
    )
    partial = json.loads(Path(partial_ref["path"]).read_text())
    assert partial["status"] == "SHARD_PARTIAL_HOLD_SESSION_FAILURES_ARTIFACT_EXISTS"
    assert partial["full_shard_complete"] is False
    assert partial["shard_success_claim"] is False
    assert partial["successful_sessions"] == ["s1", "s3"]
    assert partial["failed_sessions"] == ["s2"]
    assert not list(shard_dir.glob("SHARD_COMPLETE_*.json"))


def test_resume_revalidates_failure_and_retries_only_failed_or_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    (
        contract,
        _shard_dir,
        sessions_dir,
        failed_session_dir,
        failure_index,
        failure_attempt,
        failed_session,
        plan_ref,
        invocation,
    ) = _failure_layout(module, tmp_path, "s2")
    failure = module._write_session_failure(
        module._failure_terminal_path(failed_session_dir, failure_index),
        failure_attempt,
        failed_session,
        failure_index,
        "PRODUCER_EXECUTION",
        RuntimeError("first pass failed"),
        plan_ref,
        invocation,
        contract,
    )
    before = _snapshot_files(failed_session_dir)
    complete = {"session_id": "s1", "attempt": "/already/complete"}
    sessions = [
        {
            "session_id": "s1",
            "frame_count": 1,
            "input_metadata_bundle_sha256": "1" * 64,
        },
        failed_session,
        {
            "session_id": "s3",
            "frame_count": 1,
            "input_metadata_bundle_sha256": "3" * 64,
        },
    ]
    calls: list[tuple[str, int]] = []

    def fake_find_completed(session_dir, session, contract_arg, invocation_arg):
        assert contract_arg is contract
        assert invocation_arg == invocation
        return complete if session["session_id"] == "s1" else None

    def fake_run_session(
        session_dir,
        session,
        plan_ref_arg,
        smoke_inputs,
        producer,
        producer_ref,
        producer_admission_ref,
        producer_args,
        contract_arg,
        invocation_arg,
    ):
        index, attempt, _ = module._next_attempt(session_dir, contract_arg)
        calls.append((session["session_id"], index))
        return {"session_id": session["session_id"], "attempt": str(attempt)}

    monkeypatch.setattr(module, "_find_completed", fake_find_completed)
    monkeypatch.setattr(module, "_run_session", fake_run_session)
    outcomes = module._collect_completed_sessions(
        sessions_dir,
        sessions,
        True,
        plan_ref,
        None,
        Path("/frozen/adapter.py"),
        invocation["producer_adapter"],
        invocation["producer_admission"],
        [],
        contract,
    )
    assert outcomes.failed == []
    assert [row["session_id"] for row in outcomes.successful] == ["s1", "s2", "s3"]
    assert calls == [("s2", 2), ("s3", 1)]
    assert _snapshot_files(failed_session_dir) == before
    assert module._find_failures(
        failed_session_dir,
        failed_session,
        plan_ref,
        contract,
        invocation,
    ) == [failure]


def test_failure_terminal_is_create_once_and_rejects_alias_without_writes(
    tmp_path: Path,
) -> None:
    module = _module()
    (
        contract,
        _shard_dir,
        _sessions_dir,
        session_dir,
        index,
        attempt,
        session,
        plan_ref,
        invocation,
    ) = _failure_layout(module, tmp_path)
    terminal_path = module._failure_terminal_path(session_dir, index)
    failure = module._write_session_failure(
        terminal_path,
        attempt,
        session,
        index,
        "PRODUCER_EXECUTION",
        RuntimeError("immutable failure"),
        plan_ref,
        invocation,
        contract,
    )
    before_retry = _snapshot_files(session_dir)
    with pytest.raises(FileExistsError):
        module._write_session_failure(
            terminal_path,
            attempt,
            session,
            index,
            "PRODUCER_EXECUTION",
            RuntimeError("must not overwrite"),
            plan_ref,
            invocation,
            contract,
        )
    assert _snapshot_files(session_dir) == before_retry
    alias = attempt / "FAILURE_RECORD_ALIAS.json"
    alias.hardlink_to(attempt / "FAILURE_RECORD.json")
    after_alias = _snapshot_files(session_dir)
    with pytest.raises(module.OutputIdentityError, match="single-link"):
        module._validate_failure_terminal(
            Path(failure["path"]),
            session,
            index,
            plan_ref,
            contract,
            invocation,
        )
    assert _snapshot_files(session_dir) == after_alias


def test_failure_terminal_rejects_directory_race_without_writes(tmp_path: Path) -> None:
    module = _module()
    (
        contract,
        _shard_dir,
        _sessions_dir,
        session_dir,
        index,
        attempt,
        session,
        plan_ref,
        invocation,
    ) = _failure_layout(module, tmp_path)
    failure = module._write_session_failure(
        module._failure_terminal_path(session_dir, index),
        attempt,
        session,
        index,
        "PRODUCER_EXECUTION",
        RuntimeError("directory race fixture"),
        plan_ref,
        invocation,
        contract,
    )
    moved = session_dir.with_name("s1_original")
    session_dir.rename(moved)
    session_dir.mkdir()
    for child in moved.iterdir():
        child.rename(session_dir / child.name)
    before = _snapshot_files(session_dir)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._validate_failure_terminal(
            Path(failure["path"]),
            session,
            index,
            plan_ref,
            contract,
            invocation,
        )
    assert _snapshot_files(session_dir) == before


def test_attempt_claim_never_reuses_directory_that_appears_in_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    session_dir = module._ensure_directory(base / "sessions" / "s1", contract)
    original_mkdir = module.os.mkdir
    raced = False

    def race_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if path == "attempt_001" and dir_fd is not None and not raced:
            raced = True
            original_mkdir(path, mode=mode, dir_fd=dir_fd)
        return original_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", race_mkdir)
    index, attempt, terminal = module._next_attempt(session_dir, contract)
    assert raced is True
    assert index == 2
    assert attempt == session_dir / "attempt_002"
    assert terminal == session_dir / "SESSION_COMPLETE_002.json"
    assert (session_dir / "attempt_001").is_dir()
    assert attempt.is_dir()


def test_resume_rejects_smoke_to_production_and_changed_producer_chain() -> None:
    module = _module()
    smoke_chain = {
        "run_mode": "CPU_EXISTING_MASK_SMOKE",
        "source_root": "/tmp/real-smoke",
        "source_manifest": {
            "path": "/tmp/real-smoke.json",
            "bytes": 10,
            "sha256": "1" * 64,
        },
    }
    production = {
        "run_mode": "PRODUCTION",
        "producer_adapter": {
            "path": "/tmp/adapter.py",
            "bytes": 10,
            "sha256": "2" * 64,
        },
        "producer_admission": {
            "path": "/tmp/admission.json",
            "bytes": 10,
            "sha256": "3" * 64,
        },
        "producer_args": [],
    }
    with pytest.raises(module.InvocationDriftError, match="run-mode"):
        module._validate_chain_against_invocation(smoke_chain, production)

    persisted = {
        **production,
        "adapter_command": ["/tmp/adapter.py"],
        "live_producer": {},
    }
    changed = copy.deepcopy(production)
    changed["producer_adapter"]["sha256"] = "4" * 64
    with pytest.raises(module.InvocationDriftError, match="invocation drift"):
        module._validate_chain_against_invocation(persisted, changed)


def test_top_chain_binding_rejects_mixed_mode_or_live_source() -> None:
    module = _module()
    invocation = {
        "run_mode": "PRODUCTION",
        "producer_adapter": {"path": "/a", "bytes": 1, "sha256": "1" * 64},
        "producer_admission": {"path": "/b", "bytes": 1, "sha256": "2" * 64},
        "producer_args": [],
    }
    common = {
        **invocation,
        "executed_live_source_bytes": 100,
        "executed_live_source_sha256": "3" * 64,
        "live_capabilities": {"strict_dual": True},
    }
    sessions = [
        {
            "session_id": "s1",
            "run_mode": "PRODUCTION",
            "execution_chain_sha256": "4" * 64,
            "execution_chain_common": common,
        },
        {
            "session_id": "s2",
            "run_mode": "CPU_EXISTING_MASK_SMOKE",
            "execution_chain_sha256": "5" * 64,
            "execution_chain_common": common,
        },
    ]
    with pytest.raises(module.ShardRunError, match="run-mode mixture"):
        module._top_chain_binding(sessions, invocation)

    sessions[1]["run_mode"] = "PRODUCTION"
    sessions[1]["execution_chain_common"] = {
        **common,
        "executed_live_source_sha256": "6" * 64,
    }
    with pytest.raises(module.ShardRunError, match="inconsistent producer"):
        module._top_chain_binding(sessions, invocation)


def test_forbidden_population_path_is_rejected() -> None:
    module = _module()
    with pytest.raises(module.ShardRunError, match="forbidden path"):
        module._absolute(Path("/mnt/workspace/code/chaoyang/grap_a_cap_025/x"), "test")


def test_default_output_base_remains_project_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    run_root = tmp_path / "_run"
    run_root.mkdir()
    monkeypatch.setattr(module, "RUN_ROOT", run_root)
    contract = module._bind_output_base(None)
    output_root = run_root / "eligible68"
    assert contract.base == run_root
    assert contract.approval_mode == "DEFAULT_PROJECT_RUN_ROOT"
    assert module._ensure_directory(output_root, contract) == output_root
    assert output_root.is_dir()


def test_explicit_output_base_must_exist_be_absolute_and_canonical(
    tmp_path: Path,
) -> None:
    module = _module()
    with pytest.raises(module.ShardRunError, match="filesystem anchor"):
        module._bind_output_base(Path("/"))
    with pytest.raises(module.ShardRunError, match="must be absolute"):
        module._bind_output_base(Path("relative/output"))
    with pytest.raises(module.ShardRunError, match="does not exist"):
        module._bind_output_base(tmp_path / "missing")

    base = tmp_path / "approved"
    base.mkdir()
    with pytest.raises(module.ShardRunError, match="must not contain '..'"):
        module._bind_output_base(base / "child" / "..")


def test_output_root_must_be_strict_descendant_without_parent_traversal(
    tmp_path: Path,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    with pytest.raises(module.ShardRunError, match="strict descendant"):
        module._output_parts(base, contract, "output root")
    with pytest.raises(module.ShardRunError, match="must be a descendant"):
        module._output_parts(tmp_path / "outside", contract, "output root")
    with pytest.raises(module.ShardRunError, match="must not contain '..'"):
        module._output_parts(base / "safe" / ".." / "escape", contract, "output root")


def test_symlink_base_and_output_component_are_rejected_without_escape(
    tmp_path: Path,
) -> None:
    module = _module()
    real_base = tmp_path / "real_base"
    real_base.mkdir()
    linked_base = tmp_path / "linked_base"
    linked_base.symlink_to(real_base, target_is_directory=True)
    with pytest.raises(module.ShardRunError, match="symlink"):
        module._bind_output_base(linked_base)

    contract = module._bind_output_base(real_base)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_component = real_base / "linked_component"
    linked_component.symlink_to(outside, target_is_directory=True)
    with pytest.raises(module.ShardRunError, match="symlink|non-directory"):
        module._ensure_directory(linked_component / "must_not_exist", contract)
    assert not (outside / "must_not_exist").exists()


def test_hardlinked_output_file_is_rejected(tmp_path: Path) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not an owned output inode")
    hardlink = output_dir / "artifact.bin"
    hardlink.hardlink_to(outside)
    with pytest.raises(module.ShardRunError, match="single-link"):
        module._assert_output_file(hardlink, contract, "test artifact")


def test_nonregular_output_file_is_rejected_without_opening_it(tmp_path: Path) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    fifo = output_dir / "artifact.pipe"
    module.os.mkfifo(fifo)
    with pytest.raises(module.OutputIdentityError, match="single-link regular"):
        module._assert_output_file(fifo, contract, "test artifact")


def test_bound_base_and_created_component_replacement_are_rejected(
    tmp_path: Path,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    moved_output = base / "job_original"
    output_dir.rename(moved_output)
    output_dir.mkdir()
    with pytest.raises(module.ShardRunError, match="identity drift"):
        module._ensure_directory(output_dir / "child", contract)

    moved_base = tmp_path / "approved_original"
    base.rename(moved_base)
    base.mkdir()
    with pytest.raises(module.ShardRunError, match="identity drift"):
        module._ensure_directory(base / "new_job", contract)


@pytest.mark.parametrize("drift_level", ["base", "output_root", "shard_root"])
def test_persisted_output_binding_rejects_cross_process_resume_drift_without_writes(
    tmp_path: Path,
    drift_level: str,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_root = module._ensure_directory(base / "output", contract)
    shard_root = module._ensure_directory(
        output_root / "eligible68-test-shard", contract
    )
    session_dir = module._ensure_directory(shard_root / "sessions" / "s1", contract)
    session_terminal = session_dir / "SESSION_COMPLETE_001.json"
    _fake_session_terminal(session_terminal)
    session_ref = _fake_session_ref(module, session_terminal, "s1")
    shard = {
        "shard_id": "eligible68-test-shard",
        "sessions": [{"session_id": "s1", "frame_count": 1}],
    }
    terminal_ref = module._write_run_terminal(
        shard_root,
        "SHARD_COMPLETE",
        shard,
        [session_ref],
        False,
        contract,
    )
    terminal_value = json.loads(Path(terminal_ref["path"]).read_text())
    binding = terminal_value["output_binding"]
    assert binding["approved_output_base"] == {
        "approval_mode": "EXPLICIT_CALLER_APPROVED_BASE",
        "canonical_path": str(base),
        "device": base.stat().st_dev,
        "inode": base.stat().st_ino,
    }
    assert binding["output_root"]["canonical_path"] == str(output_root)
    assert binding["shard_root"]["canonical_path"] == str(shard_root)

    if drift_level == "base":
        moved = tmp_path / "approved_original"
        base.rename(moved)
        base.mkdir()
        (moved / output_root.name).rename(output_root)
    elif drift_level == "output_root":
        moved = base / "output_original"
        output_root.rename(moved)
        output_root.mkdir()
        (moved / shard_root.name).rename(shard_root)
    else:
        moved = output_root / "eligible68-test-shard_original"
        shard_root.rename(moved)
        shard_root.mkdir()
        for child in list(moved.iterdir()):
            child.rename(shard_root / child.name)

    before = _snapshot_files(output_root)
    rebound = module._bind_output_base(base)
    with pytest.raises(module.OutputIdentityError, match="persisted output"):
        module._find_valid_run_terminal(
            shard_root,
            "SHARD_COMPLETE",
            shard,
            [session_ref],
            False,
            rebound,
        )
    assert _snapshot_files(output_root) == before


def test_video_encode_held_fd_produces_full_decodable_mp4(tmp_path: Path) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    attempt = module._ensure_directory(base / "attempt", contract)
    left = module._ensure_directory(attempt / "masks" / "left", contract)
    right = module._ensure_directory(attempt / "masks" / "right", contract)
    png = (
        SMOKE / "sessions/grap_a_cap_004/attempt_001/masks/left/frame_00000.png"
    ).read_bytes()
    module._write_exclusive(left / "frame_00000.png", png, output_contract=contract)
    module._write_exclusive(right / "frame_00000.png", png, output_contract=contract)

    video = module._encode_video(attempt, 1, contract)
    observed = module._validate_video(video, 1, contract)
    assert video == attempt / "MASK_LEFT_RIGHT.mp4"
    assert observed["frames"] == 1
    assert observed["full_decode"] is True
    assert observed["bytes"] > 0
    assert video.stat().st_nlink == 1


@pytest.mark.parametrize("encoder_raises", [False, True])
def test_video_encode_uses_held_exclusive_fd_and_rolls_back_parent_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder_raises: bool,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    contract = module._bind_output_base(base)
    attempt = module._ensure_directory(base / "attempt", contract)
    module._ensure_directory(attempt / "masks" / "left", contract)
    module._ensure_directory(attempt / "masks" / "right", contract)
    moved_attempt = outside / "moved_attempt"
    replacement_payload = b"independent textual-path replacement"

    def raced_encoder(command, *, pass_fds=()):
        assert len(pass_fds) == 1
        output_fd = pass_fds[0]
        assert command[-4:] == [
            "-f",
            "mp4",
            "-y",
            f"/proc/self/fd/{output_fd}",
        ]
        attempt.rename(moved_attempt)
        attempt.mkdir()
        (attempt / "MASK_LEFT_RIGHT.mp4").write_bytes(replacement_payload)
        module.os.write(output_fd, b"partial held video bytes")
        if encoder_raises:
            raise RuntimeError("encoder launch/runtime exception after rename")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "_run_checked", raced_encoder)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._encode_video(attempt, 1, contract)
    assert (attempt / "MASK_LEFT_RIGHT.mp4").read_bytes() == replacement_payload
    assert not (moved_attempt / "MASK_LEFT_RIGHT.mp4").exists()


def test_exclusive_write_rolls_back_if_parent_moves_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    original_open = module.os.open
    moved = outside / "moved_job"
    fired = False

    def hostile_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal fired
        if path == "artifact.bin" and dir_fd is not None and not fired:
            fired = True
            output_dir.rename(moved)
            output_dir.mkdir()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", hostile_open)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._write_exclusive(
            output_dir / "artifact.bin",
            b"must not escape",
            output_contract=contract,
        )
    assert not (output_dir / "artifact.bin").exists()
    assert not (moved / "artifact.bin").exists()


def test_exclusive_write_rolls_back_if_parent_moves_during_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    original_write = module.os.write
    moved = outside / "moved_job"
    fired = False

    def hostile_write(fd: int, payload: bytes) -> int:
        nonlocal fired
        if not fired:
            fired = True
            output_dir.rename(moved)
            output_dir.mkdir()
        return original_write(fd, payload)

    monkeypatch.setattr(module.os, "write", hostile_write)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._write_exclusive(
            output_dir / "artifact.bin",
            b"must be rolled back",
            output_contract=contract,
        )
    assert not (output_dir / "artifact.bin").exists()
    assert not (moved / "artifact.bin").exists()


def test_exclusive_rollback_never_deletes_replacement_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    artifact = output_dir / "artifact.bin"
    moved_created = output_dir / "created_inode_moved.bin"
    replacement = b"independent replacement inode"
    original_write = module.os.write
    fired = False

    def hostile_write(fd: int, payload: bytes) -> int:
        nonlocal fired
        if not fired:
            fired = True
            artifact.rename(moved_created)
            artifact.write_bytes(replacement)
        return original_write(fd, payload)

    monkeypatch.setattr(module.os, "write", hostile_write)
    with pytest.raises(module.OutputIdentityError, match="entry identity drift"):
        module._write_exclusive(
            artifact,
            b"created inode payload",
            output_contract=contract,
        )
    assert artifact.read_bytes() == replacement
    assert moved_created.read_bytes() == b"created inode payload"


def test_exclusive_write_is_create_once_and_never_overwrites(tmp_path: Path) -> None:
    module = _module()
    base = tmp_path / "approved"
    base.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    artifact = output_dir / "artifact.bin"
    module._write_exclusive(artifact, b"first", output_contract=contract)
    before = _snapshot_files(base)
    with pytest.raises(FileExistsError):
        module._write_exclusive(artifact, b"second", output_contract=contract)
    assert artifact.read_bytes() == b"first"
    assert _snapshot_files(base) == before


def test_directory_create_rolls_back_if_parent_moves_before_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    moved = outside / "moved_job"
    original_mkdir = module.os.mkdir
    fired = False

    def hostile_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal fired
        if path == "child" and dir_fd is not None and not fired:
            fired = True
            output_dir.rename(moved)
            output_dir.mkdir()
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "mkdir", hostile_mkdir)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._ensure_directory(output_dir / "child", contract)
    assert not (output_dir / "child").exists()
    assert not (moved / "child").exists()


def test_directory_create_rolls_back_if_parent_moves_after_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    moved = outside / "moved_job"
    original_verify = module._verify_output_directory_chain
    fired = False

    def hostile_verify(absolute, parts, output_contract):
        nonlocal fired
        if absolute == output_dir / "child" and not fired:
            fired = True
            output_dir.rename(moved)
            output_dir.mkdir()
        return original_verify(absolute, parts, output_contract)

    monkeypatch.setattr(module, "_verify_output_directory_chain", hostile_verify)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._ensure_directory(output_dir / "child", contract)
    assert not (output_dir / "child").exists()
    assert not (moved / "child").exists()


def test_directory_rollback_never_deletes_replacement_or_nonempty_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    base = tmp_path / "approved"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    contract = module._bind_output_base(base)
    output_dir = module._ensure_directory(base / "job", contract)
    child = output_dir / "child"
    moved_created = outside / "created_child_moved"
    marker = b"independent replacement contents"
    original_verify = module._verify_output_directory_chain
    fired = False

    def hostile_verify(absolute, parts, output_contract):
        nonlocal fired
        if absolute == child and not fired:
            fired = True
            child.rename(moved_created)
            child.mkdir()
            (child / "marker.bin").write_bytes(marker)
        return original_verify(absolute, parts, output_contract)

    monkeypatch.setattr(module, "_verify_output_directory_chain", hostile_verify)
    with pytest.raises(module.OutputIdentityError, match="identity drift"):
        module._ensure_directory(child, contract)
    assert (child / "marker.bin").read_bytes() == marker
    assert moved_created.is_dir()
