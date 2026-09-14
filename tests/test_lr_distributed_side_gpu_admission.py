from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from pipeline import lr_distributed_side_evidence as selector
from pipeline import lr_distributed_side_gpu_admission as gate
from tools import run_lr_distributed_side_gpu as runner


PROJECT = Path(__file__).resolve().parents[1]


def ref(path: Path, kind: str) -> selector.EvidenceRef:
    return selector.EvidenceRef(
        str(path), path.stat().st_size, selector.sha256_file(path), kind
    )


def test_frozen_task26_and_cpu_bindings_are_exact() -> None:
    for relative, expected in gate.FIXED_SHA256.items():
        assert selector.sha256_file(PROJECT / relative) == expected
    assert gate.FIXED_SHA256[selector.TASK26_RELATIVE_PATH] == (
        "ac58a593b4642e562686f018ba75f0bc75a9c8563ad943f61c2d9bd4363a2843"
    )


def test_gpu_config_has_only_preregistered_semantics() -> None:
    config = gate.GPU_CONFIG
    assert config["prompt"] == "an arm"
    assert (
        config["min_joint_support_ratio"],
        config["min_side_margin"],
        config["max_object_overlap_over_instance"],
    ) == (0.20, 0.05, 0.12)
    assert config["wrist_supported_role"] == "DIAGNOSTIC_ONLY"
    assert config["authority_wrist_outside_image_role"] == "DIAGNOSTIC_ONLY"
    assert config["side_routing_basis"] == "IN_IMAGE_HAWOR_JOINT_SUPPORT_DISTRIBUTION_ONLY"
    assert set(config["fixed_frame_indices"]) == set(selector.FIXED_FRAME_INDICES)
    assert sum(map(len, config["fixed_frame_indices"].values())) == 39
    assert config["wrong_side_sentinels"] == [
        {
            "session_id": "grap_a_cap_005",
            "frame_index": 220,
            "raw_instance_offset": 1,
            "expected_side": "right",
        },
        {
            "session_id": "grap_a_cap_005",
            "frame_index": 270,
            "raw_instance_offset": 1,
            "expected_side": "right",
        },
    ]
    assert config["forbidden_session"] == "grap_a_cap_025"
    assert config["advancement_authorized"] is False
    assert config["formal_consumer_allowed"] is False


def test_canonical_independent_qa_is_exact_p0_zero() -> None:
    qa_ref = gate.evidence_ref(
        PROJECT, gate.CANONICAL_QA_RELATIVE_PATH, "INDEPENDENT_QA_JSON"
    )
    assert qa_ref.sha256 == gate.INDEPENDENT_QA_SHA256
    qa = gate._verify_independent_qa(
        qa_ref.read_verified(allowed_root=PROJECT),
        project_root=PROJECT,
        qa_ref=qa_ref,
    )
    assert qa["p0_findings"] == 0
    assert qa["claim_limits"]["gpu_execution_authorized_by_this_qa"] is False


def test_nonzero_p0_fails_before_canonical_path_check(tmp_path: Path) -> None:
    relative = Path(gate.CANONICAL_QA_RELATIVE_PATH)
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    value = json.loads((PROJECT / relative).read_text())
    value["p0_findings"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(gate.DistributedSideGpuAdmissionError, match="p0_findings"):
        gate._verify_independent_qa(
            path.read_bytes(),
            project_root=tmp_path,
            qa_ref=ref(path, "INDEPENDENT_QA_JSON"),
        )


def test_release_requires_separate_authorization_source(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.json"
    prep_path = tmp_path / "prep.json"
    release_path = tmp_path / gate.CANONICAL_RELEASE_RELATIVE_PATH
    release_path.parent.mkdir(parents=True)
    qa_path.write_text("{}")
    prep_path.write_text("{}")
    release = {
        "schema_version": "a-prime-distributed-side-gpu-release-v1",
        "authorization_scope": "TASK26_FIXED_39_FRAME_GPU_REPLAY_ONLY",
        "route_of_evidence": selector.ROUTE_OF_EVIDENCE,
        "release_written_by": "ROOT_PROJECT_MAINLINE_AFTER_QA",
        "gpu_release_authorized": True,
        "gpu_idle_verified_after_handoff": True,
        "task27_baseline_state": "NOT_STARTED_NO_GPU_OWNERSHIP",
        "task27_handoff_ref": None,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "independent_qa_ref": asdict(ref(qa_path, "INDEPENDENT_QA_JSON")),
        "gpu_prep_freeze_ref": asdict(ref(prep_path, "TASK26_GPU_PREP_FREEZE")),
    }
    release_path.write_text(json.dumps(release))
    with pytest.raises(
        gate.DistributedSideGpuAdmissionError, match="authorization source missing"
    ):
        gate._verify_release(
            release_path.read_bytes(),
            project_root=tmp_path,
            release_ref=ref(release_path, "OWNER_GPU_RELEASE"),
            qa_ref=ref(qa_path, "INDEPENDENT_QA_JSON"),
            prep_ref=ref(prep_path, "TASK26_GPU_PREP_FREEZE"),
        )


def test_o_excl_helper_creates_direct_child_and_rejects_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_parent = tmp_path / "_run"
    run_parent.mkdir()
    prep_path = tmp_path / "prep.json"
    qa_path = tmp_path / "qa.json"
    release_path = tmp_path / "release.json"
    for path in (prep_path, qa_path, release_path):
        path.write_text("{}")
    monkeypatch.setattr(gate, "_verify_fixed_sources", lambda project: {})
    monkeypatch.setattr(gate, "_verify_prep", lambda *args, **kwargs: {})
    monkeypatch.setattr(gate, "_verify_independent_qa", lambda *args, **kwargs: {})
    monkeypatch.setattr(gate, "_verify_release", lambda *args, **kwargs: {})
    kwargs = {
        "project_root": tmp_path,
        "project_run_root": run_parent,
        "run_root": run_parent / "lr_distributed_side_candidate_t1_fixture",
        "prep_ref": ref(prep_path, "PREP"),
        "independent_qa_ref": ref(qa_path, "QA"),
        "owner_release_ref": ref(release_path, "RELEASE"),
    }
    owner = gate.require_distributed_side_gpu_admission(**kwargs)
    assert Path(owner.path).is_file()
    with pytest.raises(gate.DistributedSideGpuAdmissionError, match="already exists"):
        gate.require_distributed_side_gpu_admission(**kwargs)


def test_storage_probe_writes_fsyncs_and_retains_no_bytes(tmp_path: Path) -> None:
    result = runner.storage_probe(tmp_path)
    assert result["probe_bytes_written_and_fsynced"] == runner.STORAGE_PROBE_BYTES
    assert result["probe_retained"] is False
    assert list(tmp_path.iterdir()) == []


def test_runner_has_small_hard_cap_and_no_top_level_torch_import() -> None:
    assert runner.MAX_RUN_BYTES == 1536 * 1024**2
    assert runner.MIN_FREE_BYTES_DURING_RUN == 8 * 1024**3
    source = (PROJECT / gate.GPU_RUNNER_RELATIVE_PATH).read_text()
    assert "        import torch\n" in source
    assert "\nimport torch\n" not in source
    assert "sys.path.insert(0, str(official_code))" in source
    assert "execution interpreter differs from frozen runtime" in source
    assert "pipeline_package = types.ModuleType" in source
    assert "pipeline/__init__.py" not in source


def test_wrong_side_gate_does_not_accept_wrist_or_oob_inputs() -> None:
    import inspect

    signature = inspect.signature(selector.check_wrong_side_acceptance)
    assert set(signature.parameters) == {"frames", "sentinels"}
    assert "wrist_supported" not in signature.parameters
    assert "authority_state" not in signature.parameters


def test_paired_raw_inventory_binds_order_id_and_mask(tmp_path: Path) -> None:
    import numpy as np

    mask = np.zeros((8, 9), dtype=np.bool_)
    mask[2:5, 3:7] = True
    payload = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": "grap_a_cap_005:220",
        "instance_id": 1,
        "mask_sha256": selector.mask_sha256(mask),
    }
    source_path = tmp_path / "old.json"
    source_path.write_text(json.dumps(payload))
    source = {
        "raw_instance_evidence": {
            "0": {
                "instance_id": 1,
                "raw_source": asdict(ref(source_path, "SAM_RAW_INSTANCE")),
            }
        }
    }
    support = selector.JointSupportEvidence(1, 1, 1.0, 1.0, True)
    raw = selector.RawInstance(
        "grap_a_cap_005", 220, 0, 1,
        ref(source_path, "SAM_RAW_INSTANCE"), mask, 0.9,
        {"left": support, "right": support},
        {"left": False, "right": False}, 0.0, float(mask.mean()),
    )
    exact = runner.compare_raw_inventory(
        tmp_path, source, [raw], session_id="grap_a_cap_005", frame_index=220
    )
    assert exact["status"] == "PASS_EXACT_V3_RAW_INVENTORY"
    changed = mask.copy()
    changed[0, 0] = True
    drifted = selector.RawInstance(
        "grap_a_cap_005", 220, 0, 1,
        ref(source_path, "SAM_RAW_INSTANCE"), changed, 0.9,
        {"left": support, "right": support},
        {"left": False, "right": False}, 0.0, float(changed.mean()),
    )
    with pytest.raises(runner.SemanticsOnlyInventoryDrift) as caught:
        runner.compare_raw_inventory(
            tmp_path, source, [drifted],
            session_id="grap_a_cap_005", frame_index=220,
        )
    assert caught.value.inventory["status"] == "HOLD_NOT_SEMANTICS_ONLY"


def test_terminal_manifest_is_published_after_artifact_manifest() -> None:
    source = (PROJECT / gate.GPU_RUNNER_RELATIVE_PATH).read_text()
    artifact_write = source.index('component / "ARTIFACT_MANIFEST.json"')
    terminal_comment = source.index("RUN_MANIFEST is the terminal disposition")
    terminal_write = source.index("write_json_exclusive(manifest_path, result)", terminal_comment)
    assert artifact_write < terminal_comment < terminal_write
    assert "HOLD_NOT_SEMANTICS_ONLY_RAW_SAM_INVENTORY_DRIFT" in source
    assert "HOLD_GENERALIZATION_AND_SIDE_GAP_EXCEED_PREREGISTERED_GATES_" in source
