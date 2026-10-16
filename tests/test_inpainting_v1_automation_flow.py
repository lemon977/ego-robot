"""Synthetic CPU checks for one fenced V1-compatible six-stage submission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import automation as auto
from chaoyang.inpainting_candidate.automation_mask import associate_original_objects
from chaoyang.inpainting_candidate.contracts import SCHEMA, STAGES, canonical_sha, file_ref


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_ref(path)


@pytest.fixture
def project(tmp_path, monkeypatch):
    # Synthetic fixtures remain inside the authorized project-local _run root.
    # Real parent requests retain the production derived-source guard.
    monkeypatch.setattr(auto, "DERIVED_COMPONENTS", auto.DERIVED_COMPONENTS - {"_run"})
    raw = tmp_path / "raw_capture"
    rows, leaves = [], []
    for sid, count in (("050", 3), ("007", 5), ("001", 7)):
        root = raw / sid
        pins = {}
        for role, relative in auto.RAW_NAMES.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if role == "video_qpc":
                path.write_text("\n".join(str((i + 1) * 40_000_000) for i in range(count)))
            elif role == "source_manifest":
                path.write_text(json.dumps({"session": "pick_place_1008_" + sid}))
            else:
                path.write_bytes((sid + relative).encode())
            pins[role] = file_ref(path)
        seed = save(tmp_path / "seeds" / (sid + ".json"), {"session_id": sid, "objects": []})
        leaves.append({"session_id": sid, "seed": seed})
        rows.append({"session_id": sid, "raw_frame_count": count, "input_status": "READY_RAW_AUDIT", "source_refs": pins})
    manifest = save(tmp_path / "raw_manifest.json", {"source_root": str(raw), "sessions": rows})
    camera = save(tmp_path / "camera.json", {"frozen_camera": True})
    seeds = save(tmp_path / "seeds.json", {"schema": "pick_place_mask_seed_index_v1", "records": leaves})
    save(tmp_path / "inpainting_v1/MANIFEST.json", {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1",
        "inputs": {"raw_manifest": manifest, "camera": camera, "mask_seeds": seeds, "per_session_seed_leaves": leaves}})
    selection = tmp_path / "selection.json"
    save(selection, {"schema": "inpainting-raw-session-manifest-v1", "sessions": [str(raw / "050"), str(raw / "007")]})
    return SimpleNamespace(root=tmp_path, raw=raw, selection=selection)


def request(project, sid=None):
    return auto.build_v1_request(project_root=project.root, session=project.raw / sid if sid else None,
                                manifest=None if sid else project.selection)


def packet(project, req, name="attempt_0001", config=None):
    closure = {}
    for stage in STAGES:
        path = project.root / "code" / (stage + ".py")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stage)
        closure[stage] = [file_ref(path)]
    output = project.root / name
    output.mkdir()
    request_path = output / "REQUEST.json"
    save(request_path, req)
    value = {"task_id": "synthetic_automation", "project_root": str(project.root), "output_dir": str(output),
        "request_path": str(request_path), "run_signature": "a" * 64, "writer": {"fencing_token": "synthetic-fence"},
        "component_code": closure, "component_weights": {}, "component_config": config or {}, "reuse_stages": {}}
    value["component_signatures"] = auto.component_signatures(req, closure, {}, config)
    return value


def authority(value):
    return lambda: {"task_id": value["task_id"], "run_signature": value["run_signature"],
        "fencing_token": value["writer"]["fencing_token"], "execution_allowed": True, "algorithm_execution_allowed": True}


def synthetic_builders(calls):
    def factory(stage):
        def build(*, row, output_dir, signature, upstream, request):
            calls.append((row["session_id"], stage))
            output_dir.mkdir(parents=True)
            extras = {}
            if stage == "Source":
                rgb = np.zeros((row["native_frames"], 480, 640, 3), np.uint8)
                with (output_dir / "RAW_RGB.npz").open("xb") as stream:
                    np.savez_compressed(stream, rgb=rgb)
                mapping = [{"session_id": row["session_id"], "native_frame_index": i, "decoded_display_index": i,
                    "model_input_frame": i, "output_frame": i, "qpc_declared_ns": (i + 1) * 40_000_000,
                    "pixel_sha256": hashlib.sha256(rgb[i].tobytes()).hexdigest()} for i in range(row["native_frames"])]
                save(output_dir / "FRAME_MAP.json", mapping)
                save(output_dir / "SOURCE_REPORT.json", {"session_id": row["session_id"],
                    "native_sidecar_frames": row["native_frames"], "actual_inputs": row["raw_inputs"]})
                extras["actual_inputs"] = row["raw_inputs"]
            else:
                (output_dir / "synthetic.txt").write_text(stage)
            outputs = [file_ref(p) for p in sorted(output_dir.iterdir())]
            receipt = {"schema": SCHEMA, "status": "SEALED_COMPLETE", "stage": stage,
                "session_id": row["session_id"], "native_frames": row["native_frames"], "component_signature": signature,
                "upstream": {p: upstream[p] for p in auto.DEPENDENCIES[stage]}, "outputs": outputs,
                "robot_video_produced": False, **extras}
            save(output_dir / "RECEIPT.json", receipt)
        return build
    return {stage: factory(stage) for stage in STAGES}


def test_real_native_counts_are_not_fixed001_238(project):
    req = request(project)
    assert [(r["session_id"], r["native_frames"]) for r in req["sessions"]] == [("050", 3), ("007", 5)]
    assert req["status"] == "FULL_FLOW_REQUEST_NOT_EXECUTED"
    assert req["producer_policy"] == auto.POLICY


@pytest.mark.parametrize("field", ["historical_mask", "historical_clean", "actions", "manual_placement"])
def test_historical_or_manual_producer_fields_rejected(project, field):
    req = request(project, "050")
    req["sessions"][0][field] = {"path": "historical"}
    payload = {k: v for k, v in req.items() if k not in {"submission_input_sha256", "status", "robot_video_produced",
        "training_eligible", "control_ground_truth", "candidate_adopted"}}
    req["submission_input_sha256"] = canonical_sha(payload)
    with pytest.raises(auto.AutomationError, match="FORBIDDEN_INPUT"):
        auto.verify_v1_request(req, project_root=project.root)


def test_raw_selection_cannot_supply_configuration(project):
    value = json.loads(project.selection.read_text())
    value["mask_seeds"] = "another"
    save(project.selection, value)
    with pytest.raises(auto.AutomationError, match="NO_TASK_CONFIG"):
        request(project)


def test_duplicate_session_rejected(project):
    save(project.selection, {"schema": "inpainting-raw-session-manifest-v1",
        "sessions": [str(project.raw / "050")] * 2})
    with pytest.raises(auto.AutomationError, match="PARENT_RAW_SESSIONS"):
        request(project)


def test_actual_qpc_count_drift_rejected(project):
    req = request(project, "050")
    Path(req["sessions"][0]["raw_inputs"]["video_qpc"]["path"]).write_text("1\n2")
    with pytest.raises(auto.AutomationError, match="INPUT_SHA_DRIFT"):
        auto.verify_v1_request(req, project_root=project.root)


def test_mask_edit_only_invalidates_mask_clean_composite(project):
    req = request(project)
    value = packet(project, req)
    original = value["component_signatures"]
    changed = auto.component_signatures(req, value["component_code"], stage_config={"Mask": {"recipe_change": "candidate"}})
    for sid in original:
        assert {s for s in STAGES if original[sid][s] != changed[sid][s]} == {"Mask", "Clean", "Composite"}


def test_one_submit_executes_all_stages_serially(project):
    req, calls = request(project), []
    value = packet(project, req)
    result = auto.execute_automation(value, authorization=authority(value), builders=synthetic_builders(calls))
    assert calls == [(sid, s) for sid in ("050", "007") for s in STAGES]
    assert result["status"] == "SEALED_COMPLETE"
    assert result["robot_video_produced"] is False  # Synthetic files are not a production claim.
    assert result["model_calls"] == result["render_calls"] == 0
    assert all(set(r["stage_receipts"]) == set(STAGES) for r in result["sessions"])


def test_sealed_same_signatures_reuse_without_second_producer(project):
    req = request(project, "007")
    first = packet(project, req)
    result = auto.execute_automation(first, authorization=authority(first), builders=synthetic_builders([]))
    second = packet(project, req, "attempt_0002")
    second["reuse_stages"] = {"007": result["sessions"][0]["stage_receipts"]}
    calls = []
    resumed = auto.execute_automation(second, authorization=authority(second), builders=synthetic_builders(calls))
    assert calls == []
    assert resumed["sessions"][0]["reused_stages"] == list(STAGES)


def test_partial_stage_cannot_resume(project):
    req = request(project, "050")
    value = packet(project, req)
    partial = save(project.root / "partial/RECEIPT.json", {"schema": SCHEMA, "stage": "Source", "status": "PARTIAL"})
    value["reuse_stages"] = {"050": {"Source": partial}}
    with pytest.raises(ValueError, match="PARTIAL_NOT_CONSUMABLE"):
        auto.execute_automation(value, authorization=authority(value), builders=synthetic_builders([]))


def test_existing_unsealed_output_rejects_second_writer(project):
    req = request(project, "050")
    value = packet(project, req)
    (Path(value["output_dir"]) / "sessions/050/Source").mkdir(parents=True)
    with pytest.raises(auto.AutomationError, match="SECOND_WRITER_OUTPUT_CONFLICT"):
        auto.execute_automation(value, authorization=authority(value), builders=synthetic_builders([]))


def test_fence_drift_stops_before_next_stage(project):
    req, calls = request(project, "050"), []
    value = packet(project, req)
    checks = 0
    def gate():
        nonlocal checks
        checks += 1
        permit = authority(value)()
        if checks >= 3:
            permit["fencing_token"] = "another-writer"
        return permit
    with pytest.raises(auto.AutomationError, match="WRITER_FENCE_DRIFT"):
        auto.execute_automation(value, authorization=gate, builders=synthetic_builders(calls))
    assert calls == [("050", "Source")]


def test_cpu_packet_cannot_authorize_algorithms(project):
    req, calls = request(project, "050"), []
    value = packet(project, req)
    denied = authority(value)()
    denied["algorithm_execution_allowed"] = False
    with pytest.raises(auto.AutomationError, match="REGISTERED_EXECUTION_REQUIRED"):
        auto.execute_automation(value, authorization=lambda: denied, builders=synthetic_builders(calls))
    assert calls == []


def test_component_code_bytes_drift_rejected_before_builder(project):
    req = request(project, "050")
    value = packet(project, req)
    Path(value["component_code"]["Mask"][0]["path"]).write_text("drift")
    with pytest.raises(auto.AutomationError, match="INPUT_SHA_DRIFT"):
        auto.execute_automation(value, authorization=authority(value), builders=synthetic_builders([]))


def test_side_unknown_or_nonunique_glove_not_promoted():
    bundle = {"instance_association": [{"role": "human", "track_id": 2, "side": "left",
        "association": "UNIQUE_MAX_SAME_CAPTURE_PROJECTION_HIT_RATE", "side_hit_rates": [1.0, 0.0]}]}
    assigned, unknown = associate_original_objects(bundle, [{"role": "glove", "side": "left", "obj_id": 1}])
    assert assigned[0]["obj_id"] == 1 and assigned[0]["identity_unique"] is True and not unknown
    assigned, unknown = associate_original_objects(bundle, [{"role": "glove", "side": "left", "obj_id": 1}] * 2)
    assert not assigned and unknown[0]["decision"] == "ORIGINAL_IDENTITY_UNKNOWN_NO_REPAIR"


def test_fresh_rgb_three_point_repair_preserves_original_support(monkeypatch):
    from chaoyang.inpainting_candidate.automation_mask import repair_current_masks
    from chaoyang.exact78_baseline import mask as parent_mask
    from chaoyang.pick_place_1008 import visual
    from chaoyang.inpainting_candidate.automatic_masks import PROMPTS
    shape = (3, 480, 640)
    rgb = np.zeros((*shape, 3), np.uint8)
    region = np.zeros(shape, bool)
    region[:, 10:40, 10:40] = True
    base = {key: np.zeros(shape, bool) for key in ("core", "sam_core", "equipment", "removal", "protected_visible")}
    base["core"][:, 10:15, 10:15] = True
    base["sam_core"][:] = base["core"]
    base["removal"][:] = base["core"]
    base["protected_visible"][:, 100:110, 100:110] = True
    base["metadata"] = {"removal": {"observed_segmentation_dilation_px": 0, "temporal_bracket_frames": 0}}
    hints = np.empty((2, 3, 1, 2), float)
    hints[0] = [20.0, 20.0]
    hints[1] = [120.0, 120.0]
    geometry = {k: hints.copy() for k in ("hands", "forearm", "tracker")}
    monkeypatch.setattr(parent_mask, "pico_geometry", lambda session, scale: geometry)
    monkeypatch.setattr(visual, "_gate_diagnostic_geometry", lambda value, session: (value, []))
    class FakeSam:
        aborted = []
        def track_text(self, frames, prompt, anchor):
            assert len(frames) == shape[0] and anchor == 0
            return {1: {i: np.packbits(region[i]) for i in range(3)}} if prompt == PROMPTS["human"] else {}
    observed = []
    def tracker(sam, pixels, prompts, anchor):
        observed.extend(prompts)
        assert len(prompts) == 1 and len(prompts[0]["points_px"]) == 3
        return {2: {i: np.packbits(region[i]) for i in range(3)}}, {"status": "COMPLETE", "errors": [], "aborted": []}
    before = deepcopy(base)
    candidate, audit, proposals, identity_hints = repair_current_masks(sam=FakeSam(), rgb=rgb,
        session=SimpleNamespace(width=640), original_masks=base,
        original_objects=[{"role": "glove", "side": "left", "obj_id": 2}],
        raw_source_sha256="a" * 64, frame_map_sha256="b" * 64, model_sha256="c" * 64, tracker=tracker)
    assert audit["repair_prompt_policy"] == "SPREAD_MAX_GAP_THREE"
    assert audit["manual_points_added"] == audit["original_removal_lost_pixels"] == audit["protected_mask_changed_pixels"] == 0
    assert np.array_equal(candidate["protected_visible"], before["protected_visible"])
    assert not np.any(before["core"] & ~candidate["core"])
    assert np.count_nonzero(candidate["core"] & ~before["core"]) > 0
    assert not np.any(candidate["core"] & ~region)
    assert len(observed) == len(proposals) == 1 and identity_hints.shape == (2, 3, 3, 2)


@pytest.mark.parametrize('wall_args', [[], ['--wall-seconds', '120']])
def test_cli_v1_mode_isolated_from_raw_auto(project, monkeypatch, capsys, wall_args):
    from chaoyang.cli import inpainting_submit
    from chaoyang import paths
    from chaoyang.governance import inpainting_candidate_registration as reg
    seen = []
    monkeypatch.setattr(paths, "get_paths", lambda: SimpleNamespace(repo_root=project.root))
    monkeypatch.setattr(reg, "submit_v1_compatible_request", lambda req, **kw: seen.append((req, kw)) or {"status": "PREVIEW"}, raising=False)
    monkeypatch.setattr(reg, "submit_request", lambda *a, **kw: pytest.fail("raw-auto route called"))
    assert inpainting_submit.main(["--mode", "v1-compatible", "--manifest", str(project.selection), "--preview", *wall_args]) == 0
    assert seen[0][0]["mode"] == "v1-compatible" and seen[0][1]["preview"] is True
    if wall_args:
        assert seen[0][1]['wall_seconds'] == 120
    assert json.loads(capsys.readouterr().out)["status"] == "PREVIEW"


def test_raw_auto_still_default(monkeypatch, capsys):
    from chaoyang.cli import inpainting_submit
    from chaoyang.inpainting_candidate import entry
    from chaoyang.governance import inpainting_candidate_registration as reg
    seen = []
    monkeypatch.setattr(entry, "build_request", lambda **kw: seen.append(kw) or {"old_cpu": True})
    monkeypatch.setattr(reg, "submit_request", lambda req, **kw: {"status": "CPU_ADMISSION_REQUEST_ONLY"})
    assert inpainting_submit.main(["--session", "/path/to/private_storage/datasets/ego/pick_place_1008/050", "--preview"]) == 0
    assert seen[0]["mode"] == "raw-auto"
    assert json.loads(capsys.readouterr().out)["status"] == "CPU_ADMISSION_REQUEST_ONLY"


@pytest.mark.parametrize('mode,wall', [('raw-auto', '120'), ('v1-compatible', '0'), ('v1-compatible', '3601')])
def test_invalid_wall_or_raw_auto_rejected_before_raw_read(monkeypatch, mode, wall):
    from chaoyang.cli import inpainting_submit
    from chaoyang.inpainting_candidate import entry
    monkeypatch.setattr(entry, 'build_request', lambda **kwargs: pytest.fail('input read before option rejection'))
    with pytest.raises(SystemExit) as error:
        inpainting_submit.main(['--session', '/unread/raw', '--mode', mode, '--wall-seconds', wall])
    assert error.value.code == 2
