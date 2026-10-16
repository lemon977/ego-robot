"""Isolated native consumers: CPU fake models, no weights, GPU, IK or render.

Fake PP/codec providers emit real immutable receipts and arrays. The only
synthetic Source exemption is the fixture's enclosing _run path (tests are
required to stay there); all byte/map/signature checks remain real.
"""
from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import SETTINGS
from chaoyang.inpainting_candidate import clean_worker as clean
from chaoyang.inpainting_candidate import composite_worker as composite
from chaoyang.inpainting_candidate import native_contracts as nc
from chaoyang.inpainting_candidate.automatic_masks import frames_sha256
from chaoyang.inpainting_candidate.contracts import SCHEMA, canonical_sha, file_ref

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


model_runner = module("_native_model_runner_cpu", ROOT / "scripts/maintenance/inpainting_candidate_model_task_v1.py")
composite_runner = module("_native_composite_runner_cpu", ROOT / "scripts/maintenance/inpainting_candidate_composite_task_v1.py")
model_tests = module("_native_shared_mask_fixture", Path(__file__).with_name("test_inpainting_candidate_model_execution.py"))
source_worker_batch = model_tests.source_worker_batch


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_ref(path)


def arrays(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    return file_ref(path)


def seal(folder, stage, signature, outputs, n, **extra):
    value = {"schema": SCHEMA, "stage": stage, "status": "SEALED_COMPLETE",
        "component_signature": signature, "outputs": outputs, "native_frames": n, **extra}
    if stage in {"Mask", "Clean"}:
        value.pop("native_frames")
        value.update(native_input_frames=n, native_output_frames=n)
    return put(folder / "RECEIPT.json", value)


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-CPU-fake")
    monkeypatch.setattr(model_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(composite_runner, "REPO_ROOT", tmp_path)
    original_require = nc.require

    def fixture_require(condition, code):
        if code in {"NATIVE_HISTORICAL_RAW_INPUT_FORBIDDEN", "NATIVE_CLEAN_RAW_VIDEO_REQUIRED"}:
            # Synthetic fixture is deliberately under this Task's _run lane.
            return
        original_require(condition, code)

    monkeypatch.setattr(nc, "require", fixture_require)
    settings = tmp_path / "src/chaoyang/exact78_baseline/clean.py"
    settings.parent.mkdir(parents=True)
    settings.write_text("SETTINGS = " + repr(SETTINGS) + "\n")
    for name in ("composite.py", "occlusion.py", "contracts.py", "clean_references.py", "source_frames.py"):
        codepath = tmp_path / "src/chaoyang/inpainting_candidate" / name
        codepath.parent.mkdir(parents=True, exist_ok=True)
        codepath.write_bytes((ROOT / "src/chaoyang/inpainting_candidate" / name).read_bytes())
    code = put(tmp_path / "runtime.json", {"synthetic_cpu": True})
    weights = [put(tmp_path / "models" / name, {"never_loaded": name}) for name in
        ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")]
    calls = {"load": 0, "pp": [], "video": [], "composite": []}

    def source(key, n, *, full_size=False, size=(8, 8)):
        sid, shape = "raw_" + key, (480, 640) if full_size else size
        rawdir = tmp_path / "synthetic_raw" / key
        video = put(rawdir / "vst.h264", {"synthetic": True})
        qpc_path = rawdir / "qpc.jsonl"
        qpc_path.write_text("\n".join(str(10000 + i * 40000) for i in range(n)))
        actual = {"video": video, "video_qpc": file_ref(qpc_path),
            "video_wall": put(rawdir / "wall.jsonl", {}), "manifest": put(rawdir / "manifest.json", {})}
        mapping = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
            "output_frame": i, "qpc_declared_ns": 10000 + i * 40000} for i in range(n)]
        raw = np.full((n, *shape, 3), 17, np.uint8)
        for index, frame in enumerate(mapping):
            frame["pixel_sha256"] = hashlib.sha256(np.ascontiguousarray(raw[index]).data).hexdigest()
        folder = tmp_path / "Source" / key
        outputs = [arrays(folder / "RAW_RGB.npz", rgb=raw), put(folder / "FRAME_MAP.json", mapping),
            put(folder / "SOURCE_REPORT.json", {"session_id": sid, "native_sidecar_frames": n,
                "actual_inputs": actual, "timebase_status": "TIMEBASE_UNVERIFIED"})]
        ref = seal(folder, "Source", "a" * 64, outputs, n, session_id=sid,
            timebase_status="TIMEBASE_UNVERIFIED", actual_inputs=actual)
        row = {"session_key": key, "session_id": sid, "native_frames": n, "upstream_status": "READY",
            "source_receipt": ref, "source_signature": "a" * 64, "frame_map_ref": outputs[1],
            "raw_source_ref": video}
        maskfolder = tmp_path / "Mask" / key
        e = np.zeros((n, *shape), bool)
        e[:, 3, 3] = True
        p = np.zeros_like(e)
        p[:, 0, 0] = True
        masks = {"removal": e, "effective_removal": e, "protected_visible": p,
            "core": e, "human": e, "equipment": np.zeros_like(e), "object_observed": p,
            "unknown": np.zeros_like(e), "held": np.zeros_like(e)}
        maskref = seal(maskfolder, "Mask", "b" * 64,
            [arrays(maskfolder / "MASK_ARRAYS.npz", **masks), put(maskfolder / "FRAME_MAP.json", mapping),
             put(maskfolder / "FRAME_STATUS.json", [{"native_frame_index": i} for i in range(n)])], n,
            session_id=sid, actual_inputs={"raw_source": video, "frames_sha256": frames_sha256(raw),
                "frame_map_sha256": canonical_sha(mapping)})
        row.update(mask_receipt=maskref, mask_signature="b" * 64, clean_signature="c" * 64)
        return SimpleNamespace(row=row, raw=raw, mapping=mapping, masks=masks)

    def blocked(key, n):
        diagnostic = put(tmp_path / (key + "_BLOCKED.json"), {"status": "BLOCKED",
            "session_key": key, "session_id": "raw_" + key, "native_frames": n, "failure_code": "MISSING_UPSTREAM"})
        return {"session_key": key, "session_id": "raw_" + key, "native_frames": n,
            "upstream_status": "BLOCKED", "failure_code": "MISSING_UPSTREAM", "diagnostic_ref": diagnostic}

    def packet(rows, *, kind="Clean"):
        isclean = kind == "Clean"
        scope = nc.NATIVE_CLEAN_SCOPE if isclean else nc.NATIVE_COMPOSITE_SCOPE
        value = {"task_id": "inpainting_native_" + kind.lower() + "_fixture", "run_signature": "9" * 64,
            "spec_schema": nc.NATIVE_CLEAN_SCHEMA if isclean else nc.NATIVE_COMPOSITE_SCHEMA,
            "experiment_scope": scope, "actual_stage_scope": scope,
            "operation": "run_inpainting_candidate_visual_stages_v1" if isclean else "run_inpainting_candidate_composite_stage_v1",
            "output_dir": str(tmp_path / (kind + "_out")), "runtime_code": [code],
            "inputs": {}, "schema": {}, "calibration": {"scene_depth": "ABSENT"},
            "config": {"stage": "NATIVE_BASELINE_CLEAN"} if isclean else
                {"scene_depth": "ABSENT", "display_policy": "RAW_MODEL_OBJECT_FIRST_DIAGNOSTIC",
                 "image_size": [640, 480], "complete_robot": True},
            "budgets": {"gpu_count": int(isclean), "cpu_threads_per_worker": 2, "candidates_max": 1,
                "training_steps_max": 0, "wall_seconds_max": 7200, "render_calls_max": 0,
                "model_calls_max": sum(r["upstream_status"] == "READY" for r in rows) if isclean else 0},
            "sessions": rows, "writer": {"fencing_token": "cpu-fake"}}
        if isclean:
            value.update(model_stages=["Clean"], model_weights=weights,
                clean_config={"frozen_settings": dict(SETTINGS), "reference_policy": "FROZEN_BASELINE", "candidates": ["A"]})
            value["budgets"]["model_calls_unit"] = "PROPAINTER_INPAINT_API_INVOCATION"
        else:
            value["stages"] = ["Composite"]
        Path(value["output_dir"]).mkdir()
        return value

    owner = {"task_id": "", "run_signature": "9" * 64, "fencing_token": "cpu-fake",
        "execution_allowed": True, "algorithm_execution_allowed": True,
        "gpu_uuid": "GPU-CPU-fake", "worker_logical_index": 0}

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        assert not kwargs["need_sam"] and kwargs["need_clean"]
        return {"propainter": object(), "load_receipt": {"synthetic_cpu": True}}

    def fake_pp(rgb, *, masks, output_dir, raw_source_ref, frame_map, model_refs,
                component_signature, mask_receipt_ref, authorization, **options):
        assert options["reference_selector"] is None and options["clean_context_refs"] is None
        assert options["record_reference_selection"] is False
        assert np.array_equal(options["sealed_effective_removal"], masks["effective_removal"])
        calls["pp"].append(len(rgb))
        folder = Path(output_dir)
        folder.mkdir(parents=True)
        fixed = rgb.copy()
        fixed[masks["effective_removal"]] = 91
        mask_stage = json.loads(Path(mask_receipt_ref["path"]).read_text())
        mask_arrays = next(ref for ref in mask_stage["outputs"]
            if Path(ref["path"]).name == "MASK_ARRAYS.npz")
        ref = seal(folder, "Clean", component_signature,
            [arrays(folder / "CLEAN_ARRAYS.npz", clean_rgb=fixed,
                effective_removal=masks["effective_removal"], protected_visible=masks["protected_visible"]),
             put(folder / "FRAME_MAP.json", frame_map),
             put(folder / "FRAME_STATUS.json", [{"native_frame_index": i} for i in range(len(rgb))])], len(rgb),
            settings=dict(SETTINGS), reference_mode="FROZEN_BASELINE", reference_selection_diagnostic_hook=False,
            authority=authorization(), actual_inputs={"raw_source": raw_source_ref, "mask_receipt": mask_receipt_ref,
                "models": model_refs, "clean_context_refs": {}, "frames_sha256": frames_sha256(rgb),
                "frame_map_sha256": canonical_sha(frame_map), "effective_removal_source": {
                    "mask_arrays": mask_arrays, "array_key": "effective_removal",
                    "mode": "EXACT_SEALED_SUBSET_OF_FROZEN_DILATION"}})
        return json.loads(Path(ref["path"]).read_text())

    def fake_video(path, rgb, mapping):
        calls["video"].append(len(rgb))
        return {"video": put(Path(path), {"fake_codec_frames": len(rgb)}), "decoded_frames": len(rgb)}

    monkeypatch.setattr(clean, "load_frozen_models", fake_load)
    monkeypatch.setattr(clean, "run_clean_stage", fake_pp)
    monkeypatch.setattr(clean, "write_review_video", fake_video)
    return SimpleNamespace(root=tmp_path, source=source, blocked=blocked, packet=packet,
        calls=calls, owner=owner, authority=lambda **kwargs: dict(owner))


def test_dynamic_clean_3_5_frames_and_local_blocked_no_source(native):
    f = native
    samples = [f.source("one", 3), f.source("two", 5)]
    packet = f.packet([samples[0].row, f.blocked("blocked", 7), samples[1].row])
    f.owner["task_id"] = packet["task_id"]
    result = clean.run_clean_batch(packet, authority=f.authority)
    assert f.calls["load"] == 1 and f.calls["pp"] == f.calls["video"] == [3, 5]
    assert result["completed_clean_sessions"] == 2 and result["blocked_sessions"] == 1
    assert result["model_call_accounting"]["attempted_api_calls"] == 2
    assert result["model_call_accounting"]["completed_api_calls"] == 2
    clean.validate_clean_result(packet, result, decoder=lambda path: json.loads(path.read_text())["fake_codec_frames"])
    assert model_runner.planned_session_keys(packet) == ["one", "blocked", "two"]


def test_all_blocked_clean_zero_models_not_success(native):
    f = native
    packet = f.packet([f.blocked("missing", 7)])
    f.owner["task_id"] = packet["task_id"]
    result = clean.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 0 and not result["execution_diagnostic_complete"]
    assert f.calls["load"] == 0 and f.calls["pp"] == []
    with pytest.raises(ValueError, match="ACCOUNTING"):
        clean.validate_clean_result(packet, result, decoder=lambda p: 7)


@pytest.mark.parametrize("kind", ["source_count", "source_identity", "raw_video", "mask_signature", "partial", "mask_map", "native_map", "qpc", "seed"])
def test_native_clean_corrupt_sources_rejected_before_load(native, kind):
    f = native
    sample = f.source("one", 3)
    row = sample.row
    packet = f.packet([row])
    f.owner["task_id"] = packet["task_id"]
    if kind == "source_count": row["native_frames"] = 2
    elif kind == "source_identity": row["session_id"] = "different"
    elif kind == "raw_video": row["raw_source_ref"] = put(f.root / "other_raw.h264", {})
    elif kind == "mask_signature": row["mask_signature"] = "e" * 64
    elif kind == "seed": row["manual_points"] = [[1, 2]]
    elif kind == "qpc":
        path = f.root / "synthetic_raw/one/qpc.jsonl"
        path.write_text("10000\n10000\n90000")
    else:
        field = "source_receipt" if kind == "native_map" else "mask_receipt"
        ref = row[field]
        value = json.loads(Path(ref["path"]).read_text())
        if kind == "partial": value["status"] = "PARTIAL"
        else:
            mp = next(r for r in value["outputs"] if Path(r["path"]).name == "FRAME_MAP.json")
            mapped = json.loads(Path(mp["path"]).read_text())
            mapped[1]["output_frame"] = 0
            replacement = put(Path(mp["path"]), mapped)
            value["outputs"] = [replacement if r == mp else r for r in value["outputs"]]
            if kind == "native_map": row["frame_map_ref"] = replacement
        row[field] = put(Path(ref["path"]), value)
    with pytest.raises(ValueError):
        clean.run_clean_batch(packet, authority=f.authority)
    assert f.calls["load"] == 0 and not f.calls["pp"]


@pytest.mark.parametrize("kind", ["decode", "mask_binding", "protected", "completion", "calls", "adoption",
    "effective_source_missing", "effective_source_ref", "effective_source_key", "effective_source_mode"])
def test_native_clean_final_checks_real_receipts_and_arrays(native, kind):
    f = native
    packet = f.packet([f.source("one", 3).row])
    f.owner["task_id"] = packet["task_id"]
    result = clean.run_clean_batch(packet, authority=f.authority)
    decoder = lambda p: 3
    if kind == "decode": decoder = lambda p: 2
    elif kind == "completion": result["completed_clean_sessions"] = 2
    elif kind == "calls": result["model_call_accounting"]["completed_api_calls"] = 0
    elif kind == "adoption": result["candidate_adopted"] = True
    else:
        row = result["sessions"][0]
        ref = row["clean_receipt"]
        value = json.loads(Path(ref["path"]).read_text())
        if kind == "mask_binding": value["actual_inputs"]["mask_receipt"] = {}
        elif kind == "effective_source_missing": value["actual_inputs"].pop("effective_removal_source")
        elif kind.startswith("effective_source_"):
            binding = value["actual_inputs"]["effective_removal_source"]
            key = {"effective_source_ref": "mask_arrays", "effective_source_key": "array_key",
                "effective_source_mode": "mode"}[kind]
            binding[key] = {} if key == "mask_arrays" else "changed"
        else:
            ar = next(r for r in value["outputs"] if Path(r["path"]).name == "CLEAN_ARRAYS.npz")
            with np.load(ar["path"]) as data: payload = {k: data[k] for k in data.files}
            payload["protected_visible"][:] = False
            replacement = arrays(Path(ar["path"]), **payload)
            value["outputs"] = [replacement if r == ar else r for r in value["outputs"]]
        row["clean"] = row["clean_receipt"] = put(Path(ref["path"]), value)
    with pytest.raises(ValueError):
        clean.validate_clean_result(packet, result, decoder=decoder)


@pytest.mark.parametrize("prefix", ["inpainting_native_clean_", "inpainting_post_g1_source_aware_"])
@pytest.mark.parametrize("old_field", ["resume_context", "original_g1_deadline_at"])
def test_new_consumers_do_not_revive_old_timebox(prefix, old_field):
    spec = {"task_id": prefix + "new", "budgets": {"wall_seconds_max": 7200}, old_field: None}
    with pytest.raises(ValueError, match="FRESH_TIMEBOX"):
        model_runner.effective_deadline(spec, datetime.now(timezone.utc))


def test_blocked_runner_does_not_require_fake_source():
    seen = []
    row = {"upstream_status": "BLOCKED", "diagnostic_ref": {"path": "synthetic"}}
    model_runner.verify_session_inputs(SimpleNamespace(_verify=lambda ref: seen.append(ref)), row, clean=True)
    assert seen == [row["diagnostic_ref"]]


@pytest.mark.parametrize("enabled", [False, True])
def test_source_aware_worker_opt_in_receipt_and_default_unchanged(source_worker_batch, enabled):
    from chaoyang.inpainting_candidate import model_worker
    fixture = source_worker_batch
    if enabled:
        fixture.packet.setdefault("config", {})["source_aware_retention_veto"] = "SOURCE_AWARE_POINT_WEARER_V1"
        fixture.packet["experiment_scope"] = "POST_G1_SOURCE_AWARE_RETENTION_VETO"
        put(Path(fixture.packet["packet_path"]), fixture.packet)
    result = model_worker.run(fixture.packet)
    assert result["completed_mask_sessions"] == 6
    assert result["model_call_accounting"]["attempted_api_calls"] == 36
    for row in result["sessions"]:
        actual = json.loads(Path(row["mask"]["path"]).read_text())["actual_inputs"]
        if enabled:
            assert actual["source_aware_retention_veto"] == "SOURCE_AWARE_POINT_WEARER_V1"
            assert row["experiment_scope"] == fixture.packet["experiment_scope"]
        else:
            assert "source_aware_retention_veto" not in actual
            assert "experiment_scope" not in row


def test_unknown_source_aware_flag_rejected_before_provider(source_worker_batch):
    from chaoyang.inpainting_candidate import model_worker
    fixture = source_worker_batch
    fixture.packet.setdefault("config", {})["source_aware_retention_veto"] = "UNKNOWN"
    put(Path(fixture.packet["packet_path"]), fixture.packet)
    with pytest.raises(ValueError, match="SOURCE_AWARE"):
        model_worker.run(fixture.packet)
    assert not fixture.sam.calls


def composite_input(f, key, n):
    sample = f.source(key, n, full_size=True)
    row = sample.row
    raw = sample.raw
    folder = f.root / "preclean" / key
    fixed = raw.copy()
    fixed[sample.masks["effective_removal"]] = 91
    row["clean_receipt"] = seal(folder, "Clean", row["clean_signature"],
        [arrays(folder / "CLEAN_ARRAYS.npz", clean_rgb=fixed,
            effective_removal=sample.masks["effective_removal"], protected_visible=sample.masks["protected_visible"]),
         put(folder / "FRAME_MAP.json", sample.mapping),
         put(folder / "FRAME_STATUS.json", [{"native_frame_index": i} for i in range(n)])], n,
        actual_inputs={"raw_source": row["raw_source_ref"], "mask_receipt": row["mask_receipt"],
            "frames_sha256": frames_sha256(raw), "frame_map_sha256": canonical_sha(sample.mapping)})
    folder = f.root / "prerender" / key
    rgba = np.zeros((*raw.shape[:3], 4), np.uint8)
    labels = np.full(raw.shape[:3], -1, np.int32)
    rgba[:, 1, 1] = [1, 2, 3, 255]
    labels[:, 1, 1] = 2000
    depth = np.full(raw.shape[:3], np.inf, np.float32)
    depth[:, 1, 1] = 0.7
    row.update(render_signature="d" * 64, composite_signature="e" * 64)
    row["render_receipt"] = seal(folder, "Render", row["render_signature"],
        [arrays(folder / "ROBOT_BUFFERS.npz", rgba=rgba, link_id=labels, depth_m=depth,
            qpc_ns=np.asarray([m["qpc_declared_ns"] for m in sample.mapping], np.int64)),
         put(folder / "FRAME_MAP.json", sample.mapping),
         put(folder / "FRAME_STATUS.json", [{"native_frame_index": i} for i in range(n)]),
         put(folder / "LABEL_IDENTITY.json", {"2000": {"instance_id": "left_hand", "part_id": "palm", "link_id": "base_link"}})], n,
        source_receipt=row["source_receipt"], source_signature=row["source_signature"], complete_robot=True,
        session_id=row["session_id"])
    row["upstream_quality_evidence"] = {name: put(f.root / key / (name + "_QUALITY.json"),
        {"schema": "inpainting-upstream-quality-status-v1", "stage": name,
         "upstream_receipt": row[name.lower() + "_receipt"], "visual_quality": "FAILED",
         "human_review": "PENDING", "hidden_regions": "UNKNOWN",
         "review_reference": put(f.root / key / (name + "_REVIEW.json"), {"synthetic_cpu": True})})
         for name in ("Mask", "Clean")}
    return sample


@pytest.mark.parametrize("quality_keys", [("Mask", "Clean"), ("Mask",), ()])
def test_native_composite_reuses_producer_dynamic_n_with_local_block(native, monkeypatch, quality_keys):
    from chaoyang.inpainting_candidate import composite as producer
    f = native
    monkeypatch.setattr(producer, "PROJECT_ROOT", f.root)
    samples = [composite_input(f, "one", 2), composite_input(f, "two", 3)]
    for sample in samples:
        for name in ("Mask", "Clean"):
            if name not in quality_keys:
                reference = sample.row[name.lower() + "_receipt"]
                stage = json.loads(Path(reference["path"]).read_text())
                stage["visual_quality"] = "NOT_REVIEWED"
                replacement = put(Path(reference["path"]), stage)
                sample.row[name.lower() + "_receipt"] = replacement
                if name == "Mask":
                    old_clean = sample.row["clean_receipt"]
                    clean_stage = json.loads(Path(old_clean["path"]).read_text())
                    clean_stage["actual_inputs"]["mask_receipt"] = replacement
                    sample.row["clean_receipt"] = put(Path(old_clean["path"]), clean_stage)
        sample.row["upstream_quality_evidence"] = {k: v for k, v in
            sample.row["upstream_quality_evidence"].items() if k in quality_keys}
    packet = f.packet([samples[0].row, f.blocked("missing", 9), samples[1].row], kind="Composite")
    f.owner["task_id"] = packet["task_id"]

    def fake_codec(path, rgb, mapping, **kwargs):
        Path(path).write_text(json.dumps({"fake_codec_frames": len(rgb)}))
        return {"video": file_ref(path), "decoded_frames": len(rgb), "native_frames": len(rgb)}

    monkeypatch.setattr(producer, "_write_review_video", fake_codec)
    result = composite.run_native_composite_batch(packet, authority=f.authority, repo_root=f.root)
    assert result["completed_composite_sessions"] == 2 and result["blocked_sessions"] == 1
    assert result["native_frames"] == 5 and result["planned_native_frames"] == 14
    assert result["model_calls"] == result["render_calls"] == 0
    for row in result["sessions"]:
        if row["execution"] == "COMPLETE_COMPOSITE":
            for name in ("Mask", "Clean"):
                assert row[name.lower() + "_quality"] == ("FAILED" if name in quality_keys else "NOT_REVIEWED")
    composite_runner.validate_composite_result(packet,
        decoder=lambda p: json.loads(p.read_text())["fake_codec_frames"])


def test_all_blocked_composite_never_imports_or_calls_builder(native, monkeypatch):
    from chaoyang.inpainting_candidate import composite as producer
    f = native
    packet = f.packet([f.blocked("missing", 9)], kind="Composite")
    monkeypatch.setattr(producer, "build_composite_stage", lambda **kw: pytest.fail("BLOCKED producer call"))
    result = composite.run_native_composite_batch(packet, authority=f.authority, repo_root=f.root)
    assert result["completed_composite_sessions"] == 0 and not result["execution_diagnostic_complete"]
    with pytest.raises(ValueError, match="COMPLETENESS"):
        composite_runner.validate_composite_result(packet, decoder=lambda p: 9)


@pytest.mark.parametrize("kind", ["clean_mask", "partial", "map", "source", "quality"])
def test_native_composite_bad_binding_rejected_before_producer(native, monkeypatch, kind):
    from chaoyang.inpainting_candidate import composite as producer
    f = native
    sample = composite_input(f, "one", 2)
    row = sample.row
    packet = f.packet([row], kind="Composite")
    ref = row["clean_receipt"]
    value = json.loads(Path(ref["path"]).read_text())
    if kind == "clean_mask": value["actual_inputs"]["mask_receipt"] = {}
    elif kind == "partial": value["status"] = "PARTIAL"
    elif kind == "source": row["session_id"] = "swapped"
    elif kind == "quality":
        q = row["upstream_quality_evidence"]["Clean"]
        row["upstream_quality_evidence"]["Clean"] = put(Path(q["path"]), {"visual_quality": "PASS"})
    else:
        mp = next(r for r in value["outputs"] if Path(r["path"]).name == "FRAME_MAP.json")
        replacement = put(Path(mp["path"]), sample.mapping[:1])
        value["outputs"] = [replacement if r == mp else r for r in value["outputs"]]
    row["clean_receipt"] = put(Path(ref["path"]), value)
    monkeypatch.setattr(producer, "build_composite_stage", lambda **kw: pytest.fail("invalid producer call"))
    with pytest.raises(ValueError):
        composite.run_native_composite_batch(packet, authority=f.authority, repo_root=f.root)


def cpu_fake_pp_parent():
    """Real Torch CPU tensor path; providers have no learned model or weights."""
    import torch
    calls = {"raft": 0, "flow": 0, "propagate": 0, "generate": 0, "masks": []}

    class CPUTorch:
        cuda = SimpleNamespace(empty_cache=lambda: None, synchronize=lambda *a, **k: None)

        def __getattr__(self, name):
            return getattr(torch, name)

    class Flow:
        def forward_bidirect_flow(self, flows, masks):
            calls["flow"] += 1
            calls["masks"].append(masks.detach().cpu().numpy().copy())
            return flows, None

        def combine_flow(self, original, predicted, masks):
            return predicted

    class Generator:
        def img_propagation(self, masked, flows, masks, mode):
            calls["propagate"] += 1
            assert mode == "nearest" and masked.device.type == "cpu"
            calls["masks"].append(masks.detach().cpu().numpy().copy())
            return masked.flatten(0, 1), torch.zeros_like(masks).flatten(0, 1)

        def __call__(self, frames, flows, masks, updated, neighbors):
            calls["generate"] += 1
            calls["masks"].append(masks.detach().cpu().numpy().copy())
            return torch.zeros_like(frames[:, :neighbors])

    def raft(frames, iters):
        calls["raft"] += 1
        assert frames.device.type == "cpu" and iters == SETTINGS["raft_iter"]
        shape = (1, frames.shape[1] - 1, 2, *frames.shape[-2:])
        return torch.zeros(shape), torch.zeros(shape)

    return SimpleNamespace(settings=dict(SETTINGS), torch=CPUTorch(), device=torch.device("cpu"),
        raft=raft, flow=Flow(), model=Generator()), calls


def nonempty_veto_fixture(f):
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    sample = f.source("veto", 3, size=(16, 16))
    reference = sample.row["mask_receipt"]
    stage = json.loads(Path(reference["path"]).read_text())
    mask_ref = next(r for r in stage["outputs"] if Path(r["path"]).name == "MASK_ARRAYS.npz")
    sample.masks["retention_veto"] = np.zeros_like(sample.masks["effective_removal"])
    sample.masks["retention_veto"][:, 3, 4] = True
    baseline, _ = effective_clean_masks(sample.masks["removal"], sample.masks["protected_visible"])
    sample.masks["effective_removal"] = baseline & ~sample.masks["retention_veto"]
    replacement = arrays(Path(mask_ref["path"]), **sample.masks)
    stage["outputs"] = [replacement if r == mask_ref else r for r in stage["outputs"]]
    sample.row["mask_receipt"] = put(Path(reference["path"]), stage)
    return sample


def test_native_clean_real_stage_cpu_fake_pp_preserves_nonempty_veto_and_p(native, monkeypatch):
    from chaoyang.inpainting_candidate import model_stages as stages
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    f = native
    sample = nonempty_veto_fixture(f)
    packet = f.packet([sample.row])
    f.owner["task_id"] = packet["task_id"]
    parent, calls = cpu_fake_pp_parent()
    monkeypatch.setattr(stages, "PROJECT_ROOT", f.root)
    monkeypatch.setattr(clean, "run_clean_stage", stages.run_clean_stage)
    monkeypatch.setattr(clean, "load_frozen_models", lambda *a, **k:
        {"propainter": parent, "load_receipt": {"synthetic_cpu_no_weights": True}})
    result = clean.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 1, result
    stage_ref = result["sessions"][0]["clean_receipt"]
    stage = json.loads(Path(stage_ref["path"]).read_text())
    output = next(r for r in stage["outputs"] if Path(r["path"]).name == "CLEAN_ARRAYS.npz")
    with np.load(output["path"]) as archive:
        e, p, rgb = archive["effective_removal"], archive["protected_visible"], archive["clean_rgb"]
    reexpanded, _ = effective_clean_masks(sample.masks["removal"], sample.masks["protected_visible"])
    assert np.any(reexpanded & sample.masks["retention_veto"])
    assert np.count_nonzero(reexpanded ^ e) > 0  # Test would catch the old recompute path.
    assert np.array_equal(e, sample.masks["effective_removal"])
    assert np.array_equal(p, sample.masks["protected_visible"])
    assert np.array_equal(rgb[p], sample.raw[p])
    assert np.array_equal(rgb[~e], sample.raw[~e])
    assert not np.any(e & sample.masks["retention_veto"])
    assert calls["raft"] == calls["flow"] == calls["propagate"] == calls["generate"] == 1
    assert all(np.array_equal(value[0, :, 0].astype(bool), e) for value in calls["masks"])
    assert parent.settings == SETTINGS and stage["settings"] == SETTINGS
    assert stage["reference_mode"] == "FROZEN_BASELINE"
    clean.validate_clean_result(packet, result, decoder=lambda path: 3)


@pytest.mark.parametrize("bad", ["dtype", "shape", "protected", "outside_dilation"])
def test_native_bad_sealed_e_rejected_before_model_load(native, bad):
    f = native
    sample = nonempty_veto_fixture(f)
    provided = sample.masks["effective_removal"].copy()
    if bad == "dtype": provided = provided.astype(np.uint8)
    elif bad == "shape": provided = provided[:-1]
    else:
        point = (0, 0) if bad == "protected" else (14, 14)
        provided[:, point[0], point[1]] = True
    ref = sample.row["mask_receipt"]
    stage = json.loads(Path(ref["path"]).read_text())
    array_ref = next(r for r in stage["outputs"] if Path(r["path"]).name == "MASK_ARRAYS.npz")
    sample.masks["effective_removal"] = provided
    replacement = arrays(Path(array_ref["path"]), **sample.masks)
    stage["outputs"] = [replacement if r == array_ref else r for r in stage["outputs"]]
    sample.row["mask_receipt"] = put(Path(ref["path"]), stage)
    packet = f.packet([sample.row])
    f.owner["task_id"] = packet["task_id"]
    result = clean.run_clean_batch(packet, authority=f.authority)
    assert result["completed_clean_sessions"] == 0 and result["failed_sessions"] == 1
    assert result["model_call_accounting"]["attempted_api_calls"] == 0
    assert f.calls["load"] == 0 and f.calls["pp"] == []
    assert "SEALED" in result["sessions"][0]["failure"]


@pytest.mark.parametrize("bad", ["dtype", "shape", "changed", "protected", "outside_dilation"])
def test_sealed_clean_wrong_e_rejected_before_any_cpu_fake_provider(native, monkeypatch, bad):
    from chaoyang.inpainting_candidate import model_stages as stages
    f = native
    sample = nonempty_veto_fixture(f)
    packet = f.packet([sample.row])
    f.owner["task_id"] = packet["task_id"]
    parent, calls = cpu_fake_pp_parent()
    monkeypatch.setattr(stages, "PROJECT_ROOT", f.root)
    provided = sample.masks["effective_removal"].copy()
    if bad == "dtype": provided = provided.astype(np.uint8)
    elif bad == "shape": provided = provided[:-1]
    elif bad == "changed": provided[:, 3, 4] = True
    else:
        point = (0, 0) if bad == "protected" else (14, 14)
        provided[:, point[0], point[1]] = True
        ref = sample.row["mask_receipt"]
        upstream = json.loads(Path(ref["path"]).read_text())
        array_ref = next(r for r in upstream["outputs"] if Path(r["path"]).name == "MASK_ARRAYS.npz")
        sample.masks["effective_removal"] = provided
        replacement = arrays(Path(array_ref["path"]), **sample.masks)
        upstream["outputs"] = [replacement if r == array_ref else r for r in upstream["outputs"]]
        sample.row["mask_receipt"] = put(Path(ref["path"]), upstream)
    with pytest.raises(ValueError, match="CLEAN_SEALED_EFFECTIVE"):
        stages.run_clean_stage(sample.raw, masks=sample.masks, mask_receipt_ref=sample.row["mask_receipt"],
            parent_propainter=parent, output_dir=f.root / "wrong_effective",
            raw_source_ref=sample.row["raw_source_ref"], frame_map=sample.mapping,
            model_refs=packet["model_weights"], component_signature=sample.row["clean_signature"],
            authorization=f.authority, sealed_effective_removal=provided)
    assert calls["raft"] == calls["flow"] == calls["propagate"] == calls["generate"] == 0
