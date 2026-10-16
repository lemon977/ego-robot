"""Synthetic CPU evidence only: no 199-session preflight, GPU or model calls."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from chaoyang.humanego_ict.data import TrainingDataError, VerifiedInputCache, _npz
from chaoyang.humanego_ict.production_data import ProductionDerivedICTDataset
from chaoyang.humanego_ict.production_rgb_cache_v1 import (
    MAX_DECODED_RGB_CACHE_BYTES, RGB_PAYLOAD_BYTES, ResidentRGBProductionDatasetV1,
    VerifiedDecodedRGBCacheV1,
)


def ref(path):
    encoded = path.read_bytes()
    return {"path": str(path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def png(path, shift=0, mode="RGB"):
    pixels = ((np.arange(19 * 27 * 3).reshape(19, 27, 3) + shift) % 256).astype(np.uint8)
    image = Image.fromarray(pixels)
    if mode != "RGB":
        image = image.convert(mode)
    image.save(path, format="PNG")
    return ref(path)


@pytest.fixture
def original(tmp_path):
    # Build only the exact already-admitted reader state, not a fake production
    # admission receipt. The fixture cannot be submitted through a real CLI.
    image = png(tmp_path / "rgb.png")
    manifest = tmp_path / "fixture_manifest.json"
    manifest.write_text(json.dumps({"fixture_only": True}))
    named = {}
    for role in ("shared", "session", "evidence", "trace", "receipt"):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"fixture_only": True, "role": role}))
        named[role] = ref(path)
    label_path = tmp_path / "labels.npz"
    mask = np.ones((50, 30), bool)
    mask[0, 0] = False
    np.savez(label_path, x_ict=np.arange(8 * 29, dtype=np.float32).reshape(8, 29),
             ict_mask=np.array([True, False] * 4),
             y_action=np.arange(50 * 30, dtype=np.float32).reshape(50, 30),
             action_valid_mask=mask)
    labels = ref(label_path)
    sample = {"sample_id": "fixture_001_f0", "session_id": "001", "source_frame_id": 0,
              "source_group_id": "fixture_group", "sample_weight": 0.5, "labels": labels,
              "derivation_trace_ref": named["trace"], "causal_receipts": {"raw": named["receipt"]},
              "rgb": {"raw": image}}
    dataset = object.__new__(ProductionDerivedICTDataset)
    dataset.manifest_path = manifest
    dataset.manifest_metadata = list(VerifiedInputCache._stat(manifest))
    dataset.samples = [sample]
    dataset.shared_refs = [named["shared"]]
    dataset.session_refs = {"001": [named["session"]]}
    dataset.sample_evidence_refs = {sample["sample_id"]: [named["evidence"], named["shared"]]}
    dataset.dependencies = {r["path"]: list(VerifiedInputCache._stat(r["path"]))
                            for r in [*named.values(), labels]}
    dataset.input_cache = VerifiedInputCache()
    _npz(labels, cache=dataset.input_cache)
    dataset.branch, dataset.split = "raw", "train"
    dataset.dataset_sha = "fixture_not_production"
    dataset.manifest = {"fixture_only": True}
    return dataset


def assert_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert right[key].dtype == left[key].dtype
            assert right[key].device.type == "cpu"
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_all_fields_bitwise_equal_cold_and_hit(original):
    candidate = ResidentRGBProductionDatasetV1(original)
    assert len(candidate) == len(original)
    assert candidate.dataset_sha == original.dataset_sha
    expected = original[0]
    assert_equal(expected, candidate[0])
    assert_equal(expected, candidate[0])
    candidate.verify_all_dependencies()
    assert candidate.rgb_cache_stats()["decode_calls"] == 1
    assert candidate.rgb_cache_stats()["cache_hits"] == 1


def test_identical_guard_stat_multiset_on_every_hit(original, monkeypatch):
    candidate = ResidentRGBProductionDatasetV1(original)
    candidate[0]
    before = VerifiedInputCache._stat
    paths = []
    def observed(path):
        paths.append(str(path))
        return before(path)
    monkeypatch.setattr(VerifiedInputCache, "_stat", staticmethod(observed))
    original[0]
    expected = Counter(paths)
    paths.clear()
    candidate[0]
    assert Counter(paths) == expected


@pytest.mark.parametrize("role", ["manifest", "shared", "session", "evidence", "trace", "receipt", "labels"])
def test_metadata_mutation_rejected_after_warm_cache(original, role):
    candidate = ResidentRGBProductionDatasetV1(original)
    candidate[0]
    if role == "manifest":
        path = original.manifest_path
    elif role == "labels":
        path = original.manifest_path.parent / "labels.npz"
    else:
        path = original.manifest_path.parent / f"{role}.json"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(TrainingDataError):
        candidate[0]


def test_rgb_sha_rechecked_on_hit_even_with_restored_metadata(original, monkeypatch):
    candidate = ResidentRGBProductionDatasetV1(original)
    candidate[0]
    r = original.samples[0]["rgb"]["raw"]
    from pathlib import Path
    path = Path(r["path"])
    frozen_stat = VerifiedInputCache._stat(path)
    encoded = bytearray(path.read_bytes())
    encoded[-1] ^= 1
    path.write_bytes(encoded)
    real = VerifiedInputCache._stat
    monkeypatch.setattr(VerifiedInputCache, "_stat", staticmethod(
        lambda p: frozen_stat if str(p) == str(path) else real(p)))
    with pytest.raises(TrainingDataError, match="cannot be decoded") as error:
        candidate[0]
    assert "SHA/size drift" in str(error.value.__cause__)


def test_returned_tensor_cannot_mutate_resident_pixels(original):
    candidate = ResidentRGBProductionDatasetV1(original)
    expected = original[0]
    first = candidate[0]
    first["x_rgb"].fill_(123)
    assert_equal(expected, candidate[0])
    payload = next(iter(candidate._decoded_rgb_cache._entries.values()))
    assert isinstance(payload, bytes)
    view = np.frombuffer(payload, np.float32)
    with pytest.raises(ValueError):
        view.flags.writeable = True


def test_rng_and_requested_sample_order_unchanged(original):
    initial = original.samples[0]
    evidence = original.sample_evidence_refs[initial["sample_id"]]
    original.samples = []
    for index in range(3):
        image = png(original.manifest_path.parent / f"order_{index}.png", shift=index)
        sample = dict(initial, sample_id=f"fixture_{index}", source_frame_id=index,
                      rgb={"raw": image})
        original.samples.append(sample)
        original.sample_evidence_refs[sample["sample_id"]] = evidence
    candidate = ResidentRGBProductionDatasetV1(original)
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    for index in [2, 0, 1, 2]:
        assert_equal(original[index], candidate[index])
    assert random.getstate() == py_rng
    after = np.random.get_state()
    assert after[0] == np_rng[0] and np.array_equal(after[1], np_rng[1]) and after[2:] == np_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, MAX_DECODED_RGB_CACHE_BYTES + 1])
def test_invalid_memory_budget_rejected(bad):
    with pytest.raises(ValueError, match="512_MIB"):
        VerifiedDecodedRGBCacheV1(bad)


def test_eviction_is_lru_and_never_exceeds_bound(tmp_path):
    refs = [png(tmp_path / f"{i}.png", shift=i) for i in range(3)]
    cache = VerifiedDecodedRGBCacheV1(2 * RGB_PAYLOAD_BYTES)
    cache.tensor(refs[0]); cache.tensor(refs[1]); cache.tensor(refs[0]); cache.tensor(refs[2])
    assert cache.current_bytes == 2 * RGB_PAYLOAD_BYTES
    assert (refs[1]["sha256"], refs[1]["bytes"]) not in cache._entries
    assert cache.stats()["decode_calls"] == 3
    cache.tensor(refs[1])
    assert cache.current_bytes <= cache.max_bytes and cache.stats()["decode_calls"] == 4
    cache.clear()
    assert cache.current_bytes == 0 and cache.stats()["entries"] == 0


def test_too_small_cap_returns_exact_pixels_without_retention(tmp_path):
    r = png(tmp_path / "image.png")
    cache = VerifiedDecodedRGBCacheV1(RGB_PAYLOAD_BYTES - 1)
    first, second = cache.tensor(r), cache.tensor(r)
    assert torch.equal(first, second)
    assert cache.current_bytes == 0 and cache.stats()["decode_calls"] == 2


def test_same_sha_different_paths_reuses_decode_but_still_reads(tmp_path, monkeypatch):
    r1 = png(tmp_path / "one.png")
    p2 = tmp_path / "two.png"
    p2.write_bytes((tmp_path / "one.png").read_bytes())
    r2 = ref(p2)
    from chaoyang.humanego_ict import production_rgb_cache_v1 as module
    real = module.reference_bytes
    calls = []
    def read(r):
        calls.append(r["path"])
        return real(r)
    monkeypatch.setattr(module, "reference_bytes", read)
    cache = VerifiedDecodedRGBCacheV1()
    assert torch.equal(cache.tensor(r1), cache.tensor(r2))
    assert calls == [r1["path"], r2["path"]]
    assert cache.stats()["decode_calls"] == 1 and cache.stats()["cache_hits"] == 1


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_original_pil_conversion_is_exact(tmp_path, mode):
    r = png(tmp_path / f"{mode}.png", mode=mode)
    with Image.open(r["path"]) as image:
        rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), np.float32) / 255
    expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    cache = VerifiedDecodedRGBCacheV1()
    assert torch.equal(expected, cache.tensor(r))
    assert torch.equal(expected, cache.tensor(r))


def test_wrapper_requires_completed_original_reader():
    with pytest.raises(TypeError):
        ResidentRGBProductionDatasetV1(object())
    with pytest.raises(ValueError, match="COMPLETE_ORIGINAL"):
        ResidentRGBProductionDatasetV1(object.__new__(ProductionDerivedICTDataset))


def test_valid_sha_invalid_png_never_enters_cache(tmp_path):
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not a PNG, but has an honestly matching immutable SHA")
    cache = VerifiedDecodedRGBCacheV1()
    with pytest.raises(TrainingDataError):
        cache.tensor(ref(path))
    assert cache.current_bytes == 0 and cache.stats()["entries"] == 0


def test_missing_rgb_on_hit_is_not_replaced_by_cached_pixels(tmp_path):
    path = tmp_path / "removed.png"
    r = png(path)
    cache = VerifiedDecodedRGBCacheV1()
    cache.tensor(r)
    path.unlink()
    with pytest.raises(TrainingDataError):
        cache.tensor(r)
    assert cache.stats()["cache_hits"] == 0
