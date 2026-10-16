"""Candidate CPU-only RGB decode cache; not an active production entrypoint.

Wrap an already admitted ProductionDerivedICTDataset only in a newly admitted
successor. Every item keeps the original metadata checks and fresh RGB
read/size/SHA verification. Only deterministic PIL decode/resize is cached.
No source guard, sampling, RNG, model or optimizer is changed by this module.
"""
from __future__ import annotations

import io
from collections import OrderedDict

import numpy as np

from .data import TrainingDataError, VerifiedInputCache, _npz, reference_bytes
from .production_data import ProductionDerivedICTDataset, SUPERVISION_CLASS, TRAINING_SCOPE


MAX_DECODED_RGB_CACHE_BYTES = 512 * 1024 * 1024
RGB_SHAPE = (3, 240, 320)
RGB_PAYLOAD_BYTES = 3 * 240 * 320 * np.dtype(np.float32).itemsize


class VerifiedDecodedRGBCacheV1:
    """Bounded LRU of immutable CPU FP32 payloads, keyed by encoded SHA/size.

    The cap covers retained pixel payloads, not caller-owned tensor copies or
    temporary decoding memory. Cached arrays are never exposed: immutable bytes
    back each entry, and each returned CPU tensor owns a separate writable copy.
    A cache hit still rereads and SHA-verifies the original encoded image.
    Intended for the original serial reader; no prefetch threads are introduced.
    """

    def __init__(self, max_bytes=MAX_DECODED_RGB_CACHE_BYTES):
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_DECODED_RGB_CACHE_BYTES:
            raise ValueError("DECODED_RGB_CACHE_LIMIT_MUST_BE_1_TO_512_MIB")
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._entries = OrderedDict()
        self._hits = 0
        self._decodes = 0

    @property
    def max_bytes(self):
        return self._max_bytes

    @property
    def current_bytes(self):
        return self._current_bytes

    def stats(self):
        return {"max_bytes": self.max_bytes, "current_bytes": self.current_bytes,
                "entries": len(self._entries), "cache_hits": self._hits,
                "decode_calls": self._decodes}

    def clear(self):
        self._entries.clear()
        self._current_bytes = 0

    def tensor(self, reference):
        """Return the exact old-reader RGB tensor after its original SHA guard."""
        from PIL import Image
        import torch

        try:
            # Deliberately no VerifiedInputCache or cached encoded-byte path.
            # Keep reference_bytes' path, before/after stat, size and SHA gates.
            encoded = reference_bytes(reference)
            key = (reference["sha256"], reference["bytes"])
            payload = self._entries.get(key)
            if payload is None:
                with Image.open(io.BytesIO(encoded)) as image:
                    rgb = np.asarray(image.convert("RGB").resize(
                        (320, 240), Image.Resampling.BILINEAR), np.float32) / 255
                payload = rgb.transpose(2, 0, 1).copy().tobytes(order="C")
                self._decodes += 1
                if len(payload) <= self.max_bytes:
                    while self._entries and self.current_bytes + len(payload) > self.max_bytes:
                        _, old = self._entries.popitem(last=False)
                        self._current_bytes -= len(old)
                    self._entries[key] = payload
                    self._current_bytes += len(payload)
            else:
                self._hits += 1
                self._entries.move_to_end(key)
        except (OSError, ValueError) as error:
            raise TrainingDataError("admitted production RGB cannot be decoded") from error
        # bytes-backed arrays cannot be made writable, and callers never receive
        # that view. This copy matches the old reader's independently owned CHW.
        return torch.from_numpy(np.frombuffer(payload, dtype=np.float32).reshape(RGB_SHAPE).copy())


class ResidentRGBProductionDatasetV1(ProductionDerivedICTDataset):
    """Candidate wrapper reusing one completed original dataset admission.

    Construct as ResidentRGBProductionDatasetV1(admitted_dataset), never by
    attaching it to a running worker. This performs no second source preflight.
    Inherited len(), manifest identity and verify_all_dependencies() are intact.
    """

    def __init__(self, admitted_dataset, *, max_cache_bytes=MAX_DECODED_RGB_CACHE_BYTES):
        if type(admitted_dataset) is not ProductionDerivedICTDataset:
            raise TypeError("ORIGINAL_ADMITTED_PRODUCTION_DATASET_REQUIRED")
        required = ("manifest_path", "manifest_metadata", "dependencies", "samples",
                    "shared_refs", "session_refs", "sample_evidence_refs", "input_cache",
                    "branch", "split", "dataset_sha", "manifest")
        if any(not hasattr(admitted_dataset, field) for field in required):
            raise ValueError("COMPLETE_ORIGINAL_DATASET_ADMISSION_REQUIRED")
        self.__dict__.update(admitted_dataset.__dict__)
        self._decoded_rgb_cache = VerifiedDecodedRGBCacheV1(max_cache_bytes)

    def rgb_cache_stats(self):
        return self._decoded_rgb_cache.stats()

    def clear_rgb_cache(self):
        self._decoded_rgb_cache.clear()

    def __getitem__(self, index):
        # These guards and all non-RGB fields deliberately mirror the frozen
        # original __getitem__, not batch-local or less frequent replacements.
        if list(VerifiedInputCache._stat(self.manifest_path)) != self.manifest_metadata:
            raise TrainingDataError("immutable production manifest changed")
        sample = self.samples[index]
        refs = self.shared_refs + self.session_refs[sample["session_id"]] + self.sample_evidence_refs[sample["sample_id"]] + [sample["labels"], sample["derivation_trace_ref"], sample["causal_receipts"][self.branch]]
        for reference in {r["path"]: r for r in refs}.values():
            path = reference["path"]
            if list(VerifiedInputCache._stat(path)) != self.dependencies[path]:
                raise TrainingDataError("immutable production source/evidence dependency changed")
        labels = _npz(sample["labels"], cache=self.input_cache)
        rgb = self._decoded_rgb_cache.tensor(sample["rgb"][self.branch])
        import torch
        mask = labels["action_valid_mask"]
        return {"x_rgb": rgb, "x_ict": torch.from_numpy(labels["x_ict"].astype(np.float32)),
            "ict_mask": torch.from_numpy(labels["ict_mask"].copy()), "y_action": torch.from_numpy(np.where(mask, labels["y_action"], 0).astype(np.float32)),
            "action_valid_mask": torch.from_numpy(mask.copy()), "sample_weight": torch.tensor(sample.get("sample_weight", 1), dtype=torch.float32),
            "sample_id": sample["sample_id"], "session_id": sample["session_id"], "source_frame_id": sample["source_frame_id"],
            "source_group_id": sample["source_group_id"], "sample_index": index, "branch": self.branch, "split": self.split,
            "training_scope": TRAINING_SCOPE, "supervision_class": SUPERVISION_CLASS,
            "control_ground_truth": False, "physical_deployment_authorized": False}
