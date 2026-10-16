"""Capture the frozen ProPainter Raw RAFT calls, without changing their results.

Raw optical flow is not registered background evidence. This first G2 bridge
archives real inputs/flows and existing vendor FB diagnostics; with registration
ABSENT every donor remains UNKNOWN and the reference policy is a true no-op.
Importing this module neither imports Torch nor constructs a model.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .automatic_masks import frames_sha256
from .clean_references import ReferenceEvidence, select_reference_indices
from .contracts import SCHEMA, ContractError, canonical_sha, file_ref, verify_ref, verify_sealed_stage
from .model_stages import _gate, _same_gate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_SCHEMA = "inpainting-raw-raft-capture-config-v1"
PREPROCESSING = "RGB_UINT8_FLOAT32_B1_T3HW_DIV127_5_MINUS1"


def write_json(path: Path, value: Any) -> dict:
    """Immutable stage writes. The receipt is always written last."""
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def write_npz(path: Path, **arrays: np.ndarray) -> dict:
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ContractError("OBJECT_ARRAY_NOT_ALLOWED")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return file_ref(path)


def fresh_directory(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root.exists() or root == PROJECT_ROOT or not root.is_relative_to(PROJECT_ROOT):
        raise ContractError("FRESH_PROJECT_STAGE_REQUIRED")
    root.mkdir(parents=True)
    return root


def array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.data)
    return digest.hexdigest()


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def output_named(receipt: Mapping, name: str) -> dict:
    refs = [dict(ref) for ref in receipt["outputs"] if Path(ref["path"]).name == name]
    if len(refs) != 1:
        raise ContractError("SEALED_OUTPUT_REQUIRED:" + name)
    return refs[0]


def sealed_receipt(reference: Mapping, *, expected_stage: str | None = None) -> dict:
    verify_ref(reference)
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT):
        raise ContractError("SEALED_STAGE_PROJECT_REAL_PATH_REQUIRED")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    verify_sealed_stage(receipt, receipt.get("component_signature"), output_root=path.parent)
    if expected_stage is not None and receipt.get("stage") != expected_stage:
        raise ContractError("SEALED_STAGE_ROLE_MISMATCH")
    return receipt


def chunk_plan(native_frames: int, chunk_length: int = 12) -> list[dict]:
    if type(native_frames) is not int or native_frames < 2 or chunk_length != 12:
        raise ContractError("FROZEN_RAW_RAFT_CHUNK_DOMAIN")
    rows = []
    for start in range(0, native_frames, chunk_length):
        first, end = (start if start == 0 else start - 1), min(native_frames, start + chunk_length)
        if end - first < 2:
            continue
        rows.append({"chunk_index": len(rows), "first_frame": first, "end_frame_exclusive": end,
                     "first_edge": first, "end_edge_exclusive": end - 1})
    return rows


class RawRaftCaptureReplay:
    """Private copy observer; capture always returns the original tuple identity.

    Runtime authority is checked before and after each original call. Diagnostic
    capture failures are latched *after* a legal original call and cannot erase
    an independently valid A Clean. SHA/authority drift is not swallowed.
    Replay consumes only a complete, verified same-configuration archive.
    """

    def __init__(self, parent: Any, *, raw_rgb: np.ndarray, frame_map: Sequence[Mapping],
                 capture_config_ref: Mapping, output_dir: str | Path,
                 authorization: Callable, replay_receipt_ref: Mapping | None = None):
        self.owner = _gate(authorization)
        self.authorization, self.original_parent = authorization, parent
        self.config_ref = dict(capture_config_ref)
        verify_ref(self.config_ref)
        config_path = Path(self.config_ref["path"]).resolve(strict=True)
        if not config_path.is_relative_to(PROJECT_ROOT):
            raise ContractError("CAPTURE_CONFIG_PROJECT_REAL_PATH_REQUIRED")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        required = {"schema", "session_id", "native_shape", "source_receipt", "raw_source_ref",
                    "raw_rgb_ref", "frame_map_ref", "raw_frames_sha256", "frame_map_payload_sha256",
                    "raft_weight_ref", "runtime_code", "raft_iter", "chunk_length", "preprocessing"}
        if not required <= set(self.config):
            raise ContractError("RAW_RAFT_CAPTURE_CONFIG_INCOMPLETE")
        if (self.config["schema"] != CAPTURE_SCHEMA or self.config["raft_iter"] != 20
                or self.config["chunk_length"] != 12 or self.config["preprocessing"] != PREPROCESSING):
            raise ContractError("FROZEN_RAW_RAFT_SETTINGS_DRIFT")
        self.raw = np.asarray(raw_rgb)
        self.mapping = [dict(row) for row in frame_map]
        if (self.raw.dtype != np.uint8 or self.raw.ndim != 4 or self.raw.shape[-1] != 3
                or self.raw.shape[2] > 640 or list(self.raw.shape[:3]) != self.config["native_shape"]
                or len(self.mapping) != len(self.raw)
                or frames_sha256(self.raw) != self.config["raw_frames_sha256"]
                or canonical_sha(self.mapping) != self.config["frame_map_payload_sha256"]):
            raise ContractError("RAW_RAFT_CAPTURE_SOURCE_OR_MAP_IDENTITY_CHANGED")
        for index, row in enumerate(self.mapping):
            if row.get("native_frame_index") != index or row.get("decoded_display_index") != index:
                raise ContractError("RAW_RAFT_NATIVE_FRAME_ORDER_CHANGED")
        self.input_refs = [self.config_ref, *(self.config[key] for key in (
            "source_receipt", "raw_source_ref", "raw_rgb_ref", "frame_map_ref", "raft_weight_ref")),
            *self.config["runtime_code"]]
        for ref in self.input_refs:
            verify_ref(ref)
        source = sealed_receipt(self.config["source_receipt"], expected_stage="Source")
        if (source.get("native_frames") != len(self.raw)
                or output_named(source, "RAW_RGB.npz") != self.config["raw_rgb_ref"]
                or output_named(source, "FRAME_MAP.json") != self.config["frame_map_ref"]):
            raise ContractError("CAPTURE_SOURCE_RECEIPT_OUTPUT_BINDING_MISMATCH")
        self.plan = chunk_plan(len(self.raw))
        self.calls: list[dict] = []
        self.error: str | None = None
        self.mode = "CAPTURE" if replay_receipt_ref is None else "REPLAY"
        self.root = fresh_directory(output_dir)
        self.replay = None
        if replay_receipt_ref is not None:
            self.replay = sealed_receipt(replay_receipt_ref, expected_stage="RawFlow")
            if (self.replay.get("actual_inputs", {}).get("capture_config") != self.config_ref
                    or self.replay.get("actual_native_shape") != list(self.raw.shape[:3])):
                raise ContractError("RAW_RAFT_REPLAY_SIGNATURE_CHANGED")
            index_ref = output_named(self.replay, "CHUNK_INDEX.json")
            self.replay_rows = json.loads(Path(index_ref["path"]).read_text(encoding="utf-8"))
            if len(self.replay_rows) != len(self.plan):
                raise ContractError("RAW_RAFT_REPLAY_INCOMPLETE")
        self.parent = copy.copy(parent)
        self.original_raft = parent.raft
        self.parent.raft = self

    def __call__(self, tensor: Any, *, iters: int):
        _same_gate(self.authorization, self.owner)
        verify_ref(self.config_ref)
        if iters != self.config["raft_iter"] or len(self.calls) >= len(self.plan):
            raise ContractError("RAW_RAFT_CALL_SETTINGS_OR_COUNT_CHANGED")
        plan = self.plan[len(self.calls)]
        value = as_numpy(tensor)
        expected = self.raw[plan["first_frame"]:plan["end_frame_exclusive"]]
        domain = (1, len(expected), 3, expected.shape[1], expected.shape[2])
        if value.dtype != np.float32 or value.shape != domain or not np.all(np.isfinite(value)):
            raise ContractError("RAW_RAFT_ACTUAL_NORMALIZED_TENSOR_DOMAIN_CHANGED")
        # GPU and CPU arithmetic may round differently. Exact uint8 inversion
        # binds the actual native pixels; the exact *actual* float tensor SHA is
        # archived and must match bit-for-bit on replay. No tolerance is added.
        recovered = np.rint((value[0].transpose(0, 2, 3, 1) + 1) * 127.5)
        if (np.any(value < -1) or np.any(value > 1)
                or not np.array_equal(recovered, expected)):
            raise ContractError("RAW_RAFT_ACTUAL_INPUT_NOT_NATIVE_RAW_CHUNK")
        row = {**plan, "raft_iter": iters, "normalized_input_sha256": array_sha(value),
               "raw_chunk_sha256": frames_sha256(expected),
               "model_input_frames": list(range(plan["first_frame"], plan["end_frame_exclusive"])),
               "frame_map_payload_sha256": canonical_sha(self.mapping[
                   plan["first_frame"]:plan["end_frame_exclusive"]])}
        if self.mode == "REPLAY":
            frozen = self.replay_rows[len(self.calls)]
            if any(frozen.get(key) != row[key] for key in row):
                raise ContractError("RAW_RAFT_REPLAY_ACTUAL_INPUT_SHA_CHANGED")
            verify_ref(frozen["arrays"])
            with np.load(frozen["arrays"]["path"], allow_pickle=False) as archive:
                arrays = (archive["forward"], archive["backward"])
            if (array_sha(arrays[0]) != frozen["forward_array_sha256"]
                    or array_sha(arrays[1]) != frozen["backward_array_sha256"]):
                raise ContractError("RAW_RAFT_REPLAY_ARRAY_SHA_CHANGED")
            row.update({key: frozen[key] for key in (
                "arrays", "forward_array_sha256", "backward_array_sha256")})
            torch = getattr(self.original_parent, "torch", None)
            result = (tuple(torch.from_numpy(item).to(tensor.device) for item in arrays)
                      if torch is not None else arrays)
        else:
            result = self.original_raft(tensor, iters=iters)
            _same_gate(self.authorization, self.owner)
            try:
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise ContractError("RAW_RAFT_RETURN_DOMAIN_CHANGED")
                arrays = tuple(as_numpy(item) for item in result)
                flow_shape = (1, len(expected) - 1, 2, expected.shape[1], expected.shape[2])
                if any(item.shape != flow_shape or item.dtype != np.float32
                       or not np.all(np.isfinite(item)) for item in arrays):
                    raise ContractError("RAW_RAFT_FLOW_DOMAIN_OR_TYPE_CHANGED")
                row["forward_array_sha256"], row["backward_array_sha256"] = map(array_sha, arrays)
                if self.error is None:
                    row["arrays"] = write_npz(self.root / f"chunk_{plan['chunk_index']:04d}.npz",
                        forward=arrays[0], backward=arrays[1])
            except Exception as error:
                self.error = self.error or type(error).__name__ + ":" + str(error)
                row["capture_failure"] = self.error
        self.calls.append(row)
        _same_gate(self.authorization, self.owner)
        return result

    def seal(self) -> dict:
        _same_gate(self.authorization, self.owner)
        for ref in self.input_refs:
            verify_ref(ref)
        if self.error or len(self.calls) != len(self.plan) or any("arrays" not in row for row in self.calls):
            raise ContractError("RAW_FLOW_CAPTURE_INCOMPLETE:" + (self.error or str(len(self.calls))))
        if self.mode == "REPLAY":
            return dict(self.replay)
        outputs = [row["arrays"] for row in self.calls]
        outputs.append(write_json(self.root / "CHUNK_INDEX.json", self.calls))
        payload = {"schema": SCHEMA, "stage": "RawFlow", "status": "SEALED_COMPLETE",
            "actual_native_shape": list(self.raw.shape[:3]), "native_input_frames": len(self.raw),
            "adjacent_edges": len(self.raw) - 1, "session_id": self.config["session_id"],
            "actual_inputs": {"capture_config": self.config_ref,
                "source_receipt": self.config["source_receipt"], "raw_source": self.config["raw_source_ref"],
                "raw_rgb": self.config["raw_rgb_ref"], "frame_map": self.config["frame_map_ref"],
                "raft_weight": self.config["raft_weight_ref"], "runtime_code": self.config["runtime_code"],
                "frames_sha256": frames_sha256(self.raw), "frame_map_sha256": canonical_sha(self.mapping)},
            "outputs": outputs, "model_subcalls": {"RAFT_bi": len(self.calls),
                "fix_raft_forward_backward": 2 * len(self.calls)},
            "returned_original_output_identity": True, "flow_source": "RAW_RGB_FORWARD_BACKWARD_FLOW",
            "completed_inpainted_flow_consumed": False, "background_registration": "ABSENT",
            "background_geometry_qualified": False, "training_eligible": False,
            "control_ground_truth": False}
        payload["component_signature"] = canonical_sha({"schema": SCHEMA, "stage": "RawFlow",
            "actual_inputs": payload["actual_inputs"], "actual_native_shape": payload["actual_native_shape"]})
        write_json(self.root / "RECEIPT.json", payload)
        return payload


def vendor_fb_diagnostics(forward: np.ndarray, backward: np.ndarray, parent: Any) -> dict[str, np.ndarray]:
    """Actual frozen vendor warp/default FB rule, not a background qualifier.

    This has no neural-network calls. Its Torch import is delayed until the
    worker has already validated authority and constructed the existing PP.
    """
    path = PROJECT_ROOT / "vendor/ProPainter/model/modules/flow_loss_utils.py"
    spec = importlib.util.spec_from_file_location("_candidate_frozen_pp_fb_utils", path)
    if spec is None or spec.loader is None:
        raise ContractError("FROZEN_VENDOR_FB_CODE_REQUIRED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch = parent.torch
    f = torch.from_numpy(forward[0]).to(parent.device)
    b = torch.from_numpy(backward[0]).to(parent.device)
    with torch.no_grad():
        occ_f, occ_b = module.fbConsistencyCheck(f, b)  # vendor defaults unchanged
        residual_f = (f + module.flow_warp(b, f.permute(0, 2, 3, 1))).square().sum(dim=1)
        residual_b = (b + module.flow_warp(f, b.permute(0, 2, 3, 1))).square().sum(dim=1)
    return {"fb_residual_squared_forward": as_numpy(residual_f),
            "fb_residual_squared_backward": as_numpy(residual_b),
            "vendor_fb_consistent_forward": as_numpy(occ_f[:, 0]) == 0,
            "vendor_fb_consistent_backward": as_numpy(occ_b[:, 0]) == 0}


class RawVisibleReferenceBuilder:
    """One PP-invocation cache; verify and freeze input arrays once, never across Tasks.

    Raw unremoved visible support is not static background/geometry/hidden truth.
    Only existing vendor-default FB consistency is used, with no new threshold.
    A path requires all contributing bilinear donor samples outside E/P/U and
    wearer/device/held domains; invalid/nonfinite/out-of-bounds paths are UNKNOWN.
    Error is the maximum existing squared FB residual along the composed path.
    """
    def __init__(self, *, session_id: str, raw_rgb: np.ndarray, frame_map: Sequence[Mapping],
                 masks: Mapping[str, np.ndarray], forward: np.ndarray, backward: np.ndarray,
                 fb_consistent_forward: np.ndarray, fb_consistent_backward: np.ndarray,
                 binding: Mapping):
        source = sealed_receipt(binding["source_receipt"], expected_stage="Source")
        mask = sealed_receipt(binding["mask_receipt"], expected_stage="Mask")
        flow = sealed_receipt(binding["raw_flow_receipt"], expected_stage="RawFlow")
        raw = np.array(raw_rgb, copy=True)
        masks = {k: np.array(v, copy=True) for k, v in masks.items()}
        forward, backward = np.array(forward, copy=True), np.array(backward, copy=True)
        fb_consistent_forward = np.array(fb_consistent_forward, copy=True)
        fb_consistent_backward = np.array(fb_consistent_backward, copy=True)
        if (raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[-1] != 3
                or raw.shape[0] < 2 or source.get("session_id") != session_id
                or flow.get("session_id") != session_id):
            raise ContractError("RAW_VISIBLE_REFERENCE_SOURCE_DOMAIN")
        n, h, w = raw.shape[:3]
        mapping = [dict(m) for m in frame_map]
        actual = flow.get("actual_inputs", {})
        if (source.get("native_frames") != n or flow.get("actual_native_shape") != [n, h, w]
                or flow.get("flow_source") != "RAW_RGB_FORWARD_BACKWARD_FLOW"
                or flow.get("completed_inpainted_flow_consumed") is not False
                or actual.get("source_receipt") != binding["source_receipt"]
                or actual.get("raw_source") != mask.get("actual_inputs", {}).get("raw_source")
                or actual.get("raw_rgb") != output_named(source, "RAW_RGB.npz")
                or actual.get("frame_map") != output_named(source, "FRAME_MAP.json")
                or len(mapping) != n or any(m.get("native_frame_index") != i for i, m in enumerate(mapping))
                or actual.get("frames_sha256") != frames_sha256(raw)
                or actual.get("frame_map_sha256") != canonical_sha(mapping)
                or mask.get("actual_inputs", {}).get("frames_sha256") != frames_sha256(raw)
                or mask.get("actual_inputs", {}).get("frame_map_sha256") != canonical_sha(mapping)):
            raise ContractError("RAW_VISIBLE_REFERENCE_BINDING_DRIFT")
        if json.loads(Path(actual["frame_map"]["path"]).read_text()) != mapping:
            raise ContractError("RAW_VISIBLE_REFERENCE_MAP_DRIFT")
        with np.load(actual["raw_rgb"]["path"], allow_pickle=False) as archive:
            if not np.array_equal(archive["rgb"], raw):
                raise ContractError("RAW_VISIBLE_REFERENCE_RAW_PIXELS_DRIFT")
        excluded = np.zeros((n, h, w), bool)
        with np.load(output_named(mask, "MASK_ARRAYS.npz")["path"], allow_pickle=False) as archive:
            for key in ("effective_removal", "protected_visible", "unknown", "removal", "core", "equipment", "held"):
                value = masks.get(key)
                if key in ("effective_removal", "protected_visible", "unknown") and value is None:
                    raise ContractError("RAW_VISIBLE_REFERENCE_MASK_REQUIRED:" + key)
                if value is None and key in archive:
                    value = archive[key]
                if value is not None:
                    if (value.dtype != np.bool_ or value.shape != excluded.shape
                            or key not in archive or not np.array_equal(value, archive[key])):
                        raise ContractError("RAW_VISIBLE_REFERENCE_MASK_DRIFT:" + key)
                    excluded |= value
        if np.any(masks["effective_removal"] & masks["protected_visible"]):
            raise ContractError("RAW_VISIBLE_REFERENCE_PROTECTED_HOLE")
        for a in (forward, backward):
            if a.shape != (n - 1, 2, h, w) or a.dtype != np.float32:
                raise ContractError("RAW_VISIBLE_REFERENCE_FLOW_DOMAIN")
        if binding.get("vendor_fb_diagnostic_policy") != "EXISTING_VENDOR_DEFAULTS_NOT_BACKGROUND_CONFIDENCE":
            raise ContractError("RAW_VISIBLE_REFERENCE_VENDOR_FB_POLICY")
        for name, a in (("fb_consistent_forward", fb_consistent_forward), ("fb_consistent_backward", fb_consistent_backward)):
            if a.shape != (n - 1, h, w) or a.dtype != np.bool_ or array_sha(a) != binding.get(name + "_sha256"):
                raise ContractError("RAW_VISIBLE_REFERENCE_VENDOR_FB_BINDING")
        rows = json.loads(Path(output_named(flow, "CHUNK_INDEX.json")["path"]).read_text())
        covered = []
        for row in rows:
            first, end = row["first_edge"], row["end_edge_exclusive"]
            verify_ref(row["arrays"])
            if (array_sha(forward[None, first:end]) != row["forward_array_sha256"]
                    or array_sha(backward[None, first:end]) != row["backward_array_sha256"]):
                raise ContractError("RAW_VISIBLE_REFERENCE_ACTUAL_FLOW_DRIFT")
            covered.extend(range(first, end))
        if covered != list(range(n - 1)):
            raise ContractError("RAW_VISIBLE_REFERENCE_FLOW_EDGE_CLOSURE")
        self.session_id, self.shape = session_id, (n, h, w)
        self.frame_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in raw]
        self.binding_sha = canonical_sha(dict(binding))
        self.flow_receipt_ref = dict(binding["raw_flow_receipt"])
        self.evidence_metadata = {"binding_sha256": self.binding_sha,
            "raw_flow_receipt": self.flow_receipt_ref,
            "mask_receipt": dict(binding["mask_receipt"]),
            "source_receipt": dict(binding["source_receipt"]),
            "vendor_fb_diagnostic_policy": binding["vendor_fb_diagnostic_policy"],
            "fb_consistent_forward_sha256": binding["fb_consistent_forward_sha256"],
            "fb_consistent_backward_sha256": binding["fb_consistent_backward_sha256"],
            "background_geometry_qualified": False, "static_background_truth": False}
        self.excluded, self.holes = excluded, masks["effective_removal"]
        self.forward, self.backward = forward, backward
        self.consistent_forward, self.consistent_backward = fb_consistent_forward, fb_consistent_backward
        for a in (self.excluded, self.holes, forward, backward, fb_consistent_forward, fb_consistent_backward):
            a.setflags(write=False)

    def build(self, target_frame: int, reference_frames: Sequence[int]) -> list[ReferenceEvidence]:
        """No file reads, hashes of full clips, model calls or registration here."""
        n, h, w = self.shape
        if (type(target_frame) is not int or not 0 <= target_frame < n
                or len(set(reference_frames)) != len(reference_frames)
                or any(type(i) is not int or not 0 <= i < n or i == target_frame for i in reference_frames)):
            raise ContractError("RAW_VISIBLE_REFERENCE_NATIVE_DOMAIN")
        forward, backward = self.forward, self.backward
        fb_consistent_forward, fb_consistent_backward = self.consistent_forward, self.consistent_backward
        excluded = self.excluded
        hole = self.holes[target_frame]
        yy, xx = np.nonzero(hole)
        def corners(x, y):
            good = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
            x = np.where(good, x, 0); y = np.where(good, y, 0)
            x0 = np.floor(x).astype(np.intp); y0 = np.floor(y).astype(np.intp)
            return good, x0, y0, np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1), x - x0, y - y0
        def sample(a, x, y):
            good, x0, y0, x1, y1, dx, dy = corners(x, y)
            values = (a[..., y0, x0], a[..., y0, x1], a[..., y1, x0], a[..., y1, x1])
            if a.dtype == np.bool_:
                out = values[0] & ((dx == 0) | values[1]) & ((dy == 0) | values[2]) & (((dx == 0) | (dy == 0)) | values[3])
            else:
                out = (values[0] * (1-dx)*(1-dy) + values[1]*dx*(1-dy)
                       + values[2]*(1-dx)*dy + values[3]*dx*dy)
            return out, good
        results = {}
        for direction in (-1, 1):
            donors = {i for i in reference_frames if (i - target_frame) * direction > 0}
            if not donors:
                continue
            x, y = xx.astype(np.float64), yy.astype(np.float64)
            valid = np.ones(len(xx), bool); error = np.zeros(len(xx), np.float32)
            last = max(donors) if direction == 1 else min(donors)
            for arrived in range(target_frame + direction, last + direction, direction):
                edge = arrived - 1 if direction == 1 else arrived
                f, b = (forward[edge], backward[edge]) if direction == 1 else (backward[edge], forward[edge])
                cf, cb = ((fb_consistent_forward[edge], fb_consistent_backward[edge]) if direction == 1
                          else (fb_consistent_backward[edge], fb_consistent_forward[edge]))
                move, inside = sample(f, x, y); consistent, _ = sample(cf, x, y)
                nx, ny = x + move[0], y + move[1]
                reverse, arrived_inside = sample(b, nx, ny); reverse_consistent, _ = sample(cb, nx, ny)
                visible, _ = sample(~excluded[arrived], nx, ny)
                residual = np.square(move + reverse).sum(axis=0)
                valid &= inside & arrived_inside & consistent & reverse_consistent & visible & np.isfinite(residual)
                error = np.maximum(error, np.where(np.isfinite(residual), residual, np.inf))
                x, y = nx, ny
                if arrived in donors:
                    support = np.zeros((h, w), bool); errors = np.full((h, w), np.inf, np.float32)
                    support[yy[valid], xx[valid]] = True; errors[yy[valid], xx[valid]] = error[valid]
                    results[arrived] = ReferenceEvidence(self.session_id, target_frame, arrived, support, errors,
                        self.frame_hashes[arrived], array_sha(support),
                        self.flow_receipt_ref["sha256"], support_source="RAW_UNREMOVED_VISIBLE_SUPPORT",
                        evidence_metadata=copy.deepcopy(self.evidence_metadata))
        return [results[i] for i in reference_frames]



def build_raw_visible_reference_evidence(*, session_id: str, target_frame: int,
        reference_frames: Sequence[int], raw_rgb: np.ndarray, frame_map: Sequence[Mapping],
        masks: Mapping[str, np.ndarray], forward: np.ndarray, backward: np.ndarray,
        fb_consistent_forward: np.ndarray, fb_consistent_backward: np.ndarray,
        binding: Mapping) -> list[ReferenceEvidence]:
    """One-shot compatibility helper; repeated PP calls must reuse the builder."""
    return RawVisibleReferenceBuilder(session_id=session_id, raw_rgb=raw_rgb,
        frame_map=frame_map, masks=masks, forward=forward, backward=backward,
        fb_consistent_forward=fb_consistent_forward, fb_consistent_backward=fb_consistent_backward,
        binding=binding).build(target_frame, reference_frames)


def seal_reference_evidence(*, raw_flow_receipt_ref: Mapping, mask_receipt_ref: Mapping,
        clean_receipt_ref: Mapping,
        masks: Mapping[str, np.ndarray], reference_calls: Sequence[Mapping], output_dir: str | Path,
        parent: Any, authorization: Callable, diagnostic_provider: Callable | None = None) -> dict:
    """Seal UNKNOWN background support and a real, unchanged reference plan."""
    owner = _gate(authorization)
    flow = sealed_receipt(raw_flow_receipt_ref, expected_stage="RawFlow")
    mask = sealed_receipt(mask_receipt_ref, expected_stage="Mask")
    clean = sealed_receipt(clean_receipt_ref, expected_stage="Clean")
    recorded_calls = json.loads(Path(output_named(clean, "REFERENCE_SELECTION.json")["path"]).read_text(encoding="utf-8"))
    if list(reference_calls) != recorded_calls:
        raise ContractError("REFERENCE_CALLS_NOT_FROM_SEALED_A")
    shape = tuple(flow["actual_native_shape"])
    if (mask["actual_inputs"]["frames_sha256"] != flow["actual_inputs"]["frames_sha256"]
            or mask["actual_inputs"]["frame_map_sha256"] != flow["actual_inputs"]["frame_map_sha256"]):
        raise ContractError("REFERENCE_RAW_FLOW_MASK_SOURCE_IDENTITY_CHANGED")
    root = fresh_directory(output_dir)
    diagnostics = diagnostic_provider or vendor_fb_diagnostics
    flow_rows = json.loads(Path(output_named(flow, "CHUNK_INDEX.json")["path"]).read_text(encoding="utf-8"))
    outputs, diagnostic_rows = [], []
    for row in flow_rows:
        _same_gate(authorization, owner)
        verify_ref(row["arrays"])
        with np.load(row["arrays"]["path"], allow_pickle=False) as archive:
            values = diagnostics(archive["forward"], archive["backward"], parent)
        edge_shape = (row["end_edge_exclusive"] - row["first_edge"], *shape[1:])
        if set(values) != {"fb_residual_squared_forward", "fb_residual_squared_backward",
                          "vendor_fb_consistent_forward", "vendor_fb_consistent_backward"}:
            raise ContractError("FB_DIAGNOSTIC_FIELDS_INVALID")
        for key, value in values.items():
            if value.shape != edge_shape or (value.dtype != np.bool_ if "consistent" in key else
                    value.dtype != np.float32 or not np.all(np.isfinite(value)) or np.any(value < 0)):
                raise ContractError("FB_DIAGNOSTIC_DOMAIN_INVALID")
        ref = write_npz(root / f"fb_{row['chunk_index']:04d}.npz", **values)
        outputs.append(ref)
        diagnostic_rows.append({"first_edge": row["first_edge"],
            "end_edge_exclusive": row["end_edge_exclusive"], "arrays": ref,
            "source_raw_flow": row["arrays"], "support_qualification": "UNKNOWN"})
    removal, protected = masks["effective_removal"], masks["protected_visible"]
    if any(value.dtype != np.bool_ or value.shape != shape for value in (removal, protected)):
        raise ContractError("REFERENCE_FULL_NATIVE_MASK_DOMAIN_REQUIRED")
    with np.load(output_named(mask, "MASK_ARRAYS.npz")["path"], allow_pickle=False) as archive:
        if any(name not in archive or not np.array_equal(value, archive[name]) for name, value in masks.items()):
            raise ContractError("REFERENCE_MASK_ARRAYS_DIFFER_FROM_SEALED_UPSTREAM")
    # ~E is only a bookkeeping candidate, NOT positive background truth.
    excluded = np.zeros(shape, bool)
    for key in ("core", "equipment", "unknown", "held", "removal", "protected_visible"):
        value = masks.get(key)
        if value is not None:
            if value.shape != shape or value.dtype != np.bool_:
                raise ContractError("REFERENCE_MASK_DOMAIN_CHANGED:" + key)
            excluded |= value
    outputs.append(write_npz(root / "SUPPORT_ARRAYS.npz",
        raw_unremoved_candidate=~excluded, qualified_visible_background=np.zeros(shape, bool),
        background_support_unknown=np.ones(shape, bool)))
    from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
    expected_centers = list(range(0, shape[0], int(SETTINGS["neighbor_length"]) // 2))
    if [row["target_frame"] for row in reference_calls] != expected_centers:
        raise ContractError("ACTUAL_FROZEN_PP_REFERENCE_CALL_COUNT_OR_ORDER_CHANGED")
    plans = []
    for call in reference_calls:
        mid = call["target_frame"]
        neighbors = list(range(max(0, mid - 5), min(shape[0], mid + 6)))
        ref_num = SETTINGS["subvideo_length"] // SETTINGS["ref_stride"] if shape[0] > SETTINGS["subvideo_length"] else -1
        original = _get_ref_index(mid, neighbors, shape[0], SETTINGS["ref_stride"], ref_num)
        if (call["neighbors"] != neighbors or call["original_references"] != original
                or call["references"] != original or call["reference_count"] != len(original)):
            raise ContractError("ACTUAL_A_REFERENCE_LIST_NOT_FROZEN_BASELINE")
        selection = select_reference_indices(mid, neighbors, shape[0], session_id=flow["session_id"],
            effective_delete_mask=removal[mid], evidence=[])
        plans.append({"target_frame": mid, "neighbors": neighbors, **selection})
    outputs += [write_json(root / "FB_INDEX.json", diagnostic_rows),
                write_json(root / "REFERENCE_PLAN.json", plans)]
    actual_inputs = {"raw_flow_receipt": dict(raw_flow_receipt_ref), "mask_receipt": dict(mask_receipt_ref),
                     "clean_baseline_receipt": dict(clean_receipt_ref)}
    for ref in actual_inputs.values():
        verify_ref(ref)
    _same_gate(authorization, owner)
    payload = {"schema": SCHEMA, "stage": "ReferenceEvidence", "status": "SEALED_COMPLETE",
        "component_signature": canonical_sha({"schema": SCHEMA, "stage": "ReferenceEvidence",
            "actual_inputs": actual_inputs, "policy": "RAW_FB_DIAGNOSTIC_UNKNOWN_NO_QUALIFIED_SUPPORT"}),
        "actual_inputs": actual_inputs, "outputs": outputs, "actual_native_shape": list(shape),
        "session_id": flow["session_id"], "background_registration": "ABSENT",
        "support_status": "UNKNOWN_BACKGROUND_REGISTRATION_ABSENT", "qualified_reference_pairs": 0,
        "qualified_visible_background_pixels": 0, "reference_calls": len(plans),
        "changed_reference_calls": sum(row["references"] != row["original_references"] for row in plans),
        "fb_diagnostic_policy": "EXISTING_VENDOR_DEFAULTS_NOT_BACKGROUND_CONFIDENCE",
        "causal_training_input": False, "training_eligible": False, "control_ground_truth": False,
        "visual_improvement": "NOT_ESTABLISHED"}
    write_json(root / "RECEIPT.json", payload)
    return payload
