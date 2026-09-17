#!/usr/bin/env python3
"""Observe SAM3.1 predictor state for Poker015 frames 0..2 without mutation."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import types
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAM_ROOT = ROOT / "vendor/SAM3"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.pipeline.sam31_temporal_fullsession_integration_r22 import FrozenSeedReplay, replay_seed_then_collect
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers

ANNOTATIONS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/PROMPT_ANNOTATIONS_FROZEN.json"
BOOTSTRAP = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/attempts/attempt_0002/PROMPT_CANDIDATES.json"
CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return sorted(int(x) for x in np.asarray(list(value) if isinstance(value, set) else value).reshape(-1))


def internal_row(frame: int, out: dict[str, Any], removed: Any, suppressed: Any, unconfirmed: Any, public: dict[str, Any]) -> dict[str, Any]:
    masks = out.get("obj_id_to_mask", {})
    pre_ids = sorted(int(x) for x in masks)
    nonempty = {}
    for object_id, mask in masks.items():
        nonempty[str(int(object_id))] = bool(torch.as_tensor(mask).any().item())
    scores = {str(int(k)): scalar(v) for k, v in out.get("obj_id_to_score", {}).items()}
    sam2_scores = {str(int(k)): scalar(v) for k, v in out.get("obj_id_to_sam2_score", {}).items()}
    return {
        "frame_index": frame,
        "pre_public_object_ids": pre_ids,
        "pre_public_mask_nonempty": nonempty,
        "removed_obj_ids": ids(removed),
        "suppressed_obj_ids": ids(suppressed),
        "unconfirmed_obj_ids": ids(unconfirmed),
        "detector_scores": scores,
        "sam2_scores": sam2_scores,
        "public_object_ids": ids(public.get("out_obj_ids", [])),
        "public_mask_nonempty": [bool(np.asarray(mask).any()) for mask in public.get("out_binary_masks", [])],
    }


class Instrumentation:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.enabled = False
        self.rows: list[dict[str, Any]] = []
        self.next_frame = 0
        self.mode = ""
        self.original_scalar = model._postprocess_output
        self.original_batch = model._postprocess_output_batched

    def install(self) -> None:
        owner = self

        def scalar_hook(this: Any, inference_state: Any, out: Any, removed_obj_ids=None, suppressed_obj_ids=None, unconfirmed_obj_ids=None):
            public = owner.original_scalar(inference_state, out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids)
            if owner.enabled and int(getattr(this, "postprocess_batch_size", 1)) == 1:
                owner.mode = "_postprocess_output"
                owner.rows.append(internal_row(owner.next_frame, out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids, public))
                owner.next_frame += 1
            return public

        def batch_hook(this: Any, height: int, width: int, batched_outs: Any):
            public_rows = owner.original_batch(height, width, batched_outs)
            if owner.enabled and int(getattr(this, "postprocess_batch_size", 1)) > 1:
                owner.mode = "_postprocess_output_batched"
                for packed, public in zip(batched_outs, public_rows):
                    out, removed, suppressed, unconfirmed = packed
                    owner.rows.append(internal_row(owner.next_frame, out, removed, suppressed, unconfirmed, public))
                    owner.next_frame += 1
            return public_rows

        self.model._postprocess_output = types.MethodType(scalar_hook, self.model)
        self.model._postprocess_output_batched = types.MethodType(batch_hook, self.model)

    def restore(self) -> None:
        self.model._postprocess_output = self.original_scalar
        self.model._postprocess_output_batched = self.original_batch


class StreamProxy:
    def __init__(self, adapter: Any, instrumentation: Instrumentation) -> None:
        self.adapter = adapter
        self.instrumentation = instrumentation
        self.model = adapter.model

    def _session(self, session_id: str) -> Any:
        return self.adapter._session(session_id)

    def handle_request(self, request: dict[str, Any]) -> Any:
        return self.adapter.handle_request(request)

    def handle_stream_request(self, request: dict[str, Any]):
        self.instrumentation.enabled = True
        try:
            yield from self.adapter.handle_stream_request(request)
        finally:
            self.instrumentation.enabled = False


def seal(out: Path, status: str, observations: list[dict[str, Any]], public_rows: list[dict[str, Any]], error: str | None, started: float) -> None:
    write_json(out / "INTERNAL_STATE_OBSERVATIONS.json", {"status": status, "observations": observations, "public_rows": public_rows})
    first_loss = next((row["frame_index"] for row in observations if 0 not in row["public_object_ids"]), None)
    metrics = {"status": status, "observed_frames": len(observations), "first_public_loss_frame": first_loss, "wall_seconds": time.perf_counter()-started, "inference_changed": False}
    write_json(out / "METRICS.json", metrics)
    write_json(out / "RUN_RECEIPT.json", {"task_id": "mask_sam31_predictor_internal_state_instrumentation_r22", "attempt_id": out.name, "status": status, "created_at": now(), "host": socket.gethostname(), "pid": os.getpid(), "code": ref(Path(__file__)), "error": error, "inference_changed": False, "current_ledger_modified": False})
    conclusion = "See INTERNAL_STATE_OBSERVATIONS.json; the hook only observes arguments and return values of the existing postprocess method."
    (out / "DECISION.md").write_text(f"# SAM3.1 predictor 内部状态观测\n\n状态：`{status}`。\n\n{conclusion}\n", encoding="utf-8")
    write_json(out / "NEXT_ACTION.json", {"status": status, "next_task_id": None, "claim_limit": "One Poker target, frames 0..2, read-only instrumentation only."})
    artifacts = [out/name for name in ["INTERNAL_STATE_OBSERVATIONS.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json"]]
    write_json(out / "ARTIFACT_MANIFEST.json", {"status": status, "artifacts": [ref(p) for p in artifacts]})
    write_json(out / "RESULT.json", {"task_id": "mask_sam31_predictor_internal_state_instrumentation_r22", "attempt_id": out.name, "status": status, "created_at": now(), "metrics": ref(out/"METRICS.json"), "observations": ref(out/"INTERNAL_STATE_OBSERVATIONS.json"), "manifest": ref(out/"ARTIFACT_MANIFEST.json"), "error": error, "authority_promoted": False, "claim_limit": "Read-only 3-frame internal diagnostic; no quality retry or authority."})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); status = "BLOCKED_REFERENCE_PROOF"; error = None
    observations: list[dict[str, Any]] = []; public_rows: list[dict[str, Any]] = []; adapter = None; hook = None
    try:
        if sha(CHECKPOINT) != CHECKPOINT_SHA:
            raise RuntimeError("checkpoint SHA mismatch")
        annotations = json.loads(ANNOTATIONS.read_text())["annotations"]["poker015"]
        bootstrap = json.loads(BOOTSTRAP.read_text())["tasks"]["poker015"]["object"]
        if annotations["session"] != "play_cards_0901_015" or annotations["source_frame"] != 0:
            raise RuntimeError("frozen Poker identity mismatch")
        frame_root = Path(annotations["source_rgb"]["path"]).parents[1]
        frames = out / "input_frames"; frames.mkdir()
        for frame in range(3):
            source = frame_root / f"{frame:05d}" / "rgb.png"
            if not source.is_file(): raise RuntimeError(f"missing frame {frame}")
            os.symlink(source, frames/f"{frame:05d}.png")
        adapter, _ = sam31_compat_adapter_v1.build_pinned_adapter(official_code_root=SAM_ROOT, checkpoint_path=CHECKPOINT)
        hook = Instrumentation(adapter.model); hook.install()
        proxy = StreamProxy(adapter, hook)
        spec = FrozenSeedReplay(session_id=f"poker015-internal-{out.name}", accepted_object_id=int(bootstrap["selected_object_id"]), point_xy=tuple(annotations["positive_point_xy"]), box_xyxy=tuple(annotations["box_xyxy"]), text="a purple-backed playing card", start_frame_index=0, frame_count=3, source_frame_offset=0, height=960, width=1280)
        _, public_rows, _ = replay_seed_then_collect(proxy, frames, spec, sam_helpers.normalize)
        observations = hook.rows
        if len(observations) != 3 or [r["frame_index"] for r in observations] != [0,1,2]:
            raise RuntimeError(f"instrumentation coverage mismatch: {len(observations)}")
        status = "PASSED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if hook is not None: hook.restore()
        if adapter is not None: adapter.predictor.shutdown()
    seal(out, status, observations, public_rows, error, started)
    print(json.dumps({"status": status, "result": ref(out/"RESULT.json")}, ensure_ascii=False))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
