#!/usr/bin/env python3
"""Bounded SAM3.1 state-route diagnostic for the frozen Poker015 target.

The experiment does not retry the failed RC1 Mask task.  It compares the
pinned public predictor entry, the compatibility adapter's low-level mapping,
and stable-ID tracking with/without the historical ignored priming object on
three tiny sequences: repeated seed image, known translation, and a real
16-frame prefix.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAM_ROOT = ROOT / "vendor/SAM3"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.pipeline.sam31_temporal_fullsession_integration_r22 import (
    FrozenSeedReplay,
    replay_detector_then_track_stable_point_id,
)
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
ANNOTATIONS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/PROMPT_ANNOTATIONS_FROZEN.json"
BOOTSTRAP = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/attempts/attempt_0002/PROMPT_CANDIDATES.json"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


class DirectModelRouter:
    """Minimal direct route over the pinned official low-level model."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._sessions: dict[str, dict[str, Any]] = {}

    def _session(self, session_id: str) -> dict[str, Any]:
        return self._sessions[session_id]

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request["type"] == "start_session":
            session_id = str(request["session_id"])
            self._sessions[session_id] = self.model.init_state(
                resource_path=request["resource_path"],
                offload_video_to_cpu=False,
                async_loading_frames=False,
            )
            return {"session_id": session_id}
        if request["type"] == "close_session":
            self._sessions.pop(str(request["session_id"])).clear()
            return {"is_success": True}
        raise RuntimeError(f"unsupported direct route request: {request['type']}")

    def handle_stream_request(self, request: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        for frame, output in self.model.propagate_in_video(
            inference_state=self._session(str(request["session_id"])),
            start_frame_idx=request.get("start_frame_index"),
            max_frame_num_to_track=request.get("max_frame_num_to_track"),
            reverse=request.get("propagation_direction", "forward") == "backward",
            output_prob_thresh=float(request.get("output_prob_thresh", 0.5)),
        ):
            yield {"frame_index": frame, "outputs": output}


def _prepare_sequences(output: Path, source_rgb: Path) -> dict[str, dict[str, Any]]:
    image = cv2.imread(str(source_rgb), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("frozen source decode failed")
    frame_root = source_rgb.parents[1]
    result: dict[str, dict[str, Any]] = {}

    repeated = output / "inputs/repeated_seed_8"
    repeated.mkdir(parents=True)
    for frame in range(8):
        os.symlink(source_rgb, repeated / f"{frame:05d}.png")
    result["repeated_seed_8"] = {"path": repeated, "frames": 8, "source_loader": lambda i: image.copy()}

    translated = output / "inputs/known_translation_8"
    translated.mkdir(parents=True)
    translated_images: list[np.ndarray] = []
    for frame in range(8):
        matrix = np.float32([[1, 0, 2 * frame], [0, 1, frame]])
        shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        path = translated / f"{frame:05d}.png"
        if not cv2.imwrite(str(path), shifted):
            raise RuntimeError("translation fixture write failed")
        translated_images.append(shifted)
    result["known_translation_8"] = {"path": translated, "frames": 8, "source_loader": lambda i: translated_images[i].copy()}

    real = output / "inputs/real_prefix_16"
    real.mkdir(parents=True)
    real_paths = []
    for frame in range(16):
        source = frame_root / f"{frame:05d}" / "rgb.png"
        if not source.is_file():
            raise RuntimeError(f"missing real frame {frame}")
        os.symlink(source, real / f"{frame:05d}.png")
        real_paths.append(source)
    result["real_prefix_16"] = {
        "path": real,
        "frames": 16,
        "source_loader": lambda i: cv2.imread(str(real_paths[i]), cv2.IMREAD_COLOR),
    }
    return result


def _run_variant(
    router: Any,
    *,
    sequence: str,
    sequence_spec: dict[str, Any],
    seed: dict[str, Any],
    accepted_id: int,
    priming: bool,
    route_name: str,
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    spec = FrozenSeedReplay(
        session_id=f"diag-{sequence}-{route_name}",
        accepted_object_id=accepted_id,
        point_xy=tuple(seed["positive_point_xy"]),
        box_xyxy=tuple(seed["box_xyxy"]),
        text="a purple-backed playing card",
        start_frame_index=0,
        frame_count=int(sequence_spec["frames"]),
        source_frame_offset=0,
        height=960,
        width=1280,
    )
    seed_rows, rows, masks = replay_detector_then_track_stable_point_id(
        router,
        Path(sequence_spec["path"]),
        spec,
        sam_helpers.normalize,
        tracking_object_id=10001 + accepted_id,
        priming_object_id=9901 if priming else None,
    )
    present = sum(bool(row["present"]) for row in rows)
    return {
        "route": route_name,
        "priming": priming,
        "frames": len(rows),
        "present_frames": present,
        "presence_fraction": present / max(len(rows), 1),
        "rows": rows,
        "seed_rows": seed_rows,
    }, masks


def _render(output: Path, scenario: str, spec: dict[str, Any], variants: list[tuple[str, dict[str, Any], dict[int, np.ndarray]]]) -> Path:
    cell_w, cell_h = 640, 480
    columns = 1 + len(variants)
    target = output / f"SAM31_{scenario}_路由与Priming对照.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{cell_w * columns}x{cell_h}", "-r", "4",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    font = ImageFont.truetype(str(FONT), 22)
    for frame in range(int(spec["frames"])):
        raw = spec["source_loader"](frame)
        cells = []
        for label, row, masks in [("原始画面", {}, {})] + variants:
            cell = raw.copy()
            mask = masks.get(frame)
            if mask is not None and mask.any():
                cell[mask] = (0.45 * cell[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            cell = cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, cell_w, 52), fill=(0, 0, 0))
            reason = ""
            if row.get("rows") and frame < len(row["rows"]):
                reason = str(row["rows"][frame].get("reason", ""))
            draw.text((8, 6), f"{label} | 帧{frame:02d} {reason}", font=font, fill=(255, 255, 255))
            cells.append(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        assert encoder.stdin is not None
        encoder.stdin.write(np.concatenate(cells, axis=1).tobytes())
    assert encoder.stdin is not None
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("diagnostic video encode failed")
    return target


def _public_entry_smoke(predictor: Any, frames: Path) -> dict[str, Any]:
    session_id = None
    try:
        response = predictor.handle_request({"type": "start_session", "resource_path": str(frames)})
        session_id = response["session_id"]
        prompt = predictor.handle_request({
            "type": "add_prompt", "session_id": session_id, "frame_index": 0,
            "text": "a purple-backed playing card",
        })
        output = prompt.get("outputs", {})
        ids = output.get("out_obj_ids", [])
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().numpy()
        return {"status": "PASSED", "seed_public_ids": [int(x) for x in np.asarray(ids).reshape(-1)]}
    except Exception as error:
        return {"status": "FAILED_RUNTIME", "error": f"{type(error).__name__}: {error}"}
    finally:
        if session_id is not None:
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    adapter = None
    try:
        annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]["poker015"]
        bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))["tasks"]["poker015"]["object"]
        source_rgb = Path(annotations["source_rgb"]["path"]).resolve(strict=True)
        sequences = _prepare_sequences(output, source_rgb)
        adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
            official_code_root=SAM_ROOT, checkpoint_path=CHECKPOINT
        )
        direct = DirectModelRouter(adapter.model)
        public = _public_entry_smoke(adapter.predictor, Path(sequences["repeated_seed_8"]["path"]))
        matrix: dict[str, Any] = {}
        videos = []
        accepted_id = int(bootstrap["selected_object_id"])
        for scenario, sequence_spec in sequences.items():
            definitions = [
                ("适配器_有Priming", adapter, True),
                ("适配器_无Priming", adapter, False),
            ]
            if scenario == "repeated_seed_8":
                definitions.append(("官方低层直连_无Priming", direct, False))
            variant_payloads = []
            matrix[scenario] = {}
            for label, router, priming in definitions:
                row, masks = _run_variant(
                    router, sequence=scenario, sequence_spec=sequence_spec,
                    seed=annotations, accepted_id=accepted_id, priming=priming,
                    route_name=label,
                )
                matrix[scenario][label] = row
                variant_payloads.append((label, row, masks))
            videos.append(artifact_ref(_render(output, scenario, sequence_spec, variant_payloads)))
        repeated = matrix["repeated_seed_8"]
        adapter_direct_equal = (
            repeated["适配器_无Priming"]["rows"]
            == repeated["官方低层直连_无Priming"]["rows"]
        )
        result = {
            "schema_version": "chaoyang-sam31-state-route-diagnostic-v1",
            "created_at": now_iso(),
            "task_id": "research_sam31_state_route_diagnostic",
            "status": "PASSED_DIAGNOSTIC",
            "public_official_entry": public,
            "adapter_vs_direct_low_level_equal_on_repeated_fixture": adapter_direct_equal,
            "matrix": matrix,
            "videos": videos,
            "inputs": {
                "annotations": artifact_ref(ANNOTATIONS),
                "bootstrap": artifact_ref(BOOTSTRAP),
                "checkpoint": artifact_ref(CHECKPOINT),
                "adapter": artifact_ref(Path(sam31_compat_adapter_v1.__file__).resolve()),
            },
            "build_evidence": build_evidence,
            "wall_seconds": time.perf_counter() - started,
            "authority_promoted": False,
            "claim_limit": "Short frozen Poker015 state-route diagnostic only; not Mask accuracy, RC1 retry, or authority.",
        }
        atomic_json(output / "METRICS.json", {"matrix": matrix})
        atomic_json(output / "RUN_RECEIPT.json", {
            "task_id": result["task_id"], "status": result["status"], "created_at": result["created_at"],
            "host": socket.gethostname(), "pid": os.getpid(), "code": artifact_ref(Path(__file__).resolve()),
        })
        (output / "DECISION.md").write_text(
            "# SAM3.1 状态路由诊断\n\n这是开发诊断，不修改 RC1 Mask 终态。结论以 RESULT.json 的逐路由记录为准。\n",
            encoding="utf-8",
        )
        atomic_json(output / "NEXT_ACTION.json", {
            "status": "DIAGNOSTIC_COMPLETE", "next_task_id": None,
            "claim_limit": result["claim_limit"],
        })
        artifacts = [output / name for name in ("METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json")]
        atomic_json(output / "ARTIFACT_MANIFEST.json", {
            "status": result["status"], "artifacts": [artifact_ref(path) for path in artifacts] + videos,
        })
        result["manifest"] = artifact_ref(output / "ARTIFACT_MANIFEST.json")
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
        return 0
    finally:
        if adapter is not None:
            adapter.predictor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
