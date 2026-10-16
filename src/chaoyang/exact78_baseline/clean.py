"""In-process ProPainter (pinned vendor weights), same inference path as
``vendor/ProPainter/inference_propainter.py`` without tracing or file IO."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


VENDOR = Path("vendor/ProPainter")
SETTINGS = {"mask_dilation": 4, "raft_iter": 20, "neighbor_length": 10, "ref_stride": 10,
            "subvideo_length": 80, "fp16": True}


def _get_ref_index(mid: int, neighbors: list[int], length: int, stride: int, ref_num: int) -> list[int]:
    if ref_num == -1:
        return [i for i in range(0, length, stride) if i not in neighbors]
    start = max(0, mid - stride * (ref_num // 2))
    end = min(length, mid + stride * (ref_num // 2))
    refs = [i for i in range(start, end, stride) if i not in neighbors]
    return refs[:ref_num + 1]


class ProPainter:
    def __init__(self, repo_root: Path, settings: dict | None = None):
        import torch

        root = (repo_root / VENDOR).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from model.modules.flow_comp_raft import RAFT_bi
        from model.propainter import InpaintGenerator
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet

        self.torch = torch
        self.settings = {**SETTINGS, **(settings or {})}
        self.device = torch.device("cuda")
        weights = root / "weights"
        self.raft = RAFT_bi(str(weights / "raft-things.pth"), self.device)
        self.flow = RecurrentFlowCompleteNet(str(weights / "recurrent_flow_completion.pth"))
        for parameter in self.flow.parameters():
            parameter.requires_grad = False
        self.flow.to(self.device).eval()
        self.model = InpaintGenerator(model_path=str(weights / "ProPainter.pth")).to(self.device).eval()
        if self.settings["fp16"]:
            self.flow, self.model = self.flow.half(), self.model.half()
        self.weights = {name: str(weights / name) for name in
                        ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")}

    def inpaint(self, frames_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """frames (T,H,W,3) uint8 RGB, masks (T,H,W) bool -> (T,H,W,3) uint8 RGB (only masked pixels change)."""
        torch, s = self.torch, self.settings
        length, height, width = masks.shape
        if height % 8 or width % 8:
            raise ValueError("PROPAINTER_SIZE_NOT_MULTIPLE_OF_8")
        kernel = np.ones((3, 3), np.uint8)
        dilated = np.stack([cv2.dilate(m.astype(np.uint8), kernel, iterations=s["mask_dilation"])
                            for m in masks]).astype(np.float32)
        # SAM3.1 enables a process-wide bf16 autocast; ProPainter runs in its own fp16/fp32.
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            frames = torch.from_numpy(frames_rgb).to(self.device).permute(0, 3, 1, 2).float()[None] / 127.5 - 1
            mask_t = torch.from_numpy(dilated).to(self.device)[None, :, None]
            short = 12 if width <= 640 else (8 if width <= 720 else 4)
            forward, backward = [], []
            for f in range(0, length, short):
                end = min(length, f + short)
                chunk = frames[:, f:end] if f == 0 else frames[:, f - 1:end]
                if chunk.shape[1] < 2:
                    continue
                flow_f, flow_b = self.raft(chunk, iters=s["raft_iter"])
                forward.append(flow_f)
                backward.append(flow_b)
            gt = (torch.cat(forward, 1), torch.cat(backward, 1))
            torch.cuda.empty_cache()
            if s["fp16"]:
                frames, mask_t = frames.half(), mask_t.half()
                gt = (gt[0].half(), gt[1].half())
            flow_length = gt[0].shape[1]
            sub = s["subvideo_length"]
            if flow_length > sub:
                pred_f, pred_b, pad = [], [], 5
                for f in range(0, flow_length, sub):
                    s_f, e_f = max(0, f - pad), min(flow_length, f + sub + pad)
                    pad_s, pad_e = f - s_f, e_f - min(flow_length, f + sub)
                    part = (gt[0][:, s_f:e_f], gt[1][:, s_f:e_f])
                    pred, _ = self.flow.forward_bidirect_flow(part, mask_t[:, s_f:e_f + 1])
                    pred = self.flow.combine_flow(part, pred, mask_t[:, s_f:e_f + 1])
                    pred_f.append(pred[0][:, pad_s:e_f - s_f - pad_e])
                    pred_b.append(pred[1][:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                flows = (torch.cat(pred_f, 1), torch.cat(pred_b, 1))
            else:
                pred, _ = self.flow.forward_bidirect_flow(gt, mask_t)
                flows = self.flow.combine_flow(gt, pred, mask_t)
            masked = frames * (1 - mask_t)
            prop_len = min(100, sub)
            if length > prop_len:
                updated, updated_masks, pad = [], [], 10
                for f in range(0, length, prop_len):
                    s_f, e_f = max(0, f - pad), min(length, f + prop_len + pad)
                    pad_s, pad_e = f - s_f, e_f - min(length, f + prop_len)
                    b, t = mask_t[:, s_f:e_f].shape[:2]
                    imgs, local = self.model.img_propagation(
                        masked[:, s_f:e_f], (flows[0][:, s_f:e_f - 1], flows[1][:, s_f:e_f - 1]),
                        mask_t[:, s_f:e_f], "nearest")
                    part = frames[:, s_f:e_f] * (1 - mask_t[:, s_f:e_f]) + imgs.view(b, t, 3, height, width) * mask_t[:, s_f:e_f]
                    updated.append(part[:, pad_s:e_f - s_f - pad_e])
                    updated_masks.append(local.view(b, t, 1, height, width)[:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                updated_frames, updated_mask = torch.cat(updated, 1), torch.cat(updated_masks, 1)
            else:
                b, t = mask_t.shape[:2]
                imgs, local = self.model.img_propagation(masked, flows, mask_t, "nearest")
                updated_frames = frames * (1 - mask_t) + imgs.view(b, t, 3, height, width) * mask_t
                updated_mask = local.view(b, t, 1, height, width)
            stride = s["neighbor_length"] // 2
            ref_num = sub // s["ref_stride"] if length > sub else -1
            result = np.zeros((length, height, width, 3), np.uint8)
            visits = np.zeros(length, np.int32)
            binary = dilated[..., None]
            for f in range(0, length, stride):
                neighbors = list(range(max(0, f - stride), min(length, f + stride + 1)))
                refs = _get_ref_index(f, neighbors, length, s["ref_stride"], ref_num)
                ids = neighbors + refs
                pred = self.model(updated_frames[:, ids], (flows[0][:, neighbors[:-1]], flows[1][:, neighbors[:-1]]),
                                  mask_t[:, ids], updated_mask[:, ids], len(neighbors))
                pred = ((pred.view(-1, 3, height, width).float() + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
                for i, index in enumerate(neighbors):
                    image = pred[i].astype(np.uint8) * binary[index] + frames_rgb[index] * (1 - binary[index])
                    if visits[index]:
                        image = result[index].astype(np.float32) * 0.5 + image * 0.5
                    result[index] = image.astype(np.uint8)
                    visits[index] += 1
            torch.cuda.empty_cache()
        keep = dilated == 0
        result[keep] = frames_rgb[keep]
        return result
