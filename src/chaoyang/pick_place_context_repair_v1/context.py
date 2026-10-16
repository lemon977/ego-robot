from __future__ import annotations
import copy
import cv2
import numpy as np
from chaoyang.pick_place_1008.visual import _rgb_frames, effective_clean_masks
from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index

def model_context_hole_masks(write_effective, protected_visible, *, padding_px=4):
    e=np.asarray(write_effective); p=np.asarray(protected_visible)
    if e.dtype!=np.bool_ or p.dtype!=np.bool_ or e.ndim!=3 or e.shape!=p.shape or not len(e):
        raise ValueError('CONTEXT_HOLE_BOOL_NATIVE_DOMAIN')
    if type(padding_px) is not int or not 0<=padding_px<=8:
        raise ValueError('CONTEXT_HOLE_PADDING_FINITE_0_8')
    if np.any(e&p): raise ValueError('WRITE_E_PROTECTED_OVERLAP')
    if padding_px:
        hole=np.stack([cv2.dilate(v.astype(np.uint8),np.ones((3,3),np.uint8),iterations=padding_px)>0 for v in e]) & ~p
    else: hole=e.copy()
    if np.any(e & ~hole) or np.any(hole&p):raise RuntimeError('CONTEXT_HOLE_CONSERVATION')
    return hole,{'recipe':'MODEL_CONTEXT_EXTRA_PAD4_FINAL_WRITE_E_FIXED_V1' if padding_px==4 else 'CPU_GEOMETRY_ADVISORY_ONLY',
      'padding_px':padding_px,'write_E_pixels_per_frame':e.sum((1,2)).tolist(),
      'model_hole_pixels_per_frame':hole.sum((1,2)).tolist(),
      'context_only_pixels_per_frame':(hole&~e).sum((1,2)).tolist(),
      'model_protected_overlap_pixels_per_frame':(hole&p).sum((1,2)).tolist(),
      'final_write_E_and_P_unchanged':True,'context_only_pixels_written_to_FINAL':False}

def restore_fixed_write_domain(model_result, raw_rgb, write_effective, protected_visible):
    result=np.asarray(model_result); raw=np.asarray(raw_rgb);e=np.asarray(write_effective);p=np.asarray(protected_visible)
    if result.shape!=raw.shape or raw.dtype!=np.uint8 or result.dtype!=np.uint8 or raw.ndim!=4 or raw.shape[-1]!=3 or e.shape!=raw.shape[:3] or p.shape!=e.shape or e.dtype!=np.bool_ or p.dtype!=np.bool_:
        raise ValueError('RESTORE_DOMAIN')
    if np.any(e&p):raise ValueError('WRITE_E_PROTECTED_OVERLAP')
    out=result.copy();out[~e]=raw[~e];out[p]=raw[p]
    if np.any(out[~e]!=raw[~e]) or np.any(out[p]!=raw[p]):raise RuntimeError('RESTORE_FAILED')
    return out
class PrivateContextHolePad4ProtectedProPainter:
    """Protect before every model path, using already-loaded frozen model objects."""

    def __init__(self, parent: Any):
        if dict(parent.settings) != SETTINGS:
            raise ValueError('PROPAINTER_FROZEN_SETTINGS_DRIFT')
        self.parent = parent
        self.settings = dict(parent.settings)
        self.last_validation: dict[str, Any] = {}
        self.effective_masks: np.ndarray | None = None
        self.context_padding_px = 4

    def inpaint(self, frames_rgb: np.ndarray, masks: np.ndarray, protected_visible_masks: np.ndarray) -> np.ndarray:
        frames_rgb = _rgb_frames(frames_rgb)
        if masks.shape != frames_rgb.shape[:3]:
            raise ValueError('PROPAINTER_FRAME_MASK_DOMAIN')
        if dict(self.parent.settings) != SETTINGS or self.settings != SETTINGS:
            raise ValueError('PROPAINTER_FROZEN_SETTINGS_DRIFT')
        effective, validation = effective_clean_masks(masks, protected_visible_masks, self.settings)
        self.effective_masks = effective
        self.last_validation = validation
        torch, s = (self.parent.torch, self.settings)
        length, height, width = masks.shape
        if height % 8 or width % 8:
            raise ValueError('PROPAINTER_SIZE_NOT_MULTIPLE_OF_8')
        if length < 2:
            raise ValueError('PROPAINTER_SINGLE_FRAME_UNSUPPORTED')
        context_hole, context_proof = model_context_hole_masks(effective, protected_visible_masks, padding_px=4)
        validation['private_context_hole_pad4'] = context_proof
        dilated = context_hole.astype(np.float32)
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            frames = torch.from_numpy(frames_rgb).to(self.parent.device).permute(0, 3, 1, 2).float()[None] / 127.5 - 1
            mask_t = torch.from_numpy(dilated).to(self.parent.device)[None, :, None]
            protected_t = torch.from_numpy(protected_visible_masks).to(self.parent.device)[None, :, None]
            if torch.any(mask_t[protected_t] != 0).item():
                raise RuntimeError('PROTECTED_MASK_TENSOR_NOT_ZERO')
            short = 12 if width <= 640 else 8 if width <= 720 else 4
            forward, backward = ([], [])
            for f in range(0, length, short):
                end = min(length, f + short)
                chunk = frames[:, f:end] if f == 0 else frames[:, f - 1:end]
                if chunk.shape[1] < 2:
                    continue
                flow_f, flow_b = self.parent.raft(chunk, iters=s['raft_iter'])
                forward.append(flow_f)
                backward.append(flow_b)
            gt = (torch.cat(forward, 1), torch.cat(backward, 1))
            torch.cuda.empty_cache()
            if s['fp16']:
                frames, mask_t = (frames.half(), mask_t.half())
                gt = (gt[0].half(), gt[1].half())
            flow_length = gt[0].shape[1]
            sub = s['subvideo_length']
            if flow_length > sub:
                pred_f, pred_b, pad = ([], [], 5)
                for f in range(0, flow_length, sub):
                    s_f, e_f = (max(0, f - pad), min(flow_length, f + sub + pad))
                    pad_s, pad_e = (f - s_f, e_f - min(flow_length, f + sub))
                    part = (gt[0][:, s_f:e_f], gt[1][:, s_f:e_f])
                    pred, _ = self.parent.flow.forward_bidirect_flow(part, mask_t[:, s_f:e_f + 1])
                    pred = self.parent.flow.combine_flow(part, pred, mask_t[:, s_f:e_f + 1])
                    pred_f.append(pred[0][:, pad_s:e_f - s_f - pad_e])
                    pred_b.append(pred[1][:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                flows = (torch.cat(pred_f, 1), torch.cat(pred_b, 1))
            else:
                pred, _ = self.parent.flow.forward_bidirect_flow(gt, mask_t)
                flows = self.parent.flow.combine_flow(gt, pred, mask_t)
            masked = frames * (1 - mask_t)
            support = protected_t.expand(-1, -1, 3, -1, -1)
            if not torch.equal(masked[support], frames[support]):
                raise RuntimeError('PROTECTED_RGB_MASKED_BEFORE_MODEL')
            validation['model_protected_rgb_unchanged'] = True
            prop_len = min(100, sub)
            if length > prop_len:
                updated, updated_masks, pad = ([], [], 10)
                for f in range(0, length, prop_len):
                    s_f, e_f = (max(0, f - pad), min(length, f + prop_len + pad))
                    pad_s, pad_e = (f - s_f, e_f - min(length, f + prop_len))
                    b, t = mask_t[:, s_f:e_f].shape[:2]
                    imgs, local = self.parent.model.img_propagation(masked[:, s_f:e_f], (flows[0][:, s_f:e_f - 1], flows[1][:, s_f:e_f - 1]), mask_t[:, s_f:e_f], 'nearest')
                    part = frames[:, s_f:e_f] * (1 - mask_t[:, s_f:e_f]) + imgs.view(b, t, 3, height, width) * mask_t[:, s_f:e_f]
                    updated.append(part[:, pad_s:e_f - s_f - pad_e])
                    updated_masks.append(local.view(b, t, 1, height, width)[:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                updated_frames, updated_mask = (torch.cat(updated, 1), torch.cat(updated_masks, 1))
            else:
                b, t = mask_t.shape[:2]
                imgs, local = self.parent.model.img_propagation(masked, flows, mask_t, 'nearest')
                updated_frames = frames * (1 - mask_t) + imgs.view(b, t, 3, height, width) * mask_t
                updated_mask = local.view(b, t, 1, height, width)
            stride = s['neighbor_length'] // 2
            ref_num = sub // s['ref_stride'] if length > sub else -1
            result = np.zeros((length, height, width, 3), np.uint8)
            visits = np.zeros(length, np.int32)
            binary = dilated[..., None]
            for f in range(0, length, stride):
                neighbors = list(range(max(0, f - stride), min(length, f + stride + 1)))
                refs = _get_ref_index(f, neighbors, length, s['ref_stride'], ref_num)
                ids = neighbors + refs
                pred = self.parent.model(updated_frames[:, ids], (flows[0][:, neighbors[:-1]], flows[1][:, neighbors[:-1]]), mask_t[:, ids], updated_mask[:, ids], len(neighbors))
                pred = ((pred.view(-1, 3, height, width).float() + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
                for i, index in enumerate(neighbors):
                    image = pred[i].astype(np.uint8) * binary[index] + frames_rgb[index] * (1 - binary[index])
                    if visits[index]:
                        image = result[index].astype(np.float32) * 0.5 + image * 0.5
                    result[index] = image.astype(np.uint8)
                    visits[index] += 1
            torch.cuda.empty_cache()
        if not (visits > 0).all():
            raise RuntimeError('PROPAINTER_OUTPUT_FRAME_NOT_VISITED')
        keep = ~effective
        result[keep] = frames_rgb[keep]
        result[protected_visible_masks] = frames_rgb[protected_visible_masks]
        changed = np.any(result != frames_rgb, axis=-1)
        outside = np.count_nonzero(changed & ~effective, axis=(1, 2))
        protected_changed = np.count_nonzero(changed & protected_visible_masks, axis=(1, 2))
        validation['outside_delete_changed_pixels_per_frame'] = outside.tolist()
        validation['protected_changed_pixels_per_frame'] = protected_changed.tolist()
        validation['output_frame_visits'] = visits.tolist()
        validation['status'] = 'PASS' if not outside.any() and (not protected_changed.any()) else 'FAIL'
        if validation['status'] != 'PASS':
            raise RuntimeError('CLEAN_ORIGINAL_PIXEL_PROTECTION_FAILED')
        return result
