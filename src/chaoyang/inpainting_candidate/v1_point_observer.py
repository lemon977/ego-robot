"""Read-only copies of public POINT outputs before the project adapter.

This proxy changes no prompts, thresholds, state, IDs, return objects or masks.
Binary public outputs are not neural logits or independently observed truth.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np


class PointOutputObserver:
    def __init__(self, sam, native_shape, *, max_bytes=128 * 1024 * 1024):
        self.native_shape = tuple(native_shape)
        if (len(self.native_shape) != 3 or any(type(x) is not int or x <= 0 for x in self.native_shape)
                or type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError('POINT_OBSERVER_NATIVE_DOMAIN_AND_RESOURCE_BUDGET_REQUIRED')
        self.max_bytes = max_bytes
        self.copied_bytes = 0
        self.events = []
        self.arrays = {}
        self.sam = SimpleNamespace(model=_ObservedPointModel(sam.model, self))

    def capture(self, phase, returned):
        from chaoyang.pick_place_1008.sam_point_adapter import _normalized_outputs
        frame, outputs = returned
        n, h, w = self.native_shape
        if not isinstance(frame, (int, np.integer)) or not 0 <= frame < n:
            raise ValueError('POINT_OBSERVER_NATIVE_FRAME_REQUIRED')
        masks, ids = _normalized_outputs(outputs, h, w)
        # packbits owns independent storage even if vendor reuses CPU buffers.
        packed = np.stack([np.packbits(mask.reshape(-1)) for mask in masks]) if len(masks) else np.zeros((0, (h*w+7)//8), np.uint8)
        required = packed.nbytes + ids.nbytes
        if self.copied_bytes + required > self.max_bytes:
            raise ValueError('POINT_OBSERVER_RESOURCE_LIMIT')
        key = f'event_{len(self.events):04d}'
        self.arrays[key] = packed
        self.copied_bytes += required
        self.events.append({'event': len(self.events), 'phase': phase,
                            'native_frame_index': int(frame), 'obj_ids': ids.tolist(),
                            'mask_areas': masks.sum(axis=(1, 2)).astype(int).tolist(),
                            'array_key': key})

    def packed_adapter_tracks(self, tracks):
        n, h, w = self.native_shape
        expected_size = (h * w + 7) // 8
        arrays, objects = {}, []
        for obj, frames in sorted(tracks.items()):
            packed = np.zeros((n, expected_size), np.uint8)
            presence = np.zeros(n, bool)
            for frame, value in frames.items():
                value = np.asarray(value)
                if (not 0 <= frame < n or value.shape != (expected_size,)
                        or value.dtype != np.uint8):
                    raise ValueError('POINT_OBSERVER_ADAPTER_TRACK_DOMAIN')
                packed[frame] = value
                presence[frame] = True
            key = f'obj_{obj}'
            arrays[key] = packed
            arrays[key + '_present'] = presence
            objects.append({'obj_id': int(obj), 'array_key': key,
                            'presence_key': key + '_present',
                            'native_frame_indices': np.flatnonzero(presence).tolist()})
        return arrays, objects

    def receipt(self, objects):
        return {'schema': 'inpainting-v1-point-output-readonly-observer-v1',
                'native_shape': list(self.native_shape), 'copied_bytes': self.copied_bytes,
                'resource_limit_bytes': self.max_bytes, 'events': self.events,
                'adapter_track_objects': objects,
                'observation_level': 'PUBLIC_BINARY_MODEL_OUTPUT_BEFORE_PROJECT_ADAPTER',
                'pre_threshold_logits_observed': False, 'model_inputs_changed': False,
                'return_objects_replaced': False, 'vendor_or_default_modified': False,
                'visual_improvement_claim': False, 'human_review': 'PENDING',
                'control_ground_truth': False, 'training_eligible': False}


class _ObservedPointModel:
    def __init__(self, provider, observer):
        self.provider = provider
        self.observer = observer

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def add_prompt(self, *args, **kwargs):
        result = self.provider.add_prompt(*args, **kwargs)
        self.observer.capture('add_prompt', result)
        return result

    def propagate_in_video(self, *args, **kwargs):
        phase = 'propagate_reverse' if kwargs.get('reverse') else 'propagate_forward'
        for result in self.provider.propagate_in_video(*args, **kwargs):
            self.observer.capture(phase, result)
            yield result
