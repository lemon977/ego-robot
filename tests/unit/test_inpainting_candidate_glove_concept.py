"""Strict opt-in glove concept: CPU synthetic providers, no model/quality claim."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import automatic_masks as masks
from chaoyang.inpainting_candidate import model_stages as stages
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref, verify_sealed_stage

SIG = {'raw_source_sha256': 'a' * 64, 'frame_map_sha256': 'b' * 64,
       'model_sha256': 'c' * 64}
BAD = ['', 'glove', 'gloved hand', 'A gloved hand', masks.PROMPTS['glove'],
       0, True, [], {}, np.str_('a gloved hand')]


def rgb():
    return np.zeros((3, 48, 64, 3), np.uint8)


class TextOnlySam:
    aborted = []

    def __init__(self):
        self.calls = []

    def track_text(self, frames, prompt, anchor):
        self.calls.append((prompt, anchor, len(frames)))
        return {}


def test_collect_omitted_and_none_preserve_exact_legacy_receipt_bytes():
    omitted, explicit = TextOnlySam(), TextOnlySam()
    old = masks.collect_sam_rgb_proposals(omitted, rgb(), **SIG)
    same = masks.collect_sam_rgb_proposals(explicit, rgb(), **SIG, glove_text_prompt=None)
    assert old == same
    expected = {'calls': [{'role': role, 'prompt': prompt, 'anchor_native_frame': 0,
                          'tracks': 0, 'propagation_aborted': []}
                         for role, prompt in masks.PROMPTS.items()],
                'object_provider': 'SAM31_GENERIC_OBJECT_CONCEPT_NOT_ALL_INSTANCES',
                'model_execution_performed_by_caller': True, **SIG,
                'frames_sha256': masks.frames_sha256(rgb())}
    assert old == ([], expected)
    assert canonical_sha(old[1]) == canonical_sha(same[1])
    assert 'glove_text_prompt' not in old[1]
    assert omitted.calls == explicit.calls


def test_collect_changes_one_role_only_and_preserves_global_prompts():
    defaults = dict(masks.PROMPTS)
    before, after = TextOnlySam(), TextOnlySam()
    _, old = masks.collect_sam_rgb_proposals(before, rgb(), **SIG)
    _, new = masks.collect_sam_rgb_proposals(after, rgb(), **SIG, glove_text_prompt='a gloved hand')
    changed = [i for i, pair in enumerate(zip(old['calls'], new['calls'])) if pair[0] != pair[1]]
    assert changed == [list(defaults).index('glove')]
    assert new['calls'][changed[0]] == {**old['calls'][changed[0]], 'prompt': 'a gloved hand'}
    assert new['glove_text_prompt'] == 'a gloved hand'
    assert masks.PROMPTS == defaults
    assert all(call[1:] == (0, 3) for call in after.calls)


def test_collect_glove_and_existing_controller_overrides_are_independent():
    sam = TextOnlySam()
    _, receipt = masks.collect_sam_rgb_proposals(sam, rgb(), **SIG,
        controller_text_prompt='VR controller', glove_text_prompt='a gloved hand')
    actual = {row['role']: row['prompt'] for row in receipt['calls']}
    assert actual == {**masks.PROMPTS, 'glove': 'a gloved hand', 'controller': 'VR controller'}


@pytest.mark.parametrize('bad', BAD)
def test_invalid_collect_override_rejected_before_provider(bad):
    sam = TextOnlySam()
    with pytest.raises(TypeError, match='GLOVE_TEXT_PROMPT_NONE_OR_A_GLOVED_HAND_REQUIRED'):
        masks.collect_sam_rgb_proposals(sam, rgb(), **SIG, glove_text_prompt=bad)
    assert sam.calls == []


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(stages, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fake-glove')
    source, weight = tmp_path / 'raw.h264', tmp_path / 'sam.pt'
    source.write_bytes(b'synthetic raw only'); weight.write_bytes(b'never loaded')
    owner = {'execution_allowed': True, 'algorithm_execution_allowed': True,
             'task_id': 'CPU_FAKE_GLOVE_NOT_MODEL_TASK', 'run_signature': 'a' * 64,
             'fencing_token': 'fake-fence', 'gpu_uuid': 'GPU-fake-glove', 'worker_logical_index': 0}
    return SimpleNamespace(root=tmp_path, source=file_ref(source), weight=file_ref(weight),
                           raw=rgb(), mapping=[{'native_frame_index': i, 'decoded_display_index': i}
                                               for i in range(3)], owner=owner)


class FakeTensor:
    def __init__(self, values, dtype):
        self.value = np.asarray(values, dtype=dtype)

    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


@pytest.fixture
def fake_tensors(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        tensor=lambda values, dtype: FakeTensor(values, dtype), float32=np.float32, int64=np.int64))


class PointModel:
    def __init__(self, sam):
        self.sam = sam
        self.submitted = []
        self.propagations = []

    def init_state(self, **kwargs):
        self.count = len(kwargs['resource_path'])
        return {'ids': [], 'cached_frame_outputs': {}}

    def _cache_frame_outputs(self, state, frame, outputs):
        state['cached_frame_outputs'][frame] = outputs

    def out(self, state):
        return {'out_binary_masks': np.stack([self.sam.mask] * len(state['ids'])),
                'out_obj_ids': np.asarray(state['ids'], np.int64)}

    def add_prompt(self, **kwargs):
        self.submitted.append(kwargs)
        state, obj = kwargs['inference_state'], kwargs['obj_id']
        if obj not in state['ids']: state['ids'].append(obj)
        return kwargs['frame_idx'], self.out(state)

    def propagate_in_video(self, **kwargs):
        self.propagations.append(kwargs)
        indices = (range(kwargs['start_frame_idx'], -1, -1) if kwargs['reverse']
                   else range(kwargs['start_frame_idx'], self.count))
        for i in indices: yield i, self.out(kwargs['inference_state'])


class GloveSam(TextOnlySam):
    def __init__(self):
        super().__init__()
        self.mask = np.zeros((48, 64), bool)
        self.mask[12:30, 14:24] = True
        self.model = PointModel(self)

    def track_text(self, frames, prompt, anchor):
        super().track_text(frames, prompt, anchor)
        if prompt in (masks.PROMPTS['glove'], 'a gloved hand'):
            return {7: {i: np.packbits(self.mask) for i in range(len(frames))}}
        return {}


def stage_run(env, folder, sam, **kwargs):
    return stages.run_automatic_mask_stage(env.raw, sam=sam, output_dir=env.root / folder,
        raw_source_ref=env.source, frame_map=env.mapping, model_ref=env.weight,
        component_signature='d' * 64, authorization=lambda: dict(env.owner), **kwargs)


def test_stage_none_does_not_forward_new_keyword_or_change_metadata(stage_env, fake_tensors, monkeypatch):
    original = stages.collect_sam_rgb_proposals
    passed = []
    def collect(*args, **kwargs):
        passed.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(stages, 'collect_sam_rgb_proposals', collect)
    baseline = stage_run(stage_env, 'default', GloveSam())
    explicit = stage_run(stage_env, 'none', GloveSam(), glove_text_prompt=None)
    assert all('glove_text_prompt' not in kw for kw in passed)
    for receipt in (baseline, explicit):
        assert 'glove_text_prompt' not in receipt['actual_inputs']
    for name in ('SAM_TEXT.json', 'MASK_METADATA.json', 'AUTOMATIC_POINTS.json', 'FRAME_STATUS.json'):
        assert (stage_env.root / 'default' / name).read_bytes() == (stage_env.root / 'none' / name).read_bytes()


def test_opt_in_real_stage_fake_text_and_point_seal_actual_metadata(stage_env, fake_tensors):
    sam = GloveSam()
    receipt = stage_run(stage_env, 'candidate', sam, glove_text_prompt='a gloved hand',
                        controller_text_prompt='VR controller',
                        text_wearer_retention=masks.TEXT_WEARER_RETENTION,
                        source_aware_retention_veto=masks.SOURCE_AWARE_RETENTION_VETO)
    root = stage_env.root / 'candidate'
    verify_sealed_stage(receipt, 'd' * 64, output_root=root)
    assert receipt['native_input_frames'] == receipt['native_output_frames'] == 3
    assert receipt['actual_inputs']['glove_text_prompt'] == 'a gloved hand'
    assert receipt['actual_inputs']['controller_text_prompt'] == 'VR controller'
    text = json.loads((root / 'SAM_TEXT.json').read_text())
    assert text['glove_text_prompt'] == 'a gloved hand'
    assert {r['role']: r['prompt'] for r in text['calls']} == {
        **masks.PROMPTS, 'glove': 'a gloved hand', 'controller': 'VR controller'}
    metadata = json.loads((root / 'MASK_METADATA.json').read_text())
    assert metadata['prompts']['glove'] == metadata['glove_text_prompt'] == 'a gloved hand'
    assert metadata['source_aware_retention_veto'] == masks.SOURCE_AWARE_RETENTION_VETO
    assert metadata['text_wearer_retention'] == masks.TEXT_WEARER_RETENTION
    assert metadata['glove_support_rule'] == 'INDEPENDENT_CURRENT_RGB_PROPOSAL_NOT_HUMAN_MEASUREMENT'
    assert sam.model.submitted and sam.model.propagations
    assert all(p['text_str'] is None and p['output_prob_thresh'] == .5 for p in sam.model.submitted)
    assert all(p['output_prob_thresh'] == .5 for p in sam.model.propagations)
    assert receipt['manual_points_submitted'] == 0 and receipt['text_prompt_invocations'] == 6
    with np.load(root / 'MASK_ARRAYS.npz', allow_pickle=False) as arrays:
        assert not arrays['human'].any()
        assert arrays['equipment__glove'][:, 20, 18].all()
        assert arrays['accepted_point_wearer_support'][:, 20, 18].all()
        assert 'projection_association_hints' not in arrays.files


@pytest.mark.parametrize('bad', BAD)
def test_invalid_stage_override_refused_before_source_output_or_provider(stage_env, bad):
    sam = TextOnlySam()
    with pytest.raises(TypeError, match='GLOVE_TEXT_PROMPT_NONE_OR_A_GLOVED_HAND_REQUIRED'):
        stages.run_automatic_mask_stage(stage_env.raw, sam=sam, output_dir=stage_env.root / 'bad',
            raw_source_ref={}, frame_map=[], model_ref={}, component_signature='d' * 64,
            authorization=lambda: dict(stage_env.owner), glove_text_prompt=bad)
    assert sam.calls == [] and not (stage_env.root / 'bad').exists()


def test_new_prompt_does_not_bypass_existing_authority_gate(stage_env):
    stage_env.owner['algorithm_execution_allowed'] = False
    sam = TextOnlySam()
    with pytest.raises(stages.ModelStageError, match='REGISTERED_GPU_MODEL_AUTHORITY_REQUIRED'):
        stage_run(stage_env, 'unfenced', sam, glove_text_prompt='a gloved hand')
    assert sam.calls == [] and not (stage_env.root / 'unfenced').exists()


def test_new_prompt_rechecks_fence_around_actual_provider_call(stage_env):
    class DriftSam(TextOnlySam):
        def track_text(self, frames, prompt, anchor):
            result = super().track_text(frames, prompt, anchor)
            stage_env.owner['fencing_token'] = 'changed-fence'
            return result
    sam = DriftSam()
    with pytest.raises(stages.ModelStageError, match='MODEL_TASK_WRITER_OR_CARD_IDENTITY_DRIFT'):
        stage_run(stage_env, 'drift', sam, glove_text_prompt='a gloved hand')
    assert len(sam.calls) == 1
    assert (stage_env.root / 'drift/FAILED.json').exists()
    assert not (stage_env.root / 'drift/RECEIPT.json').exists()
