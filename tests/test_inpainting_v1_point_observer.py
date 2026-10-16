from types import SimpleNamespace
import numpy as np
import pytest

from chaoyang.inpainting_candidate.v1_point_observer import PointOutputObserver


class Provider:
    def __init__(self):
        self.buffer = np.zeros((1, 3, 4), bool)
        self.output = {'out_binary_masks': self.buffer, 'out_obj_ids': np.array([5])}
        self.returned = (1, self.output)
        self.calls = []
        self.state = {'unchanged': True}

    def init_state(self, **kwargs):
        return self.state

    def add_prompt(self, *args, **kwargs):
        self.calls.append(('add', args, kwargs))
        self.buffer[0, 1, 2] = True
        return self.returned

    def propagate_in_video(self, *args, **kwargs):
        self.calls.append(('propagate', args, kwargs))
        yield self.returned
        self.buffer[:] = False
        self.buffer[0, 0, 0] = True
        yield (2, self.output)


def fixture_observer():
    provider = Provider()
    return provider, PointOutputObserver(SimpleNamespace(model=provider), (3, 3, 4))


def test_prompt_exact_return_identity_input_and_state():
    provider, observer = fixture_observer()
    kwargs = {'obj_id': 5, 'frame_idx': 1, 'points': np.array([[.5, .5]])}
    random_before = np.random.get_state()
    returned = observer.sam.model.add_prompt(**kwargs)
    assert returned is provider.returned
    assert returned[1] is provider.output
    assert provider.calls[0][2]['points'] is kwargs['points']
    assert observer.sam.model.init_state() is provider.state
    random_after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(random_before, random_after))
    assert observer.receipt([])['return_objects_replaced'] is False


def test_reused_vendor_buffer_is_copied_not_aliased():
    provider, observer = fixture_observer()
    observer.sam.model.add_prompt()
    events = list(observer.sam.model.propagate_in_video(reverse=False))
    assert events[0] is provider.returned
    first = np.unpackbits(observer.arrays['event_0000'][0], count=12).reshape(3, 4)
    assert first[1, 2] and not first[0, 0]
    assert [x['phase'] for x in observer.events] == ['add_prompt', 'propagate_forward', 'propagate_forward']


def test_adapter_presence_distinguishes_missing_from_empty():
    _, observer = fixture_observer()
    value = np.packbits(np.zeros(12, bool))
    arrays, objects = observer.packed_adapter_tracks({5: {1: value}})
    assert arrays['obj_5_present'].tolist() == [False, True, False]
    assert objects[0]['native_frame_indices'] == [1]
    value[:] = 255
    assert not arrays['obj_5'].any()


def test_resource_limit_stops_without_mutating_returned_masks():
    provider = Provider()
    observer = PointOutputObserver(SimpleNamespace(model=provider), (3, 3, 4), max_bytes=1)
    with pytest.raises(ValueError, match='RESOURCE_LIMIT'):
        observer.sam.model.add_prompt()
    assert provider.buffer.sum() == 1


@pytest.mark.parametrize('shape', [(0, 3, 4), (3, 4), (True, 3, 4)])
def test_invalid_domain(shape):
    with pytest.raises(ValueError, match='NATIVE_DOMAIN'):
        PointOutputObserver(SimpleNamespace(model=Provider()), shape)


def test_delegate_error_is_not_silenced():
    class Broken(Provider):
        def propagate_in_video(self, **kwargs):
            raise RuntimeError('ORIGINAL_MODEL_ERROR')
            yield
    observer = PointOutputObserver(SimpleNamespace(model=Broken()), (3, 3, 4))
    with pytest.raises(RuntimeError, match='ORIGINAL_MODEL_ERROR'):
        list(observer.sam.model.propagate_in_video())


def test_reverse_phase_and_empty_output():
    _, observer = fixture_observer()
    observer.capture('propagate_reverse', (0, None))
    assert observer.events[0]['phase'] == 'propagate_reverse'
    assert observer.events[0]['obj_ids'] == []
    assert observer.arrays['event_0000'].shape == (0, 2)
