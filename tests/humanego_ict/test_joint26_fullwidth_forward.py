"""Default-width synthetic CPU forward/backward; no data, weights or optimizer."""
import hashlib
from pathlib import Path

import pytest
import torch

from chaoyang.humanego_ict.joint26_model import Joint26ICTModel


FROZEN_SHARED_SHA = "5f108794b93dde56db90ba0c793f658b890d36e6e177a00f37143801c05997d7"
SHARED_SOURCE = (Path(__file__).resolve().parents[2]
                 / "src/chaoyang/human_ego/training/FlowMatchingModel.py")


@pytest.fixture(autouse=True)
def bounded_cpu_random_state():
    """Keep synthetic execution to two CPU threads and restore caller state."""
    threads = torch.get_num_threads()
    random_state = torch.random.get_rng_state()
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(7)
    try:
        yield
    finally:
        torch.random.set_rng_state(random_state)
        torch.set_num_threads(threads)


def _state_signature(model):
    return tuple((name, tuple(value.shape), value.dtype, value.device)
                 for name, value in model.state_dict().items())


@pytest.mark.parametrize("batch", [8, 3, 1])
def test_default_384_by_6_accepts_public_batch_time_and_backward(batch):
    shared_before = SHARED_SOURCE.read_bytes()
    assert hashlib.sha256(shared_before).hexdigest() == FROZEN_SHARED_SHA
    with torch.device("cpu"):
        model = Joint26ICTModel().to(dtype=torch.float32).train()
    assert model.action_proj.out_features == 384
    assert len(model.decoder) == 6
    assert model.patch_size == 16 and model.num_heads == 8
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    signature_before = _state_signature(model)
    generator = torch.Generator(device="cpu").manual_seed(100 + batch)

    def random_tensor(*shape):
        return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)

    public_time = torch.linspace(.15, .85, batch, device="cpu", dtype=torch.float32)
    public_time.requires_grad_()
    time_before = public_time.detach().clone()
    inputs = {
        "x_rgb": random_tensor(batch, 3, 240, 320),
        "x_ict": random_tensor(batch, 8, 29),
        "ict_mask": torch.ones((batch, 8), device="cpu", dtype=torch.bool),
        "x_state26": random_tensor(batch, 26),
        "state_mask": torch.ones((batch, 26), device="cpu", dtype=torch.bool),
        "x_t": random_tensor(batch, 30, 26),
        "t": public_time,
    }
    assert inputs["t"].shape == (batch,)
    output = model(**inputs)
    assert set(output) == {"v_pred"}
    prediction = output["v_pred"]
    assert prediction.shape == (batch, 30, 26)
    assert prediction.dtype == torch.float32 and prediction.device.type == "cpu"
    assert torch.isfinite(prediction).all()
    loss = prediction.square().mean()
    assert torch.isfinite(loss)
    loss.backward()

    parameters = dict(model.named_parameters())
    for name in ("time_mlp.1.weight", "time_mlp.3.weight", "action_proj.weight",
                 "head_v.weight", "robot_state_proj.weight", "state_proj.weight",
                 "rgb_embed.proj.weight"):
        assert parameters[name].grad is not None, name
        assert torch.isfinite(parameters[name].grad).all(), name
    gradients = [parameter.grad for parameter in parameters.values() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert public_time.grad is not None and public_time.grad.shape == (batch,)
    assert torch.isfinite(public_time.grad).all()
    assert public_time.shape == (batch,) and torch.equal(public_time.detach(), time_before)
    assert _state_signature(model) == signature_before
    assert SHARED_SOURCE.read_bytes() == shared_before
