from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.causal_modal_mask_challenger_v71 import BackendFrameOutput, MaskChallengerError
from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import (
    Sam21StreamingBackend,
    execute_single_session_attempt,
    real_attempt,
    validate_backend_prompt_compatibility,
)
from chaoyang.governance.v71_contracts import validate_artifact_revision


def _evidence(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path, *, future_suffix: str = "a", staggered: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_id in range(3):
        path = tmp_path / f"rgb_{frame_id}.bin"
        path.write_bytes(f"rgb-{frame_id}".encode())
        frames.append({"frame_id": frame_id, **_evidence(path)})
    rgb = {
        "schema_version": "causal-rgb-manifest-v71",
        "artifact_id": "rgb.fixture",
        "artifact_revision": "R7_0",
        "session_id": "chips_fixture",
        "frozen": True,
        "execution_mode": "CAUSAL_PROCESSING",
        "frames": frames,
    }
    audit = tmp_path / "audit.json"
    _write_json(
        audit,
        {
            "immutable": True,
            "authority": False,
            "gold_accuracy_authorized": False,
            "rows": [{"session_id": "chips_fixture"}],
        },
    )
    instances = []
    for index in range(3):
        initial_frame = index if staggered else 0
        initial = tmp_path / f"prompt_{index}_{initial_frame}.bin"
        initial.write_bytes(f"initial-{index}".encode())
        future = tmp_path / f"prompt_{index}_2_{future_suffix}.bin"
        future.write_bytes(f"future-{index}-{future_suffix}".encode())
        events = [
            {
                "frame_id": initial_frame,
                "event_type": "INITIAL_MASK",
                "mask": _evidence(initial),
            }
        ]
        if initial_frame < 2:
            events.append(
                {
                    "frame_id": 2,
                    "event_type": "CORRECTION_MASK",
                    "mask": _evidence(future),
                }
            )
        instances.append({"instance_id": f"chips_{index + 1}", "prompt_events": events})
    prompts = {
        "schema_version": "causal-modal-mask-prompts-v71",
        "artifact_id": "prompts.fixture",
        "artifact_revision": "R7_1",
        "session_id": "chips_fixture",
        "task": "CHIPS",
        "frozen": True,
        "allow_instance_union": False,
        "audit_selection": _evidence(audit),
        "instances": instances,
    }
    rgb_path = tmp_path / "rgb.json"
    prompt_path = tmp_path / "prompts.json"
    _write_json(rgb_path, rgb)
    _write_json(prompt_path, prompts)
    return rgb_path, prompt_path


class FakeLease:
    def __init__(self, marker: dict):
        self.marker = marker
        self.value = None
        self.released = None

    def __enter__(self):
        assert not self.marker["active"]
        self.marker["active"] = True
        self.value = {
            "schema_version": "chaoyang-gpu-lease-v71",
            "status": "ACQUIRED",
            "task_id": "s1",
            "attempt_id": "attempt",
            "fencing_token": "0123456789abcdef",
        }
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.marker["active"] = False
        self.released = {**self.value, "status": "RELEASED"}


class StatefulMockBackend:
    """Output depends on current/past prompts, never a future prompt."""

    def __init__(self, marker: dict):
        assert marker["active"], "backend construction must occur under lease"
        self.state = {f"chips_{index}": 0 for index in range(1, 4)}

    def step(self, *, frame_id, rgb_path, prompt_masks):
        for instance_id, path in prompt_masks.items():
            self.state[instance_id] = int(hashlib.sha256(path.read_bytes()).hexdigest()[:6], 16)
        masks = {}
        for index, instance_id in enumerate(self.state):
            mask = np.zeros((4, 4), dtype=bool)
            mask[index, (self.state[instance_id] + frame_id) % 4] = True
            masks[instance_id] = mask
        return BackendFrameOutput(
            masks=masks,
            visibility={instance_id: "VISIBLE" for instance_id in self.state},
        )


def _execute(tmp_path: Path, input_root: Path, *, suffix: str):
    rgb_path, prompt_path = _inputs(input_root, future_suffix=suffix)
    marker = {"active": False}
    receipt = execute_single_session_attempt(
        rgb_manifest_path=rgb_path,
        prompt_manifest_path=prompt_path,
        output_root=tmp_path / f"out_{suffix}",
        backend_name="cutie",
        task_id="s1_modal_mask",
        attempt_id="attempt_0001",
        artifact_revision="R7_1",
        executor_epoch=1,
        fencing_token="0123456789abcdef",
        gpu_id=0,
        lease_factory=lambda: FakeLease(marker),
        backend_factory=lambda: StatefulMockBackend(marker),
        repository_root=ROOT,
    )
    assert marker["active"] is False
    return receipt


def _frame_shas(receipt: dict) -> list[list[str | None]]:
    output = json.loads(Path(receipt["payload"]["output"]["path"]).read_text())
    return [
        [None if item["mask"] is None else item["mask"]["sha256"] for item in frame["instances"]]
        for frame in output["frames"]
    ]


def test_mock_attempt_is_immutable_leased_and_candidate_only(tmp_path):
    receipt = _execute(tmp_path, tmp_path / "inputs", suffix="a")
    assert validate_artifact_revision(receipt) == []
    payload = receipt["payload"]
    assert payload["status"] == "PASSED"
    assert payload["gpu_lease_scope"] == "MODEL_CONSTRUCTION_AND_CAUSAL_INFERENCE_ONLY"
    assert payload["gpu_lease_evidence"]["released"]["status"] == "RELEASED"
    assert payload["authority_limit"]["maximum_publication"] == "MODAL_MASK_SUCCESSOR_CANDIDATE"
    result_path = Path(payload["output"]["path"]).parent / "RESULT.json"
    assert result_path.is_file()
    with pytest.raises(FileExistsError):
        _execute(tmp_path, tmp_path / "inputs", suffix="a")


def test_future_prompt_change_does_not_change_past_mask_sha(tmp_path):
    first = _execute(tmp_path, tmp_path / "inputs_a", suffix="a")
    second = _execute(tmp_path, tmp_path / "inputs_b", suffix="b")
    first_shas = _frame_shas(first)
    second_shas = _frame_shas(second)
    assert first_shas[:2] == second_shas[:2]
    assert first_shas[2] != second_shas[2]


def test_sam_streaming_declares_staggered_initial_prompt_api_gap(tmp_path):
    _, prompt_path = _inputs(tmp_path, staggered=True)
    prompts = json.loads(prompt_path.read_text())
    errors = validate_backend_prompt_compatibility("sam2_1", prompts)
    assert len(errors) == 1
    assert "cannot safely register a new object" in errors[0]
    assert validate_backend_prompt_compatibility("cutie", prompts) == []


def test_sam_streaming_sets_video_dimensions_before_first_prompt(tmp_path, monkeypatch):
    events = []

    class Predictor:
        def init_state(self, video_path=None):
            assert video_path is None
            return {"video_height": None, "video_width": None}

        def add_new_frame(self, state, image):
            return 0

        def add_new_mask(self, state, frame_index, instance_id, mask):
            events.append((state["video_height"], state["video_width"]))

        def infer_single_frame(self, state, frame_index):
            class Logit:
                def detach(self): return self
                def float(self): return self
                def cpu(self): return np.ones((1, 3, 5), dtype=np.float32)
            return frame_index, ["card"], [Logit()]

    monkeypatch.setattr(
        "chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71._read_rgb",
        lambda path: np.zeros((3, 5, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71._read_mask",
        lambda path, shape: np.ones(shape, dtype=bool),
    )
    backend = Sam21StreamingBackend(
        predictor=Predictor(), instance_ids=["card"], initial_frame_id=0
    )
    backend.step(frame_id=0, rgb_path=tmp_path / "rgb", prompt_masks={"card": tmp_path / "mask"})
    assert events == [(3, 5)]


def test_real_attempt_requires_large_weight_sha_verification() -> None:
    import inspect

    source = inspect.getsource(real_attempt)
    assert "verify_large_sha=True" in source


class UnavailableLease:
    def __enter__(self):
        raise MaskChallengerError("GPU lease unavailable; held by exact78 clean")

    def __exit__(self, *args):
        return None


def test_busy_gpu_is_terminal_blocked_resource_not_runtime_failure(tmp_path):
    rgb_path, prompt_path = _inputs(tmp_path / "inputs")
    receipt = execute_single_session_attempt(
        rgb_manifest_path=rgb_path,
        prompt_manifest_path=prompt_path,
        output_root=tmp_path / "out",
        backend_name="cutie",
        task_id="s1_modal_mask",
        attempt_id="attempt_0001",
        artifact_revision="R7_1",
        executor_epoch=1,
        fencing_token="0123456789abcdef",
        gpu_id=0,
        lease_factory=UnavailableLease,
        backend_factory=lambda: pytest.fail("backend must not be built without lease"),
        repository_root=ROOT,
    )
    assert receipt["payload"]["status"] == "BLOCKED_RESOURCE"
    assert receipt["payload"]["execution_performed"] is False
    assert receipt["payload"]["output"] is None
