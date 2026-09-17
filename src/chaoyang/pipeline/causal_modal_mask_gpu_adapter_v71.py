from __future__ import annotations

"""Leased, strictly causal GPU adapters for the S1 modal-mask challenger.

The module is safe to import on a CPU worker: model frameworks and pinned
third-party implementations are imported lazily.  The production entrypoint
performs manifest/asset/import checks before acquiring a GPU lease, holds the
lease only while constructing and running the model, and publishes an
immutable attempt receipt afterwards.  No function in this module updates the
current governance ledger.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable, ContextManager, Mapping, Sequence

import numpy as np

from chaoyang.pipeline.causal_modal_mask_challenger_v71 import (
    BACKENDS,
    FORBIDDEN_CLAIMS,
    BackendFrameOutput,
    MaskChallengerError,
    canonical_sha256,
    file_sha256,
    preflight_backend,
    run_causal_sequence,
    validate_backend_environment,
    validate_prompt_manifest,
    validate_rgb_manifest,
)
from chaoyang.governance.common import atomic_json, load_json
from chaoyang.governance.gpu_lease_v71 import build_lease, heartbeat, query_gpu_memory_mib, query_gpu_pids, release
from chaoyang.governance.v71_contracts import build_artifact_revision, publish_new_revision


ROOT = Path(__file__).resolve().parents[3]
MAXIMUM_AUTHORITY = "MODAL_MASK_SUCCESSOR_CANDIDATE"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaskChallengerError(f"JSON root must be an object: {path}")
    return value


def _artifact_pin(value: Mapping[str, Any], label: str) -> tuple[str, str]:
    artifact_id = value.get("artifact_id")
    revision = value.get("artifact_revision")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise MaskChallengerError(f"{label} artifact_id is missing")
    if not isinstance(revision, str) or not revision.startswith("R7_"):
        raise MaskChallengerError(f"{label} artifact_revision is not R7_*")
    return artifact_id, revision


def _read_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:  # pragma: no cover - real input boundary
        raise MaskChallengerError(f"cannot decode RGB frame {path}: {exc}") from exc


def _read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        if path.suffix.lower() == ".npy":
            value = np.load(path, allow_pickle=False)
        else:
            from PIL import Image

            with Image.open(path) as image:
                value = np.asarray(image.convert("L"))
    except Exception as exc:  # pragma: no cover - real input boundary
        raise MaskChallengerError(f"cannot decode prompt mask {path}: {exc}") from exc
    value = np.asarray(value)
    if value.ndim != 2 or value.shape != expected_shape:
        raise MaskChallengerError(
            f"prompt mask shape mismatch at {path}: {value.shape} != {expected_shape}"
        )
    binary = value > 0
    if not binary.any():
        raise MaskChallengerError(f"empty prompt mask: {path}")
    return binary


def _exclusive_masks(
    scores: Mapping[str, np.ndarray], instance_ids: Sequence[str], *, threshold: float = 0.0
) -> dict[str, np.ndarray]:
    """Assign each positive pixel to exactly one instance; never union identities."""

    if not scores:
        return {}
    arrays = [np.asarray(scores[item], dtype=np.float32) for item in instance_ids]
    shape = arrays[0].shape
    if any(item.ndim != 2 or item.shape != shape for item in arrays):
        raise MaskChallengerError("backend score shape drift")
    stack = np.stack(arrays, axis=0)
    winner = np.argmax(stack, axis=0)
    positive = np.max(stack, axis=0) > threshold
    return {
        instance_id: np.logical_and(positive, winner == index)
        for index, instance_id in enumerate(instance_ids)
    }


def validate_backend_prompt_compatibility(
    backend: str, prompt_manifest: Mapping[str, Any]
) -> list[str]:
    """Declare local API limitations before touching a GPU."""

    if backend not in BACKENDS:
        return [f"unknown backend: {backend}"]
    errors: list[str] = []
    if backend == "sam2_1":
        starts = {
            int(instance["prompt_events"][0]["frame_id"])
            for instance in prompt_manifest.get("instances", [])
            if isinstance(instance, Mapping) and instance.get("prompt_events")
        }
        if len(starts) != 1:
            errors.append(
                "pinned SAM2.1 streaming API cannot safely register a new object after "
                "tracking starts; all per-instance INITIAL_MASK prompts must share one frame"
            )
    return errors


def runtime_import_preflight(
    backend: str,
    *,
    repository_root: Path = ROOT,
    python_executable: str | None = None,
    verify_large_sha: bool = False,
) -> dict[str, Any]:
    """Import and inspect pinned APIs without model construction or GPU execution."""

    asset = preflight_backend(
        backend, repository_root=repository_root, verify_large_sha=verify_large_sha
    )
    if not asset.get("execution_ready"):
        return {
            "backend": backend,
            "status": "BLOCKED_REFERENCE_PROOF",
            "execution_ready": False,
            "model_instantiated": False,
            "gpu_touched": False,
            "reason": asset.get("reason", "asset closure is not executable"),
            "asset_preflight": asset,
        }
    pin = _load_json(Path(str(asset["asset_pin"])))
    local = pin["local"]
    if backend == "sam2_1":
        implementation = (repository_root / local["implementation_ref"]).resolve()
        checks = (
            "from sam2.build_sam import build_sam2_video_predictor;"
            "from sam2.sam2_video_predictor import SAM2VideoPredictor;"
            "required=('init_state','add_new_frame','add_new_mask','infer_single_frame');"
            "missing=[x for x in required if not callable(getattr(SAM2VideoPredictor,x,None))];"
            "assert not missing, missing"
        )
        extra_path = str(implementation)
    else:
        implementation = (repository_root / local["implementation_root"]).resolve()
        checks = (
            "from tracker.config import CONFIG;"
            "from tracker.model.cutie import CUTIE;"
            "from tracker.inference.inference_core import InferenceCore;"
            "assert isinstance(CONFIG,dict);"
            "assert callable(getattr(InferenceCore,'step',None));"
            "assert callable(getattr(CUTIE,'load_weights',None))"
        )
        extra_path = str(implementation.parent)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = extra_path + os.pathsep + environment.get("PYTHONPATH", "")
    environment.update({str(k): str(v) for k, v in asset.get("runtime_environment", {}).items()})
    executable = python_executable or sys.executable
    completed = subprocess.run(
        [executable, "-c", checks],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode:
        return {
            "backend": backend,
            "status": "BLOCKED_REFERENCE_PROOF",
            "execution_ready": False,
            "model_instantiated": False,
            "gpu_touched": False,
            "reason": "pinned import/API preflight failed",
            "stderr_tail": completed.stderr[-4000:],
            "asset_preflight": asset,
            "python_executable": executable,
        }
    return {
        "backend": backend,
        "status": "READY_DEVELOPMENT_CANARY",
        "execution_ready": True,
        "model_instantiated": False,
        "gpu_touched": False,
        "reason": "pinned import and required causal APIs are available",
        "asset_preflight": asset,
        "python_executable": executable,
    }


class Sam21StreamingBackend:
    """SAM2.1 video-predictor adapter using only frames/prompts seen so far."""

    def __init__(
        self,
        *,
        predictor: Any,
        instance_ids: Sequence[str],
        initial_frame_id: int,
        min_area_pixels: int = 4,
    ) -> None:
        self.predictor = predictor
        self.instance_ids = tuple(instance_ids)
        self.initial_frame_id = initial_frame_id
        self.min_area_pixels = min_area_pixels
        self.state = predictor.init_state(video_path=None)
        self.previous_frame_id: int | None = None
        self.internal_frame_index = -1

    @classmethod
    def from_pinned_assets(
        cls,
        *,
        repository_root: Path,
        prompt_manifest: Mapping[str, Any],
        device: str,
    ) -> "Sam21StreamingBackend":  # pragma: no cover - requires leased GPU
        pin = _load_json(repository_root / "assets/models/sam2_1_hiera_large/ASSET_PIN.json")
        local = pin["local"]
        implementation = (repository_root / local["implementation_ref"]).resolve()
        if str(implementation) not in sys.path:
            sys.path.insert(0, str(implementation))
        from sam2.build_sam import build_sam2_video_predictor

        config = local["config_path"]
        prefix = "sam2/"
        if config.startswith(prefix):
            config = config[len(prefix):]
        predictor = build_sam2_video_predictor(
            config,
            str((repository_root / local["weight_path"]).resolve()),
            device=device,
            mode="eval",
        )
        starts = {
            int(item["prompt_events"][0]["frame_id"])
            for item in prompt_manifest["instances"]
        }
        if len(starts) != 1:
            raise MaskChallengerError("SAM2.1 initial prompt frame mismatch")
        return cls(
            predictor=predictor,
            instance_ids=[item["instance_id"] for item in prompt_manifest["instances"]],
            initial_frame_id=starts.pop(),
        )

    def step(
        self, *, frame_id: int, rgb_path: Path, prompt_masks: Mapping[str, Path]
    ) -> BackendFrameOutput:  # pragma: no cover - requires leased GPU
        if self.previous_frame_id is not None and frame_id <= self.previous_frame_id:
            raise MaskChallengerError("SAM2.1 frames must be strictly chronological")
        self.previous_frame_id = frame_id
        image = _read_rgb(rgb_path)
        self.internal_frame_index = int(self.predictor.add_new_frame(self.state, image))
        # The pinned SAM2.1 streaming extension leaves the original video
        # dimensions unset when init_state(video_path=None).  add_new_mask()
        # consolidates prompts at video resolution and therefore requires both
        # values before the first prompt; without this bridge torch.full() sees
        # None in its size tuple.
        if self.state.get("video_height") is None:
            self.state["video_height"] = int(image.shape[0])
        if self.state.get("video_width") is None:
            self.state["video_width"] = int(image.shape[1])
        if frame_id < self.initial_frame_id:
            return BackendFrameOutput(
                masks={},
                visibility={item: "TRACK_LOST_UNKNOWN" for item in self.instance_ids},
            )
        if frame_id == self.initial_frame_id and set(prompt_masks) != set(self.instance_ids):
            raise MaskChallengerError("SAM2.1 initial frame must prompt every independent instance")
        for instance_id, path in prompt_masks.items():
            self.predictor.add_new_mask(
                self.state,
                self.internal_frame_index,
                instance_id,
                _read_mask(path, image.shape[:2]),
            )
        _, object_ids, logits = self.predictor.infer_single_frame(
            self.state, self.internal_frame_index
        )
        score_map = {
            str(object_id): np.asarray(logits[index].detach().float().cpu()).squeeze()
            for index, object_id in enumerate(object_ids)
        }
        if set(score_map) != set(self.instance_ids):
            raise MaskChallengerError("SAM2.1 object identity drift")
        masks = _exclusive_masks(score_map, self.instance_ids)
        visible: dict[str, np.ndarray] = {}
        visibility: dict[str, str] = {}
        warnings: list[str] = []
        for instance_id, mask in masks.items():
            if int(mask.sum()) < self.min_area_pixels:
                visibility[instance_id] = "TRACK_LOST_UNKNOWN"
                warnings.append(f"{instance_id}:mask_below_min_area")
            else:
                visible[instance_id] = mask
                visibility[instance_id] = "VISIBLE"
        return BackendFrameOutput(visible, visibility, warnings)


class CutieMemoryBackend:
    """Cutie adapter whose memory receives frames and prompts in forward order only."""

    def __init__(
        self,
        *,
        torch_module: Any,
        core: Any,
        instance_ids: Sequence[str],
        device: str,
        min_area_pixels: int = 4,
    ) -> None:
        self.torch = torch_module
        self.core = core
        self.instance_ids = tuple(instance_ids)
        self.device = device
        self.numeric_ids = {item: index + 1 for index, item in enumerate(instance_ids)}
        self.previous_frame_id: int | None = None
        self.min_area_pixels = min_area_pixels

    @classmethod
    def from_pinned_assets(
        cls,
        *,
        repository_root: Path,
        prompt_manifest: Mapping[str, Any],
        device: str,
    ) -> "CutieMemoryBackend":  # pragma: no cover - requires leased GPU
        pin = _load_json(repository_root / "assets/models/cutie/ASSET_PIN.json")
        local = pin["local"]
        implementation = (repository_root / local["implementation_root"]).resolve()
        module_root = implementation.parent
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        os.environ["TORCH_HOME"] = str(
            (repository_root / "assets/models/cutie/torch_home").resolve()
        )
        import torch
        from omegaconf import OmegaConf
        from tracker.config import CONFIG
        from tracker.inference.inference_core import InferenceCore
        from tracker.model.cutie import CUTIE

        config = OmegaConf.create(CONFIG)
        network = CUTIE(config).to(device).eval()
        weights = torch.load(
            (repository_root / local["weight_path"]).resolve(),
            map_location=device,
            weights_only=True,
        )
        network.load_weights(weights)
        return cls(
            torch_module=torch,
            core=InferenceCore(network, config),
            instance_ids=[item["instance_id"] for item in prompt_manifest["instances"]],
            device=device,
        )

    def step(
        self, *, frame_id: int, rgb_path: Path, prompt_masks: Mapping[str, Path]
    ) -> BackendFrameOutput:  # pragma: no cover - requires leased GPU
        if self.previous_frame_id is not None and frame_id <= self.previous_frame_id:
            raise MaskChallengerError("Cutie frames must be strictly chronological")
        self.previous_frame_id = frame_id
        image = _read_rgb(rgb_path)
        indexed = np.zeros(image.shape[:2], dtype=np.int64)
        prompted_numeric: list[int] = []
        for instance_id, path in prompt_masks.items():
            mask = _read_mask(path, image.shape[:2])
            if np.any(np.logical_and(indexed > 0, mask)):
                raise MaskChallengerError("Cutie prompt masks overlap across physical instances")
            numeric = self.numeric_ids[instance_id]
            indexed[mask] = numeric
            prompted_numeric.append(numeric)
        frame_tensor = self.torch.from_numpy(image.transpose(2, 0, 1)).float()
        frame_tensor = frame_tensor.to(self.device, non_blocking=True) / 255.0
        mask_tensor = None
        if prompted_numeric:
            mask_tensor = self.torch.from_numpy(indexed).long().to(self.device)
        probabilities = self.core.step(
            frame_tensor,
            mask_tensor,
            objects=prompted_numeric if prompted_numeric else None,
            idx_mask=True,
            force_permanent=bool(prompted_numeric),
        )
        winner = self.torch.argmax(probabilities, dim=0).detach().cpu().numpy()
        active = list(self.core.object_manager.all_obj_ids)
        masks: dict[str, np.ndarray] = {}
        visibility: dict[str, str] = {}
        warnings: list[str] = []
        for instance_id in self.instance_ids:
            numeric = self.numeric_ids[instance_id]
            if numeric not in active:
                visibility[instance_id] = "TRACK_LOST_UNKNOWN"
                continue
            temporary = active.index(numeric) + 1
            mask = winner == temporary
            if int(mask.sum()) < self.min_area_pixels:
                visibility[instance_id] = "TRACK_LOST_UNKNOWN"
                warnings.append(f"{instance_id}:mask_below_min_area")
            else:
                masks[instance_id] = mask
                visibility[instance_id] = "VISIBLE"
        return BackendFrameOutput(masks, visibility, warnings)


def build_pinned_backend(
    backend: str,
    *,
    repository_root: Path,
    prompt_manifest: Mapping[str, Any],
    gpu_id: int,
) -> Any:  # pragma: no cover - requires leased GPU
    device = f"cuda:{gpu_id}"
    if backend == "sam2_1":
        return Sam21StreamingBackend.from_pinned_assets(
            repository_root=repository_root,
            prompt_manifest=prompt_manifest,
            device=device,
        )
    if backend == "cutie":
        return CutieMemoryBackend.from_pinned_assets(
            repository_root=repository_root,
            prompt_manifest=prompt_manifest,
            device=device,
        )
    raise MaskChallengerError(f"unknown backend: {backend}")


class FileGpuLease:
    """Central V7.1 lease guard.  It never waits or reclaims ambiguous leases."""

    def __init__(
        self,
        *,
        lease_path: Path,
        lock_path: Path,
        task_id: str,
        attempt_id: str,
        gpu_id: int,
        executor_epoch: int,
        fencing_token: str,
        min_free_mib: int = 61440,
    ) -> None:
        self.lease_path = lease_path
        self.lock_path = lock_path
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.gpu_id = gpu_id
        self.executor_epoch = executor_epoch
        self.fencing_token = fencing_token
        self.min_free_mib = min_free_mib
        self.pid = os.getpid()
        self.value: dict[str, Any] | None = None
        self.released: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _locked_update(self, callback: Callable[[dict[str, Any] | None], dict[str, Any]]) -> dict[str, Any]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            current = load_json(self.lease_path) if self.lease_path.is_file() else None
            updated = callback(current)
            atomic_json(self.lease_path, updated)
            return updated

    def __enter__(self) -> "FileGpuLease":
        def acquire(current: dict[str, Any] | None) -> dict[str, Any]:
            if current is not None and current.get("status") != "RELEASED":
                raise MaskChallengerError(
                    f"GPU lease unavailable; held by {current.get('task_id') or current.get('holder')}"
                )
            gpu_pids = query_gpu_pids()
            memory = query_gpu_memory_mib(self.gpu_id)
            if gpu_pids is None or memory is None:
                raise MaskChallengerError(
                    "GPU lease unavailable; physical process evidence is unavailable"
                )
            if memory["free_mib"] < self.min_free_mib:
                raise MaskChallengerError(
                    "GPU lease unavailable; insufficient free memory: "
                    f"{memory['free_mib']} MiB < {self.min_free_mib} MiB; "
                    f"physical GPU processes={sorted(gpu_pids)}"
                )
            value = build_lease(
                task_id=self.task_id,
                attempt_id=self.attempt_id,
                pid=self.pid,
                gpu_id=self.gpu_id,
                executor_epoch=self.executor_epoch,
                fencing_token=self.fencing_token,
                priority="CANARY",
            )
            value["gpu_process_pid"] = self.pid
            return value

        self.value = self._locked_update(acquire)

        def beat() -> None:
            while not self._stop.wait(30):
                def update(current: dict[str, Any] | None) -> dict[str, Any]:
                    if current is None:
                        raise MaskChallengerError("GPU lease disappeared")
                    return heartbeat(current, pid=self.pid, fencing_token=self.fencing_token)

                try:
                    self.value = self._locked_update(update)
                except Exception:
                    self._stop.set()

        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        def finish(current: dict[str, Any] | None) -> dict[str, Any]:
            if current is None:
                raise MaskChallengerError("GPU lease disappeared before release")
            return release(
                current,
                pid=self.pid,
                fencing_token=self.fencing_token,
                reason="INFERENCE_FAILED" if exc_type else "INFERENCE_COMPLETE",
            )

        self.released = self._locked_update(finish)


def _lease_evidence(guard: Any) -> dict[str, Any]:
    acquired = dict(getattr(guard, "value", {}) or {})
    released = dict(getattr(guard, "released", {}) or {})
    token = acquired.pop("fencing_token", None)
    released.pop("fencing_token", None)
    return {
        "acquired": acquired,
        "released": released,
        "fencing_token_sha256": hashlib.sha256(str(token).encode()).hexdigest() if token else None,
    }


def publish_nonexecution_attempt(
    *,
    rgb_manifest_path: Path,
    prompt_manifest_path: Path,
    output_root: Path,
    backend_name: str,
    task_id: str,
    attempt_id: str,
    artifact_revision: str,
    status: str,
    reason: str,
    backend_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal an input/reference blocker without touching the GPU or current ledger."""

    if status not in {"BLOCKED_REFERENCE_PROOF", "BLOCKED_PREREQ"}:
        raise ValueError(f"invalid non-execution status: {status}")
    rgb = _load_json(rgb_manifest_path)
    prompts = _load_json(prompt_manifest_path)
    rgb_pin = _artifact_pin(rgb, "RGB manifest")
    prompt_pin = _artifact_pin(prompts, "prompt manifest")
    session_id = str(rgb.get("session_id", "UNKNOWN"))
    attempt_directory = (
        output_root.resolve() / "sessions" / session_id / "attempts" / attempt_id
    )
    if attempt_directory.exists():
        raise FileExistsError(f"immutable attempt already exists: {attempt_directory}")
    attempt_directory.mkdir(parents=True)
    input_sha = canonical_sha256(
        {
            "rgb_manifest_sha": file_sha256(rgb_manifest_path),
            "prompt_manifest_sha": file_sha256(prompt_manifest_path),
        }
    )
    producer_signature = canonical_sha256(
        {
            "adapter_sha": file_sha256(Path(__file__)),
            "backend": backend_name,
            "backend_preflight": backend_preflight,
            "input_manifest_sha": input_sha,
        }
    )
    payload = {
        "schema_version": "causal-modal-mask-gpu-attempt-v71",
        "status": status,
        "session_id": session_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "backend": backend_name,
        "execution_performed": False,
        "gpu_lease_scope": "MODEL_CONSTRUCTION_AND_CAUSAL_INFERENCE_ONLY",
        "gpu_lease_evidence": None,
        "output": None,
        "runtime_error": reason,
        "identity_policy": "PER_INSTANCE_NO_UNION_FAIL_CLOSED_UNKNOWN",
        "prompt_policy": "CURRENT_OR_PAST_ONLY",
        "authority_limit": {
            "maximum_publication": MAXIMUM_AUTHORITY,
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
        },
    }
    receipt = build_artifact_revision(
        artifact_id=f"S1_MODAL_MASK.{backend_name}.{session_id}.{attempt_id}",
        artifact_revision=artifact_revision,
        input_artifacts=[rgb_pin, prompt_pin],
        input_manifest_sha=input_sha,
        producer_signature=producer_signature,
        payload=payload,
    )
    publish_new_revision(attempt_directory / "RESULT.json", receipt)
    return receipt


def execute_single_session_attempt(
    *,
    rgb_manifest_path: Path,
    prompt_manifest_path: Path,
    output_root: Path,
    backend_name: str,
    task_id: str,
    attempt_id: str,
    artifact_revision: str,
    executor_epoch: int,
    fencing_token: str,
    gpu_id: int,
    lease_factory: Callable[[], ContextManager[Any]],
    backend_factory: Callable[[], Any],
    repository_root: Path = ROOT,
    backend_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one immutable session attempt with an injectable test backend."""

    rgb = _load_json(rgb_manifest_path)
    prompts = _load_json(prompt_manifest_path)
    errors = validate_rgb_manifest(rgb) + validate_prompt_manifest(prompts, rgb)
    errors += validate_backend_prompt_compatibility(backend_name, prompts)
    if errors:
        raise MaskChallengerError("; ".join(errors))
    rgb_pin = _artifact_pin(rgb, "RGB manifest")
    prompt_pin = _artifact_pin(prompts, "prompt manifest")
    session_id = str(rgb["session_id"])
    attempt_directory = (
        output_root.resolve() / "sessions" / session_id / "attempts" / attempt_id
    )
    if attempt_directory.exists():
        raise FileExistsError(f"immutable attempt already exists: {attempt_directory}")
    attempt_directory.mkdir(parents=True)
    status = "FAILED_RUNTIME_FINAL"
    output_ref: dict[str, Any] | None = None
    runtime_error: str | None = None
    guard: Any = None
    execution_started = False
    try:
        with lease_factory() as held:
            guard = held
            backend = backend_factory()
            execution_started = True
            output = run_causal_sequence(
                rgb,
                prompts,
                backend,
                output_directory=attempt_directory / "masks",
            )
        output_path = attempt_directory / "MASK_OUTPUT.json"
        atomic_json(output_path, output)
        output_ref = {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
        }
        status = "PASSED"
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        if "GPU lease unavailable" in runtime_error:
            status = "BLOCKED_RESOURCE"
    combined_input_sha = canonical_sha256(
        {
            "rgb_manifest_sha": file_sha256(rgb_manifest_path),
            "prompt_manifest_sha": file_sha256(prompt_manifest_path),
        }
    )
    producer_signature = canonical_sha256(
        {
            "adapter_sha": file_sha256(Path(__file__)),
            "challenger_sha": file_sha256(repository_root / "src/chaoyang/pipeline/causal_modal_mask_challenger_v71.py"),
            "backend": backend_name,
            "backend_preflight": backend_preflight or {"status": "INJECTED_TEST_BACKEND"},
            "input_manifest_sha": combined_input_sha,
        }
    )
    payload = {
        "schema_version": "causal-modal-mask-gpu-attempt-v71",
        "status": status,
        "session_id": session_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "backend": backend_name,
        "execution_performed": execution_started,
        "gpu_lease_scope": "MODEL_CONSTRUCTION_AND_CAUSAL_INFERENCE_ONLY",
        "gpu_lease_evidence": _lease_evidence(guard) if guard is not None else None,
        "output": output_ref,
        "runtime_error": runtime_error,
        "identity_policy": "PER_INSTANCE_NO_UNION_FAIL_CLOSED_UNKNOWN",
        "prompt_policy": "CURRENT_OR_PAST_ONLY",
        "authority_limit": {
            "maximum_publication": MAXIMUM_AUTHORITY,
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
        },
    }
    receipt = build_artifact_revision(
        artifact_id=f"S1_MODAL_MASK.{backend_name}.{session_id}.{attempt_id}",
        artifact_revision=artifact_revision,
        input_artifacts=[rgb_pin, prompt_pin],
        input_manifest_sha=combined_input_sha,
        producer_signature=producer_signature,
        payload=payload,
    )
    publish_new_revision(attempt_directory / "RESULT.json", receipt)
    return receipt


def real_attempt(
    *,
    rgb_manifest_path: Path,
    prompt_manifest_path: Path,
    output_root: Path,
    backend_name: str,
    task_id: str,
    attempt_id: str,
    artifact_revision: str,
    executor_epoch: int,
    fencing_token: str,
    gpu_id: int,
    lease_path: Path,
    lease_lock_path: Path,
    repository_root: Path = ROOT,
) -> dict[str, Any]:  # pragma: no cover - intentionally not run in CPU tests
    rgb = _load_json(rgb_manifest_path)
    prompts = _load_json(prompt_manifest_path)
    # A real GPU attempt must bind the actual model bytes, not only their file
    # size.  The lightweight CLI import probe may skip this hash, but execution
    # cannot.
    import_check = runtime_import_preflight(
        backend_name, repository_root=repository_root, verify_large_sha=True
    )
    compatibility_errors = validate_backend_prompt_compatibility(backend_name, prompts)
    if not import_check.get("execution_ready") or compatibility_errors:
        reason = compatibility_errors or [str(import_check.get("reason"))]
        return publish_nonexecution_attempt(
            rgb_manifest_path=rgb_manifest_path,
            prompt_manifest_path=prompt_manifest_path,
            output_root=output_root,
            backend_name=backend_name,
            task_id=task_id,
            attempt_id=attempt_id,
            artifact_revision=artifact_revision,
            status="BLOCKED_REFERENCE_PROOF",
            reason="; ".join(reason),
            backend_preflight=import_check,
        )
    required_environment = import_check["asset_preflight"].get("runtime_environment", {})
    environment_errors = validate_backend_environment(
        import_check["asset_preflight"], {**os.environ, **required_environment}
    )
    if environment_errors:
        return publish_nonexecution_attempt(
            rgb_manifest_path=rgb_manifest_path,
            prompt_manifest_path=prompt_manifest_path,
            output_root=output_root,
            backend_name=backend_name,
            task_id=task_id,
            attempt_id=attempt_id,
            artifact_revision=artifact_revision,
            status="BLOCKED_REFERENCE_PROOF",
            reason="; ".join(environment_errors),
            backend_preflight=import_check,
        )
    os.environ.update({str(k): str(v) for k, v in required_environment.items()})
    return execute_single_session_attempt(
        rgb_manifest_path=rgb_manifest_path,
        prompt_manifest_path=prompt_manifest_path,
        output_root=output_root,
        backend_name=backend_name,
        task_id=task_id,
        attempt_id=attempt_id,
        artifact_revision=artifact_revision,
        executor_epoch=executor_epoch,
        fencing_token=fencing_token,
        gpu_id=gpu_id,
        lease_factory=lambda: FileGpuLease(
            lease_path=lease_path,
            lock_path=lease_lock_path,
            task_id=task_id,
            attempt_id=attempt_id,
            gpu_id=gpu_id,
            executor_epoch=executor_epoch,
            fencing_token=fencing_token,
        ),
        backend_factory=lambda: build_pinned_backend(
            backend_name,
            repository_root=repository_root,
            prompt_manifest=prompts,
            gpu_id=gpu_id,
        ),
        repository_root=repository_root,
        backend_preflight=import_check,
    )
