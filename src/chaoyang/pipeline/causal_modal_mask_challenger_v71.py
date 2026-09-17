from __future__ import annotations

"""Fail-closed causal modal-mask challenger contract for S1.

This module deliberately does not import or instantiate SAM 2.1 or Cutie.  It
validates frozen inputs, asset provenance, causal prompt ordering, immutable
attempt fencing, and backend-neutral outputs.  A future GPU worker may inject a
``ModalMaskBackend`` implementation after the corresponding asset preflight is
ready; this contract never upgrades a modal mask to amodal, Contact, or
Object6D authority.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


BACKENDS = ("sam2_1", "cutie")
UNKNOWN_VISIBILITY = frozenset(
    {"FULLY_OCCLUDED_UNKNOWN", "OUT_OF_FRAME_UNKNOWN", "TRACK_LOST_UNKNOWN"}
)
VISIBLE_VISIBILITY = frozenset({"VISIBLE", "PARTIAL"})
ALL_VISIBILITY = UNKNOWN_VISIBILITY | VISIBLE_VISIBILITY
FORBIDDEN_CLAIMS = (
    "AMODAL_MASK_TRUTH",
    "CONTACT_GROUND_TRUTH",
    "OBJECT6D_AUTHORITY",
    "EXTERNAL_PHYSICAL_TRUTH",
    "OCCLUSION_GOLD_ACCURACY_WITHOUT_INDEPENDENT_LABELS",
    "PHYSICAL_DEPLOYMENT_AUTHORITY",
)


class MaskChallengerError(RuntimeError):
    pass


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def directory_inventory_sha256(root: Path) -> str:
    """Bind a minimal source closure without relying on nested Git metadata."""
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(f"{relative}\0{path.stat().st_size}\0{file_sha256(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _validate_file_evidence(record: Mapping[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        errors.append(f"{label}.path must be absolute")
        return errors
    if not path.is_file():
        errors.append(f"{label}.path missing: {path}")
        return errors
    if int(record.get("bytes", -1)) != path.stat().st_size:
        errors.append(f"{label}.bytes mismatch: {path}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{label}.sha256 must be lowercase SHA-256")
    elif file_sha256(path) != expected:
        errors.append(f"{label}.sha256 mismatch: {path}")
    return errors


def validate_rgb_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "causal-rgb-manifest-v71":
        errors.append("rgb manifest schema_version mismatch")
    if manifest.get("frozen") is not True:
        errors.append("rgb manifest must be frozen")
    if manifest.get("execution_mode") != "CAUSAL_PROCESSING":
        errors.append("rgb manifest must use CAUSAL_PROCESSING")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("rgb manifest frames must be non-empty")
        return errors
    ids = [item.get("frame_id") for item in frames if isinstance(item, Mapping)]
    if len(ids) != len(frames) or any(not isinstance(value, int) or value < 0 for value in ids):
        errors.append("all frame_id values must be non-negative integers")
    elif ids != sorted(ids) or len(set(ids)) != len(ids):
        errors.append("frame_id values must be strictly increasing and unique")
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            errors.append(f"frames[{index}] must be an object")
            continue
        errors.extend(_validate_file_evidence(frame, label=f"frames[{index}]"))
    return errors


def validate_prompt_manifest(
    prompts: Mapping[str, Any], rgb_manifest: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if prompts.get("schema_version") != "causal-modal-mask-prompts-v71":
        errors.append("prompt manifest schema_version mismatch")
    if prompts.get("frozen") is not True:
        errors.append("prompt manifest must be frozen")
    if prompts.get("allow_instance_union") is not False:
        errors.append("allow_instance_union must be false")
    if prompts.get("session_id") != rgb_manifest.get("session_id"):
        errors.append("prompt/rgb session mismatch")
    audit_evidence = prompts.get("audit_selection")
    if not isinstance(audit_evidence, Mapping):
        errors.append("prompt manifest must pin frozen audit_selection evidence")
    else:
        errors.extend(_validate_file_evidence(audit_evidence, label="audit_selection"))
        audit_path = Path(str(audit_evidence.get("path", "")))
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append("audit_selection is not readable JSON")
            else:
                if audit.get("immutable") is not True or audit.get("authority") is not False:
                    errors.append("audit_selection must be immutable development evidence")
                if audit.get("gold_accuracy_authorized") is not False:
                    errors.append("audit_selection must not authorize Gold accuracy")
                sessions = {
                    row.get("session_id") for row in audit.get("rows", []) if isinstance(row, Mapping)
                }
                if prompts.get("session_id") not in sessions:
                    errors.append("session is absent from frozen audit_selection")

    instances = prompts.get("instances")
    if not isinstance(instances, list) or not instances:
        errors.append("instances must be non-empty")
        return errors
    instance_ids = [item.get("instance_id") for item in instances if isinstance(item, Mapping)]
    if len(instance_ids) != len(instances) or len(set(instance_ids)) != len(instance_ids):
        errors.append("instance_id values must be present and unique")
    task = prompts.get("task")
    if task == "CHIPS" and len(instances) != 3:
        errors.append("CHIPS requires exactly three independent physical instances")

    frame_ids = {
        item.get("frame_id")
        for item in rgb_manifest.get("frames", [])
        if isinstance(item, Mapping)
    }
    mask_paths: set[str] = set()
    for instance_index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            errors.append(f"instances[{instance_index}] must be an object")
            continue
        events = instance.get("prompt_events")
        if not isinstance(events, list) or not events:
            errors.append(f"instance {instance.get('instance_id')} has no prompt events")
            continue
        event_ids = [event.get("frame_id") for event in events if isinstance(event, Mapping)]
        if len(event_ids) != len(events) or event_ids != sorted(event_ids):
            errors.append(f"instance {instance.get('instance_id')} prompts are not chronological")
        if events[0].get("event_type") != "INITIAL_MASK":
            errors.append(f"instance {instance.get('instance_id')} must start with INITIAL_MASK")
        for event_index, event in enumerate(events):
            label = f"instances[{instance_index}].prompt_events[{event_index}].mask"
            if event.get("event_type") not in {"INITIAL_MASK", "CORRECTION_MASK"}:
                errors.append(f"{label} has unsupported event_type")
            if event.get("frame_id") not in frame_ids:
                errors.append(f"{label} frame is outside frozen RGB manifest")
            evidence = event.get("mask")
            if not isinstance(evidence, Mapping):
                errors.append(f"{label} must be evidence object")
                continue
            errors.extend(_validate_file_evidence(evidence, label=label))
            path = str(evidence.get("path", ""))
            if path in mask_paths:
                errors.append("prompt mask paths must not be shared across instances/events")
            mask_paths.add(path)
    return errors


def _resolve_pin_path(repository_root: Path, asset_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repository_candidate = repository_root / candidate
    if repository_candidate.exists():
        return repository_candidate
    return asset_root / candidate


def preflight_backend(
    backend: str,
    *,
    repository_root: Path,
    verify_large_sha: bool = False,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise MaskChallengerError(f"unknown backend: {backend}")
    repository_root = repository_root.resolve()
    if backend == "sam2_1":
        asset_root = repository_root / "assets/models/sam2_1_hiera_large"
        pin_path = asset_root / "ASSET_PIN.json"
    else:
        asset_root = repository_root / "assets/models/cutie"
        pin_path = asset_root / "ASSET_PIN.json"

    if not pin_path.is_file():
        partials = sorted(str(path.resolve()) for path in asset_root.glob("*.part")) if asset_root.is_dir() else []
        return {
            "backend": backend,
            "status": "BLOCKED_REFERENCE_PROOF",
            "execution_ready": False,
            "reason": "missing immutable ASSET_PIN.json; embedded code or partial weights are not an executable closure",
            "asset_pin": None,
            "partial_files": partials,
        }

    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "backend": backend,
            "status": "BLOCKED_REFERENCE_PROOF",
            "execution_ready": False,
            "reason": f"asset pin cannot be parsed: {exc}",
            "asset_pin": str(pin_path.resolve()),
            "partial_files": [],
        }

    local = pin.get("local", {})
    common_required = (
        "config_path", "config_sha256", "weight_path", "weight_bytes",
        "weight_sha256", "license_path", "license_sha256",
    )
    backend_required = (
        ("implementation_ref", "implementation_inventory_sha256")
        if backend == "sam2_1"
        else (
            "implementation_root", "model_path", "model_bytes", "model_sha256",
            "resnet50_path", "resnet50_bytes", "resnet50_sha256",
            "resnet18_path", "resnet18_bytes", "resnet18_sha256",
        )
    )
    required = common_required + backend_required
    missing = [key for key in required if key not in local]
    if missing:
        return {
            "backend": backend,
            "status": "BLOCKED_REFERENCE_PROOF",
            "execution_ready": False,
            "reason": f"asset pin missing closure fields: {missing}",
            "asset_pin": str(pin_path.resolve()),
            "partial_files": [],
        }

    weight = _resolve_pin_path(repository_root, asset_root, str(local["weight_path"]))
    license_path = _resolve_pin_path(repository_root, asset_root, str(local["license_path"]))
    errors = []
    runtime_environment: dict[str, str] = {}
    if backend == "sam2_1":
        implementation = _resolve_pin_path(
            repository_root, asset_root, str(local["implementation_ref"])
        )
        config_path = implementation / str(local["config_path"])
        if not implementation.is_dir():
            errors.append("implementation closure is missing")
        if not config_path.is_file() or file_sha256(config_path) != local["config_sha256"]:
            errors.append("config missing or SHA mismatch")
        inventory_sha = local.get("implementation_inventory_sha256")
        if not isinstance(inventory_sha, str) or len(inventory_sha) != 64:
            errors.append("implementation inventory SHA is invalid")
    else:
        implementation = _resolve_pin_path(
            repository_root, asset_root, str(local["implementation_root"])
        )
        model_path = _resolve_pin_path(repository_root, asset_root, str(local["model_path"]))
        config_path = _resolve_pin_path(repository_root, asset_root, str(local["config_path"]))
        if not implementation.is_dir():
            errors.append("Cutie implementation root is missing")
        for name, path in (("model", model_path), ("config", config_path)):
            if (
                not path.is_file()
                or path.stat().st_size != int(local[f"{name}_bytes"])
                or file_sha256(path) != local[f"{name}_sha256"]
            ):
                errors.append(f"Cutie {name} missing or closure mismatch")
        source = pin.get("source", {})
        if source.get("identity_mode") == "PINNED_FILE_CLOSURE":
            expected_inventory = source.get("implementation_inventory_sha256")
            if (
                not isinstance(expected_inventory, str)
                or len(expected_inventory) != 64
                or directory_inventory_sha256(implementation) != expected_inventory
            ):
                errors.append("Cutie source file closure mismatch")
        else:
            source_commit = source.get("commit")
            source_root = implementation
            while source_root != source_root.parent and not (source_root / ".git").exists():
                source_root = source_root.parent
            if not (source_root / ".git").exists():
                errors.append("Cutie source Git closure is unavailable")
            else:
                head = subprocess.run(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=False,
                )
                dirty = subprocess.run(
                    ["git", "-C", str(source_root), "status", "--porcelain"],
                    capture_output=True, text=True, check=False,
                )
                if head.returncode or head.stdout.strip() != source_commit:
                    errors.append("Cutie source commit mismatch")
                if dirty.returncode or dirty.stdout.strip():
                    errors.append("Cutie source checkout is dirty")
        for prefix in ("resnet50", "resnet18"):
            dependency = _resolve_pin_path(repository_root, asset_root, str(local[f"{prefix}_path"]))
            if not dependency.is_file() or dependency.stat().st_size != int(local[f"{prefix}_bytes"]):
                errors.append(f"{prefix} dependency missing or byte count mismatch")
            elif verify_large_sha and file_sha256(dependency) != local[f"{prefix}_sha256"]:
                errors.append(f"{prefix} dependency SHA mismatch")
        torch_home = asset_root / "torch_home"
        runtime_environment["TORCH_HOME"] = str(torch_home.resolve())
    if not weight.is_file() or weight.stat().st_size != int(local["weight_bytes"]):
        errors.append("weight missing or byte count mismatch")
    elif verify_large_sha and file_sha256(weight) != local["weight_sha256"]:
        errors.append("weight SHA mismatch")
    if not license_path.is_file() or file_sha256(license_path) != local["license_sha256"]:
        errors.append("license missing or SHA mismatch")
    if not str(pin.get("load_smoke", {}).get("status", "")).startswith("PASS"):
        errors.append("load-smoke receipt is not PASS")
    if errors:
        return {
            "backend": backend,
            "status": "BLOCKED_REFERENCE_PROOF",
            "execution_ready": False,
            "reason": "; ".join(errors),
            "asset_pin": str(pin_path.resolve()),
            "partial_files": [],
        }

    return {
        "backend": backend,
        "status": "READY_DEVELOPMENT_CANARY",
        "execution_ready": True,
        "reason": "asset closure is pinned; publication remains development modal-mask candidate only",
        "asset_pin": str(pin_path.resolve()),
        "asset_pin_sha256": file_sha256(pin_path),
        "weight_sha_verified_now": bool(verify_large_sha),
        "formal_production_allowed": bool(pin.get("execution", {}).get("formal_production_allowed", False)),
        "runtime_environment": runtime_environment,
    }


def validate_execution_lease(
    lease: Mapping[str, Any],
    *,
    task_id: str,
    attempt_id: str,
    fencing_token: str,
    executor_epoch: int | None = None,
    owner_pid: int | None = None,
    at: datetime | None = None,
) -> list[str]:
    errors: list[str] = []
    if lease.get("schema_version") != "chaoyang-gpu-lease-v71":
        errors.append("lease schema mismatch")
    if lease.get("status") != "ACQUIRED":
        errors.append("lease is not ACQUIRED")
    if lease.get("task_id") != task_id or lease.get("attempt_id") != attempt_id:
        errors.append("lease task/attempt mismatch")
    if lease.get("fencing_token") != fencing_token:
        errors.append("lease fencing token mismatch")
    if executor_epoch is not None and lease.get("executor_epoch") != executor_epoch:
        errors.append("lease executor epoch mismatch")
    if owner_pid is not None:
        if lease.get("pid") != owner_pid:
            errors.append("lease owner PID mismatch")
        else:
            stat_path = Path(f"/proc/{owner_pid}/stat")
            try:
                startticks = int(stat_path.read_text(encoding="utf-8").split()[21])
            except (OSError, IndexError, ValueError):
                errors.append("lease owner process identity is unavailable")
            else:
                if lease.get("process_startticks") != startticks:
                    errors.append("lease owner process startticks mismatch")
    expires_at = lease.get("expires_at")
    if isinstance(expires_at, str):
        try:
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            current = at or datetime.now().astimezone()
            if expires.tzinfo is None:
                expires = expires.astimezone()
            if current > expires:
                errors.append("lease TTL expired")
        except ValueError:
            errors.append("lease expires_at is invalid")
    else:
        errors.append("lease expires_at is missing")
    return errors


def validate_backend_environment(
    backend_preflight: Mapping[str, Any], environment: Mapping[str, str]
) -> list[str]:
    """Reject a worker whose process environment differs from the asset pin."""

    errors: list[str] = []
    if backend_preflight.get("execution_ready") is not True:
        errors.append("backend preflight is not execution-ready")
    required = backend_preflight.get("runtime_environment", {})
    if not isinstance(required, Mapping):
        return errors + ["backend runtime_environment is invalid"]
    for key, expected in required.items():
        if environment.get(str(key)) != expected:
            errors.append(f"backend environment mismatch: {key}")
    return errors


@dataclass(frozen=True)
class BackendFrameOutput:
    masks: Mapping[str, np.ndarray]
    visibility: Mapping[str, str]
    identity_warnings: Sequence[str] = ()


class ModalMaskBackend(Protocol):
    def step(
        self,
        *,
        frame_id: int,
        rgb_path: Path,
        prompt_masks: Mapping[str, Path],
    ) -> BackendFrameOutput: ...


def run_causal_sequence(
    rgb_manifest: Mapping[str, Any],
    prompt_manifest: Mapping[str, Any],
    backend: ModalMaskBackend,
    *,
    output_directory: Path,
) -> dict[str, Any]:
    """Run an injected backend sequentially and publish lossless uint8 NPY masks.

    This function is model-neutral and intended for unit/integration testing or
    a later leased GPU adapter.  Future prompt events are never passed to a
    target frame.
    """

    errors = validate_rgb_manifest(rgb_manifest) + validate_prompt_manifest(
        prompt_manifest, rgb_manifest
    )
    if errors:
        raise MaskChallengerError("; ".join(errors))
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)

    instance_ids = [item["instance_id"] for item in prompt_manifest["instances"]]
    event_map: dict[int, dict[str, Path]] = {}
    for instance in prompt_manifest["instances"]:
        for event in instance["prompt_events"]:
            event_map.setdefault(event["frame_id"], {})[instance["instance_id"]] = Path(
                event["mask"]["path"]
            )

    frame_records = []
    all_warnings: list[dict[str, Any]] = []
    for frame in rgb_manifest["frames"]:
        frame_id = frame["frame_id"]
        current_prompts = event_map.get(frame_id, {})
        result = backend.step(
            frame_id=frame_id,
            rgb_path=Path(frame["path"]),
            prompt_masks=current_prompts,
        )
        if set(result.masks) - set(instance_ids) or set(result.visibility) != set(instance_ids):
            raise MaskChallengerError("backend instance identity drift")

        visible_masks: list[np.ndarray] = []
        instances_out = []
        shape: tuple[int, int] | None = None
        for instance_id in instance_ids:
            visibility = result.visibility[instance_id]
            if visibility not in ALL_VISIBILITY:
                raise MaskChallengerError(f"invalid visibility: {visibility}")
            mask = result.masks.get(instance_id)
            if visibility in UNKNOWN_VISIBILITY:
                if mask is not None and bool(np.asarray(mask).any()):
                    raise MaskChallengerError("UNKNOWN visibility must not carry a pseudo-mask")
                instances_out.append(
                    {"instance_id": instance_id, "visibility": visibility, "mask": None}
                )
                continue
            if mask is None:
                raise MaskChallengerError("visible instance is missing modal mask")
            array = np.asarray(mask, dtype=bool)
            if array.ndim != 2 or not array.any():
                raise MaskChallengerError("visible modal mask must be non-empty HxW")
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise MaskChallengerError("instance mask shape drift")
            visible_masks.append(array)
            instance_directory = output_directory / instance_id
            instance_directory.mkdir(exist_ok=True)
            mask_path = instance_directory / f"{frame_id:06d}.npy"
            np.save(mask_path, array.astype(np.uint8), allow_pickle=False)
            instances_out.append(
                {
                    "instance_id": instance_id,
                    "visibility": visibility,
                    "mask": {
                        "path": str(mask_path.resolve()),
                        "bytes": mask_path.stat().st_size,
                        "sha256": file_sha256(mask_path),
                        "encoding": "NPY_UINT8_MODAL_BINARY",
                    },
                }
            )
        overlap = 0
        if visible_masks:
            occupancy = np.sum(np.stack(visible_masks, axis=0), axis=0)
            overlap = int(np.count_nonzero(occupancy > 1))
            if overlap:
                raise MaskChallengerError("independent instance modal masks overlap")
        warnings = list(result.identity_warnings)
        all_warnings.extend({"frame_id": frame_id, "warning": item} for item in warnings)
        frame_records.append(
            {
                "frame_id": frame_id,
                "prompt_instance_ids": sorted(current_prompts),
                "instances": instances_out,
                "identity_qa": {"overlap_pixels": overlap, "warnings": warnings},
            }
        )
    return {
        "schema_version": "causal-modal-mask-output-v71",
        "session_id": rgb_manifest["session_id"],
        "execution_mode": "CAUSAL_PROCESSING",
        "mask_semantics": "VISIBLE_MODAL_SURFACE_ONLY",
        "allow_instance_union": False,
        "frames": frame_records,
        "identity_qa": {
            "warning_count": len(all_warnings),
            "warnings": all_warnings,
            "instance_ids": instance_ids,
        },
        "authority_limit": {
            "maximum_publication": "MODAL_MASK_SUCCESSOR_CANDIDATE",
            "forbidden_claims": list(FORBIDDEN_CLAIMS),
        },
    }
