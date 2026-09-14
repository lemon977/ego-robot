"""Auditable shared-weight transfer from the legacy HumanEgo checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from utils.atomic_io import atomic_write_json
from utils.frozen_contract import file_reference, load_verified_torch_checkpoint


EXCLUDED_PREFIXES = (
    "action_proj.",
    "head_v.",
    "state_proj.",
    "head_future_state.",
    "robot_state_proj.",
    "robot_state_pos_emb",
)


def load_compatible_pretrained(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    report_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load only explicitly shared, shape-compatible layers and report all keys."""
    checkpoint_path = Path(checkpoint_path).absolute()
    reference = file_reference(checkpoint_path)
    if expected_sha256 is not None and reference["sha256"] != expected_sha256:
        raise ValueError("pretrained checkpoint differs from frozen run authority")
    checkpoint_path, payload = load_verified_torch_checkpoint(
        reference,
        allowed_roots=[checkpoint_path.parent],
        label="pretrained shared-weight checkpoint",
    )
    source = payload.get("model", payload)
    destination = model.state_dict()
    loaded = {}
    skipped_shape = []
    excluded = []
    unexpected = []
    for key, tensor in source.items():
        if key not in destination:
            unexpected.append(key)
        elif key.startswith(EXCLUDED_PREFIXES):
            excluded.append(key)
        elif tuple(tensor.shape) != tuple(destination[key].shape):
            skipped_shape.append({
                "key": key,
                "source_shape": list(tensor.shape),
                "destination_shape": list(destination[key].shape),
            })
        else:
            loaded[key] = tensor
    result = model.load_state_dict(loaded, strict=False)
    loaded_elements = sum(int(t.numel()) for t in loaded.values())
    total_elements = sum(int(t.numel()) for t in destination.values())
    report = {
        "policy": "shared layers only; state encoder and action head always reinitialized",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": reference["sha256"],
        "loaded_keys": sorted(loaded),
        "loaded_key_count": len(loaded),
        "loaded_parameter_elements": loaded_elements,
        "model_parameter_elements": total_elements,
        "loaded_parameter_fraction": loaded_elements / max(1, total_elements),
        "excluded_new_io_keys": sorted(excluded),
        "shape_mismatch": skipped_shape,
        "unexpected_source_keys": sorted(unexpected),
        "missing_destination_keys": sorted(result.missing_keys),
        "load_unexpected_keys": sorted(result.unexpected_keys),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    return report
