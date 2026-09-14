"""Independent regression for the D4 frozen-raw pixel lineage boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pipeline import lr_contact_phase_object6d_evidence as d4
from pipeline import lr_distributed_side_evidence as base


def _load_author_fixture():
    path = Path(__file__).with_name("test_lr_contact_phase_object6d_evidence.py")
    spec = importlib.util.spec_from_file_location("d4_author_fixture_for_independent_qa", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_raw_mask_must_be_reverified_at_copy_boundary(tmp_path, monkeypatch):
    fixture = _load_author_fixture()
    authorities, raws, contact = fixture.inputs(tmp_path, left_contact=True)
    original_measure = d4.measure_contact_phase
    changed = False

    def measure_and_change_in_memory_raw(value):
        nonlocal changed
        result = original_measure(value)
        if not changed:
            raws[0].mask[:] = False
            raws[0].mask[44, 45] = True
            changed = True
        return result

    monkeypatch.setattr(d4, "measure_contact_phase", measure_and_change_in_memory_raw)
    with pytest.raises(base.DistributedSideEvidenceError, match="raw source/mask payload mismatch"):
        d4.select_frame_contact_phase(
            authorities,
            raws,
            contact,
            evidence_root=tmp_path,
            image_shape=(100, 100),
        )
