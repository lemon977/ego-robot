import importlib.util
from pathlib import Path
import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools/run_generic_same_session_real_donor_v1.py"
SPEC = importlib.util.spec_from_file_location("donor", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(module)


def test_two_clear_agreeing_donors_are_exactly_copied(monkeypatch):
    monkeypatch.setattr(module, "OFFSETS", (-1, 1))
    rgbs = [np.full((3, 4, 3), 20, np.uint8), np.full((3, 4, 3), 200, np.uint8), np.full((3, 4, 3), 25, np.uint8)]
    removals = [np.zeros((3, 4), bool), np.ones((3, 4), bool), np.zeros((3, 4), bool)]
    objects = [np.zeros((3, 4), bool) for _ in rgbs]
    clean, provenance, metrics = module.build_frame(1, rgbs, removals, objects)
    assert np.all(clean == 20)
    assert np.all(provenance["source_kind"] == module.SOURCE_DONOR)
    assert metrics["supported_donor_pixels"] == 12


def test_disagreeing_or_masked_donors_remain_raw(monkeypatch):
    monkeypatch.setattr(module, "OFFSETS", (-1, 1))
    rgbs = [np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 100, np.uint8), np.full((2, 2, 3), 50, np.uint8)]
    removals = [np.zeros((2, 2), bool), np.ones((2, 2), bool), np.zeros((2, 2), bool)]
    objects = [np.zeros((2, 2), bool) for _ in rgbs]
    clean, provenance, metrics = module.build_frame(1, rgbs, removals, objects)
    assert np.all(clean == 100)
    assert np.all(provenance["source_kind"] == module.SOURCE_UNSUPPORTED)
    assert metrics["supported_donor_pixels"] == 0


def test_protected_donor_never_selected(monkeypatch):
    monkeypatch.setattr(module, "OFFSETS", (-1, 1))
    rgbs = [np.full((2, 2, 3), 20, np.uint8), np.full((2, 2, 3), 100, np.uint8), np.full((2, 2, 3), 20, np.uint8)]
    removals = [np.zeros((2, 2), bool), np.ones((2, 2), bool), np.zeros((2, 2), bool)]
    objects = [np.ones((2, 2), bool), np.zeros((2, 2), bool), np.zeros((2, 2), bool)]
    clean, _, metrics = module.build_frame(1, rgbs, removals, objects)
    assert np.all(clean == 100)
    assert metrics["supported_donor_pixels"] == 0


def test_checked_reference_allows_semantic_metadata(tmp_path):
    path = tmp_path / "mask.bin"
    path.write_bytes(b"mask")
    reference = module.ref(path)
    reference.update(observed=True, valid=True, global_physical_instance_id=0)
    assert module.checked(reference, "mask") == path.resolve()
