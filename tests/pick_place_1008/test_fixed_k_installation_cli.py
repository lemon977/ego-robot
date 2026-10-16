"""CPU-only diagnostic routing; no real authorization or data/model execution."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

from chaoyang.ops import run_exact78_pick_place_raw_to_robot_candidate_v1 as operation


def _setup(monkeypatch, tmp_path, status="PASSED"):
    import chaoyang.governance.exact78_operation_gate as gate
    import chaoyang.paths as paths
    calls = []
    monkeypatch.setattr(gate, "authorize_operation_entry", lambda *a, **k: None)
    monkeypatch.setattr(paths, "get_paths", lambda: SimpleNamespace(repo_root=tmp_path))
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR",
                 "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(name, "test-original")
    def run(*args):
        calls.append(args)
        return {"status": status}
    monkeypatch.setitem(sys.modules, "chaoyang.pick_place_1008.installation_fixed_k",
                        SimpleNamespace(run_fixed_k_installation_fit=run))
    return calls


@pytest.mark.parametrize("status,code", [("PASSED", 0), ("REVIEW_REQUIRED", 2)])
def test_fixed_k_diagnostic_routes_all_five_pinned_paths_without_gpu(monkeypatch, tmp_path, status, code):
    calls = _setup(monkeypatch, tmp_path, status)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU diagnostic entered motion/model pipeline")
    monkeypatch.setitem(sys.modules, "chaoyang.pick_place_1008.pipeline", SimpleNamespace(run=forbidden))
    output = tmp_path / "candidate"
    assert operation.main(["--stage", "fixed-k-installation-fit",
                           "--manifest", "manifest.json", "--camera-config", "camera.json",
                           "--annotations", "bundle.json", "--source-index", "source.json",
                           "--output-root", str(output)]) == code
    assert calls == [(Path("manifest.json"), Path("bundle.json"), Path("camera.json"),
                      Path("source.json"), output)]


@pytest.mark.parametrize("missing", ["--camera-config", "--annotations", "--source-index"])
def test_fixed_k_diagnostic_rejects_incomplete_input_bundle_before_call(monkeypatch, tmp_path, missing):
    calls = _setup(monkeypatch, tmp_path)
    args = ["--stage", "fixed-k-installation-fit", "--manifest", "manifest.json",
            "--output-root", str(tmp_path / "candidate")]
    for option, value in [("--camera-config", "camera.json"), ("--annotations", "bundle.json"),
                          ("--source-index", "source.json")]:
        if option != missing:
            args.extend([option, value])
    with pytest.raises(SystemExit) as error:
        operation.main(args)
    assert error.value.code == 2
    assert calls == []
