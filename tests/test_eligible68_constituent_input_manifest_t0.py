from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import author_qa_eligible68_constituent_input_manifest_t0 as author_qa
from tools import build_eligible68_constituent_input_manifest_t0 as subject
from tools import eligible68_no_discovery_guard_t0 as no_discovery_guard


class NoPathConstructionRoot:
    def __init__(self) -> None:
        self.divisions = 0

    def __truediv__(self, _other: object) -> "NoPathConstructionRoot":
        self.divisions += 1
        raise AssertionError("a root/session path was constructed before rejection")


def test_explicit_population_has_exact_order_and_disjoint_excluded_union() -> None:
    assert len(subject.TRAIN60) == 60
    assert len(subject.VALIDATION8) == 8
    assert subject.ELIGIBLE68 == subject.TRAIN60 + subject.VALIDATION8
    assert len(subject.ELIGIBLE68) == len(set(subject.ELIGIBLE68)) == 68
    assert len(subject.FORBIDDEN10) == 10
    assert set(subject.ELIGIBLE68).isdisjoint(subject.FORBIDDEN10)
    assert subject.TRAIN60[0] == "grap_a_cap_004"
    assert subject.VALIDATION8[0] == "grap_a_cap_002"


@pytest.mark.parametrize("session_id", sorted(subject.FORBIDDEN10))
def test_every_forbidden_identity_is_rejected_before_any_path_division(
    session_id: str,
) -> None:
    production = NoPathConstructionRoot()
    sidecar = NoPathConstructionRoot()
    with pytest.raises(subject.ManifestError, match="before path construction"):
        subject.construct_session_paths(session_id, production, sidecar)
    assert production.divisions == 0
    assert sidecar.divisions == 0


def test_unknown_identity_is_rejected_before_any_path_division() -> None:
    production = NoPathConstructionRoot()
    sidecar = NoPathConstructionRoot()
    with pytest.raises(subject.ManifestError, match="absent from explicit"):
        subject.construct_session_paths("grap_a_cap_999", production, sidecar)
    assert production.divisions == 0
    assert sidecar.divisions == 0


def test_allowed_path_contract_has_exact_ten_constituents() -> None:
    production = Path("/production")
    sidecar = Path("/sidecar")
    paths = subject.construct_session_paths("grap_a_cap_004", production, sidecar)
    assert len(paths) == 10
    assert paths["production_status"] == Path("/production/grap_a_cap_004/status.json")
    assert paths["retarget_sidecar"] == Path(
        "/sidecar/kai22/grap_a_cap_004/sidecar.npz"
    )
    assert paths["physical_hand_axis_metadata"] == Path(
        "/production/grap_a_cap_004/08_final_v3/joint_optimized_3d.json"
    )


@pytest.mark.parametrize("module", [subject, author_qa])
def test_metadata_tools_have_no_discovery_api_calls(module: object) -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert no_discovery_guard.find_forbidden_discovery_calls(source) == ()


@pytest.mark.parametrize(
    "source",
    [
        "from os import listdir as names\nnames('/not-opened')",
        "from os import walk as traverse\ntraverse('/not-opened')",
        "import os as filesystem\nindirect = filesystem\nindirect.scandir('/not-opened')",
        "import os\nlauncher = os.system\nlauncher('find /not-opened')",
        "from os import popen as launch\nlaunch('find /not-opened')",
        "import subprocess as process\nprocess.run(['find', '/not-opened'])",
        "from subprocess import Popen as launch\nlaunch(['find', '/not-opened'])",
        "from glob import glob as discover\ndiscover('/not-opened/*')",
        "from pathlib import Path\nPath('/not-opened').glob('*')",
        "from pathlib import Path as P\nscan = P.rglob\nscan(P('/not-opened'), '*')",
        "from pathlib import Path\nscan = Path('/not-opened').iterdir\nscan()",
        "from pathlib import Path\np = Path('/not-opened')\nscan = p.rglob\nscan('*')",
        "import os\ngetattr(os, 'listdir')('/not-opened')",
        "from builtins import getattr as lookup\nimport os\nlookup(os, 'walk')('/not-opened')",
        "import importlib\nimportlib.import_module('subprocess').run(['find'])",
        "module = __import__('os')\nmodule.listdir('/not-opened')",
    ],
)
def test_no_discovery_guard_rejects_alias_process_pathlib_and_dynamic_forms(
    source: str,
) -> None:
    assert no_discovery_guard.find_forbidden_discovery_calls(source)


def test_no_discovery_guard_accepts_literal_point_path_code() -> None:
    source = "from pathlib import Path\nitem = Path('/fixed') / 'literal.json'"
    assert no_discovery_guard.find_forbidden_discovery_calls(source) == ()


def test_live_metadata_build_closes_all_68_constituent_rows_without_admission() -> None:
    manifest = subject.build_manifest()
    assert manifest["population"]["eligible68_sessions"] == 68
    assert manifest["population"]["eligible68_frames"] == 28265
    assert manifest["aggregate_checks"] == {
        "session_rows": 68,
        "constituent_rows": 680,
        "sessions_complete": 68,
        "unique_session_ids": 68,
        "ordered_ids_equal_explicit_allowlist": True,
        "all_legacy_bundle_digests_match": True,
        "anomalies": [],
    }
    assert [row["session_id"] for row in manifest["sessions"]] == list(
        subject.ELIGIBLE68
    )
    assert all(row["constituent_count"] == 10 for row in manifest["sessions"])
    assert all(all(row["completeness"].values()) for row in manifest["sessions"])
    assert manifest["admission"]["pixel_or_selector_execution_admission"] is False
    assert manifest["admission"]["gpu_queueable_sessions"] == 0
    assert manifest["access_boundary"]["raw_pixels_opened"] == 0

    qa = author_qa.run_qa(
        manifest,
        SimpleNamespace(
            path="/synthetic/subject.json",
            bytes=1,
            sha256="0" * 64,
            device=0,
            inode=0,
        ),
    )
    assert qa["status"].startswith("PASS_68_OF_68")
    assert qa["recomputed"]["constituent_files_exact_rehashed"] == 680
    assert qa["verdict"]["pixel_or_selector_execution_admission"] is False
