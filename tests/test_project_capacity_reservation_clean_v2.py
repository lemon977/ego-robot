from __future__ import annotations

import copy
from pathlib import Path

from tools import project_capacity_reservation_clean_v2 as adapter


def test_v2_profile_changes_only_fresh_receipt_and_pins_base() -> None:
    before = copy.deepcopy(adapter.base.CLI_PROFILES)
    before_parser = adapter.base.build_parser
    profile = adapter.install_profile()
    try:
        assert adapter.sha256(adapter.BASE) == adapter.BASE_SHA256
        expected = dict(before["clean-capacity-v1"])
        expected["output"] = adapter.RECEIPT
        assert profile == expected
        assert adapter.base.CLI_PROFILES["clean-capacity-v1"] == before[
            "clean-capacity-v1"
        ]
        for name, value in before.items():
            assert adapter.base.CLI_PROFILES[name] == value
        assert profile["reservation"] == before["clean-capacity-v1"]["reservation"]
        assert profile["authority"] == before["clean-capacity-v1"]["authority"]
        assert profile["bytes"] == 128 * 1024 * 1024
        assert profile["output"].endswith("RELEASE_V2.json")
    finally:
        adapter.base.CLI_PROFILES.clear()
        adapter.base.CLI_PROFILES.update(before)
        adapter.base.build_parser = before_parser


def test_exact_v2_public_argv_parses_without_translation() -> None:
    parser = adapter.build_parser()
    command = [
        "--project-root",
        str(adapter.PROJECT),
        "run",
        "--profile",
        "clean-capacity-v2",
        "--reservation",
        str(adapter.PROJECT / "tasks/control/.clean-post-formal-mask-zip-128m-v1.reserve"),
        "--authority",
        str(
            adapter.PROJECT
            / "tasks/control/.clean-post-formal-mask-zip-128m-v1.LIVE_AUTHORITY.json"
        ),
        "--receipt",
        str(adapter.PROJECT / adapter.RECEIPT),
        "--bytes",
        str(128 * 1024 * 1024),
        "--confirm-bytes",
        str(128 * 1024 * 1024),
        "--",
        str(adapter.PROJECT / "tools/clean_python.sh"),
        str(adapter.PROJECT / "tools/launch_clean_spatial12_bounded_v7.py"),
    ]
    args = parser.parse_args(command)
    assert args.profile == adapter.PROFILE
    assert args.profile != "clean-capacity-v1"
    assert args.receipt == Path(adapter.PROJECT / adapter.RECEIPT)
    assert args.command[-1].endswith("launch_clean_spatial12_bounded_v7.py")


def test_parser_only_appends_v2_to_run_choices() -> None:
    parser = adapter.build_parser()
    mode = next(action for action in parser._actions if action.dest == "mode")
    run = mode.choices["run"]
    run_profile = next(action for action in run._actions if action.dest == "profile")
    probe = mode.choices["probe"]
    probe_profile = next(
        action for action in probe._actions if action.dest == "profile"
    )
    assert tuple(run_profile.choices) == (
        *adapter.EXPECTED_RUN_PROFILE_CHOICES,
        adapter.PROFILE,
    )
    assert tuple(probe_profile.choices) == ("cpfs-probe-8m-v1",)
