"""Public submit routing only; all sources and the T0 freeze are synthetic."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chaoyang.cli import exact78_submit as submit
from chaoyang.cli import exact78_newraw_cpu_submit_candidate as newraw
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    ColdStartInputError, PROJECT_ROOT, file_ref,
)


class Exact78SubmitNewrawDispatchCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix="exact78-submit-dispatch-test-", dir=PROJECT_ROOT / "_run/maintenance")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "raw_data"
        self.data.mkdir()
        self.selected: dict[str, dict[str, str]] = {}
        for session_id in sorted(submit.T0_SELECTED_IDS):
            session = self.data / "frozen" / session_id
            session.mkdir(parents=True)
            self.selected[session_id] = {"source_path": str(session)}
        self.t0 = Path(self.selected["play_cards_0902_042"]["source_path"])
        self.new = self.data / "tracker_0902/playing_cards/play_cards_0902_001"
        self.new.mkdir(parents=True)
        self.freeze = self.root / "INPUT_FREEZE.json"
        self._write_freeze()

    def _write_freeze(self) -> None:
        self.freeze.write_text(json.dumps({
            "schema_version": "exact78-raw-t0-input-freeze-v1",
            "selected": self.selected,
        }, sort_keys=True), encoding="utf-8")

    def _call(self, args: list[str], *, freeze_sha: str | None = None) -> dict:
        printed = StringIO()
        with patch.object(submit, "DATA_ROOT", self.data), \
                patch.object(submit, "T0_FREEZE", self.freeze), \
                patch.object(submit, "T0_FREEZE_SHA", freeze_sha or file_ref(self.freeze)["sha256"]), \
                redirect_stdout(printed):
            self.assertEqual(submit.main(args), 0)
        return json.loads(printed.getvalue())

    def test_selected_t0_preview_and_attach_keep_legacy_behavior(self) -> None:
        request = {"status": "CANDIDATE_REQUEST_ONLY", "path": str(self.t0)}
        attached = {"status": "ATTACHED_TO_ACTIVE_CANDIDATE"}
        with patch.object(submit, "build_request", return_value=request) as build, \
                patch.object(submit, "_attach_existing", return_value=attached) as attach, \
                patch.object(newraw, "submit_one") as new_submit:
            self.assertEqual(self._call(["--session", str(self.t0), "--preview"]), request)
            self.assertEqual(self._call(["--session", str(self.t0)]), attached)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(attach.call_count, 1)
        new_submit.assert_not_called()

    def test_non_t0_raw_preview_and_submit_use_new_cpu_only_candidate(self) -> None:
        planned = {"status": "FIRST_RUN_READY_NOT_REGISTERED"}
        with patch.object(submit, "build_request") as old_build, \
                patch.object(submit, "_attach_existing") as old_attach, \
                patch.object(newraw, "submit_one", return_value=planned) as new_submit:
            self.assertEqual(self._call(["--session", str(self.new), "--preview"]), planned)
            self.assertEqual(self._call(["--session", str(self.new)]), planned)
        old_build.assert_not_called()
        old_attach.assert_not_called()
        self.assertEqual(new_submit.call_count, 2)
        self.assertEqual(new_submit.call_args_list[0].kwargs, {"preview": True})
        self.assertEqual(new_submit.call_args_list[1].kwargs, {"preview": False})
        self.assertEqual(new_submit.call_args_list[0].args, (self.new,))

    def test_manifest_branch_is_unchanged_and_does_not_classify_t0(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        request = {"status": "CANDIDATE_REQUEST_ONLY"}
        with patch.object(submit, "build_request", return_value=request) as build, \
                patch.object(submit, "_frozen_t0_selected_paths") as classify, \
                patch.object(newraw, "submit_one") as new_submit:
            self.assertEqual(self._call(["--manifest", str(manifest), "--preview"]), request)
        build.assert_called_once_with(manifest=manifest)
        classify.assert_not_called()
        new_submit.assert_not_called()

    def test_t0_freeze_sha_drift_rejects_before_either_dispatch(self) -> None:
        with patch.object(submit, "build_request") as old_build, \
                patch.object(newraw, "submit_one") as new_submit:
            with self.assertRaisesRegex(ColdStartInputError, "T0_FREEZE_SHA_DRIFT"):
                self._call(["--session", str(self.new)], freeze_sha="0" * 64)
        old_build.assert_not_called()
        new_submit.assert_not_called()

    def test_selected_path_duplicate_rejects_even_with_matching_freeze_sha(self) -> None:
        self.selected["play_cards_0901_001"]["source_path"] = str(self.t0)
        self._write_freeze()
        with patch.object(newraw, "submit_one") as new_submit:
            with self.assertRaisesRegex(ColdStartInputError, "T0_SELECTED_PATH_DUPLICATE"):
                self._call(["--session", str(self.new)])
        new_submit.assert_not_called()

    def test_derived_session_and_t0_symlink_alias_never_dispatch(self) -> None:
        derived = self.data / "preprocess/all_data/play_cards_0902_001"
        derived.mkdir(parents=True)
        alias = self.data / "t0_alias"
        alias.symlink_to(self.t0, target_is_directory=True)
        with patch.object(submit, "build_request") as old_build, \
                patch.object(newraw, "submit_one") as new_submit:
            with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_SESSION_FORBIDDEN"):
                self._call(["--session", str(derived)])
            with self.assertRaisesRegex(ColdStartInputError, "SYMLINK_SESSION_FORBIDDEN"):
                self._call(["--session", str(alias)])
        old_build.assert_not_called()
        new_submit.assert_not_called()

    def test_t0_selected_child_cannot_be_submitted_as_a_new_session(self) -> None:
        child = self.t0 / "historical_candidate"
        child.mkdir()
        with patch.object(submit, "build_request") as old_build, \
                patch.object(newraw, "submit_one") as new_submit:
            with self.assertRaisesRegex(ColdStartInputError, "T0_SELECTED_CHILD_NOT_RAW_SESSION"):
                self._call(["--session", str(child)])
        old_build.assert_not_called()
        new_submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
