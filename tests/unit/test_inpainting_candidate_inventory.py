from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.inpainting_candidate.inventory import (
    REQUIRED_0918, RootBoundReader, ordered_refs, refresh_jobs, select_0918,
)


def reference(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def make_session(root, sid, *, group=None, missing=None):
    session = root / "task" / sid
    for name in REQUIRED_0918:
        if name == missing:
            continue
        path = session / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "manifest.json":
            payload = {"session": sid}
            if group:
                payload["original_acquisition_group"] = group
            path.write_text(json.dumps(payload))
        else:
            path.write_bytes(b"original-" + sid.encode())
    return session


def test_reader_hashes_actual_bytes_without_per_file_realpath(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.bin").write_bytes(b"actual")
    expected = reference(raw / "a.bin")
    with RootBoundReader(raw, raw) as reader:
        monkeypatch.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("per-file resolve forbidden"))
        assert reader.reference(raw / "a.bin", expected) == expected


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_reader_rejects_internal_symlinks(tmp_path, kind):
    raw, outside = tmp_path / "raw", tmp_path / "outside"
    raw.mkdir()
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"not-authorised")
    if kind == "file":
        (raw / "link.bin").symlink_to(outside / "secret.bin")
        target = raw / "link.bin"
    else:
        (raw / "link").symlink_to(outside, target_is_directory=True)
        target = raw / "link" / "secret.bin"
    with RootBoundReader(raw, raw) as reader, pytest.raises(OSError):
        reader.reference(target)


@pytest.mark.parametrize("path", ["../outside.bin", "a/../../outside.bin"])
def test_reader_rejects_traversal(tmp_path, path):
    with RootBoundReader(tmp_path, tmp_path) as reader, pytest.raises(ValueError, match="TRAVERSAL"):
        reader.reference(tmp_path / path)


def test_reader_rejects_outside_root(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with RootBoundReader(raw, raw) as reader, pytest.raises(ValueError, match="OUTSIDE"):
        reader.reference(tmp_path / "outside.bin")


def test_reader_rejects_fifo_before_read(tmp_path):
    import os
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with RootBoundReader(tmp_path, tmp_path) as reader, pytest.raises(ValueError, match="NOT_REGULAR"):
        reader.reference(fifo)


@pytest.mark.parametrize("drift", ["bytes", "sha256"])
def test_reader_rejects_actual_source_drift(tmp_path, drift):
    path = tmp_path / "source.bin"
    path.write_bytes(b"source")
    expected = reference(path)
    expected[drift] = 0 if drift == "bytes" else "0" * 64
    with RootBoundReader(tmp_path, tmp_path) as reader, pytest.raises(ValueError, match="DRIFT"):
        reader.reference(path, expected)


def test_directory_fds_are_cached(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("a", "b"):
        (raw / name).write_bytes(b"actual")
    with RootBoundReader(tmp_path, tmp_path) as reader:
        reader.reference(raw / "a")
        cached = reader._fds.copy()
        reader.reference(raw / "b")
        assert cached == reader._fds


def test_order_checks_fixed_development_and_challenges_first():
    rows = [{"session_id": sid, "source_refs": {"video": {"path": "/raw/" + sid, "sha256": sid}}}
            for sid in ("199", "002", "050", "001", "003", "008", "100", "114", "039", "007")]
    assert [ref["path"].split("/")[-1] for ref in ordered_refs({"sessions": rows})] == [
        "001", "003", "008", "100", "114", "050", "039", "007", "199", "002"]


def test_conflicting_frozen_reference_is_not_silently_deduplicated():
    rows = [{"session_id": sid, "source_refs": {"x": {"path": "/raw/a", "sha256": sid}}} for sid in ("001", "002")]
    with pytest.raises(ValueError, match="REFERENCE_CONFLICT"):
        ordered_refs({"sessions": rows})


def test_0918_selection_is_path_order_not_pixel_quality(tmp_path):
    for sid in ("003", "001", "002"):
        make_session(tmp_path, sid)
    result = select_0918(tmp_path, tmp_path, maximum=2)
    assert [row["identity"] for row in result["selected"]] == ["001", "002"]
    assert all(row["group_independence"] == "SPLIT_NOT_ESTABLISHED" for row in result["selected"])
    assert all(not row["blind_test"] and not row["visual_reviewed_for_selection"] and not row["hdf5_datasets_read"] for row in result["selected"])
    assert all(set(row["source_refs"]) == set(REQUIRED_0918) for row in result["selected"])


def test_0918_incomplete_first_source_does_not_select_old_processed(tmp_path):
    first = make_session(tmp_path, "001", missing="raw/vst.h264")
    (first / "processed").mkdir()
    (first / "processed" / "vst.mp4").write_bytes(b"forbidden-history")
    make_session(tmp_path, "002")
    result = select_0918(tmp_path, tmp_path, maximum=1)
    assert result["selected"][0]["identity"] == "002"
    assert result["rejected"][0]["status"] == "INPUT_INCOMPLETE_OR_UNSAFE"


def test_0918_same_explicit_group_not_counted_twice(tmp_path):
    make_session(tmp_path, "001", group="same")
    make_session(tmp_path, "002", group="same")
    make_session(tmp_path, "003", group="other")
    result = select_0918(tmp_path, tmp_path)
    assert [row["identity"] for row in result["selected"]] == ["001", "003"]
    assert result["rejected"][0]["status"] == "REPEATED_EXPLICIT_SOURCE_GROUP"


def test_refresh_real_subprocesses_actual_hash_and_first_progress(tmp_path, monkeypatch):
    import chaoyang.inpainting_candidate.inventory as inventory
    # ``python -m inventory`` gives this module __name__ == '__main__'.
    monkeypatch.setattr(inventory, "__name__", "__main__")
    raw, output = tmp_path / "raw", tmp_path / "output"
    raw.mkdir()
    output.mkdir()
    files = []
    for name in ("a.bin", "b.bin"):
        path = raw / name
        path.write_bytes(name.encode())
        files.append(reference(path))
    jobs = [{"kind": "hash", "job_id": str(index), "root": str(raw), "resolved_root": str(raw), "expected": ref}
            for index, ref in enumerate(files)]
    result = refresh_jobs(jobs, output, datetime.now(timezone.utc) + timedelta(seconds=20), 2)
    assert len(result["verified"]) == 2 and not result["failures"]
    assert not any(row["still_alive"] for row in result["worker_states"])
    progress = json.loads((output / "INPUT_PROGRESS.json").read_text())
    assert progress["verified_files"] == 2
    assert progress["status"] == "COMPLETED_ACTUAL_INPUT_SHA"


def test_refresh_deadline_has_no_fake_verification(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    job = {"kind": "hash", "job_id": "a", "root": str(tmp_path), "resolved_root": str(tmp_path),
           "expected": {"path": str(tmp_path / "missing"), "bytes": 0, "sha256": "0" * 64}}
    result = refresh_jobs([job], output, datetime.now(timezone.utc) - timedelta(seconds=1), 1)
    assert result["verified"] == []
    assert result["failures"][0]["failure_code"] == "FINITE_G0_DEADLINE"


def test_cross_source_failure_does_not_discard_valid_regression(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    path = tmp_path / "good.bin"
    path.write_bytes(b"good")
    jobs = [{"kind": "hash", "job_id": "a", "root": str(tmp_path), "resolved_root": str(tmp_path), "expected": reference(path)},
            {"kind": "select_0918", "job_id": "cross", "root": str(tmp_path / "absent"), "resolved_root": str(tmp_path / "absent"), "maximum": 8}]
    result = refresh_jobs(jobs, output, datetime.now(timezone.utc) + timedelta(seconds=20), 1)
    assert len(result["verified"]) == 1 and not result["failures"]
    assert result["selection"]["status"] == "BLOCKED_0918_METADATA"
