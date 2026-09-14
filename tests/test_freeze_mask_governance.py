from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import tools.freeze_mask_governance as freezer

from tools.freeze_mask_governance import (
    FreezeError,
    atomic_publish,
    authorization_evidence,
    canonical_json,
    read_regular_nofollow,
    sha256_bytes,
)


def test_authorization_preserves_raw_text_and_context_separately() -> None:
    evidence = authorization_evidence()
    assert evidence["raw_user_text"] == "可以全部都推进，我都同意，以任务进行为最高优先"
    assert evidence["contextual_scope"] == [
        "授权 T2-B0a 签发 benchmark v2 freeze ref。",
        "授权 T2-B0b 按 HUMAN_REVIEW_POLICY 冻结 auditor_a1。",
    ]
    assert evidence["contextual_scope_source"] == "immediately_preceding_authorization_request"


def test_atomic_publish_is_noreplace_readonly_and_digest_bound(tmp_path: Path) -> None:
    parent = tmp_path / "freezes"
    identity = {"kind": "fixture", "authorized": True}
    record = {**identity, "freeze_identity_sha256": sha256_bytes(canonical_json(identity))}
    path, digest = atomic_publish(parent, identity, "FREEZE.json", record)
    assert path.is_file() and not path.is_symlink()
    assert sha256_bytes(path.read_bytes()) == digest
    assert path.stat().st_mode & 0o222 == 0
    assert path.parent.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        atomic_publish(parent, identity, "FREEZE.json", record)


def test_noreplace_fallback_uses_exclusive_reservation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target"

    class UnsupportedRenameAt2:
        def renameat2(self, *_args: object) -> int:
            freezer.ctypes.set_errno(freezer.errno.EINVAL)
            return -1

    monkeypatch.setattr(freezer.ctypes, "CDLL", lambda *_args, **_kwargs: UnsupportedRenameAt2())
    freezer._rename_noreplace(source, target)
    assert target.is_dir()
    assert not source.exists()
    assert not target.with_name(f".{target.name}.publish.lock").exists()


def test_secure_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"x": 1}), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FreezeError, match="symlink"):
        read_regular_nofollow(link)


def test_identity_digest_is_key_order_independent() -> None:
    assert sha256_bytes(canonical_json({"a": 1, "b": 2})) == sha256_bytes(
        canonical_json({"b": 2, "a": 1})
    )


def test_auditor_decoder_contract_requires_0_or_255() -> None:
    source = (Path(__file__).parents[1] / "pipeline/auditor/mask_auditor_a1.py").read_text()
    assert "(values != 0) & (values != 255)" in source
    assert "mask must be lossless binary 0/255" in source
