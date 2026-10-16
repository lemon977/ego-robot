from __future__ import annotations

from pathlib import Path

import pytest

from chaoyang.ops import run_exact78_0915_cards001_base_temporal_replay_v1 as op


def test_run_cli_gate_denial_precedes_candidate_import_or_write(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(*_args, **_kwargs):
        raise RuntimeError("FENCED_GATE_DENIED")

    monkeypatch.setattr(op, "authorize_exact78_operation", deny)
    monkeypatch.setattr(op, "run_candidate", lambda *_: pytest.fail("candidate invoked"))
    output = Path(op.__file__).parent / "forbidden_base_temporal_output"
    assert not output.exists()
    with pytest.raises(RuntimeError, match="FENCED_GATE_DENIED"):
        op.main(["--output-root", str(output)])
    assert not output.exists()


def test_check_only_rejects_output_before_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(op, "check_only", lambda: pytest.fail("preflight invoked"))
    output = Path(op.__file__).parent / "forbidden_base_temporal_output"
    with pytest.raises(RuntimeError, match="CHECK_ONLY_MUST_NOT_SET_OUTPUT"):
        op.main(["--check-only", "--output-root", str(output)])
    assert not output.exists()


def test_direct_candidate_requires_packet_after_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(op, "authorize_exact78_operation", lambda *_: None)
    monkeypatch.setattr(op, "_active_packet", lambda: (_ for _ in ()).throw(
        RuntimeError("NO_MATCHING_PACKET")))
    output = Path(op.__file__).parent / "forbidden_base_temporal_output"
    with pytest.raises(RuntimeError, match="NO_MATCHING_PACKET"):
        op.run_candidate(output)
    assert not output.exists()
