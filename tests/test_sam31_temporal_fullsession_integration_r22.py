from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chaoyang.pipeline.sam31_temporal_fullsession_integration_r22 import (
    FrozenSeedReplay,
    TemporalIntegrationError,
    persist_compact_evidence,
    replay_seed_then_collect,
)


class FakeAdapter:
    def __init__(self, seed_id=7):
        self.seed_id = seed_id
        self.model = self
        self.state = None
        self.request = None
        self.closed = False

    def handle_request(self, request):
        if request["type"] == "start_session": self.state = {}; return {}
        if request["type"] == "close_session": self.closed = True; self.state = None; return {}
        raise AssertionError(request)

    def _session(self, session_id): return self.state

    def add_prompt(self, **kwargs):
        mask = np.zeros((1, 6, 8), bool); mask[0, 1:4, 2:5] = True
        return 0, {"masks": mask, "ids": np.array([self.seed_id])}

    def handle_stream_request(self, request):
        self.request = request
        for frame in range(3):
            mask = np.zeros((1, 6, 8), bool)
            if frame != 1: mask[0, 1:4, 2:5] = True
            yield {"frame_index": frame, "outputs": {"masks": mask, "ids": np.array([self.seed_id])}}


def normalize(output, height, width):
    return output["masks"], np.ones(len(output["ids"])), output["ids"]


def spec(seed_id=7):
    return FrozenSeedReplay("s", seed_id, (3, 2), (1, 0, 6, 5), None, 0, 3, 10, 6, 8)


def test_integration_replays_id_then_forwards_and_persists(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    seeds, rows, masks = replay_seed_then_collect(adapter, tmp_path, spec(), normalize)
    assert adapter.request["propagation_direction"] == "forward"
    assert all(row["accepted_object_id"] == 7 for row in rows)
    assert adapter.closed and len(seeds) == 2 and len(rows) == 3
    artifacts = persist_compact_evidence(tmp_path / "out", seeds, rows, masks, lambda _: np.zeros((6, 8, 3), np.uint8))
    assert len(Path(artifacts["compact_rows"]).read_text().splitlines()) == 3
    assert len(artifacts["failure_pngs"]) == 1


def test_integration_rejects_seed_id_drift_and_closes(tmp_path: Path) -> None:
    adapter = FakeAdapter(seed_id=8)
    try:
        replay_seed_then_collect(adapter, tmp_path, spec(seed_id=7), normalize)
        raise AssertionError("expected fail-closed ID mismatch")
    except TemporalIntegrationError as exc:
        assert "accepted object ID absent" in str(exc)
    assert adapter.closed
