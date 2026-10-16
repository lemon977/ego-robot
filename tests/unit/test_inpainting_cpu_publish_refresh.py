"""Synthetic CPU-only checks for the finite bootstrap's narrow route refresh."""
from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/maintenance/inpainting_optimization_task_v1.py"
TARGET = "chaoyang.governance.inpainting_candidate_registration"
spec = importlib.util.spec_from_file_location("_candidate_cpu_publish_refresh_tests", SCRIPT)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def fake_refresh(monkeypatch, events, *, failure_at=None, error=None):
    candidate = SimpleNamespace(__name__=TARGET)

    def invalidate():
        events.append("invalidate")
        if failure_at == "invalidate":
            raise error

    def load(name):
        events.append(("import", name))
        assert name == TARGET
        if failure_at == "import":
            raise error
        return candidate

    def reload(module):
        events.append(("reload", module))
        assert module is candidate
        if failure_at == "reload":
            raise error
        return candidate

    monkeypatch.setattr(bootstrap.importlib, "invalidate_caches", invalidate)
    monkeypatch.setattr(bootstrap.importlib, "import_module", load)
    monkeypatch.setattr(bootstrap.importlib, "reload", reload)
    return candidate


def test_refresh_precedes_publish_and_forwards_identical_arguments(monkeypatch):
    events = []
    candidate = fake_refresh(monkeypatch, events)
    state, index, expected, returned = {}, {}, object(), object()

    def publish(*args, **kwargs):
        events.append("publish")
        assert args == (state, index, "REGISTERED")
        assert args[0] is state and args[1] is index
        assert kwargs == {"expected_revision": expected}
        return returned

    result = bootstrap._publish(SimpleNamespace(_publish=publish), state, index,
                                "REGISTERED", expected_revision=expected)
    assert result is returned
    assert events == ["invalidate", ("import", TARGET), ("reload", candidate), "publish"]


def test_each_publication_refreshes_only_candidate_registration(monkeypatch):
    events = []
    candidate = fake_refresh(monkeypatch, events)
    infra = SimpleNamespace(_publish=lambda *a, **k: events.append("publish"))
    for phase in ("REGISTERED", "HEARTBEAT", "SEALED"):
        bootstrap._publish(infra, {}, {}, phase, expected_revision=0)
    assert events == ["invalidate", ("import", TARGET), ("reload", candidate), "publish"] * 3
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "reload"] == [candidate] * 3


@pytest.mark.parametrize("failure_at", ["invalidate", "import", "reload"])
def test_refresh_failure_prevents_publishing_and_propagates_original_error(monkeypatch, failure_at):
    events, error = [], RuntimeError("synthetic refresh error")
    fake_refresh(monkeypatch, events, failure_at=failure_at, error=error)
    infra = SimpleNamespace(_publish=lambda *a, **k: pytest.fail("published stale contract"))
    with pytest.raises(RuntimeError) as caught:
        bootstrap._publish(infra, {}, {}, "HEARTBEAT", expected_revision=4)
    assert caught.value is error
    assert "publish" not in events


def test_existing_publisher_exception_is_not_wrapped_or_retried(monkeypatch):
    events, error = [], ValueError("existing CAS conflict")
    fake_refresh(monkeypatch, events)

    def publish(*args, **kwargs):
        events.append("publish")
        raise error

    with pytest.raises(ValueError) as caught:
        bootstrap._publish(SimpleNamespace(_publish=publish), {}, {}, "SEALED")
    assert caught.value is error
    assert events.count("publish") == 1


def test_real_cached_module_refresh_changes_route_seen_by_fake_publisher(monkeypatch, tmp_path):
    """Real import/reload; fixture files are confined to --basetemp in this task."""
    module_name = "inpainting_cpu_refresh_fixture"
    module_path = tmp_path / (module_name + ".py")
    module_path.write_text("ROUTE = 'OLD_MASK_ONLY'\n", encoding="utf-8")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(tmp_path))
    original_import = importlib.import_module
    original_reload = importlib.reload
    cached = original_import(module_name)
    try:
        module_path.write_text("ROUTE = 'NEW_MASK_AND_CLEAN_ROUTE'\n", encoding="utf-8")
        assert cached.ROUTE == "OLD_MASK_ONLY"
        reloaded = []

        def candidate_import(name):
            assert name == TARGET
            return cached

        def reload_only_candidate(module):
            assert module is cached
            reloaded.append(module)
            return original_reload(module)

        monkeypatch.setattr(bootstrap.importlib, "import_module", candidate_import)
        monkeypatch.setattr(bootstrap.importlib, "reload", reload_only_candidate)
        infra = SimpleNamespace(_publish=lambda *a, **k: cached.ROUTE)
        assert bootstrap._publish(infra, {}, {}, "SEALED") == "NEW_MASK_AND_CLEAN_ROUTE"
        assert reloaded == [cached]
    finally:
        # Remove only this disposable CPU fixture from the module cache.
        sys.modules.pop(module_name, None)


def test_all_three_main_publications_use_refresh_helper_without_infra_edits():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    assert len([node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_publish"]) == 3
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "_publish" for node in calls)
