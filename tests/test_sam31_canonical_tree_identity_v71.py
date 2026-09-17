from pathlib import Path

from chaoyang.pipeline.sam31_compat_adapter_v1 import (
    OFFICIAL_CODE_COMMIT,
    OFFICIAL_GIT_TREE,
    RUNTIME_CLOSURE_GIT_TREE,
    RUNTIME_CLOSURE_REGULAR_BYTES,
    RUNTIME_CLOSURE_REGULAR_FILES,
    _canonical_tree_identity,
    verify_canonical_source_identity,
)


def test_metadata_free_sam31_matches_pinned_runtime_closure() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "vendor/SAM3"
    identity = _canonical_tree_identity(root)
    assert identity["git_tree"] == RUNTIME_CLOSURE_GIT_TREE
    assert identity["tracked_regular_files"] == RUNTIME_CLOSURE_REGULAR_FILES
    assert identity["tracked_regular_bytes"] == RUNTIME_CLOSURE_REGULAR_BYTES
    verified = verify_canonical_source_identity(root, project)
    assert verified["method"] == "PINNED_RUNTIME_FILE_CLOSURE"
    assert verified["official_code_commit"] == OFFICIAL_CODE_COMMIT
    assert verified["official_git_tree"] == OFFICIAL_GIT_TREE
    assert verified["complete_upstream_tree_retained"] is False
