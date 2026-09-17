import json
from pathlib import Path

from chaoyang.ops.run_object6d_selected_camera_adapter_batch_v71 import adapter_command


def test_old_single_object_layout_gets_explicit_object_zero(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "registration_authority": {"path": "/reg.npz"},
        "camera_to_world": {"path": "/c2w.npz"},
        "camera_to_world_key": "c2w",
    }))
    result_dir = tmp_path / "poker" / "session"
    result_dir.mkdir(parents=True)
    result = result_dir / "RESULT.json"
    result.write_text(json.dumps({
        "task": "poker", "session_id": "session",
        "inputs": {"spec": {"path": str(spec)}},
        "artifacts": {"trajectory": {"path": "/object.npz"}},
    }))
    command, destination = adapter_command(result, tmp_path / "out")
    assert destination == tmp_path / "out" / "poker" / "session" / "physical_object_0"
    assert "--artifact-revision" in command
