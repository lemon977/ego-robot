import numpy as np

from chaoyang.ops.run_human_to_robot_result_robot import pose_errors


def test_position_gate_measures_hand_root_not_tool(monkeypatch):
    from chaoyang.pipeline import robot_scene_state_cpu as arm

    tool = np.eye(4)
    mount = np.eye(4)
    mount[0, 3] = .0684
    target = np.eye(4)
    target[0, 3] = .0684
    target[1, 3] = .030
    monkeypatch.setattr(arm, "_tool_fk", lambda assets, side, q: tool)
    position, rotation = pose_errors(None, 0, np.zeros(7), target, mount)
    assert np.isclose(position, .030)
    assert np.isclose(rotation, 0.)
    assert position > .020
