import unittest

import numpy as np

from inference.receding_horizon import CausalPoseRateLimiter


def pose(x: float, yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    output = np.eye(4, dtype=np.float64)
    output[0, 3] = x
    output[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return output


class CausalPoseRateLimiterTests(unittest.TestCase):
    def test_translation_and_rotation_are_rate_limited(self) -> None:
        controller = CausalPoseRateLimiter(
            max_translation_step_m=0.01,
            max_rotation_step_rad=np.deg2rad(8.0),
            ema_alpha=0.65,
        )
        initial = np.stack([pose(0.0, 0.0), pose(0.0, 0.0)])
        controller.process(initial, np.asarray([True, True]))
        output, diagnostics = controller.process(
            np.stack([pose(0.2, 60.0), pose(0.2, 60.0)]),
            np.asarray([True, True]),
        )

        np.testing.assert_allclose(output[:, 0, 3], 0.01, atol=1e-8)
        self.assertAlmostEqual(diagnostics["rotation_step_deg"], 8.0, places=6)
        np.testing.assert_allclose(
            output[:, :3, :3].transpose(0, 2, 1) @ output[:, :3, :3],
            np.broadcast_to(np.eye(3), (2, 3, 3)),
            atol=1e-7,
        )

    def test_inactive_hand_keeps_previous_command(self) -> None:
        controller = CausalPoseRateLimiter()
        initial = np.stack([pose(0.0, 0.0), pose(0.0, 0.0)])
        controller.process(initial, np.asarray([True, True]))
        output, _ = controller.process(
            np.stack([pose(0.2, 60.0), pose(0.2, 60.0)]),
            np.asarray([True, False]),
        )
        np.testing.assert_allclose(output[1], initial[1])


if __name__ == "__main__":
    unittest.main()
