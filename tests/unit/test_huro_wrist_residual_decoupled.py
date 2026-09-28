"""CPU-only regression of the production HuRo wrist residual, no solver/model.

Run in the pinned HuRo environment with JAX_PLATFORMS=cpu and two CPU threads.
Synthetic correctness does not certify scene alignment or physical accuracy.
"""
import math
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
from chaoyang.ops.run_v5_huro import (
    wrist_pose_residual, WRIST_POSITION_SCALE_M, WRIST_ROTATION_SCALE_RAD,
)


def pose(x=0., angle=0.):
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.exp(jnp.array([0., 0., angle])), jnp.array([x, 0., 0.]))


class WristResidualTests(unittest.TestCase):
    def test_cpu_backend(self):
        self.assertTrue(all(device.platform == 'cpu' for device in jax.devices()))

    def test_legacy_mixed_pose_negative_control(self):
        actual, target = pose(.01, math.pi / 2), pose()
        legacy = np.asarray((target.inverse() @ actual).log())[:3]
        # Independent planar SE(3) logarithm oracle: ||rho|| = d*a/(2*sin(a/2)).
        oracle = .01 * (math.pi / 2) / (2 * math.sin(math.pi / 4))
        self.assertAlmostEqual(float(np.linalg.norm(legacy)), oracle, places=7)
        self.assertGreater(float(np.linalg.norm(legacy)), .011)
        corrected = np.asarray(wrist_pose_residual(actual, target, jnp.array(1.), jnp.array(0.)))
        np.testing.assert_allclose(corrected, [2., 0., 0., 0., 0., 0.], atol=1e-6)

    def test_position_only_rotation_invariance_and_gradient(self):
        for angle in (0., .7, math.pi / 2, 2.7):
            value = wrist_pose_residual(pose(.01, angle), pose(), jnp.array(1.), jnp.array(0.))
            np.testing.assert_allclose(value, [2., 0., 0., 0., 0., 0.], atol=1e-6)
        def cost(angle):
            value = wrist_pose_residual(pose(.01, angle), pose(), jnp.array(1.), jnp.array(0.))
            return jnp.sum(value ** 2)
        self.assertAlmostEqual(float(jax.grad(cost)(jnp.array(math.pi / 2))), 0., places=7)

    def test_rotation_only_translation_invariance(self):
        for translation in (0., .01, 3.):
            value = wrist_pose_residual(pose(translation, math.pi / 2), pose(), jnp.array(0.), jnp.array(1.))
            np.testing.assert_allclose(value, [0., 0., 0., 0., 0., 45.], atol=1e-4)

    def test_common_world_transform_cost_invariance(self):
        actual, target, world = pose(.01, .7), pose(-.02, -.3), pose(4., 1.2)
        a = wrist_pose_residual(actual, target, jnp.array(1.), jnp.array(1.))
        b = wrist_pose_residual(world @ actual, world @ target, jnp.array(1.), jnp.array(1.))
        self.assertAlmostEqual(float(jnp.sum(a * a)), float(jnp.sum(b * b)), places=2)

    def test_independent_side_masks(self):
        actual = jaxlie.SE3(jnp.stack((pose(.01, .7).wxyz_xyz, pose(.02, 1.1).wxyz_xyz)))
        target = jaxlie.SE3(jnp.stack((pose().wxyz_xyz, pose().wxyz_xyz)))
        value = np.asarray(wrist_pose_residual(actual, target, jnp.array([1., 0.]), jnp.array([0., 1.]))).reshape(2, 6)
        np.testing.assert_allclose(value[0], [2., 0., 0., 0., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(value[1, :3], 0., atol=1e-6)
        self.assertAlmostEqual(float(value[1, 5]), 1.1 / WRIST_ROTATION_SCALE_RAD, places=4)

    def test_disabled_masks_have_zero_gradient(self):
        def cost(x):
            value = wrist_pose_residual(pose(x, .8), pose(), jnp.array(0.), jnp.array(0.))
            return jnp.sum(value ** 2)
        self.assertEqual(float(cost(.01)), 0.)
        self.assertEqual(float(jax.grad(cost)(jnp.array(.01))), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
