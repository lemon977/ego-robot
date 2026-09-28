import unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jaxls
from chaoyang.pipeline.huro_projected_joints_v1 import (
    make_projected_joint_type, validate_physical_seed,
    intersect_affine_full_limits, projected_core_source,
)
from chaoyang.pipeline.huro_constrained_core_v1 import check_raw_joint_limits

@jaxls.Cost.factory
def goal(values, variable, target):
    return values[variable] - target

class ProjectedJointTests(unittest.TestCase):
    def test_retraction_all_trial_values_in_box(self):
        cls = make_projected_joint_type([-1., -.5, -.1], [1., .5, .1], [0., 0., 0.], [1, 1, 0])
        for delta in [-1e10, -10., -.01, 0., .01, 10., 1e10]:
            q = np.asarray(cls.retract_fn(jnp.array([.2, -.2, .0]), jnp.ones(3) * delta))
            self.assertTrue(check_raw_joint_limits(q, [-1., -.5, -.1], [1., .5, .1])['pass'])
            self.assertEqual(q[2], 0.)

    def test_inward_float32_rounding(self):
        cls = make_projected_joint_type([-.1], [.1], [0.], [1])
        for delta in [-100., 100.]:
            q = np.asarray(cls.retract_fn(jnp.zeros(1), jnp.array([delta])))
            self.assertTrue(check_raw_joint_limits(q, [-.1], [.1])['pass'])

    def test_conflict_and_interior_real_solver_raw_q(self):
        cls = make_projected_joint_type([-1., -.5], [1., .5], [0., 0.], [1, 1])
        for target in ([2., -.8], [.3, -.2]):
            v = cls(0)
            solution = jaxls.LeastSquaresProblem([goal(v, jnp.array(target))], [v]).analyze().solve(
                initial_vals=jaxls.VarValues.make([v.with_value(jnp.zeros(2))]),
                termination=jaxls.TerminationConfig(max_iterations=100), verbose=False)
            raw = np.asarray(solution[v])
            gate = check_raw_joint_limits(raw, [-1., -.5], [1., .5])
            print('PROJECTED_RAW_SOLUTION', {'target': target, 'raw_q': raw.tolist(), 'gate': gate}, flush=True)
            self.assertTrue(gate['pass'])
            np.testing.assert_allclose(raw, [1., -.5] if target[0] == 2. else target, atol=1e-5)

    def test_boundary_can_move_back_inside(self):
        cls = make_projected_joint_type([-1.], [1.], [0.], [1])
        q = cls.retract_fn(jnp.array([1.]), jnp.array([-.25]))
        np.testing.assert_array_equal(q, [.75])

    def test_home_seed_semantics_and_bad_seed_rejection(self):
        seed = validate_physical_seed([.7, -.4], [0., .2], [1, 0], [-1., -.5], [1., .5], 2)
        np.testing.assert_array_equal(seed, np.array([[.7, .2], [.7, .2]], dtype=seed.dtype))
        with self.assertRaises(ValueError):
            validate_physical_seed([.1], [0.], [1], [-.1], [.1], 1)
        with self.assertRaises(ValueError):
            validate_physical_seed([1.01, 0.], [0., 0.], [1, 1], [-1., -.5], [1., .5], 2)

    def test_mimic_negative_multiplier_and_fixed_limits(self):
        lo, hi = intersect_affine_full_limits([-2.], [2.], [-1., -.3], [1., .7], [[1.], [-2.]], [0., .1])
        np.testing.assert_allclose(lo, [-.3])
        np.testing.assert_allclose(hi, [.2])
        with self.assertRaises(ValueError):
            intersect_affine_full_limits([-1., -1.], [1., 1.], [-1.], [1.], [[1., 1.]], [0.])
        with self.assertRaises(ValueError):
            intersect_affine_full_limits([-1.], [1.], [-1.], [1.], [[0.]], [2.])

    def test_actual_official_source_all_consumers_and_compile(self):
        from chaoyang.ops.run_v5_huro import adapted_core
        core, _, _, source = adapted_core(Path('/mnt/workspace/code/chaoyang/vendor/HuRo/pipeline/retargeting/retargeter.py'))
        result = projected_core_source(source)
        self.assertNotIn('robot.joint_var_cls(', result)
        self.assertEqual(result.count('_completion_projected_joint_cls('), 3)
        self.assertIn('return solution[var_joints], final_cost', result)
        self.assertIn('rest_pose=initial_cfg[None]', result)
        core.__dict__['_completion_projected_joint_cls'] = make_projected_joint_type([-1.], [1.], [0.], [1])
        exec(compile(result, '<projected-source>', 'exec'), core.__dict__)
        self.assertTrue(callable(core.solve_retargeting_with_projected_joints))
        with self.assertRaises(ValueError):
            projected_core_source(source.replace('robot.joint_var_cls(', 'changed(', 1))

if __name__ == '__main__':
    unittest.main(verbosity=2)
