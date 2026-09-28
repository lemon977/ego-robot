"""Known-answer JAX/jaxls CPU constraints; no real robot, images or model."""
import ast
from pathlib import Path
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jaxls
from chaoyang.pipeline.huro_constrained_core_v1 import (
    joint_limit_inequalities, prepare_initial_guess, check_raw_joint_limits,
    constrained_core_source,
)

class SyntheticJoint(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
    pass

@jaxls.Cost.factory
def target_cost(values, variable, target):
    return values[variable] - target

@jaxls.Cost.factory(kind='constraint_leq_zero')
def bounded_cost(values, variable):
    return joint_limit_inequalities(values[variable], jnp.array([-1., -.5]), jnp.array([1., .5]))

def solve(target, bounded=True):
    variable = SyntheticJoint(0)
    costs = [target_cost(variable, jnp.array(target))]
    if bounded:
        costs.append(bounded_cost(variable))
    problem = jaxls.LeastSquaresProblem(costs, [variable]).analyze()
    kwargs = {'augmented_lagrangian': jaxls.AugmentedLagrangianConfig(tolerance_absolute=1e-7, tolerance_relative=1e-7)} if bounded else {}
    solution = problem.solve(initial_vals=jaxls.VarValues.make([variable.with_value(jnp.array([0., 0.]))]),
                             termination=jaxls.TerminationConfig(max_iterations=200), verbose=False, **kwargs)
    return np.asarray(solution[variable])

class NativeConstraintTests(unittest.TestCase):
    def test_cpu_and_native_constraint_api(self):
        self.assertTrue(all(d.platform == 'cpu' for d in jax.devices()))
        self.assertEqual(bounded_cost(SyntheticJoint(0)).kind, 'constraint_leq_zero')

    def test_signed_constraints_and_negative_control(self):
        np.testing.assert_allclose(joint_limit_inequalities(jnp.array([0., .2]), jnp.array([-1., -.5]), jnp.array([1., .5])), [-1., -.3, -1., -.7], atol=1e-7)
        raw = np.array([1.02, 0.])
        self.assertFalse(check_raw_joint_limits(raw, [-1., -.5], [1., .5])['pass'])
        np.testing.assert_array_equal(raw, [1.02, 0.])

    def test_feasible_target_actual_solve(self):
        raw = solve([.3, -.2])
        np.testing.assert_allclose(raw, [.3, -.2], atol=1e-4)
        self.assertTrue(check_raw_joint_limits(raw, [-1., -.5], [1., .5])['pass'])

    def test_conflicting_target_actual_solve_no_output_clipping(self):
        free = solve([2., -.8], bounded=False)
        constrained = solve([2., -.8], bounded=True)
        self.assertFalse(check_raw_joint_limits(free, [-1., -.5], [1., .5])['pass'])
        # Numerical AL residual is reported separately, never hidden by clipping.
        report = check_raw_joint_limits(constrained, [-1., -.5], [1., .5])
        print('CONFLICT_ORACLE', {'raw_q': constrained.tolist(), 'gate': report}, flush=True)
        np.testing.assert_allclose(constrained, [1., -.5], atol=1e-5)
        self.assertFalse(report['posthoc_clip_used'])

    def test_seed_does_not_redefine_home(self):
        home = np.array([0., .2])
        seed = np.array([.7, -.4])
        result = prepare_initial_guess(home, seed, 3, [1., 0.])
        np.testing.assert_array_equal(result, [[.7, .2]] * 3)
        np.testing.assert_array_equal(home, [0., .2])
        np.testing.assert_array_equal(seed, [.7, -.4])
        with self.assertRaises(ValueError):
            prepare_initial_guess(home, [np.nan, 0.], 3, [1., 0.])

    def test_source_drift_rejected(self):
        with self.assertRaises(ValueError):
            constrained_core_source('def unknown(): pass')

    def test_real_pinned_core_adapter_compiles_without_solving(self):
        from chaoyang.ops.run_v5_huro import adapted_core
        root = Path('/mnt/workspace/code/chaoyang')
        core, _, _, wrist_source = adapted_core(root / 'vendor/HuRo/pipeline/retargeting/retargeter.py')
        candidate = constrained_core_source(wrist_source)
        self.assertIn('kind="constraint_leq_zero"', candidate)
        self.assertIn('rest_pose=initial_cfg[None]', candidate)
        self.assertIn('return solution[var_joints], final_cost', candidate)
        self.assertNotIn('jnp.clip', candidate)
        core.__dict__['_completion_joint_limits'] = joint_limit_inequalities
        exec(compile(candidate, '<completion-core-candidate>', 'exec'), core.__dict__)
        self.assertTrue(callable(core.solve_retargeting_with_constraints))

if __name__ == '__main__':
    unittest.main(verbosity=2)
