"""Regression tests for constraint-aware optimization and field publication."""
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fmri_rt_preproc.pyhysco_apply import EPIMRIDistortionCorrection, DataObject
from optimization.GaussNewton import GaussNewton, PCG
from optimization.EPIOptimize import EPIOptimize
from EPI_MRI.utils import m_plus


class ScalarProblem:
    """A quadratic whose unconstrained minimizer is outside the barrier."""
    def __init__(self, target=1.5):
        self.D = SimpleNamespace(mat_mul=lambda x: x)
        self.beta = 1e-4
        self.target = target
        self.trials = []
        self.accepted = []
        self.barrier = EPIMRIDistortionCorrection.__new__(EPIMRIDistortionCorrection)

    def eval(self, x, yref=None, do_derivative=False, calc_hessian=False):
        g, dg, ddg = self.barrier.phi_EPI(x, do_derivative, calc_hessian)
        self.Dc = ((x - self.target).square() / 2).sum()
        self.Sc = torch.zeros((), dtype=x.dtype)
        value = self.Dc + self.beta*g.sum()
        if not do_derivative:
            self.trials.append((float(x.item()), float(value)))
            return value
        self.accepted.append(float(x.item()))
        gradient = x - self.target + self.beta*dg
        return value, gradient, lambda p: (1 + self.beta*ddg)*p, None


def test_optimizer_backtracks_instead_of_accepting_folded_field(tmp_path):
    problem = ScalarProblem()
    optimizer = GaussNewton(problem, max_iter=50, path=str(tmp_path))
    optimizer.run_correction(torch.zeros(1, dtype=torch.float64))
    assert any(abs(x) >= 1 and np.isinf(value) for x, value in problem.trials)
    assert all(abs(x) < 1 for x in problem.accepted)
    # A feasible field alone is insufficient: the constrained objective should
    # be near its stationary point, not merely at the first shortened step.
    _, gradient, _, _ = problem.eval(optimizer.Bc, do_derivative=True, calc_hessian=True)
    assert torch.linalg.vector_norm(gradient) < 1e-2
    assert optimizer.converged
    assert optimizer.linear_solver.max_iter == 50


def test_explicit_linear_solver_budget_is_preserved(tmp_path):
    problem = ScalarProblem()
    solver = PCG(max_iter=4, tol=.1)
    optimizer = GaussNewton(problem, max_iter=50, linear_solver=solver, path=str(tmp_path))
    optimizer.run_correction(torch.zeros(1, dtype=torch.float64))
    assert optimizer.barrier_limited and optimizer.converged
    assert solver.max_iter == 4


def test_stationary_initialization_skips_pcg_and_reports_convergence(tmp_path, monkeypatch):
    problem = ScalarProblem(target=0)
    optimizer = GaussNewton(problem, path=str(tmp_path))
    def unexpected_solve(*args, **kwargs):
        pytest.fail('An already-stationary field must not enter PCG')
    monkeypatch.setattr(optimizer.linear_solver, 'eval', unexpected_solve)
    initial = torch.zeros(1, dtype=torch.float64)
    result = optimizer.run_correction(initial)
    torch.testing.assert_close(result, initial)
    assert optimizer.converged and optimizer.convergence_reason == 'gradient_tolerance'
    assert not optimizer.barrier_limited and not problem.trials
    assert len(problem.accepted) == 1
    log = (tmp_path/'log_file.txt').read_text()
    assert 'gradient tolerance at initialization' in log
    assert 'total function evaluations: 1' in log


def test_reused_optimizer_resets_adaptive_budget_and_barrier_state(tmp_path):
    problem = ScalarProblem()
    optimizer = GaussNewton(problem, max_iter=50, path=str(tmp_path))
    initial = torch.zeros(1, dtype=torch.float64)
    optimizer.run_correction(initial)
    assert optimizer.barrier_limited and optimizer.linear_solver.max_iter == 50

    problem.target = 0
    optimizer.run_correction(initial)
    assert optimizer.converged and optimizer.convergence_reason == 'gradient_tolerance'
    assert not optimizer.barrier_limited
    assert optimizer.linear_solver.max_iter == 10


@pytest.mark.parametrize('component', ['objective', 'gradient'])
@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
@pytest.mark.parametrize('evaluation', ['initial', 'accepted'])
def test_nonfinite_evaluations_cannot_report_convergence(tmp_path, component, bad, evaluation):
    class NonfiniteProblem(ScalarProblem):
        def eval(self, x, yref=None, do_derivative=False, calc_hessian=False):
            result = super().eval(x, yref, do_derivative, calc_hessian)
            target_call = 1 if evaluation == 'initial' else 2
            if do_derivative and len(self.accepted) == target_call:
                result = list(result)
                index = 0 if component == 'objective' else 1
                result[index] = torch.full_like(result[index], bad)
                return tuple(result)
            return result
    optimizer = GaussNewton(NonfiniteProblem(), path=str(tmp_path))
    with pytest.raises(FloatingPointError, match=f'{evaluation} objective and gradient must be finite'):
        optimizer.run_correction(torch.zeros(1, dtype=torch.float64))
    assert not optimizer.converged


def test_small_backtracked_steps_are_not_declared_convergence(tmp_path):
    problem = ScalarProblem()
    optimizer = GaussNewton(problem, max_iter=4, tolG=0, path=str(tmp_path))
    class ShortSteps:
        t = 1.0
        rejected_nonfinite = True
        def eval(self, obj, current, direction, value, gradient, yref=None):
            # Models a constraint-limited line search, without conflating the
            # accepted tiny step with an already-small Newton search direction.
            return 1e-4, current + 1e-4*direction, 14, True
    optimizer.line_search = ShortSteps()
    optimizer.run_correction(torch.zeros(1, dtype=torch.float64))
    assert len(problem.accepted) == 5  # initial point plus all four iterations
    assert 'change tolerance' not in (tmp_path/'log_file.txt').read_text()
    assert not optimizer.converged
    assert optimizer.convergence_reason == 'maximum_iterations'


def test_no_barrier_contact_preserves_historical_change_stopping(tmp_path):
    problem = ScalarProblem()
    optimizer = GaussNewton(problem, max_iter=4, tolG=0, path=str(tmp_path))
    class SmallUnconstrainedSteps:
        def eval(self, obj, current, direction, value, gradient, yref=None):
            return 1.0, current + 1e-4*direction, 0, True
    optimizer.line_search = SmallUnconstrainedSteps()
    optimizer.run_correction(torch.zeros(1, dtype=torch.float64))
    assert len(problem.accepted) == 3
    assert optimizer.converged and not optimizer.barrier_limited
    assert optimizer.linear_solver.max_iter == 10


@pytest.mark.parametrize('bad', [1.0, -1.2, float('nan')])
def test_invalid_initialization_is_rejected_before_optimization(tmp_path, bad):
    problem = ScalarProblem()
    optimizer = GaussNewton(problem, path=str(tmp_path))
    with pytest.raises(ValueError, match='initialization'):
        optimizer.run_correction(torch.tensor([bad], dtype=torch.float64))
    assert not problem.accepted and not problem.trials


def test_invalid_field_is_not_written(tmp_path):
    loss = SimpleNamespace(D=SimpleNamespace(mat_mul=lambda x: x))
    optimizer = EPIOptimize(loss, path=str(tmp_path/'result'))
    optimizer.B0 = torch.zeros(1)
    optimizer.Bc = torch.tensor([1.1])
    with pytest.raises(ValueError, match='Refusing to save'):
        optimizer.apply_correction()
    assert not list(tmp_path.glob('*.nii*'))


def test_corrected_pair_uses_accepted_field_not_last_trial_cache(tmp_path):
    data = np.arange(120, dtype=np.float32).reshape(6, 5, 4) + 10
    image = tmp_path/'input.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), image)
    images = DataObject(str(image), str(image), 2, device='cpu', dtype=torch.float64)
    loss = EPIMRIDistortionCorrection(images, alpha=300, beta=1e-4)
    optimizer = EPIOptimize(loss, path=str(tmp_path/'result'))
    optimizer.Bc = torch.zeros(tuple(m_plus(images.m)), dtype=torch.float64)
    optimizer.B0 = optimizer.Bc.clone()
    loss.corr1 = loss.corr2 = torch.full(tuple(images.m), -999., dtype=torch.float64)
    actual_ap, actual_pa = optimizer.apply_correction('jac')
    expected = images.I1.data.permute(images.p)
    torch.testing.assert_close(actual_ap, expected)
    torch.testing.assert_close(actual_pa, expected)
    assert (nib.load(tmp_path/'result-im2Corrected.nii.gz').get_fdata() >= 0).all()
