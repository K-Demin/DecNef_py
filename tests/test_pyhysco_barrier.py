"""The actual PyHySCO barrier must keep both PE Jacobians positive."""
import nibabel as nib
import numpy as np
import pytest
import torch

from fmri_rt_preproc.pyhysco_apply import DataObject, EPIMRIDistortionCorrection


@pytest.fixture
def loss():
    # phi_EPI has no image dependencies; exercise the real class without a fit.
    return EPIMRIDistortionCorrection.__new__(EPIMRIDistortionCorrection)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_valid_barrier_preserves_formula_symmetry_and_input(loss, dtype):
    x = torch.tensor([[-0.9, -0.4, 0.0], [0.9, 0.4, 0.2]], dtype=dtype).T
    before = x.clone()
    value, first, second = loss.phi_EPI(x, do_derivative=True, calc_hessian=True)
    expected = x.pow(4) / (1 - x.square())
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert value.shape == first.shape == second.shape == x.shape
    assert value.dtype == first.dtype == second.dtype == dtype
    assert torch.isfinite(value).all() and torch.isfinite(first).all() and torch.isfinite(second).all()
    assert (value >= 0).all() and (second >= 0).all()
    opposite = loss.phi_EPI(-x, do_derivative=True, calc_hessian=True)
    torch.testing.assert_close(value, opposite[0])
    torch.testing.assert_close(first, -opposite[1])
    torch.testing.assert_close(second, opposite[2])


@pytest.mark.parametrize("dtype,step,tolerance", [
    (torch.float32, 1e-3, 8e-4),
    (torch.float64, 1e-5, 1e-7),
])
def test_interior_derivatives_agree_with_finite_differences(loss, dtype, step, tolerance):
    x = torch.tensor([-0.85, -0.4, 0.0, 0.35, 0.8], dtype=dtype)
    _, first, second = loss.phi_EPI(x, do_derivative=True, calc_hessian=True)
    plus, plus_first, _ = loss.phi_EPI(x + step, do_derivative=True)
    minus, minus_first, _ = loss.phi_EPI(x - step, do_derivative=True)
    torch.testing.assert_close(first, (plus - minus) / (2 * step), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(second, (plus_first - minus_first) / (2 * step), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_invalid_objective_is_infinite_without_mutating_input(loss, dtype):
    x = torch.tensor([float('-inf'), -1.2, -1.0, -0.5, 0.0, 0.5, 1.0, 1.2,
                      float('inf'), float('nan')], dtype=dtype)
    before = x.clone()
    value, first, second = loss.phi_EPI(x)
    invalid = ~torch.isfinite(x) | (x.abs() >= 1)
    assert torch.isposinf(value[invalid]).all()
    assert torch.isfinite(value[~invalid]).all()
    assert not torch.isnan(value).any()
    assert first is None and second is None
    torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)
    # A line search must reject a mixed feasible/infeasible field as a whole.
    assert torch.isposinf(value.sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bad", [-1.2, -1.0, 1.0, 1.2, float('-inf'), float('inf'), float('nan')])
@pytest.mark.parametrize("derivative,hessian", [(True, False), (False, True), (True, True)])
def test_invalid_derivative_state_raises_explicitly(loss, dtype, bad, derivative, hessian):
    x = torch.tensor([0.25, bad], dtype=dtype)
    before = x.clone()
    with pytest.raises(ValueError, match="finite and strictly between -1 and 1"):
        loss.phi_EPI(x, do_derivative=derivative, calc_hessian=hessian)
    torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_representable_interior_near_boundary_keeps_large_finite_penalty(loss, dtype):
    inside = torch.nextafter(torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype))
    x = torch.stack([-inside, inside])
    value, first, second = loss.phi_EPI(x, do_derivative=True, calc_hessian=True)
    assert torch.isfinite(value).all() and torch.isfinite(first).all() and torch.isfinite(second).all()
    assert (value > 1e6).all() and (second > 0).all()
    assert first[0] < 0 < first[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_image_objective_uses_field_derivative_without_jacobian_roundtrip(tmp_path, dtype):
    image_path = tmp_path / "input.nii.gz"
    values = np.arange(60, dtype=np.float32).reshape(4, 5, 3) + 10
    nib.save(nib.Nifti1Image(values, np.eye(4)), image_path)
    data = DataObject(str(image_path), str(image_path), phase_encoding_direction=2,
                      do_normalize=False, dtype=dtype, device="cpu")
    correction = EPIMRIDistortionCorrection(data, alpha=300, beta=1e-4, PC=None)
    shape = [int(n) for n in data.m]
    shape[-1] += 1
    field = torch.zeros(shape, dtype=dtype)
    field[..., 1] = torch.nextafter(torch.tensor(1.0, dtype=dtype),
                                    torch.tensor(0.0, dtype=dtype))
    before = field.clone()
    derivative = correction.D.mat_mul(field)
    assert (derivative.abs() < 1).all()
    # The old route made this valid field appear to hit the barrier exactly.
    assert (((1 + derivative) - 1).abs() >= 1).any()

    objective = correction.eval(field)
    value, gradient, hessian, preconditioner = correction.eval(
        field, do_derivative=True, calc_hessian=True)
    assert torch.isfinite(objective) and torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian(torch.ones_like(field))).all()
    assert preconditioner is None
    torch.testing.assert_close(value, objective)
    torch.testing.assert_close(correction.Pc, correction.phi_EPI(derivative)[0].sum())
    torch.testing.assert_close(field, before, rtol=0, atol=0)
