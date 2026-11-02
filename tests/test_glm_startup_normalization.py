"""Initialize voxel z-scoring from the existing first cGLM fit, without replay."""
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
from test_regression_coverage import RtpRegress


@pytest.mark.parametrize("reference", ["fixed", "running"])
def test_first_fit_exports_supported_residual_moments_once(tmp_path, reference):
    rng = np.random.default_rng(416)
    shape = (2, 2, 3)
    brain = np.ones(shape, bool)
    brain[1, 1, 1] = False
    wm = np.zeros(shape, bool)
    wm[..., 2] = True
    brain_path, wm_path = tmp_path / "brain.nii", tmp_path / "wm.nii"
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), np.eye(4)), brain_path)
    nib.save(nib.Nifti1Image(wm.astype(np.uint8), np.eye(4)), wm_path)
    motion = rng.normal(size=(30, 6)).astype(np.float32)
    motion[:, 1] = 2 * motion[:, 0]  # A rank-deficient nuisance design is valid.
    values = (1000. + rng.normal(size=(15, *shape)) * 40).astype(np.float32)
    reg = RtpRegress(mask_file=brain_path, WM_reg=True, WM_mask=wm_path,
        mask_src_proc=SimpleNamespace(proc_data=None),
        volreg=SimpleNamespace(_motion=motion), mot_reg="mot6",
        TR=1., max_poly_order=1, max_scan_length=30, wait_num=11,
        onGPU=False, reg_retro_proc=False, scale_before_glm=False,
        startup_ref_volumes=6, startup_ref_reference=reference)
    assert reg.ready_proc()
    supports, first_fit = [], None
    for index, frame in enumerate(values, 1):
        support = np.ones(shape, bool)
        if index == 2:
            support[0, 0, 0] = False  # Permanently unavailable GLM voxel.
        if index == 3:
            support[wm] = False  # Empty WM makes this entire design row invalid.
        supports.append(support)
        data = np.where(support, frame, 0).astype(np.float32)
        reg.mask_src_proc.proc_data = data
        image = nib.Nifti1Image(data, np.eye(4))
        image.set_filename("current.nii")
        reg.do_proc(image, vol_idx=index - 1, support_mask=support)
        if not reg.last_proc_ready:
            assert reg.startup_residual_moments is None
            continue
        if first_fit is not None:
            assert reg.startup_residual_moments is None  # Consuming it cannot regenerate it.
            continue
        first_fit = index
        assert first_fit == 13  # Twelve valid rows, plus one invalid WM row.
        payload = reg.startup_residual_moments
        assert payload is not None
        valid_rows = np.ones(index, bool)
        valid_rows[2] = False
        wm_mean = values[:index, wm].mean(axis=1)
        design = np.column_stack((motion[:index], wm_mean, np.ones(index), np.linspace(-1., 1., index)))
        assert np.linalg.matrix_rank(design[valid_rows]) < design.shape[1]
        complete = brain & np.logical_and.reduce(supports)
        target = values[:index, complete].astype(np.float64)
        beta = np.linalg.lstsq(design[valid_rows], target[valid_rows], rcond=None)[0]
        residuals = target - design @ beta
        rows = np.flatnonzero(valid_rows)
        if reference == "fixed":
            rows = rows[:6]
        residuals = residuals[rows]
        expected_mean = residuals.mean(axis=0)
        expected_m2 = np.sum((residuals - expected_mean) ** 2, axis=0)
        assert payload["reference_count"] == len(rows)
        np.testing.assert_array_equal(payload["count"], np.where(complete, len(rows), 0))
        np.testing.assert_allclose(payload["mean"][complete], expected_mean, atol=.004, rtol=1e-4)
        np.testing.assert_allclose(payload["m2"][complete], expected_m2, atol=.15, rtol=.002)
        np.testing.assert_array_equal(payload["mean"][~complete], 0)
        np.testing.assert_array_equal(payload["m2"][~complete], 0)
        normalizer = VoxelNormalizer("zscore", reference, 6, brain)
        normalizer.initialize_from_moments(**payload)
        assert normalizer.ready
        assert not normalizer.output_support.any()
        np.testing.assert_allclose(normalizer.std[complete], residuals.std(axis=0, ddof=1),
                                   atol=.004, rtol=.001)
        reg.startup_residual_moments = None
    assert first_fit is not None


@pytest.mark.parametrize("reference", ["fixed", "running"])
def test_seeded_moments_match_observation_math_and_preserve_prior_sample_semantics(reference):
    shape = (1, 2, 2)
    brain = np.array([[[True, True], [True, False]]])
    history = np.array([[[[2., 4.], [0., 0.]]],
                        [[[4., 8.], [0., 0.]]],
                        [[[8., 0.], [12., 0.]]]])
    supports = np.array([[[[True, True], [False, False]]],
                         [[[True, True], [False, False]]],
                         [[[True, False], [True, False]]]])
    count = supports.sum(axis=0)
    mean = np.divide(np.where(supports, history, 0).sum(axis=0), count,
                     out=np.zeros(shape), where=count > 0)
    m2 = np.where(supports, (history - mean) ** 2, 0).sum(axis=0)
    normalizer = VoxelNormalizer("zscore", reference, 3, brain)
    normalizer.initialize_from_moments(mean, m2, count, 3)
    # Inputs must not alias mutable reference state.
    mean_copy, m2_copy, count_copy = mean.copy(), m2.copy(), count.copy()
    mean.fill(99)
    m2.fill(99)
    count.fill(99)
    np.testing.assert_array_equal(normalizer.mean, mean_copy)
    np.testing.assert_array_equal(normalizer.count, count_copy)
    current = np.array([[[10., 10.], [99., 100.]]])
    output = normalizer.transform(current, brain)
    eligible = count_copy >= 2
    expected = np.zeros(shape)
    expected[eligible] = ((current[eligible] - mean_copy[eligible]) /
                          np.sqrt(m2_copy[eligible] / (count_copy[eligible] - 1)))
    np.testing.assert_allclose(output, expected, atol=1e-6)
    assert normalizer.reference_count == 3  # Transform never observes current data.
    np.testing.assert_array_equal(normalizer.output_support, eligible)
    normalizer.observe(current, brain)
    if reference == "fixed":
        assert normalizer.reference_count == 3
        np.testing.assert_array_equal(normalizer.mean, mean_copy)
    else:
        assert normalizer.reference_count == 4
        np.testing.assert_allclose(normalizer.mean[brain],
                                   (mean_copy * count_copy + current)[brain] / (count_copy[brain] + 1))
        np.testing.assert_array_equal(normalizer.count[brain], count_copy[brain] + 1)
    with pytest.raises(ValueError, match="already contains"):
        normalizer.initialize_from_moments(mean_copy, m2_copy, count_copy, 3)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "negative_m2", "fractional_count",
    "count_exceeds_rows", "count_zero_with_mean", "single_with_m2", "empty", "fractional_rows",
    "boolean_rows", "fixed_overflow", "outside_mask"])
def test_invalid_moment_import_does_not_mutate_normalizer(invalid):
    shape = (1, 1, 2)
    brain = np.array([[[True, False]]])
    normalizer = VoxelNormalizer("zscore", "fixed", 3, brain)
    mean = np.array([[[2., 0.]]])
    m2 = np.array([[[2., 0.]]])
    count = np.array([[[3., 0.]]])
    rows = 3
    if invalid == "shape":
        m2 = m2.ravel()
    elif invalid == "nonfinite":
        mean[0, 0, 0] = np.nan
    elif invalid == "negative_m2":
        m2[0, 0, 0] = -1
    elif invalid == "fractional_count":
        count[0, 0, 0] = 1.5
    elif invalid == "count_exceeds_rows":
        count[0, 0, 0] = 4
    elif invalid == "count_zero_with_mean":
        mean[0, 0, 1] = 1
    elif invalid == "single_with_m2":
        count[0, 0, 0] = 1
    elif invalid == "empty":
        mean.fill(0)
        m2.fill(0)
        count.fill(0)
    elif invalid == "fractional_rows":
        rows = 2.5
    elif invalid == "boolean_rows":
        rows = True
    elif invalid == "fixed_overflow":
        rows = 4
    elif invalid == "outside_mask":
        count[0, 0, 1] = 2
    with pytest.raises(ValueError):
        normalizer.initialize_from_moments(mean, m2, count, rows)
    assert normalizer.reference_count == 0
    assert not normalizer.count.any()
    assert not normalizer.mean.any()
    assert not normalizer.output_support.any()


def test_startup_export_is_disabled_by_default_and_validates_configuration():
    reg = RtpRegress(max_scan_length=20, onGPU=False)
    assert reg.startup_ref_volumes is None
    assert reg.startup_residual_moments is None
    for invalid in (True, 1, 2.5, np.nan):
        with pytest.raises(ValueError, match="integer count"):
            RtpRegress(max_scan_length=20, onGPU=False, startup_ref_volumes=invalid)
    with pytest.raises(ValueError, match="fixed or running"):
        RtpRegress(max_scan_length=20, onGPU=False, startup_ref_reference="unknown")


def test_capture_waits_for_requested_rows_without_extra_fits_or_replay(tmp_path, monkeypatch):
    mask = np.ones((2, 2, 2), bool)
    mask_path = tmp_path / "brain.nii"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), mask_path)
    reg = RtpRegress(mask_file=mask_path, max_scan_length=20, max_poly_order=1,
        TR=1., wait_num=3, onGPU=False, reg_retro_proc=False,
        scale_before_glm=False, startup_ref_volumes=6, save_proc=False)
    assert reg.ready_proc()
    calls, emitted = [], []
    solver = reg.do_proc.__globals__["lstsq_SVDsolver"]

    def count_solve(*args, **kwargs):
        calls.append(1)
        return solver(*args, **kwargs)

    monkeypatch.setitem(reg.do_proc.__globals__, "lstsq_SVDsolver", count_solve)
    reg.next_proc = SimpleNamespace(do_proc=lambda image, **kwargs: emitted.append(kwargs["vol_idx"]))
    values = np.array([1000., 1002., 998., 1006., 997., 1001., 1008.])
    for index, value in enumerate(values, 1):
        image = nib.Nifti1Image(np.full(mask.shape, value, np.float32), np.eye(4))
        image.set_filename("current.nii")
        reg.do_proc(image, vol_idx=index - 1, support_mask=mask)
        assert reg.last_proc_ready == (index >= 4)
        if index < 6:
            assert reg.startup_residual_moments is None
        elif index == 6:
            assert reg.startup_residual_moments["reference_count"] == 6
            reg.startup_residual_moments = None
        else:
            assert reg.startup_residual_moments is None
    assert len(calls) == 4  # One solve for each ordinary ready volume.
    assert emitted == [3, 4, 5, 6]  # No retrospective downstream emissions.
    # The test's lightweight RTP base includes only __init__; supply its no-op
    # reset so the real RtpRegress reset can be exercised independently.
    monkeypatch.setattr(RtpRegress.__mro__[1], "end_reset", lambda self: True, raising=False)
    assert reg.end_reset()
    assert reg.startup_residual_moments is None
    assert not reg._startup_reference_generated
