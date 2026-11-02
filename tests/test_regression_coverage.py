"""Actual RTPSpy/wrapper numerical paths, without loading their optional Qt UI."""
import ast
import logging
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Optional

import nibabel as nib
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _classes():
    def parse(path):
        return ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))

    base = next(n for n in parse("fmri_rt_preproc/RTPSpy_tools/rtp_common.py").body
                if isinstance(n, ast.ClassDef) and n.name == "RTP")
    # Use the original initializer; no GUI/export methods are called here.
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    reg = parse("fmri_rt_preproc/RTPSpy_tools/rtp_regress.py")
    selected = [n for n in reg.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in {"RtpRegress", "lstsq_SVDsolver"}]
    pipeline = parse("rt_pipeline.py")
    wrappers = [n for n in pipeline.body if isinstance(n, ast.ClassDef)
                and n.name in {"MotionRegressor", "RTPStyleVoxelNormalizer"}]
    namespace = dict(np=np, torch=torch, nib=nib, Path=Path, time=time, sys=sys,
                     traceback=traceback, logging=logging, string_types=(str,),
                     gpu_available=False, Optional=Optional, RtpVolreg=object,
                     log=logging.getLogger("regression-coverage-test"))
    exec(compile(ast.Module(body=[base, *selected, *wrappers], type_ignores=[]),
                 "actual_regression_classes", "exec"), namespace)
    return namespace["MotionRegressor"], namespace["RTPStyleVoxelNormalizer"], namespace["RtpRegress"]


MotionRegressor, Normalizer, RtpRegress = _classes()


def _wrapper(tmp_path, mask, *, n=12, motion="mot6", wm=None, scale_before_glm=True,
             normalization_mode=None, normalization_reference="fixed"):
    mask_path = tmp_path / "brain.nii"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), mask_path)
    wm_path = None
    if wm is not None:
        wm_path = tmp_path / "wm.nii"
        nib.save(nib.Nifti1Image(wm.astype(np.uint8), np.eye(4)), wm_path)
    wrapper = MotionRegressor(SimpleNamespace(_motion=np.zeros((40, 6), np.float32)),
                              reg_mask=mask_path, wm_mask=wm_path, mot_reg=motion,
                              max_poly_order=1, TR=1., max_scan_length=40,
                              norm_ref_volumes=n, normalization_mode=normalization_mode,
                              normalization_reference=normalization_reference)
    wrapper._regress.mask_src_proc = SimpleNamespace(proc_data=None)
    # These existing coverage cases deliberately exercise legacy GLM scaling.
    # The runtime wrapper may independently choose native units by default.
    wrapper._regress.scale_before_glm = scale_before_glm
    return wrapper


def _apply(wrapper, value, index, support=None):
    data = np.asarray(value, dtype=np.float32)
    image = nib.Nifti1Image(data, np.eye(4))
    image.set_filename("test_volume.nii")
    wrapper._regress.mask_src_proc.proc_data = data
    result, ready = wrapper.apply(image, index, support_mask=support)
    return result.copy(), ready


def test_warmup_is_brain_masked_and_first_fit_uses_scaled_units(tmp_path):
    mask = np.ones((2, 3, 4), bool)
    mask[0, 0] = False
    wrapper = _wrapper(tmp_path, mask)
    data = np.full(mask.shape, 1000., np.float32)
    data[~mask] = 37.  # Acquired extracranial signal is not missing FOV.
    for index in range(1, 14):
        out, ready = _apply(wrapper, data, index, np.ones_like(mask))
        assert np.all(out[~mask] == 0.)
        if index < 12:
            assert not ready
            np.testing.assert_array_equal(out[mask], 1000.)
        else:
            assert ready
            np.testing.assert_allclose(out[mask], 0., atol=1e-3)
    np.testing.assert_allclose(wrapper._regress.Y_mean.cpu(), 1000.)
    # No motion is a redundant constant regressor, not division by zero.
    assert np.isfinite(out).all()


def test_missing_warmup_samples_do_not_lower_means_or_reenter_glm(tmp_path):
    mask = np.ones((2, 2, 2), bool)
    wrapper = _wrapper(tmp_path, mask, n=4, motion="None")
    for index in range(1, 7):
        support = mask.copy()
        if index == 2:
            support[..., 1] = False
        data = np.full(mask.shape, 1000., np.float32)
        data[~support] = np.nan  # Missing values must never enter the mean.
        out, ready = _apply(wrapper, data, index, support)
        if index >= 2:
            assert not wrapper.output_support[..., 1].any()
            assert np.all(out[..., 1] == 0.)
        if index >= 4:
            assert ready
            np.testing.assert_allclose(out[..., 0], 0., atol=1e-3)
    np.testing.assert_allclose(wrapper._regress.Y_mean.cpu(), 1000.)
    assert wrapper._regress.YMtx.shape[1] == mask.sum()  # Never squeeze decoder feature indexing.


def test_lost_coverage_after_calibration_is_unavailable_for_rest_of_run(tmp_path):
    mask = np.ones((2, 2, 2), bool)
    wrapper = _wrapper(tmp_path, mask, n=4, motion="None")
    for index in range(1, 7):
        support = mask.copy()
        support[..., 1] = index != 5
        out, ready = _apply(wrapper, np.full(mask.shape, 1000.), index, support)
        if index >= 4:
            assert ready
        if index >= 5:
            assert np.all(out[..., 1] == 0.)
            assert not wrapper.output_support[..., 1].any()
            np.testing.assert_allclose(out[..., 0], 0., atol=1e-3)


def test_valid_negative_residual_is_retained(tmp_path):
    mask = np.ones((2, 2, 2), bool)
    wrapper = _wrapper(tmp_path, mask, n=4, motion="None")
    for index in range(1, 5):
        _apply(wrapper, np.full(mask.shape, 1000.), index, mask)
    out, ready = _apply(wrapper, np.broadcast_to([990., 1010.], mask.shape).copy(), 5, mask)
    assert ready
    assert out[0, 0, 0] < 0 < out[0, 0, 1]
    assert wrapper.output_support.all()


@pytest.mark.parametrize("scale_before_glm", [True, False])
def test_invalid_frame_does_not_publish_raw_or_leave_stale_valid_support(tmp_path, scale_before_glm):
    mask = np.ones((2, 2, 2), bool)
    wrapper = _wrapper(tmp_path, mask, n=4, motion="None", scale_before_glm=scale_before_glm)
    for index in range(1, 5):
        _apply(wrapper, np.full(mask.shape, 1000.), index, mask)
    assert wrapper.output_support.all()
    invalid = np.full(mask.shape, 1000.)
    invalid[0, 0, 0] = np.nan
    out, ready = _apply(wrapper, invalid, 5, mask)
    assert not ready
    assert not wrapper.output_support.any()
    np.testing.assert_array_equal(out, 0.)
    out, ready = _apply(wrapper, np.full(mask.shape, 1000.), 6, mask)
    assert ready
    np.testing.assert_allclose(out, 0., atol=1e-3)


def test_tissue_mean_excludes_unavailable_samples_and_empty_rows(tmp_path):
    mask = np.ones((2, 2, 3), bool)
    wm = mask.copy()
    wm[..., 2] = False
    wrapper = _wrapper(tmp_path, mask, n=6, motion="None", wm=wm)
    for index in range(1, 10):
        support = mask.copy()
        if index == 2:
            support[..., 0] = False
        if index == 7:
            support[..., :2] = False  # No tissue signal: exclude this time point.
        data = np.full(mask.shape, 1000., np.float32)
        data[~support] = 0.
        out, ready = _apply(wrapper, data, index, support)
        if index == 2:
            assert wrapper._regress.desMtx[1, wrapper._regress.WM_col] == 1000.
        if index == 7:
            assert not ready
            assert not out.any()
        if index >= 8:
            assert ready
            np.testing.assert_allclose(out, 0., atol=1e-3)
    assert not wrapper._regress._valid_design_rows[6]


def test_no_regression_mean_uses_observed_counts_and_recovers_coverage():
    normalizer = Normalizer(3, np.ones((1, 1, 2), bool))
    normalizer.apply([[[10., np.nan]]], np.array([[[True, False]]]))
    normalizer.apply([[[20., 30.]]], np.ones((1, 1, 2), bool))
    third = normalizer.apply([[[np.nan, 50.]]], np.array([[[False, True]]]))
    np.testing.assert_allclose(normalizer._y_mean, [[[15., 40.]]])
    np.testing.assert_allclose(third, [[[0., 125.]]])
    np.testing.assert_allclose(normalizer.apply([[[15., 40.]]], np.ones((1, 1, 2), bool)), 100.)
    np.testing.assert_allclose(normalizer._y_mean, [[[15., 40.]]])


def test_no_regression_never_calibrated_voxel_stays_unavailable():
    normalizer = Normalizer(2, np.array([[[True, True, False]]]))
    support = np.array([[[True, False, True]]])
    for _ in range(2):
        normalizer.apply([[[10., np.nan, 99.]]], support)
    out = normalizer.apply([[[10., 20., 99.]]], np.ones((1, 1, 3), bool))
    np.testing.assert_array_equal(out, [[[100., 0., 0.]]])
    np.testing.assert_array_equal(normalizer.output_support, [[[True, False, False]]])


def test_no_regression_rejects_nonfinite_acquired_values():
    with pytest.raises(ValueError, match="Nonfinite acquired"):
        Normalizer(2).apply([[[np.nan]]], np.ones((1, 1, 1), bool))


def _native_regressor(tmp_path, mask, *, n, motion=None):
    path = tmp_path / "native_brain.nii"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), path)
    reg = RtpRegress(
        mask_file=path, max_scan_length=40, max_poly_order=1, TR=1.,
        wait_num=n - 1, onGPU=False, reg_retro_proc=False,
        mot_reg="None" if motion is None else "mot6",
        volreg=None if motion is None else SimpleNamespace(_motion=motion),
        scale_before_glm=False,
    )
    assert reg.ready_proc()
    return reg


def _native_apply(reg, values, index, support):
    image = nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4))
    image.set_filename("native_volume.nii")
    reg.do_proc(image, vol_idx=index - 1, support_mask=support)
    return np.asarray(image.dataobj).copy(), reg.last_proc_ready


def test_native_glm_matches_numpy_with_motion_polynomials_and_coverage_gaps(tmp_path):
    rng = np.random.default_rng(109)
    mask = np.ones((2, 2, 2), bool)
    mask[1, 1, 1] = False
    motion = rng.normal(size=(40, 6)).astype(np.float32)
    n_time, n_voxels = 22, int(mask.sum())
    raw = (1000. + motion[:n_time] @ rng.normal(size=(6, n_voxels)) * 40.
           + np.arange(n_time)[:, None] * 7.
           + rng.normal(size=(n_time, n_voxels)) * 700.).astype(np.float32)
    raw[14, 0] = 20000.  # Native fitting must not apply the old >200 clip.
    reg = _native_regressor(tmp_path, mask, n=12, motion=motion)
    full_support = np.ones_like(mask)
    observed_history = np.ones_like(raw, dtype=bool)
    complete = np.ones(n_voxels, bool)
    frozen_mean = None
    for index in range(1, n_time + 1):
        support = full_support.copy()
        observed = np.ones(n_voxels, bool)
        if index == 3:
            observed[1] = False
        if index == 15:
            observed[2] = False
        observed_history[index - 1] = observed
        support[mask] = observed
        complete &= observed
        frame = np.full(mask.shape, 777., np.float32)
        frame[mask] = np.where(observed, raw[index - 1], np.nan)
        out, ready = _native_apply(reg, frame, index, support)
        np.testing.assert_array_equal(out[~mask], 0.)
        np.testing.assert_array_equal(reg.output_support[mask], complete)
        np.testing.assert_array_equal(out[mask][~complete], 0.)
        if index < 12:
            assert not ready
            assert reg.raw_reference_mean is None
            continue
        assert ready
        design = np.column_stack((motion[:index], np.ones(index), np.linspace(-1., 1., index)))
        target = raw[:index, complete].astype(np.float64)
        expected = target[-1] - design[-1] @ np.linalg.lstsq(design, target, rcond=None)[0]
        np.testing.assert_allclose(out[mask][complete], expected, atol=0.012, rtol=1e-5)
        if index == 12:
            counts = observed_history[:12].sum(axis=0)
            expected_mean = np.where(observed_history[:12], raw[:12], 0.).sum(axis=0) / counts
            frozen_mean = reg.raw_reference_mean.cpu().numpy().copy()
            np.testing.assert_allclose(frozen_mean, expected_mean, atol=1e-4)
            np.testing.assert_array_equal(reg.raw_reference_count.cpu(), counts)
            np.testing.assert_array_equal(reg.raw_reference_valid.cpu(), complete)
        else:
            np.testing.assert_array_equal(reg.raw_reference_mean.cpu(), frozen_mean)
    # A later loss changes the fit support, not the frozen calibration record.
    assert reg.raw_reference_valid[2]
    assert not reg.output_support[mask][2]


def test_native_glm_preserves_zero_mean_and_zero_signals_without_clipping(tmp_path):
    mask = np.ones((2, 2, 2), bool)
    reg = _native_regressor(tmp_path, mask, n=4)
    raw = np.asarray([[1000., -600., 0.], [1000., 600., 0.],
                      [1000., 600., 0.], [1000., -600., 0.],
                      [20000., 900., 0.], [1000., -900., 0.]], dtype=np.float32)
    for index, row in enumerate(raw, 1):
        frame = np.full(mask.shape, row[0], np.float32)
        frame.flat[1], frame.flat[2] = row[1:]
        out, ready = _native_apply(reg, frame, index, mask)
        if index < 4:
            assert not ready
            continue
        assert ready
        assert reg.output_support.all()
        design = np.column_stack((np.ones(index), np.linspace(-1., 1., index)))
        expected = raw[index - 1] - design[-1] @ np.linalg.lstsq(design, raw[:index], rcond=None)[0]
        np.testing.assert_allclose(out.flat[:3], expected, atol=0.005)
        np.testing.assert_array_equal(reg.raw_reference_mean.cpu().numpy()[:3], [1000., 0., 0.])
        np.testing.assert_array_equal(reg.raw_reference_valid.cpu().numpy()[:3], True)
    assert reg.Y_mean_mask.all()  # No abs(mean)>epsilon criterion in native GLM.


def test_standalone_regressor_retains_legacy_scaling_default():
    reg = RtpRegress(max_scan_length=40, onGPU=False)
    assert reg.scale_before_glm is True
