"""Native unsmoothed nuisance signals can accompany final-grid responses."""
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from test_regression_coverage import RtpRegress


def _save_mask(path, data):
    nib.save(nib.Nifti1Image(np.asarray(data, np.uint8), np.eye(4)), path)
    return path


class _ForbiddenSource:
    @property
    def proc_data(self):
        raise AssertionError("Supplied nuisance means must bypass the source image")


def _apply(reg, frame, index, support, means=None, supplied=False):
    image = nib.Nifti1Image(np.asarray(frame, np.float32).copy(), np.eye(4))
    image.set_filename("current.nii")
    kwargs = {"nuisance_means": means} if supplied else {}
    reg.do_proc(image, vol_idx=index - 1, support_mask=support, **kwargs)
    return np.asarray(image.dataobj).copy()


def test_supplied_native_means_reproduce_legacy_masks_support_and_startup(tmp_path):
    rng = np.random.default_rng(731)
    shape = (3, 4, 5)
    brain = np.ones(shape, bool)
    brain[0, 0, 0] = False
    masks = {"GS": np.ones(shape, bool), "WM": np.zeros(shape, bool),
             "Vent": np.zeros(shape, bool)}
    masks["WM"][..., 0] = True
    masks["Vent"][..., 1] = True
    mask_paths = {name: _save_mask(tmp_path / f"{name}.nii", mask) for name, mask in masks.items()}
    kwargs = dict(mask_file=_save_mask(tmp_path / "brain.nii", brain),
        GS_reg=True, GS_mask=mask_paths["GS"], WM_reg=True, WM_mask=mask_paths["WM"],
        Vent_reg=True, Vent_mask=mask_paths["Vent"],
        max_scan_length=30, max_poly_order=1, TR=1., wait_num=7,
        onGPU=False, reg_retro_proc=False, scale_before_glm=False,
        startup_ref_volumes=8)
    legacy = RtpRegress(**kwargs, mask_src_proc=SimpleNamespace(proc_data=None))
    supplied = RtpRegress(**kwargs, mask_src_proc=_ForbiddenSource())
    assert legacy.ready_proc() and supplied.ready_proc()
    history = (1000 + rng.normal(size=(16, *shape)) * 50).astype(np.float32)
    history[4] = 0  # An acquired zero tissue mean is a valid nuisance value.
    saw_ready = False
    for index, raw in enumerate(history, 1):
        support = np.ones(shape, bool)
        if index == 3:
            support[masks["WM"]] = False
        if index == 12:
            support[masks["Vent"]] = False
        frame = np.where(support, raw, 0)
        legacy.mask_src_proc.proc_data = frame
        means = {}
        for name, mask in masks.items():
            values = frame[mask & brain & support]
            means[name] = float(values.mean()) if values.size else np.nan
        actual_legacy = _apply(legacy, frame, index, support)
        actual_supplied = _apply(supplied, frame, index, support, means, supplied=True)
        np.testing.assert_array_equal(actual_supplied, actual_legacy)
        np.testing.assert_array_equal(supplied.desMtx.cpu(), legacy.desMtx.cpu())
        np.testing.assert_array_equal(supplied._valid_design_rows, legacy._valid_design_rows)
        np.testing.assert_array_equal(supplied.output_support, legacy.output_support)
        assert supplied.last_proc_ready == legacy.last_proc_ready
        saw_ready |= supplied.last_proc_ready
        if index in (3, 12):
            assert not supplied._valid_design_rows[index - 1]
        if index == 5:
            assert supplied._valid_design_rows[index - 1]
            assert all(supplied.desMtx[index - 1, getattr(supplied, name + "_col")] == 0
                       for name in masks)
        if index == 4:
            # A tissue mean recovers with CURRENT coverage even though those
            # response voxels remain excluded by cumulative GLM support.
            assert supplied._valid_design_rows[index - 1]
            assert not supplied.output_support[masks["WM"]].any()
            assert supplied.desMtx[index - 1, supplied.WM_col] == means["WM"]
    assert saw_ready
    assert legacy.startup_residual_moments is not None
    for name in ("mean", "m2", "count"):
        np.testing.assert_array_equal(supplied.startup_residual_moments[name],
                                      legacy.startup_residual_moments[name])
    assert supplied.startup_residual_moments["reference_count"] == 8
    assert supplied.GS_maskdata is supplied.WM_maskdata is supplied.Vent_maskdata is None


def test_native_nuisance_means_fit_different_response_grid_against_numpy(tmp_path):
    rng = np.random.default_rng(105)
    native_shape, response_shape = (3, 4, 5), (2, 3, 4)
    masks = {"GS": np.ones(native_shape, bool), "WM": np.zeros(native_shape, bool),
             "Vent": np.zeros(native_shape, bool)}
    masks["WM"][0] = True
    masks["Vent"][-1] = True
    paths = {name: _save_mask(tmp_path / f"native_{name}.nii", mask) for name, mask in masks.items()}
    brain = np.ones(response_shape, bool)
    reg = RtpRegress(mask_file=_save_mask(tmp_path / "final_brain.nii", brain),
        GS_reg=True, GS_mask=paths["GS"], WM_reg=True, WM_mask=paths["WM"],
        Vent_reg=True, Vent_mask=paths["Vent"], mask_src_proc=_ForbiddenSource(),
        max_scan_length=30, max_poly_order=1, TR=1., wait_num=11,
        onGPU=False, reg_retro_proc=False, scale_before_glm=False)
    assert reg.ready_proc()
    native = (1000 + rng.normal(size=(16, *native_shape)) * 100).astype(np.float32)
    means = np.asarray([[frame[masks[name]].mean() for name in ("GS", "WM", "Vent")]
                        for frame in native], np.float32)
    coefficients = rng.normal(size=(3, int(brain.sum())))
    response = (means @ coefficients + rng.normal(size=(16, int(brain.sum()))) * 40).astype(np.float32)
    complete = brain.copy()
    for index, values in enumerate(response, 1):
        support = brain.copy()
        if index == 4:
            support[0, 0, 0] = False
        complete &= support
        frame = values.reshape(response_shape).copy()
        frame[~support] = 0
        output = _apply(reg, frame, index, support,
                        dict(zip(("GS", "WM", "Vent"), means[index - 1])), supplied=True)
        if index < 12:
            assert not reg.last_proc_ready
            continue
        assert reg.last_proc_ready
        design = np.column_stack((means[:index], np.ones(index), np.linspace(-1., 1., index)))
        target = response[:index, complete.ravel()].astype(np.float64)
        residual = target[-1] - design[-1] @ np.linalg.lstsq(design, target, rcond=None)[0]
        np.testing.assert_allclose(output[complete], residual, rtol=2e-4, atol=.03)
        np.testing.assert_array_equal(output[~complete], 0)
        for column, name in enumerate(("GS", "WM", "Vent")):
            assert reg.desMtx[index - 1, getattr(reg, name + "_col")] == means[index - 1, column]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, None, [1.], "2", True, "missing"])
def test_invalid_enabled_mean_invalidates_row_without_affecting_disabled_columns(tmp_path, invalid):
    brain = np.ones((2, 2, 2), bool)
    reg = RtpRegress(mask_file=_save_mask(tmp_path / "brain.nii", brain),
        WM_reg=True, WM_mask=None, mask_src_proc=_ForbiddenSource(),
        max_scan_length=20, max_poly_order=1, TR=1., wait_num=5,
        onGPU=False, reg_retro_proc=False, scale_before_glm=False)
    assert reg.ready_proc()
    assert "GS" not in reg.reg_names and "Vent" not in reg.reg_names
    for index in range(1, 9):
        nuisance = {"WM": float(index * index), "GS": np.nan, "Vent": np.nan}
        if index == 7:
            if isinstance(invalid, str) and invalid == "missing":
                nuisance.pop("WM")
            else:
                nuisance["WM"] = invalid
        output = _apply(reg, np.full(brain.shape, 1000 + index, np.float32), index,
                        brain, nuisance, supplied=True)
        if index == 7:
            assert not reg._valid_design_rows[index - 1]
            assert not reg.last_proc_ready
            assert not reg.output_support.any()
            np.testing.assert_array_equal(output, 0)
        elif index >= 6:
            assert reg._valid_design_rows[index - 1]
            assert reg.last_proc_ready
