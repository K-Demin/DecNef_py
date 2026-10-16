import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from fmri_rt_preproc.volume_smooth import MaskedGaussianSmoother, prepare_smoother, smooth_file
from rt_global_settings import RegressorSettings, load_regressor_settings, save_regressor_settings


def test_matches_supplied_normalized_convolution():
    rng = np.random.default_rng(13)
    mask = rng.random((11, 12, 13)) > .4
    sizes = np.array([2., 2.5, 3.])
    smoother = MaskedGaussianSmoother(mask, np.diag([*sizes, 1]), 6.)
    sigma = 6. / (2 * np.sqrt(2 * np.log(2)) * sizes)
    density = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0., truncate=4.)[mask]
    for _ in range(3):
        data = rng.normal(size=mask.shape)
        buf = np.zeros(mask.shape)
        buf[mask] = data[mask]
        expected = gaussian_filter(buf, sigma, mode="constant", cval=0., truncate=4.)[mask] / density
        actual = smoother.apply(data)
        np.testing.assert_allclose(actual[mask], expected, rtol=1e-14, atol=1e-14)
        assert np.all(actual[~mask] == 0)
    constant = np.full(mask.shape, 7.)
    constant[~mask] = np.nan  # excluded voxels cannot contaminate the result
    np.testing.assert_allclose(smoother.apply(constant)[mask], 7.)


def test_dynamic_support_keeps_constant_signal_at_missing_boundary():
    mask = np.ones((11, 12, 13), dtype=bool)
    mask[0] = False
    support = np.ones_like(mask)
    support[:, :, 8:] = False
    support[4:7, 4:8, 3:6] = False
    active = mask & support
    data = np.full(mask.shape, 7.)
    data[~active] = np.nan
    smoother = MaskedGaussianSmoother(mask, np.eye(4), 4.)

    result = smoother.apply(data, support_mask=support)

    np.testing.assert_allclose(result[active], 7., rtol=1e-14, atol=1e-14)
    assert np.all(result[~active] == 0.)


def test_dynamic_support_counts_acquired_zero_and_negative_residuals():
    mask = np.ones((7, 8, 9), dtype=bool)
    support = np.ones_like(mask)
    support[:, :, -2:] = False
    data = np.full(mask.shape, -3.)
    data[2:5, 2:6, 2:5] = 0.  # Acquired zeros still contribute kernel density.
    data[~support] = 1000.  # Missing values must not contribute, whatever their value.
    smoother = MaskedGaussianSmoother(mask, np.eye(4), 3.)
    numerator = gaussian_filter(np.where(support, data, 0.), smoother.sigma,
                                mode="constant", cval=0., truncate=4.)
    density = gaussian_filter(support.astype(float), smoother.sigma,
                              mode="constant", cval=0., truncate=4.)

    result = smoother.apply(data, support)

    np.testing.assert_allclose(result[support], numerator[support] / density[support],
                               rtol=1e-14, atol=1e-14)
    assert -3. < result[3, 4, 3] < 0.
    assert np.all(result[~support] == 0.)


@pytest.mark.parametrize("fwhm", [0., 4.])
def test_changing_support_and_return_to_static_mask_do_not_leak_frames(fwhm):
    mask = np.ones((7, 8, 9), dtype=bool)
    mask[0] = False
    first = np.ones_like(mask)
    first[:, :, 5:] = False
    second = np.ones_like(mask)
    second[:, :, :4] = False
    smoother = MaskedGaussianSmoother(mask, np.eye(4), fwhm)

    for value, support in ((19., first), (-3., second),
                           (0., np.zeros_like(mask)), (7., None)):
        active = mask if support is None else mask & support
        data = np.full(mask.shape, value)
        data[~active] = np.nan
        result = smoother.apply(data, support)
        np.testing.assert_allclose(result[active], value, rtol=1e-14, atol=1e-14)
        assert np.all(result[~active] == 0.)

    # Switching back to the cached static denominator preserves the old result.
    data = np.random.default_rng(25).normal(size=mask.shape)
    expected = MaskedGaussianSmoother(mask, np.eye(4), fwhm).apply(data)
    np.testing.assert_array_equal(smoother.apply(data), expected)


def test_mask_only_support_preserves_acquired_values_exactly():
    mask = np.ones((3, 4, 5), dtype=bool)
    support = np.ones_like(mask)
    support[:, :, -1] = False
    data = np.arange(mask.size, dtype=float).reshape(mask.shape) - 15.
    data[~support] = np.nan
    smoother = MaskedGaussianSmoother(mask, np.eye(4), 0.)

    result = smoother.apply(data, support)

    np.testing.assert_array_equal(result[support], data[support])
    assert np.all(result[~support] == 0.)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, .5, -1., 2.])
def test_dynamic_support_rejects_nonfinite_or_nonbinary_masks(invalid):
    mask = np.ones((3, 4, 5), dtype=bool)
    support = mask.astype(float)
    support[0, 0, 0] = invalid
    smoother = MaskedGaussianSmoother(mask, np.eye(4), 4.)
    with pytest.raises(ValueError, match="finite and binary"):
        smoother.apply(np.zeros(mask.shape), support)


def test_nonfinite_acquired_value_is_still_rejected():
    mask = np.ones((3, 4, 5), dtype=bool)
    data = np.ones(mask.shape)
    data[1, 1, 1] = np.nan
    smoother = MaskedGaussianSmoother(mask, np.eye(4), 4.)
    with pytest.raises(ValueError, match="Nonfinite values"):
        smoother.apply(data, mask)


def test_smooth_file_uses_support_and_rejects_its_wrong_shape(tmp_path):
    mask = np.ones((5, 6, 7), dtype=bool)
    support = np.ones_like(mask)
    support[:, :, -2:] = False
    data = np.full(mask.shape, 5.)
    data[~support] = np.nan
    affine = np.diag([2., 2., 2., 1.])
    source, destination = tmp_path / "source.nii", tmp_path / "out.nii"
    nib.save(nib.Nifti1Image(data, affine), source)
    smoother = MaskedGaussianSmoother(mask, affine, 4.)

    smooth_file(smoother, source, destination, support_mask=support.astype(np.uint8))

    saved = nib.load(destination)
    np.testing.assert_array_equal(saved.affine, affine)
    np.testing.assert_allclose(saved.get_fdata()[support], 5.)
    assert np.all(saved.get_fdata()[~support] == 0.)
    assert not (tmp_path / ".out.nii").exists()
    with pytest.raises(ValueError, match="support and mask shapes differ"):
        smooth_file(smoother, source, tmp_path / "bad.nii", np.ones((5, 6, 6), bool))
    assert not (tmp_path / "bad.nii").exists()


def test_disabled_is_a_complete_bypass():
    assert prepare_smoother(None, 0, None, None, None) is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_settings_rejected(value):
    with pytest.raises(ValueError):
        RegressorSettings().update({"smoothing_fwhm_mm": value})


def test_settings_json_roundtrip(tmp_path):
    settings = RegressorSettings()
    assert settings.smoothing_fwhm_mm == 0
    settings.update({"smoothing_fwhm_mm": 5.})
    path = tmp_path / "settings.json"
    save_regressor_settings(path, settings)
    assert load_regressor_settings(path).smoothing_fwhm_mm == 5.


@pytest.mark.parametrize("space", ["epi", "t1", "mni"])
def test_mask_preparation_uses_final_grid_and_transform_chain(tmp_path, space):
    source = tmp_path / "brain.nii"
    reference = tmp_path / "reference.nii"
    mask = np.ones((4, 5, 6), np.uint8)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), source)
    final_affine = np.diag([2., 2., 2., 1.]) if space != "epi" else np.eye(4)
    nib.save(nib.Nifti1Image(mask, final_affine), reference)
    (tmp_path / "anat").mkdir()
    epi_transform = tmp_path / "epi2t1_Composite.h5"
    mni_transform = tmp_path / "anat" / "warp_T1_to_MNI_synth.nii"
    epi_transform.touch(); mni_transform.touch()
    cfg = SimpleNamespace(rt_unwarped_analysis_ref_mask=source,
        rt_work_dir=tmp_path, trans_dir=tmp_path, subject_root=tmp_path)
    calls = []

    def run(cmd):
        calls.append(cmd)
        assert cmd[cmd.index("-n") + 1] == "NearestNeighbor"
        assert cmd[cmd.index("-i") + 1] == str(source)
        assert cmd[cmd.index("-r") + 1] == str(reference)
        transforms = [cmd[i+1] for i, v in enumerate(cmd) if v == "-t"]
        assert transforms == ([str(mni_transform)] if space == "mni" else []) + [str(epi_transform)]
        nib.save(nib.Nifti1Image(mask, final_affine), cmd[cmd.index("-o") + 1])

    smoother = prepare_smoother(cfg, 4., space, reference, run)
    assert len(calls) == (0 if space == "epi" else 1)
    np.testing.assert_array_equal(smoother.affine, final_affine)
    for idx in range(2):
        output = tmp_path / "smooth" / f"vol_{idx:05d}_smooth.nii"
        smooth_file(smoother, reference, output)
        np.testing.assert_allclose(nib.load(output).get_fdata(), 1.)
    assert len(calls) == (0 if space == "epi" else 1)
    assert not list((tmp_path / "smooth").glob(".vol_*"))


def test_grid_mismatch_and_empty_mask_fail(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        MaskedGaussianSmoother(np.zeros((3, 4, 5)), np.eye(4), 4)
    smoother = MaskedGaussianSmoother(np.ones((3, 4, 5)), np.eye(4), 4)
    source = tmp_path / "wrong.nii"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5)), np.diag([2., 2., 2., 1.])), source)
    with pytest.raises(ValueError, match="grids differ"):
        smooth_file(smoother, source, tmp_path / "out.nii")


@pytest.mark.parametrize("space,kind", [("epi", "reg"), ("t1", "t1"), ("mni", "mni")])
def test_pca_waits_for_final_smoothing(tmp_path, space, kind):
    from rs_pca_runtime import volume_path_for_kind

    metadata = {"smoothing_fwhm_mm": 4, "regression": {"analysis_space": space}}
    (tmp_path / "session_metadata.json").write_text(json.dumps(metadata))
    assert volume_path_for_kind(tmp_path, 1, kind) == tmp_path / "smooth" / "vol_00001_smooth.nii"
    assert volume_path_for_kind(tmp_path, 1, "mc") == tmp_path / "mc" / "vol_00001_mc.nii"
    metadata["smoothing_fwhm_mm"] = 0
    (tmp_path / "session_metadata.json").write_text(json.dumps(metadata))
    assert volume_path_for_kind(tmp_path, 1, kind).parent.name == kind
    with pytest.raises(ValueError, match="disabled"):
        volume_path_for_kind(tmp_path, 1, "smooth")


@pytest.mark.parametrize("kind", ["smooth", "masked"])
def test_pca_preparation_excludes_mask_and_original_stream(tmp_path, monkeypatch, kind):
    import roi_rs_pca_decoder_prep as prep

    folder = tmp_path / kind
    folder.mkdir()
    expected = []
    for name in (f"vol_00001_{kind}.nii", f"vol_00002_{kind}.nii", "mask.nii", f"vol_00001_{kind}_orig.nii"):
        path = folder / name
        path.touch()
        if name.endswith(f"_{kind}.nii"):
            expected.append(path)
    (tmp_path / "session_metadata.json").write_text(json.dumps(
        {"smoothing_fwhm_mm": 4 if kind == "smooth" else 0, "analysis_mask": "cortical_gm",
         "regression": {"analysis_space": "t1"}}))
    calls = []

    def merge(volumes, output):
        calls.append(volumes)
        output.touch()
        return True

    monkeypatch.setattr(prep, "_run_fslmerge", merge)
    pca, tsnr = prep._discover_rs_inputs(tmp_path, "auto")
    assert pca == tsnr
    assert pca.name.endswith(f"_{kind}.nii.gz")
    assert calls == [expected]
