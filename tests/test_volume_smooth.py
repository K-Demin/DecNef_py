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


def test_pca_preparation_excludes_mask_and_original_stream(tmp_path, monkeypatch):
    import roi_rs_pca_decoder_prep as prep

    folder = tmp_path / "smooth"
    folder.mkdir()
    expected = []
    for name in ("vol_00001_smooth.nii", "vol_00002_smooth.nii", "mask.nii", "vol_00001_smooth_orig.nii"):
        path = folder / name
        path.touch()
        if name.endswith("_smooth.nii"):
            expected.append(path)
    (tmp_path / "session_metadata.json").write_text(json.dumps(
        {"smoothing_fwhm_mm": 4, "regression": {"analysis_space": "t1"}}))
    calls = []

    def merge(volumes, output):
        calls.append(volumes)
        output.touch()
        return True

    monkeypatch.setattr(prep, "_run_fslmerge", merge)
    pca, tsnr = prep._discover_rs_inputs(tmp_path, "auto")
    assert pca == tsnr
    assert pca.name.endswith("_smooth.nii.gz")
    assert calls == [expected]
