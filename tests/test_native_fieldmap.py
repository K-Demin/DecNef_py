import json

import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc.native_fieldmap import (
    calibration_spec, find_source, load_calibration, native_paths,
    require_same_grid, save_native_mean, validate_bold,
)


def make_pair(folder, axis="j"):
    folder.mkdir(parents=True, exist_ok=True)
    for name, pe in (("AP", axis), ("PA", axis + "-")):
        nib.save(nib.Nifti1Image(np.ones((3, 4, 5, 2), np.float32), np.eye(4)), folder / (name + ".nii"))
        (folder / (name + ".json")).write_text(json.dumps({"PhaseEncodingDirection": pe, "TotalReadoutTime": 0.03}))
    native, field, manifest = native_paths(folder)
    native.mkdir()
    for name in ("AP", "PA"):
        save_native_mean(folder / (name + ".nii"), native / (name + "_mean.nii"))
    field.touch()  # Geometry of PyHySCO's staggered field is deliberately not image geometry.
    manifest.write_text(json.dumps(calibration_spec(folder / "AP.nii", folder / "PA.nii")))
    return folder


def test_native_mean_does_not_realign_volumes(tmp_path):
    data = np.zeros((3, 4, 5, 2), np.float32)
    data[0, 1, 2, 0] = 8
    data[2, 1, 2, 1] = 8
    affine = np.diag([2., 3., 4., 1.])
    source, out = tmp_path / "AP.nii", tmp_path / "mean.nii"
    nib.save(nib.Nifti1Image(data, affine), source)
    save_native_mean(source, out)
    result = nib.load(out)
    np.testing.assert_array_equal(result.get_fdata(), data.mean(axis=3))
    np.testing.assert_array_equal(result.affine, affine)


@pytest.mark.parametrize("axis,expected", [("i", 1), ("j", 2), ("k", 3)])
def test_axis_is_independent_of_polarity(tmp_path, axis, expected):
    pair = make_pair(tmp_path / "pair", axis)
    img = nib.load(native_paths(pair)[0] / "AP_mean.nii")
    for polarity in ("AP", "PA"):
        assert validate_bold(img, pair, polarity)["phase_encoding_axis"] == expected


def test_rejects_stale_calibration_and_derived_input(tmp_path):
    pair = make_pair(tmp_path / "pair")
    (pair / "AP.json").write_text('{"PhaseEncodingDirection":"j", "TotalReadoutTime":0.031}')
    with pytest.raises(ValueError):
        load_calibration(pair)
    (pair / "AP.nii").rename(pair / "AP_mc.nii")
    with pytest.raises(FileNotFoundError):
        find_source(pair, "AP")


def test_rejects_geometry_and_readout_mismatch(tmp_path):
    pair = make_pair(tmp_path / "pair")
    img = nib.load(native_paths(pair)[0] / "AP_mean.nii")
    shifted = img.affine.copy()
    shifted[0, 3] += 1
    with pytest.raises(ValueError, match="grids differ"):
        require_same_grid(img, nib.Nifti1Image(img.get_fdata(), shifted))
    source = tmp_path / "bold.nii"
    source.with_suffix(".json").write_text('{"TotalReadoutTime":0.06}')
    with pytest.raises(ValueError, match="readout times differ"):
        validate_bold(img, pair, "PA", source)


def test_same_sign_pair_is_rejected(tmp_path):
    pair = make_pair(tmp_path / "pair")
    (pair / "PA.json").write_text('{"PhaseEncodingDirection":"j"}')
    with pytest.raises(ValueError, match="opposite signs"):
        calibration_spec(pair / "AP.nii", pair / "PA.nii")


def test_launcher_accepts_native_subdirectory(tmp_path):
    from rs_realtime_parallel import _check_rt_fieldmap_exists

    pair = make_pair(tmp_path / "pair-ap001_pa002")
    assert not list(pair.glob("*EstFieldMap*"))
    _check_rt_fieldmap_exists(pair)


def test_launcher_rejects_legacy_only_and_incomplete_calibration(tmp_path):
    from rs_realtime_parallel import _check_rt_fieldmap_exists

    pair = make_pair(tmp_path / "pair")
    native_paths(pair)[1].unlink()
    (pair / "pyhysco_epi-EstFieldMap.nii").touch()
    with pytest.raises(FileNotFoundError, match="native_unwarp_v1"):
        _check_rt_fieldmap_exists(pair)
