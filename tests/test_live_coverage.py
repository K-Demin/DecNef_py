"""Follow geometric coverage through MC, denoising, space selection and viewer."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import shift

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from fmri_rt_preproc.volume_smooth import MaskedGaussianSmoother
from test_native_fieldmap import make_pair


@pytest.mark.parametrize("space,gpu", [("epi", False), ("t1", False), ("t1", True), ("mni", False), ("mni", True)])
@pytest.mark.parametrize("fwhm", [0, 3])
def test_missing_mc_slices_do_not_enter_final_image_or_smoothing(tmp_path, monkeypatch, space, gpu, fwhm):
    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    cfg.subject_root = tmp_path
    cfg.trans_dir = tmp_path / "trans"
    cfg.trans_dir.mkdir()
    (tmp_path / "anat").mkdir()
    for name in ("mc", "reg", "unwarp", "mni"):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, "rt_" + name + "_dir", folder)
    cfg.enable_original_score = True
    shape = (3, 4, 5)
    raw = tmp_path / "raw.nii"
    nib.save(nib.Nifti1Image(np.full(shape, 7., np.float32), np.eye(4)), raw)
    cfg.rt_motion_ref_epi = raw
    cfg.decoder_template = raw
    (cfg.trans_dir / "epi2t1_Composite.h5").touch()
    (tmp_path / "anat" / "warp_T1_to_MNI_synth.nii").touch()
    support = np.ones(shape, np.float32)
    support[..., -1] = 0  # One acquired slice moved outside the reference FOV.
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda data: data)

    def mc(img, vol_idx):
        # Native MC may clip zero fill to original minimum: geometry, not
        # nonzero intensity, must establish that the top slice is unavailable.
        img._dataobj = np.full(shape, 7., np.float32)

    handler.volreg = SimpleNamespace(do_proc=mc, _motion=np.zeros((1, 6)),
        resample_support=lambda data, idx: support.copy())
    handler.biopac_receiver = None

    def regress(img, *args, **kwargs):
        data = np.full(shape, -4., np.float32)  # Legitimate negative residual.
        data[..., -1] = -999.  # Missing sample minus fitted nuisance prediction.
        return data, True

    handler.motion_regressor = SimpleNamespace(apply=regress, get_regressors=lambda idx: ([], None))
    kind = "smooth" if fwhm else "masked"
    (tmp_path / kind).mkdir()
    handler.smoother = MaskedGaussianSmoother(np.ones(shape, bool), np.eye(4), fwhm)
    streamed = []
    handler.volume_streamer = SimpleNamespace(publish=lambda idx, path, t1: streamed.append((idx, path, t1)))
    transform_inputs = []

    def transform(data):
        # A fractional spatial transform combines neighboring EPI samples.
        if np.all((data == 0) | (data == 1)):
            # Even with smoothing disabled, PCA must not see an intermediate
            # T1/MNI file while its per-volume support is still being built.
            for stage in ("t1", "mni"):
                assert not (tmp_path / stage / f"vol_00001_{stage}.nii").exists()
        return shift(data, (0, 0, -.25), order=1, mode="constant", cval=0, prefilter=False)

    def apply_file(source, destination):
        data = nib.load(source).get_fdata()
        if Path(source).parent.name == "reg":
            transform_inputs.append(data.copy())
        nib.save(nib.Nifti1Image(transform(data), np.eye(4)), destination)

    def run(cmd):
        assert cmd[0] == "antsApplyTransforms"
        chain = [Path(cmd[i + 1]).name for i, value in enumerate(cmd) if value == "-t"]
        assert chain == (["warp_T1_to_MNI_synth.nii"] if space == "mni" else []) + ["epi2t1_Composite.h5"]
        assert cmd[cmd.index("-n") + 1] == "Linear"
        apply_file(cmd[cmd.index("-i") + 1], cmd[cmd.index("-o") + 1])

    handler.gpu_resampler = SimpleNamespace(resample_array=transform, resample_nifti_to_nifti=apply_file) if gpu else None
    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space=space, volume_stream_kind="mc"))
    assert rt.process_volume(cfg, handler, raw, 1, raw_nii=raw, volume_timestamp=1.)
    expected_support = support.astype(bool) if space == "epi" else shift(
        support, (0, 0, -.25), order=1, mode="constant", cval=0, prefilter=False) >= rt.SUPPORT_THRESHOLD
    coverage = tmp_path / "coverage" / f"vol_00001_{space}.nii"
    np.testing.assert_array_equal(nib.load(coverage).get_fdata(), expected_support)
    cleaned = nib.load(cfg.rt_reg_dir / "vol_00001_reg.nii").get_fdata()
    np.testing.assert_array_equal(cleaned[..., -1], 0)
    if space != "epi":
        assert len(transform_inputs) == 1
        np.testing.assert_array_equal(transform_inputs[0][..., -1], 0)
    for suffix, value in (("", -4), ("_orig", 7)):
        final = nib.load(tmp_path / kind / f"vol_00001_{kind}{suffix}.nii").get_fdata()
        np.testing.assert_allclose(final[expected_support], value, atol=1e-6)
        np.testing.assert_array_equal(final[~expected_support], 0)
    assert streamed[0][0] == 1
    assert streamed[0][1] == cfg.rt_mc_dir / "vol_00001_mc.nii"
    assert streamed[0][2] == (tmp_path / "t1" / "vol_00001_t1.nii" if space == "t1" else None)
    if space == "t1":
        before_smoothing = nib.load(streamed[0][2]).get_fdata()
        np.testing.assert_array_equal(before_smoothing[~expected_support], 0)


def test_unavailable_zero_is_not_inferred_from_residual_value(tmp_path):
    data = np.array([[[0., -3., 12.]]])
    path = tmp_path / "reg.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    support = np.array([[[True, True, False]]])
    rt.zero_outside_support(path, support)
    np.testing.assert_array_equal(nib.load(path).get_fdata(), [[[0., -3., 0.]]])
