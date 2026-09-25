"""Exercise real orchestration with synthetic, noncommuting correction operators.

Reuse the existing test suite's headless RTPSpy stubs; scanner/native libraries
are intentionally outside these orchestration tests.
"""
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
from fmri_rt_preproc.native_fieldmap import native_paths, load_calibration
from test_native_fieldmap import make_pair


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("save_intermediate", [False, True])
def test_raw_unwarp_then_mc_reaches_regression_and_qc(tmp_path, monkeypatch, fallback, save_intermediate):
    pair = make_pair(tmp_path / "pair")
    raw = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    ramp = np.arange(3, dtype=np.float32)[:, None, None] + 1
    corrected = raw * ramp
    expected = np.roll(corrected, 1, axis=0)
    assert not np.array_equal(expected, np.roll(raw, 1, axis=0) * ramp)
    source, ref = tmp_path / "raw.nii", tmp_path / "reference.nii"
    nib.save(nib.Nifti1Image(raw, np.eye(4)), source)
    nib.save(nib.Nifti1Image(expected, np.eye(4)), ref)
    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.fmap_dir, cfg.rt_motion_ref_epi = pair, ref
    cfg.enable_original_score = False
    for name in ("mc", "reg", "unwarp"):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, "rt_" + name + "_dir", folder)
    events = []

    def apply(data):
        np.testing.assert_array_equal(data, raw)
        if fallback:
            raise RuntimeError("force file fallback")
        events.append("unwarp")
        return data * ramp

    def file_apply(**kwargs):
        assert kwargs["phase_encoding_direction"] == 2
        assert kwargs["polarity"] == -1
        assert kwargs["fieldmap_path"] == native_paths(pair)[1]
        np.testing.assert_array_equal(nib.load(kwargs["epi_path"]).get_fdata(), raw)
        events.append("unwarp")
        nib.save(nib.Nifti1Image(corrected, np.eye(4)), kwargs["out_path"])

    def mc(img, vol_idx):
        assert vol_idx == 0
        np.testing.assert_array_equal(img.get_fdata(), corrected)
        events.append("mc")
        img._dataobj = np.roll(np.asarray(img.dataobj), 1, axis=0)

    handler.pyhysco_applier = SimpleNamespace(apply_volume=apply)
    handler.volreg = SimpleNamespace(do_proc=mc, _motion=np.zeros((1, 6)))
    handler.biopac_receiver = None
    handler.motion_regressor.get_regressors = lambda idx: ([], None)
    handler.volume_streamer = SimpleNamespace(publish=lambda *args: None)
    monkeypatch.setattr(rt, "apply_pyhysco_fieldmap", file_apply)
    settings = SimpleNamespace(epi_phase_encoding="PA", fieldmap_method="pyhysco",
        save_intermediate_unwarped=save_intermediate, enable_fd_censor_reg=False,
        enable_dvars_censor_reg=False, biopac_timelag=False, analysis_space="epi")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", settings)
    assert rt.process_volume(cfg, handler, source, 1, raw_nii=source, volume_timestamp=1.)
    assert events == ["unwarp", "mc"]
    np.testing.assert_array_equal(handler.prev_mc_for_dvars, expected)
    np.testing.assert_array_equal(handler.proc_src.proc_data, expected)
    np.testing.assert_array_equal(nib.load(cfg.rt_reg_dir / "vol_00001_reg.nii").get_fdata(), expected)


def test_prepare_calibration_never_registers_ap_pa(tmp_path):
    pair = make_pair(tmp_path / "pair")
    folder, field, manifest = native_paths(pair)
    manifest.unlink()
    field.unlink()
    pipe = object.__new__(FMRIRealtimePreprocessor)
    pipe.fmap_dir = pair
    pipe.fieldmap_method = "pyhysco"
    pipe.cfg = SimpleNamespace(ap_file=pair / "AP.nii", pa_file=pair / "PA.nii")
    calls = []

    def estimate(ap, pa, output_stem):
        calls.append((ap, pa))
        for original, mean in ((pipe.cfg.ap_file, ap), (pipe.cfg.pa_file, pa)):
            np.testing.assert_array_equal(nib.load(mean).get_fdata(), nib.load(original).get_fdata().mean(axis=3))
        field.touch()

    pipe._run_ap_pa_pyhysco = estimate
    pipe._prepare_fieldmap()
    pipe._prepare_fieldmap()
    assert len(calls) == 1
    assert load_calibration(pair)["motion_correction"] == "none"


def test_new_reference_is_made_from_unwarped_bold(tmp_path, monkeypatch):
    pipe = object.__new__(FMRIRealtimePreprocessor)
    pipe.rt_unwarped_analysis_ref_epi = tmp_path / "fixed.nii"
    data = np.arange(120, dtype=np.float32).reshape(3, 4, 5, 2)
    source, mean = tmp_path / "epi_unwarped.nii", tmp_path / "mean.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    refs = []

    class Volreg:
        def __init__(self, **kwargs):
            self._motion = np.zeros((2, 6))

        def set_ref_vol(self, ref):
            refs.append(ref)

        def do_proc(self, img, vol_idx):
            np.testing.assert_array_equal(img.get_fdata(), data[..., vol_idx])
            img._dataobj = np.roll(np.asarray(img.dataobj), 1, axis=0)

    monkeypatch.setattr("fmri_rt_preproc.pipeline.RtpVolreg", Volreg)
    pipe._motion_correct_and_mean(source, mean)
    assert refs == [str(source) + "[0]"]
    np.testing.assert_array_equal(nib.load(mean).get_fdata(), np.roll(data, 1, axis=0).mean(axis=3))
    nib.save(nib.load(mean), pipe.rt_unwarped_analysis_ref_epi)
    mean.unlink()
    pipe._motion_correct_and_mean(source, mean)
    assert refs[-1] == str(pipe.rt_unwarped_analysis_ref_epi)


def test_run_rejects_changed_pair_and_legacy_outputs(tmp_path, monkeypatch):
    pair = make_pair(tmp_path / "pair")
    ref = native_paths(pair)[0] / "AP_mean.nii"
    work = tmp_path / "run"
    work.mkdir()
    cfg = SimpleNamespace(rt_work_dir=work, fmap_dir=pair, rt_motion_ref_epi=ref)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(epi_phase_encoding="PA"))
    rt.validate_run_provenance(cfg)
    cfg.fmap_dir = make_pair(tmp_path / "other_pair")
    with pytest.raises(ValueError, match="changed"):
        rt.validate_run_provenance(cfg)
    (work / "preprocessing_order.json").unlink()
    (work / "mc").mkdir()
    (work / "mc" / "vol_00001_mc.nii").touch()
    with pytest.raises(ValueError, match="Legacy"):
        rt.validate_run_provenance(cfg)
