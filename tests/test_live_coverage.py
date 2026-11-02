"""Follow geometric coverage through MC, denoising, space selection and viewer."""
import csv
import json
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
from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
from test_native_fieldmap import make_pair


@pytest.mark.parametrize("space,gpu", [("epi", False), ("t1", False), ("t1", True), ("mni", False), ("mni", True)])
@pytest.mark.parametrize("fwhm", [0, 3])
@pytest.mark.parametrize("brain_and_history", [False, True])
def test_missing_mc_slices_do_not_enter_final_image_or_smoothing(tmp_path, monkeypatch, space, gpu, fwhm, brain_and_history):
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

    brain = np.ones(shape, bool)
    eligible = support.astype(bool)
    if brain_and_history:
        brain[0] = False  # Acquired extracranial voxels are not brain support.
        eligible &= brain
        eligible[:, 0] = False  # A prior unavailable sample excludes this GLM voxel.
    handler.motion_regressor = SimpleNamespace(apply=regress, get_regressors=lambda idx: ([], None),
        output_support=eligible, brain_mask=brain)
    decoder_mask = np.ones(shape, bool)
    handler.scorer = SimpleNamespace(mask=decoder_mask)
    kind = "smooth" if fwhm else "masked"
    (tmp_path / kind).mkdir()
    handler.smoother = MaskedGaussianSmoother(np.ones(shape, bool), np.eye(4), fwhm)
    streamed = []
    handler.volume_streamer = SimpleNamespace(publish=lambda idx, path, t1: streamed.append((idx, path, t1)))
    transform_inputs = []
    support_transforms = []

    def transform(data):
        # A fractional spatial transform combines neighboring EPI samples.
        if np.all((data == 0) | (data == 1)):
            support_transforms.append(data.copy())
            # Even with smoothing disabled, PCA must not see an intermediate
            # T1/MNI file while its per-volume support is still being built.
            for stage in ("t1", "mni"):
                assert not (tmp_path / stage / f"vol_00001_{stage}.nii").exists()
        return shift(data, (0, 0, -.25), order=1, mode="constant", cval=0, prefilter=False)

    def apply_file(source, destination):
        data = nib.load(source).get_fdata()
        if Path(source).parent.name == "norm":
            transform_inputs.append(data.copy())
        nib.save(nib.Nifti1Image(transform(data), np.eye(4)), destination)

    def run(cmd):
        assert cmd[0] == "antsApplyTransforms"
        chain = [Path(cmd[i + 1]).name for i, value in enumerate(cmd) if value == "-t"]
        assert chain == (["warp_T1_to_MNI_synth.nii"] if space == "mni" else []) + ["epi2t1_Composite.h5"]
        assert cmd[cmd.index("-n") + 1] == "Linear"
        apply_file(cmd[cmd.index("-i") + 1], cmd[cmd.index("-o") + 1])

    handler.gpu_resampler = SimpleNamespace(resample_support=transform, resample_nifti_to_nifti=apply_file) if gpu else None
    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space=space, volume_stream_kind="mc"))
    assert rt.process_volume(cfg, handler, raw, 1, raw_nii=raw, volume_timestamp=1.)
    expected_support = support.astype(bool) if space == "epi" else shift(
        support, (0, 0, -.25), order=1, mode="constant", cval=0, prefilter=False) >= rt.SUPPORT_THRESHOLD
    def mapped(mask):
        return mask if space == "epi" else shift(mask.astype(float), (0, 0, -.25),
            order=1, mode="constant", cval=0, prefilter=False) >= rt.SUPPORT_THRESHOLD
    expected_cleaned = mapped(eligible)
    expected_orig = mapped(brain & support.astype(bool))
    coverage = tmp_path / "coverage" / f"vol_00001_{space}.nii"
    np.testing.assert_array_equal(nib.load(coverage).get_fdata(), expected_support)
    np.testing.assert_array_equal(nib.load(tmp_path / "support" / f"vol_00001_{space}.nii").get_fdata(),
                                  expected_cleaned)
    cleaned = nib.load(cfg.rt_reg_dir / "vol_00001_reg.nii").get_fdata()
    np.testing.assert_array_equal(cleaned[..., -1], 0)
    np.testing.assert_array_equal(cleaned[~eligible], 0)
    if space != "epi":
        assert len(transform_inputs) == 1
        np.testing.assert_array_equal(transform_inputs[0][..., -1], 0)
    for suffix, value, final_valid in (("", -4, expected_cleaned), ("_orig", 7, expected_orig)):
        final = nib.load(tmp_path / kind / f"vol_00001_{kind}{suffix}.nii").get_fdata()
        np.testing.assert_allclose(final[final_valid], value, atol=1e-6)
        np.testing.assert_array_equal(final[~final_valid], 0)
    with (tmp_path / "coverage_rt.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert int(row["brain_voxels"]) == brain.sum()
    assert int(row["usable_brain_voxels"]) == eligible.sum()
    assert int(row["decoder_voxels"]) == decoder_mask.sum()
    assert int(row["usable_decoder_voxels"]) == expected_cleaned.sum()
    assert float(row["decoder_coverage_fraction"]) == pytest.approx(expected_cleaned.mean())
    assert handler.decoder_support_complete is False
    assert handler.scorer.mask is decoder_mask  # No feature squeezing/reweighting.
    # Identical acquisition/analysis/original support is resampled only once.
    assert len(support_transforms) == (0 if space == "epi" else 3 if brain_and_history else 1)
    assert streamed[0][0] == 1
    assert streamed[0][1] == cfg.rt_mc_dir / "vol_00001_mc.nii"
    assert streamed[0][2] == (tmp_path / "t1" / "vol_00001_t1.nii" if space == "t1" else None)
    if space == "t1":
        before_smoothing = nib.load(streamed[0][2]).get_fdata()
        np.testing.assert_array_equal(before_smoothing[~expected_cleaned], 0)


def test_unavailable_zero_is_not_inferred_from_residual_value(tmp_path):
    data = np.array([[[0., -3., 12.]]])
    path = tmp_path / "reg.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    support = np.array([[[True, True, False]]])
    rt.zero_outside_support(path, support)
    np.testing.assert_array_equal(nib.load(path).get_fdata(), [[[0., -3., 0.]]])


def test_dvars_excludes_entering_and_leaving_acquisition_coverage():
    previous = np.array([[[1000., 0., 1000.]]])
    current = np.array([[[1002., 1000., 0.]]])
    previous_support = np.array([[[True, False, True]]])
    current_support = np.array([[[True, True, False]]])
    assert rt.compute_dvars(previous, current, np.ones_like(previous_support),
        previous_support, current_support) == 2.
    assert np.isnan(rt.compute_dvars(previous, current, np.ones_like(previous_support),
        ~current_support, current_support))


@pytest.mark.parametrize("mode", ["zscore", "mean"])
@pytest.mark.parametrize("reference", ["fixed", "running"])
def test_native_glm_then_prior_sample_normalization_then_mapping_and_score(
        tmp_path, monkeypatch, mode, reference):
    # Execute the real CPU Torch GLM and real normalizer, with only scanner/MC/
    # spatial native-library work replaced by known synthetic operators.
    from test_regression_coverage import _wrapper

    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    cfg.trans_dir = tmp_path / "trans"
    cfg.trans_dir.mkdir()
    (cfg.trans_dir / "epi2t1_Composite.h5").touch()
    cfg.subject_root = tmp_path
    cfg.enable_scoring, cfg.enable_original_score = True, False
    for name in ("mc", "reg", "unwarp"):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, "rt_" + name + "_dir", folder)
    shape = (3, 4, 5)
    brain = np.ones(shape, bool)
    brain[0, 0] = False
    acquired = np.ones(shape, bool)
    acquired[..., -1] = False
    valid = brain & acquired
    gain = np.linspace(.8, 1.5, np.prod(shape)).reshape(shape)
    offset = np.arange(np.prod(shape)).reshape(shape) / 3.
    timecourse = np.array([1000., 1100., 950., 1300., 900., 1050., 1500., 980., 1250., 1020.])
    native = np.asarray([value * gain + offset for value in timecourse], dtype=np.float32)
    ref_path = tmp_path / "reference.nii"
    nib.save(nib.Nifti1Image(native[0], np.eye(4)), ref_path)
    cfg.rt_motion_ref_epi = cfg.decoder_template = ref_path
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda data: data)
    handler.volreg = SimpleNamespace(do_proc=lambda *args, **kwargs: None,
        _motion=np.zeros((len(native), 6)), resample_support=lambda data, idx: acquired.astype(float))
    handler.biopac_receiver = None
    handler.motion_regressor = _wrapper(tmp_path, brain, n=4, motion="None", scale_before_glm=False)
    handler.motion_regressor._regress.mask_src_proc = handler.proc_src
    handler.voxel_normalizer = VoxelNormalizer(mode, reference, 3, brain)
    handler.smoother = None
    handler.gpu_resampler = None
    handler.volume_streamer = SimpleNamespace(publish=lambda *args: None)
    handler.reference_score_stats = None
    handler.score_event_tracker = SimpleNamespace(for_volume=lambda idx: None)
    scored, baseline = [], []
    score_mask = np.flip(valid, axis=0)

    def score(data):
        scored.append(data.copy())
        return float(np.sum(data[score_mask]) + 5.)

    def observe_baseline(data):
        baseline.append(data.copy())
        handler.scorer.baseline_count += 1

    handler.scorer = SimpleNamespace(mask=score_mask, baseline_count=0, n_baseline=100,
        score_from_array=score, accumulate_baseline=observe_baseline)
    transformed_normalized = []

    def run(command):
        assert command[0] == "antsApplyTransforms"
        source = Path(command[command.index("-i") + 1])
        destination = Path(command[command.index("-o") + 1])
        data = nib.load(source).get_fdata()
        if source.parent.name == "norm":
            transformed_normalized.append(data.copy())
        # An exactly known spatial permutation preserves values and support.
        nib.save(nib.Nifti1Image(np.flip(data, axis=0).copy(), np.eye(4)), destination)

    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space="t1", volume_stream_kind="score_input",
        voxel_norm_mode=mode, voxel_norm_reference=reference, voxel_norm_ref_volumes=3))

    history, expected_scores, expected_ready, expected_prior_counts = [], [], [], []
    for index, frame in enumerate(native, 1):
        source = tmp_path / f"raw_{index}.nii"
        nib.save(nib.Nifti1Image(frame, np.eye(4)), source)
        glm_ready = index >= 4
        if glm_ready:
            design = np.column_stack((np.ones(index), np.linspace(-1., 1., index)))
            target = native[:index, valid].astype(np.float64)
            native_residual = target[-1] - design[-1] @ np.linalg.lstsq(design, target, rcond=None)[0]
        else:
            native_residual = frame[valid]
        expected_reg = np.zeros(shape)
        expected_reg[valid] = native_residual
        expected_norm = np.zeros(shape)
        ready = glm_ready and len(history) >= 3
        expected_ready.append(ready)
        expected_prior_counts.append(len(history))
        if ready:
            samples = np.asarray(history)
            if mode == "zscore":
                expected_norm[valid] = (native_residual - samples.mean(axis=0)) / samples.std(axis=0, ddof=1)
            else:
                expected_norm[valid] = 100. * native_residual / samples.mean(axis=0)
            expected_scores.append(np.flip(expected_norm, axis=0))
        assert rt.process_volume(cfg, handler, source, index, raw_nii=source, volume_timestamp=float(index))
        np.testing.assert_allclose(nib.load(cfg.rt_reg_dir / f"vol_{index:05d}_reg.nii").get_fdata(),
                                   expected_reg, atol=.002, rtol=1e-5)
        normalized = nib.load(tmp_path / "norm" / f"vol_{index:05d}_norm.nii").get_fdata()
        np.testing.assert_allclose(normalized, expected_norm, atol=.002, rtol=1e-5)
        mapped = nib.load(tmp_path / "t1" / f"vol_{index:05d}_t1.nii").get_fdata()
        np.testing.assert_allclose(mapped, np.flip(expected_norm, axis=0), atol=.002, rtol=1e-5)
        assert handler.score_queue.get_nowait()["reg_ready"] == ready
        if (mode == "mean" or glm_ready) and (reference == "running" or len(history) < 3):
            history.append(frame[valid] if mode == "mean" else native_residual.copy())

    assert len(transformed_normalized) == len(native)
    assert len(scored) == len(baseline) == sum(expected_ready)
    for actual, expected in zip(scored, expected_scores):
        np.testing.assert_allclose(actual, expected, atol=.002, rtol=1e-5)
    for actual, expected in zip(baseline, expected_scores):
        np.testing.assert_allclose(actual, expected, atol=.002, rtol=1e-5)
    with (tmp_path / "regression_status_rt.csv").open() as stream:
        status = list(csv.DictReader(stream))
    assert [bool(int(row["reg_ready"])) for row in status] == expected_ready
    assert [int(row["normalization_reference_count"]) for row in status] == expected_prior_counts
    with (tmp_path / "scores.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == len(native)
    for row, ready in zip(scores, expected_ready):
        assert np.isfinite(float(row["score_raw"])) == ready
    # The startup snapshot must reproduce the reference used for the first
    # normalized volume; running updates must not overwrite that snapshot here.
    saved = json.loads((tmp_path / "norm" / "reference.json").read_text())
    assert saved["reference_count"] == 3
    assert saved["sample_source"] == ("raw_mc" if mode == "mean" else "regression_ready_output")
    assert handler.voxel_normalizer.reference_count == len(history)
