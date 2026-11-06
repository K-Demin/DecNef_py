"""Warmup anatomical previews must not become calibrated decoder features."""
import csv
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from fmri_rt_preproc.volume_smooth import MaskedGaussianSmoother
from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
from fmri_rt_preproc.decoder_qc import DecoderSpatialCorrelation
from test_native_fieldmap import make_pair


def _save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4)), path)
    return path


def _smooth_expected(values, support, fwhm):
    """Independent normalized convolution over eligible residuals only."""
    source = np.where(support, values, 0).astype(np.float64)
    if not fwhm:
        return source.astype(np.float32)
    sigma = fwhm / np.sqrt(8 * np.log(2))
    density = gaussian_filter(support.astype(float), sigma, mode="constant", cval=0., truncate=4.)
    numerator = gaussian_filter(source, sigma, mode="constant", cval=0., truncate=4.)
    result = np.zeros(source.shape, np.float32)
    result[support] = numerator[support] / density[support]
    return result


@pytest.mark.parametrize("space", ["t1", "mni", "epi"])
@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("fwhm", [0, 3])
@pytest.mark.parametrize("norm_mode", ["zscore", "mean"])
def test_warmup_transforms_mc_for_viewing_without_scoring(
        tmp_path, monkeypatch, space, gpu, glm, fwhm, norm_mode):
    """Exercise actual normalization, file publication, preview and score gates.

    A known spatial permutation replaces native registration/GPU libraries.
    The GLM double supplies known residuals and a history-excluded voxel, so
    current visual coverage is distinct from calibrated analysis eligibility.
    """
    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    cfg.subject_root = tmp_path
    cfg.trans_dir = tmp_path / "trans"
    cfg.trans_dir.mkdir()
    (cfg.trans_dir / "epi2t1_Composite.h5").touch()
    (tmp_path / "anat").mkdir()
    (tmp_path / "anat" / "warp_T1_to_MNI_synth.nii").touch()
    cfg.enable_scoring, cfg.enable_original_score = True, False
    for name in ("mc", "reg", "unwarp", "mni"):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, "rt_" + name + "_dir", folder)

    shape = (3, 4, 5)
    gain = np.linspace(.8, 1.2, np.prod(shape)).reshape(shape)
    offset = np.arange(np.prod(shape)).reshape(shape) / 20.
    native = np.asarray([v * gain + offset for v in (1000, 1100, 1200, 1300)], np.float32)
    # Spatially different time courses distinguish smoothing-before-normalizing
    # from smoothing already standardized voxels.
    native[1] += (30 * np.sin(np.arange(np.prod(shape))).reshape(shape)).astype(np.float32)
    native[2] += (50 * np.cos(np.arange(np.prod(shape))).reshape(shape)).astype(np.float32)
    residuals = np.asarray([v * gain + offset for v in (0, -10, 10, 20)], np.float32)
    # A legitimate zero z-score remains an available feature.
    residuals[3, 1, 2, 1] = residuals[:2, 1, 2, 1].mean()
    acquired = np.ones(shape, bool)
    acquired[..., -1] = False
    brain = np.zeros(shape, bool)
    brain[0:2, 1:3, :] = True
    final_brain = np.flip(brain, axis=0).copy()
    fit_support = np.flip(acquired, axis=0).copy() & final_brain
    if glm:
        fit_support[1, 2, 2] = False
        residuals[:, 1, 2, 2] = 1e6  # Rejected neighbors must not leak into smoothing.
    # Startup moments must describe the same smoothed residuals scored later,
    # including their covariance, rather than smoothed per-voxel SD maps.
    startup_residuals = np.asarray([_smooth_expected(frame, fit_support, fwhm)
                                   for frame in residuals[:2]], dtype=np.float64)
    startup_mean, startup_m2 = np.zeros(shape), np.zeros(shape)
    startup_mean[fit_support] = startup_residuals[:, fit_support].mean(axis=0)
    startup_m2[fit_support] = ((startup_residuals[:, fit_support] - startup_mean[fit_support]) ** 2).sum(axis=0)
    startup = [dict(mean=startup_mean, m2=startup_m2,
                    count=fit_support.astype(np.int64) * 2, reference_count=2)]
    handler.anatomical_brain_mask = brain
    handler.final_brain_mask = final_brain
    handler.processing_mask = final_brain.copy()
    handler.native_nuisance_masks = {}
    handler.voxel_normalizer = VoxelNormalizer(norm_mode, "fixed", 2, final_brain)
    glm_inputs = []

    def regress(image, index, **kwargs):
        glm_inputs.append(np.asarray(image.dataobj).copy())
        np.testing.assert_array_equal(kwargs["support_mask"], np.flip(acquired, axis=0) & final_brain)
        assert kwargs["nuisance_means"] == {}
        return residuals[index - 1].copy(), index >= 2

    handler.motion_regressor = (SimpleNamespace(
        apply=regress,
        get_regressors=lambda index: ([], None),
        take_startup_reference=lambda: startup.pop() if startup else None,
        output_support=fit_support, brain_mask=final_brain) if glm else None)
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda data: data)
    handler.volreg = SimpleNamespace(do_proc=lambda *args, **kwargs: None,
        _motion=np.zeros((len(native), 6)), resample_support=lambda *args: acquired.astype(float))
    handler.biopac_receiver = None
    cfg.rt_motion_ref_epi = cfg.decoder_template = _save(tmp_path / "reference.nii", native[0])
    cfg.rt_unwarped_analysis_ref_epi = cfg.rt_motion_ref_epi
    if space == "epi":
        (cfg.trans_dir / "session_to_reference_Composite.h5").touch()
        _save(cfg.trans_dir / "reference_epi.nii", native[0])
    stage = "epi_ref" if space == "epi" else space

    def spatial(values):
        return np.flip(values, axis=0).copy()

    def apply_file(source, destination):
        _save(Path(destination), spatial(nib.load(source).get_fdata()))

    def run(command):
        assert command[0] == "antsApplyTransforms"
        apply_file(command[command.index("-i") + 1], command[command.index("-o") + 1])

    monkeypatch.setattr(rt, "run", run)
    handler.gpu_resampler = (SimpleNamespace(
        resample_support=spatial, resample_nifti_to_nifti=apply_file) if gpu else None)
    handler.smoother = (MaskedGaussianSmoother(handler.final_brain_mask, np.eye(4), fwhm)
                        if fwhm else None)
    if fwhm:
        (tmp_path / "smooth").mkdir()
    scored, baseline, previews = [], [], []
    handler.scorer = SimpleNamespace(mask=handler.final_brain_mask,
        baseline_count=0, n_baseline=100,
        score_from_array=lambda data: scored.append(data.copy()) or float(data.sum()))

    def observe_baseline(data):
        baseline.append(data.copy())
        handler.scorer.baseline_count += 1

    handler.scorer.accumulate_baseline = observe_baseline
    handler.score_event_tracker = SimpleNamespace(for_volume=lambda index: None)
    handler.reference_score_stats = None

    def publish(index, primary, t1):
        # Public paths must already hold supported images when the viewer reads.
        assert primary == tmp_path / "norm" / f"vol_{index:05d}_norm.nii"
        assert primary.is_file()
        previews.append((index, primary, t1, nib.load(primary).get_fdata()))
        if space == "t1":
            assert t1 == tmp_path / "t1" / f"vol_{index:05d}_t1.nii"
            assert t1.is_file()
            assert nib.load(t1).get_fdata().max() > 500
        else:
            assert t1 is None

    handler.volume_streamer = SimpleNamespace(publish=publish)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space=space, volume_stream_kind="score_input"))

    visual_support = spatial(acquired) & handler.final_brain_mask
    qc_reference = spatial(native[0])
    handler.decoder_qc = DecoderSpatialCorrelation(
        nib.Nifti1Image(qc_reference, np.eye(4)), handler.scorer.mask,
        fixed_roi_mask=visual_support)
    analysis_support = fit_support.copy()
    history, expected_readiness, expected_scores = [], [], []
    for index, frame in enumerate(native, 1):
        glm_ready = not glm or index >= 2
        ready = glm_ready and len(history) >= 2
        expected_readiness.append(ready)
        expected_image = np.where(visual_support, spatial(np.where(acquired, frame, 0)), 0)
        expected_pre_glm = expected_image.astype(np.float32)
        clean = residuals[index - 1] if glm else expected_pre_glm
        score_input = _smooth_expected(clean, fit_support, fwhm)
        normalized = np.zeros(shape, np.float32)
        if ready:
            prior = np.asarray(history, dtype=np.float64)
            if norm_mode == "zscore":
                normalized[fit_support] = ((score_input[fit_support] - prior.mean(axis=0)) /
                                           prior.std(axis=0, ddof=1))
            else:
                normalized[fit_support] = score_input[fit_support] / prior.mean(axis=0) * 100.
        source = _save(tmp_path / f"raw_{index}.nii", frame)
        assert rt.process_volume(cfg, handler, source, index, raw_nii=source,
                                 volume_timestamp=float(index))
        public_image = nib.load(tmp_path / stage / f"vol_{index:05d}_{stage}.nii").get_fdata()
        np.testing.assert_allclose(public_image, expected_image, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(public_image[~visual_support], 0)
        np.testing.assert_allclose(nib.load(tmp_path / "norm" / f"vol_{index:05d}_norm.nii").get_fdata(),
                                   normalized, rtol=2e-5, atol=2e-5)
        if fwhm:
            smoothed = nib.load(tmp_path / "smooth" / f"vol_{index:05d}_smooth.nii").get_fdata()
            np.testing.assert_allclose(smoothed, score_input, rtol=2e-6, atol=2e-5)
        if glm:
            np.testing.assert_allclose(glm_inputs[-1], expected_pre_glm, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(nib.load(tmp_path / "reg" / f"vol_{index:05d}_reg.nii").get_fdata(),
                                   np.where(fit_support, clean, 0), rtol=2e-6, atol=2e-5)
        saved_support = nib.load(tmp_path / "support" / f"vol_{index:05d}_{stage}.nii").get_fdata()
        np.testing.assert_array_equal(saved_support, analysis_support if ready else np.zeros(shape))
        # Publishing raw-intensity warmup images must not alter the native MC file.
        np.testing.assert_array_equal(nib.load(cfg.rt_mc_dir / f"vol_{index:05d}_mc.nii").get_fdata(),
                                      np.where(acquired, frame, 0))
        preview = previews[-1][3]
        np.testing.assert_array_equal(preview[~visual_support], 0)
        np.testing.assert_allclose(preview, normalized, rtol=2e-5, atol=2e-5)
        if not ready:
            np.testing.assert_array_equal(preview, 0)
        if ready:
            expected_scores.append(normalized)
        assert len(scored) == len(baseline) == sum(expected_readiness)
        payload = handler.score_queue.get_nowait()
        assert payload["reg_ready"] == ready
        assert np.isfinite(payload["score_raw"]) == ready
        if norm_mode == "mean" and len(history) < 2:
            # Percent scaling keeps a raw-intensity denominator, independently
            # smoothed using current coverage rather than historical GLM loss.
            history.append(_smooth_expected(expected_pre_glm, visual_support, fwhm)[fit_support])
        elif norm_mode == "zscore" and glm and index == 2:
            history = list(startup_residuals[:, fit_support].copy())
        elif norm_mode == "zscore" and glm_ready and len(history) < 2:
            history.append(score_input[fit_support].copy())

    for actual, expected in zip(scored, expected_scores):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(baseline, expected_scores):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    if glm and not fwhm and norm_mode == "zscore":
        # This sample is normalized to zero, but its geometric support remains.
        assert normalized[1, 2, 1] == pytest.approx(0, abs=1e-6)
        assert saved_support[1, 2, 1] == 1
    with (tmp_path / "scores.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    assert [bool(int(row["reg_ready"])) for row in scores] == expected_readiness
    with (tmp_path / "decoder_qc_rt.csv").open() as stream:
        correlation_rows = list(csv.DictReader(stream))
    with (tmp_path / "regression_status_rt.csv").open() as stream:
        status_rows = list(csv.DictReader(stream))
    assert len(correlation_rows) == len(native)
    for index, (row, status) in enumerate(zip(correlation_rows, status_rows)):
        expected_r = np.corrcoef(qc_reference[visual_support], spatial(native[index])[visual_support])[0, 1]
        assert int(row["volume_idx"]) == index + 1
        assert row["correlation_status"] == status["correlation_status"] == "ok"
        assert float(row["reference_coverage_fraction"]) == 1
        assert float(row["reference_correlation"]) == pytest.approx(expected_r, abs=1e-7)
        assert float(status["reference_correlation"]) == pytest.approx(expected_r, abs=1e-7)
    assert not list((tmp_path / stage).glob(".vol*"))
