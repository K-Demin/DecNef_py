"""Voxel contributions must explain the actual ready decoder score."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from decoder_score import DecoderScorer
from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
from rt_global_settings import load_regressor_settings
from test_native_fieldmap import make_pair


def _save(path, values, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32),
                           np.eye(4) if affine is None else affine), path)
    return path


def _scorer(tmp_path, shape, affine, use_roi):
    weights = np.zeros(shape, np.float32)
    weights[0, 0, 0] = 2.
    weights[0, 1, 1] = -3.
    weights[-1, -1, -1] = .5
    weights[0, 0, -1] = np.nan  # Nonfeatures are not contributions.
    decoder = _save(tmp_path / 'decoder.nii', weights, affine)
    roi = None
    if use_roi:
        roi = tmp_path / 'ROI_DECODER.txt'
        # The ROI deliberately overrides the NIfTI support/weights, including
        # an explicit zero-weight feature and a scalar intercept.
        roi.write_text(
            f'{shape[0]} {shape[1]} {shape[2]}\n3\n'
            '1 1 1 -1.25 1000\n'
            '1 2 2 2.5 1100\n'
            f'{shape[0]} {shape[1]} {shape[2]} 0 1200\n'
            '0 0 0 7.125 0\n')
    return DecoderScorer(decoder, roi_txt=roi, n_baseline=2)


@pytest.mark.parametrize('use_roi', [False, True])
def test_contributions_use_effective_weights_and_keep_bias_scalar(tmp_path, use_roi):
    shape = (2, 3, 4)
    scorer = _scorer(tmp_path, shape, np.eye(4), use_roi)
    values = np.linspace(-4., 7., np.prod(shape)).reshape(shape)
    values[0, 1, 1] = 0.  # An acquired zero contributes zero without changing its mask.
    before = values.copy()
    weights_before = scorer.w.copy()
    contributions = scorer.contributions_from_array(values)
    expected = np.zeros(shape, np.float32)
    expected[scorer.mask] = values.astype(np.float32)[scorer.mask] * scorer.w
    assert contributions.shape == shape
    assert contributions.dtype == np.float32
    np.testing.assert_array_equal(contributions, expected)
    np.testing.assert_array_equal(contributions[~scorer.mask], 0.)
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(scorer.w, weights_before)
    assert scorer.baseline_count == 0
    assert float(contributions.sum(dtype=np.float64)) + scorer.bias == pytest.approx(
        scorer.score_from_array(values), rel=2e-6, abs=2e-6)
    if use_roi:
        assert scorer.bias == 7.125
        # The intercept must not be distributed among voxels.
        assert float(contributions.sum(dtype=np.float64)) != pytest.approx(scorer.score_from_array(values))


def test_contribution_maps_default_on_and_json_can_disable_them(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text('{}')
    assert load_regressor_settings(path).save_decoder_contributions is True
    path.write_text(json.dumps({'save_decoder_contributions': False}))
    assert load_regressor_settings(path).save_decoder_contributions is False


@pytest.mark.parametrize('use_roi', [False, True])
def test_probability_contributions_preserve_score_handshake_and_effective_intercept(tmp_path, use_roi):
    from rt_nf_events_prob_parallel import _make_direct_scorer

    shape = (2, 3, 4)
    base = _scorer(tmp_path, shape, np.eye(4), use_roi)
    intercept = -.375
    scorer = _make_direct_scorer(intercept)(base.decoder_path, roi_txt=base.roi_txt)
    values = np.linspace(-2., 3., np.prod(shape), dtype=np.float32).reshape(shape)
    scorer.accumulate_baseline(values)
    raw_score = scorer.score_from_array(values)
    assert np.isfinite(raw_score)
    assert scorer._pending is None
    contributions = scorer.contributions_from_array(values)
    assert scorer._pending is None
    assert scorer.baseline_count == 1
    assert scorer.bias == intercept  # The launcher overrides any ROI-TXT bias.
    assert contributions.sum(dtype=np.float64) + intercept == pytest.approx(raw_score, rel=2e-6, abs=2e-6)
    assert np.isnan(scorer.score_from_array(values))  # A map cannot re-arm scoring.

    next_values = values + 1.
    scorer.accumulate_baseline(next_values)
    pending = scorer._pending
    np.testing.assert_array_equal(scorer.contributions_from_array(values), contributions)
    assert scorer._pending is pending  # Nor can a map consume the next ready volume.
    assert scorer.baseline_count == 2
    next_score = scorer.score_from_array(next_values)
    assert np.isfinite(next_score)
    assert scorer.contributions_from_array(next_values).sum(dtype=np.float64) + intercept == pytest.approx(
        next_score, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize('use_roi,direct', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('scoring,finite_score', [(False, True), (True, True), (True, False)])
def test_runtime_saves_ready_contributions_on_actual_final_grid(
        tmp_path, monkeypatch, use_roi, direct, enabled, scoring, finite_score):
    """Exercise publication through real normalization and real scoring.

    The final grid differs from native MC, and saved features differ from raw
    BOLD. This distinguishes a true score decomposition from raw-intensity maps.
    """
    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.fmap_dir = make_pair(tmp_path / 'pair')
    cfg.trans_dir = tmp_path / 'trans'
    cfg.trans_dir.mkdir()
    (cfg.trans_dir / 'epi2t1_Composite.h5').touch()
    cfg.subject_root = tmp_path
    cfg.enable_scoring, cfg.enable_original_score = scoring, True
    for name in ('mc', 'reg', 'unwarp'):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, 'rt_' + name + '_dir', folder)
    native_shape, final_shape = (3, 4, 5), (2, 3, 4)
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [3., -5., 7.]
    pattern = np.arange(np.prod(native_shape), dtype=np.float32).reshape(native_shape)
    native = np.asarray([1000 + 3 * pattern + i * (pattern + 10) + i * i
                         for i in range(4)], np.float32)
    cfg.rt_motion_ref_epi = _save(tmp_path / 'motion_reference.nii', native[0])
    handler.scorer = _scorer(tmp_path, final_shape, affine, use_roi)
    if direct:
        from rt_nf_events_prob_parallel import _make_direct_scorer
        handler.scorer = _make_direct_scorer(-.375)(
            handler.scorer.decoder_path, roi_txt=handler.scorer.roi_txt)
    contribution_calls = []
    contributions = handler.scorer.contributions_from_array

    def record_contributions(values):
        contribution_calls.append(values.copy())
        return contributions(values)

    handler.scorer.contributions_from_array = record_contributions
    if not finite_score:
        handler.scorer.score_from_array = lambda values: float('inf')
    cfg.decoder_template = handler.scorer.decoder_path
    handler.motion_regressor = None
    handler.voxel_normalizer = VoxelNormalizer('zscore', 'fixed', 2, np.ones(final_shape, bool))
    handler.anatomical_brain_mask = np.ones(native_shape, bool)
    handler.final_brain_mask = handler.processing_mask = np.ones(final_shape, bool)
    handler.native_nuisance_masks = {}
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda values: values)
    handler.volreg = SimpleNamespace(do_proc=lambda *a, **k: None,
        _motion=np.zeros((4, 6)), resample_support=lambda values, index: values.copy())
    handler.biopac_receiver = handler.gpu_resampler = handler.smoother = None
    handler.volume_streamer = SimpleNamespace(publish=lambda *args: None)
    handler.reference_score_stats = None
    handler.score_event_tracker = SimpleNamespace(for_volume=lambda index: None)

    def spatial(command):
        assert command[0] == 'antsApplyTransforms'
        source = nib.load(command[command.index('-i') + 1])
        values = np.asarray(source.dataobj)[1:, 1:, 1:]
        _save(Path(command[command.index('-o') + 1]), values, affine)

    monkeypatch.setattr(rt, 'run', spatial)
    monkeypatch.setattr(rt, 'REGRESSOR_SETTINGS', SimpleNamespace(
        epi_phase_encoding='PA', save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space='t1', volume_stream_kind='score_input',
        save_decoder_contributions=enabled))
    contribution_dir = tmp_path / 'decoder_contributions'
    for volume, values in enumerate(native, 1):
        source = _save(tmp_path / f'raw_{volume}.nii', values)
        assert rt.process_volume(cfg, handler, source, volume, raw_nii=source,
                                 volume_timestamp=float(volume))
        output = contribution_dir / f'vol_{volume:05d}_decoder_contributions.nii'
        expected_file = enabled and scoring and finite_score and volume >= 3
        assert output.is_file() == expected_file
        if expected_file:
            image = nib.load(output)
            norm = nib.load(tmp_path / 'norm' / f'vol_{volume:05d}_norm.nii')
            expected = np.zeros(final_shape, np.float32)
            expected[handler.scorer.mask] = np.asarray(norm.dataobj)[handler.scorer.mask] * handler.scorer.w
            assert image.shape == final_shape
            assert image.get_data_dtype() == np.dtype(np.float32)
            np.testing.assert_array_equal(image.affine, norm.affine)
            np.testing.assert_array_equal(image.header.get_zooms(), norm.header.get_zooms())
            np.testing.assert_array_equal(image.get_fdata(), expected)
            np.testing.assert_array_equal(image.get_fdata()[~handler.scorer.mask], 0)
            # Scoring an original-intensity diagnostic must not replace the
            # normalized feature contributions published for the main score.
            raw_contributions = native[volume - 1, 1:, 1:, 1:][handler.scorer.mask] * handler.scorer.w
            assert not np.allclose(expected[handler.scorer.mask], raw_contributions)
            with (tmp_path / 'scores.csv').open() as stream:
                row = list(csv.DictReader(stream))[-1]
            assert float(expected.sum(dtype=np.float64)) + handler.scorer.bias == pytest.approx(
                float(row['score_raw']), rel=2e-6, abs=2e-6)
        if volume <= 2 or not (enabled and scoring and finite_score):
            assert not list(contribution_dir.glob('vol_*.nii'))
    expected_count = 2 if enabled and scoring and finite_score else 0
    assert len(list(contribution_dir.glob('vol_*.nii'))) == expected_count
    assert len(contribution_calls) == expected_count
    assert not list(contribution_dir.glob('.vol*'))
    if expected_count:
        metadata = json.loads((contribution_dir / 'decoder.json').read_text())
        assert metadata['bias'] == handler.scorer.bias
        assert metadata['bias_included_in_image'] is False
        assert metadata['input_folder'] == 'norm'
    if not (enabled and scoring and finite_score):
        assert not contribution_dir.exists()
