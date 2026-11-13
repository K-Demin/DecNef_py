"""Spatial decoder QC reports correlation, coverage, and decoder weight loss."""
import csv
import json

import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc.decoder_qc import (
    DecoderSpatialCorrelation, add_reference_correlation_panel, has_decoder_qc,
)


def image(values, affine=None):
    return nib.Nifti1Image(np.asarray(values, dtype=np.float64), np.eye(4) if affine is None else affine)


def test_spatial_pearson_uses_effective_features_not_decoder_weights():
    ref = np.arange(24., dtype=float).reshape(2, 3, 4)
    decoder = np.ones(ref.shape, bool)
    weights = np.arange(1., ref.size + 1.)
    qc = DecoderSpatialCorrelation(image(ref), decoder, decoder_weights=weights)
    incoming = 3 * ref + 1000
    result = qc.evaluate(image(incoming), coverage=np.ones(ref.shape, bool))
    assert result.as_dict() == {
        "reference_correlation": pytest.approx(1.),
        "reference_coverage_fraction": 1.,
        "correlation_status": "ok",
        "decoder_weight_loss_fraction": 0.,
    }
    assert qc.evaluate(-incoming).reference_correlation == pytest.approx(-1.)
    curved = ref ** 2
    assert qc.evaluate(curved).reference_correlation == pytest.approx(
        np.corrcoef(ref.ravel(), curved.ravel())[0, 1])
    # A reference zero is an observed voxel and must remain in the correlation.
    assert qc.metadata['fixed_qc_voxel_count'] == ref.size
    assert qc.roi[0, 0, 0]
    np.testing.assert_array_equal(incoming, 3 * ref + 1000)


def test_fixed_session_roi_reports_partial_correlation_without_changing_denominator():
    ref = np.arange(8.).reshape(2, 2, 2)
    initial = np.ones(ref.shape, bool); initial[0, 0, :] = False
    qc = DecoderSpatialCorrelation(image(ref), np.ones(ref.shape, bool), fixed_roi_mask=initial,
                                  reference_kind='prepared_session_reference')
    assert qc.metadata['full_decoder_voxel_count'] == 8
    assert qc.metadata['fixed_qc_voxel_count'] == 6
    assert qc.metadata['excluded_decoder_fraction'] == .25
    assert 'fallback' in qc.reference_label
    assert qc.evaluate(ref, coverage=initial).correlation_status == 'ok'
    later = initial.copy(); later[1, 0, 0] = False
    incoming = ref ** 2
    incoming[1, 0, 0] = 1000
    result = qc.evaluate(incoming, coverage=later)
    assert result.correlation_status == 'partial_coverage'
    assert result.reference_correlation == pytest.approx(np.corrcoef(ref[later], incoming[later])[0, 1])
    assert result.reference_coverage_fraction == pytest.approx(5 / 6)
    assert np.isnan(result.decoder_weight_loss_fraction)
    assert qc.roi.sum() == 6
    assert qc.evaluate(ref, coverage=initial).reference_correlation == pytest.approx(1.)
    with pytest.raises(ValueError):
        qc.roi[0, 0, 0] = True


def test_weight_loss_uses_full_decoder_and_counts_nonfinite_features():
    ref = np.arange(8.).reshape(2, 2, 2)
    decoder = np.ones(ref.shape, bool)
    decoder.ravel()[1] = False
    weights = np.array([-10., 2., 1., 1., 1., 1., 1.])
    fixed_roi = decoder.copy()
    fixed_roi.ravel()[0] = False  # The strongest weight is outside the QC ROI.
    qc = DecoderSpatialCorrelation(image(ref), decoder, fixed_roi_mask=fixed_roi,
                                  decoder_weights=weights)
    coverage = np.ones(ref.shape, bool)
    coverage.ravel()[0] = False
    incoming = ref.copy()
    incoming.ravel()[2] = np.nan
    result = qc.evaluate(incoming, coverage=coverage)
    assert result.correlation_status == 'partial_coverage'
    assert result.reference_correlation == pytest.approx(1.)
    assert result.reference_coverage_fraction == pytest.approx(5 / 6)
    assert result.decoder_weight_loss_fraction == pytest.approx((100 + 4) / np.dot(weights, weights))


def test_fewer_than_two_covered_voxels_keeps_weight_loss():
    ref = np.arange(8.).reshape(2, 2, 2)
    qc = DecoderSpatialCorrelation(image(ref), np.ones(ref.shape, bool),
                                  decoder_weights=np.ones(ref.size))
    coverage = np.zeros(ref.shape, bool)
    coverage.ravel()[0] = True
    result = qc.evaluate(ref, coverage=coverage)
    assert result.correlation_status == 'insufficient_overlap'
    assert np.isnan(result.reference_correlation)
    assert result.reference_coverage_fraction == 1 / 8
    assert result.decoder_weight_loss_fraction == 7 / 8


def test_float64_centering_handles_large_common_bold_offset():
    pattern = np.array([0, 1, 4, 3, 7, 12, 10, 16.]).reshape(2, 2, 2)
    qc = DecoderSpatialCorrelation(image(1e14 + pattern), np.ones(pattern.shape, bool))
    result = qc.evaluate(2e14 + pattern * 2)
    assert result.reference_correlation == pytest.approx(1., abs=2e-15)


@pytest.mark.parametrize('ref,current,status', [
    (np.ones(8), np.arange(8.), 'constant_reference'),
    (np.arange(8.), np.ones(8), 'constant_volume'),
    (np.array([np.nan] + list(range(7))), np.arange(8.), 'nonfinite_reference'),
])
def test_degenerate_patterns_report_status_without_changing_roi(ref, current, status):
    qc = DecoderSpatialCorrelation(image(ref.reshape(2, 2, 2)), np.ones((2, 2, 2), bool))
    result = qc.evaluate(current.reshape(2, 2, 2))
    assert result.correlation_status == status
    assert np.isnan(result.reference_correlation)
    assert qc.roi.sum() == 8
    assert result.reference_coverage_fraction == 1.


@pytest.mark.parametrize('count,status', [(0, 'empty_roi'), (1, 'insufficient_voxels')])
def test_small_roi_is_diagnostic_unavailability(count, status):
    mask = np.zeros((2, 2, 2), bool); mask.ravel()[:count] = True
    qc = DecoderSpatialCorrelation(image(np.arange(8.).reshape(mask.shape)), mask)
    assert qc.evaluate(np.ones(mask.shape)).correlation_status == status


def test_reference_is_cached_and_nonfinite_voxels_outside_roi_are_irrelevant():
    ref = np.arange(8.).reshape(2, 2, 2)
    mask = np.ones(ref.shape, bool); mask[0, 0, 0] = False
    img = image(ref)
    qc = DecoderSpatialCorrelation(img, mask)
    img.dataobj[:] = 0  # Later external changes must not change cached reference.
    incoming = np.arange(8.).reshape(ref.shape); incoming[0, 0, 0] = np.nan
    assert qc.evaluate(incoming).reference_correlation == pytest.approx(1.)


def test_singleton_reference_and_explicit_roi_require_same_physical_grid(tmp_path):
    ref = np.arange(8.).reshape(2, 2, 2)
    path = tmp_path / 'mean.nii.gz'; nib.save(image(ref[..., None]), path)
    roi = np.ones(ref.shape); roi[0, 0, 0] = 0
    qc = DecoderSpatialCorrelation(path, np.ones(ref.shape, bool), analysis_image=image(ref), roi_image=image(roi))
    assert qc.roi.sum() == 7
    assert qc.evaluate(image(ref[..., None])).correlation_status == 'ok'
    shifted = np.eye(4); shifted[1, 3] = 1.
    with pytest.raises(ValueError, match='grids differ'):
        DecoderSpatialCorrelation(path, np.ones(ref.shape, bool), analysis_image=image(ref, shifted))
    with pytest.raises(ValueError, match='grids differ'):
        DecoderSpatialCorrelation(path, np.ones(ref.shape, bool), roi_image=image(roi, shifted))
    with pytest.raises(ValueError, match='grids differ'):
        qc.evaluate(image(ref, shifted))
    with pytest.raises(ValueError, match='singleton'):
        DecoderSpatialCorrelation(image(np.zeros((2, 2, 2, 3))), np.ones(ref.shape, bool))
    with pytest.raises(ValueError, match='shapes differ'):
        qc.evaluate(np.ones((3, 3, 3)))
    with pytest.raises(ValueError, match='coverage'):
        qc.evaluate(ref, coverage=np.ones((3, 3, 3)))


def test_panel_keeps_warmup_and_invalid_coverage_gap_for_legacy_csv(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (tmp_path / 'decoder_qc_rt.csv').open('w', newline='') as stream:
        w = csv.writer(stream); w.writerow(['volume_idx', 'reference_correlation', 'reference_coverage_fraction', 'correlation_status'])
        w.writerows([[1, .95, 1., 'ok'], [2, 'nan', .9, 'incomplete_coverage'], [31, .93, 1., 'ok']])
    (tmp_path / 'decoder_qc_reference.json').write_text(json.dumps({'reference_kind': 'prepared_session_reference', 'fixed_qc_voxel_count': 7, 'full_decoder_voxel_count': 10}))
    assert has_decoder_qc(tmp_path)
    fig, ax = plt.subplots()
    assert add_reference_correlation_panel(ax, tmp_path)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 31])
    assert np.isnan(ax.lines[0].get_ydata()[1])
    assert len(fig.axes[1].lines) == 1
    assert 'fallback' in ax.get_title()
    plt.close(fig)


def test_panel_connects_partial_correlation_and_plots_weight_loss(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (tmp_path / 'decoder_qc_rt.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['volume_idx', 'reference_correlation', 'reference_coverage_fraction',
                         'correlation_status', 'decoder_weight_loss_fraction'])
        writer.writerows([[1, .95, 1., 'ok', 0.], [2, .9, .99, 'partial_coverage', .15],
                          [3, .93, 1., 'ok', 0.]])
    fig, ax = plt.subplots()
    assert add_reference_correlation_panel(ax, tmp_path)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [.95, .9, .93])
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [100., 99., 100.])
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(), [0., 15., 0.])
    plt.close(fig)
