"""Spatial decoder QC must use fixed voxels and never repair a missing pattern."""
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
    qc = DecoderSpatialCorrelation(image(ref), decoder)
    incoming = 3 * ref + 1000
    result = qc.evaluate(image(incoming), coverage=np.ones(ref.shape, bool))
    assert result.as_dict() == {"reference_correlation": pytest.approx(1.), "reference_coverage_fraction": 1., "correlation_status": "ok"}
    assert qc.evaluate(-incoming).reference_correlation == pytest.approx(-1.)
    # A reference zero is an observed voxel and must remain in the correlation.
    assert qc.metadata['fixed_qc_voxel_count'] == ref.size
    assert qc.roi[0, 0, 0]
    np.testing.assert_array_equal(incoming, 3 * ref + 1000)


def test_fixed_session_roi_records_exclusions_and_later_loss_is_nan():
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
    result = qc.evaluate(ref, coverage=later)
    assert result.correlation_status == 'incomplete_coverage'
    assert np.isnan(result.reference_correlation)
    assert result.reference_coverage_fraction == pytest.approx(5 / 6)
    assert qc.roi.sum() == 6  # No dynamic ROI intersection, even after a bad TR.
    assert qc.evaluate(ref, coverage=initial).reference_correlation == pytest.approx(1.)
    with pytest.raises(ValueError):
        qc.roi[0, 0, 0] = True


def test_float64_centering_handles_large_common_bold_offset():
    pattern = np.array([0, 1, 4, 3, 7, 12, 10, 16.]).reshape(2, 2, 2)
    qc = DecoderSpatialCorrelation(image(1e14 + pattern), np.ones(pattern.shape, bool))
    result = qc.evaluate(2e14 + pattern * 2)
    assert result.reference_correlation == pytest.approx(1., abs=2e-15)


@pytest.mark.parametrize('ref,current,status', [
    (np.ones(8), np.arange(8.), 'constant_reference'),
    (np.arange(8.), np.ones(8), 'constant_volume'),
    (np.array([np.nan] + list(range(7))), np.arange(8.), 'nonfinite_reference'),
    (np.arange(8.), np.array([np.nan] + list(range(7))), 'nonfinite_volume'),
])
def test_degenerate_patterns_report_status_without_changing_roi(ref, current, status):
    qc = DecoderSpatialCorrelation(image(ref.reshape(2, 2, 2)), np.ones((2, 2, 2), bool))
    result = qc.evaluate(current.reshape(2, 2, 2))
    assert result.correlation_status == status
    assert np.isnan(result.reference_correlation)
    assert qc.roi.sum() == 8
    assert result.reference_coverage_fraction == (7 / 8 if status == 'nonfinite_volume' else 1.)


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


def test_panel_keeps_warmup_and_invalid_coverage_gap(tmp_path):
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
    assert 'fallback' in ax.get_title()
    plt.close(fig)
