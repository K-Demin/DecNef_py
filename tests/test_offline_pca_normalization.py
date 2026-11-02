import json

import nibabel as nib
import numpy as np
import pytest

import roi_rs_pca_decoder_prep as prep
from fmri_rt_preproc.processing_mask import PROCESSING_MASK_POLICY, PROCESSING_MASK_POLICY_VERSION


def _metadata(run_dir, *, space='epi', mapped=False, smoothing=0, modern=True):
    info = {'regression': {'analysis_space': space}, 'epi_reference_mapped': mapped,
            'smoothing_fwhm_mm': smoothing, 'analysis_mask': 'whole_brain'}
    if modern:
        info['normalization'] = {'mode': 'zscore', 'reference': 'fixed', 'ref_volumes': 2}
        info['coverage_policy'] = 'test_separate_support'
        info['processing_mask'] = {'policy': PROCESSING_MASK_POLICY,
                                  'version': PROCESSING_MASK_POLICY_VERSION, 'interpolation': 'linear'}
    (run_dir / 'session_metadata.json').write_text(json.dumps(info))


def _series(run_dir, folder, suffix, values, *, start=1, affine=None):
    target = run_dir / folder
    target.mkdir(exist_ok=True)
    for index, value in enumerate(values, start):
        path = target / f'vol_{index:05d}_{suffix}.nii'
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), value, np.float32),
                               np.eye(4) if affine is None else affine), path)


@pytest.fixture
def fake_merge(monkeypatch):
    calls = []

    def merge(volumes, output):
        calls.append(list(volumes))
        images = [nib.load(path) for path in volumes]
        nib.save(nib.Nifti1Image(np.stack([image.get_fdata() for image in images], axis=-1),
                               images[0].affine), output)
        return True

    monkeypatch.setattr(prep, '_run_fslmerge', merge)
    monkeypatch.setattr(prep, 'FORCE_REBUILD_4D', False)
    monkeypatch.setattr(prep, 'AUTO_FSLMERGE_4D', True)
    return calls


def test_modern_epi_uses_norm_excludes_reference_maps_and_invalidates_merge_cache(tmp_path, fake_merge):
    _metadata(tmp_path)
    _series(tmp_path, 'norm', 'norm', [0, 0, 2])
    _series(tmp_path, 'reg', 'reg', [1000, 1000, 1000])
    _series(tmp_path, 'mc', 'mc', [100, 101, 102])
    for name in ('reference_mean', 'reference_std', 'reference_count'):
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), 999., np.float32), np.eye(4)),
                 tmp_path / 'norm' / f'{name}.nii')
    pca, tsnr = prep._discover_rs_inputs(tmp_path, 'reg')
    assert pca.name == 'rs_4d_norm.nii.gz'
    assert tsnr.name == 'rs_4d_mc.nii.gz'
    np.testing.assert_allclose(nib.load(pca).get_fdata()[0, 0, 0], [0, 0, 2])
    assert len(fake_merge) == 2 and len(fake_merge[0]) == 3
    prep._discover_rs_inputs(tmp_path, 'reg')
    assert len(fake_merge) == 2
    _series(tmp_path, 'norm', 'norm', [0, 0, 4])
    prep._discover_rs_inputs(tmp_path, 'reg')
    assert len(fake_merge) == 3
    assert nib.load(pca).get_fdata()[0, 0, 0, 2] == 4


@pytest.mark.parametrize('space,mapped,mode,folder,suffix', [
    ('t1', False, 't1', 't1', 't1'),
    ('epi', True, 'reg', 'epi_ref', 'epi_ref'),
    ('t1', False, 'auto', 'smooth', 'smooth'),
])
def test_modern_transformed_input_uses_matching_original_intensity_tsnr(
        tmp_path, fake_merge, space, mapped, mode, folder, suffix):
    _metadata(tmp_path, space=space, mapped=mapped, smoothing=4 if folder == 'smooth' else 0)
    _series(tmp_path, folder, suffix, [0, 0, -2])
    original_folder = 'epi_ref' if mapped else space
    _series(tmp_path, original_folder, original_folder + '_orig', [100, 101, 102])
    pca, tsnr = prep._discover_rs_inputs(tmp_path, mode)
    assert pca.name == f'rs_4d_{suffix}.nii.gz'
    assert tsnr.name == f'rs_4d_{original_folder}_orig.nii.gz'
    np.testing.assert_allclose(nib.load(pca).get_fdata()[0, 0, 0], [0, 0, -2])
    np.testing.assert_allclose(nib.load(tsnr).get_fdata()[0, 0, 0], [100, 101, 102])


def test_modern_transformed_input_cannot_rank_tsnr_on_zscore_residuals(tmp_path, fake_merge):
    _metadata(tmp_path, space='t1')
    _series(tmp_path, 't1', 't1', [0, 0, 2])
    with pytest.raises(FileNotFoundError, match='original-intensity.*tSNR'):
        prep._discover_rs_inputs(tmp_path, 't1')


def test_explicit_reg_on_t1_run_still_means_normalized_epi(tmp_path, fake_merge):
    _metadata(tmp_path, space='t1')
    _series(tmp_path, 'norm', 'norm', [0, 0, 2])
    _series(tmp_path, 'mc', 'mc', [100, 101, 102])
    pca, tsnr = prep._discover_rs_inputs(tmp_path, 'reg')
    assert pca.name == 'rs_4d_norm.nii.gz' and tsnr.name == 'rs_4d_mc.nii.gz'


def test_modern_tsnr_grid_mismatch_is_rejected(tmp_path, fake_merge):
    _metadata(tmp_path, space='t1')
    _series(tmp_path, 't1', 't1', [0, 0, 2])
    shifted = np.eye(4)
    shifted[0, 3] = 2
    _series(tmp_path, 't1', 't1_orig', [100, 101, 102], affine=shifted)
    with pytest.raises(ValueError, match='matching grids'):
        prep._discover_rs_inputs(tmp_path, 't1')


def test_modern_missing_volume_cannot_shift_readiness_rows(tmp_path, fake_merge):
    _metadata(tmp_path)
    _series(tmp_path, 'norm', 'norm', [0, 2], start=2)
    with pytest.raises(ValueError, match='complete ordered series'):
        prep._discover_rs_inputs(tmp_path, 'reg')


def test_modern_readiness_excludes_calibration_but_keeps_first_ready_and_valid_zeros(tmp_path):
    _metadata(tmp_path)
    (tmp_path / 'regression_status_rt.csv').write_text(
        'volume_idx,reg_ready,glm_ready,normalization_ready\n'
        '1,0,0,0\n2,1,1,0\n3,1,1,1\n4,1,1,1\n')
    keep, meta = prep._initial_keep_mask(tmp_path, 4, 'auto', tmp_path / 'rs_4d_norm.nii.gz')
    np.testing.assert_array_equal(keep, [False, False, True, True])
    assert meta['drop_first_ready'] is False
    assert meta['normalization']['mode'] == 'zscore'
    assert meta['readiness_semantics'] == 'glm_and_normalization_ready'
    assert meta['upstream_processing']['normalization'] == meta['normalization']
    assert meta['upstream_processing']['processing_mask']['policy'] == PROCESSING_MASK_POLICY


def test_modern_missing_normalization_readiness_fails_closed(tmp_path):
    _metadata(tmp_path)
    (tmp_path / 'regression_status_rt.csv').write_text('volume_idx,reg_ready\n1,1\n2,1\n')
    with pytest.raises(FileNotFoundError, match='missing_columns'):
        prep._initial_keep_mask(tmp_path, 2, 'reg')


@pytest.mark.parametrize('has_metadata', [False, True])
def test_legacy_discovery_and_first_ready_policy_are_preserved(tmp_path, fake_merge, has_metadata):
    if has_metadata:
        _metadata(tmp_path, modern=False)
    _series(tmp_path, 'reg', 'reg', [100, 101, 102])
    _series(tmp_path, 'mc', 'mc', [1000, 1001, 1002])
    _series(tmp_path, 'norm', 'norm', [0, 0, 9])  # No metadata: never infer the new policy.
    pca, tsnr = prep._discover_rs_inputs(tmp_path, 'reg')
    assert pca.name == 'rs_4d_reg.nii.gz' and tsnr.name == 'rs_4d_mc.nii.gz'
    (tmp_path / 'regression_status_rt.csv').write_text('volume_idx,reg_ready\n1,0\n2,1\n3,1\n')
    keep, meta = prep._initial_keep_mask(tmp_path, 3, 'reg', pca)
    np.testing.assert_array_equal(keep, [False, False, True])
    assert meta['drop_first_ready'] is True


def test_trained_pca_normalization_is_unchanged():
    score = prep.score_components(np.array([104., 5.]), np.array([0, 1]),
                                  np.array([[1., 0.]]), normalize='z',
                                  norm_mean=np.array([100., 0.]), norm_std=np.array([2., 1.]))
    np.testing.assert_array_equal(score, [2.])
