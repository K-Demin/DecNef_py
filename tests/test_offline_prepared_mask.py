"""Prepared-mask export handles fractional FLIRT edges without relaxing import."""
import hashlib
from types import SimpleNamespace

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pytest

import prepare_offline as offline
from fmri_rt_preproc.offline_assets import load_offline_assets


def save(path, values, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32),
                           np.eye(4) if affine is None else affine), path)
    return path


@pytest.fixture
def prepared(tmp_path):
    day = tmp_path / 'sub-001' / 'prepared'
    trans, anat = day / 'func/trans', day.parent / 'anat'
    affine = np.eye(4)
    affine[:3, 3] = .5
    shape = (5, 5, 5)
    native = np.zeros(shape, np.float32)
    native[1:4, 1:4, 1:4] = 1
    # Reproduce the preparation operation that creates fractional mask edges:
    # trilinear resampling of a binary segmentation onto a shifted T1 grid.
    fractional = np.asarray(resample_from_to(
        nib.Nifti1Image(native, np.eye(4)), (shape, affine), order=1).dataobj)
    assert np.any((fractional > 0) & (fractional < .5))
    mask = save(anat / 'T1_combined_mask.nii', fractional, affine)
    save(anat / 'T1_N4.nii', np.ones(shape), affine)
    save(anat / 'cortical_gm_mask.nii', np.ones(shape), affine)
    for name in ('WM_T1', 'Vent_T1'):
        save(trans / (name + '.nii'), np.ones(shape), affine)
    for name in ('epi_unwarped_mean', 'epi_mask_mean'):
        save(trans / (name + '.nii'), np.ones((3, 3, 2)))
    save(anat / 'warp_T1_to_MNI_synth.nii', np.ones(shape))
    template = save(tmp_path / 'template.nii', np.ones(shape))
    (trans / 'epi2t1_Composite.h5').write_bytes(b'forward unchanged')
    (trans / 'epi2t1_InverseComposite.h5').write_bytes(b'inverse unchanged')
    args = SimpleNamespace(subject='001', prepared_day=day, mni_template=template,
                           mni_space='MNI152NLin2009cAsym', output=tmp_path / 'bundle')
    return args, mask, fractional, affine


def test_export_binarizes_generated_edges_preserves_selection_and_sources(prepared):
    args, source_mask, fractional, affine = prepared
    source_hash = hashlib.sha256(source_mask.read_bytes()).hexdigest()
    assets = load_offline_assets(offline.export_prepared(args), '001')
    expected = fractional > 0
    for name in ('t1_mask', 'wm_mask', 'vent_mask', 'cortical_gm_mask'):
        result = nib.load(assets[name])
        np.testing.assert_array_equal(np.asarray(result.dataobj), expected)
        np.testing.assert_array_equal(result.affine, affine)
        assert result.get_data_dtype() == np.dtype('uint8')
    assert hashlib.sha256(source_mask.read_bytes()).hexdigest() == source_hash
    assert assets['epi_to_t1'].read_bytes() == b'forward unchanged'
    assert assets['t1_to_epi'].read_bytes() == b'inverse unchanged'
    recorded = assets['provenance']['t1_mask_preparation']
    assert recorded['binarization'] == 'value > 0'
    assert recorded['fractional_voxels'] == np.count_nonzero((fractional > 0) & (fractional < 1))
    assert recorded['included_voxels'] == np.count_nonzero(expected)
    assert assets['provenance']['inputs']['t1_mask']['sha256'] == source_hash


@pytest.mark.parametrize('bad_value', [-.01, 1.01, np.nan])
def test_export_rejects_invalid_prepared_mask_without_partial_bundle(prepared, bad_value):
    args, source_mask, values, affine = prepared
    values[0, 0, 0] = bad_value
    save(source_mask, values, affine)
    with pytest.raises(ValueError, match='finite|between 0 and 1'):
        offline.export_prepared(args)
    assert not args.output.exists()


def test_export_rejects_empty_prepared_mask(prepared):
    args, source_mask, values, affine = prepared
    save(source_mask, np.zeros_like(values), affine)
    with pytest.raises(ValueError, match='empty'):
        offline.export_prepared(args)
    assert not args.output.exists()


def test_fmriprep_import_still_rejects_fractional_mask(prepared, tmp_path):
    args, fractional_mask, values, affine = prepared
    binary = save(tmp_path / 'binary.nii', values > 0, affine)
    affine_path = tmp_path / 'affine.txt'
    affine_path.write_text('#Insight Transform File V1.0\n#Transform 0\n'
        'Transform: AffineTransform_double_3_3\nParameters: 1 0 0 0 1 0 0 0 1 0 0 0\n'
        'FixedParameters: 0 0 0\n')
    mni_transform = tmp_path / 'mni.h5'
    mni_transform.write_bytes(b'unused transform')
    imported = SimpleNamespace(subject='001', output=tmp_path / 'import',
        mni_space=args.mni_space, epi_reference=binary, epi_mask=binary,
        epi_to_t1=affine_path, t1_to_epi=None, t1=binary, t1_mask=fractional_mask,
        wm_probseg=binary, aseg=binary, t1_to_mni=mni_transform,
        mni_template=args.mni_template)
    with pytest.raises(ValueError, match='binary mask'):
        offline.import_fmriprep(imported, lambda _: pytest.fail('ANTs called for invalid import'))
    assert not imported.output.exists()
