"""A singleton time axis must not change image values or spatial coordinates."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import prepare_offline as offline
from fmri_rt_preproc.offline_assets import copy_image, load_offline_assets, load_single_volume_image
from test_offline_import import inputs, fake_ants


@pytest.mark.parametrize("image_class", [nib.Nifti1Image, nib.Nifti2Image])
@pytest.mark.parametrize("scaled", [False, True])
def test_singleton_copy_preserves_values_and_spatial_forms(tmp_path, image_class, scaled):
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "canonical.nii.gz"
    raw = np.arange(1 * 4 * 5, dtype=np.int16).reshape(1, 4, 5, 1)
    affine = np.diag([2., 2.5, 3., 1.])
    affine[:3, 3] = [12., -19., 4.]
    img = image_class(raw, affine)
    qform = affine.copy()
    qform[0, 3] += 1.5  # Different stored forms must not be silently unified.
    img.set_qform(qform, 1)
    img.set_sform(affine, 4)
    if scaled:
        img.header.set_slope_inter(.125, 3.5)
    nib.save(img, source)
    original_bytes = source.read_bytes()
    original = nib.load(source)
    loaded = load_single_volume_image(source)
    assert loaded.shape == (1, 4, 5)  # Retain the singleton SPATIAL dimension.
    copy_image(source, target)
    restored = nib.load(target)
    assert source.read_bytes() == original_bytes
    assert restored.shape == (1, 4, 5)
    for converted in (loaded, restored):
        np.testing.assert_array_equal(np.asarray(converted.dataobj), np.asarray(original.dataobj)[..., 0])
        np.testing.assert_array_equal(converted.affine, original.affine)
        np.testing.assert_allclose(converted.get_qform(), original.get_qform(), atol=1e-7)
        np.testing.assert_array_equal(converted.get_sform(), original.get_sform())
        assert converted.header['qform_code'] == original.header['qform_code']
        assert converted.header['sform_code'] == original.header['sform_code']
    if not scaled:
        assert restored.get_data_dtype() == original.get_data_dtype()


def test_import_singleton_references_passes_3d_files_to_ants_and_writes_3d_bundle(inputs):
    source_bytes = {}
    source_values = {}
    for key in ('epi_reference', 'epi_mask', 't1', 't1_mask', 'wm_probseg', 'aseg', 'mni_template'):
        path = getattr(inputs, key)
        img = nib.load(path)
        values = np.asarray(img.dataobj)
        expanded = nib.Nifti1Image(values[..., None], img.affine, img.header)
        expanded.set_qform(img.get_qform(), int(img.header['qform_code']))
        expanded.set_sform(img.get_sform(), int(img.header['sform_code']))
        nib.save(expanded, path)
        source_bytes[key] = path.read_bytes()
        source_values[key] = values
    commands = []
    def run(command):
        reference = Path(command[command.index('-r') + 1])
        assert reference not in {inputs.t1, inputs.epi_reference}
        assert len(nib.load(reference).shape) == 3
        fake_ants(commands)(command)
    manifest = offline.import_fmriprep(inputs, run)
    assets = load_offline_assets(manifest)
    assert len(commands) == 2
    for key in ('epi_reference', 'epi_mask', 't1', 't1_mask', 'mni_template'):
        img = nib.load(assets[key])
        assert len(img.shape) == 3
        np.testing.assert_array_equal(np.asarray(img.dataobj), source_values[key])
        assert assets['provenance']['inputs'][key]['path'] == str(getattr(inputs, key).resolve())
    assert all(getattr(inputs, key).read_bytes() == values for key, values in source_bytes.items())


@pytest.mark.parametrize('shape', [(3, 4, 2, 2), (3, 4, 2, 1, 1)])
def test_import_rejects_multivolume_and_higher_dimensions_before_commands(inputs, shape):
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), inputs.epi_reference)
    with pytest.raises(ValueError, match='3D image or a 4D image with one volume'):
        offline.import_fmriprep(inputs, lambda command: pytest.fail('Invalid reference reached ANTs'))
    assert not inputs.output.exists()


def test_singleton_nonfinite_reference_remains_rejected(inputs):
    values = np.ones((3, 4, 2, 1))
    values[0, 0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, np.eye(4)), inputs.epi_reference)
    with pytest.raises(ValueError, match='finite 3D'):
        offline.import_fmriprep(inputs, lambda command: pytest.fail('Nonfinite reference reached ANTs'))
    assert not inputs.output.exists()
