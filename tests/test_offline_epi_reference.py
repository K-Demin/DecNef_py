"""Portable fitting references preserve original EPI and verify N4 provenance."""
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import prepare_offline as offline
from fmri_rt_preproc.epi_reference import (
    epi_reference_metadata_path, prepare_epi_registration_reference,
)
from fmri_rt_preproc.offline_assets import load_offline_assets
from test_offline_import import fake_ants, image, inputs


def test_fmriprep_import_reuses_supplied_contrast_without_refitting_n4(inputs):
    commands = []
    original = inputs.epi_reference.read_bytes()
    manifest = offline.import_fmriprep(inputs, fake_ants(commands))
    assets = load_offline_assets(manifest)
    np.testing.assert_array_equal(nib.load(assets['epi_registration_reference']).get_fdata(),
                                  nib.load(assets['epi_reference']).get_fdata())
    assert assets['epi_registration_reference'] != assets['epi_reference']
    assert inputs.epi_reference.read_bytes() == original
    assert all(command[0] == 'antsApplyTransforms' for command in commands)
    assert assets['provenance']['epi_registration_reference'] == {
        'source': 'provided_fmriprep_reference', 'n4_rerun': False,
    }


def test_old_bundle_without_optional_fitting_reference_still_loads(inputs):
    manifest = offline.import_fmriprep(inputs, fake_ants([]))
    contents = json.loads(manifest.read_text())
    contents.pop('epi_registration_reference')
    manifest.write_text(json.dumps(contents))
    assets = load_offline_assets(manifest)
    assert 'epi_registration_reference' not in assets


@pytest.mark.parametrize('invalid', ['grid', 'nonfinite', 'multivolume'])
def test_optional_fitting_reference_cannot_change_geometry_or_contain_bad_data(inputs, invalid):
    manifest = offline.import_fmriprep(inputs, fake_ants([]))
    assets = load_offline_assets(manifest)
    path = assets['epi_registration_reference']
    original = nib.load(path)
    values, affine = original.get_fdata(), original.affine.copy()
    if invalid == 'grid':
        affine[0, 3] += 2
    elif invalid == 'nonfinite':
        values[0, 0, 0] = np.nan
    else:
        values = np.stack([values, values], axis=-1)
    image(path, values, affine)
    with pytest.raises(ValueError, match='grids differ|finite 3D'):
        load_offline_assets(manifest)


@pytest.fixture
def prepared(tmp_path):
    day = tmp_path / 'sub-001' / 'prepared'
    trans, anat = day / 'func' / 'trans', day.parent / 'anat'
    epi = image(trans / 'epi_unwarped_mean.nii', np.full((3, 4, 2), 100.))
    mask = image(trans / 'epi_mask_mean.nii', np.ones((3, 4, 2)))
    for name in ('WM_T1', 'Vent_T1'):
        image(trans / (name + '.nii'), np.ones((4, 4, 4)))
    for name in ('T1_N4', 'T1_combined_mask'):
        image(anat / (name + '.nii'), np.ones((4, 4, 4)))
    image(anat / 'warp_T1_to_MNI_synth.nii', np.ones((5, 5, 5)))
    template = image(tmp_path / 'template.nii', np.ones((5, 5, 5)))
    (trans / 'epi2t1_Composite.h5').write_bytes(b'forward')
    (trans / 'epi2t1_InverseComposite.h5').write_bytes(b'inverse')
    args = SimpleNamespace(subject='001', prepared_day=day, mni_template=template,
                           mni_space='MNI152NLin2009cAsym', output=tmp_path / 'bundle')
    return args, epi, mask


def make_n4_copy(epi, mask, suffix='.nii'):
    output = epi.parent / ('epi_registration_reference' + suffix)

    def fake_n4(command):
        assert command[0] == 'N4BiasFieldCorrection'
        original = nib.load(epi)
        image(output, original.get_fdata() / 2, original.affine)

    prepare_epi_registration_reference(epi, mask, output, fake_n4)
    return output


@pytest.mark.parametrize('suffix', ['.nii', '.nii.gz'])
def test_export_preserves_original_and_verified_n4_copy(prepared, suffix):
    args, epi, mask = prepared
    n4 = make_n4_copy(epi, mask, suffix)
    original = epi.read_bytes()
    assets = load_offline_assets(offline.export_prepared(args))
    np.testing.assert_array_equal(nib.load(assets['epi_reference']).get_fdata(),
                                  nib.load(epi).get_fdata())
    np.testing.assert_array_equal(nib.load(assets['epi_registration_reference']).get_fdata(),
                                  nib.load(n4).get_fdata())
    assert epi.read_bytes() == original
    assert assets['provenance']['epi_registration_reference']['source'] == 'verified_prepared_n4'


def test_export_without_n4_remains_compatible(prepared):
    args, _, _ = prepared
    assets = load_offline_assets(offline.export_prepared(args))
    assert 'epi_registration_reference' not in assets


@pytest.mark.parametrize('changed', ['source', 'mask', 'output', 'sidecar_missing', 'output_missing'])
def test_export_rejects_stale_or_incomplete_n4_before_creating_bundle(prepared, changed):
    args, epi, mask = prepared
    n4 = make_n4_copy(epi, mask)
    if changed == 'sidecar_missing':
        epi_reference_metadata_path(n4).unlink()
    elif changed == 'output_missing':
        n4.unlink()
    else:
        path = {'source': epi, 'mask': mask, 'output': n4}[changed]
        source = nib.load(path)
        values = source.get_fdata()
        values[0, 0, 0] = 0 if changed == 'mask' else values[0, 0, 0] + 1
        image(path, values, source.affine)
    with pytest.raises(ValueError, match='N4'):
        offline.export_prepared(args)
    assert not args.output.exists()
