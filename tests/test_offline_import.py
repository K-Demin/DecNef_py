"""Offline import/export contracts with synthetic images and mocked ANTs output."""
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import prepare_offline as offline
from fmri_rt_preproc.offline_assets import load_offline_assets


AFFINE = """#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 2 3 4
FixedParameters: 0 0 0
"""


def image(path, values, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4) if affine is None else affine), path)
    return path


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / 'source files'
    epi = np.ones((3, 4, 2))
    t1 = np.ones((5, 5, 5))
    wm = np.zeros(t1.shape)
    wm[0, 0, :3] = [.94, .95, 1.]
    labels = np.zeros(t1.shape)
    labels[1, 1, 1] = 4
    labels[2, 2, 2] = 42
    labels[3, 3, 3] = 2001
    values = dict(epi_reference=epi, epi_mask=epi, t1=t1, t1_mask=t1,
                  wm_probseg=wm, aseg=labels, mni_template=np.ones((7, 7, 7)))
    paths = {key: image(folder / (key + '.nii.gz'), data) for key, data in values.items()}
    forward = folder / 'forward affine.txt'
    forward.write_text(AFFINE)
    mni = folder / 't1-to-mni.h5'
    mni.write_bytes(b'fake native transform')
    return SimpleNamespace(subject='001', mni_space='MNI152NLin2009cAsym', output=tmp_path / 'bundle',
                           epi_to_t1=forward, t1_to_epi=None, t1_to_mni=mni, **paths)


def fake_ants(commands):
    def run(args):
        commands.append(args)
        output = args[args.index('-o') + 1]
        assert output.startswith('[') and output.endswith(',1]')
        Path(output[1:-3]).write_bytes(b'fake composed transform')
    return run


def test_import_affine_inverse_uses_epi_reference_and_derives_tissue_masks(inputs):
    commands = []
    manifest = offline.import_fmriprep(inputs, fake_ants(commands))
    assets = load_offline_assets(manifest, '001')
    assert len(commands) == 2
    assert commands[0][commands[0].index('-r') + 1] == str(inputs.t1)
    assert commands[0][commands[0].index('-t') + 1] == str(inputs.epi_to_t1)
    assert commands[1][commands[1].index('-r') + 1] == str(inputs.epi_reference)
    assert commands[1][commands[1].index('-t') + 1] == f'[{inputs.epi_to_t1},1]'
    wm = nib.load(assets['wm_mask']).get_fdata()
    assert wm[0, 0, 0] == 0 and wm[0, 0, 1] == 1 and wm[0, 0, 2] == 1
    vent = nib.load(assets['vent_mask']).get_fdata()
    cortex = nib.load(assets['cortical_gm_mask']).get_fdata()
    assert vent.sum() == 1 and vent[1, 1, 1] == 1
    assert cortex.sum() == 2 and cortex[2, 2, 2] == cortex[3, 3, 3] == 1
    assert assets['epi_to_t1'].suffix == assets['t1_to_epi'].suffix == '.h5'
    assert assets['provenance']['alignment_validation'] == 'not_performed_by_import'
    assert all('fieldmap' not in path.name for path in inputs.output.iterdir())
    with pytest.raises(ValueError, match='immutable'):
        offline.import_fmriprep(inputs, lambda cmd: pytest.fail('existing bundle was modified'))


@pytest.mark.parametrize('mask_key', ['epi_mask', 't1_mask'])
def test_bad_mask_geometry_fails_before_external_command_or_destination(inputs, mask_key):
    mask = nib.load(getattr(inputs, mask_key))
    affine = mask.affine.copy()
    affine[0, 3] += 10
    image(getattr(inputs, mask_key), mask.get_fdata(), affine)
    with pytest.raises(ValueError, match='grids differ'):
        offline.import_fmriprep(inputs, lambda cmd: pytest.fail('ANTs ran before validation'))
    assert not inputs.output.exists()


def test_nonlinear_forward_requires_explicit_inverse(inputs):
    inputs.epi_to_t1 = inputs.epi_to_t1.with_suffix('.h5')
    inputs.epi_to_t1.write_bytes(b'forward')
    with pytest.raises(ValueError, match='explicit --t1-to-epi'):
        offline.import_fmriprep(inputs, lambda cmd: pytest.fail('attempted nonlinear inversion'))
    inputs.t1_to_epi = inputs.epi_to_t1.with_name('inverse.h5')
    inputs.t1_to_epi.write_bytes(b'inverse')
    assets = load_offline_assets(offline.import_fmriprep(inputs, lambda cmd: pytest.fail('unnecessary composition')))
    assert assets['epi_to_t1'].read_bytes() == b'forward'
    assert assets['t1_to_epi'].read_bytes() == b'inverse'


@pytest.mark.parametrize('bad_transform', ['1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1',
    AFFINE.replace('AffineTransform_double_3_3', 'DisplacementFieldTransform_double_3_3'),
    AFFINE.replace('1 0 0 0 1 0 0 0 1', '0 0 0 0 0 0 0 0 0'), AFFINE + AFFINE])
def test_unverified_text_transform_cannot_be_inverted(inputs, bad_transform):
    inputs.epi_to_t1.write_text(bad_transform)
    with pytest.raises(ValueError, match='ITK affine'):
        offline.import_fmriprep(inputs, lambda cmd: pytest.fail('invalid affine reached ANTs'))


def test_aseg_resampled_as_labels_to_t1_physical_grid(inputs):
    labels = np.zeros((3, 3, 3))
    labels[1, 1, 1] = 43
    labels[2, 2, 2] = 3
    image(inputs.aseg, labels, np.diag([2., 2., 2., 1.]))
    assets = load_offline_assets(offline.import_fmriprep(inputs, fake_ants([])))
    vent = nib.load(assets['vent_mask'])
    assert vent.shape == (5, 5, 5) and vent.get_fdata()[2, 2, 2] == 1
    np.testing.assert_array_equal(vent.affine, nib.load(inputs.t1).affine)


def test_missing_source_is_reported_without_partial_bundle(inputs):
    inputs.aseg = inputs.aseg.with_name('missing.nii.gz')
    with pytest.raises(FileNotFoundError):
        offline.import_fmriprep(inputs, lambda cmd: pytest.fail('missing source reached ANTs'))
    assert not inputs.output.exists()


def test_export_prepared_preserves_canonical_directions_and_no_fieldmaps(tmp_path):
    day = tmp_path / 'sub-001' / 'old-day'
    trans, anat = day / 'func' / 'trans', day.parent / 'anat'
    for name in ('epi_unwarped_mean', 'epi_mask_mean'):
        image(trans / (name + '.nii'), np.ones((3, 3, 2)))
    for name in ('WM_T1', 'Vent_T1'):
        image(trans / (name + '.nii'), np.ones((4, 4, 4)))
    for name in ('T1_N4', 'T1_combined_mask'):
        image(anat / (name + '.nii'), np.ones((4, 4, 4)))
    image(anat / 'warp_T1_to_MNI_synth.nii', np.ones((5, 5, 5)))
    template = image(tmp_path / 'template.nii', np.ones((5, 5, 5)))
    (trans / 'epi2t1_Composite.h5').write_bytes(b'epi-to-t1')
    (trans / 'epi2t1_InverseComposite.h5').write_bytes(b't1-to-epi')
    args = SimpleNamespace(subject='001', prepared_day=day, mni_template=template,
                           mni_space='MNI152NLin2009cAsym', output=tmp_path / 'export')
    assets = load_offline_assets(offline.export_prepared(args), '001')
    assert assets['epi_to_t1'].read_bytes() == b'epi-to-t1'
    assert assets['t1_to_epi'].read_bytes() == b't1-to-epi'
    assert assets['t1_to_mni'].name.endswith('.nii')
    assert set(assets) >= {'wm_mask', 'vent_mask', 'mni_template'}
    assert 'calibration' not in assets


def test_cli_requires_explicit_space_and_passes_arguments(inputs, monkeypatch):
    argv = ['import-fmriprep', '--subject', inputs.subject, '--mni-space', inputs.mni_space,
            '--output', str(inputs.output)]
    for key in ('epi_reference', 'epi_mask', 'epi_to_t1', 't1', 't1_mask', 'wm_probseg',
                'aseg', 't1_to_mni', 'mni_template'):
        argv += ['--' + key.replace('_', '-'), str(getattr(inputs, key))]
    captured = []
    monkeypatch.setattr(offline, 'import_fmriprep', lambda args: captured.append(args) or inputs.output / 'assets.json')
    offline.main(argv)
    assert captured[0].epi_reference == inputs.epi_reference
    assert captured[0].mni_space == inputs.mni_space
    with pytest.raises(SystemExit):
        offline.build_parser().parse_args(['export-prepared', '--subject', '001'])
