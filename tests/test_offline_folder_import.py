"""Folder discovery and transform direction for a copied fMRIPrep session."""
import hashlib
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pytest

import prepare_offline as offline
from fmri_rt_preproc.offline_assets import load_offline_assets


AFFINE = """#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_float_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
"""
PREFIX = 'sub-55555_ses-01'
RUN = PREFIX + '_task-pic_run-01'
SPACE = 'MNI152NLin2009cAsym'


def nifti(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4)), path)
    return path


def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / 'copied sources'
    anat, func = root / 'anat', root / 'func'
    for suffix, values in [('desc-preproc_T1w', np.ones((5, 5, 5))),
                           ('desc-brain_mask', np.ones((5, 5, 5))),
                           ('label-WM_probseg', np.ones((5, 5, 5)))]:
        nifti(anat / f'{PREFIX}_{suffix}.nii.gz', values)
    # These are not native anatomical candidates, despite the matching suffix.
    nifti(anat / f'{PREFIX}_space-{SPACE}_desc-preproc_T1w.nii.gz', np.ones((2, 2, 2)))
    nifti(anat / 'sub-555550_ses-01_desc-preproc_T1w.nii.gz', np.ones((2, 2, 2)))
    nifti(func / f'{RUN}_desc-coreg_boldref.nii.gz', np.ones((4, 4, 4)))
    text(func / f'{RUN}_desc-coreg_boldref.nii.gz.copying', 'incomplete transfer')
    nifti(func / f'{RUN}_desc-hmc_boldref.nii.gz', np.ones((3, 3, 3)))
    text(func / f'{RUN}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt', AFFINE)
    text(anat / f'{PREFIX}_from-fsnative_to-T1w_mode-image_xfm.txt', AFFINE)
    text(anat / f'{PREFIX}_from-T1w_to-fsnative_mode-image_xfm.txt', AFFINE.replace('0 0 0\nFixed', '9 8 7\nFixed'))
    text(anat / f'{PREFIX}_from-T1w_to-{SPACE}_mode-image_xfm.h5', 'test composite')
    nifti(root / 'templates/canonical.nii.gz', np.ones((6, 6, 6)))
    labels = np.zeros((5, 5, 5), np.int16)
    labels[1, 1, 1], labels[2, 2, 2] = 4, 42
    aseg = root / 'FreeSurfer/mri/aseg.mgz'
    aseg.parent.mkdir(parents=True)
    nib.save(nib.MGHImage(labels, np.eye(4)), aseg)
    return root


def arguments(folder, *extra):
    return offline.build_parser().parse_args(['import-fmriprep-folder', '--subject', '55555',
                                              '--input', str(folder), *extra])


def fake_ants(commands):
    def execute(command):
        commands.append(command)
        output = command[command.index('-o') + 1]
        if output.startswith('['):
            Path(output[1:-3]).write_bytes(b'composed transform')
            return
        assert command[command.index('-n') + 1] == 'NearestNeighbor'
        image = nib.load(command[command.index('-i') + 1])
        reference = nib.load(command[command.index('-r') + 1])
        assert len(image.shape) == len(reference.shape) == 3
        nib.save(resample_from_to(image, reference, order=0), output)
    return execute


def source_hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and not set(p.relative_to(folder).parts) & {'derived', 'prepared'}}


def test_folder_import_preserves_sources_and_uses_correct_inverse_and_fs_direction(folder):
    before, commands = source_hashes(folder), []
    manifest = offline.import_fmriprep_folder(arguments(folder), fake_ants(commands))
    assets = load_offline_assets(manifest, '55555')
    assert manifest == folder / 'prepared/assets.json'
    assert source_hashes(folder) == before
    assert len(commands) == 4
    assert commands[0][commands[0].index('-t') + 1] == str(folder / 'anat' / f'{PREFIX}_from-fsnative_to-T1w_mode-image_xfm.txt')
    coreg = folder / 'func' / f'{RUN}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
    assert commands[1][commands[1].index('-t') + 1] == f'[{coreg},1]'
    assert commands[1][commands[1].index('-i') + 1] == str(folder / 'anat' / f'{PREFIX}_desc-brain_mask.nii.gz')
    assert commands[1][commands[1].index('-r') + 1] == str(folder / 'func' / f'{RUN}_desc-coreg_boldref.nii.gz')
    assert assets['mni_space'] == SPACE
    assert nib.load(assets['t1']).shape == (5, 5, 5)
    assert nib.load(assets['epi_reference']).shape == (4, 4, 4)
    assert nib.load(assets['vent_mask']).get_fdata()[1, 1, 1] == 1
    assert nib.load(assets['cortical_gm_mask']).get_fdata()[2, 2, 2] == 1
    provenance = assets['provenance']['folder_preparation']
    assert Path(provenance['fs_aseg']['path']).as_posix().endswith('FreeSurfer/mri/aseg.mgz')
    assert provenance['reference_run'] == '01'


@pytest.mark.parametrize('which', ['epi', 't1', 'both'])
def test_singleton_references_are_converted_for_ants_without_rewriting_originals(folder, which):
    paths = {'epi': folder / 'func' / f'{RUN}_desc-coreg_boldref.nii.gz',
             't1': folder / 'anat' / f'{PREFIX}_desc-preproc_T1w.nii.gz'}
    for name in paths if which == 'both' else [which]:
        image = nib.load(paths[name])
        nifti(paths[name], image.get_fdata()[..., None])
    before = source_hashes(folder)
    manifest = offline.import_fmriprep_folder(arguments(folder), fake_ants([]))
    assert source_hashes(folder) == before
    assets = load_offline_assets(manifest)
    assert len(nib.load(assets['epi_reference']).shape) == len(nib.load(assets['t1']).shape) == 3


def test_missing_selected_run_never_substitutes_other_run(folder):
    with pytest.raises(ValueError, match='selected run'):
        offline.import_fmriprep_folder(arguments(folder, '--reference-run', '02'), lambda cmd: pytest.fail('wrong run reached ANTs'))
    assert not (folder / 'derived').exists()


def test_ambiguous_tasks_require_explicit_selection(folder):
    nifti(folder / 'func' / f'{PREFIX}_task-other_run-01_desc-coreg_boldref.nii.gz', np.ones((3, 3, 3)))
    with pytest.raises(ValueError, match='reference-task'):
        offline.import_fmriprep_folder(arguments(folder), lambda cmd: pytest.fail('ambiguous run reached ANTs'))
    manifest = offline.import_fmriprep_folder(arguments(folder, '--reference-task', 'pic', '--reference-run', '1'), fake_ants([]))
    assert load_offline_assets(manifest)['provenance']['inputs']['epi_reference']['path'].endswith(f'{RUN}_desc-coreg_boldref.nii.gz')


def test_ambiguous_mni_space_requires_selection(folder):
    text(folder / 'anat' / f'{PREFIX}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5', 'other transform')
    with pytest.raises(ValueError, match='T1-to-MNI transform'):
        offline.import_fmriprep_folder(arguments(folder), lambda cmd: pytest.fail('ambiguous MNI reached ANTs'))
    manifest = offline.import_fmriprep_folder(arguments(folder, '--mni-space', SPACE), fake_ants([]))
    assert load_offline_assets(manifest)['mni_space'] == SPACE


def test_existing_bundle_is_rejected_before_derived_files_change(folder):
    existing = folder / 'prepared'
    existing.mkdir()
    marker = existing / 'keep.txt'
    marker.write_text('preserve')
    with pytest.raises(ValueError, match='immutable'):
        offline.import_fmriprep_folder(arguments(folder), lambda cmd: pytest.fail('existing bundle changed'))
    assert marker.read_text() == 'preserve' and not (folder / 'derived').exists()


def test_missing_fs_mapping_does_not_use_reverse_mapping(folder):
    (folder / 'anat' / f'{PREFIX}_from-fsnative_to-T1w_mode-image_xfm.txt').unlink()
    with pytest.raises(FileNotFoundError, match='from-fsnative_to-T1w'):
        offline.import_fmriprep_folder(arguments(folder), lambda cmd: pytest.fail('reverse FS mapping substituted'))
    assert not (folder / 'derived').exists()


def test_bad_anatomical_mask_grid_fails_before_external_commands(folder):
    path = folder / 'anat' / f'{PREFIX}_desc-brain_mask.nii.gz'
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] = 20
    nib.save(nib.Nifti1Image(image.get_fdata(), affine), path)
    with pytest.raises(ValueError, match='grids differ'):
        offline.import_fmriprep_folder(arguments(folder), lambda cmd: pytest.fail('bad grid reached ANTs'))
    assert not (folder / 'derived').exists()


def test_main_dispatches_folder_command_with_inferred_defaults(folder, monkeypatch):
    captured = []
    monkeypatch.setattr(offline, 'import_fmriprep_folder', lambda args: captured.append(args) or folder / 'prepared/assets.json')
    offline.main(['import-fmriprep-folder', '--subject', '55555', '--input', str(folder)])
    assert captured[0].reference_run == '01'
    assert captured[0].output is None and captured[0].mni_space is None
