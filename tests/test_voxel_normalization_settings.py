import json
from pathlib import Path

import pytest

from rt_global_settings import (
    RegressorSettings, load_regressor_settings, save_regressor_settings,
)


def test_default_normalization_is_fixed_zscore_with_thirty_reference_volumes():
    settings = load_regressor_settings()
    assert (settings.voxel_norm_mode, settings.voxel_norm_reference,
            settings.voxel_norm_ref_volumes) == ('zscore', 'fixed', 30)


@pytest.mark.parametrize('mode,count', [('zscore', 2), ('mean', 1)])
@pytest.mark.parametrize('reference', ['fixed', 'running'])
def test_normalization_choices_roundtrip(tmp_path, mode, count, reference):
    settings = RegressorSettings(voxel_norm_mode=mode.upper(),
                                 voxel_norm_reference=reference.upper(),
                                 voxel_norm_ref_volumes=count)
    target = tmp_path / 'runtime.json'
    save_regressor_settings(target, settings)
    restored = load_regressor_settings(target)
    assert (restored.voxel_norm_mode, restored.voxel_norm_reference,
            restored.voxel_norm_ref_volumes) == (mode, reference, count)


@pytest.mark.parametrize('invalid', [
    {'voxel_norm_mode': 'none'}, {'voxel_norm_mode': None},
    {'voxel_norm_reference': 'latest'}, {'voxel_norm_reference': None},
    {'voxel_norm_ref_volumes': 1}, {'voxel_norm_ref_volumes': 0},
    {'voxel_norm_ref_volumes': -1}, {'voxel_norm_ref_volumes': 2.5},
    {'voxel_norm_ref_volumes': True}, {'voxel_norm_ref_volumes': 'many'},
    {'voxel_norm_ref_volumes': float('inf')}, {'voxel_norm_ref_volumes': None},
    {'voxel_norm_mode': 'mean', 'voxel_norm_ref_volumes': 0},
])
def test_invalid_normalization_is_rejected_atomically(invalid):
    with pytest.raises(ValueError, match='voxel_norm_'):
        RegressorSettings(**invalid)
    settings = RegressorSettings()
    before = settings.to_json_dict()
    with pytest.raises(ValueError, match='voxel_norm_'):
        settings.update({'TR': 3.0, 'registration_direction': 't1_to_epi', **invalid})
    assert settings.to_json_dict() == before


def test_mode_and_count_are_validated_as_one_change():
    settings = RegressorSettings(voxel_norm_mode='mean', voxel_norm_ref_volumes=1)
    with pytest.raises(ValueError, match='>= 2'):
        settings.update({'voxel_norm_mode': 'zscore'})
    assert settings.voxel_norm_mode == 'mean'
    settings.update({'voxel_norm_mode': 'zscore', 'voxel_norm_ref_volumes': 30})
    assert settings.voxel_norm_ref_volumes == 30 and settings.voxel_norm_mode == 'zscore'


def test_profile_omission_uses_new_normalization_defaults(tmp_path):
    target = tmp_path / 'runtime.json'
    target.write_text(json.dumps({'TR': 2., 'voxel_norm_ref_volumes': 30}))
    settings = load_regressor_settings(target)
    assert settings.voxel_norm_mode == 'zscore'
    assert settings.voxel_norm_reference == 'fixed'
    assert settings.TR == 2.


@pytest.mark.parametrize('name', ['rt', 'jane', 'kostya', 'minjoo', 'olga', 'sara'])
def test_runtime_profiles_explicitly_select_fixed_zscore(name):
    target = Path(__file__).resolve().parents[1] / f'{name}_settings.json'
    payload = json.loads(target.read_text(encoding='utf-8'))
    assert payload['voxel_norm_mode'] == 'zscore'
    assert payload['voxel_norm_reference'] == 'fixed'
    assert payload['voxel_norm_ref_volumes'] == 30
    settings = load_regressor_settings(target)
    assert settings.voxel_norm_mode == 'zscore' and settings.voxel_norm_reference == 'fixed'
