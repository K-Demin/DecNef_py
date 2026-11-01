import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmri_rt_preproc.config import SubjectDayConfig, validate_registration_settings
from rt_global_settings import (
    RegressorSettings, add_registration_arguments,
    load_regressor_settings, save_regressor_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f'registration_test_{name}', ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_config(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv('FREESURFER_HOME', str(tmp_path / 'freesurfer'))
    root = tmp_path / 'sub-001' / 'day'
    return SubjectDayConfig.for_single_run(
        subject_id='001', day_id='day', root=root, run_id='trans',
        epi_file=root / 'func' / 'trans' / 'epi_4d.nii.gz', **kwargs,
    )


def test_default_registration_and_config_json_roundtrip(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    assert (cfg.registration_initialization, cfg.registration_syn) == ('identity', False)
    cfg.registration_initialization = 'center_of_mass'
    cfg.registration_syn = True
    target = tmp_path / 'config.json'
    cfg.to_json(target)
    loaded = SubjectDayConfig.from_json(target)
    assert (loaded.registration_initialization, loaded.registration_syn) == ('center_of_mass', True)
    payload = json.loads(target.read_text())
    assert 'registration_mode' not in payload and 'partial_epi_syn' not in payload
    payload.pop('registration_initialization')
    payload.pop('registration_syn')
    target.write_text(json.dumps(payload))
    loaded = SubjectDayConfig.from_json(target)
    assert (loaded.registration_initialization, loaded.registration_syn) == ('identity', False)


@pytest.mark.parametrize('initialization,syn', [
    ('random', False), ('identity', 'false'), ('identity', 1), (None, False),
])
def test_invalid_registration_settings_are_rejected(tmp_path, monkeypatch, initialization, syn):
    with pytest.raises(ValueError):
        validate_registration_settings(initialization, syn)
    with pytest.raises(ValueError):
        session_config(tmp_path, monkeypatch,
                       registration_initialization=initialization, registration_syn=syn)
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(dict(registration_initialization=initialization, registration_syn=syn)))
    with pytest.raises(ValueError):
        load_regressor_settings(path)


def test_runtime_settings_roundtrip_and_atomic_registration_validation(tmp_path):
    settings = RegressorSettings(registration_initialization='AUTO', registration_syn=True)
    assert settings.registration_initialization == 'identity'
    settings.update({'registration_initialization': 'CENTER_OF_MASS', 'registration_syn': False})
    path = tmp_path / 'runtime.json'
    save_regressor_settings(path, settings)
    payload = json.loads(path.read_text())
    assert 'registration_mode' not in payload and 'partial_epi_syn' not in payload
    restored = load_regressor_settings(path)
    assert (restored.registration_initialization, restored.registration_syn) == ('center_of_mass', False)
    with pytest.raises(ValueError):
        settings.update({'registration_initialization': 'random', 'TR': 3})
    assert settings.registration_initialization == 'center_of_mass' and settings.TR == 1


@pytest.mark.parametrize('mode', ['full_brain', 'partial_epi'])
def test_legacy_json_keys_warn_and_cannot_select_old_recipe(tmp_path, monkeypatch, mode):
    prep_path = tmp_path / 'config.json'
    session_config(tmp_path, monkeypatch).to_json(prep_path)
    payload = json.loads(prep_path.read_text())
    payload.pop('registration_syn')
    legacy = {'registration_mode': mode, 'registration_initialization': 'auto', 'partial_epi_syn': True}
    payload.update(legacy)
    prep_path.write_text(json.dumps(payload))
    runtime_path = tmp_path / 'runtime.json'
    runtime_path.write_text(json.dumps(legacy))
    for loader, path in [(SubjectDayConfig.from_json, prep_path), (load_regressor_settings, runtime_path)]:
        with pytest.warns(FutureWarning) as warnings:
            loaded = loader(path)
        assert len(warnings) == 2
        assert (loaded.registration_initialization, loaded.registration_syn) == ('identity', True)
        assert not hasattr(loaded, 'registration_mode')
        assert not hasattr(loaded, 'partial_epi_syn')


@pytest.mark.parametrize('legacy', [
    {'registration_mode': 'slab'},
    {'partial_epi_syn': 'false'},
    {'partial_epi_syn': True, 'registration_syn': False},
    {'partial_epi_syn': False, 'registration_syn': True},
])
def test_invalid_or_conflicting_legacy_json_is_rejected(tmp_path, monkeypatch, legacy):
    prep_path = tmp_path / 'config.json'
    session_config(tmp_path, monkeypatch).to_json(prep_path)
    payload = json.loads(prep_path.read_text())
    payload.update(legacy)
    prep_path.write_text(json.dumps(payload))
    runtime_path = tmp_path / 'runtime.json'
    runtime_path.write_text(json.dumps(legacy))
    for loader, path in [(SubjectDayConfig.from_json, prep_path), (load_regressor_settings, runtime_path)]:
        with pytest.raises(ValueError):
            loader(path)


def test_matching_legacy_alias_is_accepted_but_not_serialized(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps({'partial_epi_syn': True, 'registration_syn': True}))
    with pytest.warns(FutureWarning, match='partial_epi_syn'):
        loaded = load_regressor_settings(path)
    save_regressor_settings(path, loaded)
    assert json.loads(path.read_text())['registration_syn'] is True
    assert 'partial_epi_syn' not in json.loads(path.read_text())


def test_removed_mode_is_not_an_active_cli_option():
    parser = argparse.ArgumentParser()
    add_registration_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(['--registration-mode', 'partial_epi'])
    with pytest.raises(SystemExit):
        parser.parse_args(['--partial-epi-syn'])


def test_preparation_refuses_cache_before_staging_or_config_write(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    cfg.root.mkdir(parents=True)
    config_path = cfg.root / 'config.json'
    config_path.write_text('existing configuration')
    calls = []

    class RejectingPipeline:
        def __init__(self, request):
            calls.append(request)

        def validate_registration_request(self):
            raise ValueError('Incompatible registration cache; use a fresh preparation directory.')

    monkeypatch.setitem(sys.modules, 'fmri_rt_preproc.pipeline',
                        SimpleNamespace(FMRIRealtimePreprocessor=RejectingPipeline))
    script = load_script('run_preproc')
    monkeypatch.setattr(script, '_stage_convert_to_target', lambda *a: pytest.fail('staging occurred'))
    monkeypatch.setattr(script, 'run', lambda *a, **kw: pytest.fail('external command occurred'))
    monkeypatch.setattr(sys, 'argv', ['run_preproc.py', '--sub', '001', '--day', 'day',
        '--base-data', str(tmp_path), '--incoming-root', str(tmp_path / 'not-created-yet')])
    with pytest.raises(ValueError, match='fresh preparation'):
        script.main()
    assert config_path.read_text() == 'existing configuration'
    assert len(calls) == 1
    assert calls[0].registration_initialization == 'identity'
    assert not calls[0].registration_syn


def test_preparation_records_profile_selection_only_after_preflight(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    for path in (cfg.ap_file, cfg.pa_file, cfg.t1_file, cfg.runs[0].epi_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config_path = cfg.root / 'config.json'
    config_path.write_text('old')
    profile = tmp_path / 'runtime.json'
    profile.write_text(json.dumps({'registration_initialization': 'center_of_mass', 'registration_syn': True}))
    checks = []

    class Pipeline:
        def __init__(self, request):
            self.cfg = request

        def validate_registration_request(self):
            assert config_path.read_text() == 'old'
            checks.append('preflight')

        def run_all(self):
            recorded = json.loads(config_path.read_text())
            assert recorded['registration_initialization'] == 'center_of_mass'
            assert recorded['registration_syn'] is True
            assert 'registration_mode' not in recorded and 'partial_epi_syn' not in recorded
            checks.append('run')

    monkeypatch.setitem(sys.modules, 'fmri_rt_preproc.pipeline',
                        SimpleNamespace(FMRIRealtimePreprocessor=Pipeline))
    script = load_script('run_preproc')
    monkeypatch.setattr(script, 'run', lambda *a, **kw: pytest.fail('unexpected conversion'))
    monkeypatch.setattr(sys, 'argv', ['run_preproc.py', '--sub', '001', '--day', 'day',
        '--base-data', str(tmp_path), '--settings-file', str(profile)])
    script.main()
    assert checks == ['preflight', 'preflight', 'run']


@pytest.mark.parametrize('launcher', ['run_preproc', 'rs_realtime_parallel', 'rs_pca_score_all_rois'])
@pytest.mark.parametrize('profile_values,overrides,expected', [
    (None, [], ('identity', False)),
    ({'registration_initialization': 'center_of_mass', 'registration_syn': True}, [], ('center_of_mass', True)),
    ({'registration_initialization': 'center_of_mass', 'registration_syn': True},
     ['--registration-initialization', 'auto', '--no-registration-syn'], ('identity', False)),
])
def test_all_preparation_launchers_resolve_profile_before_early_preflight(
        tmp_path, monkeypatch, launcher, profile_values, overrides, expected):
    settings = RegressorSettings()
    monkeypatch.setenv('FREESURFER_HOME', str(tmp_path / 'freesurfer'))
    monkeypatch.setitem(sys.modules, 'rt_pipeline',
                        SimpleNamespace(REGRESSOR_SETTINGS=settings, RTSessionConfig=object))
    monkeypatch.setitem(sys.modules, 'biopac_rt.biopac_receiver',
                        SimpleNamespace(BiopacReceiverConfig=object))
    received = []

    class StopAfterPreflight(Exception):
        pass

    def preflight(*args):
        received.append(args[-2:])
        raise StopAfterPreflight

    class Pipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        def validate_registration_request(self):
            received.append((self.cfg.registration_initialization, self.cfg.registration_syn))
            raise StopAfterPreflight

    monkeypatch.setitem(sys.modules, 'fmri_rt_preproc.pipeline',
                        SimpleNamespace(FMRIRealtimePreprocessor=Pipeline))
    monkeypatch.setitem(sys.modules, 'run_preproc', SimpleNamespace(preflight_registration=preflight))
    script = load_script(launcher)
    if launcher != 'run_preproc':
        monkeypatch.setattr(script.mp, 'set_start_method', lambda *a, **kw: None)
        monkeypatch.setattr(script.mp, 'get_context', lambda *a: pytest.fail('started before preflight'))
    argv = [f'{launcher}.py', '--sub', '001', '--day', 'day',
            '--base-data', str(tmp_path), '--incoming-root', str(tmp_path)]
    if launcher != 'run_preproc':
        argv += ['--run', '4']
    if profile_values is not None:
        profile = tmp_path / 'runtime.json'
        profile.write_text(json.dumps(profile_values))
        argv += ['--settings-file', str(profile)]
    monkeypatch.setattr(sys, 'argv', argv + overrides)
    with pytest.raises(StopAfterPreflight):
        script.main()
    assert received == [expected]
    assert not (tmp_path / 'sub-001').exists()


@pytest.mark.parametrize('syn', [False, True])
def test_rs_forwards_registration_values_to_preparation(tmp_path, monkeypatch, syn):
    script = load_script('rs_realtime_parallel')
    commands = []
    monkeypatch.setattr(script.subprocess, 'run', lambda cmd, **kw: commands.append((cmd, kw)))
    script._run_preproc('001', 'day', tmp_path, tmp_path, 1, 2, 3, 4, 'center_of_mass', syn)
    command, options = commands[0]
    assert command[command.index('--registration-initialization') + 1] == 'center_of_mass'
    assert ('--registration-syn' if syn else '--no-registration-syn') in command
    assert '--registration-mode' not in command and '--partial-epi-syn' not in command
    assert options == {'check': True}


def test_pca_rs_forwards_profile_into_actual_preparation_call(tmp_path, monkeypatch):
    """Regression: PCA-RS used to load the profile but omit it when preparing."""
    profile = tmp_path / 'runtime.json'
    profile.write_text(json.dumps({'registration_initialization': 'center_of_mass', 'registration_syn': True}))
    monkeypatch.setitem(sys.modules, 'rt_pipeline',
                        SimpleNamespace(REGRESSOR_SETTINGS=RegressorSettings(), RTSessionConfig=object))
    monkeypatch.setitem(sys.modules, 'biopac_rt.biopac_receiver',
                        SimpleNamespace(BiopacReceiverConfig=object))
    monkeypatch.setitem(sys.modules, 'run_preproc', SimpleNamespace(preflight_registration=lambda *a: None))
    script = load_script('rs_pca_score_all_rois')
    context = SimpleNamespace(
        Queue=lambda **kw: object(),
        Event=lambda: SimpleNamespace(set=lambda: None),
        Process=lambda **kw: SimpleNamespace(start=lambda: None, join=lambda **kw: None),
    )
    monkeypatch.setattr(script.mp, 'set_start_method', lambda *a, **kw: None)
    monkeypatch.setattr(script.mp, 'get_context', lambda *a: context)
    monkeypatch.setattr(script, '_merge_session_metadata', lambda *a: None)
    monkeypatch.setattr(script, '_wait_for_epi_dicoms', lambda *a, **kw: None)
    received = []

    class StopAtPreparation(Exception):
        pass

    def preparation(**kwargs):
        received.append(kwargs)
        raise StopAtPreparation

    monkeypatch.setattr(script, '_run_preproc', preparation)
    monkeypatch.setattr(sys, 'argv', ['rs_pca_score_all_rois.py', '--sub', '001', '--day', 'day',
        '--run', '4', '--base-data', str(tmp_path), '--incoming-root', str(tmp_path),
        '--settings-file', str(profile)])
    with pytest.raises(StopAtPreparation):
        script.main()
    assert len(received) == 1
    assert received[0]['registration_initialization'] == 'center_of_mass'
    assert received[0]['registration_syn'] is True
