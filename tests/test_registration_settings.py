import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmri_rt_preproc.config import (
    SubjectDayConfig, validate_registration_direction, validate_registration_settings,
)
from rt_global_settings import (
    RegressorSettings, add_registration_arguments, apply_registration_arguments,
    load_regressor_settings, registration_cli_arguments, save_regressor_settings,
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
    assert (cfg.registration_initialization, cfg.registration_syn, cfg.registration_direction) == ('identity', True, 'epi_to_t1')
    cfg.registration_initialization = 'center_of_mass'
    cfg.registration_syn = False
    cfg.registration_direction = 't1_to_epi'
    target = tmp_path / 'config.json'
    cfg.to_json(target)
    loaded = SubjectDayConfig.from_json(target)
    assert (loaded.registration_initialization, loaded.registration_syn, loaded.registration_direction) == ('center_of_mass', False, 't1_to_epi')
    payload = json.loads(target.read_text())
    assert 'registration_mode' not in payload and 'partial_epi_syn' not in payload
    payload.pop('registration_initialization')
    payload.pop('registration_syn')
    payload.pop('registration_direction')
    target.write_text(json.dumps(payload))
    loaded = SubjectDayConfig.from_json(target)
    assert (loaded.registration_initialization, loaded.registration_syn, loaded.registration_direction) == ('identity', True, 'epi_to_t1')


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


@pytest.mark.parametrize('mode,old_syn,expected_syn', [
    ('full_brain', False, True), ('partial_epi', False, False), ('partial_epi', True, True),
])
def test_legacy_json_keys_warn_and_cannot_select_old_recipe(tmp_path, monkeypatch, mode, old_syn, expected_syn):
    prep_path = tmp_path / 'config.json'
    session_config(tmp_path, monkeypatch).to_json(prep_path)
    payload = json.loads(prep_path.read_text())
    payload.pop('registration_syn')
    legacy = {'registration_mode': mode, 'registration_initialization': 'auto', 'partial_epi_syn': old_syn}
    payload.update(legacy)
    prep_path.write_text(json.dumps(payload))
    runtime_path = tmp_path / 'runtime.json'
    runtime_path.write_text(json.dumps(legacy))
    for loader, path in [(SubjectDayConfig.from_json, prep_path), (load_regressor_settings, runtime_path)]:
        with pytest.warns(FutureWarning) as warnings:
            loaded = loader(path)
        assert len(warnings) == 2
        assert (loaded.registration_initialization, loaded.registration_syn) == ('identity', expected_syn)
        assert not hasattr(loaded, 'registration_mode')
        assert not hasattr(loaded, 'partial_epi_syn')


@pytest.mark.parametrize('legacy', [
    {'registration_mode': 'slab'},
    {'partial_epi_syn': 'false'},
    {'partial_epi_syn': True, 'registration_syn': False},
    {'partial_epi_syn': False, 'registration_syn': True},
    {'registration_mode': 'full_brain', 'partial_epi_syn': True},
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


@pytest.mark.parametrize('syn', [False, True])
def test_full_brain_partial_flag_does_not_override_explicit_shared_syn_setting(tmp_path, monkeypatch, syn):
    prep_path = tmp_path / 'config.json'
    session_config(tmp_path, monkeypatch).to_json(prep_path)
    legacy = {'registration_mode': 'full_brain', 'partial_epi_syn': False, 'registration_syn': syn}
    payload = json.loads(prep_path.read_text())
    payload.update(legacy)
    prep_path.write_text(json.dumps(payload))
    runtime_path = tmp_path / 'runtime.json'
    runtime_path.write_text(json.dumps(legacy))
    for loader, path in [(SubjectDayConfig.from_json, prep_path), (load_regressor_settings, runtime_path)]:
        with pytest.warns(FutureWarning) as warnings:
            loaded = loader(path)
        assert any('had no effect' in str(w.message) for w in warnings)
        assert loaded.registration_syn is syn


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
    assert calls[0].registration_syn
    assert calls[0].registration_direction == 'epi_to_t1'


def test_preparation_records_profile_selection_only_after_preflight(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    for path in (cfg.ap_file, cfg.pa_file, cfg.t1_file, cfg.runs[0].epi_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config_path = cfg.root / 'config.json'
    config_path.write_text('old')
    profile = tmp_path / 'runtime.json'
    profile.write_text(json.dumps({'registration_initialization': 'center_of_mass', 'registration_syn': True,
                                  'epi_phase_encoding': 'AP', 'fieldmap_method': 'pyhysco'}))
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
            assert recorded['epi_phase_encoding'] == 'AP'
            assert self.cfg.epi_phase_encoding == 'AP'
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
    (None, [], ('identity', True, 'epi_to_t1')),
    ({'registration_initialization': 'center_of_mass', 'registration_syn': True,
      'registration_direction': 't1_to_epi'}, [], ('center_of_mass', True, 't1_to_epi')),
    ({'registration_initialization': 'center_of_mass', 'registration_syn': True,
      'registration_direction': 't1_to_epi'},
     ['--registration-initialization', 'auto', '--no-registration-syn',
      '--registration-direction', 'epi_to_t1'], ('identity', False, 'epi_to_t1')),
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

    def preflight(*args, **kwargs):
        received.append((kwargs.get('registration_initialization', args[5] if len(args) > 5 else 'identity'),
                         kwargs.get('registration_syn', args[6] if len(args) > 6 else True),
                         kwargs.get('registration_direction', args[7] if len(args) > 7 else 'epi_to_t1')))
        raise StopAfterPreflight

    class Pipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        def validate_registration_request(self):
            received.append((self.cfg.registration_initialization, self.cfg.registration_syn,
                             self.cfg.registration_direction))
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
    assets = tmp_path / 'offline assets.json'
    policy = tmp_path / 'qc policy.json'
    profile = tmp_path / 'AP study settings.json'
    script._run_preproc('001', 'day', tmp_path, tmp_path, 1, 2, 3, 4, 'center_of_mass', syn,
                        registration_direction='t1_to_epi', offline_assets=assets,
                        registration_qc_policy=str(policy), settings_file=profile)
    command, options = commands[0]
    assert command[command.index('--registration-initialization') + 1] == 'center_of_mass'
    assert ('--registration-syn' if syn else '--no-registration-syn') in command
    assert command[command.index('--registration-direction') + 1] == 't1_to_epi'
    assert command[command.index('--offline-assets') + 1] == str(assets)
    assert command[command.index('--registration-qc-policy') + 1] == str(policy)
    assert command[command.index('--settings-file') + 1] == str(profile.resolve())
    assert '--registration-mode' not in command and '--partial-epi-syn' not in command
    assert options == {'check': True}


def test_pca_rs_forwards_profile_into_actual_preparation_call(tmp_path, monkeypatch):
    """Regression: PCA-RS used to load the profile but omit it when preparing."""
    profile = tmp_path / 'runtime.json'
    assets = tmp_path / 'offline assets.json'
    policy = tmp_path / 'qc policy.json'
    profile.write_text(json.dumps({'registration_initialization': 'center_of_mass', 'registration_syn': True,
                                  'registration_direction': 't1_to_epi', 'offline_assets': str(assets),
                                  'registration_qc_policy': str(policy)}))
    monkeypatch.setitem(sys.modules, 'rt_pipeline',
                        SimpleNamespace(REGRESSOR_SETTINGS=RegressorSettings(), RTSessionConfig=object))
    monkeypatch.setitem(sys.modules, 'biopac_rt.biopac_receiver',
                        SimpleNamespace(BiopacReceiverConfig=object))
    monkeypatch.setitem(sys.modules, 'run_preproc', SimpleNamespace(preflight_registration=lambda *a, **kw: None))
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
    assert received[0]['registration_direction'] == 't1_to_epi'
    assert Path(received[0]['offline_assets']) == assets
    assert received[0]['registration_qc_policy'] == str(policy)
    assert received[0]['settings_file'] == str(profile)


@pytest.mark.parametrize('direction', ['bad', None, 1])
def test_registration_direction_rejects_invalid_values_atomically(tmp_path, monkeypatch, direction):
    with pytest.raises(ValueError, match='registration_direction'):
        validate_registration_direction(direction)
    with pytest.raises(ValueError, match='registration_direction'):
        session_config(tmp_path, monkeypatch, registration_direction=direction)
    settings = RegressorSettings()
    with pytest.raises(ValueError, match='registration_direction'):
        settings.update({'registration_direction': direction, 'registration_syn': False, 'TR': 3})
    assert settings.registration_direction == 'epi_to_t1' and settings.registration_syn and settings.TR == 1


@pytest.mark.parametrize('key', ['offline_assets', 'registration_qc_policy'])
@pytest.mark.parametrize('value', ['', '   ', False, 4, []])
def test_invalid_optional_paths_rejected_before_settings_mutation(tmp_path, monkeypatch, key, value):
    settings = RegressorSettings()
    with pytest.raises(ValueError, match=key):
        settings.update({key: value, 'TR': 3})
    assert settings.TR == 1
    with pytest.raises(ValueError, match=key):
        session_config(tmp_path, monkeypatch, **{key: value})


def test_explicit_offline_paths_roundtrip_without_freesurfer_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('FREESURFER_HOME', raising=False)
    root = tmp_path / 'subject' / 'day'
    assets = tmp_path / 'offline assets.json'
    policy = tmp_path / 'QC policy.json'
    cfg = SubjectDayConfig.for_single_run(
        subject_id='001', day_id='day', root=root, run_id='trans',
        epi_file=tmp_path / 'external EPI.nii', t1_file=tmp_path / 'external T1.nii',
        ap_file=tmp_path / 'AP.nii', pa_file=tmp_path / 'PA.nii',
        mni_template=str(tmp_path / 'MNI template.nii'), offline_assets=str(assets),
        registration_qc_policy=policy, registration_direction='T1_TO_EPI',
    )
    assert cfg.offline_assets == assets and cfg.registration_qc_policy == str(policy)
    assert cfg.registration_direction == 't1_to_epi'
    path = tmp_path / 'config.json'
    cfg.to_json(path)
    loaded = SubjectDayConfig.from_json(path)
    for key in ('t1_file', 'mni_template', 'ap_file', 'pa_file', 'fieldmap_dir',
                'offline_assets', 'registration_qc_policy', 'registration_direction'):
        assert getattr(loaded, key) == getattr(cfg, key)
    assert loaded.runs == cfg.runs
    settings = RegressorSettings(offline_assets=str(assets), registration_qc_policy=policy,
                                 registration_direction='T1_TO_EPI')
    save_regressor_settings(path, settings)
    restored = load_regressor_settings(path)
    assert restored.offline_assets == assets
    assert restored.registration_qc_policy == str(policy)
    assert restored.registration_direction == 't1_to_epi'


@pytest.mark.parametrize('syn', [False, True])
@pytest.mark.parametrize('direction', ['epi_to_t1', 't1_to_epi'])
def test_shared_cli_roundtrip_preserves_complete_registration_request(tmp_path, syn, direction):
    parser = argparse.ArgumentParser()
    add_registration_arguments(parser)
    settings = RegressorSettings(registration_syn=syn, registration_direction=direction,
        registration_initialization='center_of_mass', offline_assets=tmp_path / 'asset manifest.json',
        registration_qc_policy=str(tmp_path / 'QC policy.json'))
    forwarded = registration_cli_arguments(settings)
    child = RegressorSettings()
    apply_registration_arguments(child, parser.parse_args(forwarded))
    assert child.to_json_dict() == settings.to_json_dict()
    # Omitting every CLI override retains all profile values.
    apply_registration_arguments(child, parser.parse_args([]))
    assert child.to_json_dict() == settings.to_json_dict()


def test_shared_cli_defaults_use_syn_and_manual_qc_without_implicit_paths():
    settings = load_regressor_settings(None)
    assert settings.registration_direction == 'epi_to_t1'
    assert settings.registration_syn and settings.registration_initialization == 'identity'
    assert settings.offline_assets is None and settings.registration_qc_policy is None
    assert registration_cli_arguments(settings) == ['--registration-direction', 'epi_to_t1',
        '--registration-initialization', 'identity', '--registration-syn', '--epi-reference-n4']
