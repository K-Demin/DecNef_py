import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmri_rt_preproc.config import SubjectDayConfig
from rt_global_settings import (
    RegressorSettings, add_registration_arguments, apply_registration_arguments,
    load_regressor_settings, registration_cli_arguments, save_regressor_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f'n4_settings_test_{name}', ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path, **kwargs):
    return SubjectDayConfig.for_single_run(
        subject_id='001', day_id='day', root=tmp_path / 'sub-001' / 'day',
        run_id='trans', epi_file=tmp_path / 'epi.nii',
        mni_template=tmp_path / 'template.nii', **kwargs,
    )


@pytest.mark.parametrize('enabled', [True, False])
def test_n4_profile_and_preparation_roundtrip(tmp_path, enabled):
    profile = tmp_path / 'study.json'
    save_regressor_settings(profile, RegressorSettings(epi_reference_n4=enabled))
    assert load_regressor_settings(profile).epi_reference_n4 is enabled
    prep = tmp_path / 'config.json'
    config(tmp_path, epi_reference_n4=enabled).to_json(prep)
    assert SubjectDayConfig.from_json(prep).epi_reference_n4 is enabled
    payload = json.loads(prep.read_text())
    payload.pop('epi_reference_n4')
    prep.write_text(json.dumps(payload))
    assert SubjectDayConfig.from_json(prep).epi_reference_n4 is True
    assert RegressorSettings().epi_reference_n4 is True


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None, [], {}])
def test_n4_requires_boolean_and_rejects_before_settings_mutation(tmp_path, value):
    with pytest.raises(ValueError, match='epi_reference_n4'):
        config(tmp_path, epi_reference_n4=value)
    with pytest.raises(ValueError, match='epi_reference_n4'):
        RegressorSettings(epi_reference_n4=value)
    settings = RegressorSettings()
    with pytest.raises(ValueError, match='epi_reference_n4'):
        settings.update({'epi_reference_n4': value, 'TR': 7})
    assert settings.epi_reference_n4 is True and settings.TR == 1
    profile = tmp_path / 'study.json'
    profile.write_text(json.dumps({'epi_reference_n4': value}))
    with pytest.raises(ValueError, match='epi_reference_n4'):
        load_regressor_settings(profile)


@pytest.mark.parametrize('enabled', [True, False])
def test_n4_cli_roundtrip_and_explicit_override(enabled):
    parser = argparse.ArgumentParser()
    add_registration_arguments(parser)
    source = RegressorSettings(epi_reference_n4=enabled)
    child = RegressorSettings(epi_reference_n4=not enabled)
    apply_registration_arguments(child, parser.parse_args(registration_cli_arguments(source)))
    assert child.epi_reference_n4 is enabled
    apply_registration_arguments(child, parser.parse_args([]))
    assert child.epi_reference_n4 is enabled
    flag = '--no-epi-reference-n4' if enabled else '--epi-reference-n4'
    apply_registration_arguments(child, parser.parse_args([flag]))
    assert child.epi_reference_n4 is not enabled


@pytest.mark.parametrize('launcher', ['run_preproc', 'rs_realtime_parallel', 'rs_pca_score_all_rois'])
@pytest.mark.parametrize('cli,expected', [([], False), (['--epi-reference-n4'], True)])
def test_preparation_launchers_forward_resolved_n4_to_preflight(tmp_path, monkeypatch, launcher, cli, expected):
    profile = tmp_path / 'study.json'
    profile.write_text(json.dumps({'epi_reference_n4': False}))
    monkeypatch.setenv('FREESURFER_HOME', str(tmp_path / 'freesurfer'))
    monkeypatch.setitem(sys.modules, 'rt_pipeline',
                        SimpleNamespace(REGRESSOR_SETTINGS=RegressorSettings(), RTSessionConfig=object))
    monkeypatch.setitem(sys.modules, 'biopac_rt.biopac_receiver',
                        SimpleNamespace(BiopacReceiverConfig=object))
    received = []

    class StopAfterPreflight(Exception):
        pass

    class Pipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        def validate_registration_request(self):
            received.append(self.cfg.epi_reference_n4)
            raise StopAfterPreflight

    def preflight(*args, **kwargs):
        received.append(kwargs['epi_reference_n4'])
        raise StopAfterPreflight

    monkeypatch.setitem(sys.modules, 'fmri_rt_preproc.pipeline',
                        SimpleNamespace(FMRIRealtimePreprocessor=Pipeline))
    monkeypatch.setitem(sys.modules, 'run_preproc', SimpleNamespace(preflight_registration=preflight))
    script = load_script(launcher)
    if launcher != 'run_preproc':
        monkeypatch.setattr(script.mp, 'set_start_method', lambda *a, **kw: None)
        monkeypatch.setattr(script.mp, 'get_context', lambda *a: pytest.fail('started before preflight'))
    argv = [f'{launcher}.py', '--sub', '001', '--day', 'day', '--base-data', str(tmp_path),
            '--incoming-root', str(tmp_path), '--settings-file', str(profile)]
    if launcher != 'run_preproc':
        argv += ['--run', '4']
    monkeypatch.setattr(sys, 'argv', argv + cli)
    with pytest.raises(StopAfterPreflight):
        script.main()
    assert received == [expected]
    assert not (tmp_path / 'sub-001').exists()


@pytest.mark.parametrize('enabled', [True, False])
def test_rs_preparation_child_gets_explicit_n4_selection(tmp_path, monkeypatch, enabled):
    script = load_script('rs_realtime_parallel')
    commands = []
    monkeypatch.setattr(script.subprocess, 'run', lambda command, **kw: commands.append(command))
    script._run_preproc('001', 'day', tmp_path, tmp_path, 1, 2, 3, 4,
                        settings_file=tmp_path / 'study.json', epi_reference_n4=enabled)
    assert ('--epi-reference-n4' if enabled else '--no-epi-reference-n4') in commands[0]


@pytest.mark.parametrize('launcher', ['rs_realtime_parallel', 'rs_pca_score_all_rois'])
def test_rs_parent_passes_profile_n4_into_actual_preparation(tmp_path, monkeypatch, launcher):
    profile = tmp_path / 'study.json'
    profile.write_text(json.dumps({'epi_reference_n4': False, 'enable_biopac_physio': False}))
    monkeypatch.setitem(sys.modules, 'rt_pipeline',
                        SimpleNamespace(REGRESSOR_SETTINGS=RegressorSettings(), RTSessionConfig=object))
    monkeypatch.setitem(sys.modules, 'biopac_rt.biopac_receiver',
                        SimpleNamespace(BiopacReceiverConfig=object))
    monkeypatch.setitem(sys.modules, 'run_preproc', SimpleNamespace(preflight_registration=lambda *a, **kw: None))
    script = load_script(launcher)
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
        received.append(kwargs['epi_reference_n4'])
        raise StopAtPreparation

    monkeypatch.setattr(script, '_run_preproc', preparation)
    monkeypatch.setattr(sys, 'argv', [f'{launcher}.py', '--sub', '001', '--day', 'day',
        '--run', '4', '--base-data', str(tmp_path), '--incoming-root', str(tmp_path),
        '--settings-file', str(profile)])
    with pytest.raises(StopAtPreparation):
        script.main()
    assert received == [False]
