import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmri_rt_preproc.config import SubjectDayConfig, validate_registration_settings
from rt_global_settings import RegressorSettings, load_regressor_settings, save_regressor_settings


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"registration_test_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_config(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path / "freesurfer"))
    root = tmp_path / "sub-001" / "day"
    return SubjectDayConfig.for_single_run(
        subject_id="001", day_id="day", root=root, run_id="trans",
        epi_file=root / "func" / "trans" / "epi_4d.nii.gz", **kwargs,
    )


def test_default_registration_and_config_json_roundtrip(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    assert (cfg.registration_mode, cfg.registration_initialization, cfg.partial_epi_syn) == (
        "full_brain", "auto", False,
    )
    cfg.registration_mode = "partial_epi"
    cfg.registration_initialization = "identity"
    cfg.partial_epi_syn = True
    target = tmp_path / "config.json"
    cfg.to_json(target)
    loaded = SubjectDayConfig.from_json(target)
    assert (loaded.registration_mode, loaded.registration_initialization, loaded.partial_epi_syn) == (
        "partial_epi", "identity", True,
    )
    # Older preparation JSONs retain the existing full-brain behavior.
    payload = json.loads(target.read_text())
    for key in ("registration_mode", "registration_initialization", "partial_epi_syn"):
        payload.pop(key)
    target.write_text(json.dumps(payload))
    legacy = SubjectDayConfig.from_json(target)
    assert (legacy.registration_mode, legacy.registration_initialization, legacy.partial_epi_syn) == (
        "full_brain", "auto", False,
    )


@pytest.mark.parametrize("values", [
    ("slab", "auto", False),
    ("partial_epi", "random", False),
    ("partial_epi", "auto", "false"),
    ("full_brain", "auto", True),
])
def test_invalid_registration_settings_are_rejected(tmp_path, monkeypatch, values):
    mode, initialization, syn = values
    with pytest.raises(ValueError):
        validate_registration_settings(*values)
    with pytest.raises(ValueError):
        session_config(tmp_path, monkeypatch, registration_mode=mode,
                       registration_initialization=initialization, partial_epi_syn=syn)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(dict(registration_mode=mode,
                                   registration_initialization=initialization, partial_epi_syn=syn)))
    with pytest.raises(ValueError):
        load_regressor_settings(path)


def test_runtime_settings_roundtrip_and_combined_update(tmp_path):
    settings = RegressorSettings()
    # Order of JSON keys must not cause validation against the old mode.
    settings.update({"partial_epi_syn": True, "registration_mode": "PARTIAL_EPI",
                     "registration_initialization": "IDENTITY"})
    path = tmp_path / "runtime.json"
    save_regressor_settings(path, settings)
    restored = load_regressor_settings(path)
    assert (restored.registration_mode, restored.registration_initialization, restored.partial_epi_syn) == (
        "partial_epi", "identity", True,
    )
    with pytest.raises(ValueError):
        settings.update({"registration_mode": "full_brain"})
    assert settings.registration_mode == "partial_epi"
    settings.update({"registration_mode": "full_brain", "partial_epi_syn": False})
    assert settings.registration_mode == "full_brain"


def test_preparation_refuses_cache_before_staging_or_config_write(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    cfg.root.mkdir(parents=True)
    config_path = cfg.root / "config.json"
    config_path.write_text("existing configuration")
    calls = []

    class RejectingPipeline:
        def __init__(self, request):
            calls.append(request)

        def validate_registration_request(self):
            raise ValueError("Incompatible registration cache; use a fresh preparation directory.")

    monkeypatch.setitem(sys.modules, "fmri_rt_preproc.pipeline",
                        SimpleNamespace(FMRIRealtimePreprocessor=RejectingPipeline))
    script = load_script("run_preproc")
    monkeypatch.setattr(script, "_stage_convert_to_target", lambda *a: pytest.fail("staging occurred"))
    monkeypatch.setattr(script, "run", lambda *a, **kw: pytest.fail("external command occurred"))
    monkeypatch.setattr(sys, "argv", ["run_preproc.py", "--sub", "001", "--day", "day",
        "--base-data", str(tmp_path), "--incoming-root", str(tmp_path / "not-created-yet"),
        "--registration-mode", "partial_epi"])
    with pytest.raises(ValueError, match="fresh preparation"):
        script.main()
    assert config_path.read_text() == "existing configuration"
    assert len(calls) == 1
    assert calls[0].registration_mode == "partial_epi"
    assert calls[0].registration_initialization == "auto"
    assert not calls[0].partial_epi_syn


def test_preparation_records_selection_only_after_preflight(tmp_path, monkeypatch):
    cfg = session_config(tmp_path, monkeypatch)
    for path in (cfg.ap_file, cfg.pa_file, cfg.t1_file, cfg.runs[0].epi_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config_path = cfg.root / "config.json"
    config_path.write_text("old")
    checks = []

    class Pipeline:
        def __init__(self, request):
            self.cfg = request

        def validate_registration_request(self):
            assert config_path.read_text() == "old"
            checks.append("preflight")

        def run_all(self):
            recorded = json.loads(config_path.read_text())
            assert recorded["registration_mode"] == "partial_epi"
            assert recorded["registration_initialization"] == "center_of_mass"
            assert recorded["partial_epi_syn"] is True
            checks.append("run")

    monkeypatch.setitem(sys.modules, "fmri_rt_preproc.pipeline",
                        SimpleNamespace(FMRIRealtimePreprocessor=Pipeline))
    script = load_script("run_preproc")
    monkeypatch.setattr(script, "run", lambda *a, **kw: pytest.fail("unexpected conversion"))
    monkeypatch.setattr(sys, "argv", ["run_preproc.py", "--sub", "001", "--day", "day",
        "--base-data", str(tmp_path), "--registration-mode", "partial_epi",
        "--registration-initialization", "center_of_mass", "--partial-epi-syn"])
    script.main()
    assert checks == ["preflight", "preflight", "run"]


@pytest.mark.parametrize("overrides,expected", [
    ([], ("partial_epi", "identity", True)),
    (["--registration-initialization", "center_of_mass", "--no-partial-epi-syn"],
     ("partial_epi", "center_of_mass", False)),
])
def test_rs_loads_registration_profile_and_only_explicit_overrides(tmp_path, monkeypatch, overrides, expected):
    settings = RegressorSettings()
    profile = tmp_path / "runtime.json"
    profile.write_text(json.dumps({"registration_mode": "partial_epi",
                                  "registration_initialization": "identity", "partial_epi_syn": True}))
    monkeypatch.setitem(sys.modules, "rt_pipeline",
                        SimpleNamespace(REGRESSOR_SETTINGS=settings, RTSessionConfig=object))
    monkeypatch.setitem(sys.modules, "biopac_rt.biopac_receiver",
                        SimpleNamespace(BiopacReceiverConfig=object))
    received = []

    class StopAfterPreflight(Exception):
        pass

    def preflight(*args):
        received.append(args[-3:])
        raise StopAfterPreflight

    monkeypatch.setitem(sys.modules, "run_preproc", SimpleNamespace(preflight_registration=preflight))
    script = load_script("rs_realtime_parallel")
    monkeypatch.setattr(script.mp, "set_start_method", lambda *a, **kw: None)
    monkeypatch.setattr(script.mp, "get_context", lambda *a: pytest.fail("started before preflight"))
    monkeypatch.setattr(sys, "argv", ["rs_realtime_parallel.py", "--sub", "001", "--day", "day",
        "--run", "4", "--base-data", str(tmp_path), "--incoming-root", str(tmp_path),
        "--settings-file", str(profile), *overrides])
    with pytest.raises(StopAfterPreflight):
        script.main()
    assert received == [expected]
    assert not (tmp_path / "sub-001" / "day" / "func" / "4").exists()


@pytest.mark.parametrize("syn", [False, True])
def test_rs_forwards_registration_values_to_preparation(tmp_path, monkeypatch, syn):
    script = load_script("rs_realtime_parallel")
    commands = []
    monkeypatch.setattr(script.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    script._run_preproc("001", "day", tmp_path, tmp_path, 1, 2, 3, 4,
                        "partial_epi", "identity", syn)
    command, options = commands[0]
    assert command[command.index("--registration-mode") + 1] == "partial_epi"
    assert command[command.index("--registration-initialization") + 1] == "identity"
    assert ("--partial-epi-syn" if syn else "--no-partial-epi-syn") in command
    assert options == {"check": True}
