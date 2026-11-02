"""Exercise the real CLI entrypoint without importing native runtime libraries."""
import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rt_global_settings import RegressorSettings, load_regressor_settings


ROOT = Path(__file__).resolve().parents[1]


def run_main(tmp_path, monkeypatch, profile=None, flags=()):
    tree = ast.parse((ROOT / "rt_pipeline.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    settings = RegressorSettings()
    calls = []

    def session(**kwargs):
        return SimpleNamespace(**kwargs, incoming_dir=kwargs["incoming_root"])

    namespace = dict(argparse=argparse, Path=Path, REGRESSOR_SETTINGS=settings,
                     load_regressor_settings=load_regressor_settings,
                     RTSessionConfig=session, run_rt_pipeline=calls.append)
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(ROOT / "rt_pipeline.py"), "exec"), namespace)
    argv = ["rt_pipeline.py", "--sub", "001", "--day", "test", "--run", "4",
            "--incoming-root", str(tmp_path), "--base-data", str(tmp_path)]
    if profile is not None:
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        argv += ["--settings-file", str(profile_path)]
    monkeypatch.setattr(sys, "argv", argv + list(flags))
    namespace["main"]()
    assert len(calls) == 1
    return settings, calls[0]


def test_omitted_cli_options_preserve_runtime_profile(tmp_path, monkeypatch):
    profile = {
        "analysis_space": "t1", "voxel_norm_ref_volumes": 60,
        "max_workers": 2, "rt_max_scan_length": 800, "max_retries": 5,
        "pipeline_engine": "legacy", "commit_wait_timeout_s": 2.5,
        "enable_biopac_physio": True, "biopac_host": "127.0.0.2",
        "biopac_port": 15010, "biopac_timeout": .8, "biopac_phys_reg": "RVT5",
        "biopac_handshake": False, "biopac_start_online_only": True,
        "biopac_mode": "file", "biopac_file": str(tmp_path / "physio.csv"),
        "biopac_poll_interval": .15, "biopac_timelag": True,
        "enable_volume_streamer": True, "volume_stream_kind": "score_input",
        "volume_stream_every_n": 3,
        "registration_direction": "t1_to_epi", "registration_syn": False,
        "voxel_norm_mode": "mean", "voxel_norm_reference": "running",
    }
    actual, _ = run_main(tmp_path, monkeypatch, profile)
    expected = RegressorSettings()
    expected.update(profile)
    assert actual.to_json_dict() == expected.to_json_dict()


def test_explicit_cli_values_override_profile(tmp_path, monkeypatch):
    settings, _ = run_main(tmp_path, monkeypatch, {
        "analysis_space": "t1", "voxel_norm_ref_volumes": 60, "max_workers": 2,
        "enable_biopac_physio": False, "biopac_handshake": False,
        "biopac_file": str(tmp_path / "old.csv"), "registration_syn": True,
    }, ["--analysis-space", "epi", "--voxel-norm-ref-volumes", "20",
        "--max-workers", "3", "--biopac-enable", "--biopac-handshake",
        "--biopac-file", str(tmp_path / "new.csv"), "--no-registration-syn"])
    assert settings.analysis_space == "epi"
    assert settings.voxel_norm_ref_volumes == 20
    assert settings.max_workers == 3
    assert settings.enable_biopac_physio and settings.biopac_handshake
    assert settings.biopac_file == tmp_path / "new.csv"
    assert not settings.registration_syn


def test_cli_without_profile_keeps_existing_defaults(tmp_path, monkeypatch):
    settings, _ = run_main(tmp_path, monkeypatch)
    expected = RegressorSettings(enable_biopac_physio=False)
    assert settings.to_json_dict() == expected.to_json_dict()


def test_cli_cannot_bypass_zscore_calibration_validation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="voxel_norm_ref_volumes"):
        run_main(tmp_path, monkeypatch, {"voxel_norm_mode": "zscore"},
                 ["--voxel-norm-ref-volumes", "1"])
