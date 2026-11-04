"""Resolved RS registration choices must reach preflight and the prep child."""
import ast
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("overrides, expected", [
    ({}, {"--registration-initialization": "center_of_mass",
          "--registration-threads": "1", "--registration-bridge-initialization": "identity"}),
    ({"registration_initialization": "identity", "registration_threads": 4,
      "registration_bridge_initialization": "auto"},
     {"--registration-initialization": "identity", "--registration-threads": "4",
      "--registration-bridge-initialization": "auto"}),
])
def test_rs_preparation_child_receives_registration_options(overrides, expected):
    source = ROOT / "rs_realtime_parallel.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_preproc")
    calls = []
    namespace = dict(Path=Path, Optional=Optional, sys=sys, __file__=str(source),
                     log=logging.getLogger(__name__),
                     subprocess=SimpleNamespace(run=lambda command, **kwargs: calls.append((command, kwargs))))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    namespace["_run_preproc"](sub="001", day="day", base_data=Path("data"),
                              incoming_root=Path("incoming"), struct_block=None,
                              ap_block=2, pa_block=3, epi_block=4, **overrides)
    command, kwargs = calls[0]
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert "--registration-syn" in command
    assert "--epi-reference-n4" in command
    assert kwargs["check"] is True


@pytest.mark.parametrize("launcher", ["rs_realtime_parallel.py", "rs_pca_score_all_rois.py"])
@pytest.mark.parametrize("function_name", ["preflight_registration", "_run_preproc"])
def test_rs_main_forwards_user_overrides_to_preflight_and_preparation(launcher, function_name):
    source = ROOT / launcher
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == function_name)
    captured = []
    settings = SimpleNamespace(registration_initialization="identity", registration_threads=4,
                               registration_bridge_initialization="auto", registration_syn=False,
                               registration_direction="t1_to_epi", offline_assets=None,
                               registration_qc_policy=None, epi_reference_n4=True)
    args = SimpleNamespace(sub="001", day="day", ap_block=2, pa_block=3,
                           struct_block=None, settings_file=None)
    namespace = {function_name: lambda *a, **kw: captured.append((a, kw)),
                 "REGRESSOR_SETTINGS": settings, "args": args, "sub": "001",
                 "base_data": Path("data"), "incoming_root": Path("incoming"), "epi_block": 4}
    eval(compile(ast.Expression(call), str(source), "eval"), namespace)
    kwargs = captured[0][1]
    assert kwargs["registration_threads"] == 4
    assert kwargs["registration_bridge_initialization"] == "auto"
    assert kwargs["registration_direction"] == "t1_to_epi"
