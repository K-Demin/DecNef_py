"""Presentation duration must not change the feature-calibration reference."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmri_rt_preproc.normalization_timing import normalization_startup_hint, first_trial_window_volumes
from rt_global_settings import load_regressor_settings


@pytest.mark.parametrize("reference", ["fixed", "running"])
@pytest.mark.parametrize("mode,regression,expected", [("zscore", True, 60), ("mean", True, 31),
                                                      ("zscore", False, 31), ("mean", False, 31)])
def test_startup_bound_includes_prior_calibration_values(mode, regression, expected, reference):
    settings = SimpleNamespace(voxel_norm_mode=mode, voxel_norm_reference=reference,
                               voxel_norm_ref_volumes=30, enable_motion_regression=regression)
    assert normalization_startup_hint(settings)["first_score_volume_lower_bound"] == expected


def test_extending_presentation_baseline_moves_trial_window_without_changing_calibration():
    settings = SimpleNamespace(voxel_norm_mode="zscore", voxel_norm_reference="fixed",
                               voxel_norm_ref_volumes=30, enable_motion_regression=True)
    timing = dict(skip_first_trs=10, iti_trs=3, cue_trs=4, scans_trs=3, score_delay=3)
    assert first_trial_window_volumes(baseline_trs=30, **timing) == (47, 50)
    assert first_trial_window_volumes(baseline_trs=60, **timing) == (77, 80)
    assert normalization_startup_hint(settings)["first_score_volume_lower_bound"] == 60
    assert settings.voxel_norm_ref_volumes == 30


@pytest.mark.parametrize("filename", ["rt_nf_events_parallel.py", "rt_nf_events_prob_parallel.py",
                                      "rs_realtime_parallel.py", "rs_pca_score_all_rois.py",
                                      "rt_robot_parallel.py", "rt_psychopy_parallel.py"])
def test_launcher_payload_preserves_json_window_when_cli_baseline_is_longer(tmp_path, filename):
    """Execute actual argument/payload statements without starting scanner/UI code."""
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"voxel_norm_mode": "zscore", "voxel_norm_reference": "fixed",
                                   "voxel_norm_ref_volumes": 30}), encoding="utf-8")
    settings = load_regressor_settings(profile)
    tree = ast.parse((Path(__file__).resolve().parents[1] / filename).read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add_argument"
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == "--baseline-trs"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), {"parser": parser})
            args = parser.parse_args(["--baseline-trs", "120"])
            break
    else:
        args = SimpleNamespace(baseline_trs=120)  # These launchers use other presentation controls.
    for key, value in dict(biopac_enable=False, biopac_host="localhost", biopac_port=1,
                           biopac_timeout=.1, biopac_phys_reg="None", biopac_handshake=False,
                           biopac_start_online=False, biopac_mode="file", biopac_file=None,
                           biopac_poll=.1, skip_first_trs=10, rt_max_scan_length=2000,
                           max_workers=2, pca_space="t1").items():
        setattr(args, key, value)
    payload_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                base = target.value if isinstance(target, ast.Subscript) else target
                if isinstance(base, ast.Name) and base.id == "settings_payload":
                    payload_nodes.append(node)
                    break
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == "settings_payload"):
            payload_nodes.append(node)
    assert payload_nodes, "The launcher's child configuration must be exercised"
    namespace = dict(REGRESSOR_SETTINGS=settings, args=args, baseline_trs=args.baseline_trs,
                     Path=Path, intercept=.7, biopac_output=None)
    exec(compile(ast.Module(body=sorted(payload_nodes, key=lambda n: n.lineno), type_ignores=[]), filename, "exec"), namespace)
    assert namespace["settings_payload"]["voxel_norm_ref_volumes"] == 30
    assert settings.voxel_norm_ref_volumes == 30


@pytest.mark.parametrize("requested", [0., 45., 75.])
def test_robot_warns_without_extending_requested_lead(requested):
    filename = Path(__file__).resolve().parents[1] / "rt_robot_parallel.py"
    tree = ast.parse(filename.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.If)):
            body = node.body
            for index, statement in enumerate(body):
                if (isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "rest_lead_s"):
                    # The production lead block ends at child-payload construction.
                    end = next(i for i in range(index + 1, len(body))
                               if isinstance(body[i], ast.Assign) and isinstance(body[i].targets[0], ast.Name)
                               and body[i].targets[0].id == "settings_payload")
                    selected = body[index:end]
                    break
            else:
                continue
            break
    warnings = []
    namespace = dict(args=SimpleNamespace(rest_lead_sec=requested),
                     REGRESSOR_SETTINGS=SimpleNamespace(TR=1., voxel_norm_mode="zscore",
                         voxel_norm_reference="fixed", voxel_norm_ref_volumes=30, enable_motion_regression=True),
                     normalization_startup_hint=normalization_startup_hint,
                     log=SimpleNamespace(warning=lambda *args: warnings.append(args), info=lambda *args: None))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(filename), "exec"), namespace)
    assert namespace["rest_lead_s"] == requested
    assert bool(warnings) == (requested < 60)
