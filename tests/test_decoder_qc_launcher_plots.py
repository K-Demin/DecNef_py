"""Run actual launcher QC functions without importing scanner or GUI modules."""
import ast
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ["rs_realtime_parallel.py", "rt_nf_events_parallel.py",
             "rt_nf_events_prob_parallel.py", "rt_robot_parallel.py", "rt_psychopy_parallel.py"]


def load_plot(launcher, ready):
    source = ROOT / launcher
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_plot_qc")
    namespace = dict(Path=Path, csv=csv, json=json, np=np, log=logging.getLogger(launcher),
                     _load_reg_ready_map=lambda _: ready)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_plot_qc"]


def write_qc(folder, count=6):
    with (folder / "decoder_qc_rt.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["volume_idx", "reference_correlation",
                                "reference_coverage_fraction", "correlation_status"])
        writer.writeheader()
        for index in range(1, count + 1):
            writer.writerow(dict(volume_idx=index,
                                 reference_correlation=float("nan") if index == 2 else .9 - index * .01,
                                 reference_coverage_fraction=.75 if index == 2 else 1.,
                                 correlation_status="incomplete_coverage" if index == 2 else "ok"))
    (folder / "decoder_qc_reference.json").write_text(json.dumps({
        "reference_kind": "prepared_session_reference", "reference_label": "Prepared session EPI",
        "fixed_qc_voxel_count": 9, "full_decoder_voxel_count": 10,
    }))


def write_scores_and_motion(folder):
    (folder / "scores.csv").write_text("volume_idx,score_raw\n" +
                                      "".join(f"{index},{index / 10}\n" for index in range(1, 7)))
    np.savetxt(folder / "motion_rt.1D", np.arange(36).reshape(6, 6))
    (folder / "session_metadata.json").write_text(json.dumps({"voxel_norm_ref_volumes": 3,
                                                            "pre_trial_scans": 2}))


@pytest.fixture
def figures(monkeypatch):
    saved = []
    original = Figure.savefig
    def capture(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    yield saved
    plt.close("all")


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_warmup_reference_qc_survives_missing_score_stream(launcher, tmp_path, figures):
    from fmri_rt_preproc import run_qc_report
    write_qc(tmp_path, count=3)
    called = []
    original = run_qc_report.write_run_qc_workbook
    def export(folder):
        result = original(folder)
        called.append(result)
        return result
    # Spy through the real workbook writer; this verifies the parent finalizer
    # still exports a short run whose decoder score stream never became ready.
    from unittest.mock import patch
    with patch.object(run_qc_report, "write_run_qc_workbook", export):
        load_plot(launcher, ready=None)(tmp_path)
    assert called and called[0] is not None and Path(called[0]).is_file()
    assert (tmp_path / "qc_scores_motion.png").is_file()
    assert len(figures[-1].axes) == 2  # Correlation plus its coverage-percent axis.
    line = figures[-1].axes[0].lines[0]
    assert list(line.get_xdata()) == [1, 2, 3]
    assert np.isnan(line.get_ydata()[1])
    assert line.get_marker() not in {None, "None", "", " "}  # Isolated finite TR1 is visible.
    assert "fallback" in figures[-1].axes[0].get_title()
    assert list(figures[-1].axes[1].lines[0].get_ydata()) == [100, 75, 100]


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_reference_panel_uses_all_trs_while_score_motion_filters_stay_intact(launcher, tmp_path, figures):
    write_qc(tmp_path)
    write_scores_and_motion(tmp_path)
    ready = {index: index >= 4 for index in range(1, 7)}
    load_plot(launcher, ready)(tmp_path)
    axes = figures[-1].axes
    assert len(axes) == 4  # Three panels plus correlation's coverage-percent axis.
    assert list(axes[0].lines[0].get_xdata()) == [5, 6]
    assert list(axes[1].lines[0].get_xdata()) == [5, 6]
    assert list(axes[2].lines[0].get_xdata()) == [1, 2, 3, 4, 5, 6]
    assert np.isnan(axes[2].lines[0].get_ydata()[1])


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_legacy_run_without_correlation_retains_two_panels(launcher, tmp_path, figures):
    write_scores_and_motion(tmp_path)
    ready = {index: index >= 4 for index in range(1, 7)}
    load_plot(launcher, ready)(tmp_path)
    assert len(figures[-1].axes) == 2


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_workbook_error_does_not_drop_short_run_png(launcher, tmp_path, figures, monkeypatch, caplog):
    from fmri_rt_preproc import run_qc_report
    write_qc(tmp_path, count=3)
    def fail(_):
        raise OSError("workbook locked")
    monkeypatch.setattr(run_qc_report, "write_run_qc_workbook", fail)
    load_plot(launcher, ready=None)(tmp_path)
    assert (tmp_path / "qc_scores_motion.png").is_file()
    assert "workbook locked" in caplog.text
