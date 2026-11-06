"""Pure score-log integration checks; no GUI, registration or native MRI library."""
import ast
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc import reference_pool as pool
from fmri_rt_preproc.native_fieldmap import PIPELINE_ORDER
from fmri_rt_preproc.offline_assets import file_identity
from fmri_rt_preproc.processing_mask import (FINAL_PROCESSING_MASK_POLICY,
    FINAL_PROCESSING_MASK_POLICY_VERSION, FINAL_NUISANCE_SOURCE, FINAL_PROCESSING_ORDER,
    canonical_final_processing_mask)
from rt_global_settings import RegressorSettings


def runtime(settings):
    path = Path(__file__).resolve().parents[1] / "rt_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"COVERAGE_POLICY", "TEMPORAL_PROCESSING_ORDER"}:
                    constants[target.id] = ast.literal_eval(node.value)
    namespace = dict(np=np, Path=Path, json=json, csv=csv, Optional=Optional,
        RTSessionConfig=object, REGRESSOR_SETTINGS=settings,
        log=logging.getLogger("reference-pool-test"), **constants,
        FINAL_PROCESSING_MASK_POLICY=FINAL_PROCESSING_MASK_POLICY,
        FINAL_PROCESSING_MASK_POLICY_VERSION=FINAL_PROCESSING_MASK_POLICY_VERSION,
        FINAL_NUISANCE_SOURCE=FINAL_NUISANCE_SOURCE,
        FINAL_PROCESSING_ORDER=FINAL_PROCESSING_ORDER, PIPELINE_ORDER=PIPELINE_ORDER,
        canonical_final_processing_mask=canonical_final_processing_mask)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"normalization_spec", "load_reference_score_stats"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def native_runs(tmp_path):
    settings = RegressorSettings(analysis_space="t1", enable_biopac_physio=False,
                                 smoothing_fwhm_mm=0, analysis_mask="whole_brain")
    rt = runtime(settings)
    decoder = tmp_path / "decoder.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), decoder)
    sources = []
    for number, values in ((4, (2., 6.)), (7, (14., 18.))):
        folder = tmp_path / "data" / "sub-55555" / "historical" / "func" / str(number)
        folder.mkdir(parents=True)
        metadata = {"subject": "55555", "run": str(number), "decoder_template": str(decoder),
            "decoder_roi_txt": None, "normalization": rt.normalization_spec(),
            "coverage_policy": rt.COVERAGE_POLICY, "preprocessing_order": rt.PIPELINE_ORDER,
            "temporal_processing_order": rt.TEMPORAL_PROCESSING_ORDER,
            "processing_mask": {"policy": FINAL_PROCESSING_MASK_POLICY,
                "version": FINAL_PROCESSING_MASK_POLICY_VERSION, "nuisance_source": FINAL_NUISANCE_SOURCE,
                "processing_order": list(FINAL_PROCESSING_ORDER), "interpolation": "linear",
                "brain_mask": {"path": f"/acquisition-{number}/mask.nii", "sha256": str(number)},
                "margin_vox": number},
            "tr": settings.TR, "smoothing_fwhm_mm": settings.smoothing_fwhm_mm,
            "analysis_mask": settings.analysis_mask, "analysis_mask_space": settings.analysis_mask_space,
            "analysis_mask_file": None, "motion_reference": f"/reference-{number}.nii",
            "fieldmap_dir": f"/pair-{number}",
            "regression": {key: getattr(settings, key) for key in pool.REGRESSION_KEYS},
            "biopac": {"enabled": False, "phys_reg": settings.biopac_phys_reg}}
        write_json(folder / "session_metadata.json", metadata)
        rows = [{"volume_idx": 1, "score_raw": float("nan"), "reg_ready": 0},
                {"volume_idx": 2, "score_raw": values[0], "reg_ready": 1},
                {"volume_idx": 3, "score_raw": values[1], "reg_ready": 1}]
        write_csv(folder / "scores.csv", rows)
        write_csv(folder / "regression_status_rt.csv", [{"volume_idx": row["volume_idx"],
            "reg_ready": row["reg_ready"]} for row in rows])
        sources.append(folder)
    kwargs = dict(expected_volumes=[3, 3], settings=settings, subject="55555",
                  decoder_template=decoder, runtime=rt)
    return sources, sources[0].parent / "pooled_reference", kwargs


def test_pool_roundtrips_through_ordinary_loader_preserving_run_displacement(native_runs):
    runs, output, kwargs = native_runs
    stats = pool.pool_native_references(runs, output, **kwargs)
    loaded = kwargs["runtime"].load_reference_score_stats(SimpleNamespace(day_root=Path("unused")), str(output))
    assert (stats["mean"], stats["n"], stats["ddof"]) == (10., 4, 0)
    assert stats["std"] == pytest.approx(np.std([2., 6., 14., 18.], ddof=0))
    assert stats["std"] > 2.  # Averaging per-run SD or runwise scalar z-scoring would lose this.
    assert (loaded["mean"], loaded["std"], loaded["n"]) == (stats["mean"], stats["std"], 4)
    assert loaded["skipped_first_reg_ready"] is False
    with (output / "scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["volume_idx"]) for row in rows] == list(range(1, 7))
    assert [int(row["source_volume_idx"]) for row in rows] == [1, 2, 3, 1, 2, 3]
    assert [int(row["reg_ready"]) for row in rows] == [0, 1, 1, 0, 1, 1]
    metadata = json.loads((output / "session_metadata.json").read_text())
    assert metadata["processing_mask"]["nuisance_source"] == FINAL_NUISANCE_SOURCE
    assert "offline_score_adapter" not in metadata
    assert "fieldmap_dir" not in metadata and "motion_reference" not in metadata
    assert [source["provenance_scope"] for source in metadata["reference_pool"]["sources"]] == [
        "ordinary_RS_metadata_only", "ordinary_RS_metadata_only"]
    assert "Infinity" not in (output / "session_metadata.json").read_text()
    for folder in runs:
        assert (folder / "scores.csv").is_file()  # Source run layout is retained.


@pytest.mark.parametrize("table", ["scores.csv", "regression_status_rt.csv"])
@pytest.mark.parametrize("problem", ["missing", "duplicate", "extra"])
def test_refuse_incomplete_or_ambiguous_source_TRs(native_runs, table, problem):
    runs, output, kwargs = native_runs
    path = runs[0] / table
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if problem == "missing": rows.pop()
    elif problem == "duplicate": rows.append(rows[-1].copy())
    else: rows.append({**rows[-1], "volume_idx": 4})
    write_csv(path, rows)
    with pytest.raises(ValueError, match="TRs|Duplicate"):
        pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()


def test_refuse_disagreeing_readiness_before_export(native_runs):
    runs, output, kwargs = native_runs
    write_csv(runs[0] / "regression_status_rt.csv", [
        {"volume_idx": 1, "reg_ready": 0}, {"volume_idx": 2, "reg_ready": 0},
        {"volume_idx": 3, "reg_ready": 1}])
    with pytest.raises(ValueError):
        pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["subject", "smoothing", "nuisance", "policy", "order", "decoder"])
def test_refuse_changed_subject_method_or_model(native_runs, problem, tmp_path):
    runs, output, kwargs = native_runs
    path = runs[1] / "session_metadata.json"
    metadata = json.loads(path.read_text())
    if problem == "subject": metadata["subject"] = "other"
    elif problem == "smoothing": metadata["smoothing_fwhm_mm"] = 6.
    elif problem == "nuisance": metadata["regression"]["use_vent"] = False
    elif problem == "policy": metadata["processing_mask"]["nuisance_source"] = "final_t1_unsmoothed_fmriprep"
    elif problem == "order": metadata["temporal_processing_order"] = "old_order"
    else:
        other = tmp_path / "different_decoder.nii.gz"
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), 2., np.float32), np.eye(4)), other)
        metadata["decoder_template"] = str(other)
    write_json(path, metadata)
    with pytest.raises(ValueError):
        pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()


def test_pool_normalization_metadata_difference_warns_without_blocking(native_runs, caplog):
    runs, output, kwargs = native_runs
    path = runs[1] / "session_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["normalization"]["ref_volumes"] = 60
    write_json(path, metadata)

    result = pool.pool_native_references(runs, output, **kwargs)

    assert result["n"] == 4
    assert (output / "scores.csv").is_file()
    assert "Reference voxel normalization differs or is unrecorded" in caplog.text
    assert "Recorded=" in caplog.text and "current=" in caplog.text
    # Selecting that reference must not rewrite its source metadata.
    assert json.loads(path.read_text())["normalization"]["ref_volumes"] == 60


def test_existing_pool_is_immutable(native_runs):
    runs, output, kwargs = native_runs
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing")
    with pytest.raises(FileExistsError):
        pool.pool_native_references(runs, output, **kwargs)
    assert sentinel.read_text() == "existing"


@pytest.mark.parametrize("problem", ["duplicate_run", "missing_count", "fractional_count", "boolean_count"])
def test_pool_request_cannot_duplicate_or_miscount_calibration_samples(native_runs, problem):
    runs, output, kwargs = native_runs
    if problem == "duplicate_run": runs = [runs[0], runs[0]]
    elif problem == "missing_count": kwargs["expected_volumes"] = [3]
    elif problem == "fractional_count": kwargs["expected_volumes"] = [3., 3]
    else: kwargs["expected_volumes"] = [True, 3]
    with pytest.raises(ValueError):
        pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["no_ready", "constant"])
def test_refuse_unusable_score_distributions(native_runs, problem):
    runs, output, kwargs = native_runs
    rows = [{"volume_idx": n, "score_raw": 3., "reg_ready": int(problem != "no_ready")}
            for n in (1, 2, 3)]
    write_csv(runs[0] / "scores.csv", rows)
    write_csv(runs[0] / "regression_status_rt.csv", [{"volume_idx": row["volume_idx"],
        "reg_ready": row["reg_ready"]} for row in rows])
    with pytest.raises(ValueError, match="usable score distribution"):
        pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()


def receipt(folder, kwargs):
    return {"version": 1, "status": "complete", "subject": "55555", "run": folder.name,
            "expected_volumes": 3, "settings": kwargs["settings"].to_json_dict(),
            "decoder": file_identity(kwargs["decoder_template"]), "roi_txt": None,
            "inputs_snapshot": [{"path": "/scanner/001_000004_000001.dcm", "size": 100, "mtime_ns": 1}],
            "inputs_unchanged": True, "processing_error_count": 0}


def test_complete_replay_receipts_are_hashed_in_pool(native_runs):
    runs, output, kwargs = native_runs
    for folder in runs:
        write_json(folder / "recorded_reference.json", receipt(folder, kwargs))
    pool.pool_native_references(runs, output, **kwargs)
    sources = json.loads((output / "session_metadata.json").read_text())["reference_pool"]["sources"]
    assert all(source["provenance_scope"] == "completed_recorded_replay" for source in sources)
    assert all(source["recorded_replay_receipt"]["sha256"] == file_identity(
        Path(source["run_dir"]) / "recorded_reference.json")["sha256"] for source in sources)


def test_wrapper_grouped_DICOM_snapshot_schema_is_accepted(native_runs):
    runs, output, kwargs = native_runs
    for folder in runs:
        value = receipt(folder, kwargs)
        record = value["inputs_snapshot"][0]
        value["inputs_snapshot"] = {folder.name: [dict(record)] * 3,
                                    "2": [dict(record)], "3": [dict(record)]}
        value.update(worker_exit_code=0, decoder_unchanged=True)
        write_json(folder / "recorded_reference.json", value)
    assert pool.pool_native_references(runs, output, **kwargs)["n"] == 4


@pytest.mark.parametrize("problem", ["running", "changed_inputs", "errors", "count", "settings", "model"])
def test_receipt_cannot_certify_failed_or_different_replay(native_runs, problem):
    runs, output, kwargs = native_runs
    value = receipt(runs[0], kwargs)
    if problem == "running": value["status"] = "running"
    elif problem == "changed_inputs": value["inputs_unchanged"] = False
    elif problem == "errors": value["processing_error_count"] = 1
    elif problem == "count": value["expected_volumes"] = 2
    elif problem == "settings": value["settings"]["skip_first_trs"] += 1
    else: value["decoder"]["sha256"] = "b" * 64
    write_json(runs[0] / "recorded_reference.json", value)
    with pytest.raises(ValueError):
        pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()
