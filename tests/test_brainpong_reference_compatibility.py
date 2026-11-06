"""BrainPong accepts legacy score references and current gameplay exports."""
import json
from types import SimpleNamespace

import pytest
import test_rt_pipeline_parallel  # headless imports
import rt_pipeline as rt

from test_normalized_score_reference import reference_run


def legacy_run(tmp_path, folder="7"):
    run = tmp_path / "func" / folder
    run.mkdir(parents=True)
    return run, SimpleNamespace(day_root=tmp_path)


@pytest.mark.parametrize("location", ["run", "gameplay", "absolute"])
def test_legacy_reference_paths_preserve_historical_sample_selection(tmp_path, location, caplog):
    folder = "7/reference_gameplay" if location == "gameplay" else "7"
    run, cfg = legacy_run(tmp_path, folder)
    (run / "scores.csv").write_text(
        "volume_idx,score_raw\n1,1000\n2,3\n3,5\n4,nan\n5,inf\n6,-inf\n")
    original_files = {path.name: path.read_bytes() for path in run.iterdir()}
    run_id = str(run) if location == "absolute" else folder

    result = rt.load_reference_score_stats(cfg, run_id)

    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert result["run"] == run_id
    assert result["skipped_first_reg_ready"]
    assert not result["used_reg_ready"]
    assert "Legacy score reference" in caplog.text
    # Compatibility does not fabricate provenance or alter the selected data.
    assert {path.name: path.read_bytes() for path in run.iterdir()} == original_files


def test_basic_legacy_metadata_is_not_mistaken_for_modern_contract(tmp_path, caplog):
    run, cfg = legacy_run(tmp_path)
    (run / "session_metadata.json").write_text(json.dumps({"subject": "88888", "analysis_space": "epi"}))
    (run / "scores.csv").write_text("volume_idx,score_raw\n1,1000\n2,3\n3,5\n")
    result = rt.load_reference_score_stats(cfg, "7")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert "Legacy score reference" in caplog.text


def test_legacy_ready_selection_uses_status_log_and_drops_first_ready(tmp_path):
    run, cfg = legacy_run(tmp_path)
    (run / "scores.csv").write_text(
        "volume_idx,score_raw,reg_ready\n1,2000,1\n2,1000,1\n3,3,1\n4,5,1\n")
    (run / "regression_status_rt.csv").write_text(
        "volume_idx,reg_ready\n1,0\n2,1\n3,1\n4,1\n")
    result = rt.load_reference_score_stats(cfg, "7")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert result["used_reg_ready"] and result["skipped_first_reg_ready"]


def test_legacy_all_false_readiness_retains_old_finite_score_fallback(tmp_path, caplog):
    run, cfg = legacy_run(tmp_path)
    (run / "scores.csv").write_text(
        "volume_idx,score_raw,reg_ready\n1,3,0\n2,5,0\n3,inf,0\n")
    result = rt.load_reference_score_stats(cfg, "7")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert not result["used_reg_ready"]
    assert "falling back to its finite recorded scores" in caplog.text


@pytest.mark.parametrize("modern_metadata", [False, True])
@pytest.mark.parametrize("transient", ["1000", "nan", "inf", "invalid"])
def test_explicit_first_ready_transient_is_removed_once(tmp_path, modern_metadata, transient):
    if modern_metadata:
        parent_run, cfg = reference_run(tmp_path)
        run = parent_run / "reference_gameplay"
        run.mkdir()
        (run / "session_metadata.json").write_bytes((parent_run / "session_metadata.json").read_bytes())
        run_id = "1/reference_gameplay"
    else:
        run, cfg = legacy_run(tmp_path, "7/reference_gameplay")
        run_id = "7/reference_gameplay"
    (run / "reference_selection.json").write_text(json.dumps({
        "status": "ready", "reference_policy": "first_ready_drop",
    }))
    (run / "scores.csv").write_text(
        "volume_idx,score_raw,reg_ready,reference_selection_role\n"
        f"20,{transient},1,first_ready_transient\n21,3,1,gameplay\n22,5,1,gameplay\n")

    result = rt.load_reference_score_stats(cfg, run_id)

    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert result["used_reg_ready"] and result["skipped_first_reg_ready"]


def test_current_gameplay_normalization_ready_keeps_first_ready_sample(tmp_path):
    parent_run, cfg = reference_run(tmp_path)
    run = parent_run / "reference_gameplay"
    run.mkdir()
    (run / "session_metadata.json").write_bytes((parent_run / "session_metadata.json").read_bytes())
    (run / "reference_selection.json").write_text(json.dumps({
        "status": "ready", "reference_policy": "normalization_ready",
    }))
    (run / "scores.csv").write_text(
        "volume_idx,score_raw,reg_ready,reference_selection_role\n"
        "1,1000,0,calibration\n30,3,1,gameplay\n31,5,1,gameplay\n32,inf,1,gameplay\n")
    result = rt.load_reference_score_stats(cfg, "1/reference_gameplay")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert result["used_reg_ready"] and not result["skipped_first_reg_ready"]


def test_whole_current_run_is_not_blocked_by_failed_gameplay_export(tmp_path):
    run, cfg = reference_run(tmp_path)
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n1,1000,0\n2,3,1\n3,5,1\n")
    failed = {"status": "failed", "reason": "No gameplay samples"}
    (run / "reference_selection.json").write_text(json.dumps(failed))
    gameplay = run / "reference_gameplay"
    gameplay.mkdir()
    (gameplay / "reference_selection.json").write_text(json.dumps(failed))
    result = rt.load_reference_score_stats(cfg, "1")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert not result["skipped_first_reg_ready"]


@pytest.mark.parametrize("key", [
    "normalization", "coverage_policy", "processing_mask", "temporal_processing_order",
])
def test_partial_modern_contract_cannot_silently_become_legacy(tmp_path, key):
    run, cfg = legacy_run(tmp_path)
    (run / "session_metadata.json").write_text(json.dumps({key: None}))
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n1,3,1\n2,5,1\n")
    with pytest.raises(ValueError, match="spatial masking/interpolation policy"):
        rt.load_reference_score_stats(cfg, "7")


@pytest.mark.parametrize("metadata", [None, [], "broken"])
def test_invalid_metadata_container_is_not_accepted_as_legacy(tmp_path, metadata):
    run, cfg = legacy_run(tmp_path)
    (run / "session_metadata.json").write_text(json.dumps(metadata))
    (run / "scores.csv").write_text("volume_idx,score_raw\n1,1000\n2,3\n3,5\n")
    with pytest.raises(ValueError, match="Invalid reference session metadata"):
        rt.load_reference_score_stats(cfg, "7")
