"""Score-reference normalization differences warn without blocking live runs."""
import json
from types import SimpleNamespace

import pytest
import test_rt_pipeline_parallel  # headless imports
import rt_pipeline as rt


def reference_run(tmp_path):
    run = tmp_path / "func" / "1"
    run.mkdir(parents=True)
    metadata = {
        "normalization": rt.normalization_spec(),
        "coverage_policy": rt.COVERAGE_POLICY,
        "processing_mask": {
            "policy": rt.FINAL_PROCESSING_MASK_POLICY,
            "version": rt.FINAL_PROCESSING_MASK_POLICY_VERSION,
            "interpolation": getattr(rt.REGRESSOR_SETTINGS, "gpu_resampler_interpolation", "linear"),
            "processing_order": list(rt.FINAL_PROCESSING_ORDER),
            "nuisance_source": rt.FINAL_NUISANCE_SOURCE,
        },
    }
    (run / "session_metadata.json").write_text(json.dumps(metadata))
    return run, SimpleNamespace(day_root=tmp_path)


@pytest.mark.parametrize("change", [
    "missing", "mode", "reference", "ref_volumes", "version", "glm_input_units", "zscore_startup",
])
def test_reference_with_unverified_or_different_voxel_units_warns_and_loads(tmp_path, change, caplog):
    run, cfg = reference_run(tmp_path)
    path = run / "session_metadata.json"
    metadata = json.loads(path.read_text())
    spec = rt.normalization_spec()
    if change == "missing":
        spec = None
    else:
        spec[change] = "different"
    metadata["normalization"] = spec
    path.write_text(json.dumps(metadata))
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n1,3,1\n2,5,1\n")
    result = rt.load_reference_score_stats(cfg, "1")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert "Reference voxel normalization differs or is unrecorded" in caplog.text
    assert "Recorded=" in caplog.text and "current=" in caplog.text


def test_use_first_ready_score_and_exclude_calibration_and_nonfinite(tmp_path):
    run, cfg = reference_run(tmp_path)
    (run / "scores.csv").write_text(
        "volume_idx,score_raw,reg_ready\n1,1000,0\n2,3,1\n3,5,1\n4,inf,1\n")
    result = rt.load_reference_score_stats(cfg, "1")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert not result["skipped_first_reg_ready"]


def test_reference_from_different_calibration_window_warns_and_loads(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        voxel_norm_mode="zscore", voxel_norm_reference="fixed", voxel_norm_ref_volumes=30,
        enable_motion_regression=True, analysis_space="epi"))
    run, cfg = reference_run(tmp_path)
    path = run / "session_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["normalization"].pop("zscore_startup")
    metadata["normalization"]["version"] = 1
    path.write_text(json.dumps(metadata))
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n60,3,1\n61,5,1\n")
    result = rt.load_reference_score_stats(cfg, "1")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
    assert "Reference voxel normalization differs or is unrecorded" in caplog.text


@pytest.mark.parametrize("rows", ["1,1000,0\n2,2000,0\n", "1,1000,\n2,2000,\n"])
def test_never_fallback_to_calibration_or_unknown_readiness(tmp_path, rows):
    run, cfg = reference_run(tmp_path)
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n" + rows)
    assert rt.load_reference_score_stats(cfg, "1") is None


@pytest.mark.parametrize("change", [
    "missing_coverage", "old_coverage", "missing_processing", "invalid_processing",
    "missing_policy", "old_policy", "missing_version", "old_version",
    "missing_interpolation", "different_interpolation",
    "missing_processing_order", "different_processing_order",
    "missing_nuisance_source", "different_nuisance_source",
])
def test_reject_reference_with_old_or_unrecorded_spatial_processing(tmp_path, change):
    run, cfg = reference_run(tmp_path)
    path = run / "session_metadata.json"
    metadata = json.loads(path.read_text())
    processing = metadata["processing_mask"]
    if change == "missing_coverage":
        metadata.pop("coverage_policy")
    elif change == "old_coverage":
        metadata["coverage_policy"] = "mc_fov_v1"
    elif change == "missing_processing":
        metadata.pop("processing_mask")
    elif change == "invalid_processing":
        metadata["processing_mask"] = "source_grid_cubic_halo"
    elif change.startswith("missing_"):
        processing.pop(change.removeprefix("missing_"))
    elif change == "old_policy":
        processing["policy"] = "strict_brain_stencil"
    elif change == "old_version":
        processing["version"] -= 1
    elif change == "different_processing_order":
        processing["processing_order"] = ["denoising", "normalization", "spatial_transform", "smoothing"]
    elif change == "different_nuisance_source":
        processing["nuisance_source"] = "smoothed_final_grid"
    else:
        processing["interpolation"] = (
            "lanczos" if processing["interpolation"] == "linear" else "linear")
    path.write_text(json.dumps(metadata))
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n1,3,1\n2,5,1\n")
    with pytest.raises(ValueError, match="spatial masking/interpolation policy.*fresh RS reference"):
        rt.load_reference_score_stats(cfg, "1")


@pytest.mark.parametrize("margin,spatial", [(0, False), (1, True), (3, True)])
def test_reference_allows_different_acquisition_masks_and_grids(tmp_path, margin, spatial):
    run, cfg = reference_run(tmp_path)
    path = run / "session_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["processing_mask"].update({
        "brain_mask": {"path": "/other/session/brain.nii", "sha256": "different-source"},
        "margin_vox": margin,
        "spatial_resampling": spatial,
        "brain_voxels": 100,
        "processing_voxels": 150,
    })
    metadata["motion_reference"] = "/other/session/reference.nii"
    metadata["epi_reference_mapped"] = spatial
    path.write_text(json.dumps(metadata))
    (run / "scores.csv").write_text("volume_idx,score_raw,reg_ready\n1,3,1\n2,5,1\n")
    result = rt.load_reference_score_stats(cfg, "1")
    assert (result["mean"], result["std"], result["n"]) == (4, 1, 2)
