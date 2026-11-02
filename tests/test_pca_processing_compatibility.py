"""PCA models and RS references must match the live preprocessing method."""
from copy import deepcopy
import json
import pickle
from queue import Queue
from threading import Event

import nibabel as nib
import numpy as np
import pytest

import rs_pca_runtime as runtime
from fmri_rt_preproc.processing_mask import upstream_processing_spec


def run_metadata():
    return {"normalization": {"mode": "zscore", "reference": "fixed", "ref_volumes": 2},
            "coverage_policy": "separate_brain_processing_observed_support_v3",
            "processing_mask": {"policy": "source_grid_cubic_halo", "version": 1,
                                "interpolation": "linear", "margin_vox": 1},
            "regression": {"analysis_space": "epi"},
            "analysis_mask": "whole_brain", "smoothing_fwhm_mm": 0}


def fixture(tmp_path, modern=True):
    info = run_metadata() if modern else {}
    (tmp_path / "session_metadata.json").write_text(json.dumps(info))
    spec = upstream_processing_spec(info)
    decoder = {"_metadata": {"upstream_processing": deepcopy(spec)}}
    stats = runtime.PCAReferenceStats({"ROI_PC01": {"mean": 3., "std": 1.}},
                                      upstream_processing=deepcopy(spec))
    return decoder, stats


@pytest.mark.parametrize("asset", ["decoder", "reference"])
@pytest.mark.parametrize("change", ["missing", "normalization", "coverage", "mask", "version", "kernel", "stage"])
def test_rejects_unverifiable_and_mismatched_modern_assets(tmp_path, asset, change):
    decoder, stats = fixture(tmp_path)
    spec = (decoder["_metadata"]["upstream_processing"] if asset == "decoder"
            else stats.upstream_processing)
    if change == "missing":
        if asset == "decoder":
            decoder["_metadata"].pop("upstream_processing")
        else:
            stats = dict(stats)  # Plain legacy/in-memory stats carry no provenance.
    elif change == "normalization":
        spec["normalization"]["mode"] = "mean"
    elif change == "coverage":
        spec["coverage_policy"] = "old_support"
    elif change == "mask":
        spec["processing_mask"]["policy"] = "strict_brain_stencil"
    elif change == "version":
        spec["processing_mask"]["version"] += 1
    elif change == "kernel":
        spec["processing_mask"]["interpolation"] = "lanczos"
    else:
        spec["input_stage"] = "raw_mc"
        spec["normalization"] = None
    with pytest.raises(runtime.PCAReferenceMismatchError, match="Rebuild PCA assets"):
        runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                        volume_kind="reg", reference_stats=stats)


def test_acquisition_specific_mask_and_grid_changes_are_allowed(tmp_path):
    decoder, stats = fixture(tmp_path)
    info = run_metadata()
    info["processing_mask"].update(brain_mask={"path": "/new/mask.nii", "sha256": "new"},
                                   margin_vox=0, spatial_resampling=False)
    info["motion_reference"] = "/new/reference.nii"
    info["epi_reference_mapped"] = False
    (tmp_path / "session_metadata.json").write_text(json.dumps(info))
    runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                    volume_kind="reg", reference_stats=stats)


def test_legacy_only_works_when_both_sides_are_legacy(tmp_path):
    decoder, stats = fixture(tmp_path, modern=False)
    runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                    volume_kind="reg", reference_stats=dict(stats))
    decoder["_metadata"]["upstream_normalization"] = {"mode": "zscore"}
    with pytest.raises(runtime.PCAReferenceMismatchError):
        runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                        volume_kind="reg", reference_stats=dict(stats))
    decoder["_metadata"]["upstream_processing"] = upstream_processing_spec(run_metadata())
    with pytest.raises(runtime.PCAReferenceMismatchError):
        runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                        volume_kind="reg", reference_stats=dict(stats))


def test_raw_mc_is_identified_without_claiming_voxel_normalization(tmp_path):
    fixture(tmp_path)
    spec = runtime.run_upstream_processing(tmp_path, "mc")
    assert spec["input_stage"] == "raw_mc" and spec["normalization"] is None
    decoder = {"_metadata": {"upstream_processing": spec}}
    runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path, volume_kind="mc")
    with pytest.raises(runtime.PCAReferenceMismatchError):
        runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path, volume_kind="reg")


def test_reference_provenance_survives_json_csv_and_process_pickling(tmp_path):
    decoder, stats = fixture(tmp_path)
    path = tmp_path / "pca_reference_stats.json"
    runtime.save_reference_stats(path, stats)
    loaded = pickle.loads(pickle.dumps(runtime.load_reference_stats(path)))
    assert loaded == stats and loaded.upstream_processing == stats.upstream_processing
    csv = tmp_path / "scores_pca_all_rois.csv"
    csv.write_text("volume_idx,ROI_PC01\n1,2\n2,4\n")
    csv.with_name("scores_pca_all_rois_metadata.json").write_text(json.dumps({
        "upstream_processing": stats.upstream_processing}))
    from_csv = runtime.load_reference_stats_from_score_csv(csv, ["ROI_PC01"])
    assert from_csv["ROI_PC01"]["mean"] == 3
    runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                    volume_kind="reg", reference_stats=from_csv)
    csv.with_name("scores_pca_all_rois_metadata.json").unlink()
    unverified = runtime.load_reference_stats_from_score_csv(csv, ["ROI_PC01"])
    with pytest.raises(runtime.PCAReferenceMismatchError):
        runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path,
                                        volume_kind="reg", reference_stats=unverified)


def test_normalized_run_with_missing_spatial_metadata_is_not_legacy(tmp_path):
    decoder, _ = fixture(tmp_path)
    (tmp_path / "session_metadata.json").write_text(json.dumps({"normalization": {"mode": "zscore"}}))
    with pytest.raises(runtime.PCAReferenceMismatchError):
        runtime.validate_pca_processing(decoders={"ROI": decoder}, run_dir=tmp_path, volume_kind="reg")


def test_realtime_consumer_rejects_known_mismatch_before_scoring(tmp_path):
    from test_pca_normalization_readiness import _setup, _volume, _run
    run, pca, condition = _setup(tmp_path)
    _volume(run, 1, 104.)
    (run / "regression_status_rt.csv").write_text("volume_idx,reg_ready\n1,1\n")
    path = pca / "ROI" / "decoder_metadata.json"
    info = json.loads(path.read_text())
    info["upstream_processing"]["processing_mask"]["interpolation"] = "lanczos"
    path.write_text(json.dumps(info))
    queue = Queue()
    with pytest.raises(runtime.PCAReferenceMismatchError, match="different upstream preprocessing"):
        _run(run, pca, condition, queue, Event(), 1)
    assert queue.empty()
    assert not (run / "pca_realtime_scores.csv").exists()


@pytest.mark.parametrize("no_reference", [None, {}])
def test_modern_decoder_supports_unreferenced_directed_scores(tmp_path, no_reference):
    from test_pca_normalization_readiness import _setup, _volume
    run, pca, condition = _setup(tmp_path)
    _volume(run, 1, 104.)
    (run / "regression_status_rt.csv").write_text("volume_idx,reg_ready\n1,1\n")
    queue = Queue()
    runtime.run_realtime_pca_scorer(
        run_dir=run, pca_root=pca, condition=condition, score_queue=queue,
        stop_event=Event(), reference_stats=no_reference, volume_kind="reg",
        normalization="zscore", score_metric="projection", max_trs=1)
    row = queue.get_nowait()
    assert row["raw_component_score"] == 2.
    assert row["feedback_score"] == row["directed_score"] == 2.
    assert np.isnan(row["score_z"])


@pytest.mark.parametrize("columns", [{}, {"OTHER_PC01": {"mean": 0., "std": 1.}},
                                     {"ROI_PC01": {"mean": 0., "std": 0.}},
                                     {"ROI_PC01": {"mean": "bad", "std": 1.}}])
def test_explicit_invalid_reference_cannot_silently_become_unreferenced(tmp_path, columns):
    from test_pca_normalization_readiness import _setup
    run, pca, condition = _setup(tmp_path)
    path = tmp_path / "requested_stats.json"
    path.write_text(json.dumps({"columns": columns,
                               "upstream_processing": runtime.run_upstream_processing(run, "reg")}))
    with pytest.raises(runtime.PCAReferenceMismatchError, match="Requested PCA RS reference"):
        runtime.run_realtime_pca_scorer(
            run_dir=run, pca_root=pca, condition=condition, score_queue=Queue(),
            stop_event=Event(), reference_stats_path=path, volume_kind="reg",
            normalization="zscore", score_metric="projection", max_trs=1)


def test_all_roi_producer_validates_and_stamps_reference_statistics(tmp_path):
    import rs_pca_score_all_rois as producer
    run = tmp_path / "run"
    run.mkdir()
    _, stats = fixture(run)
    pca = tmp_path / "pca"
    roi = pca / "ROI"
    roi.mkdir(parents=True)
    np.savez(roi / "decoder_bundle.npz", voxel_indices=np.array([0, 1]),
             weights=np.array([[1., 0.]], np.float32), norm_mean=np.zeros(2),
             norm_std=np.ones(2), explained=np.array([1.]))
    (roi / "decoder_metadata.json").write_text(json.dumps({"upstream_processing": stats.upstream_processing}))
    (run / "norm").mkdir()
    for index, value in enumerate([1., 2., 4.], 1):
        nib.save(nib.Nifti1Image(np.array([[[value, 0.]]], np.float32), np.eye(4)),
                 run / "norm" / f"vol_{index:05d}_norm.nii")
    (run / "regression_status_rt.csv").write_text("volume_idx,reg_ready\n1,1\n2,1\n3,1\n")
    score_root = tmp_path / "scores"
    producer.run_realtime_all_roi_pca_scorer(
        run_dir=run, pca_root=pca, score_root=score_root, score_queue=Queue(), stop_event=Event(),
        volume_kind="reg", score_label="PC01", normalization="none", score_metric="projection",
        target_pc="PC01", top_pcs="all", top_pc_variance=.1, max_trs=3, poll_interval=.01,
        qc_plot_path=None, qc_update_every=0, metadata={})
    loaded = runtime.load_reference_stats(score_root / "pca_reference_stats.json")
    assert loaded["ROI_PC01"]["n"] == 2
    assert loaded.upstream_processing == stats.upstream_processing
    summary = json.loads((score_root / "scores_pca_all_rois_metadata.json").read_text())
    assert summary["upstream_processing"] == stats.upstream_processing
