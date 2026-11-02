import csv
import json
from pathlib import Path
from queue import Queue
from threading import Event

import nibabel as nib
import numpy as np
import pytest

import rs_pca_runtime as runtime
from fmri_rt_preproc.processing_mask import PROCESSING_MASK_POLICY, PROCESSING_MASK_POLICY_VERSION


def _setup(tmp_path, metadata=True):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if metadata is not None:
        info = {"regression": {"analysis_space": "epi"},
                "analysis_mask": "whole_brain", "smoothing_fwhm_mm": 0}
        if metadata:
            info["normalization"] = {"mode": "zscore", "reference": "fixed"}
            info["coverage_policy"] = "test_separate_support"
            info["processing_mask"] = {"policy": PROCESSING_MASK_POLICY,
                                       "version": PROCESSING_MASK_POLICY_VERSION,
                                       "interpolation": "linear"}
        (run_dir / "session_metadata.json").write_text(json.dumps(info), encoding="utf-8")
    pca_root = tmp_path / "pca"
    roi = pca_root / "ROI"
    roi.mkdir(parents=True)
    # Preserve the trained PCA normalization. Scoring a calibration zero here
    # would incorrectly produce -50 instead of being unavailable.
    np.savez(roi / "decoder_bundle.npz", voxel_indices=np.array([0, 1]),
             weights=np.array([[1., 0.]], np.float32),
             norm_mean=np.array([100., 0.], np.float32),
             norm_std=np.array([2., 1.], np.float32))
    (roi / "decoder_metadata.json").write_text(json.dumps({
        "upstream_processing": runtime.run_upstream_processing(run_dir, "reg")}), encoding="utf-8")
    condition = runtime.PCACondition("condition1", "X", "ROI", "PC01", "up")
    return run_dir, pca_root, condition


def _volume(run_dir, index, first_value, folder="norm"):
    directory = run_dir / folder
    directory.mkdir(exist_ok=True)
    path = directory / f"vol_{index:05d}_{folder}.nii"
    nib.save(nib.Nifti1Image(np.array([[[first_value, 0.]]], np.float32), np.eye(4)), path)
    return path


def _run(run_dir, pca_root, condition, queue, stop, max_trs):
    decoder_info = json.loads((pca_root / "ROI" / "decoder_metadata.json").read_text())
    reference = runtime.PCAReferenceStats({"ROI_PC01": {"mean": 0., "std": 1.}},
                                          upstream_processing=decoder_info["upstream_processing"])
    runtime.run_realtime_pca_scorer(
        run_dir=run_dir, pca_root=pca_root, condition=condition,
        score_queue=queue, stop_event=stop, max_trs=max_trs,
        volume_kind="reg", normalization="zscore", score_metric="projection",
        reference_stats=reference,
    )


def test_calibration_volumes_publish_nan_and_only_ready_normalized_volume_is_scored(tmp_path, monkeypatch):
    run_dir, pca_root, condition = _setup(tmp_path)
    volumes = [_volume(run_dir, 1, 0.), _volume(run_dir, 2, 0.), _volume(run_dir, 3, 104.)]
    # A competing native-unit reg file must not become the PCA input.
    _volume(run_dir, 3, 9999., "reg")
    (run_dir / "regression_status_rt.csv").write_text(
        "volume_idx,reg_ready,glm_ready,normalization_ready\n1,0,0,0\n2,0,1,0\n3,1,1,1\n",
        encoding="utf-8")
    loaded = []
    real_load = nib.load

    def track_load(path, *args, **kwargs):
        loaded.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", track_load)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: pytest.fail("Unexpected readiness wait"))
    queue = Queue()
    _run(run_dir, pca_root, condition, queue, Event(), 3)
    messages = [queue.get_nowait() for _ in range(3)]
    assert [row["volume_idx"] for row in messages] == [1, 2, 3]
    assert [row["reg_ready"] for row in messages] == [False, False, True]
    for row in messages[:2]:
        for key in ("raw_component_score", "directed_score", "score_z", "feedback_score", "score_raw"):
            assert np.isnan(row[key])
    assert messages[2]["raw_component_score"] == 2.  # Existing trained PCA z-score still applies.
    assert np.isfinite(messages[2]["feedback_score"])
    assert loaded == [volumes[2]]
    with (run_dir / "pca_realtime_scores.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["volume_idx"]) for row in rows] == [1, 2, 3]
    assert np.isnan(float(rows[0]["feedback_score"]))
    assert float(rows[2]["raw_component_score"]) == 2.


@pytest.mark.parametrize("initial_status", [
    None,
    "volume_idx,reg_ready\n",
    "volume_idx,reg_ready\n2,1\n",  # A different volume is not an authorization.
    "volume_idx,reg_ready\n1,1",  # Producer has not completed this row yet.
    "volume_idx,reg_ready\n1,unknown\n",
])
def test_existing_volume_waits_for_matching_complete_readiness_row(tmp_path, monkeypatch, initial_status):
    run_dir, pca_root, condition = _setup(tmp_path)
    _volume(run_dir, 1, 104.)
    status = run_dir / "regression_status_rt.csv"
    if initial_status is not None:
        status.write_text(initial_status, encoding="utf-8")
    waits = []

    def publish_status(_):
        waits.append(True)
        assert len(waits) == 1
        status.write_text("volume_idx,reg_ready\n1,1\n", encoding="utf-8")

    monkeypatch.setattr(runtime.time, "sleep", publish_status)
    queue = Queue()
    _run(run_dir, pca_root, condition, queue, Event(), 1)
    assert len(waits) == 1
    assert queue.qsize() == 1
    assert queue.get_nowait()["raw_component_score"] == 2.


def test_stopped_run_with_missing_readiness_row_does_not_score_existing_volume(tmp_path, monkeypatch):
    run_dir, pca_root, condition = _setup(tmp_path)
    _volume(run_dir, 1, 104.)
    stop = Event()
    stop.set()
    monkeypatch.setattr(runtime.time, "sleep", lambda _: pytest.fail("Stopped run must not wait"))
    monkeypatch.setattr(runtime, "score_pca_scalar", lambda *args, **kwargs: pytest.fail("No readiness row"))
    queue = Queue()
    _run(run_dir, pca_root, condition, queue, stop, 1)
    assert queue.empty()
    assert not (run_dir / "pca_realtime_scores.csv").exists()


@pytest.mark.parametrize("metadata", [None, False])
def test_legacy_run_retains_existing_scoring_without_new_readiness_gate(tmp_path, monkeypatch, metadata):
    run_dir, pca_root, condition = _setup(tmp_path, metadata)
    _volume(run_dir, 1, 104., "reg")
    (run_dir / "regression_status_rt.csv").write_text("volume_idx,reg_ready\n1,0\n", encoding="utf-8")
    monkeypatch.setattr(runtime.time, "sleep", lambda _: pytest.fail("Legacy run must not wait"))
    queue = Queue()
    _run(run_dir, pca_root, condition, queue, Event(), 1)
    message = queue.get_nowait()
    assert message["reg_ready"] is True
    assert message["raw_component_score"] == 2.


def test_incomplete_metadata_waits_before_selecting_or_scoring_volume(tmp_path, monkeypatch):
    run_dir, pca_root, condition = _setup(tmp_path)
    metadata = run_dir / "session_metadata.json"
    complete_metadata = metadata.read_text(encoding="utf-8")
    metadata.write_text('{"normalization":', encoding="utf-8")
    _volume(run_dir, 1, 104.)
    (run_dir / "regression_status_rt.csv").write_text("volume_idx,reg_ready\n1,1\n", encoding="utf-8")
    waits = []

    def finish_metadata(_):
        waits.append(True)
        assert len(waits) == 1
        metadata.write_text(complete_metadata, encoding="utf-8")

    monkeypatch.setattr(runtime.time, "sleep", finish_metadata)
    queue = Queue()
    _run(run_dir, pca_root, condition, queue, Event(), 1)
    assert len(waits) == 1
    assert queue.get_nowait()["raw_component_score"] == 2.
