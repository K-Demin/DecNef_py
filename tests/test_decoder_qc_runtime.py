"""Reference preparation uses the same final grid without altering the scorer."""
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel  # headless native-library doubles
import rt_pipeline as rt
from rt_global_settings import RegressorSettings


def save(path, data, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32),
                            np.eye(4) if affine is None else affine), path)
    return path


@pytest.mark.parametrize("space", ["epi", "t1", "mni"])
@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("training_reference", [False, True])
def test_reference_roi_and_every_tr_export(tmp_path, monkeypatch, space, gpu, training_reference):
    settings = RegressorSettings(analysis_space=space)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", settings)
    values = np.arange(60, dtype=np.float32).reshape(3, 4, 5) + 100
    native = save(tmp_path / "native.nii", values)
    decoder = save(tmp_path / "decoder.nii", np.ones(values.shape))
    trans = tmp_path / "trans"
    trans.mkdir()
    (trans / "epi2t1_Composite.h5").touch()
    mni = trans / "t1_to_mni.h5"
    mni.touch()
    monkeypatch.setattr(rt, "t1_to_mni_transform", lambda cfg: mni)
    cfg = SimpleNamespace(rt_work_dir=tmp_path, trans_dir=trans,
                          rt_unwarped_analysis_ref_epi=native,
                          decoder_template=decoder)
    brain = np.ones(values.shape, dtype=bool)
    brain[0, :, :] = False
    mask = np.ones(values.shape, dtype=bool)
    mask[:, 0, :] = False

    def map_values(data):
        return data.copy() if space == "epi" else np.flip(data, axis=1).copy()

    def resample(source, output):
        save(output, map_values(nib.load(source).get_fdata()))

    commands = []

    def run(command):
        commands.append(command)
        assert command[0] == "antsApplyTransforms"
        resample(command[command.index("-i") + 1], command[command.index("-o") + 1])

    monkeypatch.setattr(rt, "run", run)
    handler = SimpleNamespace(scorer=SimpleNamespace(mask=mask), final_brain_mask=brain,
        gpu_resampler=SimpleNamespace(resample_support=map_values,
                                     resample_nifti_to_nifti=resample) if gpu else None)
    if training_reference:
        # Same coordinates, independent image intensities.
        reference = save(tmp_path / "training_mean.nii", map_values(values) ** 1.2)
        settings.decoder_qc_reference = str(reference)
    else:
        reference = None
    explicit_roi = brain.copy()
    explicit_roi[:, :, -1] = False
    settings.decoder_qc_roi_mask = str(save(tmp_path / "features.nii", explicit_roi))
    (tmp_path / "session_metadata.json").write_text('{"existing": 42}')
    handler.decoder_qc = rt.prepare_decoder_correlation_qc(cfg, handler)
    expected_roi = mask & brain & explicit_roi
    np.testing.assert_array_equal(handler.decoder_qc.roi, expected_roi)
    np.testing.assert_array_equal(handler.scorer.mask, mask)
    meta = json.loads((tmp_path / "decoder_qc_reference.json").read_text())
    assert meta["reference_kind"] == ("decoder_training_mean" if training_reference else "prepared_session_reference")
    assert meta["full_decoder_voxel_count"] == int(mask.sum())
    assert meta["fixed_qc_voxel_count"] == int(expected_roi.sum())
    assert json.loads((tmp_path / "session_metadata.json").read_text())["existing"] == 42
    np.testing.assert_array_equal(nib.load(tmp_path / "decoder_qc/fixed_roi.nii").get_fdata(), expected_roi)
    current = map_values(values)
    current_path = save(tmp_path / "current.nii", current)
    support = brain.copy()
    result = rt.record_decoder_correlation(cfg, handler, 1, current_path, support)
    ref_data = nib.load(reference).get_fdata() if reference else map_values(values)
    assert result["reference_correlation"] == pytest.approx(np.corrcoef(
        ref_data[expected_roi], current[expected_roi])[0, 1])
    assert result["reference_coverage_fraction"] == 1
    # Later coverage loss is not hidden by reducing the fixed ROI.
    lost = tuple(np.argwhere(expected_roi)[0])
    support[lost] = False
    result = rt.record_decoder_correlation(cfg, handler, 2, current_path, support)
    assert result["correlation_status"] == "incomplete_coverage"
    assert np.isnan(result["reference_correlation"])
    np.testing.assert_array_equal(handler.decoder_qc.roi, expected_roi)
    with (tmp_path / "decoder_qc_rt.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["volume_idx"] for row in rows] == ["1", "2"]
    if space == "epi" or gpu:
        assert not commands


def test_qc_disabled_or_nonvoxel_workflow_does_no_spatial_work(tmp_path, monkeypatch):
    cfg = SimpleNamespace(rt_work_dir=tmp_path)
    handler = SimpleNamespace(scorer=None)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", RegressorSettings())
    assert rt.prepare_decoder_correlation_qc(cfg, handler) is None
    result = rt.record_decoder_correlation(cfg, handler, 1, None, None)
    assert result["correlation_status"] == "no_voxel_decoder"
    handler.scorer = object()
    rt.REGRESSOR_SETTINGS.enable_decoder_correlation_qc = False
    assert rt.prepare_decoder_correlation_qc(cfg, handler) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("settings", [
    {"enable_decoder_correlation_qc": "false"},
    {"decoder_qc_reference": ""}, {"decoder_qc_roi_mask": 10},
])
def test_invalid_qc_settings_do_not_mutate_profile(settings):
    profile = RegressorSettings()
    original = vars(profile).copy()
    with pytest.raises(ValueError):
        profile.update(settings)
    assert vars(profile) == original


def test_orderly_stop_exports_after_last_active_csv_writer(tmp_path, monkeypatch):
    entered, release, exported = threading.Event(), threading.Event(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)

    def last_volume():
        entered.set()
        assert release.wait(timeout=5)
        (tmp_path / "last_volume.csv").write_text("complete")

    def export(folder):
        assert (folder / "last_volume.csv").read_text() == "complete"
        exported.set()

    monkeypatch.setattr(rt, "write_run_qc_workbook", export)
    executor.submit(last_volume)
    assert entered.wait(timeout=5)
    handler = SimpleNamespace(cfg=SimpleNamespace(rt_work_dir=tmp_path),
                              biopac_receiver=None, _executor=executor,
                              volume_streamer=SimpleNamespace(stop=lambda: None))
    try:
        rt.DICOMHandler.stop(handler)
        assert not exported.is_set()
    finally:
        release.set()
        if hasattr(handler, "_qc_export_thread"):
            handler._qc_export_thread.join(timeout=5)
        executor.shutdown(wait=True)
    assert exported.is_set()
