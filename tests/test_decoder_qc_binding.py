"""Existing run QC cannot silently switch reference patterns or CSV schemas."""
import csv
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel  # Native/GUI test doubles.
import rt_pipeline as rt
from rt_global_settings import RegressorSettings


def save(path, values):
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4)), path)
    return path


@pytest.fixture
def bound_qc(tmp_path, monkeypatch):
    settings = RegressorSettings(analysis_space='epi')
    monkeypatch.setattr(rt, 'REGRESSOR_SETTINGS', settings)
    values = np.arange(60).reshape(3, 4, 5) + 100
    native = save(tmp_path / 'native.nii', values)
    decoder = save(tmp_path / 'decoder.nii', np.ones(values.shape))
    training = save(tmp_path / 'training.nii', values ** 1.1)
    roi = save(tmp_path / 'features.nii', np.ones(values.shape))
    settings.decoder_qc_roi_mask = str(roi)
    trans = tmp_path / 'trans'; trans.mkdir()
    cfg = SimpleNamespace(rt_work_dir=tmp_path, trans_dir=trans,
                          rt_unwarped_analysis_ref_epi=native, decoder_template=decoder,
                          enable_scoring=False)
    handler = SimpleNamespace(scorer=SimpleNamespace(mask=np.ones(values.shape, bool)),
                              final_brain_mask=np.ones(values.shape, bool), gpu_resampler=None)
    handler.decoder_qc = rt.prepare_decoder_correlation_qc(cfg, handler)
    rt.record_decoder_correlation(cfg, handler, 1, native, np.ones(values.shape, bool))
    return cfg, handler, settings, training, roi


def test_same_fixed_request_can_restart_without_generated_reference_mtime_binding(bound_qc):
    cfg, handler, _, _, _ = bound_qc
    binding = (cfg.rt_work_dir / 'decoder_qc_request.json').read_bytes()
    handler.decoder_qc = rt.prepare_decoder_correlation_qc(cfg, handler)
    assert (cfg.rt_work_dir / 'decoder_qc_request.json').read_bytes() == binding
    rt.record_decoder_correlation(cfg, handler, 2, cfg.rt_unwarped_analysis_ref_epi,
                                  np.ones(handler.scorer.mask.shape, bool))
    with (cfg.rt_work_dir / 'decoder_qc_rt.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


@pytest.mark.parametrize('changed', ['session_reference', 'training_reference', 'decoder_mask',
                                    'brain_mask', 'explicit_roi', 'missing_binding'])
def test_changed_qc_request_rejects_before_overwriting_run_outputs(bound_qc, changed):
    cfg, handler, settings, training, roi = bound_qc
    outputs = [cfg.rt_work_dir / name for name in ('decoder_qc_reference.json', 'decoder_qc_rt.csv',
               'decoder_qc/session_reference.nii', 'decoder_qc/fixed_roi.nii')]
    original = {p: p.read_bytes() for p in outputs}
    if changed == 'session_reference':
        save(cfg.rt_unwarped_analysis_ref_epi, np.zeros(handler.scorer.mask.shape))
    elif changed == 'training_reference':
        settings.decoder_qc_reference = str(training)
    elif changed == 'decoder_mask':
        handler.scorer.mask[0, 0, 0] = False
    elif changed == 'brain_mask':
        handler.final_brain_mask[0, 0, 0] = False
    elif changed == 'explicit_roi':
        save(roi, np.zeros(handler.scorer.mask.shape))
    else:
        (cfg.rt_work_dir / 'decoder_qc_request.json').unlink()
    with pytest.raises(ValueError, match='fresh run directory'):
        rt.prepare_decoder_correlation_qc(cfg, handler)
    assert all(p.read_bytes() == data for p, data in original.items())


def test_old_status_header_rejects_before_preparation_or_metadata(bound_qc, monkeypatch):
    cfg, _, _, _, _ = bound_qc
    path = cfg.rt_work_dir / 'regression_status_rt.csv'
    path.write_text('volume_idx,fd,reg_ready\n1,0,0\n')
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Must reject old status before preparation/metadata')
    monkeypatch.setattr(rt, 'ensure_fieldmap_only', forbidden)
    monkeypatch.setattr(rt, 'write_session_metadata', forbidden)
    with pytest.raises(ValueError, match='column schema'):
        rt.run_rt_pipeline(cfg)
    with pytest.raises(ValueError, match='column schema'):
        rt.append_regression_status(path, 2, 0, 0, 0, 0, 0, False, False)
    assert path.read_bytes() == before


def test_current_status_header_allows_append_and_empty_file_gets_header(tmp_path):
    path = tmp_path / 'regression_status_rt.csv'; path.touch()
    rt.append_regression_status(path, 1, 0, 0, 0, 0, 0, False, False)
    rt.validate_regression_status_schema(path)
    rt.append_regression_status(path, 2, 0, 0, 0, 0, 0, True, False)
    with path.open() as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == rt.REGRESSION_STATUS_FIELDS and len(rows) == 3
