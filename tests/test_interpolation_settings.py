"""Interpolation settings must not silently change a run or its support policy."""
import json
from types import SimpleNamespace

import numpy as np
import nibabel as nib
import pytest

from rt_global_settings import RegressorSettings, load_regressor_settings, save_regressor_settings
import test_rt_pipeline_parallel  # Headless platform stubs used by pipeline tests.
import rt_pipeline as rt
from test_native_fieldmap import make_pair


def test_json_roundtrip_and_legacy_default(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{}')
    assert load_regressor_settings(path).gpu_resampler_interpolation == "linear"
    settings = RegressorSettings(use_gpu_resampler=True, gpu_resampler_interpolation="LANCZOS")
    save_regressor_settings(path, settings)
    assert load_regressor_settings(path).gpu_resampler_interpolation == "lanczos"


@pytest.mark.parametrize("changes", [
    {"gpu_resampler_interpolation": "cubic"},
    {"gpu_resampler_interpolation": "lanczos", "use_gpu_resampler": False},
    {"gpu_resampler_interpolation": "lanczos", "gpu_resampler_device": "cpu"},
])
def test_invalid_interpolation_is_rejected_before_other_updates(changes):
    settings = RegressorSettings(use_gpu_resampler=True)
    with pytest.raises(ValueError):
        settings.update({"TR": 99, **changes})
    assert settings.TR == 1 and settings.gpu_resampler_interpolation == "linear"


def test_pipeline_does_not_fall_back_to_cpu_linear_for_requested_lanczos(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        use_gpu_resampler=False, gpu_resampler_interpolation="lanczos"))
    with pytest.raises(ValueError, match="requires"):
        rt.maybe_init_gpu_resampler(SimpleNamespace())
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        use_gpu_resampler=True, gpu_resampler_interpolation="lanczos", analysis_space="t1"))
    cfg = SimpleNamespace(trans_dir=tmp_path, decoder_template=tmp_path / "missing.nii",
                          rt_unwarped_analysis_ref_epi=tmp_path / "missing_epi.nii")
    with pytest.raises(FileNotFoundError, match="Cannot initialize Lanczos"):
        rt.maybe_init_gpu_resampler(cfg)


def test_final_support_uses_linear_availability_instead_of_lanczos_intensities(tmp_path, monkeypatch):
    shape = (3, 4, 5)
    reference = tmp_path / "ref.nii"
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), reference)
    mask = np.ones(shape, bool)
    valid = mask.copy()
    valid[..., -2:] = False
    calls = []
    def coverage(support):
        calls.append(support.copy())
        return valid.copy()
    def wrong(*args):
        pytest.fail("Neither sinc signal interpolation nor strict stencil erosion defines final brain support")
    sampler = SimpleNamespace(interpolation="lanczos", resample_coverage=wrong,
                              resample_array=wrong, resample_support=coverage)
    cfg = SimpleNamespace(rt_work_dir=tmp_path, trans_dir=tmp_path, decoder_template=reference)
    handler = SimpleNamespace(gpu_resampler=sampler)
    actual = rt.final_volume_support(cfg, handler, mask, np.eye(4), 1, "t1")
    np.testing.assert_array_equal(actual, valid)
    np.testing.assert_array_equal(calls[0], mask)
    # Identical support can reuse the already computed availability result.
    reused = rt.final_volume_support(cfg, handler, mask, np.eye(4), 1, "t1",
                                    folder_name="support", mapped_support=actual)
    assert len(calls) == 1
    np.testing.assert_array_equal(reused, valid)


def test_run_rejects_switching_interpolation(tmp_path, monkeypatch):
    reference = tmp_path / "ref.nii"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4)), reference)
    cfg = SimpleNamespace(rt_work_dir=tmp_path, fmap_dir=make_pair(tmp_path / "pair"),
                          rt_motion_ref_epi=reference)
    settings = SimpleNamespace(epi_phase_encoding="PA", analysis_space="epi",
                               gpu_resampler_interpolation="linear")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", settings)
    rt.validate_run_provenance(cfg)
    marker = tmp_path / "preprocessing_order.json"
    old = marker.read_text()
    settings.gpu_resampler_interpolation = "lanczos"
    with pytest.raises(ValueError, match="fresh run"):
        rt.validate_run_provenance(cfg)
    assert marker.read_text() == old
