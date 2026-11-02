"""Calibration acceptance, reuse, and runtime precision guards."""
import json
import importlib
import sys
from types import SimpleNamespace, ModuleType

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc.native_fieldmap import (
    load_calibration, native_paths, ESTIMATOR_RECIPE,
)
from test_native_fieldmap import make_pair


def write_field(pair, derivative, *, spacing=1.0):
    folder, field, _ = native_paths(pair)
    values = np.broadcast_to(
        (np.arange(5, dtype=np.float32) * derivative * spacing)[None, :, None],
        (3, 5, 5),
    ).copy()
    nib.save(nib.Nifti1Image(values, np.eye(4)), field)
    for name in ("AP_mean.nii", "PA_mean.nii"):
        image = nib.load(folder / name)
        affine = image.affine.copy()
        affine[1, 1] = spacing
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj).copy(), affine), folder / name)
    return field


@pytest.mark.parametrize("derivative", [-1.2, -1.0, 1.0, 1.2])
def test_rejects_either_nonpositive_jacobian(tmp_path, derivative):
    pair = make_pair(tmp_path / "pair")
    write_field(pair, derivative)
    with pytest.raises(ValueError, match="nonpositive AP/PA Jacobians"):
        load_calibration(pair)


def test_saved_field_spacing_comes_from_calibration_image(tmp_path):
    pair = make_pair(tmp_path / "pair")
    # Field identity affine says 1 mm, but acquisition PE spacing is 3 mm.
    write_field(pair, .5, spacing=3.)
    assert load_calibration(pair)["estimator_recipe"] == ESTIMATOR_RECIPE


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_saved_field(tmp_path, value):
    pair = make_pair(tmp_path / "pair")
    field = native_paths(pair)[1]
    values = np.zeros((3, 5, 5), np.float32)
    values[0, 0, 0] = value
    nib.save(nib.Nifti1Image(values, np.eye(4)), field)
    with pytest.raises(ValueError, match="nonfinite/invalid"):
        load_calibration(pair)


def test_repeated_load_does_not_rescan_field_and_replacement_is_rechecked(tmp_path, monkeypatch):
    pair = make_pair(tmp_path / "pair")
    field = native_paths(pair)[1]
    original_load = nib.load
    loads = []

    def tracked_load(path, *args, **kwargs):
        if str(path) == str(field.resolve()):
            loads.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", tracked_load)
    for _ in range(10):
        load_calibration(pair)
    assert len(loads) == 1
    write_field(pair, 1.2)
    with pytest.raises(ValueError, match="nonpositive AP/PA Jacobians"):
        load_calibration(pair)
    assert len(loads) == 2


@pytest.fixture
def pipeline_for_pair(monkeypatch):
    """Only fieldmap orchestration is exercised; no Qt/native MC is needed."""
    module_name = "fmri_rt_preproc.pipeline"
    already_imported = module_name in sys.modules
    with monkeypatch.context() as scoped:
        if not already_imported:
            stub = ModuleType("fmri_rt_preproc.RTPSpy_tools.rtp_volreg")
            stub.RtpVolreg = object
            scoped.setitem(sys.modules, stub.__name__, stub)
        module = importlib.import_module(module_name)

        def make(pair):
            pipe = object.__new__(module.FMRIRealtimePreprocessor)
            pipe.fmap_dir = pair
            pipe.fieldmap_method = "pyhysco"
            pipe.cfg = SimpleNamespace(ap_file=pair / "AP.nii", pa_file=pair / "PA.nii")
            return pipe

        try:
            yield make
        finally:
            if not already_imported:
                sys.modules.pop(module_name, None)
                package = sys.modules.get("fmri_rt_preproc")
                if getattr(package, "pipeline", None) is module:
                    delattr(package, "pipeline")


def test_old_recipe_cache_is_rejected_without_overwriting(tmp_path, pipeline_for_pair):
    pair = make_pair(tmp_path / "pair")
    _, field, manifest = native_paths(pair)
    saved = json.loads(manifest.read_text())
    saved.pop("estimator_recipe")
    manifest.write_text(json.dumps(saved))
    before = {p: p.read_bytes() for p in (field, manifest)}
    with pytest.raises(ValueError, match="fresh preparation directory"):
        pipeline_for_pair(pair)._prepare_fieldmap()
    assert before == {p: p.read_bytes() for p in before}


def test_unpublished_field_is_not_silently_reused_or_overwritten(tmp_path, pipeline_for_pair):
    pair = make_pair(tmp_path / "pair")
    _, field, manifest = native_paths(pair)
    manifest.unlink()
    before = field.read_bytes()
    with pytest.raises(ValueError, match="Unpublished or incomplete"):
        pipeline_for_pair(pair)._prepare_fieldmap()
    assert field.read_bytes() == before
    assert not manifest.exists()


def test_invalid_new_estimate_is_not_published(tmp_path, pipeline_for_pair):
    pair = make_pair(tmp_path / "pair")
    folder, field, manifest = native_paths(pair)
    for path in folder.iterdir():
        path.unlink()
    pipe = pipeline_for_pair(pair)
    pipe._run_ap_pa_pyhysco = lambda *args, **kwargs: write_field(pair, 1.2)
    with pytest.raises(ValueError, match="nonpositive AP/PA Jacobians"):
        pipe._prepare_fieldmap()
    assert field.exists()
    assert not manifest.exists()


@pytest.mark.parametrize("polarity", [-1, 1])
def test_runtime_checks_both_polarities(tmp_path, polarity):
    from fmri_rt_preproc.pyhysco_apply import PreloadedPyHyscoApplier

    pair = make_pair(tmp_path / "pair")
    field = write_field(pair, 1.2)
    with pytest.raises(ValueError, match="nonpositive AP/PA Jacobians in runtime precision"):
        PreloadedPyHyscoApplier(native_paths(pair)[0] / "AP_mean.nii", field,
                               phase_encoding_direction=2, polarity=polarity, device="cpu")


def test_valid_runtime_field_preserves_identity_correction(tmp_path):
    from fmri_rt_preproc.pyhysco_apply import PreloadedPyHyscoApplier

    pair = make_pair(tmp_path / "pair")
    folder, field, _ = native_paths(pair)
    applier = PreloadedPyHyscoApplier(folder / "AP_mean.nii", field,
                                    phase_encoding_direction=2, device="cpu")
    volume = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    np.testing.assert_allclose(applier.apply_volume(volume), volume, atol=1e-5)


def test_runtime_rejects_positive_saved_jacobian_that_rounds_to_zero(tmp_path):
    from fmri_rt_preproc.pyhysco_apply import PreloadedPyHyscoApplier

    pair = make_pair(tmp_path / "pair")
    folder, field, _ = native_paths(pair)
    values = np.broadcast_to(
        (np.arange(5, dtype=np.float64) * (1 - 1e-9))[None, :, None], (3, 5, 5))
    nib.save(nib.Nifti1Image(values, np.eye(4)), field)
    # The stored double-precision field is strictly feasible mathematically.
    load_calibration(pair)
    # Executing this field in the pipeline's float32 precision is not feasible.
    with pytest.raises(ValueError, match="runtime precision"):
        PreloadedPyHyscoApplier(folder / "AP_mean.nii", field,
                               phase_encoding_direction=2, device="cpu")
