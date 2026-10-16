"""Registration consumers keep canonical directions and invalidate derived grids."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import test_rt_pipeline_parallel as headless  # Install the shared headless RTPSpy stubs.
import rt_pipeline as rt
import gpu_ants_like_resampler as gpu
from fmri_rt_preproc.analysis_mask import mask_transforms
from test_native_fieldmap import make_pair


def _image(path, shape=(4, 5, 6), affine=None):
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), affine), path)
    return path


def _assets(tmp_path):
    trans = tmp_path / "trans"
    trans.mkdir()
    (tmp_path / "anat").mkdir()
    forward = trans / "epi2t1_Composite.h5"
    inverse = trans / "epi2t1_InverseComposite.h5"
    mni = tmp_path / "anat" / "warp_T1_to_MNI_synth.nii"
    for path in (forward, inverse, mni):
        path.write_text("0")  # Synthetic pull translations, interpreted by _fake_ants.
    moving = _image(trans / "epi_unwarped_mean.nii")
    fixed = _image(tmp_path / "decoder.nii")
    return SimpleNamespace(trans_dir=trans, subject_root=tmp_path, decoder_template=fixed,
                           rt_unwarped_analysis_ref_epi=moving), forward, inverse, mni


def _fake_ants(monkeypatch):
    calls = []

    def run(cmd):
        calls.append(cmd)
        source = nib.load(cmd[cmd.index("-i") + 1])
        fixed = nib.load(cmd[cmd.index("-r") + 1])
        xyz = np.indices(fixed.shape[:3], dtype=float).reshape(3, -1)
        physical = fixed.affine @ np.vstack([xyz, np.ones(xyz.shape[1])])
        sample = (np.linalg.inv(source.affine) @ physical)[:3]
        transforms = [Path(cmd[i + 1]) for i, value in enumerate(cmd) if value == "-t"]
        sample[0] += sum(float(path.read_text()) for path in transforms)
        data = map_coordinates(source.get_fdata(), sample, order=1, mode="constant").reshape(fixed.shape[:3])
        nib.save(nib.Nifti1Image(data.astype(np.float32), fixed.affine), cmd[cmd.index("-o") + 1])

    monkeypatch.setattr(gpu, "_run_cmd", run)
    return calls


def test_transform_replaced_at_same_path_rebuilds_grid_and_all_coordinate_warps(tmp_path, monkeypatch):
    cfg, forward, _, _ = _assets(tmp_path)
    calls = _fake_ants(monkeypatch)
    out_dir = tmp_path / "grid"
    args = (cfg.rt_unwarped_analysis_ref_epi, cfg.decoder_template, [forward], out_dir)
    path = gpu.precompute_sampling_grid(*args)
    x, y, z = np.indices((4, 5, 6), dtype=np.float32)
    bold = 7 + x + 10 * y + 100 * z
    before = gpu.GpuAntsLikeResampler(path, device="cpu").resample_array(bold)
    np.testing.assert_allclose(before, bold, atol=1e-4)
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 4  # Three coordinates and coverage; unchanged inputs reuse the grid.

    stat = forward.stat()
    forward.write_text("1")
    os.utime(forward, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 8  # Same name, size and timestamp still detect new contents.
    after = gpu.GpuAntsLikeResampler(path, device="cpu").resample_array(bold)
    np.testing.assert_allclose(after[:-1], bold[1:], atol=1e-4)
    np.testing.assert_array_equal(after[-1], 0)  # Outside is not source voxel (0, 0, 0).
    support = gpu.GpuAntsLikeResampler(path, device="cpu").resample_array(np.ones_like(bold))
    np.testing.assert_array_equal(support[-1], 0)
    np.testing.assert_array_equal(support[:-1], 1)


@pytest.mark.parametrize("changed", ["moving_shape", "moving_affine", "fixed_shape", "fixed_affine"])
def test_geometry_changes_regenerate_coordinate_images(tmp_path, monkeypatch, changed):
    cfg, forward, _, _ = _assets(tmp_path)
    calls = _fake_ants(monkeypatch)
    out_dir = tmp_path / "grid"
    args = (cfg.rt_unwarped_analysis_ref_epi, cfg.decoder_template, [forward], out_dir)
    path = gpu.precompute_sampling_grid(*args)
    target = args[0] if changed.startswith("moving") else args[1]
    shape = (3, 4, 5) if changed.endswith("shape") else (4, 5, 6)
    affine = np.eye(4)
    if changed.endswith("affine"):
        affine[1, 3] = 0.5
    _image(target, shape, affine)
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 8
    for axis in "xyz":
        coord = nib.load(out_dir / f"coord_{axis}_epi_vox.nii")
        warped = nib.load(out_dir / f"coord_{axis}_in_fixed.nii")
        assert coord.shape == nib.load(args[0]).shape
        np.testing.assert_array_equal(coord.affine, nib.load(args[0]).affine)
        assert warped.shape == nib.load(args[1]).shape
        np.testing.assert_array_equal(warped.affine, nib.load(args[1]).affine)
    payload = gpu.GpuAntsLikeResampler._load_payload(path)
    assert payload["moving_shape"] == nib.load(args[0]).shape
    assert payload["fixed_shape"] == nib.load(args[1]).shape


def test_legacy_grid_or_missing_grid_does_not_reuse_coordinate_warps(tmp_path, monkeypatch):
    cfg, forward, _, _ = _assets(tmp_path)
    calls = _fake_ants(monkeypatch)
    out_dir = tmp_path / "grid"
    args = (cfg.rt_unwarped_analysis_ref_epi, cfg.decoder_template, [forward], out_dir)
    path = gpu.precompute_sampling_grid(*args)
    with np.load(path) as payload:
        legacy = {key: payload[key] for key in payload.files if key != "provenance"}
    np.savez_compressed(path, **legacy)
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 8
    path.unlink()
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 12


def test_old_grid_without_outside_field_detection_is_rebuilt(tmp_path, monkeypatch):
    cfg, forward, _, _ = _assets(tmp_path)
    calls = _fake_ants(monkeypatch)
    args = (cfg.rt_unwarped_analysis_ref_epi, cfg.decoder_template, [forward], tmp_path / "grid")
    path = gpu.precompute_sampling_grid(*args)
    with np.load(path) as stored:
        payload = {key: stored[key] for key in stored.files}
    identity = json.loads(payload["provenance"].item())
    identity["version"] = 1
    payload["provenance"] = json.dumps(identity)
    np.savez_compressed(path, **payload)
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 8


@pytest.mark.parametrize("space", ["t1", "mni"])
def test_gpu_and_mask_consumers_use_canonical_transform_directions(tmp_path, monkeypatch, space):
    cfg, forward, inverse, mni = _assets(tmp_path)
    calls = _fake_ants(monkeypatch)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        use_gpu_resampler=True, gpu_resampler_device="cpu", analysis_space=space))
    rt.maybe_init_gpu_resampler(cfg)
    expected = [str(forward)] if space == "t1" else [str(mni), str(forward)]
    assert len(calls) == 4
    for cmd in calls:
        assert [cmd[i + 1] for i, value in enumerate(cmd) if value == "-t"] == expected
        assert cmd[cmd.index("-r") + 1] == str(cfg.decoder_template)
    assert mask_transforms(cfg, "t1", "epi") == [inverse]
    assert mask_transforms(cfg, "epi", space) == [Path(value) for value in expected]
    calls.clear()
    rt.maybe_init_gpu_resampler(cfg)
    assert calls == []


@pytest.mark.parametrize("changed", ["forward", "mni", "reference"])
def test_run_rejects_spatial_changes_with_smoothing_disabled(tmp_path, monkeypatch, changed):
    cfg, forward, _, mni = _assets(tmp_path)
    cfg.rt_work_dir = tmp_path / "run"
    cfg.rt_work_dir.mkdir()
    cfg.rt_motion_ref_epi = cfg.rt_unwarped_analysis_ref_epi
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", analysis_space="mni", smoothing_fwhm_mm=0))
    rt.validate_run_provenance(cfg)
    rt.validate_run_provenance(cfg)
    if changed == "reference":
        affine = np.eye(4)
        affine[2, 3] = 1
        _image(cfg.decoder_template, affine=affine)
    else:
        path = forward if changed == "forward" else mni
        stat = path.stat()
        path.write_text("1")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="spatial transform changed"):
        rt.validate_run_provenance(cfg)


def test_epi_run_rejects_rebuilt_nuisance_mask(tmp_path, monkeypatch):
    cfg, _, _, _ = _assets(tmp_path)
    cfg.rt_work_dir = tmp_path / "run"
    cfg.rt_work_dir.mkdir()
    cfg.rt_motion_ref_epi = cfg.rt_unwarped_analysis_ref_epi
    cfg.rt_wm_mask = _image(cfg.trans_dir / "rt_WM_mask.nii")
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", analysis_space="epi", enable_motion_regression=True, use_wm=True))
    rt.validate_run_provenance(cfg)
    stat = cfg.rt_wm_mask.stat()
    os.utime(cfg.rt_wm_mask, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed"):
        rt.validate_run_provenance(cfg)


def _legacy_run(tmp_path, monkeypatch):
    cfg, _, _, _ = _assets(tmp_path)
    cfg.rt_work_dir = tmp_path / "run"
    cfg.rt_work_dir.mkdir()
    cfg.rt_motion_ref_epi = cfg.rt_unwarped_analysis_ref_epi
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", analysis_space="t1", smoothing_fwhm_mm=0))
    marker = cfg.rt_work_dir / "preprocessing_order.json"
    marker.write_text(json.dumps({
        "order": rt.PIPELINE_ORDER, "coverage_policy": rt.COVERAGE_POLICY,
        "calibration": rt.load_calibration(cfg.fmap_dir),
        "field": rt.source_identity(rt._preferred_pyhysco_fieldmap(cfg.fmap_dir)),
        "reference": rt.source_identity(cfg.rt_motion_ref_epi), "epi_phase_encoding": "PA"}, indent=2))
    (cfg.rt_work_dir / "t1").mkdir()
    (cfg.rt_work_dir / "t1" / "vol_00001_t1.nii").touch()
    return cfg, marker


def test_unchanged_legacy_full_brain_run_preserves_original_marker(tmp_path, monkeypatch):
    cfg, marker = _legacy_run(tmp_path, monkeypatch)
    saved = marker.read_bytes()
    mtime = marker.stat().st_mtime_ns
    rt.validate_run_provenance(cfg)
    rt.validate_run_provenance(cfg)
    assert marker.read_bytes() == saved
    assert marker.stat().st_mtime_ns == mtime
    assert "spatial_transform" not in json.loads(marker.read_text())


def test_run_without_current_coverage_policy_is_rejected(tmp_path, monkeypatch):
    cfg, marker = _legacy_run(tmp_path, monkeypatch)
    payload = json.loads(marker.read_text())
    payload.pop("coverage_policy")
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fresh run output directory"):
        rt.validate_run_provenance(cfg)


@pytest.mark.parametrize("mode", ["full_brain", "partial_epi"])
def test_legacy_run_rejected_after_versioned_registration(tmp_path, monkeypatch, mode):
    cfg, marker = _legacy_run(tmp_path, monkeypatch)
    saved = marker.read_bytes()
    (cfg.trans_dir / "registration.json").write_text(json.dumps({"mode": mode}))
    with pytest.raises(ValueError, match="cannot certify the new registration"):
        rt.validate_run_provenance(cfg)
    assert marker.read_bytes() == saved


@pytest.mark.parametrize("changed", ["reference", "field", "phase_encoding"])
def test_legacy_compatibility_retains_original_provenance_checks(tmp_path, monkeypatch, changed):
    cfg, marker = _legacy_run(tmp_path, monkeypatch)
    saved = marker.read_bytes()
    if changed == "phase_encoding":
        rt.REGRESSOR_SETTINGS.epi_phase_encoding = "AP"
    else:
        path = cfg.rt_motion_ref_epi if changed == "reference" else rt._preferred_pyhysco_fieldmap(cfg.fmap_dir)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed"):
        rt.validate_run_provenance(cfg)
    assert marker.read_bytes() == saved


@pytest.mark.parametrize("existing_marker", [False, True])
def test_incomplete_registration_blocks_new_and_legacy_runs(tmp_path, monkeypatch, existing_marker):
    cfg, marker = _legacy_run(tmp_path, monkeypatch)
    if not existing_marker:
        marker.unlink()
    (cfg.trans_dir / "registration_pending.json").write_text("{}")
    with pytest.raises(ValueError, match="Registration preparation is incomplete"):
        rt.validate_run_provenance(cfg)
