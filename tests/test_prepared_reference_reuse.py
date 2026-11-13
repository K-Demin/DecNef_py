"""NF consumes the existing reference run, without automatic spatial preparation."""
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel as headless  # Optional GUI/native RTPSpy dependencies.
import rt_pipeline as rt
from fmri_rt_preproc import session_alignment as session
from fmri_rt_preproc.analysis_mask import mask_transforms, session_epi_bridge
from fmri_rt_preproc.offline_assets import file_identity, t1_to_mni_transform


def _image(path, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ones((4, 5, 6), np.float32),
                           np.eye(4) if affine is None else affine)
    nib.save(image, path)
    return path


@pytest.fixture
def prepared(tmp_path):
    cfg = rt.RTSessionConfig("88888", "1", "5", tmp_path / "incoming", tmp_path)
    cfg.incoming_dir.mkdir()
    folder = cfg.day_root / "func" / "trans"
    for name in ("epi_unwarped_mean.nii", "epi_mask_mean.nii", "rt_WM_mask.nii", "rt_Vent_mask.nii"):
        _image(folder / name)
    for name in ("epi2t1_Composite.h5", "epi2t1_InverseComposite.h5", "t1_to_mni.h5"):
        (folder / name).write_text("prepared " + name)
    settings = SimpleNamespace(offline_assets=None, analysis_space="t1",
                               enable_motion_regression=True, use_wm=True, use_vent=True,
                               analysis_mask="whole_brain")
    return cfg, settings


def test_reuse_never_waits_for_bold_or_builds_a_new_reference(prepared, monkeypatch):
    cfg, settings = prepared
    import run_preproc
    from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor

    def unexpected(*args, **kwargs):
        pytest.fail("Prepared-reference reuse must not wait, import assets, or fit a reference")

    monkeypatch.setattr(run_preproc, "_wait_for_epi_dicoms", unexpected)
    monkeypatch.setattr(run_preproc, "_stage_epi_block", unexpected)
    monkeypatch.setattr(FMRIRealtimePreprocessor, "__init__", unexpected)
    monkeypatch.setattr(session, "prepared_assets", unexpected)
    monkeypatch.setattr(session, "register_session_bridge", unexpected)
    before = {path: path.stat().st_mtime_ns for path in cfg.day_root.rglob("*")}
    calls = []
    session.ensure_nf_session(cfg, settings, prepare_fieldmap=calls.append)
    assert calls == [cfg]
    assert {path: path.stat().st_mtime_ns for path in cfg.day_root.rglob("*")} == before
    assert not list(cfg.incoming_dir.iterdir())
    assert not (cfg.day_root / "func" / "session_trans").exists()


def test_existing_session_bridge_cannot_override_prepared_reference(prepared):
    cfg, settings = prepared
    old = cfg.day_root / "func" / "session_trans"
    old.mkdir()
    # Reuse must not even inspect an unrelated bridge marker.
    (old / "session_alignment.json").write_text("old bridge metadata")
    (old / "session_to_reference_Composite.h5").write_text("old bridge")
    session.ensure_nf_session(cfg, settings)
    selected = cfg.day_root / "func" / "trans"
    assert cfg.trans_dir == selected
    assert cfg.rt_motion_ref_epi == selected / "epi_unwarped_mean.nii"
    assert cfg.rt_motion_ref_mask == selected / "epi_mask_mean.nii"
    assert cfg.rt_wm_mask == selected / "rt_WM_mask.nii"
    assert cfg.rt_vent_mask == selected / "rt_Vent_mask.nii"
    assert session_epi_bridge(cfg) is None
    assert t1_to_mni_transform(cfg) == selected / "t1_to_mni.h5"
    assert mask_transforms(cfg, "epi", "t1") == [selected / "epi2t1_Composite.h5"]
    assert mask_transforms(cfg, "t1", "epi") == [selected / "epi2t1_InverseComposite.h5"]


@pytest.mark.parametrize("space", ["t1", "mni"])
def test_gpu_and_cpu_paths_use_the_original_prepared_transforms(prepared, tmp_path, monkeypatch, space):
    import gpu_ants_like_resampler as gpu
    cfg, settings = prepared
    settings.analysis_space = space
    selected = cfg.trans_dir
    old = cfg.day_root / "func" / "session_trans"
    old.mkdir()
    (old / "session_alignment.json").write_text("{}")
    cfg.decoder_template = _image(tmp_path / "decoder.nii")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(use_gpu_resampler=True,
                        analysis_space=space, gpu_resampler_device="cpu"))
    observed = []
    monkeypatch.setattr(gpu, "precompute_sampling_grid",
                        lambda **kwargs: observed.append(kwargs) or tmp_path / "grid.npy")
    monkeypatch.setattr(gpu, "GpuAntsLikeResampler",
                        lambda **kwargs: SimpleNamespace(device=kwargs["device"]))
    session.ensure_nf_session(cfg, settings)
    rt.maybe_init_gpu_resampler(cfg)
    transforms = ([selected / "t1_to_mni.h5"] if space == "mni" else []) + [selected / "epi2t1_Composite.h5"]
    assert observed[0]["moving_ref"] == selected / "epi_unwarped_mean.nii"
    assert observed[0]["transforms"] == transforms
    assert selected in observed[0]["out_dir"].parents
    calls = []

    def apply(args):
        calls.append(args)
        nib.save(nib.load(args[args.index("-r") + 1]), args[args.index("-o") + 1])

    monkeypatch.setattr(rt, "run", apply)
    rt.final_volume_support(cfg, SimpleNamespace(gpu_resampler=None), np.ones((4, 5, 6), bool),
                            np.eye(4), 1, space)
    assert [Path(calls[0][i + 1]) for i, arg in enumerate(calls[0]) if arg == "-t"] == transforms


@pytest.mark.parametrize("name", ["epi_unwarped_mean.nii", "epi_mask_mean.nii",
                                  "rt_WM_mask.nii", "rt_Vent_mask.nii", "epi2t1_Composite.h5"])
def test_missing_prepared_asset_has_clear_error_without_starting_preparation(prepared, name):
    cfg, settings = prepared
    (cfg.trans_dir / name).unlink()
    calls = []
    with pytest.raises(FileNotFoundError, match="Prepared RS/behavioral reference is incomplete") as error:
        session.ensure_nf_session(cfg, settings, prepare_fieldmap=calls.append)
    assert name in str(error.value)
    assert calls == []
    assert not (cfg.day_root / "func" / "session_trans").exists()


def test_reuse_accepts_compressed_reference_and_does_not_require_disabled_nuisance_masks(prepared):
    cfg, settings = prepared
    for name in ("epi_unwarped_mean.nii", "epi_mask_mean.nii"):
        path = cfg.trans_dir / name
        nib.save(nib.load(path), path.with_suffix(".nii.gz"))
        path.unlink()
    settings.enable_motion_regression = False
    for name in ("rt_WM_mask.nii", "rt_Vent_mask.nii"):
        (cfg.trans_dir / name).unlink()
    session.ensure_nf_session(cfg, settings)
    assert cfg.rt_motion_ref_epi.name == "epi_unwarped_mean.nii.gz"


def test_reuse_keeps_native_mask_grid_validation(prepared):
    cfg, settings = prepared
    affine = np.eye(4)
    affine[2, 3] = 10
    _image(cfg.rt_wm_mask, affine)
    with pytest.raises(ValueError, match="grids differ"):
        session.ensure_nf_session(cfg, settings)


def _offline_preparation(cfg, settings, tmp_path):
    manifest = tmp_path / "offline" / "assets.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"version": 1, "subject_id": cfg.subject}))
    settings.offline_assets = manifest
    (cfg.trans_dir / "preparation_assets.json").write_text(json.dumps({
        "version": 1, "offline_manifest": file_identity(manifest)}))
    for name in ("session_to_reference_Composite.h5", "session_to_reference_InverseComposite.h5"):
        (cfg.trans_dir / name).write_text("prepared " + name)
    affine = np.eye(4)
    affine[2, 3] = 12
    for name in ("reference_epi.nii", "reference_epi_mask.nii"):
        _image(cfg.trans_dir / name, affine)
    return manifest


def test_prepared_offline_reference_reuses_its_existing_bridge_without_loading_bundle(
        prepared, tmp_path, monkeypatch):
    cfg, settings = prepared
    manifest = _offline_preparation(cfg, settings, tmp_path)
    calls = []
    identity = session.file_identity

    def small_manifest_only(path):
        calls.append(Path(path))
        assert Path(path) == manifest
        return identity(path)

    monkeypatch.setattr(session, "file_identity", small_manifest_only)
    monkeypatch.setattr("fmri_rt_preproc.offline_assets.load_offline_assets",
                        lambda *args: pytest.fail("NF must not import anatomy or reload the bundle"))
    before = {path: path.stat().st_mtime_ns for path in cfg.day_root.rglob("*")}
    fieldmaps = []
    session.ensure_nf_session(cfg, settings, prepare_fieldmap=fieldmaps.append)
    assert calls == [manifest] and fieldmaps == [cfg]
    assert {path: path.stat().st_mtime_ns for path in cfg.day_root.rglob("*")} == before
    assert session_epi_bridge(cfg) == cfg.trans_dir / "session_to_reference_Composite.h5"


@pytest.mark.parametrize("change", ["missing", "unreadable", "version", "native", "path", "content"])
def test_explicit_offline_assets_require_matching_preparation(prepared, tmp_path, change):
    cfg, settings = prepared
    manifest = _offline_preparation(cfg, settings, tmp_path)
    metadata = cfg.trans_dir / "preparation_assets.json"
    if change == "missing":
        metadata.unlink()
    elif change == "unreadable":
        metadata.write_text("broken metadata")
    elif change == "version":
        metadata.write_text(json.dumps({"version": 2, "offline_manifest": file_identity(manifest)}))
    elif change == "native":
        metadata.write_text(json.dumps({"version": 1, "offline_manifest": None}))
    elif change == "path":
        other = manifest.with_name("other_assets.json")
        other.write_bytes(manifest.read_bytes())
        settings.offline_assets = other
    else:
        manifest.write_text(json.dumps({"version": 1, "subject_id": "different"}))
    calls = []
    with pytest.raises(ValueError, match="reference") as error:
        session.ensure_nf_session(cfg, settings, prepare_fieldmap=calls.append)
    assert str(cfg.trans_dir) in str(error.value)
    assert "reference preparation" in str(error.value)
    assert calls == []


@pytest.mark.parametrize("name", ["session_to_reference_Composite.h5",
    "session_to_reference_InverseComposite.h5", "reference_epi.nii", "reference_epi_mask.nii",
    "epi2t1_Composite.h5", "epi2t1_InverseComposite.h5"])
def test_offline_preparation_requires_complete_bridge_and_composites(prepared, tmp_path, name):
    cfg, settings = prepared
    _offline_preparation(cfg, settings, tmp_path)
    (cfg.trans_dir / name).unlink()
    with pytest.raises(FileNotFoundError, match="reference is incomplete") as error:
        session.ensure_nf_session(cfg, settings)
    assert name in str(error.value)


def test_epi_space_offline_reference_keeps_its_distinct_final_grid(prepared, tmp_path, monkeypatch):
    import gpu_ants_like_resampler as gpu
    cfg, settings = prepared
    _offline_preparation(cfg, settings, tmp_path)
    settings.analysis_space = "epi"
    cfg.decoder_template = cfg.trans_dir / "reference_epi.nii"
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(use_gpu_resampler=True,
                        analysis_space="epi", gpu_resampler_device="cpu"))
    observed = []
    monkeypatch.setattr(gpu, "precompute_sampling_grid",
                        lambda **kwargs: observed.append(kwargs) or tmp_path / "grid.npy")
    monkeypatch.setattr(gpu, "GpuAntsLikeResampler",
                        lambda **kwargs: SimpleNamespace(device=kwargs["device"]))
    session.ensure_nf_session(cfg, settings)
    rt.maybe_init_gpu_resampler(cfg)
    assert observed[0]["moving_ref"] == cfg.rt_motion_ref_epi
    assert observed[0]["fixed_ref"] == cfg.trans_dir / "reference_epi.nii"
    assert observed[0]["transforms"] == [cfg.trans_dir / "session_to_reference_Composite.h5"]
    assert not np.array_equal(nib.load(cfg.rt_motion_ref_epi).affine,
                              nib.load(observed[0]["fixed_ref"]).affine)
    assert mask_transforms(cfg, "t1", "epi") == [
        cfg.trans_dir / "session_to_reference_Composite.h5",
        cfg.trans_dir / "epi2t1_InverseComposite.h5"]


def test_no_prepared_reference_never_collects_or_fits_one(tmp_path, monkeypatch):
    cfg = rt.RTSessionConfig("88888", "1", "5", tmp_path / "incoming", tmp_path)
    settings = SimpleNamespace(offline_assets=None, analysis_space="t1")
    import run_preproc
    from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor

    def unexpected(*args, **kwargs):
        pytest.fail("NF must not prepare its own spatial reference")

    monkeypatch.setattr(run_preproc, "_wait_for_epi_dicoms", unexpected)
    monkeypatch.setattr(run_preproc, "_stage_epi_block", unexpected)
    monkeypatch.setattr(FMRIRealtimePreprocessor, "__init__", unexpected)
    with pytest.raises(FileNotFoundError, match="reference preparation") as error:
        session.ensure_nf_session(cfg, settings, prepare_fieldmap=unexpected)
    assert str(cfg.trans_dir) in str(error.value)
    assert not cfg.day_root.exists()


def test_pending_reference_is_rejected_before_fieldmap_work(prepared):
    cfg, settings = prepared
    pending = cfg.trans_dir / "registration_pending.json"
    pending.write_text("interrupted")
    calls = []
    with pytest.raises(ValueError, match="registration_pending.json") as error:
        session.ensure_nf_session(cfg, settings, prepare_fieldmap=calls.append)
    assert str(cfg.trans_dir) in str(error.value)
    assert "reference preparation" in str(error.value)
    assert calls == []


def test_changed_fitting_and_normalization_options_do_not_refit_or_block_prepared_reference(prepared):
    cfg, settings = prepared
    settings.registration_initialization = "center_of_mass"
    settings.registration_syn = False
    settings.registration_threads = 8
    settings.voxel_norm_mode = "mean"
    settings.voxel_norm_reference = "running"
    settings.voxel_norm_ref_volumes = 60
    (cfg.trans_dir / "registration_qc_epi_t1.json").write_text(json.dumps({"gate_status": "fail"}))
    session.ensure_nf_session(cfg, settings)
    assert not list(cfg.trans_dir.glob("*.approval.json"))


def test_default_and_preparation_runs_always_select_func_trans(tmp_path):
    cfg = rt.RTSessionConfig("88888", "1", "5", tmp_path, tmp_path)
    assert cfg.trans_dir == cfg.day_root / "func" / "trans"
    old = cfg.day_root / "func" / "session_trans"
    old.mkdir(parents=True)
    (old / "session_alignment.json").write_text("{}")
    assert cfg.trans_dir == cfg.day_root / "func" / "trans"
    cfg.preparation_only = True
    assert cfg.trans_dir == cfg.day_root / "func" / "trans"
