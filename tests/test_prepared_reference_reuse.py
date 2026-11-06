"""A prepared online RS reference must not trigger a second NF reference fit."""
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel as headless  # Optional GUI/native RTPSpy dependencies.
import rt_pipeline as rt
from fmri_rt_preproc import session_alignment as session
from fmri_rt_preproc.analysis_mask import mask_transforms, session_epi_bridge
from fmri_rt_preproc.offline_assets import t1_to_mni_transform


def _image(path, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ones((4, 5, 6), np.float32),
                           np.eye(4) if affine is None else affine)
    nib.save(image, path)
    return path


@pytest.fixture
def prepared(tmp_path):
    cfg = rt.RTSessionConfig("88888", "1", "5", tmp_path / "incoming", tmp_path,
                             reuse_prepared_reference=True)
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


def test_offline_assets_cannot_silently_reuse_native_rs_coordinates(prepared):
    cfg, settings = prepared
    settings.offline_assets = Path("offline_assets.json")
    with pytest.raises(ValueError, match="offline_assets needs the current-session EPI bridge"):
        session.ensure_nf_session(cfg, settings)


def test_default_offline_path_still_loads_the_imported_assets(prepared, monkeypatch):
    cfg, settings = prepared
    cfg.reuse_prepared_reference = False
    settings.offline_assets = Path("offline_assets.json")
    calls = []

    class ReachedOfflineLoader(Exception):
        pass

    def load(path, subject):
        calls.append((path, subject))
        raise ReachedOfflineLoader

    monkeypatch.setattr("fmri_rt_preproc.offline_assets.load_offline_assets", load)
    with pytest.raises(ReachedOfflineLoader):
        session.ensure_nf_session(cfg, settings)
    assert calls == [(settings.offline_assets, cfg.subject)]


def test_default_and_preparation_directory_selection_is_unchanged(tmp_path):
    cfg = rt.RTSessionConfig("88888", "1", "5", tmp_path, tmp_path)
    assert cfg.reuse_prepared_reference is False
    assert cfg.trans_dir == cfg.day_root / "func" / "trans"
    bridge = cfg.day_root / "func" / "session_trans"
    bridge.mkdir(parents=True)
    (bridge / "session_alignment.json").write_text("{}")
    assert cfg.trans_dir == bridge
    cfg.preparation_only = True
    assert cfg.trans_dir == cfg.day_root / "func" / "trans"
