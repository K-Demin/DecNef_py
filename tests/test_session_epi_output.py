"""Stable EPI output geometry must survive the current-session MC reference."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from fmri_rt_preproc.analysis_mask import mask_transforms, prepare_analysis_mask
from fmri_rt_preproc.volume_smooth import MaskedGaussianSmoother
from test_native_fieldmap import make_pair


def save(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine), path)
    return path


@pytest.fixture(params=["nf", "offline_rs"])
def session_cfg(tmp_path, request):
    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.preparation_only = request.param == "offline_rs"
    cfg.trans_dir = tmp_path / ("trans" if cfg.preparation_only else "session_trans")
    cfg.trans_dir.mkdir()
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    cfg.subject_root = tmp_path
    shape = (3, 4, 5)
    reference_shape = (5, 4, 5)
    cfg.rt_motion_ref_epi = save(tmp_path / "current_epi.nii", np.full(shape, 1000.))
    cfg.rt_unwarped_analysis_ref_epi = cfg.rt_motion_ref_epi
    cfg.rt_unwarped_analysis_ref_mask = save(tmp_path / "current_mask.nii", np.ones(shape))
    save(cfg.trans_dir / "reference_epi.nii", np.ones(reference_shape))
    cfg.decoder_template = save(tmp_path / "decoder.nii", np.ones(reference_shape))
    for name in ("session_to_reference_Composite.h5", "epi2t1_InverseComposite.h5"):
        (cfg.trans_dir / name).write_bytes(name.encode())
    if cfg.preparation_only:
        (cfg.trans_dir / "registration.json").write_text(json.dumps({"settings": {
            "offline_assets": {"manifest": {"path": "assets.json"}}}}))
    for name in ("mc", "reg", "unwarp"):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, "rt_" + name + "_dir", folder)
    return cfg, handler, shape, reference_shape


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("fwhm", [0, 3])
def test_epi_bridge_maps_cleaned_original_and_support_before_scoring(session_cfg, monkeypatch, gpu, fwhm):
    cfg, handler, shape, target_shape = session_cfg
    cfg.enable_scoring, cfg.enable_original_score = True, True
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda data: data)
    support = np.ones(shape, bool)
    support[..., -1] = False
    brain = np.ones(shape, bool)
    brain[0, 0] = False
    valid = support & brain
    handler.volreg = SimpleNamespace(do_proc=lambda *args, **kwargs: None,
        _motion=np.zeros((1, 6)), resample_support=lambda data, idx: support.astype(float))
    handler.biopac_receiver = None
    handler.motion_regressor = SimpleNamespace(apply=lambda *args, **kwargs: (np.full(shape, -2.), True),
        get_regressors=lambda idx: ([], None), output_support=valid, brain_mask=brain)
    handler.smoother = MaskedGaussianSmoother(np.ones(target_shape, bool), np.eye(4), fwhm) if fwhm else None
    if fwhm:
        (cfg.rt_work_dir / "smooth").mkdir()
    handler.volume_streamer = SimpleNamespace(publish=lambda *args: None)
    scored = []
    handler.scorer = SimpleNamespace(mask=np.ones(target_shape, bool), baseline_count=0, n_baseline=0,
        score_from_array=lambda data: scored.append(data.copy()) or float(data.mean()))
    handler.score_event_tracker = SimpleNamespace(for_volume=lambda idx: None)
    handler.reference_score_stats = None
    calls = []

    def mapped(data):
        # A larger stable preparation grid has empty margins beyond the current slab.
        assert not (cfg.rt_work_dir / "epi_ref" / "vol_00001_epi_ref.nii").exists()
        return np.pad(data, ((1, 1), (0, 0), (0, 0)))

    def apply_file(source, output):
        save(Path(output), mapped(nib.load(source).get_fdata()))

    def run(args):
        assert args[0] == "antsApplyTransforms"
        assert Path(args[args.index("-r") + 1]) == cfg.trans_dir / "reference_epi.nii"
        transforms = [Path(args[i + 1]) for i, arg in enumerate(args) if arg == "-t"]
        assert transforms == [cfg.trans_dir / "session_to_reference_Composite.h5"]
        calls.append(args)
        apply_file(args[args.index("-i") + 1], args[args.index("-o") + 1])

    handler.gpu_resampler = SimpleNamespace(resample_support=mapped, resample_nifti_to_nifti=apply_file) if gpu else None
    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(epi_phase_encoding="PA",
        save_intermediate_unwarped=False, enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space="epi", volume_stream_kind="score_input"))
    assert rt.process_volume(cfg, handler, cfg.rt_motion_ref_epi, 1,
                             raw_nii=cfg.rt_motion_ref_epi, volume_timestamp=1.)
    expected = np.pad(valid, ((1, 1), (0, 0), (0, 0)))
    output = nib.load(cfg.rt_work_dir / "epi_ref" / "vol_00001_epi_ref.nii")
    assert output.shape == target_shape
    np.testing.assert_array_equal(output.get_fdata()[~expected], 0.)
    for data, value in zip(scored, (1000., -2.)):
        assert data.shape == target_shape
        np.testing.assert_allclose(data[expected], value, atol=1e-6)
        np.testing.assert_array_equal(data[~expected], 0.)
    np.testing.assert_array_equal(nib.load(cfg.rt_work_dir / "support" / "vol_00001_epi_ref.nii").get_fdata(), expected)
    assert not list((cfg.rt_work_dir / "epi_ref").glob(".vol*"))
    assert (len(calls) == 4) if not gpu else (len(calls) == 0)


def test_epi_decoder_grid_mismatch_rejected_even_when_shape_matches(session_cfg):
    cfg, _, _, target_shape = session_cfg
    assert rt.resolve_analysis_reference(cfg, "epi") == cfg.trans_dir / "reference_epi.nii"
    affine = np.eye(4)
    affine[0, 3] = 10.
    save(cfg.decoder_template, np.ones(target_shape), affine)
    with pytest.raises(ValueError, match="stable EPI output grid"):
        rt.resolve_analysis_reference(cfg, "epi")


def test_declared_epi_mapping_cannot_silently_fall_back_when_assets_are_missing(session_cfg):
    cfg, _, _, _ = session_cfg
    (cfg.trans_dir / "session_to_reference_Composite.h5").unlink()
    (cfg.trans_dir / "reference_epi.nii").unlink()
    with pytest.raises(FileNotFoundError, match="mapping is incomplete"):
        rt.resolve_analysis_reference(cfg, "epi")


def test_gpu_epi_sampling_grid_uses_bridge_and_stable_reference(session_cfg, monkeypatch):
    import gpu_ants_like_resampler as gpu
    cfg, _, _, _ = session_cfg
    observed = []
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(use_gpu_resampler=True,
        analysis_space="epi", gpu_resampler_device="cpu"))
    monkeypatch.setattr(gpu, "precompute_sampling_grid", lambda **kwargs: observed.append(kwargs) or Path("grid.npy"))
    monkeypatch.setattr(gpu, "GpuAntsLikeResampler", lambda **kwargs: SimpleNamespace(device=kwargs["device"]))
    assert rt.maybe_init_gpu_resampler(cfg).device == "cpu"
    assert observed[0]["fixed_ref"] == cfg.trans_dir / "reference_epi.nii"
    assert observed[0]["moving_ref"] == cfg.rt_motion_ref_epi
    assert observed[0]["transforms"] == [cfg.trans_dir / "session_to_reference_Composite.h5"]
    assert observed[0]["out_dir"] == cfg.trans_dir / "gpu_grids" / "epi_to_reference"


def test_run_provenance_binds_epi_bridge_and_reference(session_cfg, monkeypatch):
    cfg, _, _, _ = session_cfg
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(epi_phase_encoding="PA", analysis_space="epi"))
    rt.validate_run_provenance(cfg)
    saved = json.loads((cfg.rt_work_dir / "preprocessing_order.json").read_text())
    assert saved["epi_reference_mapped"] is True
    assert saved["spatial_transform"]["fixed"]["shape"] == [5, 4, 5]
    rt.validate_run_provenance(cfg)
    (cfg.trans_dir / "session_to_reference_Composite.h5").write_bytes(b"changed bridge")
    with pytest.raises(ValueError, match="changed"):
        rt.validate_run_provenance(cfg)


def test_epi_masks_follow_bridge_and_t1_inverse_then_bridge(session_cfg, monkeypatch):
    cfg, _, shape, target_shape = session_cfg
    bridge = cfg.trans_dir / "session_to_reference_Composite.h5"
    assert mask_transforms(cfg, "epi", "epi") == [bridge]
    assert mask_transforms(cfg, "t1", "epi") == [bridge, cfg.trans_dir / "epi2t1_InverseComposite.h5"]
    assert mask_transforms(cfg, "final", "epi") == []
    folder = cfg.rt_work_dir / "mask_test"
    folder.mkdir()
    calls = []

    def apply(args):
        calls.append(args)
        assert [Path(args[i + 1]) for i, arg in enumerate(args) if arg == "-t"] == [bridge]
        assert args[args.index("-n") + 1] == "NearestNeighbor"
        source = nib.load(args[args.index("-i") + 1]).get_fdata()
        save(Path(args[args.index("-o") + 1]), np.pad(source, ((1, 1), (0, 0), (0, 0))))
    mask, _ = prepare_analysis_mask(cfg, "whole_brain", "epi", cfg.trans_dir / "reference_epi.nii", folder, apply)
    assert mask.shape == target_shape
    assert not mask[[0, -1]].any()
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["reg", "mc", "unwarped"])
@pytest.mark.parametrize("fwhm", [0, 4])
def test_pca_never_reads_session_grid_for_a_stable_epi_decoder(tmp_path, kind, fwhm):
    from rs_pca_runtime import volume_path_for_kind
    (tmp_path / "session_metadata.json").write_text(json.dumps({"regression": {"analysis_space": "epi"},
        "epi_reference_mapped": True, "smoothing_fwhm_mm": fwhm}))
    if kind != "reg":
        with pytest.raises(ValueError, match="stable reference EPI grid"):
            volume_path_for_kind(tmp_path, 1, kind)
    else:
        suffix = "smooth" if fwhm else "epi_ref"
        assert volume_path_for_kind(tmp_path, 1, kind) == tmp_path / suffix / f"vol_00001_{suffix}.nii"
