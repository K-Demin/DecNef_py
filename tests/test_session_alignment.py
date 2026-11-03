"""First-NF reference cache and transform composition, without native ANTs."""
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

import test_rt_pipeline_parallel as headless  # Optional GUI/native RTPSpy dependencies.
from fmri_rt_preproc import session_alignment as session
from fmri_rt_preproc.offline_assets import copy_image
from fmri_rt_preproc.registration_qc import approve_registration_qc, build_registration_qc


def _image(path, shape, affine, mask=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.indices(shape, dtype=float)
    data = np.ones(shape) if mask else 30 + 2 * np.sin(xyz[0]) + 3 * np.cos(xyz[1]) + xyz[2]
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)
    return path


def _assets(folder):
    epi_affine = np.diag([2., 2., 3., 1.])
    t1_affine = np.diag([1., 1., 1.5, 1.])
    assets = {
        "epi_reference": _image(folder / "offline_epi.nii", (7, 8, 9), epi_affine),
        "epi_mask": _image(folder / "offline_mask.nii", (7, 8, 9), epi_affine, True),
        "t1": _image(folder / "T1.nii", (11, 12, 13), t1_affine),
        "mni_template": _image(folder / "MNI.nii", (13, 14, 15), np.eye(4)),
    }
    for key in ("t1_mask", "wm_mask", "vent_mask"):
        assets[key] = _image(folder / (key + ".nii"), (11, 12, 13), t1_affine, True)
    for key in ("epi_to_t1", "t1_to_epi", "t1_to_mni"):
        assets[key] = folder / (key + ".h5")
        assets[key].write_text(key)
    return assets


class FakeAnts:
    """Record pull-transform order and emit images on distinct reference grids."""
    def __init__(self):
        self.calls = []
        self.wrong_grid = False

    def __call__(self, args):
        self.calls.append(list(args))
        if args[0] == "fslmerge":
            nib.save(nib.load(args[3]), args[2])
            return
        if args[0] == "antsRegistration":
            prefix, warped, inverse = args[args.index("--output") + 1][1:-1].split(",")
            fixed, moving = args[args.index("--metric") + 1][3:-1].split(",")[:2]
            Path(prefix + "Composite.h5").write_text("native_offline_to_session")
            Path(prefix + "InverseComposite.h5").write_text("native_session_to_offline")
            for output, reference in ((warped, moving if self.wrong_grid else fixed), (inverse, moving)):
                image = nib.load(reference)
                _image(Path(output), image.shape, image.affine)
            return
        assert args[0] == "antsApplyTransforms"
        reference = Path(args[args.index("-r") + 1])
        output = args[args.index("-o") + 1]
        transforms = [args[index + 1] for index, value in enumerate(args) if value == "-t"]
        if output.startswith("["):
            Path(output[1:-3]).write_text(json.dumps({"transforms": transforms, "reference": str(reference)}))
        else:
            image = nib.load(reference)
            _image(Path(output), image.shape, image.affine)


@pytest.fixture
def bridge(tmp_path):
    assets = _assets(tmp_path / "offline source")
    affine = np.diag([2., 2., 3., 1.])
    affine[2, 3] = 12.
    epi = _image(tmp_path / "session.nii", (7, 8, 4), affine)
    mask = _image(tmp_path / "session_mask.nii", (7, 8, 4), affine, True)
    return assets, epi, mask, tmp_path / "session transforms", FakeAnts()


def test_bridge_composition_preserves_forward_inverse_order_and_target_grids(bridge):
    assets, epi, mask, folder, runner = bridge
    plan = session.register_session_bridge(assets, epi, mask, folder, runner, epi_reference_n4=False)
    assert (plan.fixed, plan.moving) == (epi, assets["epi_reference"])
    assert (folder / "session_to_reference_Composite.h5").read_text() == "native_session_to_offline"
    assert (folder / "session_to_reference_InverseComposite.h5").read_text() == "native_offline_to_session"
    forward = json.loads((folder / "epi2t1_Composite.h5").read_text())
    inverse = json.loads((folder / "epi2t1_InverseComposite.h5").read_text())
    assert forward == {"transforms": [str(assets["epi_to_t1"]), str(folder / "session_to_reference_Composite.h5")],
                       "reference": str(assets["t1"])}
    assert inverse == {"transforms": [str(folder / "session_to_reference_InverseComposite.h5"), str(assets["t1_to_epi"])],
                       "reference": str(epi)}
    for output, reference in ((folder / "epi2t1_Warped.nii", assets["t1"]),
                              (folder / "epi2t1_InverseWarped.nii", epi)):
        actual, target = nib.load(output), nib.load(reference)
        assert actual.shape == target.shape
        np.testing.assert_array_equal(actual.affine, target.affine)


def test_bridge_rejects_native_wrong_grid_before_canonical_composition(bridge):
    assets, epi, mask, folder, runner = bridge
    runner.wrong_grid = True
    with pytest.raises(ValueError, match="grids differ"):
        session.register_session_bridge(assets, epi, mask, folder, runner, epi_reference_n4=False)
    assert not (folder / "epi2t1_Composite.h5").exists()
    assert len(runner.calls) == 1


def test_pipeline_bridge_cache_rejects_newly_selected_epi_path(bridge, monkeypatch):
    from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
    assets, epi, mask, folder, runner = bridge
    folder.mkdir()
    pipe = object.__new__(FMRIRealtimePreprocessor)
    pipe.trans_dir, pipe.offline_assets = folder, assets
    pipe.cfg = SimpleNamespace(registration_initialization="identity", registration_syn=True,
                               registration_direction="epi_to_t1", epi_reference_n4=False)
    monkeypatch.setattr("fmri_rt_preproc.pipeline.run", runner)
    pipe._register_to_offline_reference(epi, mask)
    count = len(runner.calls)
    pipe._register_to_offline_reference(epi, mask)
    assert len(runner.calls) == count
    selected = epi.with_name("another_selected_epi.nii")
    nib.save(nib.load(epi), selected)
    with pytest.raises(ValueError, match="input selection changed"):
        pipe._register_to_offline_reference(selected, mask)
    assert len(runner.calls) == count


@pytest.fixture
def frozen_session(tmp_path, monkeypatch):
    assets = _assets(tmp_path / "offline")
    day = tmp_path / "sub-001" / "day"
    day.mkdir(parents=True)
    cfg = SimpleNamespace(day_root=day, subject_root=day.parent, subject="001", day="day", run="4",
                          incoming_dir=tmp_path / "incoming", fmap_dir=day / "fmap", preparation_only=False)
    cfg.fmap_dir.mkdir()
    for direction in ("AP", "PA"):
        _image(cfg.fmap_dir / (direction + ".nii"), (7, 8, 4), np.eye(4))
    settings = SimpleNamespace(offline_assets=None, registration_qc_policy=None,
        registration_direction="epi_to_t1", registration_initialization="identity", registration_syn=True,
        epi_phase_encoding="AP", epi_reference_n4=False)
    runner, stages, preparations = FakeAnts(), [], []
    monkeypatch.setattr(session, "prepared_assets", lambda current: assets)
    monkeypatch.setattr("fmri_rt_preproc.offline_assets.load_offline_assets", lambda path, subject: assets)
    monkeypatch.setattr("fmri_rt_preproc.utils.run", runner)
    monkeypatch.setattr("run_preproc._wait_for_epi_dicoms", lambda *args, **kwargs: stages.append((args, kwargs)))

    def stage(incoming, run, folder):
        folder.mkdir()
        path = folder / "staged.nii"
        _image(path, (7, 8, 4, 20), np.eye(4))
        return [path]
    monkeypatch.setattr("run_preproc._stage_epi_block", stage)

    class Preparation:
        def __init__(self, prep_cfg):
            self.cfg = prep_cfg
            preparations.append(self)

        def _prepare_anat(self):
            self.anat_dir.mkdir()

        def _unwarp_epi(self, source, output):
            nib.save(nib.load(source), output)

        def _motion_correct_and_mean(self, source, output, count):
            assert count == 20
            image = nib.load(source)
            _image(output, image.shape[:3], image.affine)

        def _skullstrip_epi(self, source, brain, mask):
            image = nib.load(source)
            nib.save(image, brain)
            _image(mask, image.shape, image.affine, True)

        def _register_to_offline_reference(self, epi, mask):
            session.register_session_bridge(self.offline_assets, epi, mask, self.trans_dir, runner,
                                             epi_reference_n4=self.cfg.epi_reference_n4)

        def _make_rtp_nuisance_masks(self):
            image = nib.load(self.rt_unwarped_analysis_ref_epi)
            for name in ("rt_WM_mask.nii", "rt_Vent_mask.nii"):
                _image(self.trans_dir / name, image.shape, image.affine, True)

        def _write_registration_qc(self):
            reports = []
            for name in ("epi_t1", "epi_epi", "t1_mni"):
                path = self.trans_dir / ("registration_qc_" + name + ".json")
                build_registration_qc(assets["t1"], assets["t1"], assets["t1_mask"],
                    comparison="epi_epi", output_json=path,
                    transform_files=[self.trans_dir / "epi2t1_Composite.h5"])
                reports.append(path)
            return reports

        def _registration_outputs(self):
            return [self.trans_dir / ("epi2t1_" + suffix) for suffix in (
                "Composite.h5", "InverseComposite.h5", "Warped.nii", "InverseWarped.nii")]

    monkeypatch.setattr("fmri_rt_preproc.pipeline.FMRIRealtimePreprocessor", Preparation)
    with pytest.raises(ValueError, match="needs visual approval"):
        session.ensure_nf_session(cfg, settings)
    folder = day / "func" / "session_trans"
    marker = json.loads((folder / "session_alignment.json").read_text())
    assert marker["reference_run"] == "4"
    assert len(preparations) == 1
    for path in marker["qc_reports"]:
        approve_registration_qc(path, reviewer="Unit test reviewer")
    return cfg, settings, assets, folder, runner, stages, preparations


@pytest.mark.parametrize("offline", [False, True])
def test_first_nf_is_frozen_and_reused_across_runs_after_bound_review(frozen_session, offline):
    cfg, settings, assets, folder, runner, stages, preparations = frozen_session
    assert stages[0][0] == (cfg.incoming_dir, 4)
    assert stages[0][1] == {"expected": 20}
    if offline:
        settings.offline_assets = Path("assets.json")
    count = len(runner.calls)
    cfg.run = "7"
    session.ensure_nf_session(cfg, settings)
    assert len(preparations) == 1
    assert len(runner.calls) == count


def test_first_nf_uses_online_phase_encoding(frozen_session):
    _, settings, _, _, _, _, preparations = frozen_session
    assert preparations[0].cfg.epi_phase_encoding == settings.epi_phase_encoding


@pytest.mark.parametrize("change", ["source", "output", "policy", "initialization", "version", "approval", "omitted_report"])
def test_session_cache_rejects_changed_source_output_policy_or_review(frozen_session, change):
    cfg, settings, assets, folder, runner, stages, preparations = frozen_session
    if change == "source":
        assets["epi_to_t1"].write_text("changed transform")
    elif change == "output":
        (folder / "rt_WM_mask.nii").write_bytes(b"changed output")
    elif change == "policy":
        settings.registration_qc_policy = folder / "changed_policy.json"
        settings.registration_qc_policy.write_text("{}")
    elif change == "initialization":
        settings.registration_initialization = "center_of_mass"
    elif change in {"version", "omitted_report"}:
        path = folder / "session_alignment.json"
        marker = json.loads(path.read_text())
        if change == "version":
            marker["version"] = 999
        else:
            marker["qc_reports"] = marker["qc_reports"][:1]
        path.write_text(json.dumps(marker))
    else:
        approval = folder / "registration_qc_epi_epi.approval.json"
        approval.unlink()
    count = len(runner.calls)
    with pytest.raises(ValueError):
        session.ensure_nf_session(cfg, settings)
    assert len(runner.calls) == count
    assert len(preparations) == 1


def test_incomplete_session_does_not_adopt_unvalidated_outputs(tmp_path, monkeypatch):
    folder = tmp_path / "func" / "session_trans"
    folder.mkdir(parents=True)
    (folder / "epi2t1_Composite.h5").write_bytes(b"interrupted output")
    monkeypatch.setattr(session, "prepared_assets", lambda cfg: {})
    cfg = SimpleNamespace(preparation_only=False, day_root=tmp_path)
    settings = SimpleNamespace(offline_assets=None, registration_qc_policy=None,
                               registration_initialization="identity")
    with pytest.raises(ValueError, match="Incomplete session"):
        session.ensure_nf_session(cfg, settings)


def test_preparation_run_does_not_build_nf_reference(tmp_path):
    session.ensure_nf_session(SimpleNamespace(preparation_only=True), SimpleNamespace())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("space", ["t1", "mni"])
@pytest.mark.parametrize("preparation_only", [False, True])
def test_runtime_cpu_gpu_and_masks_resolve_the_selected_session_directory(tmp_path, monkeypatch, space, preparation_only):
    import rt_pipeline as rt
    import gpu_ants_like_resampler as gpu
    from fmri_rt_preproc.analysis_mask import mask_source, mask_transforms
    cfg = rt.RTSessionConfig("001", "day", "5", tmp_path / "incoming", tmp_path,
                            preparation_only=preparation_only)
    for name, offset in (("trans", 0.), ("session_trans", 20.)):
        folder = cfg.day_root / "func" / name
        affine = np.eye(4)
        affine[2, 3] = offset
        for filename in ("epi_unwarped_mean.nii", "epi_mask_mean.nii", "rt_WM_mask.nii", "rt_Vent_mask.nii"):
            _image(folder / filename, (7, 8, 4), affine, "mask" in filename)
        for filename in ("epi2t1_Composite.h5", "epi2t1_InverseComposite.h5", "t1_to_mni.h5"):
            (folder / filename).write_text(name + filename)
        _image(folder / "offline_anat" / "cortical_gm_mask.nii", (9, 10, 11), np.eye(4), True)
    session_dir = cfg.day_root / "func" / "session_trans"
    (session_dir / "session_alignment.json").write_text("{}")
    (session_dir / "session_to_reference_Composite.h5").write_text("session to stable EPI")
    _image(session_dir / "reference_epi.nii", (7, 8, 4), np.eye(4))
    selected = cfg.day_root / "func" / ("trans" if preparation_only else "session_trans")
    assert cfg.trans_dir == selected
    assert cfg.rt_motion_ref_epi == selected / "epi_unwarped_mean.nii"
    assert cfg.rt_motion_ref_mask == selected / "epi_mask_mean.nii"
    assert cfg.rt_wm_mask == selected / "rt_WM_mask.nii"
    assert cfg.rt_vent_mask == selected / "rt_Vent_mask.nii"
    assert mask_source(cfg, "cortical_gm")[0] == selected / "offline_anat" / "cortical_gm_mask.nii"
    assert mask_transforms(cfg, "t1", "epi") == (
        [] if preparation_only else [selected / "session_to_reference_Composite.h5"]
    ) + [selected / "epi2t1_InverseComposite.h5"]
    transforms = ([selected / "t1_to_mni.h5"] if space == "mni" else []) + [selected / "epi2t1_Composite.h5"]
    assert mask_transforms(cfg, "epi", space) == transforms
    cfg.decoder_template = _image(tmp_path / "decoder.nii", (9, 10, 11), np.eye(4))
    observed = []
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(use_gpu_resampler=True,
        analysis_space=space, gpu_resampler_device="cpu"))
    monkeypatch.setattr(gpu, "precompute_sampling_grid", lambda **kwargs: observed.append(kwargs) or tmp_path / "grid.npy")
    monkeypatch.setattr(gpu, "GpuAntsLikeResampler", lambda **kwargs: SimpleNamespace(device=kwargs["device"]))
    rt.maybe_init_gpu_resampler(cfg)
    assert observed[0]["moving_ref"] == cfg.rt_motion_ref_epi
    assert observed[0]["transforms"] == transforms
    assert observed[0]["fixed_ref"] == cfg.decoder_template
    assert selected in observed[0]["out_dir"].parents
    runner = FakeAnts()
    monkeypatch.setattr(rt, "run", runner)
    rt.final_volume_support(cfg, SimpleNamespace(gpu_resampler=None), np.ones((7, 8, 4), bool),
                            nib.load(cfg.rt_motion_ref_epi).affine, 1, space)
    args = runner.calls[0]
    assert [Path(args[i + 1]) for i, arg in enumerate(args) if arg == "-t"] == transforms
    assert Path(args[args.index("-r") + 1]) == cfg.decoder_template


def _prepared_day(tmp_path, monkeypatch, offline=False):
    from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
    assets = _assets(tmp_path / "source")
    cfg = SimpleNamespace(subject="001", day_root=tmp_path / "sub-001" / "day",
                          subject_root=tmp_path / "sub-001")
    folder = cfg.day_root / "func" / "trans"
    anatomy = folder / "offline_anat" if offline else cfg.subject_root / "anat"
    targets = {
        "epi_reference": folder / "epi_unwarped_mean.nii", "epi_mask": folder / "epi_mask_mean.nii",
        "t1": anatomy / "T1_N4.nii", "t1_mask": anatomy / "T1_combined_mask.nii",
        "wm_mask": folder / "WM_T1.nii", "vent_mask": folder / "Vent_T1.nii",
    }
    for key, path in targets.items():
        copy_image(assets[key], path)
    for key, name in (("epi_to_t1", "epi2t1_Composite.h5"),
                      ("t1_to_epi", "epi2t1_InverseComposite.h5"), ("t1_to_mni", "t1_to_mni.h5")):
        (folder / name).write_bytes(assets[key].read_bytes())
    pipe = object.__new__(FMRIRealtimePreprocessor)
    pipe.trans_dir, pipe.anat_dir = folder, anatomy
    pipe.rt_unwarped_analysis_ref_epi = targets["epi_reference"]
    pipe.rt_unwarped_analysis_ref_mask = targets["epi_mask"]
    pipe.cfg = SimpleNamespace(subject_root=cfg.subject_root, mni_template=assets["mni_template"])
    pipe.offline_assets = assets if offline else None
    if offline:
        assets["manifest"] = tmp_path / "source" / "assets.json"
        assets["manifest"].write_text('{"offline":true}')
    monkeypatch.setattr(session, "write_registration_reports", lambda *args, **kwargs: [])
    return cfg, pipe, assets


def test_preparation_metadata_preserves_template_without_saved_config(tmp_path, monkeypatch):
    cfg, pipe, assets = _prepared_day(tmp_path, monkeypatch)
    assert not (cfg.day_root / "config.json").exists()
    pipe._write_registration_qc()
    restored = session.prepared_assets(cfg)
    assert restored["mni_template"] == assets["mni_template"].resolve()
    assert restored["epi_reference"] == pipe.rt_unwarped_analysis_ref_epi
    assets["mni_template"].write_bytes(b"changed template")
    with pytest.raises(ValueError, match="MNI template changed"):
        session.prepared_assets(cfg)


@pytest.mark.parametrize("recipe", [None, "legacy_estimator"])
def test_same_day_reference_rejects_old_estimator_without_changing_outputs(tmp_path, monkeypatch, recipe):
    cfg, pipe, _ = _prepared_day(tmp_path, monkeypatch)
    pipe._write_registration_qc()
    marker = pipe.trans_dir / "preparation_assets.json"
    saved = json.loads(marker.read_text())
    saved["fieldmap_estimator_recipe"] = recipe
    marker.write_text(json.dumps(saved))
    transform = pipe.trans_dir / "epi2t1_Composite.h5"
    before = transform.read_bytes(), marker.read_bytes()
    with pytest.raises(ValueError, match="old or unrecorded fieldmap estimator"):
        session.prepared_assets(cfg)
    assert (transform.read_bytes(), marker.read_bytes()) == before


def test_frozen_session_rejects_old_estimator_before_reuse(frozen_session):
    cfg, settings, assets, folder, runner, stages, preparations = frozen_session
    marker = folder / "session_alignment.json"
    saved = json.loads(marker.read_text())
    saved["settings"].pop("fieldmap_estimator_recipe")
    marker.write_text(json.dumps(saved))
    before = len(runner.calls), len(stages), marker.read_bytes()
    with pytest.raises(ValueError, match="inputs/QC policy changed"):
        session.ensure_nf_session(cfg, settings)
    assert (len(runner.calls), len(stages), marker.read_bytes()) == before


@pytest.mark.parametrize("metadata", [True, False])
def test_implicit_nf_import_keeps_offline_reference_used_for_rs(tmp_path, monkeypatch, metadata):
    cfg, pipe, assets = _prepared_day(tmp_path, monkeypatch, offline=True)
    if metadata:
        pipe._write_registration_qc()
    else:
        (cfg.day_root / "config.json").write_text(json.dumps({"offline_assets": str(assets["manifest"])}))
    calls = []
    monkeypatch.setattr("fmri_rt_preproc.offline_assets.load_offline_assets",
        lambda path, subject: calls.append((path, subject)) or assets)
    restored = session.prepared_assets(cfg)
    assert restored is assets
    assert restored["epi_reference"] != pipe.rt_unwarped_analysis_ref_epi
    assert Path(calls[0][0]) == assets["manifest"]
    assert calls[0][1] == cfg.subject
    if metadata:
        assets["manifest"].write_text('{"offline":"changed"}')
        with pytest.raises(ValueError, match="offline manifest changed"):
            session.prepared_assets(cfg)
        assert len(calls) == 1
