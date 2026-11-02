"""Final-grid mask provenance and actual spatial/temporal processing together."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates, shift

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from fmri_rt_preproc.processing_mask import (
    FINAL_NUISANCE_SOURCE, FINAL_PROCESSING_MASK_POLICY,
    FINAL_PROCESSING_MASK_POLICY_VERSION, FINAL_PROCESSING_ORDER,
    PROCESSING_MASK_POLICY, canonical_upstream_processing, upstream_processing_spec,
)
from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
from fmri_rt_preproc.volume_smooth import MaskedGaussianSmoother
from test_native_fieldmap import make_pair
from test_regression_coverage import _wrapper


def _save(path, values, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32),
                           np.eye(4) if affine is None else affine), path)
    return path


@pytest.mark.parametrize("mode", ["mean", "zscore"])
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("fwhm", [0., 2.])
def test_final_grid_spatial_smoothing_glm_normalization(tmp_path, monkeypatch, mode, glm, fwhm):
    """Compare real temporal operators with interpolation/convolution/OLS math.

    Native and response grids differ in both shape and affine. Spatial mapping
    must see the acquired native values before temporal masking; cGLM must see
    the mapped, smoothed response. Missing slices remain unavailable throughout.
    """
    handler = headless._mk_handler(tmp_path)
    cfg = handler.cfg
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    cfg.trans_dir = tmp_path / "trans"
    cfg.trans_dir.mkdir()
    (cfg.trans_dir / "epi2t1_Composite.h5").touch()
    cfg.subject_root = tmp_path
    cfg.enable_original_score = True
    for name in ("mc", "reg", "unwarp"):
        folder = tmp_path / name
        folder.mkdir()
        setattr(cfg, "rt_" + name + "_dir", folder)

    native_shape, final_shape = (3, 4, 5), (2, 3, 4)
    final_affine = np.eye(4)
    final_affine[:3, 3] = (.25, 1., 1.)
    coordinates = np.indices(final_shape, dtype=float)
    coordinates += final_affine[:3, 3, None, None, None]

    def spatial(values):
        return map_coordinates(np.asarray(values, float), coordinates, order=1,
                               mode="constant", cval=0., prefilter=False)

    acquired = np.ones(native_shape, bool)
    acquired[..., 4:] = False
    brain = np.zeros(final_shape, bool)
    brain[1:, :, 1:] = True
    processing = brain.copy()
    processing[:, 0, :] = False  # A custom response selection excludes acquired brain.
    final_coverage = spatial(acquired) >= rt.SUPPORT_THRESHOLD
    valid = processing & final_coverage
    handler.anatomical_brain_mask = np.ones(native_shape, bool)
    handler.final_brain_mask = brain.copy()
    handler.processing_mask = processing.copy()
    handler.voxel_normalizer = VoxelNormalizer(mode, "fixed", 3, processing)
    handler.motion_regressor = (_wrapper(tmp_path, processing, n=4, motion="None",
        scale_before_glm=False, normalization_mode=mode) if glm else None)
    if glm:
        handler.motion_regressor._regress.mask_src_proc = handler.proc_src
        handler.motion_regressor._regress.startup_ref_volumes = 3 if mode == "zscore" else None
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda data: data)
    handler.volreg = SimpleNamespace(do_proc=lambda *a, **k: None,
        _motion=np.zeros((12, 6)), resample_support=lambda *a: acquired.astype(float))
    handler.biopac_receiver = handler.gpu_resampler = None
    handler.smoother = MaskedGaussianSmoother(processing, final_affine, fwhm)
    (tmp_path / handler.smoother.output_kind).mkdir()
    handler.scorer = SimpleNamespace(mask=processing.copy())
    previews = []
    handler.volume_streamer = SimpleNamespace(publish=lambda *args: previews.append(args))

    gain = np.linspace(.8, 1.5, np.prod(native_shape)).reshape(native_shape)
    pattern = np.indices(native_shape).sum(axis=0)
    course = np.array([1000., 1100., 950., 1300., 900., 1050., 1500., 980., 1250., 1020.])
    native = np.asarray([value * gain + 80 * np.sin(pattern + index)
                         for index, value in enumerate(course)], np.float32)
    cfg.rt_motion_ref_epi = _save(tmp_path / "reference.nii", native[0])
    cfg.decoder_template = _save(tmp_path / "target.nii", np.ones(final_shape), final_affine)
    transformed_sources = []

    def run(command):
        assert command[0] == "antsApplyTransforms"
        assert command[command.index("-n") + 1] == "Linear"
        source = Path(command[command.index("-i") + 1])
        destination = Path(command[command.index("-o") + 1])
        data = nib.load(source).get_fdata()
        assert data.shape == native_shape  # No second warp of final-grid temporal output.
        transformed_sources.append(source.parent.name)
        _save(destination, spatial(data), final_affine)

    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space="t1", volume_stream_kind="mc"))

    def smooth_oracle(values):
        result = np.zeros(final_shape)
        if fwhm == 0:
            result[valid] = values[valid]
        else:
            sigma = fwhm / np.sqrt(8 * np.log(2))
            numerator = gaussian_filter(np.where(valid, values, 0.), sigma,
                                        mode="constant", cval=0., truncate=4.)
            density = gaussian_filter(valid.astype(float), sigma,
                                      mode="constant", cval=0., truncate=4.)
            result[valid] = numerator[valid] / density[valid]
        return result

    spatial_samples = np.asarray([np.where(brain & final_coverage,
        spatial(np.where(acquired, frame, 0)), 0) for frame in native])
    response = np.asarray([smooth_oracle(frame) for frame in spatial_samples], np.float32)
    history = []
    became_ready = False
    for index, frame in enumerate(native, 1):
        raw = _save(tmp_path / f"raw_{index}.nii", frame)
        glm_ready = not glm or index >= 4
        if glm and glm_ready:
            design = np.column_stack((np.ones(index), np.linspace(-1., 1., index)))
            samples = response[:index, valid].astype(float)
            residuals = samples - design @ np.linalg.lstsq(design, samples, rcond=None)[0]
            values = residuals[-1]
        else:
            values = response[index - 1, valid]
        expected_reg = np.zeros(final_shape)
        expected_reg[valid] = values
        expected_norm = np.zeros(final_shape)
        ready = glm_ready and len(history) >= 3
        if ready:
            prior = np.asarray(history)
            expected_norm[valid] = (100 * values / prior.mean(axis=0) if mode == "mean"
                else (values - prior.mean(axis=0)) / prior.std(axis=0, ddof=1))
        assert rt.process_volume(cfg, handler, raw, index, raw_nii=raw, volume_timestamp=float(index))
        for folder, kind, expected in (
            ("t1", "t1", spatial_samples[index - 1]),
            (handler.smoother.output_kind, handler.smoother.output_kind, response[index - 1]),
            ("reg", "reg", expected_reg), ("norm", "norm", expected_norm)):
            saved = nib.load(tmp_path / folder / f"vol_{index:05d}_{kind}.nii")
            assert saved.shape == final_shape
            np.testing.assert_array_equal(saved.affine, final_affine)
            np.testing.assert_allclose(saved.get_fdata(), expected, rtol=1e-4, atol=.003)
        saved_support = nib.load(tmp_path / "support" / f"vol_{index:05d}_t1.nii")
        np.testing.assert_array_equal(saved_support.get_fdata(), valid if ready else np.zeros(final_shape))
        np.testing.assert_array_equal(saved_support.affine, final_affine)
        original = nib.load(tmp_path / "t1" / f"vol_{index:05d}_t1_orig.nii").get_fdata()
        np.testing.assert_allclose(original, spatial_samples[index - 1], rtol=1e-6)
        mc = nib.load(cfg.rt_mc_dir / f"vol_{index:05d}_mc.nii").get_fdata()
        np.testing.assert_array_equal(mc, np.where(acquired, frame, 0))
        np.testing.assert_array_equal(handler.prev_mc_for_dvars, mc)
        assert previews[-1][1] == cfg.rt_mc_dir / f"vol_{index:05d}_mc.nii"
        became_ready |= ready
        if glm and mode == "zscore" and index == 4:
            history = list(residuals[:3].copy())
        elif len(history) < 3 and (mode == "mean" or glm_ready):
            history.append(response[index - 1, valid].copy() if mode == "mean" else values.copy())

    assert became_ready
    assert set(transformed_sources) == {"mc", "coverage"}
    assert len(transformed_sources) == len(native) * 3  # response, optional original, acquisition support
    with (tmp_path / "coverage_rt.csv").open() as stream:
        last = list(csv.DictReader(stream))[-1]
    assert int(last["brain_voxels"]) == brain.sum()
    assert int(last["acquired_brain_voxels"]) == (brain & final_coverage).sum()
    assert int(last["usable_brain_voxels"]) == valid.sum()
    assert int(last["usable_decoder_voxels"]) == valid.sum()
    assert handler.voxel_normalizer.count[valid].min() == 3
    np.testing.assert_array_equal(handler.voxel_normalizer.count[~valid], 0)
    for name in ("mean", "std", "count"):
        saved = nib.load(tmp_path / "norm" / f"reference_{name}.nii")
        assert saved.shape == final_shape
        np.testing.assert_array_equal(saved.affine, final_affine)


def test_lanczos_availability_uses_linear_and_persisted_anatomical_mask(tmp_path):
    shape = (5, 5, 5)
    acquired = np.ones(shape, bool)
    acquired[-1] = False
    brain = np.zeros(shape, bool)
    brain[1:4, 1:4, 1:4] = True
    reference = _save(tmp_path / "reference.nii", np.ones(shape))
    cfg = SimpleNamespace(rt_work_dir=tmp_path, decoder_template=reference)
    calls = []

    def linear(values):
        calls.append(values.copy())
        return shift(values, (.25, 0, 0), order=1, mode="constant", cval=0, prefilter=False)

    def forbidden(*args):
        raise AssertionError("Strict Lanczos contributor rejection must not define availability")

    handler = SimpleNamespace(final_brain_mask=brain, gpu_resampler=SimpleNamespace(
        interpolation="lanczos", resample_support=linear, resample_coverage=forbidden))
    availability = rt.final_volume_support(cfg, handler, acquired, np.eye(4), 1, "t1")
    expected = shift(acquired.astype(float), (.25, 0, 0), order=1,
                     mode="constant", cval=0, prefilter=False) >= rt.SUPPORT_THRESHOLD
    np.testing.assert_array_equal(availability, expected)
    final = rt.final_volume_support(cfg, handler, acquired, np.eye(4), 1, "t1",
                                    folder_name="support", mapped_support=availability)
    np.testing.assert_array_equal(final, expected & brain)
    np.testing.assert_array_equal(nib.load(tmp_path / "support" / "vol_00001_t1.nii").get_fdata(), final)
    assert len(calls) == 1
    np.testing.assert_array_equal(availability, expected)


@pytest.mark.parametrize("kernel", ["linear", "lanczos"])
@pytest.mark.parametrize("space,bridge", [("t1", False), ("epi", False), ("epi", True)])
@pytest.mark.parametrize("selected", [False, True])
def test_preparation_uses_final_grid_and_analysis_selection(tmp_path, monkeypatch, kernel, space, bridge, selected):
    native_shape = (9, 8, 7)
    brain = np.zeros(native_shape, np.uint8)
    brain[2:7, 2:6, 2:5] = 1
    mask_path = _save(tmp_path / "brain.nii", brain)
    source = _save(tmp_path / "reference.nii", np.ones(native_shape))
    spatial = space == "t1" or bridge
    final_shape = (5, 6, 4) if spatial else native_shape
    final_affine = np.diag([2., 2., 2., 1.]) if spatial else np.eye(4)
    final_brain = np.zeros(final_shape, bool) if spatial else brain.astype(bool)
    if spatial:
        final_brain[1:4, 1:5, 1:3] = True
    selection = np.ones(final_shape, bool)
    selection[2:] = False
    trans = tmp_path / "trans"
    trans.mkdir()
    (trans / "epi2t1_Composite.h5").touch()
    target = _save(tmp_path / "target.nii", np.ones(final_shape), final_affine)
    if bridge:
        (trans / "session_to_reference_Composite.h5").touch()
        _save(trans / "reference_epi.nii", np.ones(final_shape), final_affine)
    cfg = SimpleNamespace(rt_work_dir=tmp_path / "run", trans_dir=trans, subject_root=tmp_path,
        rt_unwarped_analysis_ref_mask=mask_path, rt_unwarped_analysis_ref_epi=source,
        decoder_template=target)
    commands = []

    def run(command):
        commands.append(command)
        assert command[command.index("-n") + 1] == "NearestNeighbor"
        np.testing.assert_array_equal(nib.load(command[command.index("-i") + 1]).get_fdata(), brain)
        _save(Path(command[command.index("-o") + 1]), final_brain, final_affine)

    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        analysis_space=space, gpu_resampler_interpolation=kernel))
    anatomical, processing, final, path = rt.prepare_processing_masks(cfg, selection if selected else None)
    expected = final_brain & selection if selected else final_brain
    np.testing.assert_array_equal(anatomical, brain)
    np.testing.assert_array_equal(final, final_brain)
    np.testing.assert_array_equal(processing, expected)
    assert path.name == "processing_final.nii"
    for filename, values, affine in (("brain_epi.nii", brain, np.eye(4)),
        ("brain_final.nii", final_brain, final_affine), ("processing_final.nii", expected, final_affine)):
        saved = nib.load(path.parent / filename)
        np.testing.assert_array_equal(saved.get_fdata(), values)
        np.testing.assert_array_equal(saved.affine, affine)
    assert len(commands) == int(spatial)
    policy = json.loads((path.parent / "policy.json").read_text())
    assert policy["policy"] == FINAL_PROCESSING_MASK_POLICY
    assert policy["version"] == FINAL_PROCESSING_MASK_POLICY_VERSION
    assert policy["margin_vox"] == 0
    assert policy["nuisance_margin_vox"] == (3 if kernel == "lanczos" else 1) * spatial
    assert policy["nuisance_source"] == FINAL_NUISANCE_SOURCE
    assert policy["processing_order"] == list(FINAL_PROCESSING_ORDER)
    assert policy["interpolation"] == kernel
    assert policy["final_brain_mask_interpolation"] == "NearestNeighbor"
    assert policy["availability_interpolation"] == "Linear"
    assert policy["brain_voxels"] == final_brain.sum()
    assert policy["processing_voxels"] == expected.sum()
    old_source_identity = policy["brain_mask"]
    brain[1, 1, 1] = 1
    _save(mask_path, brain)
    assert rt.processing_mask_spec(cfg)["brain_mask"] != old_source_identity
    with pytest.raises(ValueError, match="grids differ"):
        rt.prepare_processing_masks(cfg, np.ones((2, 2, 2), bool))


def _provenance(policy=FINAL_PROCESSING_MASK_POLICY):
    return {"version": 1, "input_stage": "normalized", "normalization": {"mode": "zscore"},
        "coverage_policy": "observed_support", "processing_mask": {
            "policy": policy, "version": 1, "interpolation": "linear",
            "processing_order": list(FINAL_PROCESSING_ORDER), "nuisance_source": FINAL_NUISANCE_SOURCE,
            "brain_mask": {"path": "acquisition-specific.nii"}, "nuisance_margin_vox": 1}}


def test_final_and_legacy_policies_remain_distinct_and_ignore_acquisition_identity():
    current = _provenance()
    canonical = canonical_upstream_processing(current)
    assert canonical["processing_mask"] == {
        "policy": FINAL_PROCESSING_MASK_POLICY, "version": 1, "interpolation": "linear",
        "processing_order": list(FINAL_PROCESSING_ORDER), "nuisance_source": FINAL_NUISANCE_SOURCE}
    changed_acquisition = copy.deepcopy(current)
    changed_acquisition["processing_mask"].update(brain_mask={"path": "other.nii"}, nuisance_margin_vox=3)
    assert canonical_upstream_processing(changed_acquisition) == canonical
    legacy = canonical_upstream_processing(_provenance(PROCESSING_MASK_POLICY))
    assert legacy != canonical
    assert legacy["processing_mask"] == {
        "policy": PROCESSING_MASK_POLICY, "version": 1, "interpolation": "linear"}
    metadata = {key: value for key, value in current.items() if key not in {"version", "input_stage"}}
    assert upstream_processing_spec(metadata) == canonical


@pytest.mark.parametrize("key,value", [
    ("processing_order", None), ("processing_order", ["denoising", "spatial_transform", "normalization"]),
    ("nuisance_source", None), ("nuisance_source", "final_masked_response"), ("version", 2)])
def test_final_grid_provenance_requires_order_and_native_nuisance_source(key, value):
    spec = _provenance()
    spec["processing_mask"][key] = value
    with pytest.raises(ValueError, match="order/nuisance-source"):
        canonical_upstream_processing(spec)
