"""Exercise real temporal processing with a wider interpolation region."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import shift

import test_rt_pipeline_parallel as headless
import rt_pipeline as rt
from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
from test_native_fieldmap import make_pair
from test_regression_coverage import _wrapper


def _save(path, values):
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4)), path)
    return path


@pytest.mark.parametrize("mode", ["mean", "zscore"])
@pytest.mark.parametrize("glm", [False, True])
def test_temporal_halo_reaches_warp_but_not_final_brain(tmp_path, monkeypatch, mode, glm):
    """Compare actual GLM/normalization against independent least-squares math.

    The native scanner/registration libraries are replaced with identity MC and
    a known fractional spatial shift; the temporal stages run their real code.
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

    shape = (3, 4, 5)
    brain = np.zeros(shape, bool)
    brain[1, 1:3, 1:4] = True
    processing = np.ones(shape, bool)  # The one-voxel margin fills this tiny FOV.
    acquired = np.ones(shape, bool)
    acquired[..., 3:] = False  # Includes one anatomical slice and halo slices.
    valid = processing & acquired
    halo = valid & ~brain
    handler.anatomical_brain_mask = brain.copy()
    # A quarter-voxel shift keeps nearest-neighbour brain membership unchanged.
    handler.final_brain_mask = brain.copy()
    handler.processing_mask = processing.copy()
    handler.voxel_normalizer = VoxelNormalizer(mode, "fixed", 3, processing)
    handler.motion_regressor = (_wrapper(tmp_path, processing, n=4, motion="None",
                                        scale_before_glm=False) if glm else None)
    if glm:
        handler.motion_regressor._regress.mask_src_proc = handler.proc_src
    handler.pyhysco_applier = SimpleNamespace(apply_volume=lambda data: data)
    handler.volreg = SimpleNamespace(do_proc=lambda *a, **k: None,
        _motion=np.zeros((12, 6)), resample_support=lambda *a: acquired.astype(float))
    handler.biopac_receiver = None
    handler.gpu_resampler = handler.smoother = None
    handler.scorer = SimpleNamespace(mask=brain.copy())
    previews = []
    handler.volume_streamer = SimpleNamespace(publish=lambda *args: previews.append(args))

    gain = np.linspace(.8, 1.5, np.prod(shape)).reshape(shape)
    offset = np.arange(np.prod(shape)).reshape(shape) / 30.
    course = np.array([1000., 1100., 950., 1300., 900., 1050., 1500., 980., 1250., 1020.])
    native = np.asarray([value * gain + offset for value in course], np.float32)
    cfg.rt_motion_ref_epi = cfg.decoder_template = _save(tmp_path / "reference.nii", native[0])
    transformed_inputs = []

    def spatial(values):
        return shift(values, (.25, 0, 0), order=1, mode="constant", cval=0, prefilter=False)

    def run(command):
        assert command[0] == "antsApplyTransforms"
        assert command[command.index("-n") + 1] == "Linear"
        source = Path(command[command.index("-i") + 1])
        destination = Path(command[command.index("-o") + 1])
        data = nib.load(source).get_fdata()
        if source.parent.name == "norm":
            transformed_inputs.append(data.copy())
        _save(destination, spatial(data))

    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        epi_phase_encoding="PA", save_intermediate_unwarped=False,
        enable_fd_censor_reg=False, enable_dvars_censor_reg=False,
        biopac_timelag=False, analysis_space="t1", volume_stream_kind="mc"))
    history = []
    expected_final = (spatial(valid.astype(float)) >= rt.SUPPORT_THRESHOLD) & brain
    expected_orig = (spatial(acquired.astype(float)) >= rt.SUPPORT_THRESHOLD) & brain
    became_ready = False
    nonzero_halo_reached_warp = False
    for index, frame in enumerate(native, 1):
        raw = _save(tmp_path / f"raw_{index}.nii", frame)
        glm_ready = not glm or index >= 4
        if glm and glm_ready:
            design = np.column_stack((np.ones(index), np.linspace(-1., 1., index)))
            samples = native[:index, valid].astype(float)
            values = samples[-1] - design[-1] @ np.linalg.lstsq(design, samples, rcond=None)[0]
        else:
            values = frame[valid]
        expected_reg = np.zeros(shape)
        expected_reg[valid] = values
        expected_norm = np.zeros(shape)
        ready = glm_ready and len(history) >= 3
        if ready:
            prior = np.asarray(history)
            expected_norm[valid] = (100 * values / prior.mean(axis=0) if mode == "mean"
                else (values - prior.mean(axis=0)) / prior.std(axis=0, ddof=1))
        assert rt.process_volume(cfg, handler, raw, index, raw_nii=raw, volume_timestamp=float(index))
        reg = nib.load(cfg.rt_reg_dir / f"vol_{index:05d}_reg.nii").get_fdata()
        np.testing.assert_allclose(reg, expected_reg, rtol=1e-5, atol=.003)
        norm = nib.load(tmp_path / "norm" / f"vol_{index:05d}_norm.nii").get_fdata()
        np.testing.assert_allclose(norm, expected_norm, rtol=1e-4, atol=.003)
        np.testing.assert_allclose(transformed_inputs[-1], expected_norm, rtol=1e-4, atol=.003)
        final = nib.load(tmp_path / "t1" / f"vol_{index:05d}_t1.nii").get_fdata()
        np.testing.assert_allclose(final, np.where(expected_final, spatial(expected_norm), 0),
                                   rtol=1e-4, atol=.003)
        np.testing.assert_array_equal(final[~brain], 0)
        saved_support = nib.load(tmp_path / "support" / f"vol_{index:05d}_t1.nii").get_fdata()
        np.testing.assert_array_equal(saved_support, expected_final if ready else np.zeros(shape))
        original = nib.load(tmp_path / "t1" / f"vol_{index:05d}_t1_orig.nii").get_fdata()
        np.testing.assert_allclose(original,
            np.where(expected_orig, spatial(np.where(acquired, frame, 0)), 0), rtol=1e-6)
        # The unmasked acquired MC image remains a distinct viewer product.
        mc = nib.load(cfg.rt_mc_dir / f"vol_{index:05d}_mc.nii").get_fdata()
        np.testing.assert_array_equal(mc, np.where(acquired, frame, 0))
        assert previews[-1][1] == cfg.rt_mc_dir / f"vol_{index:05d}_mc.nii"
        if ready:
            became_ready = True
            nonzero_halo_reached_warp |= bool(np.any(np.abs(transformed_inputs[-1][halo]) > .01))
        if len(history) < 3 and (mode == "mean" or glm_ready):
            history.append(frame[valid].copy() if mode == "mean" else values.copy())

    assert became_ready
    assert nonzero_halo_reached_warp
    with (tmp_path / "coverage_rt.csv").open() as stream:
        last = list(csv.DictReader(stream))[-1]
    assert int(last["brain_voxels"]) == brain.sum()
    assert int(last["usable_brain_voxels"]) == (brain & acquired).sum()
    assert float(last["usable_brain_fraction"]) == pytest.approx(2 / 3)
    assert int(last["usable_decoder_voxels"]) == expected_final.sum()
    assert handler.voxel_normalizer.count[halo].min() == 3


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
    # Reusing an acquisition array must not mutate it into an anatomical mask.
    np.testing.assert_array_equal(availability, expected)


@pytest.mark.parametrize("kernel,space,bridge,margin", [
    ("linear", "t1", False, 1), ("lanczos", "t1", False, 3),
    ("lanczos", "epi", False, 0), ("lanczos", "epi", True, 3)])
def test_preparation_wires_source_padding_and_nn_anatomy_and_records_policy(
        tmp_path, monkeypatch, kernel, space, bridge, margin):
    shape = (9, 9, 9)
    brain = np.zeros(shape, np.uint8)
    brain[4, 4, 4] = 1
    mask_path = _save(tmp_path / "brain.nii", brain)
    source = _save(tmp_path / "reference.nii", np.ones(shape))
    trans = tmp_path / "trans"
    trans.mkdir()
    (trans / "epi2t1_Composite.h5").touch()
    if bridge:
        (trans / "session_to_reference_Composite.h5").touch()
        _save(trans / "reference_epi.nii", np.ones(shape))
    cfg = SimpleNamespace(rt_work_dir=tmp_path / "run", trans_dir=trans,
        subject_root=tmp_path,
        rt_unwarped_analysis_ref_mask=mask_path, rt_unwarped_analysis_ref_epi=source,
        decoder_template=source)
    commands = []

    def run(command):
        commands.append(command)
        assert command[command.index("-n") + 1] == "NearestNeighbor"
        source_data = nib.load(command[command.index("-i") + 1]).get_fdata()
        _save(Path(command[command.index("-o") + 1]), source_data)

    monkeypatch.setattr(rt, "run", run)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(
        analysis_space=space, gpu_resampler_interpolation=kernel))
    anatomical, processing, final, path = rt.prepare_processing_masks(cfg)
    np.testing.assert_array_equal(anatomical, brain)
    np.testing.assert_array_equal(final, brain)
    assert processing.sum() == (2 * margin + 1) ** 3
    assert processing[4, 4, 4]
    np.testing.assert_array_equal(nib.load(path).get_fdata(), processing)
    np.testing.assert_array_equal(nib.load(mask_path).get_fdata(), brain)
    assert len(commands) == int(space == "t1" or bridge)
    policy = json.loads((path.parent / "policy.json").read_text())
    assert policy["margin_vox"] == margin
    assert policy["interpolation"] == kernel
    assert policy["final_brain_mask_interpolation"] == "NearestNeighbor"
    assert policy["availability_interpolation"] == "Linear"
    assert policy["brain_voxels"] == 1
    assert policy["processing_voxels"] == processing.sum()
    # Source identity detects a changed anatomical mask even with the same path.
    old_source_identity = policy["brain_mask"]
    brain[5, 4, 4] = 1
    _save(mask_path, brain)
    assert rt.processing_mask_spec(cfg)["brain_mask"] != old_source_identity
