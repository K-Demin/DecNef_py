"""Registration direction, image-grid and cache contracts without native ANTs.

Fake ANTs emits distinct forward/inverse transforms and different EPI/T1 grids:
an accidental swap or header relabel therefore cannot pass unnoticed.
"""
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

import test_rt_pipeline_parallel as headless  # headless RTPSpy dependencies
from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
from fmri_rt_preproc.analysis_mask import mask_transforms


def values(args, flag):
    return [args[i + 1] for i, value in enumerate(args) if value == flag]


@pytest.fixture
def registration(tmp_path, monkeypatch):
    pipe = object.__new__(FMRIRealtimePreprocessor)
    # Spaces exercise actual shell argument boundaries, including ANTs brackets.
    pipe.anat_dir = tmp_path / "subject anatomy"
    pipe.trans_dir = tmp_path / "session transforms"
    pipe.anat_dir.mkdir()
    pipe.trans_dir.mkdir()
    pipe.cfg = SimpleNamespace(registration_initialization="identity", registration_syn=False)
    epi = pipe.trans_dir / "epi_unwarped_mean.nii"
    mask = pipe.trans_dir / "epi_mask_mean.nii"
    epi_affine = np.diag([2., 2., 3., 1.])
    epi_affine[:3, 3] = [-4, -6, 7]
    for path in (epi, mask):
        nib.save(nib.Nifti1Image(np.ones((4, 5, 3), np.float32), epi_affine), path)
    for name in ("T1_N4.nii", "T1_brain.nii", "T1_combined_mask.nii"):
        nib.save(nib.Nifti1Image(np.ones((9, 10, 11), np.float32), np.eye(4)), pipe.anat_dir / name)
    calls = []

    def fake_ants(cmd):
        assert cmd[:2] == ["bash", "-lc"]
        args = shlex.split(cmd[2].splitlines()[-1])
        calls.append(args)
        assert args[0] == "antsRegistration"
        prefix, warped, inverse_warped = values(args, "--output")[0][1:-1].split(",")
        fixed, moving = values(args, "--metric")[0][3:-1].split(",")[:2]
        Path(prefix + "Composite.h5").write_bytes(b"native-forward")
        Path(prefix + "InverseComposite.h5").write_bytes(b"native-inverse")
        # Fixed-grid forward diagnostic, moving-grid inverse diagnostic.
        for path, reference, signal in ((warped, fixed, 11), (inverse_warped, moving, 22)):
            img = nib.load(reference)
            nib.save(nib.Nifti1Image(np.full(img.shape, signal, np.float32), img.affine), path)

    monkeypatch.setattr("fmri_rt_preproc.pipeline.run", fake_ants)
    return pipe, epi, mask, calls


@pytest.mark.parametrize("epi_shape", [(4, 5, 3), (4, 5, 6)], ids=["partial", "whole_brain"])
def test_semantic_directions_and_image_grids(registration, epi_shape):
    pipe, epi, mask, calls = registration
    for path in (epi, mask):
        img = nib.load(path)
        nib.save(nib.Nifti1Image(np.ones(epi_shape, np.float32), img.affine), path)
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    args = calls[0]
    t1 = pipe.anat_dir / "T1_brain.nii"
    fixed, moving = epi, t1
    stages = [value.split("[")[0] for value in values(args, "--transform")]
    assert stages == ["Translation", "Rigid"]
    expected_mask = f"[{mask},NULL]"
    assert values(args, "--masks") == [expected_mask] * len(stages)
    assert all(metric.startswith(f"MI[{fixed},{moving},") for metric in values(args, "--metric"))
    assert values(args, "--initial-moving-transform") == ["Identity"]
    assert values(args, "--shrink-factors") == ["1", "2x2x1"]
    forward, inverse, warped, inv_warped = pipe._registration_outputs()
    assert forward.read_bytes() == b"native-inverse"
    assert inverse.read_bytes() == b"native-forward"
    for path, reference, signal in ((warped, t1, 22), (inv_warped, epi, 11)):
        output, target = nib.load(path), nib.load(reference)
        assert output.shape == target.shape
        np.testing.assert_allclose(output.affine, target.affine)
        np.testing.assert_array_equal(output.get_fdata(), signal)
    # The same canonical names feed BOLD, cortical/custom masks and MNI chains.
    cfg = SimpleNamespace(trans_dir=pipe.trans_dir, subject_root=pipe.anat_dir.parent)
    assert mask_transforms(cfg, "epi", "t1") == [forward]
    assert mask_transforms(cfg, "t1", "epi") == [inverse]
    assert mask_transforms(cfg, "epi", "mni")[-1] == forward
    assert len(calls) == 1  # no fslcpgeom or extra resampling
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert len(calls) == 1  # valid repeat reuses the pair together
    saved = json.loads((pipe.trans_dir / "registration.json").read_text())
    assert saved["version"] == 2
    assert saved["native_fixed"] == str(epi.resolve())
    assert saved["native_moving"] == str(t1.resolve())
    assert saved["epi_to_t1_native"] == str((pipe.trans_dir / "t1_to_epi_InverseComposite.h5").resolve())


def test_optional_syn_and_explicit_com(registration):
    pipe, epi, mask, calls = registration
    pipe.cfg.registration_initialization = "center_of_mass"
    pipe.cfg.registration_syn = True
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    args = calls[0]
    assert values(args, "--transform") == ["Translation[0.1]", "Rigid[0.1]", "SyN[0.1,3,0]"]
    assert values(args, "--shrink-factors") == ["1", "2x2x1", "2x1"]
    assert values(args, "--initial-moving-transform") == [f"[{epi},{pipe.anat_dir / 'T1_brain.nii'},1]"]
    assert values(args, "--masks") == [f"[{mask},NULL]"] * 3


@pytest.mark.parametrize("syn", [False, True])
def test_published_transforms_feed_actual_wm_vent_and_mni_preparation(registration, monkeypatch, syn):
    pipe, epi, mask, _ = registration
    pipe.cfg.registration_syn = syn
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    pipe.rt_unwarped_analysis_ref_epi = epi
    pipe.rt_unwarped_analysis_ref_mask = mask
    pipe.fastsurfer_sid = "001"
    segmentation = pipe.anat_dir / "fastsurfer" / "001" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    segmentation.parent.mkdir(parents=True)
    segmentation.touch()  # existing binarized masks below avoid external mri_binarize
    for name in ("WM_T1.nii", "Vent_T1.nii"):
        nib.save(nib.load(pipe.anat_dir / "T1_N4.nii"), pipe.trans_dir / name)
    brain = nib.load(mask)
    brain_data = brain.get_fdata()
    brain_data[0] = 0
    nib.save(nib.Nifti1Image(brain_data, brain.affine), mask)
    pipe.decoder_template = pipe.anat_dir / "T1_N4.nii"
    pipe.cfg.mni_template = pipe.decoder_template
    (pipe.anat_dir / "warp_T1_to_MNI_synth.nii").write_bytes(b"T1-to-MNI")
    transforms = []

    def fake_apply(cmd, **kwargs):
        if cmd[0] == "bash":
            args = shlex.split(cmd[2].split("antsApplyTransforms", 1)[1])
            reference = nib.load(values(args, "-r")[0])
            chain = [Path(path).read_bytes() for path in values(args, "-t")]
            transforms.append((chain, values(args, "-n"), values(args, "-r")))
            nib.save(nib.Nifti1Image(np.ones(reference.shape), reference.affine), values(args, "-o")[0])
        elif cmd[0] == "fslmaths":
            source = nib.load(cmd[1], mmap=False)
            data = source.get_fdata()
            if "-mul" in cmd:
                data *= nib.load(cmd[cmd.index("-mul") + 1]).get_fdata()
            if "-bin" in cmd:
                data = (data > 0.5).astype(np.float32)
            nib.save(nib.Nifti1Image(data, source.affine), cmd[-1])
        else:
            pytest.fail(f"Unexpected command: {cmd}")

    monkeypatch.setattr("fmri_rt_preproc.pipeline.run", fake_apply)
    _, wm, vent = pipe._make_rtp_nuisance_masks()
    pipe._warp_epi_mean_to_mni(pipe.trans_dir, epi)
    t1_to_epi = b"native-forward"
    epi_to_t1 = b"native-inverse"
    assert transforms == [
        ([t1_to_epi], ["NearestNeighbor"], [str(epi)]),
        ([t1_to_epi], ["NearestNeighbor"], [str(epi)]),
        ([b"T1-to-MNI", epi_to_t1], ["Linear"], [str(pipe.decoder_template)]),
    ]
    for path in (wm, vent):
        img = nib.load(path)
        np.testing.assert_array_equal(img.get_fdata(), brain_data)
        np.testing.assert_array_equal(img.affine, brain.affine)


@pytest.mark.parametrize("change", ["syn", "initialization", "input", "output", "missing_inverse"])
def test_changed_cache_rejected_before_any_overwrite(registration, change):
    pipe, epi, mask, calls = registration
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    forward, inverse, _, _ = pipe._registration_outputs()
    if change == "syn":
        pipe.cfg.registration_syn = True
    elif change == "initialization":
        pipe.cfg.registration_initialization = "center_of_mass"
    elif change == "input":
        mask.write_bytes(mask.read_bytes() + b"changed")
    elif change == "output":
        forward.write_bytes(b"different transform")
    else:
        inverse.unlink()
    before = forward.read_bytes()
    with pytest.raises(ValueError, match="fresh day/preparation directory"):
        pipe.validate_registration_request()
    assert forward.read_bytes() == before
    assert len(calls) == 1


@pytest.mark.parametrize("cache_kind", ["unversioned", "full_brain", "unknown_version"])
def test_incompatible_cache_is_not_reinterpreted_or_overwritten(registration, cache_kind):
    pipe, epi, mask, calls = registration
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    marker = pipe.trans_dir / "registration.json"
    if cache_kind == "unversioned":
        marker.unlink()
    else:
        saved = json.loads(marker.read_text())
        saved["version"] = 1 if cache_kind == "full_brain" else 99
        saved["settings"] = {"mode": "full_brain", "initialization": "center_of_mass", "partial_epi_syn": False}
        marker.write_text(json.dumps(saved))
    original = {p: p.read_bytes() for p in pipe.trans_dir.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="fresh day/preparation directory"):
        pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert {p: p.read_bytes() for p in original} == original
    assert len(calls) == 1


@pytest.mark.parametrize("initialization,syn", [("identity", False), ("center_of_mass", True)])
def test_equivalent_version_one_partial_cache_reused_without_rewriting(registration, initialization, syn):
    pipe, epi, mask, calls = registration
    pipe.cfg.registration_initialization = initialization
    pipe.cfg.registration_syn = syn
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    marker = pipe.trans_dir / "registration.json"
    saved = json.loads(marker.read_text())
    saved["version"] = 1
    saved["settings"] = {"mode": "partial_epi", "initialization": initialization, "partial_epi_syn": syn}
    # Prior releases retain native outputs under their original prefix.
    for key in ("epi_to_t1_native", "t1_to_epi_native"):
        path = Path(saved[key])
        old_path = path.with_name(path.name.replace("t1_to_epi_", "t1_to_partial_epi_"))
        path.rename(old_path)
        saved[key] = str(old_path)
    marker.write_text(json.dumps(saved))
    original = {p: p.read_bytes() for p in pipe.trans_dir.iterdir() if p.is_file()}
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert len(calls) == 1
    assert {p: p.read_bytes() for p in pipe.trans_dir.iterdir() if p.is_file()} == original
    pipe.cfg.registration_syn = not syn
    with pytest.raises(ValueError, match="fresh day/preparation directory"):
        pipe.validate_registration_request()


def test_fresh_preflight_does_not_require_images(registration):
    pipe, _, _, calls = registration
    pipe.trans_dir = pipe.trans_dir / "fresh"
    pipe.anat_dir = pipe.anat_dir / "not_yet_created"
    pipe.validate_registration_request()
    assert not calls


def test_different_selected_reference_is_not_silently_reused(registration):
    pipe, epi, mask, calls = registration
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    other_epi = epi.with_name("another_reference.nii")
    nib.save(nib.load(epi), other_epi)
    with pytest.raises(ValueError, match="input selection changed"):
        pipe._register_epi_to_t1(pipe.trans_dir, other_epi, mask)
    assert len(calls) == 1


def test_interrupted_publication_cannot_be_reused(registration, monkeypatch):
    pipe, epi, mask, calls = registration
    from fmri_rt_preproc import pipeline
    copy = pipeline.shutil.copy2
    copies = []

    def interrupted_copy(source, target):
        copy(source, target)
        copies.append(target)
        if len(copies) == 2:
            raise OSError("interrupted publication")

    monkeypatch.setattr(pipeline.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="interrupted publication"):
        pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert [p.exists() for p in pipe._registration_outputs()] == [True, True, False, False]
    assert not (pipe.trans_dir / "registration.json").exists()
    with pytest.raises(ValueError, match="interrupted"):
        pipe.validate_registration_request()


@pytest.mark.parametrize("failure", ["missing_inverse", "wrong_grid", "empty_mask"])
def test_invalid_outputs_never_publish_transforms(registration, monkeypatch, failure):
    pipe, epi, mask, calls = registration
    from fmri_rt_preproc import pipeline
    fake_ants = pipeline.run
    if failure == "empty_mask":
        img = nib.load(mask)
        nib.save(nib.Nifti1Image(np.zeros(img.shape), img.affine), mask)
    else:
        def broken(cmd):
            fake_ants(cmd)
            if failure == "missing_inverse":
                (pipe.trans_dir / "t1_to_epi_InverseComposite.h5").unlink()
            else:
                nib.save(nib.load(epi), pipe.trans_dir / "t1_to_epi_InverseWarped.nii")
        monkeypatch.setattr(pipeline, "run", broken)
    with pytest.raises((RuntimeError, ValueError)):
        pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert not (pipe.trans_dir / "registration.json").exists()
    assert not any(p.exists() for p in pipe._registration_outputs())
