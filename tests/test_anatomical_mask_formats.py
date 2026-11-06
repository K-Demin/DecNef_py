"""Online anatomy preparation accepts FSL's compressed NIfTI outputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import nibabel as nib
import numpy as np
import pytest

matplotlib.use("Agg")

import test_rt_pipeline_parallel as headless  # shared headless native/UI imports
import fmri_rt_preproc.pipeline as pipeline
from test_registration_pipeline import registration, values


def save(path, value=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full((4, 5, 6), value, np.float32), np.eye(4)), path)
    return path


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("existing_binarized,existing_filled", [
    (False, False), (True, False), (False, True),
], ids=["fresh", "missing-filled", "orphan-filled"])
def test_brain_mask_generation_resolves_fsl_outputs_and_partial_cache(
        tmp_path, monkeypatch, compressed, existing_binarized, existing_filled):
    pipe = object.__new__(pipeline.FMRIRealtimePreprocessor)
    pipe.anat_dir = tmp_path / "anat"
    pipe.anat_dir.mkdir()
    pipe.fastsurfer_sid = "88888"
    fastsurfer = pipe.anat_dir / "fastsurfer"
    segmentation = fastsurfer / pipe.fastsurfer_sid / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    segmentation.parent.mkdir(parents=True)
    segmentation.touch()
    t1 = save(pipe.anat_dir / "T1_N4.nii")
    if existing_binarized:
        save(pipe.anat_dir / "brainmask_noCSF.nii")
    calls = []

    def output_path(path):
        path = Path(path)
        return path.with_suffix(".nii.gz") if compressed and not path.name.endswith(".nii.gz") else path

    if existing_filled:
        # An orphaned fill must not survive regeneration of its source mask.
        save(output_path(pipe.anat_dir / "brainmask_noCSF_filled.nii"), value=4)

    def fake_run(command):
        calls.append(command)
        if command[0] == "mri_binarize":
            save(command[command.index("--o") + 1])
        elif command[0] == "mri_synthstrip":
            save(command[command.index("-o") + 1])
            save(command[command.index("-m") + 1])
        elif command[0] == "flirt":
            # Unlike FSL's basename lookup, this assertion requires Python to
            # pass the actual file produced by the earlier FSL command.
            assert Path(command[command.index("-in") + 1]).is_file()
            save(output_path(command[command.index("-out") + 1]))
        elif command[0] == "fslmaths":
            assert Path(command[1]).is_file()
            if "-mul" in command:
                assert Path(command[command.index("-mul") + 1]).is_file()
            save(output_path(command[-1]))
        else:
            pytest.fail(f"Unexpected anatomy command: {command}")

    monkeypatch.setattr(pipeline, "run", fake_run)
    pipe._make_brain_masks(t1, fastsurfer)
    combined = output_path(pipe.anat_dir / "T1_combined_mask.nii")
    filled = output_path(pipe.anat_dir / "brainmask_noCSF_filled.nii")
    assert combined.is_file() and filled.is_file()
    assert sum("-fillh" in command for command in calls) == 1
    assert sum(command[0] == "mri_binarize" for command in calls) == int(not existing_binarized)
    assert sum(command[0] == "flirt" for command in calls) == 1
    np.testing.assert_array_equal(nib.load(filled).get_fdata(), 1)
    np.testing.assert_array_equal(nib.load(combined).get_fdata(), 1)
    if compressed:
        assert not combined.with_suffix("").exists()
    count = len(calls)
    pipe._make_brain_masks(t1, fastsurfer)
    assert len(calls) == count  # Both compressed and uncompressed cache reuse.

    # The top-level online preparation must recognize the same complete cache.
    pipe.offline_assets = None
    pipe.cfg = SimpleNamespace(t1_file=t1)
    save(pipe.anat_dir / "warp_T1_to_MNI_synth.nii")
    save(pipe.anat_dir / "T1_warped_to_MNI_synth.nii")
    monkeypatch.setattr(pipe, "_make_brain_masks", lambda *args: pytest.fail("Complete mask cache was rebuilt"))
    monkeypatch.setattr(pipe, "_run_fastsurfer", lambda *args: pytest.fail("Segmentation was rerun"))
    monkeypatch.setattr(pipe, "_run_synthmorph", lambda *args: pytest.fail("SynthMorph was rerun"))
    pipe._prepare_anat()
    assert len(calls) == count


@pytest.mark.parametrize("direction", ["t1_to_epi", "epi_to_t1"])
def test_registration_and_provenance_accept_compressed_anatomical_mask(registration, direction):
    pipe, epi, mask, calls = registration
    pipe.cfg.registration_direction = direction
    uncompressed = pipe.anat_dir / "T1_combined_mask.nii"
    compressed = uncompressed.with_suffix(".nii.gz")
    nib.save(nib.load(uncompressed), compressed)
    uncompressed.unlink()
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    expected = f"[{mask},{compressed}]" if direction == "t1_to_epi" else f"[{compressed},{mask}]"
    assert values(calls[0], "--masks") == [expected] * 2
    provenance = json.loads((pipe.trans_dir / "registration.json").read_text())
    assert str(compressed.resolve()) in [record["path"] for record in provenance["inputs"]]
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert len(calls) == 1


def test_registration_qc_receives_actual_compressed_anatomical_mask(tmp_path, monkeypatch):
    import fmri_rt_preproc.session_alignment as alignment

    pipe = object.__new__(pipeline.FMRIRealtimePreprocessor)
    pipe.anat_dir = tmp_path / "anat"
    pipe.trans_dir = tmp_path / "trans"
    pipe.trans_dir.mkdir()
    pipe.offline_assets = None
    pipe.cfg = SimpleNamespace(subject_root=tmp_path, mni_template=save(tmp_path / "mni.nii"),
                               registration_qc_policy=None)
    pipe.rt_unwarped_analysis_ref_epi = save(pipe.trans_dir / "epi.nii")
    pipe.rt_unwarped_analysis_ref_mask = save(pipe.trans_dir / "epi_mask.nii")
    save(pipe.anat_dir / "T1_N4.nii")
    combined = save(pipe.anat_dir / "T1_combined_mask.nii.gz")
    save(pipe.anat_dir / "warp_T1_to_MNI_synth.nii")
    calls = []

    def fake_reports(*args):
        calls.append(args)
        return "reports"

    monkeypatch.setattr(alignment, "write_registration_reports", fake_reports)
    assert pipe._write_registration_qc() == "reports"
    assert calls[0][4] == combined


def test_brain_mask_generation_fails_at_source_if_fsl_produces_no_mask(tmp_path, monkeypatch):
    pipe = object.__new__(pipeline.FMRIRealtimePreprocessor)
    pipe.anat_dir = tmp_path
    pipe.fastsurfer_sid = "88888"
    for name in ("brainmask_noCSF.nii", "brainmask_noCSF_filled.nii", "T1_brain.nii", "T1_mask_skull.nii"):
        save(tmp_path / name)
    monkeypatch.setattr(pipeline, "run", lambda *args: None)
    with pytest.raises(FileNotFoundError, match="Brain mask generation produced neither"):
        pipe._make_brain_masks(tmp_path / "T1_N4.nii", tmp_path / "fastsurfer")
