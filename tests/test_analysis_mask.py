import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc.analysis_mask import final_output_kind, mask_provenance
from fmri_rt_preproc.volume_smooth import MaskedGaussianSmoother, prepare_smoother
from rt_global_settings import RegressorSettings, save_regressor_settings, load_regressor_settings


def assets(tmp_path):
    shape = (4, 5, 6)
    cfg = SimpleNamespace(subject="001", subject_root=tmp_path,
        trans_dir=tmp_path / "trans", rt_work_dir=tmp_path / "run",
        rt_unwarped_analysis_ref_mask=tmp_path / "brain.nii")
    cfg.trans_dir.mkdir()
    (tmp_path / "anat").mkdir()
    for path in (cfg.trans_dir / "epi2t1_Composite.h5", cfg.trans_dir / "epi2t1_InverseComposite.h5",
                 tmp_path / "anat" / "warp_T1_to_MNI_synth.nii"):
        path.touch()
    brain = np.ones(shape, np.uint8)
    brain[0] = 0
    nib.save(nib.Nifti1Image(brain, np.eye(4)), cfg.rt_unwarped_analysis_ref_mask)
    labels = np.zeros(shape, np.int16)
    labels[0] = 1002  # cortex outside acquisition mask: must be removed
    labels[1] = 1002
    labels[2] = 2003
    labels[3, 0] = 2  # WM
    labels[3, 1] = 10  # subcortical GM
    labels[3, 2] = 8  # cerebellar GM
    labels[3, 3] = 4  # CSF
    labels[3, 4] = 1000  # unknown cortex
    segmentation = tmp_path / "anat" / "fastsurfer" / "001" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    segmentation.parent.mkdir(parents=True)
    nib.save(nib.MGHImage(labels, np.eye(4)), segmentation)
    reference = tmp_path / "reference.nii"
    nib.save(nib.Nifti1Image(np.zeros(shape), np.eye(4)), reference)
    return cfg, reference, labels, brain


@pytest.mark.parametrize("space", ["epi", "t1", "mni"])
@pytest.mark.parametrize("fwhm", [0, 4])
def test_cortex_selection_and_transform_chains(tmp_path, space, fwhm):
    cfg, reference, labels, brain = assets(tmp_path)
    calls = []

    def run(cmd):
        calls.append(cmd)
        assert cmd[cmd.index("-n") + 1] == "NearestNeighbor"
        source = nib.load(cmd[cmd.index("-i") + 1])
        nib.save(source, cmd[cmd.index("-o") + 1])  # identity transforms for synthetic data

    processor = prepare_smoother(cfg, fwhm, space, reference, run, mask_type="cortical_gm")
    expected = np.isin(labels, [1002, 2003]) & (brain > 0)
    np.testing.assert_array_equal(processor.mask, expected)
    chains = [[Path(cmd[i+1]).name for i, part in enumerate(cmd) if part == "-t"] for cmd in calls]
    assert chains == {"epi": [["epi2t1_InverseComposite.h5"]],
        "t1": [["epi2t1_Composite.h5"]],
        "mni": [["warp_T1_to_MNI_synth.nii"], ["warp_T1_to_MNI_synth.nii", "epi2t1_Composite.h5"]]}[space]
    data = np.where(expected, 7., 1e9)
    result = processor.apply(data)
    np.testing.assert_allclose(result[expected], 7.)
    assert np.all(result[~expected] == 0)
    kind = "smooth" if fwhm else "masked"
    assert (cfg.rt_work_dir / kind / "mask.nii").exists()


def test_zero_fwhm_masks_without_any_gaussian_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No filtering allowed in mask-only mode")

    monkeypatch.setattr("fmri_rt_preproc.volume_smooth.gaussian_filter", forbidden)
    mask = np.zeros((3, 4, 5), bool)
    mask[1] = True
    processor = MaskedGaussianSmoother(mask, np.eye(4), 0)
    data = np.arange(60).reshape(mask.shape)
    np.testing.assert_array_equal(processor.apply(data), np.where(mask, data, 0))


def test_custom_final_mask_and_geometry_errors(tmp_path):
    cfg, reference, _, brain = assets(tmp_path)
    custom = tmp_path / "custom.nii"
    mask = np.zeros(brain.shape, np.uint8)
    mask[:2] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), custom)
    processor = prepare_smoother(cfg, 0, "epi", reference, None,
        mask_type="custom", custom_file=custom)
    np.testing.assert_array_equal(processor.mask, (mask > 0) & (brain > 0))
    provenance = mask_provenance(cfg, "custom", "epi", reference, custom)
    assert provenance["source"]["path"] == str(custom.resolve())
    nib.save(nib.Nifti1Image(mask, np.diag([2., 2., 2., 1.])), custom)
    with pytest.raises(ValueError, match="grids differ"):
        prepare_smoother(cfg, 0, "epi", reference, None, mask_type="custom", custom_file=custom)
    nib.save(nib.Nifti1Image(np.ones(brain.shape) * .7, np.eye(4)), custom)
    with pytest.raises(ValueError, match="binary"):
        prepare_smoother(cfg, 0, "epi", reference, None, mask_type="custom", custom_file=custom)
    with pytest.raises(ValueError, match="requires"):
        prepare_smoother(cfg, 0, "epi", reference, None, mask_type="custom")


def test_no_overlap_rejected(tmp_path):
    cfg, reference, _, brain = assets(tmp_path)
    mask = (brain == 0).astype(np.uint8)
    custom = tmp_path / "outside.nii"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), custom)
    with pytest.raises(ValueError, match="no overlap"):
        prepare_smoother(cfg, 4, "epi", reference, None, mask_type="custom", custom_file=custom)


def test_settings_roundtrip_and_validation(tmp_path):
    settings = RegressorSettings()
    settings.update({"analysis_mask": "custom", "analysis_mask_file": tmp_path / "mask.nii",
                     "analysis_mask_space": "t1"})
    path = tmp_path / "settings.json"
    save_regressor_settings(path, settings)
    restored = load_regressor_settings(path)
    assert restored.analysis_mask == "custom"
    assert restored.analysis_mask_file == str(tmp_path / "mask.nii")
    assert restored.analysis_mask_space == "t1"
    assert final_output_kind(0, restored.analysis_mask) == "masked"
    assert final_output_kind(0) is None
    for key in ("analysis_mask", "analysis_mask_space"):
        with pytest.raises(ValueError):
            settings.update({key: "unknown"})


@pytest.mark.parametrize("space,kind", [("epi", "reg"), ("t1", "t1"), ("mni", "mni")])
def test_pca_routes_mask_only_output(tmp_path, space, kind):
    from rs_pca_runtime import volume_path_for_kind

    info = {"smoothing_fwhm_mm": 0, "analysis_mask": "cortical_gm", "regression": {"analysis_space": space}}
    (tmp_path / "session_metadata.json").write_text(json.dumps(info))
    expected = tmp_path / "masked" / "vol_00001_masked.nii"
    assert volume_path_for_kind(tmp_path, 1, kind) == expected
    assert volume_path_for_kind(tmp_path, 1, "masked") == expected
    assert volume_path_for_kind(tmp_path, 1, "mc").parent.name == "mc"


def test_run_cannot_switch_mask_or_mask_contents(tmp_path, monkeypatch):
    import rt_pipeline as rt
    from test_native_fieldmap import make_pair
    import os

    cfg, reference, _, _ = assets(tmp_path)
    cfg.rt_work_dir.mkdir()
    cfg.fmap_dir = make_pair(tmp_path / "pair")
    cfg.rt_motion_ref_epi = cfg.rt_unwarped_analysis_ref_epi = reference
    settings = SimpleNamespace(epi_phase_encoding="PA", analysis_space="epi", smoothing_fwhm_mm=0,
        analysis_mask="custom", analysis_mask_file=str(cfg.rt_unwarped_analysis_ref_mask), analysis_mask_space="final")
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", settings)
    rt.validate_run_provenance(cfg)
    rt.validate_run_provenance(cfg)
    settings.analysis_mask = "whole_brain"
    with pytest.raises(ValueError, match="changed"):
        rt.validate_run_provenance(cfg)
    settings.analysis_mask = "custom"
    mask_path = cfg.rt_unwarped_analysis_ref_mask
    stat = mask_path.stat()
    os.utime(mask_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed"):
        rt.validate_run_provenance(cfg)
