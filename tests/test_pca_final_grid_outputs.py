"""PCA must consume final normalized features after spatial-first processing."""
import csv
import json
from queue import Queue
from threading import Event

import nibabel as nib
import numpy as np
import pytest

import roi_rs_pca_decoder_prep as prep
import rs_pca_runtime as runtime
from fmri_rt_preproc.processing_mask import (
    FINAL_PROCESSING_MASK_POLICY, FINAL_PROCESSING_MASK_POLICY_VERSION,
    FINAL_NUISANCE_SOURCE, FINAL_PROCESSING_ORDER,
)


def _metadata(run_dir, *, space="t1", mapped=False, smoothing=0, mask="whole_brain"):
    info = {
        "temporal_processing_order": "spatial_smooth_glm_norm_v1",
        "final_output": {"folder": "norm", "suffix": "norm", "space": space},
        "regression": {"analysis_space": space}, "epi_reference_mapped": mapped,
        "smoothing_fwhm_mm": smoothing, "analysis_mask": mask,
        "normalization": {"version": 3, "stage": "after_spatial_smoothing_glm",
                          "space": space, "mode": "zscore", "reference": "fixed"},
        "coverage_policy": "test_separate_support",
        "processing_mask": {"policy": FINAL_PROCESSING_MASK_POLICY,
                            "version": FINAL_PROCESSING_MASK_POLICY_VERSION, "interpolation": "linear",
                            "processing_order": list(FINAL_PROCESSING_ORDER),
                            "nuisance_source": FINAL_NUISANCE_SOURCE},
    }
    (run_dir / "session_metadata.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def _series(run_dir, kind, values):
    folder = run_dir / kind
    folder.mkdir(exist_ok=True)
    for index, value in enumerate(values, 1):
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), value, np.float32), np.eye(4)),
                 folder / f"vol_{index:05d}_{kind}.nii")


@pytest.fixture
def merge_locally(monkeypatch):
    def merge(volumes, output):
        images = [nib.load(path) for path in volumes]
        nib.save(nib.Nifti1Image(np.stack([image.get_fdata() for image in images], axis=-1),
                               images[0].affine), output)
        return True

    monkeypatch.setattr(prep, "_run_fslmerge", merge)
    monkeypatch.setattr(prep, "FORCE_REBUILD_4D", False)
    monkeypatch.setattr(prep, "AUTO_FSLMERGE_4D", True)


@pytest.mark.parametrize("space,mapped,spatial", [
    ("epi", False, "mc"), ("epi", True, "epi_ref"), ("t1", False, "t1"), ("mni", False, "mni"),
])
def test_final_aliases_select_norm_and_tsnr_uses_original_final_grid(
        tmp_path, merge_locally, space, mapped, spatial):
    _metadata(tmp_path, space=space, mapped=mapped)
    _series(tmp_path, "norm", [0, 0, 2])
    _series(tmp_path, "reg", [0, 0, 200])
    _series(tmp_path, spatial, [100, 110, 120])
    aliases = ["reg", "norm"] + ([spatial] if spatial != "mc" else [])
    for alias in aliases:
        assert runtime.volume_path_for_kind(tmp_path, 3, alias) == tmp_path / "norm" / "vol_00003_norm.nii"
    for alias in ["auto", *aliases]:
        pca, tsnr = prep._discover_rs_inputs(tmp_path, alias)
        assert prep._pca_reference_space(tmp_path, alias) == space
        assert pca.name == "rs_4d_norm.nii.gz"
        assert tsnr.name == f"rs_4d_{spatial}.nii.gz"
        np.testing.assert_array_equal(nib.load(pca).get_fdata()[0, 0, 0], [0, 0, 2])
        np.testing.assert_array_equal(nib.load(tsnr).get_fdata()[0, 0, 0], [100, 110, 120])


@pytest.mark.parametrize("kind,smoothing,mask", [("smooth", 4, "whole_brain"), ("masked", 0, "cortical_gm")])
def test_spatial_stage_alias_cannot_select_pre_denoising_input(tmp_path, merge_locally, kind, smoothing, mask):
    _metadata(tmp_path, smoothing=smoothing, mask=mask)
    _series(tmp_path, "norm", [0, 0, -2])
    _series(tmp_path, "t1", [100, 101, 102])
    _series(tmp_path, kind, [999, 999, 999])
    assert runtime.volume_path_for_kind(tmp_path, 3, kind).parent.name == "norm"
    pca, tsnr = prep._discover_rs_inputs(tmp_path, kind)
    np.testing.assert_array_equal(nib.load(pca).get_fdata()[0, 0, 0], [0, 0, -2])
    assert tsnr.name == "rs_4d_t1.nii.gz"


def test_existing_spatial_preview_does_not_replace_missing_normalized_output(tmp_path, merge_locally):
    _metadata(tmp_path)
    _series(tmp_path, "t1", [100, 101, 102])
    assert not runtime.volume_path_for_kind(tmp_path, 1, "t1").exists()
    with pytest.raises(FileNotFoundError, match="No norm/"):
        prep._discover_rs_inputs(tmp_path, "t1")


def test_normalized_residuals_cannot_replace_missing_tsnr_source(tmp_path, merge_locally):
    _metadata(tmp_path)
    _series(tmp_path, "norm", [0, 0, 2])
    with pytest.raises(FileNotFoundError, match="original-intensity.*tSNR"):
        prep._discover_rs_inputs(tmp_path, "reg")


@pytest.mark.parametrize("kind", ["mc", "unwarped"])
def test_explicit_raw_mc_selection_is_preserved(tmp_path, merge_locally, kind):
    _metadata(tmp_path, space="t1", smoothing=4)
    _series(tmp_path, "mc", [100, 101, 102])
    assert runtime.volume_path_for_kind(tmp_path, 1, kind) == tmp_path / "mc" / "vol_00001_mc.nii"
    pca, tsnr = prep._discover_rs_inputs(tmp_path, kind)
    assert pca == tsnr and pca.name == "rs_4d_mc.nii.gz"


def test_bridged_epi_still_rejects_current_session_mc_for_online_pca(tmp_path):
    _metadata(tmp_path, space="epi", mapped=True)
    with pytest.raises(ValueError, match="stable reference EPI grid"):
        runtime.volume_path_for_kind(tmp_path, 1, "mc")


@pytest.mark.parametrize("kind", ["mni", "epi_ref", "smooth", "masked"])
def test_disabled_or_different_grid_alias_is_not_silently_decoded(tmp_path, kind):
    _metadata(tmp_path, space="t1")
    with pytest.raises(ValueError):
        runtime.volume_path_for_kind(tmp_path, 1, kind)
    with pytest.raises(ValueError):
        prep._discover_rs_inputs(tmp_path, kind)


@pytest.mark.parametrize("declaration", [None, {}, {"folder": "t1", "suffix": "t1", "space": "t1"},
                                        {"folder": "norm", "suffix": "norm", "space": "epi"}])
def test_new_order_requires_consistent_final_output_metadata(tmp_path, declaration):
    info = _metadata(tmp_path)
    info["final_output"] = declaration
    (tmp_path / "session_metadata.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="final_output"):
        runtime.volume_path_for_kind(tmp_path, 1, "reg")
    with pytest.raises(ValueError, match="final_output"):
        prep._discover_rs_inputs(tmp_path, "auto")


def test_final_grid_pca_still_excludes_warmup_and_keeps_first_ready_volume(tmp_path, merge_locally):
    _metadata(tmp_path)
    _series(tmp_path, "norm", [0, 0, 0, 2])
    _series(tmp_path, "t1", [100, 101, 102, 103])
    (tmp_path / "regression_status_rt.csv").write_text(
        "volume_idx,reg_ready,glm_ready,normalization_ready\n"
        "1,0,0,0\n2,0,1,0\n3,1,1,1\n4,1,1,1\n", encoding="utf-8")
    pca, _ = prep._discover_rs_inputs(tmp_path, "auto")
    keep, meta = prep._initial_keep_mask(tmp_path, 4, "auto", pca)
    np.testing.assert_array_equal(keep, [False, False, True, True])
    assert meta["drop_first_ready"] is False
    assert meta["upstream_processing"]["normalization"]["space"] == "t1"


def test_older_normalized_order_keeps_its_previous_final_folder(tmp_path):
    info = _metadata(tmp_path, smoothing=4)
    info.pop("temporal_processing_order")
    (tmp_path / "session_metadata.json").write_text(json.dumps(info), encoding="utf-8")
    assert runtime.volume_path_for_kind(tmp_path, 1, "t1") == tmp_path / "smooth" / "vol_00001_smooth.nii"
    assert prep._pca_reference_space(tmp_path, "reg") == "reg"


@pytest.mark.parametrize("wrong_grid", [False, True])
def test_mni_mask_is_never_sent_through_t1_to_epi_transform(tmp_path, monkeypatch, wrong_grid):
    reference = nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4))
    affine = np.eye(4)
    if wrong_grid:
        affine[0, 3] = 10
    mask_path = tmp_path / "roi.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), affine), mask_path)
    monkeypatch.setattr(prep, "_find_epi2t1_composite", lambda _: pytest.fail("MNI must not use T1-to-EPI"))
    kwargs = dict(roi="ROI", roi_mask_path=mask_path, tsnr_map_3d=reference.get_fdata(),
                  tsnr_img_3d=reference, run_dir=tmp_path, roi_out_dir=tmp_path, reference_space="mni")
    if wrong_grid:
        with pytest.raises(ValueError, match="final MNI reference grid"):
            prep._load_mask_in_epi_space(**kwargs)
    else:
        mask, actual_affine = prep._load_mask_in_epi_space(**kwargs)
        assert mask.all()
        np.testing.assert_array_equal(actual_affine, reference.affine)


@pytest.mark.parametrize("kind", ["norm", "epi_ref"])
def test_all_roi_final_aliases_keep_existing_readiness_filter(tmp_path, monkeypatch, kind):
    import test_rt_pipeline_parallel  # headless launcher imports
    import rs_pca_score_all_rois as all_rois

    _metadata(tmp_path, space="epi", mapped=True)
    _series(tmp_path, "norm", [0, 0, 4, 6, 8])
    (tmp_path / "regression_status_rt.csv").write_text(
        "volume_idx,reg_ready,glm_ready,normalization_ready\n"
        "1,0,0,0\n2,0,1,0\n3,1,1,1\n4,1,1,1\n5,1,1,1\n", encoding="utf-8")
    pca_root = tmp_path / "pca"
    roi = pca_root / "ROI"
    roi.mkdir(parents=True)
    np.savez(roi / "decoder_bundle.npz", voxel_indices=np.arange(8),
             weights=np.eye(1, 8, dtype=np.float32), norm_mean=np.zeros(8, np.float32),
             norm_std=np.ones(8, np.float32))
    (roi / "decoder_metadata.json").write_text(json.dumps({
        "upstream_processing": runtime.run_upstream_processing(tmp_path, kind)}), encoding="utf-8")
    monkeypatch.setattr(all_rois.time, "sleep", lambda _: pytest.fail("Unexpected readiness wait"))
    score_root = tmp_path / "scores"
    all_rois.run_realtime_all_roi_pca_scorer(
        run_dir=tmp_path, pca_root=pca_root, score_root=score_root, score_queue=Queue(),
        stop_event=Event(), volume_kind=kind, score_label="PC01", normalization="zscore",
        score_metric="projection", target_pc="PC01", top_pcs="auto", top_pc_variance=.1,
        max_trs=5, poll_interval=.01, qc_plot_path=None, qc_update_every=0, metadata={})
    with (score_root / "scores_pca_all_rois.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Preserve the all-ROI scorer's existing first-ready exclusion, too.
    assert [int(row["volume_idx"]) for row in rows] == [4, 5]
    assert [float(row["ROI_PC01"]) for row in rows] == [6, 8]
    saved = json.loads((score_root / "scores_pca_all_rois_metadata.json").read_text())
    assert saved["initial_volume_filter"]["skipped_not_ready"] == 2
    assert saved["upstream_processing"]["input_stage"] == "normalized"


@pytest.mark.parametrize("kind", ["norm", "epi_ref", "mni"])
def test_explicit_final_alias_requires_readiness_without_filename_inference(kind):
    assert prep._requires_regression_ready_filter(kind)
