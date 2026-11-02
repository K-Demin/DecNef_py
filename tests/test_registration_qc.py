"""Registration QC must reject misleading background agreement and stale review."""
import json

import nibabel as nib
import numpy as np
import pytest

from fmri_rt_preproc.registration_qc import (
    approve_registration_qc, assert_qc_gate, build_registration_qc, inputs_match,
)


@pytest.fixture
def images(tmp_path):
    xyz = np.indices((17, 19, 21), dtype=float)
    radius = sum(((xyz[i] - [8, 9, 10][i]) / [7, 8, 9][i]) ** 2 for i in range(3))
    mask = radius < 1
    data = 30 + 4 * np.sin(xyz[0] / 1.3) + 7 * np.cos(xyz[1] / 1.9) + 3 * np.sin(xyz[2] / 2.1)
    data *= mask
    affine = np.diag([2., 2., 2.5, 1.])

    def save(name, values, matrix=None):
        path = tmp_path / (name + ".nii")
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32),
                               affine if matrix is None else matrix), path)
        return path

    return save, data, mask, [save("fixed", data), save("aligned", data), save("mask", mask)]


def test_same_contrast_is_masked_and_scale_invariant(images):
    save, data, mask, paths = images
    paths[1] = save("scaled", np.where(mask, 3 * data + 100, 0))
    report = build_registration_qc(*paths, comparison="epi_epi")
    assert report["metrics"]["pearson_r"] == pytest.approx(1)
    assert report["metrics"]["overlap_voxels"] == mask.sum()
    assert report["gate_status"] == "review_required"
    assert report["metrics"]["geometric_coverage_fraction"] is None


def test_shared_background_cannot_inflate_correlation(images):
    save, data, mask, paths = images
    shuffled = np.zeros_like(data)
    shuffled[mask] = np.random.default_rng(42).permutation(data[mask])
    paths[1] = save("shuffled", shuffled)
    raw = np.corrcoef(data.ravel(), shuffled.ravel())[0, 1]
    report = build_registration_qc(*paths, comparison="epi_epi")
    assert raw > .9  # Shared outside-brain zeros falsely suggest good agreement.
    assert abs(report["metrics"]["pearson_r"]) < .1


def test_cross_modal_negative_contrast_is_not_rejected_as_bad_pearson(images):
    save, data, mask, paths = images
    paths[1] = save("opposite_contrast", np.where(mask, 100 - data, 0))
    report = build_registration_qc(*paths, comparison="epi_t1")
    assert report["metrics"]["pearson_r"] is None
    assert report["metrics"]["nmi"] > 1.95
    assert report["metrics"]["edge_correlation"] == pytest.approx(1, abs=1e-10)
    assert not report["hard_failures"]
    assert report["gate_status"] == "review_required"
    with pytest.raises(ValueError, match="not an EPI-T1"):
        build_registration_qc(*paths, thresholds={"pearson_r_min": .8})


@pytest.mark.parametrize("kind", ["shape", "affine"])
def test_equal_array_shape_does_not_prove_registration(images, kind):
    save, data, _, paths = images
    affine = np.diag([2., 2., 2.5, 1.])
    if kind == "affine":
        affine[0, 3] = 4
    else:
        data = data[:-1]
    paths[1] = save("wrong_grid", data, affine)
    with pytest.raises(ValueError, match="grids differ"):
        build_registration_qc(*paths)


@pytest.mark.parametrize("kind", ["empty_mask", "no_signal", "constant", "nonfinite"])
def test_bad_data_is_hard_failure(images, kind):
    save, data, mask, paths = images
    if kind == "empty_mask":
        paths[2] = save("empty", np.zeros_like(mask))
    else:
        bad = data.copy()
        if kind == "no_signal":
            bad[:] = 0
        elif kind == "constant":
            bad[mask] = 10
        else:
            bad[8, 9, 10] = np.nan
        paths[1] = save("bad", bad)
    report = build_registration_qc(*paths)
    assert report["gate_status"] == "fail"
    assert report["hard_failures"]
    json.dumps(report, allow_nan=False)


def test_nonfinite_background_is_not_counted_as_brain_signal(images):
    save, data, mask, paths = images
    paths[1] = save("nan_background", np.where(mask, data, np.nan))
    report = build_registration_qc(*paths, comparison="epi_epi")
    assert not report["hard_failures"]
    assert report["metrics"]["pearson_r"] == pytest.approx(1)


def test_high_similarity_on_small_patch_does_not_hide_missing_coverage(images):
    save, data, mask, paths = images
    support = np.zeros_like(mask)
    support[:8] = True
    paths[1] = save("clipped", np.where(support, data, 0))
    coverage = save("coverage", support)
    report = build_registration_qc(*paths, comparison="epi_epi", coverage_file=coverage,
                                  thresholds={"pearson_r_min": .95, "signal_overlap_min": .9},
                                  thresholds_validated=True)
    assert report["metrics"]["pearson_r"] == pytest.approx(1)
    assert report["metrics"]["geometric_coverage_fraction"] < .5
    assert report["gate_status"] == "fail"
    assert report["threshold_failures"][0]["metric"] == "signal_overlap_fraction"


def test_auto_pass_requires_validated_similarity_and_coverage_policy(images):
    _, _, _, paths = images
    policy = {"pearson_r_min": .95, "signal_overlap_min": .9}
    assert build_registration_qc(*paths, comparison="epi_epi", thresholds=policy)["gate_status"] == "review_required"
    assert build_registration_qc(*paths, comparison="epi_epi", thresholds={"pearson_r_min": .95},
                                 thresholds_validated=True)["gate_status"] == "review_required"
    report = build_registration_qc(*paths, comparison="epi_epi", thresholds=policy, thresholds_validated=True)
    assert report["gate_status"] == "pass"


def test_missing_requested_metric_cannot_pass(images):
    _, _, _, paths = images
    report = build_registration_qc(*paths, thresholds={"geometric_coverage_min": .9})
    assert report["gate_status"] == "fail"
    assert report["threshold_failures"] == [{"metric": "geometric_coverage_fraction", "value": None, "minimum": .9}]


@pytest.mark.parametrize("policy", [{"made_up_min": .9}, {"signal_overlap_min": 1.1},
                                    {"nmi_min": float("nan")}, {"nmi_min": True}])
def test_invalid_threshold_configuration_is_rejected(images, policy):
    with pytest.raises(ValueError):
        build_registration_qc(*images[-1], thresholds=policy)


def test_repeatable_report_and_transform_hash_binding(images, tmp_path):
    _, _, _, paths = images
    transform = tmp_path / "Composite.h5"
    transform.write_bytes(b"first transform")
    first = build_registration_qc(*paths, transform_files=[transform])
    second = build_registration_qc(*paths, transform_files=[transform])
    assert first == second
    transform.write_bytes(b"other transform")
    assert not inputs_match(first)
    assert build_registration_qc(*paths, transform_files=[transform])["report_id"] != first["report_id"]


def test_visual_review_is_explicit_and_invalidated_by_input_change(images, tmp_path):
    save, data, _, paths = images
    path = tmp_path / "registration_qc.json"
    build_registration_qc(*paths, output_json=path)
    with pytest.raises(ValueError, match="visual approval"):
        assert_qc_gate([path])
    approve_registration_qc(path, reviewer="Lab operator", note="Checked aligned anatomy and coverage")
    assert len(assert_qc_gate(tmp_path)) == 1
    save("aligned", data * 2)
    with pytest.raises(ValueError, match="changed"):
        assert_qc_gate(path)


def test_changed_policy_invalidates_previous_review(images, tmp_path):
    path = tmp_path / "registration_qc.json"
    build_registration_qc(*images[-1], output_json=path)
    approve_registration_qc(path, reviewer="Lab operator")
    build_registration_qc(*images[-1], output_json=path, thresholds={"signal_overlap_min": .8})
    with pytest.raises(ValueError, match="visual approval"):
        assert_qc_gate(path)


def test_failed_report_cannot_be_approved(images, tmp_path):
    save, data, _, paths = images
    paths[1] = save("zero", np.zeros_like(data))
    path = tmp_path / "registration_qc.json"
    build_registration_qc(*paths, output_json=path)
    with pytest.raises(ValueError, match="cannot be approved"):
        approve_registration_qc(path, reviewer="Lab operator")
    with pytest.raises(ValueError, match="QC failed"):
        assert_qc_gate(path)


def test_edited_status_and_empty_report_bundle_cannot_bypass_gate(images, tmp_path):
    path = tmp_path / "registration_qc.json"
    report = build_registration_qc(*images[-1], output_json=path)
    report["gate_status"] = "pass"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        assert_qc_gate(path)
    with pytest.raises(ValueError, match="No registration QC"):
        assert_qc_gate([])


def test_orthogonal_review_figure_is_generated_headlessly(images, tmp_path):
    path = tmp_path / "registration_qc.png"
    report = build_registration_qc(*images[-1], review_png=path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 5000
    assert report["review_png"] == str(path.resolve())


def test_t1_template_comparison_uses_same_contrast_correlation(images):
    report = build_registration_qc(*images[-1], comparison="t1_t1",
                                  thresholds={"pearson_r_min": .95, "signal_overlap_min": .9},
                                  thresholds_validated=True)
    assert report["metrics"]["pearson_r"] == pytest.approx(1)
    assert report["gate_status"] == "pass"
    assert report["metrics"]["mask_dice"] is None


def test_check_registration_cli_requires_then_records_explicit_review(images, tmp_path, capsys):
    from check_registration import main
    build_registration_qc(*images[-1], output_json=tmp_path / "registration_qc_epi_t1.json")
    assert main(["--trans", str(tmp_path)]) == 1
    assert "visual approval" in capsys.readouterr().out
    assert main(["--trans", str(tmp_path), "--approve", "--reviewer", "Operator", "--note", "Inspected anatomy"]) == 0
    assert main(["--trans", str(tmp_path)]) == 0


def test_cli_preflights_failed_bundle_before_recording_any_approval(images, tmp_path):
    from check_registration import main
    save, data, _, paths = images
    build_registration_qc(*paths, output_json=tmp_path / "registration_qc_a.json")
    bad_paths = [paths[0], save("bad_bundle", data * 0), paths[2]]
    build_registration_qc(*bad_paths, output_json=tmp_path / "registration_qc_b.json")
    assert main(["--trans", str(tmp_path), "--approve", "--reviewer", "Operator"]) == 1
    assert not list(tmp_path.glob("*.approval.json"))


def test_legacy_decoder_signal_coverage_cannot_approve_missing_registration(images, tmp_path, capsys):
    from check_registration import main
    save, data, mask, _ = images
    save("epi2t1_Warped", data)
    weights = save("weights", mask)
    assert main(["--trans", str(tmp_path), "--weights", str(weights)]) == 1
    output = capsys.readouterr().out
    assert "100.0%" in output
    assert "No registration QC reports" in output


def test_cli_approval_requires_named_reviewer(tmp_path):
    from check_registration import main
    with pytest.raises(SystemExit) as error:
        main(["--trans", str(tmp_path), "--approve"])
    assert error.value.code == 2


@pytest.mark.parametrize("empty_decoder", [False, True])
def test_legacy_decoder_guard_still_blocks_low_coverage_approval(images, tmp_path, capsys, empty_decoder):
    from check_registration import main
    save, data, mask, paths = images
    build_registration_qc(*paths, output_json=tmp_path / "registration_qc_epi_t1.json")
    save("epi2t1_Warped", data * 0)
    weights = save("weights", np.zeros_like(mask) if empty_decoder else mask)
    assert main(["--trans", str(tmp_path), "--weights", str(weights),
                 "--approve", "--reviewer", "Operator"]) == 1
    assert "Decoder signal-coverage guard failed" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.approval.json"))
