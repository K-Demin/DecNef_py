"""N4 is confined to fitting copies; canonical images retain original units."""
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pytest

from test_registration_pipeline import registration, values
from test_session_alignment import bridge, frozen_session, _prepared_day
import fmri_rt_preproc.pipeline as pipeline
import fmri_rt_preproc.session_alignment as session
from fmri_rt_preproc.epi_reference import prepare_epi_registration_reference
from fmri_rt_preproc.offline_assets import file_identity


def n4_copy(args):
    source = nib.load(args[args.index("-i") + 1])
    nib.save(nib.Nifti1Image(source.get_fdata() * 2, source.affine),
             args[args.index("-o") + 1])


@pytest.mark.parametrize("direction", ["epi_to_t1", "t1_to_epi"])
def test_anatomical_fit_uses_n4_copy_but_keeps_original_mc_and_diagnostic_intensities(
        registration, monkeypatch, direction):
    pipe, epi, mask, fit_calls = registration
    pipe.cfg.registration_direction = direction
    pipe.cfg.epi_reference_n4 = True
    original_identity = file_identity(epi)
    previous_runner = pipeline.run
    additional = []

    def runner(args):
        if args[0] == "N4BiasFieldCorrection":
            additional.append(args)
            n4_copy(args)
        elif args[0] == "antsApplyTransforms":
            additional.append(args)
            source = nib.load(values(args, "-i")[0])
            target = nib.load(values(args, "-r")[0])
            nib.save(resample_from_to(source, target, order=0), values(args, "-o")[0])
        else:
            previous_runner(args)

    monkeypatch.setattr(pipeline, "run", runner)
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    corrected = pipe.trans_dir / "epi_registration_reference.nii"
    assert file_identity(epi) == original_identity
    np.testing.assert_array_equal(nib.load(corrected).get_fdata(), nib.load(epi).get_fdata() * 2)
    assert str(corrected) in values(fit_calls[0], "--metric")[0]
    # The canonical QC image is reapplied from raw mean, never from the N4 copy.
    application = [args for args in additional if args[0] == "antsApplyTransforms"]
    assert len(application) == 1
    assert values(application[0], "-i") == [str(epi)]
    assert values(application[0], "-t") == [str(pipe.trans_dir / "epi2t1_Composite.h5")]
    assert nib.load(pipe.trans_dir / "epi2t1_Warped.nii").get_fdata().max() == 1
    metadata = json.loads((pipe.trans_dir / "registration.json").read_text())
    assert metadata["settings"]["epi_reference_n4"] is True
    assert len(metadata["reference_preparation"]) == 2
    counts = len(additional), len(fit_calls)
    pipe._register_epi_to_t1(pipe.trans_dir, epi, mask)
    assert (len(additional), len(fit_calls)) == counts
    corrected.write_bytes(corrected.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="fresh day/preparation"):
        pipe.validate_registration_request()
    assert (len(additional), len(fit_calls)) == counts


@pytest.mark.parametrize("provided", ["none", "prepared", "fmriprep_legacy"])
def test_bridge_corrects_both_fit_references_without_recorrecting_prepared_inputs(bridge, provided):
    assets, epi, mask, folder, ants = bridge
    originals = {key: file_identity(path) for key, path in assets.items()}
    original_epi = file_identity(epi)
    if provided == "prepared":
        assets["epi_registration_reference"] = assets["epi_reference"]
    elif provided == "fmriprep_legacy":
        assets["provenance"] = {"source": "fmriprep_explicit_import"}
    n4_calls = []

    def runner(args):
        if args[0] == "N4BiasFieldCorrection":
            n4_calls.append(args)
            n4_copy(args)
        else:
            ants(args)

    plan = session.register_session_bridge(assets, epi, mask, folder, runner)
    assert len(n4_calls) == (2 if provided == "none" else 1)
    assert plan.fixed == folder / "epi_registration_reference.nii"
    assert plan.moving == folder / "reference_epi_registration.nii"
    assert file_identity(epi) == original_epi
    assert all(file_identity(assets[key]) == identity for key, identity in originals.items())
    np.testing.assert_array_equal(nib.load(folder / "reference_epi.nii").get_fdata(),
                                  nib.load(assets["epi_reference"]).get_fdata())
    for source, output in ((epi, "session_to_reference_Warped.nii"),
                           (assets["epi_reference"], "session_to_reference_InverseWarped.nii"),
                           (epi, "epi2t1_Warped.nii")):
        call = next(args for args in ants.calls if args[0] == "antsApplyTransforms"
                    and values(args, "-o") == [str(folder / output)])
        assert values(call, "-i") == [str(source)]


def test_qc_uses_original_intensity_bridge_diagnostic(bridge, monkeypatch):
    assets, epi, mask, folder, runner = bridge
    session.register_session_bridge(assets, epi, mask, folder, runner, epi_reference_n4=False)
    observed = []
    monkeypatch.setattr("fmri_rt_preproc.registration_qc.build_registration_qc",
                        lambda *args, **kwargs: observed.append((args, kwargs)))
    session.write_registration_reports(folder, epi, mask, assets["t1"], assets["t1_mask"],
                                        assets["mni_template"], assets["t1_to_mni"], runner)
    epi_report = next(args for args, kwargs in observed if kwargs["comparison"] == "epi_epi")
    assert epi_report[0] == epi
    assert epi_report[1] == folder / "session_to_reference_InverseWarped.nii"


def test_session_rejects_changed_n4_setting_before_reusing_approved_fit(frozen_session):
    cfg, settings, assets, folder, runner, stages, preparations = frozen_session
    assert preparations[0].cfg.epi_reference_n4 is False
    settings.epi_reference_n4 = True
    before = len(runner.calls)
    with pytest.raises(ValueError, match="inputs/QC policy changed"):
        session.ensure_nf_session(cfg, settings)
    assert len(runner.calls) == before


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_prepared_assets_reuses_verified_n4_copy(tmp_path, monkeypatch, suffix):
    cfg, pipe, assets = _prepared_day(tmp_path, monkeypatch)
    epi, mask = pipe.rt_unwarped_analysis_ref_epi, pipe.rt_unwarped_analysis_ref_mask
    corrected = pipe.trans_dir / ("epi_registration_reference" + suffix)
    prepare_epi_registration_reference(epi, mask, corrected, n4_copy)
    pipe._write_registration_qc()
    restored = session.prepared_assets(cfg)
    assert restored["epi_registration_reference"] == corrected
    assert restored["epi_reference"] == epi
    corrected.write_bytes(corrected.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="EPI N4"):
        session.prepared_assets(cfg)


def test_bridge_rejects_non_boolean_option_before_writes(bridge):
    assets, epi, mask, folder, runner = bridge
    with pytest.raises(ValueError, match="must be a boolean"):
        session.register_session_bridge(assets, epi, mask, folder, runner, epi_reference_n4="false")
    assert not folder.exists()
    assert not runner.calls
