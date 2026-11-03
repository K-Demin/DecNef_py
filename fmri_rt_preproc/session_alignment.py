"""One current-acquisition EPI alignment for all NF runs of a session."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import nibabel as nib

from .native_fieldmap import require_same_grid, ESTIMATOR_RECIPE
from .offline_assets import (apply_image, compose_transforms, file_identity, copy_image,
                             publish_mni_transform, t1_to_mni_transform)
from .registration import build_epi_bridge_registration, BRIDGE_RECIPE
from .epi_reference import (EPI_REFERENCE_N4_RECIPE, prepare_epi_registration_reference,
                            validate_epi_registration_reference, epi_reference_metadata_path)


def reference_preparation_outputs(folder):
    """Reference-only images/metadata bound to the fitted transform cache."""
    folder = Path(folder)
    paths = []
    for name in ("epi_registration_reference.nii", "reference_epi_registration.nii"):
        image = folder / name
        paths.extend(path for path in (image, epi_reference_metadata_path(image)) if path.is_file())
    return paths


def register_session_bridge(assets, epi, mask, folder, run_command, initialization="identity",
                            *, epi_reference_n4=True):
    if not isinstance(epi_reference_n4, bool):
        raise ValueError("epi_reference_n4 must be a boolean.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    copy_image(assets["epi_reference"], folder / "reference_epi.nii")
    copy_image(assets["epi_mask"], folder / "reference_epi_mask.nii")
    session_fit, reference_fit = epi, assets["epi_reference"]
    reference_policy = "original"
    if epi_reference_n4:
        session_fit = prepare_epi_registration_reference(
            epi, mask, folder / "epi_registration_reference.nii", run_command)
        prepared_reference = assets.get("epi_registration_reference")
        # Genuine fMRIPrep coreg references already include bias correction.
        # Older imported bundles may predate the optional registration-image key.
        if prepared_reference is None and (assets.get("provenance") or {}).get("source") == "fmriprep_explicit_import":
            prepared_reference = assets["epi_reference"]
        if prepared_reference is not None:
            require_same_grid(nib.load(str(prepared_reference)), nib.load(str(assets["epi_reference"])))
            reference_fit = copy_image(prepared_reference, folder / "reference_epi_registration.nii")
            reference_policy = "provided_bias_corrected_reference"
        else:
            reference_fit = prepare_epi_registration_reference(
                assets["epi_reference"], assets["epi_mask"],
                folder / "reference_epi_registration.nii", run_command)
            reference_policy = EPI_REFERENCE_N4_RECIPE
    plan = build_epi_bridge_registration(session_fit, reference_fit, mask, assets["epi_mask"],
                                         folder / "reference_to_session_", initialization=initialization)
    plan.provenance["epi_reference_n4"] = bool(epi_reference_n4)
    plan.provenance["reference_intensity_preparation"] = reference_policy
    run_command(plan.args)
    for output, reference in plan.output_references.items():
        require_same_grid(nib.load(str(output)), nib.load(str(reference)))
    for name, source in plan.semantic_outputs.items():
        if not source.is_file() or not source.stat().st_size:
            raise RuntimeError(f"Missing EPI bridge output: {source}")
        shutil.copy2(source, folder / name)
    bridge = folder / "session_to_reference_Composite.h5"
    inverse = folder / "session_to_reference_InverseComposite.h5"
    if epi_reference_n4:
        # Keep canonical diagnostics in original intensity units. Native ANTs
        # diagnostics retain the contrast-enhanced fitting images for inspection.
        apply_image(epi, assets["epi_reference"], folder / "session_to_reference_Warped.nii",
                    [bridge], run_command)
        apply_image(assets["epi_reference"], epi, folder / "session_to_reference_InverseWarped.nii",
                    [inverse], run_command)
    # ANTs transform CLI order is reverse image-mapping order.
    compose_transforms([assets["epi_to_t1"], bridge], assets["t1"],
                       folder / "epi2t1_Composite.h5", run_command)
    compose_transforms([inverse, assets["t1_to_epi"]], epi,
                       folder / "epi2t1_InverseComposite.h5", run_command)
    apply_image(epi, assets["t1"], folder / "epi2t1_Warped.nii",
                [folder / "epi2t1_Composite.h5"], run_command)
    apply_image(assets["t1"], epi, folder / "epi2t1_InverseWarped.nii",
                [folder / "epi2t1_InverseComposite.h5"], run_command)
    publish_mni_transform(assets["t1_to_mni"], folder)
    return plan


def write_registration_reports(folder, epi, epi_mask, t1, t1_mask, mni, t1_mni,
                               run_command, policy_path=None, bridge_reference_mask=None):
    from .registration_qc import build_registration_qc
    folder = Path(folder)
    policy = json.loads(Path(policy_path).read_text()) if policy_path else {}
    forward = folder / "epi2t1_Composite.h5"
    mask_t1 = folder / "epi_brain_in_t1.nii"
    apply_image(epi_mask, t1, mask_t1, [forward], run_command, "NearestNeighbor")
    reports = []

    def report(name, fixed, aligned, mask, transforms, comparison, aligned_mask=None):
        path = folder / ("registration_qc_" + name + ".json")
        build_registration_qc(fixed, aligned, mask, comparison=comparison,
                              aligned_mask_file=aligned_mask, transform_files=transforms,
                              thresholds=policy.get(name, {}),
                              thresholds_validated=policy.get("thresholds_validated", False),
                              output_json=path, review_png=path.with_suffix(".png"))
        reports.append(path)

    report("epi_t1", t1, folder / "epi2t1_Warped.nii", t1_mask, [forward], "epi_t1", mask_t1)
    reference_in_session = folder / "session_to_reference_InverseWarped.nii"
    if not reference_in_session.exists():
        reference_in_session = folder / "reference_to_session_Warped.nii"
    if reference_in_session.exists():
        mapped_mask = None
        if bridge_reference_mask is not None:
            mapped_mask = folder / "reference_brain_in_session.nii"
            apply_image(bridge_reference_mask, epi, mapped_mask,
                        [folder / "session_to_reference_InverseComposite.h5"], run_command, "NearestNeighbor")
        report("epi_epi", epi, reference_in_session, epi_mask,
               [folder / "session_to_reference_Composite.h5"], "epi_epi", mapped_mask)
    t1_in_mni = folder / "qc_t1_in_mni.nii"
    mask_mni = folder / "qc_t1_mask_in_mni.nii"
    apply_image(t1, mni, t1_in_mni, [t1_mni], run_command)
    apply_image(t1_mask, mni, mask_mni, [t1_mni], run_command, "NearestNeighbor")
    # Evaluation support is the transformed T1 brain mask, not an independent
    # template segmentation; omit aligned_mask to avoid a tautological Dice=1.
    report("t1_mni", mni, t1_in_mni, mask_mni, [t1_mni], "t1_t1")
    return reports


def prepared_assets(cfg):
    """Expose same-day preparation through the same contract as an offline bundle."""
    from .offline_assets import load_offline_assets
    folder = cfg.day_root / "func" / "trans"
    anatomy = folder / "offline_anat"
    if not anatomy.exists():
        anatomy = cfg.subject_root / "anat"
    metadata = folder / "preparation_assets.json"
    estimator_recipe = None
    if metadata.exists():
        recorded = json.loads(metadata.read_text())
        record = recorded["mni_template"]
        if recorded.get("version") != 1 or file_identity(record["path"]) != record:
            raise ValueError("Prepared MNI template changed; use a fresh preparation directory.")
        offline = recorded.get("offline_manifest")
        if offline:
            if file_identity(offline["path"]) != offline:
                raise ValueError("Prepared offline manifest changed; use a fresh preparation directory.")
            return load_offline_assets(offline["path"], cfg.subject)
        mni_template = Path(record["path"])
        estimator_recipe = recorded.get("fieldmap_estimator_recipe")
    else:
        config_path = cfg.day_root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("Missing preparation template metadata/config.json. "
                                    "Prepare a fresh directory with run_preproc.py.")
        config = json.loads(config_path.read_text())
        if config.get("offline_assets"):
            return load_offline_assets(config["offline_assets"], cfg.subject)
        mni_template = Path(config["templates"]["mni_t1"])
    if estimator_recipe != ESTIMATOR_RECIPE:
        raise ValueError("Same-day EPI preparation uses an old or unrecorded fieldmap estimator recipe. "
                         "Use a fresh preparation directory to rebuild its EPI reference and transforms.")
    result = {
        "epi_reference": folder / "epi_unwarped_mean.nii",
        "epi_mask": folder / "epi_mask_mean.nii", "t1": anatomy / "T1_N4.nii",
        "t1_mask": anatomy / "T1_combined_mask.nii", "wm_mask": folder / "WM_T1.nii",
        "vent_mask": folder / "Vent_T1.nii", "epi_to_t1": folder / "epi2t1_Composite.h5",
        "t1_to_epi": folder / "epi2t1_InverseComposite.h5",
        "t1_to_mni": t1_to_mni_transform(SimpleNamespace(trans_dir=folder, subject_root=cfg.subject_root)),
        "mni_template": mni_template,
    }
    cortex = anatomy / "cortical_gm_mask.nii"
    if cortex.exists():
        result["cortical_gm_mask"] = cortex
    for key, value in list(result.items()):
        if not value.exists() and value.suffix == ".nii" and value.with_suffix(".nii.gz").exists():
            result[key] = value.with_suffix(".nii.gz")
        if not result[key].exists():
            raise FileNotFoundError(f"Missing preparation asset {key}: {result[key]}. Run run_preproc.py first.")
    registration_reference = folder / "epi_registration_reference.nii"
    if not registration_reference.exists() and registration_reference.with_suffix(".nii.gz").exists():
        registration_reference = registration_reference.with_suffix(".nii.gz")
    if registration_reference.exists() or epi_reference_metadata_path(registration_reference).exists():
        validate_epi_registration_reference(result["epi_reference"], result["epi_mask"], registration_reference)
        result["epi_registration_reference"] = registration_reference
    return result


def ensure_nf_session(cfg, settings):
    """Build the first NF reference once, then enforce its session-bound QC gate.

    Scans 11-30 of the first reference acquisition supply a current-grid reference.
    A restart after reviewing QC reuses the frozen alignment. Subsequent NF
    runs use that same motion target; each incoming volume's grid is checked.
    """
    if getattr(cfg, "preparation_only", False):
        return
    from .offline_assets import load_offline_assets, bundle_identity
    from .registration_qc import assert_qc_gate
    from .utils import run
    folder = cfg.day_root / "func" / "session_trans"
    marker = folder / "session_alignment.json"
    assets = (load_offline_assets(settings.offline_assets, cfg.subject)
              if getattr(settings, "offline_assets", None) else prepared_assets(cfg))
    source_id = bundle_identity(assets)
    policy = getattr(settings, "registration_qc_policy", None)
    policy_id = file_identity(policy) if policy else None
    options = {"recipe": BRIDGE_RECIPE, "initialization": settings.registration_initialization,
               "fieldmap_estimator_recipe": ESTIMATOR_RECIPE,
               "fieldmap_method": getattr(settings, "fieldmap_method", "pyhysco"),
               "epi_phase_encoding": getattr(settings, "epi_phase_encoding", "PA"),
               "epi_reference_n4": getattr(settings, "epi_reference_n4", True),
               "epi_reference_n4_recipe": EPI_REFERENCE_N4_RECIPE
               if getattr(settings, "epi_reference_n4", True) else None}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved.get("version") != 1 or saved.get("settings") != options or
                saved["sources"] != source_id or saved["qc_policy"] != policy_id):
            raise ValueError("Session reference inputs/QC policy changed; prepare a new session directory.")
        expected_reports = {str((folder / ("registration_qc_" + name + ".json")).resolve())
                            for name in ("epi_epi", "epi_t1", "t1_mni")}
        if set(saved.get("qc_reports", [])) != expected_reports:
            raise ValueError("Session registration QC report set is incomplete or changed.")
        for record in saved["outputs"]:
            if file_identity(record["path"]) != record:
                raise ValueError("Session alignment output changed; prepare a new session directory.")
        assert_qc_gate([Path(p) for p in saved["qc_reports"]])
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete session preparation in {folder}; inspect it and use a fresh session directory.")
    folder.mkdir(parents=True, exist_ok=True)
    from run_preproc import _stage_epi_block, _wait_for_epi_dicoms
    from .config import SubjectDayConfig
    from .pipeline import FMRIRealtimePreprocessor
    from .native_fieldmap import find_source
    _wait_for_epi_dicoms(cfg.incoming_dir, int(cfg.run), expected=20)
    raw_dir = folder / "reference_input"
    files = _stage_epi_block(cfg.incoming_dir, int(cfg.run), raw_dir)
    merged = raw_dir / "epi_4d.nii.gz"
    run(["fslmerge", "-t", str(merged)] + [str(p) for p in files])
    prep_cfg = SubjectDayConfig.for_single_run(
        cfg.subject, cfg.day, cfg.day_root, str(cfg.run), merged,
        t1_file=assets["t1"], ap_file=find_source(cfg.fmap_dir, "AP"),
        pa_file=find_source(cfg.fmap_dir, "PA"), fieldmap_dir=cfg.fmap_dir,
        mni_template=assets["mni_template"],
        registration_direction=settings.registration_direction,
        registration_initialization=settings.registration_initialization,
        registration_syn=settings.registration_syn,
        epi_reference_n4=options["epi_reference_n4"])
    prep_cfg.registration_qc_policy = policy
    prep_cfg.fieldmap_method = options["fieldmap_method"]
    prep_cfg.epi_phase_encoding = options["epi_phase_encoding"]
    pipe = FMRIRealtimePreprocessor(prep_cfg)
    pipe.trans_dir, pipe.anat_dir, pipe.offline_assets = folder, folder / "offline_anat", assets
    pipe.rt_unwarped_analysis_ref_epi = folder / "epi_unwarped_mean.nii"
    pipe.rt_unwarped_analysis_ref_mask = folder / "epi_mask_mean.nii"
    pipe._prepare_anat()
    unwarped = raw_dir / "epi_unwarped.nii"
    pipe._unwarp_epi(merged, unwarped)
    pipe._motion_correct_and_mean(unwarped, pipe.rt_unwarped_analysis_ref_epi, 20)
    pipe._skullstrip_epi(pipe.rt_unwarped_analysis_ref_epi, folder / "epi_brain_mean.nii",
                        pipe.rt_unwarped_analysis_ref_mask)
    pipe._register_to_offline_reference(pipe.rt_unwarped_analysis_ref_epi,
                                        pipe.rt_unwarped_analysis_ref_mask)
    pipe._make_rtp_nuisance_masks()
    reports = pipe._write_registration_qc()
    outputs = list(pipe._registration_outputs()) + [pipe.rt_unwarped_analysis_ref_epi,
               pipe.rt_unwarped_analysis_ref_mask, folder / "rt_WM_mask.nii", folder / "rt_Vent_mask.nii",
               folder / "reference_epi.nii", folder / "reference_epi_mask.nii",
               folder / "session_to_reference_Composite.h5", folder / "session_to_reference_InverseComposite.h5",
               t1_to_mni_transform(SimpleNamespace(trans_dir=folder, subject_root=cfg.subject_root))]
    outputs += reference_preparation_outputs(folder)
    # The EPI/EPI QC image must also be immutable after review.
    outputs += [folder / "session_to_reference_InverseWarped.nii"]
    payload = {"version": 1, "reference_run": str(cfg.run), "sources": source_id, "settings": options,
               "qc_policy": policy_id, "qc_reports": [str(p.resolve()) for p in reports],
               "outputs": [file_identity(p) for p in outputs]}
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    assert_qc_gate(reports)
