"""Reference-run spatial preparation and validation for subsequent NF runs."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import nibabel as nib

from .native_fieldmap import require_same_grid, ESTIMATOR_RECIPE
from .offline_assets import (apply_image, compose_transforms, file_identity, copy_image,
                             publish_mni_transform, t1_to_mni_transform)
from .registration import build_epi_bridge_registration
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


def validate_prepared_reference(cfg, settings):
    """Require the spatial files made by this subject/day's reference run.

    NF only consumes these files. Importing anatomy, collecting reference EPIs
    and fitting a bridge belong to RS/behavioral reference preparation, including
    when that preparation uses an offline anatomical bundle.
    """
    folder = cfg.day_root / "func" / "trans"
    instruction = f"Run the RS/behavioral reference preparation for this subject/day first: {folder}."
    if (folder / "registration_pending.json").exists():
        raise ValueError(f"Reference preparation is incomplete in {folder}: "
                         f"registration_pending.json exists. {instruction}")
    required = {
        "corrected EPI reference": cfg.rt_unwarped_analysis_ref_epi,
        "EPI brain mask": cfg.rt_unwarped_analysis_ref_mask,
    }
    native_masks = [cfg.rt_unwarped_analysis_ref_mask]
    if getattr(settings, "enable_motion_regression", False):
        for label, enabled, path in (
                ("WM nuisance mask", "use_wm", cfg.rt_wm_mask),
                ("ventricle nuisance mask", "use_vent", cfg.rt_vent_mask)):
            if getattr(settings, enabled, False):
                required[label] = path
                native_masks.append(path)
    space = str(getattr(settings, "analysis_space", "mni")).lower()
    mask_type = getattr(settings, "analysis_mask", "whole_brain")
    mask_space = ("t1" if mask_type == "cortical_gm" else
                  getattr(settings, "analysis_mask_space", "final") if mask_type == "custom" else "epi")
    if space in {"t1", "mni"}:
        required["EPI-to-T1 transform"] = folder / "epi2t1_Composite.h5"
    elif space == "epi" and mask_space == "t1":
        required["T1-to-EPI transform"] = folder / "epi2t1_InverseComposite.h5"
    if space == "mni":
        required["T1-to-MNI transform"] = t1_to_mni_transform(cfg)

    offline = getattr(settings, "offline_assets", None)
    if offline:
        metadata = folder / "preparation_assets.json"
        try:
            recorded = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Missing or unreadable offline reference preparation metadata in {folder}. "
                             f"{instruction}") from error
        if not isinstance(recorded, dict) or recorded.get("version") != 1:
            raise ValueError(f"Invalid offline reference preparation metadata in {folder}. {instruction}")
        manifest = Path(offline).expanduser()
        if not manifest.is_file():
            raise FileNotFoundError(f"Offline assets manifest not found: {manifest}. {instruction}")
        # Only this small manifest is hashed here. Do not reload/hash/copy the
        # imported images or rebuild a reference while launching NF.
        if recorded.get("offline_manifest") != file_identity(manifest):
            raise ValueError(f"The selected offline assets do not match the prepared reference in {folder}. "
                             f"{instruction}")

    bridged = bool(offline) or any((folder / name).exists() for name in (
        "session_to_reference_Composite.h5", "session_to_reference_InverseComposite.h5",
        "reference_epi.nii", "reference_epi_mask.nii"))
    if bridged:
        required.update({
            "prepared EPI bridge": folder / "session_to_reference_Composite.h5",
            "prepared inverse EPI bridge": folder / "session_to_reference_InverseComposite.h5",
            "offline EPI reference": folder / "reference_epi.nii",
            "offline EPI reference mask": folder / "reference_epi_mask.nii",
            "composed EPI-to-T1 transform": folder / "epi2t1_Composite.h5",
            "composed T1-to-EPI transform": folder / "epi2t1_InverseComposite.h5",
        })
    missing = [f"{label}: {path}" for label, path in required.items()
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Prepared RS/behavioral reference is incomplete in {folder}. "
                                f"{instruction} Missing: " + "; ".join(missing))
    reference = nib.load(str(cfg.rt_unwarped_analysis_ref_epi))
    for path in native_masks:
        require_same_grid(nib.load(str(path)), reference)
    if bridged:
        require_same_grid(nib.load(str(folder / "reference_epi_mask.nii")),
                          nib.load(str(folder / "reference_epi.nii")))


def ensure_nf_session(cfg, settings, prepare_fieldmap=None):
    """Use reference-run preparation; NF never constructs a spatial reference.

    The selected run's AP/PA calibration may still be prepared independently.
    Spatial fitting and diagnostic registration QC remain reference-run work.
    """
    if not getattr(cfg, "preparation_only", False):
        validate_prepared_reference(cfg, settings)
    if prepare_fieldmap is not None:
        prepare_fieldmap(cfg)
