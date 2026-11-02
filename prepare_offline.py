"""Import explicit fMRIPrep assets or export a prepared DecNef session."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from fmri_rt_preproc.native_fieldmap import require_same_grid
from fmri_rt_preproc.offline_assets import (
    compose_transforms, file_identity, load_single_volume_image, t1_to_mni_transform, write_bundle,
)
from fmri_rt_preproc.utils import run

VENT_LABELS = (4, 43, 5, 44, 14, 15)


def _file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _image(path, binary=False):
    image = load_single_volume_image(_file(path))
    values = np.asarray(image.dataobj)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite 3D image: {path}")
    if binary and (not np.all((values == 0) | (values == 1)) or not values.any()):
        raise ValueError(f"Expected nonempty binary mask: {path}")
    return image


def _empty_destination(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Offline destination must be empty; existing bundles are immutable.")
    return path


def _save_mask(values, reference, path):
    values = np.asarray(values, dtype=bool)
    if not values.any():
        raise ValueError(f"Derived {Path(path).name} is empty; check image space and tissue labels.")
    nib.save(nib.Nifti1Image(values.astype(np.uint8), reference.affine), path)
    return Path(path)


def _on_t1(image, t1, order):
    return np.asarray(resample_from_to(image, t1, order=order, mode="constant", cval=0).dataobj)


def _affine_text(path):
    """Only an explicit invertible 3D ITK affine is eligible for [file,1]."""
    text = _file(path).read_text()
    transforms = re.findall(r"^Transform:\s*(\S+)", text, flags=re.MULTILINE)
    parameters = re.findall(r"^Parameters:\s*(.*)$", text, flags=re.MULTILINE)
    centers = re.findall(r"^FixedParameters:\s*(.*)$", text, flags=re.MULTILINE)
    if (not text.lstrip().startswith("#Insight Transform File") or len(transforms) != 1
            or transforms[0] not in {"AffineTransform_double_3_3", "AffineTransform_float_3_3",
                                     "MatrixOffsetTransformBase_double_3_3", "MatrixOffsetTransformBase_float_3_3"}
            or len(parameters) != 1 or len(centers) != 1):
        raise ValueError("EPI affine .txt must contain one 3D ITK affine, not FSL/AFNI matrices or HMC series.")
    try:
        values = np.asarray([float(value) for value in parameters[0].split()])
        center = np.asarray([float(value) for value in centers[0].split()])
        valid = (values.size == 12 and center.size == 3 and np.isfinite(values).all()
                 and np.isfinite(center).all() and abs(np.linalg.det(values[:9].reshape(3, 3))) > 1e-12)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("EPI ITK affine parameters must describe a finite invertible 3D transform.")


def import_fmriprep(args, run_command=run):
    output = _empty_destination(args.output)
    names = ("epi_reference", "epi_mask", "epi_to_t1", "t1", "t1_mask", "wm_probseg",
             "aseg", "t1_to_mni", "mni_template")
    sources = {key: _file(getattr(args, key)) for key in names}
    if args.t1_to_epi is not None:
        sources["t1_to_epi"] = _file(args.t1_to_epi)
    forward = sources["epi_to_t1"]
    if forward.suffix.lower() not in {".txt", ".h5"}:
        raise ValueError("--epi-to-t1 must be a 3D ITK affine .txt or an ANTs .h5 transform.")
    if forward.suffix.lower() == ".txt":
        _affine_text(forward)
    elif "t1_to_epi" not in sources:
        raise ValueError("An .h5 EPI-to-T1 transform requires explicit --t1-to-epi; nonlinear transforms are not inverted automatically.")
    if sources["t1_to_mni"].suffix.lower() != ".h5":
        raise ValueError("fMRIPrep --t1-to-mni must be the matching ANTs .h5 transform.")
    if "t1_to_epi" in sources and sources["t1_to_epi"].suffix.lower() == ".txt":
        _affine_text(sources["t1_to_epi"])
    elif "t1_to_epi" in sources and sources["t1_to_epi"].suffix.lower() != ".h5":
        raise ValueError("--t1-to-epi must be a 3D ITK affine .txt or an ANTs .h5 transform.")

    epi, epi_mask = _image(sources["epi_reference"]), _image(sources["epi_mask"], binary=True)
    t1, t1_mask = _image(sources["t1"]), _image(sources["t1_mask"], binary=True)
    require_same_grid(epi, epi_mask)
    require_same_grid(t1, t1_mask)
    mni_template = _image(sources["mni_template"])
    wm, aseg = _image(sources["wm_probseg"]), _image(sources["aseg"])
    if np.any((np.asarray(wm.dataobj) < 0) | (np.asarray(wm.dataobj) > 1)):
        raise ValueError("--wm-probseg must contain probabilities between 0 and 1.")
    if not np.all(np.asarray(aseg.dataobj) == np.rint(np.asarray(aseg.dataobj))):
        raise ValueError("--aseg must contain integer FreeSurfer labels in T1w physical space.")

    with tempfile.TemporaryDirectory(prefix="decnef-offline-") as scratch:
        scratch = Path(scratch)
        # ANTs and runtime bundles receive actual 3D files, even when the
        # source reference stores a singleton time dimension. Keep provenance
        # attached to the original files supplied by the user.
        prepared = dict(sources)
        for key, image in {"epi_reference": epi, "epi_mask": epi_mask, "t1": t1,
                           "t1_mask": t1_mask, "mni_template": mni_template}.items():
            if len(nib.load(str(sources[key])).shape) == 4:
                prepared[key] = scratch / (key + ".nii.gz")
                nib.save(image, prepared[key])
        brain = np.asarray(t1_mask.dataobj) > 0
        labels = np.rint(_on_t1(aseg, t1, 0)).astype(np.int32)
        images = {key: prepared[key] for key in ("epi_reference", "epi_mask", "t1", "t1_mask", "mni_template")}
        # fMRIPrep's supplied coregistration reference is already enhanced.
        # Preserve it as the fitting image; never fit another N4 field here.
        images["epi_registration_reference"] = prepared["epi_reference"]
        images["wm_mask"] = _save_mask((_on_t1(wm, t1, 1) >= .95) & brain, t1, scratch / "wm_mask.nii")
        images["vent_mask"] = _save_mask(np.isin(labels, VENT_LABELS) & brain, t1, scratch / "vent_mask.nii")
        cortex = ((labels == 3) | (labels == 42) | ((labels >= 1000) & (labels < 3000))) & brain
        if cortex.any():
            images["cortical_gm_mask"] = _save_mask(cortex, t1, scratch / "cortical_gm_mask.nii")
        transforms = {"t1_to_mni": sources["t1_to_mni"]}
        if forward.suffix.lower() == ".h5":
            transforms["epi_to_t1"] = forward
        else:
            transforms["epi_to_t1"] = scratch / "epi_to_t1.h5"
            compose_transforms([forward], prepared["t1"], transforms["epi_to_t1"], run_command)
        inverse = sources.get("t1_to_epi")
        if inverse is not None and inverse.suffix.lower() == ".h5":
            transforms["t1_to_epi"] = inverse
        else:
            transforms["t1_to_epi"] = scratch / "t1_to_epi.h5"
            inverse_spec = inverse if inverse is not None else f"[{forward},1]"
            compose_transforms([inverse_spec], prepared["epi_reference"], transforms["t1_to_epi"], run_command)
        provenance = {"source": "fmriprep_explicit_import", "wm_probability_threshold": .95,
                      "vent_labels": list(VENT_LABELS), "cortex_labels": "3,42,1000..2999",
                      "epi_registration_reference": {"source": "provided_fmriprep_reference",
                                                     "n4_rerun": False},
                      "inputs": {key: file_identity(path) for key, path in sources.items()},
                      "alignment_validation": "not_performed_by_import"}
        return write_bundle(output, args.subject, images, transforms, args.mni_space, provenance)


def _nifti(folder, name):
    for suffix in (".nii", ".nii.gz"):
        path = folder / (name + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(folder / (name + ".nii[.gz]"))


def export_prepared(args):
    output = _empty_destination(args.output)
    day = Path(args.prepared_day).expanduser().resolve()
    trans = day / "func" / "trans"
    if (trans / "registration_pending.json").exists():
        raise ValueError("Prepared registration is incomplete; finish preparation before exporting.")
    if day.parent.name != f"sub-{args.subject}":
        raise ValueError("--subject must match the prepared day's sub-<subject> parent directory.")
    anat = trans / "offline_anat" if (trans / "offline_anat").is_dir() else day.parent / "anat"
    sources = {"epi_reference": _nifti(trans, "epi_unwarped_mean"), "epi_mask": _nifti(trans, "epi_mask_mean"),
               "t1": _nifti(anat, "T1_N4"), "t1_mask": _nifti(anat, "T1_combined_mask"),
               "wm_mask": _nifti(trans, "WM_T1"), "vent_mask": _nifti(trans, "Vent_T1"),
               "mni_template": _file(args.mni_template)}
    images = {key: value for key, value in sources.items() if key not in {"wm_mask", "vent_mask"}}
    reference_preparation = None
    from fmri_rt_preproc.epi_reference import (
        epi_reference_metadata_path, validate_epi_registration_reference,
    )
    candidates = [trans / ("epi_registration_reference" + suffix)
                  for suffix in (".nii", ".nii.gz")]
    candidate = next((path for path in candidates if path.exists()), candidates[0])
    if candidate.exists() or epi_reference_metadata_path(candidate).exists():
        # Do not attach a cached N4 image merely because its filename fits.
        # Validate the original mean, mask and derived output identities.
        reference_preparation = validate_epi_registration_reference(
            sources["epi_reference"], sources["epi_mask"], candidate)
        sources["epi_registration_reference"] = candidate
        images["epi_registration_reference"] = candidate
    epi, t1 = _image(images["epi_reference"]), _image(images["t1"])
    require_same_grid(epi, _image(images["epi_mask"], binary=True))
    prepared_t1_mask = _image(images["t1_mask"])
    require_same_grid(t1, prepared_t1_mask)
    mask_values = np.asarray(prepared_t1_mask.dataobj)
    if np.any((mask_values < 0) | (mask_values > 1)):
        raise ValueError("Prepared T1_combined_mask must contain values between 0 and 1.")
    # Native preparation uses FLIRT to resample a binary anatomical mask and
    # can leave fractional edge values. Preserve this exporter's existing >0
    # brain/tissue selection in an explicitly binary bundle copy. This does
    # not alter the prepared mask or the already estimated transforms.
    brain = mask_values > 0
    if not brain.any():
        raise ValueError("Prepared T1_combined_mask is empty.")
    mask_preparation = {
        "source": "prepared_T1_combined_mask",
        "binarization": "value > 0",
        "fractional_voxels": int(np.count_nonzero((mask_values > 0) & (mask_values < 1))),
        "included_voxels": int(np.count_nonzero(brain)),
    }
    _image(images["mni_template"])
    transforms = {"epi_to_t1": _file(trans / "epi2t1_Composite.h5"),
                  "t1_to_epi": _file(trans / "epi2t1_InverseComposite.h5"),
                  "t1_to_mni": _file(t1_to_mni_transform(SimpleNamespace(trans_dir=trans, subject_root=day.parent)))}
    with tempfile.TemporaryDirectory(prefix="decnef-offline-") as scratch:
        scratch = Path(scratch)
        images["t1_mask"] = _save_mask(brain, t1, scratch / "t1_mask.nii")
        for key in ("wm_mask", "vent_mask"):
            images[key] = _save_mask((_on_t1(_image(sources[key], binary=True), t1, 0) > 0) & brain,
                                     t1, scratch / (key + ".nii"))
        cortex = anat / "cortical_gm_mask.nii"
        aseg = anat / "fastsurfer" / args.subject / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        if cortex.exists():
            sources["cortical_gm_mask"] = cortex
            values = _on_t1(_image(cortex, binary=True), t1, 0) > 0
        elif aseg.exists():
            sources["aseg"] = aseg
            labels = np.rint(_on_t1(_image(aseg), t1, 0)).astype(np.int32)
            values = (labels == 3) | (labels == 42) | ((labels >= 1000) & (labels < 3000))
        else:
            values = None
        if values is not None:
            images["cortical_gm_mask"] = _save_mask(values & brain, t1, scratch / "cortical_gm_mask.nii")
        provenance = {"source": "decnef_prepared_day", "prepared_day": str(day),
                      "inputs": {key: file_identity(path) for key, path in {**sources, **transforms}.items()},
                      "t1_mask_preparation": mask_preparation,
                      "alignment_validation": "not_performed_by_export"}
        if reference_preparation is not None:
            provenance["epi_registration_reference"] = {
                "source": "verified_prepared_n4", "preparation": reference_preparation,
            }
        return write_bundle(output, args.subject, images, transforms, args.mni_space, provenance)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-fmriprep", help="Import explicit, matching fMRIPrep images and transforms.")
    for option in ("epi-reference", "epi-mask", "epi-to-t1", "t1", "t1-mask", "wm-probseg", "aseg", "t1-to-mni"):
        importer.add_argument("--" + option, type=Path, required=True)
    importer.add_argument("--t1-to-epi", type=Path, help="Required inverse for a composite .h5 EPI-to-T1 transform.")
    exporter = commands.add_parser("export-prepared", help="Copy an existing corrected DecNef reference and transforms.")
    exporter.add_argument("--prepared-day", type=Path, required=True)
    for subparser in (importer, exporter):
        subparser.add_argument("--subject", required=True, help="Subject ID used by realtime, without the sub- prefix.")
        subparser.add_argument("--mni-template", type=Path, required=True)
        subparser.add_argument("--mni-space", required=True, help="Exact template identifier, e.g. MNI152NLin2009cAsym.")
        subparser.add_argument("--output", type=Path, required=True, help="New or empty immutable bundle directory.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.subject.strip() or not args.mni_space.strip():
        parser.error("--subject and --mni-space cannot be empty.")
    manifest = import_fmriprep(args) if args.command == "import-fmriprep" else export_prepared(args)
    print(f"Offline asset manifest: {manifest}")
    print("Import/export checks structure and grids; inspect anatomical alignment before use.")
    return manifest


if __name__ == "__main__":
    main()
