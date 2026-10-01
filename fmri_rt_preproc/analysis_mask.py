"""Experiment mask selection and one-time mapping into the final BOLD grid."""
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .native_fieldmap import require_same_grid, source_identity

# Cortical DKT labels used in the existing FastSurfer preparation. Excludes WM,
# CSF, subcortical GM, cerebellum, and unknown/corpus-callosum labels.
DKT_PARCELS = (2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35)


def final_output_kind(fwhm_mm, mask_type="whole_brain"):
    fwhm = float(fwhm_mm)
    if not np.isfinite(fwhm) or fwhm < 0:
        raise ValueError("smoothing_fwhm_mm must be finite and >= 0.")
    if mask_type not in {"whole_brain", "cortical_gm", "custom"}:
        raise ValueError(f"Invalid analysis_mask: {mask_type}")
    return "smooth" if fwhm > 0 else ("masked" if mask_type != "whole_brain" else None)


def mask_transforms(cfg, source_space, final_space):
    if source_space == final_space or source_space == "final":
        return []
    forward = cfg.trans_dir / "epi2t1_Composite.h5"
    mni = cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii"
    chains = {("epi", "t1"): [forward], ("epi", "mni"): [mni, forward],
              ("t1", "epi"): [cfg.trans_dir / "epi2t1_InverseComposite.h5"],
              ("t1", "mni"): [mni]}
    if (source_space, final_space) not in chains:
        raise ValueError(f"Cannot map a {source_space} mask to {final_space}; supply a mask in final coordinates.")
    return chains[source_space, final_space]


def mask_source(cfg, mask_type, custom_file=None, custom_space="final"):
    if mask_type == "whole_brain":
        return cfg.rt_unwarped_analysis_ref_mask, "epi"
    if mask_type == "cortical_gm":
        return (cfg.subject_root / "anat" / "fastsurfer" / cfg.subject / "mri"
                / "aparc.DKTatlas+aseg.deep.mgz"), "t1"
    if mask_type != "custom" or not custom_file:
        raise ValueError("Custom analysis masking requires analysis_mask_file.")
    if custom_space not in {"final", "epi", "t1", "mni"}:
        raise ValueError(f"Invalid analysis_mask_space: {custom_space}")
    return Path(custom_file).expanduser().resolve(), custom_space


def mask_provenance(cfg, mask_type, space, reference, custom_file=None, custom_space="final"):
    source, coordinates = mask_source(cfg, mask_type, custom_file, custom_space)
    transforms = list(dict.fromkeys(mask_transforms(cfg, coordinates, space)
                                    + mask_transforms(cfg, "epi", space)))
    return {"type": mask_type, "source_space": coordinates,
            "source": source_identity(source),
            "coverage": source_identity(cfg.rt_unwarped_analysis_ref_mask),
            "reference": source_identity(reference),
            "transforms": [source_identity(p) for p in transforms]}


def prepare_analysis_mask(cfg, mask_type, space, reference, folder, run_command,
                          custom_file=None, custom_space="final"):
    source, coordinates = mask_source(cfg, mask_type, custom_file, custom_space)
    reference_img = nib.load(str(reference))
    image = nib.load(str(source))
    values = np.asarray(image.dataobj).copy()
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Analysis mask source must be a finite 3D image.")
    if mask_type == "cortical_gm":
        labels = [hemisphere + parcel for hemisphere in (1000, 2000) for parcel in DKT_PARCELS]
        mask = np.isin(values, labels)
    else:
        if mask_type == "custom" and not np.all((values == 0) | (values == 1)):
            raise ValueError("Custom analysis mask must be binary (0/1), not a probabilistic map or label image.")
        mask = values > .5
    if not mask.any():
        raise ValueError("Analysis mask source is empty.")

    def map_image(input_path, input_image, input_space, output):
        transforms = mask_transforms(cfg, input_space, space)
        if input_space == "final":
            require_same_grid(input_image, reference_img)
            return np.asarray(input_image.dataobj) > .5
        if not transforms:
            return np.asarray(resample_from_to(input_image, reference_img, order=0,
                                               mode="constant", cval=0).dataobj) > .5
        for path in transforms:
            if not path.exists():
                raise FileNotFoundError(path)
        cmd = ["antsApplyTransforms", "-d", "3", "-i", str(input_path),
               "-r", str(reference), "-o", str(output), "-n", "NearestNeighbor", "--float", "1"]
        for path in transforms:
            cmd.extend(["-t", str(path)])
        run_command(cmd)
        mapped = nib.load(str(output))
        require_same_grid(mapped, reference_img)
        values = np.asarray(mapped.dataobj).copy()
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("Transformed analysis mask must be a finite 3D image.")
        return values > .5

    binary = nib.Nifti1Image(mask.astype(np.uint8), image.affine)
    # The whole-brain source is already binary; keep its existing input path.
    binary_path = source
    if mask_type != "whole_brain":
        binary_path = folder / "source_mask.nii"
        nib.save(binary, str(binary_path))
    mapped = map_image(binary_path, binary, coordinates, folder / "mask.nii")
    if mask_type != "whole_brain":
        coverage_path = cfg.rt_unwarped_analysis_ref_mask
        coverage = map_image(coverage_path, nib.load(str(coverage_path)), "epi",
                             folder / "coverage_mask.nii")
        mapped &= coverage
    if not mapped.any():
        raise ValueError("Analysis mask has no overlap with the acquired brain in the final grid.")
    return mapped, reference_img.affine
