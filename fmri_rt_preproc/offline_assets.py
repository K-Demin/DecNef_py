"""Portable, explicit offline image/transform inputs (ANTs image conventions).

Bundles never contain a fieldmap for a future acquisition or temporal GLM state.
All filenames in a manifest are relative to that manifest unless absolute.
"""
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from .native_fieldmap import require_same_grid

REQUIRED_IMAGES = ("epi_reference", "epi_mask", "t1", "t1_mask", "wm_mask", "vent_mask", "mni_template")
OPTIONAL_IMAGES = ("cortical_gm_mask", "epi_registration_reference")
REQUIRED_TRANSFORMS = ("epi_to_t1", "t1_to_epi", "t1_to_mni")


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def load_single_volume_image(path):
    """Read a spatial image, accepting one volume stored as (X, Y, Z, 1).

    Only the time axis is removed; a single acquired spatial slice stays 3D.
    Scaled voxel values are retained in their decoded dtype, avoiding another
    integer quantization when writing the canonical image. Spatial transforms
    and their NIfTI codes are preserved explicitly.
    """
    image = nib.load(str(path))
    if len(image.shape) == 3:
        return image
    if len(image.shape) != 4 or image.shape[3] != 1:
        raise ValueError(f"Expected a 3D image or a 4D image with one volume: {path}")
    values = np.asanyarray(image.dataobj)[..., 0]
    result = image.__class__(values, image.affine, image.header.copy(), extra=image.extra.copy())
    result.set_data_dtype(values.dtype)
    if hasattr(image, "get_qform"):
        result.set_qform(image.get_qform(), int(image.header["qform_code"]))
        result.set_sform(image.get_sform(), int(image.header["sform_code"]))
    return result


def load_offline_assets(path, subject=None):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get("version") != 1 or data.get("epi_distortion_corrected") is not True:
        raise ValueError("Offline assets need version=1 and epi_distortion_corrected=true.")
    if not data.get("mni_space"):
        raise ValueError("Offline assets must identify the exact mni_space.")
    if subject is not None and str(data.get("subject_id")) != str(subject):
        raise ValueError("Offline assets belong to a different subject.")
    resolved = dict(data)
    for key in REQUIRED_IMAGES + REQUIRED_TRANSFORMS + OPTIONAL_IMAGES:
        if key in OPTIONAL_IMAGES and not data.get(key):
            continue
        if not data.get(key):
            raise ValueError(f"Missing offline asset: {key}")
        resolved[key] = (path.parent / data[key]).resolve()
        if not resolved[key].is_file():
            raise FileNotFoundError(resolved[key])
    for key in REQUIRED_IMAGES + tuple(key for key in OPTIONAL_IMAGES if data.get(key)):
        img = nib.load(str(resolved[key]))
        arr = np.asarray(img.dataobj)
        if arr.ndim != 3 or not np.isfinite(arr).all():
            raise ValueError(f"Offline {key} must be a finite 3D image.")
        if key.endswith("mask") and not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"Offline {key} must be binary; threshold probability maps explicitly.")
        if key in {"epi_mask", "t1_mask"} and not arr.any():
            raise ValueError(f"Offline {key} is empty.")
    require_same_grid(nib.load(str(resolved["epi_reference"])), nib.load(str(resolved["epi_mask"])))
    if data.get("epi_registration_reference"):
        require_same_grid(nib.load(str(resolved["epi_reference"])),
                          nib.load(str(resolved["epi_registration_reference"])))
    for key in ("t1_mask", "wm_mask", "vent_mask", "cortical_gm_mask"):
        if key in resolved:
            require_same_grid(nib.load(str(resolved["t1"])), nib.load(str(resolved[key])))
    resolved["manifest"] = path
    return resolved


def bundle_identity(assets):
    return {key: file_identity(value) for key, value in assets.items() if isinstance(value, Path)}


def copy_image(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Saving via nibabel also handles .nii.gz -> .nii without mislabeling bytes.
    nib.save(load_single_volume_image(source), str(destination))
    return destination


def apply_image(source, reference, output, transforms, run_command, interpolation="Linear"):
    args = ["antsApplyTransforms", "-d", "3", "--float", "1", "-i", str(source),
            "-r", str(reference), "-o", str(output), "-n", interpolation, "--default-value", "0"]
    for transform in transforms:
        args += ["-t", str(transform)]
    run_command(args)
    require_same_grid(nib.load(str(output)), nib.load(str(reference)))


def compose_transforms(transforms, reference, output, run_command):
    """Collapse an image pull-transform list on its target grid, including inverse specs.

    The longstanding ANTs [output,1] interface works on older scanner installs.
    An HDF5 output contains a displacement transform, not relabelled NIfTI data.
    """
    args = ["antsApplyTransforms", "-d", "3", "--float", "1", "-r", str(reference),
            "-o", f"[{output},1]"]
    for transform in transforms:
        args += ["-t", str(transform)]
    run_command(args)
    if not Path(output).is_file() or not Path(output).stat().st_size:
        raise RuntimeError(f"ANTs did not write transform: {output}")


def t1_to_mni_transform(cfg):
    # Each session selects its own imported anatomy. Never silently overwrite
    # subject-level SynthMorph assets with transforms from a different T1 space.
    for name in ("t1_to_mni.h5", "t1_to_mni.nii", "t1_to_mni.nii.gz", "t1_to_mni_Composite.h5"):
        imported = cfg.trans_dir / name
        if imported.exists():
            return imported
    return cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii"


def publish_mni_transform(source, folder):
    destination = Path(folder) / ("t1_to_mni" + "".join(Path(source).suffixes))
    shutil.copy2(source, destination)
    return destination


def install_anatomy(assets, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = {"t1": "T1_N4.nii", "t1_mask": "T1_combined_mask.nii",
             "wm_mask": "WM_T1.nii", "vent_mask": "Vent_T1.nii",
             "cortical_gm_mask": "cortical_gm_mask.nii"}
    for key, name in names.items():
        if key in assets:
            copy_image(assets[key], folder / name)
    t1 = nib.load(str(assets["t1"]))
    brain = np.asarray(t1.dataobj) * (np.asarray(nib.load(str(assets["t1_mask"])).dataobj) > 0)
    nib.save(nib.Nifti1Image(brain.astype(np.float32), t1.affine, t1.header), folder / "T1_brain.nii")


def write_bundle(destination, subject_id, images, transforms, mni_space, provenance=None):
    """Copy a fully specified bundle; refuse to merge into an existing bundle."""
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Offline destination must be empty; existing bundles are immutable.")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "subject_id": str(subject_id), "mni_space": mni_space,
                "epi_distortion_corrected": True, "provenance": provenance or {}}
    for key, source in images.items():
        manifest[key] = key + ".nii.gz"
        copy_image(source, destination / manifest[key])
    for key, source in transforms.items():
        manifest[key] = key + "".join(Path(source).suffixes)
        shutil.copy2(source, destination / manifest[key])
    output = destination / "assets.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    load_offline_assets(output, subject_id)
    return output
